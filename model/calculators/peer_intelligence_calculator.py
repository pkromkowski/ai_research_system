import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

from model.calculators.calculator_base import CalculatorBase


class PeerIntelligenceCalculator(CalculatorBase):
    """Calculates peer price intelligence metrics comparing stock to peer group."""

    ROLLING_WINDOW = 63
    TAIL_QUANTILE = 0.05
    VOL_REGIME_WINDOW = 30
    MIN_DATA_POINTS = 10
    
    def __init__(self, history: pd.DataFrame):
        """Initialize calculator with price history.

        Args:
            history: DataFrame with 'Close' column and DatetimeIndex
        """
        self.history = history.copy()
        self.calculations: Dict[str, Any] = {}
        self.stock_returns_aligned: Optional[pd.Series] = None
        self.stock_prices_aligned: Optional[pd.Series] = None
        self.peer_returns: Optional[pd.DataFrame] = None
        self.peer_returns_aligned: Optional[pd.DataFrame] = None
        self.peer_median_aligned: Optional[pd.Series] = None
        self.peer_prices_aligned: Dict[str, pd.Series] = {}
        self.common_index: Optional[pd.DatetimeIndex] = None
        self._peer_data_ready = False
    
    def _require_peer_ready(self) -> bool:
        """Check if peer data is set up and aligned."""
        return self._peer_data_ready and self.stock_returns_aligned is not None and self.peer_median_aligned is not None

    def _require_min_aligned(self, min_len: int) -> bool:
        """Check if aligned stock returns meet minimum length."""
        return self._require_peer_ready() and len(self.stock_returns_aligned) >= min_len

    def _require_min_mask(self, mask: pd.Series) -> bool:
        """Check if mask has sufficient True values."""
        if mask is None:
            return False
        return mask.sum() >= self.MIN_DATA_POINTS

    @staticmethod
    def _calculate_recovery_time(prices: pd.Series) -> float:
        """Calculate maximum consecutive days in drawdown."""
        try:
            peak = prices.cummax()
            is_in_drawdown = prices < peak
            recovery_periods = is_in_drawdown.groupby(
                (~is_in_drawdown).cumsum()
            ).sum()
            return float(recovery_periods.max()) if len(recovery_periods) > 0 else 0.0
        except Exception:
            return 0.0
    
    def calculate_all(self, peer_prices: Dict[str, pd.Series] = None,
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute all or selected peer intelligence calculations.

        Args:
            peer_prices: Dictionary mapping ticker to price series
            metrics: List of metric groups to calculate
        """
        if not peer_prices or len(peer_prices) == 0:
            return {}

        if self.history.empty:
            return {}

        if not self._setup_peer_data(peer_prices):
            return {}

        group_methods = {
            'excess_returns': self.calculate_rolling_excess_return,
            'outperformance': self.calculate_outperformance_frequency,
            'down_market': self.calculate_down_market_relative_return,
            'drawdown': self.calculate_relative_max_drawdown,
            'recovery': self.calculate_relative_recovery_time,
            'tail_risk': self.calculate_left_tail_return_vs_peers,
            'rank_stability': self.calculate_rank_stability,
            'correlation': self.calculate_correlation_dispersion,
            'high_vol_regime': self.calculate_high_vol_relative_return,
        }

        if metrics is None:
            metrics = list(group_methods.keys())

        for metric_name in metrics:
            if metric_name in group_methods:
                try:
                    group_methods[metric_name]()
                except Exception:
                    pass

        return self.calculations.copy()
    
    def _setup_peer_data(self, peer_prices: Dict[str, pd.Series]) -> bool:
        """Setup aligned peer and stock data."""
        if not self._has_close():
            return False

        stock_prices = self.prices
        stock_returns = self.returns

        if stock_prices is None or stock_returns is None or len(stock_returns) < self.MIN_DATA_POINTS:
            return False

        peer_returns_dict = {}
        peer_prices_dict = {}

        for ticker, prices in peer_prices.items():
            if not isinstance(prices, pd.Series) or len(prices) <= 1:
                continue

            returns = prices.pct_change().dropna()
            if len(returns) > 0:
                if returns.index.tz is not None:
                    returns = returns.copy()
                    returns.index = returns.index.tz_localize(None)
                peer_returns_dict[ticker] = returns
                prices_naive = prices.copy()
                if prices_naive.index.tz is not None:
                    prices_naive.index = prices_naive.index.tz_localize(None)
                peer_prices_dict[ticker] = prices_naive

        if not peer_returns_dict:
            return False

        peer_returns_df = pd.DataFrame(peer_returns_dict).dropna()
        if peer_returns_df.empty:
            return False

        peer_median_ret = peer_returns_df.median(axis=1)

        common_index = stock_returns.index.intersection(peer_returns_df.index)
        if len(common_index) < self.MIN_DATA_POINTS:
            return False

        self.stock_returns_aligned = self.returns.loc[common_index]
        self.stock_prices_aligned = stock_prices.loc[common_index]
        self.peer_returns = peer_returns_df
        self.peer_returns_aligned = peer_returns_df.loc[common_index]
        self.peer_median_aligned = peer_median_ret.loc[common_index]
        self.common_index = common_index

        for ticker, prices in peer_prices_dict.items():
            aligned_prices = prices.reindex(common_index).dropna()
            if len(aligned_prices) > 0:
                self.peer_prices_aligned[ticker] = aligned_prices

        self._peer_data_ready = True
        return True
    
    def calculate_rolling_excess_return(self) -> Optional[float]:
        """Calculate rolling excess return vs peer median."""
        if not self._require_min_aligned(self.ROLLING_WINDOW):
            return None

        try:
            excess_ret = (
                (self.stock_returns_aligned - self.peer_median_aligned)
                .rolling(self.ROLLING_WINDOW).mean()
            )
            metric_name = f'rolling_excess_return_{self.ROLLING_WINDOW}d_vs_peers'
            return self._store_result(metric_name, excess_ret.iloc[-1])
        except Exception:
            return None

    def calculate_outperformance_frequency(self) -> Optional[float]:
        """Calculate frequency of outperforming peer median."""
        if not self._require_min_aligned(self.ROLLING_WINDOW):
            return None

        try:
            outperf = (
                (self.stock_returns_aligned > self.peer_median_aligned)
                .rolling(self.ROLLING_WINDOW).mean()
            )
            metric_name = f'outperformance_frequency_{self.ROLLING_WINDOW}d'
            return self._store_result(metric_name, outperf.iloc[-1])
        except Exception:
            return None
    
    def calculate_down_market_relative_return(self) -> Optional[float]:
        """Calculate relative return on days when peer median is negative."""
        if not self._require_peer_ready():
            return None

        try:
            down_days = self.peer_median_aligned < 0

            if not self._require_min_mask(down_days):
                return None

            down_market_return = (
                self.stock_returns_aligned[down_days] -
                self.peer_median_aligned[down_days]
            ).mean()

            return self._store_result('down_market_relative_return', down_market_return)
        except Exception:
            return None

    def calculate_relative_max_drawdown(self) -> Optional[float]:
        """Calculate stock max drawdown relative to peer average."""
        if not self._require_peer_ready():
            return None

        try:
            stock_dd = (
                self.stock_prices_aligned / self.stock_prices_aligned.cummax() - 1
            ).min()

            peer_dds = []
            for ticker, prices in self.peer_prices_aligned.items():
                if len(prices) > 0:
                    dd = (prices / prices.cummax() - 1).min()
                    if np.isfinite(dd):
                        peer_dds.append(dd)

            if not peer_dds:
                return None

            peer_dd_mean = np.mean(peer_dds)
            rel_dd = stock_dd - peer_dd_mean

            return self._store_result('relative_max_drawdown_vs_peers', rel_dd)
        except Exception:
            return None
    
    def calculate_relative_recovery_time(self) -> Optional[float]:
        """Calculate stock recovery time relative to peer average."""
        if not self._require_peer_ready():
            return None

        try:
            stock_recovery = self._calculate_recovery_time(self.stock_prices_aligned)

            peer_recoveries = []
            for ticker, prices in self.peer_prices_aligned.items():
                if len(prices) > 0:
                    rec = self._calculate_recovery_time(prices)
                    peer_recoveries.append(rec)

            if not peer_recoveries:
                return None

            peer_recovery_mean = np.mean(peer_recoveries)
            rel_recovery = stock_recovery - peer_recovery_mean

            return self._store_result('relative_recovery_time_days', rel_recovery)
        except Exception:
            return None

    def calculate_left_tail_return_vs_peers(self) -> Optional[float]:
        """Calculate stock left tail return relative to peer median."""
        if not self._require_peer_ready():
            return None

        try:
            stock_tail = self.stock_returns_aligned.quantile(self.TAIL_QUANTILE)
            peer_median_tail = self.peer_median_aligned.quantile(self.TAIL_QUANTILE)
            rel_tail = stock_tail - peer_median_tail

            metric_name = f'left_tail_return_{int(self.TAIL_QUANTILE * 100)}pct_vs_peers'
            return self._store_result(metric_name, rel_tail)
        except Exception:
            return None
    
    def calculate_rank_stability(self) -> Optional[float]:
        """Calculate stability of stock rank among peers."""
        if not self._require_min_aligned(self.ROLLING_WINDOW):
            return None

        try:
            ranks = (
                self.stock_returns_aligned.values.reshape(-1, 1) >
                self.peer_returns_aligned.values
            ).sum(axis=1)

            rank_series = pd.Series(ranks, index=self.stock_returns_aligned.index)
            rank_stability = rank_series.rolling(self.ROLLING_WINDOW).std()

            metric_name = f'peer_rank_stability_{self.ROLLING_WINDOW}d'
            return self._store_result(metric_name, rank_stability.iloc[-1])
        except Exception:
            return None

    def calculate_correlation_dispersion(self) -> Optional[float]:
        """Calculate dispersion of correlations with individual peers."""
        if not self._peer_data_ready:
            return None

        try:
            peer_corrs = []
            for ticker in self.peer_returns_aligned.columns:
                corr = self.stock_returns_aligned.corr(
                    self.peer_returns_aligned[ticker]
                )
                if pd.notna(corr):
                    peer_corrs.append(corr)

            if len(peer_corrs) < 2:
                return None

            corr_dispersion = np.std(peer_corrs)
            return self._store_result('correlation_dispersion_vs_peers', corr_dispersion)
        except Exception:
            return None
    
    def calculate_high_vol_relative_return(self) -> Optional[float]:
        """Calculate relative return during high volatility regimes."""
        if not self._peer_data_ready:
            return None

        if len(self.peer_median_aligned) < self.VOL_REGIME_WINDOW:
            return None

        try:
            vol = self.peer_median_aligned.rolling(self.VOL_REGIME_WINDOW).std()
            high_vol_regime = vol > vol.median()

            if high_vol_regime.sum() < self.MIN_DATA_POINTS:
                return None

            high_vol_return = (
                self.stock_returns_aligned[high_vol_regime] -
                self.peer_median_aligned[high_vol_regime]
            ).mean()

            metric_name = f'high_vol_{self.VOL_REGIME_WINDOW}d_relative_return'
            return self._store_result(metric_name, high_vol_return)
        except Exception:
            return None

