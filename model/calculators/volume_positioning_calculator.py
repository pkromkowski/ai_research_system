import pandas as pd
from typing import Dict, Optional, List, Any

from model.calculators.calculator_base import CalculatorBase


class VolumePositioningCalculator(CalculatorBase):
    """Calculates volume metrics and positioning metrics relative to market indices."""

    VOLUME_MA_PERIOD = 20
    VOLUME_VOLATILITY_PERIOD = 20
    VOLUME_PERSISTENCE_LAG = 5
    ROLLING_BETA_WINDOW = 63
    CORRELATION_WINDOW = 63
    MIN_DATA_POINTS = 5
    
    def __init__(self, history: pd.DataFrame):
        """Initialize calculator with price/volume history."""
        self.history = history.copy()
        self.calculations: Dict[str, Any] = {}
        self.sector_index: Optional[pd.DataFrame] = None
        self.broad_index: Optional[pd.DataFrame] = None
        self._benchmarks_ready = False
        self._sector_returns: Optional[pd.Series] = None
        self._broad_returns: Optional[pd.Series] = None
        self._common_index: Optional[pd.DatetimeIndex] = None
    
    def _require_volume_len(self, n: int) -> Optional[pd.Series]:
        """Return volume series if it exists and has at least n samples, else None."""
        if not self._has_volume():
            return None
        vol = self.history['Volume'].dropna()
        if len(vol) < n:
            return None
        return vol

    def _require_aligned_returns(self, min_len: int) -> Optional[pd.Series]:
        """Delegate to base `_require_returns` and align to `_common_index` if present."""
        series = self._require_returns(min_len)
        if series is None:
            return None
        if self._common_index is not None:
            aligned = series.reindex(self._common_index).dropna()
            if len(aligned) < min_len:
                return None
            return aligned
        return series

    def _get_benchmark_returns(self, benchmark_type: str) -> Optional[pd.Series]:
        """Return benchmark returns series by type ('sector'|'broad') or None."""
        if benchmark_type == 'sector':
            return self._sector_returns
        return self._broad_returns

    def _require_benchmark_returns(self, benchmark_type: str, min_len: int) -> Optional[pd.Series]:
        """Return benchmark returns if present and meets min_len else None."""
        br = self._get_benchmark_returns(benchmark_type)
        if br is None or len(br) < min_len:
            return None
        return br

    def _setup_benchmarks(self, sector_index_history: pd.DataFrame = None,
                          broad_index_history: pd.DataFrame = None):
        """Set up and align benchmark data with stock data."""
        if not self._has_close():
            return

        stock_returns = self.returns
        if stock_returns is None or len(stock_returns) < self.MIN_DATA_POINTS:
            return

        self.sector_index = sector_index_history
        self.broad_index = broad_index_history

        sector_returns = None
        broad_returns = None

        if sector_index_history is not None:
            col = 'Close' if 'Close' in sector_index_history.columns else 'close'
            if col in sector_index_history.columns:
                sector_price = sector_index_history[col].dropna()
                if getattr(sector_price.index, 'tz', None) is not None:
                    sector_price = sector_price.copy()
                    sector_price.index = sector_price.index.tz_localize(None)
                sector_returns = sector_price.pct_change().dropna()

        if broad_index_history is not None:
            col = 'Close' if 'Close' in broad_index_history.columns else 'close'
            if col in broad_index_history.columns:
                broad_price = broad_index_history[col].dropna()
                if getattr(broad_price.index, 'tz', None) is not None:
                    broad_price = broad_price.copy()
                    broad_price.index = broad_price.index.tz_localize(None)
                broad_returns = broad_price.pct_change().dropna()

        common_index = stock_returns.index

        if sector_returns is not None:
            common_index = common_index.intersection(sector_returns.index)
        if broad_returns is not None:
            common_index = common_index.intersection(broad_returns.index)

        if len(common_index) < self.MIN_DATA_POINTS:
            self._common_index = stock_returns.index
            self._benchmarks_ready = False
            return

        self._sector_returns = sector_returns.loc[common_index] if sector_returns is not None else None
        self._broad_returns = broad_returns.loc[common_index] if broad_returns is not None else None
        self._common_index = common_index
        self._benchmarks_ready = True

    def calculate_abnormal_volume(self) -> Optional[float]:
        """Calculate abnormal volume relative to moving average."""
        volume = self._require_volume_len(self.VOLUME_MA_PERIOD)
        if volume is None:
            return None

        vol_ma = volume.rolling(self.VOLUME_MA_PERIOD).mean()

        if vol_ma.iloc[-1] == 0 or pd.isna(vol_ma.iloc[-1]):
            return None

        abnormal_vol = (volume.iloc[-1] - vol_ma.iloc[-1]) / vol_ma.iloc[-1]
        metric_name = f'abnormal_volume_{self.VOLUME_MA_PERIOD}d'
        return self._store_result(metric_name, abnormal_vol)

    def calculate_volume_volatility(self) -> Optional[float]:
        """Calculate volatility of volume changes."""
        volume = self._require_volume_len(self.VOLUME_VOLATILITY_PERIOD + 1)
        if volume is None:
            return None

        vol_changes = volume.pct_change().dropna()
        vol_volatility = vol_changes.rolling(self.VOLUME_VOLATILITY_PERIOD).std().iloc[-1]

        metric_name = f'volume_volatility_{self.VOLUME_VOLATILITY_PERIOD}d'
        return self._store_result(metric_name, vol_volatility)

    def calculate_volume_persistence(self) -> Optional[float]:
        """Calculate volume autocorrelation at specified lag."""
        min_required = self.VOLUME_PERSISTENCE_LAG * 4
        volume = self._require_volume_len(min_required)
        if volume is None:
            return None

        try:
            vol_corr = volume.autocorr(lag=self.VOLUME_PERSISTENCE_LAG)
            metric_name = f'volume_persistence_{self.VOLUME_PERSISTENCE_LAG}d'
            return self._store_result(metric_name, vol_corr)
        except Exception:
            return None

    def calculate_average_daily_volume(self) -> Optional[float]:
        """Calculate average daily volume over entire period."""
        volume = self._require_volume_len(1)
        if volume is None or len(volume) == 0:
            return None

        return self._store_result('average_daily_volume', volume.mean())

    def calculate_volume_trend_ratio(self) -> Optional[float]:
        """Calculate current volume relative to moving average."""
        volume = self._require_volume_len(self.VOLUME_MA_PERIOD)
        if volume is None:
            return None

        vol_ma = volume.rolling(self.VOLUME_MA_PERIOD).mean().iloc[-1]

        if vol_ma == 0 or pd.isna(vol_ma):
            return None

        ratio = volume.iloc[-1] / vol_ma
        metric_name = f'volume_trend_ratio_{self.VOLUME_MA_PERIOD}d'
        return self._store_result(metric_name, ratio)

    def calculate_volume_group(self) -> Dict[str, Any]:
        """Calculate all volume metrics."""
        return self._collect_new_results([
            self.calculate_abnormal_volume,
            self.calculate_volume_volatility,
            self.calculate_volume_persistence,
            self.calculate_average_daily_volume,
            self.calculate_volume_trend_ratio
        ])

    def calculate_excess_return_latest(self, benchmark_type: str = 'sector') -> Optional[float]:
        """Calculate latest period excess return vs benchmark ('sector' or 'broad')."""
        stock = self._require_aligned_returns(1)
        if stock is None:
            return None

        benchmark_returns = self._require_benchmark_returns(benchmark_type, 1)
        if benchmark_returns is None:
            return None

        excess_return = stock.iloc[-1] - benchmark_returns.iloc[-1]

        metric_name = f'excess_return_vs_{benchmark_type}_latest'
        return self._store_result(metric_name, excess_return)

    def calculate_cumulative_excess_return(self, benchmark_type: str = 'sector') -> Optional[float]:
        """Calculate cumulative excess return vs benchmark over period ('sector' or 'broad')."""
        stock = self._require_aligned_returns(self.MIN_DATA_POINTS)
        if stock is None:
            return None

        benchmark_returns = self._require_benchmark_returns(benchmark_type, self.MIN_DATA_POINTS)
        if benchmark_returns is None:
            return None

        stock_cum = (1 + stock).prod() - 1
        benchmark_cum = (1 + benchmark_returns).prod() - 1
        excess = stock_cum - benchmark_cum

        metric_name = f'cumulative_excess_return_vs_{benchmark_type}'
        return self._store_result(metric_name, excess)

    def calculate_relative_drawdown(self, benchmark_type: str = 'sector') -> Optional[float]:
        """Calculate relative drawdown vs benchmark ('sector' or 'broad')."""
        stock = self._require_aligned_returns(self.MIN_DATA_POINTS)
        if stock is None:
            return None

        benchmark_returns = self._require_benchmark_returns(benchmark_type, self.MIN_DATA_POINTS)
        if benchmark_returns is None:
            return None

        stock_cum = (1 + stock).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()

        stock_dd = (stock_cum / stock_cum.cummax() - 1).iloc[-1]
        benchmark_dd = (benchmark_cum / benchmark_cum.cummax() - 1).iloc[-1]

        relative_dd = stock_dd - benchmark_dd

        metric_name = f'relative_drawdown_vs_{benchmark_type}'
        return self._store_result(metric_name, relative_dd)

    def calculate_rolling_beta(self, benchmark_type: str = 'sector') -> Optional[float]:
        """Calculate rolling beta vs benchmark ('sector' or 'broad')."""
        stock = self._require_aligned_returns(self.ROLLING_BETA_WINDOW)
        if stock is None:
            return None

        benchmark_returns = self._require_benchmark_returns(benchmark_type, self.ROLLING_BETA_WINDOW)
        if benchmark_returns is None:
            return None

        rolling_cov = stock.rolling(self.ROLLING_BETA_WINDOW).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(self.ROLLING_BETA_WINDOW).var()

        if rolling_var.iloc[-1] == 0 or pd.isna(rolling_var.iloc[-1]):
            return None

        beta = rolling_cov.iloc[-1] / rolling_var.iloc[-1]

        metric_name = f'rolling_beta_{self.ROLLING_BETA_WINDOW}d_vs_{benchmark_type}'
        return self._store_result(metric_name, beta)

    def calculate_rolling_correlation(self, benchmark_type: str = 'sector') -> Optional[float]:
        """Calculate rolling correlation with benchmark ('sector' or 'broad')."""
        stock = self._require_aligned_returns(self.CORRELATION_WINDOW)
        if stock is None:
            return None

        benchmark_returns = self._require_benchmark_returns(benchmark_type, self.CORRELATION_WINDOW)
        if benchmark_returns is None:
            return None

        rolling_corr = stock.rolling(self.CORRELATION_WINDOW).corr(benchmark_returns)

        metric_name = f'rolling_correlation_{self.CORRELATION_WINDOW}d_vs_{benchmark_type}'
        return self._store_result(metric_name, rolling_corr.iloc[-1])

    def calculate_positioning_vs_benchmark(self, benchmark_type: str = 'sector') -> Dict[str, Any]:
        """Calculate all positioning metrics vs specific benchmark ('sector' or 'broad')."""
        def _runner():
            self.calculate_excess_return_latest(benchmark_type)
            self.calculate_cumulative_excess_return(benchmark_type)
            self.calculate_relative_drawdown(benchmark_type)
            self.calculate_rolling_beta(benchmark_type)
            self.calculate_rolling_correlation(benchmark_type)

        return self._collect_new_results([_runner])

    def calculate_relative_drawdown_vs_benchmarks(self) -> Optional[float]:
        """Calculate relative drawdown vs average of sector and broad benchmarks."""
        if self._sector_returns is None or self._broad_returns is None:
            return None

        stock = self._require_aligned_returns(self.MIN_DATA_POINTS)
        if stock is None:
            return None

        stock_cum = (1 + stock).cumprod()
        sector_cum = (1 + self._sector_returns).cumprod()
        broad_cum = (1 + self._broad_returns).cumprod()

        stock_dd = (stock_cum / stock_cum.cummax() - 1).iloc[-1]
        sector_dd = (sector_cum / sector_cum.cummax() - 1).iloc[-1]
        broad_dd = (broad_cum / broad_cum.cummax() - 1).iloc[-1]

        avg_benchmark_dd = (sector_dd + broad_dd) / 2
        relative_dd = stock_dd - avg_benchmark_dd

        return self._store_result('relative_drawdown_vs_benchmarks', relative_dd)

    def calculate_avg_excess_return_vs_benchmarks(self) -> Optional[float]:
        """Calculate average excess return vs both benchmarks."""
        if self._sector_returns is None or self._broad_returns is None:
            return None

        stock = self._require_aligned_returns(self.MIN_DATA_POINTS)
        if stock is None:
            return None

        stock_cum = (1 + stock).prod() - 1
        sector_cum = (1 + self._sector_returns).prod() - 1
        broad_cum = (1 + self._broad_returns).prod() - 1

        excess_vs_sector = stock_cum - sector_cum
        excess_vs_broad = stock_cum - broad_cum
        avg_excess = (excess_vs_sector + excess_vs_broad) / 2

        return self._store_result('avg_excess_return_vs_benchmarks', avg_excess)

    def calculate_combined_benchmark_metrics(self) -> Dict[str, Any]:
        """Calculate all combined benchmark metrics."""
        return self._collect_new_results([
            self.calculate_relative_drawdown_vs_benchmarks,
            self.calculate_avg_excess_return_vs_benchmarks
        ])

    def calculate_all(self, sector_index_history: pd.DataFrame = None,
                      broad_index_history: pd.DataFrame = None,
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute all or selected volume and positioning calculations."""
        if self.history.empty:
            return {}

        self._setup_benchmarks(sector_index_history, broad_index_history)

        group_methods = {
            'volume_metrics': self.calculate_volume_group,
            'positioning_sector': lambda: self.calculate_positioning_vs_benchmark('sector'),
            'positioning_broad': lambda: self.calculate_positioning_vs_benchmark('broad'),
            'positioning_combined': self.calculate_combined_benchmark_metrics,
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
