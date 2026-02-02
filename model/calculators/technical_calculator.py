import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

from model.calculators.calculator_base import CalculatorBase


class TechnicalCalculator(CalculatorBase):
    """Calculates technical and performance metrics from historical price data."""
    
    SMA_SHORT = 20
    SMA_MID = 50
    SMA_LONG = 200
    EMA_SHORT = 20
    EMA_MID = 50
    ROC_PERIOD = 14
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MOMENTUM_SHORT = 5
    MOMENTUM_MID = 10
    MOMENTUM_LONG = 20
    VOLATILITY_WINDOW = 30
    BOLLINGER_PERIOD = 20
    BOLLINGER_STDEV = 2
    SHORT_RANGE_PERIOD = 20
    MID_RANGE_PERIOD = 63
    LONG_RANGE_PERIOD = 252
    VOLUME_MA_PERIOD = 30
    RISK_FREE_RATE_ANNUAL = 0.02
    TRADING_DAYS_PER_YEAR = 252
    MIN_DATA_POINTS = 5
    
    def __init__(self, history: pd.DataFrame):
        """Initialize calculator with price history."""
        self.history = history.copy()
        self.calculations: Dict[str, Any] = {}
    
    @property
    def _risk_free_rate_daily(self) -> float:
        """Daily risk-free rate derived from annual rate."""
        return self.RISK_FREE_RATE_ANNUAL / self.TRADING_DAYS_PER_YEAR
    
    def _has_ohlc(self) -> bool:
        """Check if High/Low/Close columns exist."""
        return all(col in self.history.columns for col in ['High', 'Low', 'Close']) and not self.history.empty

    def _latest_close(self):
        """Return latest close price or None."""
        prices = self.prices
        if prices is None or prices.empty:
            return None
        return prices.iloc[-1]

    def calculate_returns_group(self) -> Dict[str, Any]:
        """Calculate all return metrics."""
        return self._collect_new_results([
            self.calculate_daily_return_mean,
            self.calculate_daily_return_std,
            self.calculate_total_return,
            self.calculate_log_return_mean
        ])

    def calculate_daily_return_mean(self) -> Optional[float]:
        """Calculate mean daily return."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        return self._store_result('daily_return_mean', series.mean())
    
    def calculate_daily_return_std(self) -> Optional[float]:
        """Calculate daily return standard deviation."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        return self._store_result('daily_return_std', series.std())
    
    def calculate_total_return(self) -> Optional[float]:
        """Calculate total return over the period."""
        if self._require_prices(2) is None:
            return None
        prices = self.prices
        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]
        if first_price == 0 or pd.isna(first_price):
            return None
        total_return = (last_price - first_price) / first_price
        return self._store_result('total_return', total_return)
    
    def calculate_log_return_mean(self) -> Optional[float]:
        """Calculate mean log return."""
        series = self._require_log_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        return self._store_result('log_return_mean', series.mean())    
    
    def calculate_volatility_group(self) -> Dict[str, Any]:
        """Calculate all volatility metrics."""
        return self._collect_new_results([
            self.calculate_historical_volatility_daily,
            self.calculate_historical_volatility_annual,
            self.calculate_rolling_volatility_latest,
            self.calculate_rolling_volatility_mean
        ])
    
    def calculate_historical_volatility_daily(self) -> Optional[float]:
        """Calculate historical daily volatility."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        return self._store_result('historical_volatility_daily', series.std())
    
    def calculate_historical_volatility_annual(self) -> Optional[float]:
        """Calculate annualized historical volatility."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        annual_vol = series.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return self._store_result('historical_volatility_annual', annual_vol)
    
    def calculate_rolling_volatility_latest(self) -> Optional[float]:
        """Calculate latest rolling volatility."""
        series = self._require_returns(self.VOLATILITY_WINDOW - 1)
        if series is None:
            return None
        rolling_vol = series.rolling(window=self.VOLATILITY_WINDOW).std()
        metric_name = f'rolling_volatility_{self.VOLATILITY_WINDOW}d_latest'
        return self._store_result(metric_name, rolling_vol.iloc[-1])
    
    def calculate_rolling_volatility_mean(self) -> Optional[float]:
        """Calculate mean of rolling volatility."""
        series = self._require_returns(self.VOLATILITY_WINDOW - 1)
        if series is None:
            return None
        rolling_vol = series.rolling(window=self.VOLATILITY_WINDOW).std()
        metric_name = f'rolling_volatility_{self.VOLATILITY_WINDOW}d_mean'
        return self._store_result(metric_name, rolling_vol.mean())
    
    def calculate_moving_averages_group(self) -> Dict[str, Any]:
        """Calculate all moving average metrics."""
        return self._collect_new_results([
            self.calculate_current_price,
            self.calculate_sma_short,
            self.calculate_sma_mid,
            self.calculate_sma_long,
            self.calculate_ema_short,
            self.calculate_ema_mid,
            self.calculate_price_to_sma_short_ratio,
            self.calculate_price_to_sma_mid_ratio
        ])
    
    def calculate_current_price(self) -> Optional[float]:
        """Get current price."""
        if not self._has_close():
            return None
        return self._store_result('current_price', self._latest_close())
    
    def calculate_sma_short(self) -> Optional[float]:
        """Calculate short-term SMA."""
        prices = self._require_prices(self.SMA_SHORT)
        if prices is None:
            return None
        sma = prices.rolling(window=self.SMA_SHORT).mean().iloc[-1]
        metric_name = f'sma_{self.SMA_SHORT}_current'
        return self._store_result(metric_name, sma)
    
    def calculate_sma_mid(self) -> Optional[float]:
        """Calculate mid-term SMA."""
        prices = self._require_prices(self.SMA_MID)
        if prices is None:
            return None
        sma = prices.rolling(window=self.SMA_MID).mean().iloc[-1]
        metric_name = f'sma_{self.SMA_MID}_current'
        return self._store_result(metric_name, sma)
    
    def calculate_sma_long(self) -> Optional[float]:
        """Calculate long-term SMA."""
        prices = self._require_prices(self.SMA_LONG)
        if prices is None:
            return None
        sma = prices.rolling(window=self.SMA_LONG).mean().iloc[-1]
        metric_name = f'sma_{self.SMA_LONG}_current'
        return self._store_result(metric_name, sma)
    
    def calculate_ema_short(self) -> Optional[float]:
        """Calculate short-term EMA."""
        prices = self._require_prices(self.EMA_SHORT)
        if prices is None:
            return None
        ema = prices.ewm(span=self.EMA_SHORT, adjust=False).mean().iloc[-1]
        metric_name = f'ema_{self.EMA_SHORT}_current'
        return self._store_result(metric_name, ema)
    
    def calculate_ema_mid(self) -> Optional[float]:
        """Calculate mid-term EMA."""
        prices = self._require_prices(self.EMA_MID)
        if prices is None:
            return None
        ema = prices.ewm(span=self.EMA_MID, adjust=False).mean().iloc[-1]
        metric_name = f'ema_{self.EMA_MID}_current'
        return self._store_result(metric_name, ema)
    
    def calculate_price_to_sma_short_ratio(self) -> Optional[float]:
        """Calculate price to short SMA ratio."""
        if self._require_prices(self.SMA_SHORT) is None:
            return None
        current_price = self._latest_close()
        sma = self.history['Close'].rolling(window=self.SMA_SHORT).mean().iloc[-1]
        if sma == 0 or pd.isna(sma):
            return None
        metric_name = f'price_to_sma{self.SMA_SHORT}_ratio'
        return self._store_result(metric_name, current_price / sma)
    
    def calculate_price_to_sma_mid_ratio(self) -> Optional[float]:
        """Calculate price to mid SMA ratio."""
        if self._require_prices(self.SMA_MID) is None:
            return None
        current_price = self._latest_close()
        sma = self.history['Close'].rolling(window=self.SMA_MID).mean().iloc[-1]
        if sma == 0 or pd.isna(sma):
            return None
        metric_name = f'price_to_sma{self.SMA_MID}_ratio'
        return self._store_result(metric_name, current_price / sma)
    
    def calculate_momentum_group(self) -> Dict[str, Any]:
        """Calculate all momentum metrics."""
        return self._collect_new_results([
            self.calculate_roc,
            self.calculate_momentum_short,
            self.calculate_momentum_mid,
            self.calculate_momentum_long,
            self.calculate_rsi_current,
            self.calculate_rsi_mean,
            self.calculate_macd,
            self.calculate_macd_signal,
            self.calculate_macd_histogram
        ])
    
    def calculate_roc(self) -> Optional[float]:
        """Calculate Rate of Change."""
        if self._require_prices(self.ROC_PERIOD + 1) is None:
            return None
        prices = self._require_prices(self.ROC_PERIOD + 1)
        if prices is None:
            return None
        current = prices.iloc[-1]
        past = prices.iloc[-self.ROC_PERIOD - 1]
        if past == 0 or pd.isna(past):
            return None
        roc = ((current - past) / past) * 100
        metric_name = f'roc_{self.ROC_PERIOD}_current'
        return self._store_result(metric_name, roc)
    
    def calculate_momentum_short(self) -> Optional[float]:
        """Calculate short-term momentum."""
        if self._require_prices(self.MOMENTUM_SHORT + 1) is None:
            return None
        prices = self._require_prices(self.MOMENTUM_SHORT + 1)
        if prices is None:
            return None
        current = prices.iloc[-1]
        past = prices.iloc[-self.MOMENTUM_SHORT - 1]
        if past == 0 or pd.isna(past):
            return None
        momentum = (current - past) / past
        metric_name = f'momentum_{self.MOMENTUM_SHORT}d'
        return self._store_result(metric_name, momentum)
    
    def calculate_momentum_mid(self) -> Optional[float]:
        """Calculate mid-term momentum."""
        if self._require_prices(self.MOMENTUM_MID + 1) is None:
            return None
        prices = self._require_prices(self.MOMENTUM_MID + 1)
        if prices is None:
            return None
        current = prices.iloc[-1]
        past = prices.iloc[-self.MOMENTUM_MID - 1]
        if past == 0 or pd.isna(past):
            return None
        momentum = (current - past) / past
        metric_name = f'momentum_{self.MOMENTUM_MID}d'
        return self._store_result(metric_name, momentum)
    
    def calculate_momentum_long(self) -> Optional[float]:
        """Calculate long-term momentum."""
        if self._require_prices(self.MOMENTUM_LONG + 1) is None:
            return None
        prices = self._require_prices(self.MOMENTUM_LONG + 1)
        if prices is None:
            return None
        current = prices.iloc[-1]
        past = prices.iloc[-self.MOMENTUM_LONG - 1]
        if past == 0 or pd.isna(past):
            return None
        momentum = (current - past) / past
        metric_name = f'momentum_{self.MOMENTUM_LONG}d'
        return self._store_result(metric_name, momentum)
    
    def calculate_rsi_current(self) -> Optional[float]:
        """Calculate RSI using Wilder's exponential moving average."""
        if self._require_prices(self.RSI_PERIOD + 1) is None:
            return None
        delta = self.history['Close'].diff()
        alpha = 1 / self.RSI_PERIOD
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        rs = rs.where((avg_loss != 0) | (avg_gain != 0), np.nan)
        rs = rs.mask((avg_loss == 0) & (avg_gain > 0), np.inf)
        rsi = 100 - (100 / (1 + rs))
        last_val = rsi.iloc[-1]
        if pd.isna(last_val):
            last_gain = avg_gain.iloc[-1]
            last_loss = avg_loss.iloc[-1]
            if last_loss == 0 and last_gain > 0:
                last_val = 100.0
            elif last_loss == 0 and last_gain == 0:
                last_val = 50.0
            else:
                last_val = None
        metric_name = f'rsi_{self.RSI_PERIOD}_current'
        return self._store_result(metric_name, last_val)
    
    def calculate_rsi_mean(self) -> Optional[float]:
        """Calculate mean RSI over the period."""
        if self._require_prices(self.RSI_PERIOD + 1) is None:
            return None
        delta = self.history['Close'].diff()
        alpha = 1 / self.RSI_PERIOD
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        rs = rs.where((avg_loss != 0) | (avg_gain != 0), np.nan)
        rs = rs.mask((avg_loss == 0) & (avg_gain > 0), np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi_mean = rsi.dropna().mean() if not rsi.dropna().empty else None
        metric_name = f'rsi_{self.RSI_PERIOD}_mean'
        return self._store_result(metric_name, rsi_mean)
    
    def calculate_macd(self) -> Optional[float]:
        """Calculate MACD line."""
        if self._require_prices(self.MACD_SLOW) is None:
            return None
        prices = self._require_prices(self.MACD_SLOW)
        if prices is None:
            return None
        ema_fast = prices.ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = prices.ewm(span=self.MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        return self._store_result('macd_current', macd.iloc[-1])
    
    def calculate_macd_signal(self) -> Optional[float]:
        """Calculate MACD signal line."""
        if self._require_prices(self.MACD_SLOW + self.MACD_SIGNAL) is None:
            return None
        prices = self._require_prices(self.MACD_SLOW + self.MACD_SIGNAL)
        if prices is None:
            return None
        ema_fast = prices.ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = prices.ewm(span=self.MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        return self._store_result('macd_signal_current', macd_signal.iloc[-1])
    
    def calculate_macd_histogram(self) -> Optional[float]:
        """Calculate MACD histogram."""
        if self._require_prices(self.MACD_SLOW + self.MACD_SIGNAL) is None:
            return None
        prices = self._require_prices(self.MACD_SLOW + self.MACD_SIGNAL)
        if prices is None:
            return None
        ema_fast = prices.ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = prices.ewm(span=self.MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        macd_hist = macd - macd_signal
        return self._store_result('macd_histogram_current', macd_hist.iloc[-1])
    
    def calculate_momentum_group(self) -> Dict[str, Any]:
        """Calculate all momentum metrics."""
        return self._collect_new_results([
            self.calculate_roc,
            self.calculate_momentum_short,
            self.calculate_momentum_mid,
            self.calculate_momentum_long,
            self.calculate_rsi_current,
            self.calculate_rsi_mean,
            self.calculate_macd,
            self.calculate_macd_signal,
            self.calculate_macd_histogram
        ])
    
    def calculate_performance_group(self) -> Dict[str, Any]:
        """Calculate all performance metrics."""
        return self._collect_new_results([
            self.calculate_sharpe_ratio,
            self.calculate_sortino_ratio,
            self.calculate_max_drawdown,
            self.calculate_current_drawdown,
            self.calculate_calmar_ratio,
            self.calculate_win_rate,
            self.calculate_profit_factor,
            self.calculate_max_consecutive_wins,
            self.calculate_max_consecutive_losses
        ])
    
    def calculate_sharpe_ratio(self) -> Optional[float]:
        """Calculate annualized Sharpe ratio."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        excess_returns = series - self._risk_free_rate_daily
        if excess_returns.std() == 0 or pd.isna(excess_returns.std()):
            return None
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return self._store_result('sharpe_ratio', sharpe)
    
    def calculate_sortino_ratio(self) -> Optional[float]:
        """Calculate annualized Sortino ratio."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        excess_returns = series - self._risk_free_rate_daily
        downside_returns = series[series < self._risk_free_rate_daily]
        if len(downside_returns) < 2:
            return None
        downside_vol = downside_returns.std()
        if downside_vol == 0 or pd.isna(downside_vol):
            return None
        sortino = excess_returns.mean() / downside_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return self._store_result('sortino_ratio', sortino)
    
    def calculate_max_drawdown(self) -> Optional[float]:
        """Calculate maximum drawdown."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        cumulative_returns = (1 + series.fillna(0)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return self._store_result('max_drawdown', drawdown.min())
    
    def calculate_current_drawdown(self) -> Optional[float]:
        """Calculate current drawdown from peak."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        cumulative_returns = (1 + series.fillna(0)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return self._store_result('current_drawdown', drawdown.iloc[-1])
    
    def calculate_calmar_ratio(self) -> Optional[float]:
        """Calculate Calmar ratio."""
        series = self._require_returns(self.MIN_DATA_POINTS)
        if series is None:
            return None
        cumulative_returns = (1 + series.fillna(0)).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()
        if max_dd == 0 or abs(max_dd) < 0.0001 or pd.isna(max_dd):
            return None
        calmar = total_return / abs(max_dd)
        return self._store_result('calmar_ratio', calmar)
    
    def calculate_win_rate(self) -> Optional[float]:
        """Calculate win rate."""
        series = self._require_returns(self.MIN_DATA_POINTS - 1)
        if series is None:
            return None
        valid_returns = series
        win_count = (valid_returns > 0).sum()
        total_count = len(valid_returns)
        if total_count == 0:
            return None
        return self._store_result('win_rate', win_count / total_count)
    
    def calculate_profit_factor(self) -> Optional[float]:
        """Calculate profit factor."""
        series = self._require_returns(self.MIN_DATA_POINTS)
        if series is None:
            return None
        valid_returns = series
        gains = valid_returns[valid_returns > 0].sum()
        losses = abs(valid_returns[valid_returns < 0].sum())
        if losses < 0.0001:
            return None
        return self._store_result('profit_factor', gains / losses)
    
    def calculate_max_consecutive_wins(self) -> Optional[int]:
        """Calculate maximum consecutive winning days."""
        series = self._require_returns(self.MIN_DATA_POINTS)
        if series is None:
            return None
        positive = (series > 0).astype(int)
        consecutive = positive.groupby((positive != positive.shift()).cumsum()).sum()
        max_wins = consecutive[consecutive > 0].max() if len(consecutive[consecutive > 0]) > 0 else 0
        result = int(max_wins) if pd.notna(max_wins) else 0
        self.calculations['max_consecutive_wins'] = result
        return result
    
    def calculate_max_consecutive_losses(self) -> Optional[int]:
        """Calculate maximum consecutive losing days."""
        series = self._require_returns(self.MIN_DATA_POINTS)
        if series is None:
            return None
        negative = (series < 0).astype(int)
        consecutive = negative.groupby((negative != negative.shift()).cumsum()).sum()
        max_losses = consecutive[consecutive > 0].max() if len(consecutive[consecutive > 0]) > 0 else 0
        result = int(max_losses) if pd.notna(max_losses) else 0
        self.calculations['max_consecutive_losses'] = result
        return result
    
    def calculate_price_ranges_group(self) -> Dict[str, Any]:
        """Calculate all price range metrics."""
        return self._collect_new_results([
            self.calculate_price_range_high,
            self.calculate_price_range_low,
            self.calculate_price_long_range_high,
            self.calculate_price_long_range_low,
            self.calculate_pct_from_long_range_high,
            self.calculate_pct_from_long_range_low,
            self.calculate_bollinger_upper,
            self.calculate_bollinger_lower,
            self.calculate_bollinger_width,
            self.calculate_bollinger_position,
            self.calculate_price_short_range_high,
            self.calculate_price_short_range_low,
            self.calculate_price_mid_range_high,
            self.calculate_price_mid_range_low
        ])
    
    def calculate_price_range_high(self) -> Optional[float]:
        """Calculate all-time high price."""
        if not self._has_ohlc():
            return None
        return self._store_result('price_range_high', self.history['High'].max())
    
    def calculate_price_range_low(self) -> Optional[float]:
        """Calculate all-time low price."""
        if not self._has_ohlc():
            return None
        return self._store_result('price_range_low', self.history['Low'].min())
    
    def calculate_price_long_range_high(self) -> Optional[float]:
        """Calculate long-range high."""
        if 'High' not in self.history.columns or self.history.empty:
            return None
        if len(self.history) >= self.LONG_RANGE_PERIOD:
            high = self.history['High'].tail(self.LONG_RANGE_PERIOD).max()
        else:
            high = self.history['High'].max()
        metric_name = f'price_{self.LONG_RANGE_PERIOD}d_high'
        return self._store_result(metric_name, high)
    
    def calculate_price_long_range_low(self) -> Optional[float]:
        """Calculate long-range low."""
        if 'Low' not in self.history.columns or self.history.empty:
            return None
        if len(self.history) >= self.LONG_RANGE_PERIOD:
            low = self.history['Low'].tail(self.LONG_RANGE_PERIOD).min()
        else:
            low = self.history['Low'].min()
        metric_name = f'price_{self.LONG_RANGE_PERIOD}d_low'
        return self._store_result(metric_name, low)
    
    def calculate_pct_from_long_range_high(self) -> Optional[float]:
        """Calculate percentage from long-range high."""
        if not self._has_ohlc():
            return None
        if self.history.empty:
            return None
        current_price = self._latest_close()
        if len(self.history) >= self.LONG_RANGE_PERIOD:
            high = self.history['High'].tail(self.LONG_RANGE_PERIOD).max()
        else:
            high = self.history['High'].max()
        if high == 0 or pd.isna(high):
            return None
        pct = (current_price - high) / high
        metric_name = f'pct_from_{self.LONG_RANGE_PERIOD}d_high'
        return self._store_result(metric_name, pct)
    
    def calculate_pct_from_long_range_low(self) -> Optional[float]:
        """Calculate percentage from long-range low."""
        if not self._has_ohlc():
            return None
        if self.history.empty:
            return None
        current_price = self._latest_close()
        if len(self.history) >= self.LONG_RANGE_PERIOD:
            low = self.history['Low'].tail(self.LONG_RANGE_PERIOD).min()
        else:
            low = self.history['Low'].min()
        if low == 0 or pd.isna(low):
            return None
        pct = (current_price - low) / low
        metric_name = f'pct_from_{self.LONG_RANGE_PERIOD}d_low'
        return self._store_result(metric_name, pct)
    
    def calculate_bollinger_upper(self) -> Optional[float]:
        """Calculate Bollinger Band upper band."""
        prices = self._require_prices(self.BOLLINGER_PERIOD)
        if prices is None:
            return None
        sma = prices.rolling(self.BOLLINGER_PERIOD).mean()
        std = prices.rolling(self.BOLLINGER_PERIOD).std()
        bb_upper = sma + (self.BOLLINGER_STDEV * std)
        metric_name = f'bollinger_upper_{self.BOLLINGER_PERIOD}d_{self.BOLLINGER_STDEV}std'
        return self._store_result(metric_name, bb_upper.iloc[-1])
    
    def calculate_bollinger_lower(self) -> Optional[float]:
        """Calculate Bollinger Band lower band."""
        prices = self._require_prices(self.BOLLINGER_PERIOD)
        if prices is None:
            return None
        sma = prices.rolling(self.BOLLINGER_PERIOD).mean()
        std = prices.rolling(self.BOLLINGER_PERIOD).std()
        bb_lower = sma - (self.BOLLINGER_STDEV * std)
        metric_name = f'bollinger_lower_{self.BOLLINGER_PERIOD}d_{self.BOLLINGER_STDEV}std'
        return self._store_result(metric_name, bb_lower.iloc[-1])
    
    def calculate_bollinger_width(self) -> Optional[float]:
        """Calculate Bollinger Band width."""
        prices = self._require_prices(self.BOLLINGER_PERIOD)
        if prices is None:
            return None
        sma = prices.rolling(self.BOLLINGER_PERIOD).mean()
        std = prices.rolling(self.BOLLINGER_PERIOD).std()
        bb_upper = sma + (self.BOLLINGER_STDEV * std)
        bb_lower = sma - (self.BOLLINGER_STDEV * std)
        bb_width = (bb_upper - bb_lower) / sma
        metric_name = f'bollinger_width_{self.BOLLINGER_PERIOD}d_{self.BOLLINGER_STDEV}std'
        return self._store_result(metric_name, bb_width.iloc[-1])
    
    def calculate_bollinger_position(self) -> Optional[float]:
        """Calculate price position within Bollinger Bands."""
        prices = self._require_prices(self.BOLLINGER_PERIOD)
        if prices is None:
            return None
        sma = prices.rolling(self.BOLLINGER_PERIOD).mean()
        std = prices.rolling(self.BOLLINGER_PERIOD).std()
        bb_upper = sma + (self.BOLLINGER_STDEV * std)
        bb_lower = sma - (self.BOLLINGER_STDEV * std)
        band_range = bb_upper - bb_lower
        if band_range.iloc[-1] == 0 or pd.isna(band_range.iloc[-1]):
            return None
        bb_position = (prices - bb_lower) / band_range
        metric_name = f'bollinger_position_{self.BOLLINGER_PERIOD}d_{self.BOLLINGER_STDEV}std'
        return self._store_result(metric_name, bb_position.iloc[-1])
    
    def calculate_price_short_range_high(self) -> Optional[float]:
        """Calculate short-range high."""
        if 'High' not in self.history.columns or len(self.history) < self.SHORT_RANGE_PERIOD:
            return None
        high = self.history['High'].tail(self.SHORT_RANGE_PERIOD).max()
        metric_name = f'price_{self.SHORT_RANGE_PERIOD}d_high'
        return self._store_result(metric_name, high)
    
    def calculate_price_short_range_low(self) -> Optional[float]:
        """Calculate short-range low."""
        if 'Low' not in self.history.columns or len(self.history) < self.SHORT_RANGE_PERIOD:
            return None
        low = self.history['Low'].tail(self.SHORT_RANGE_PERIOD).min()
        metric_name = f'price_{self.SHORT_RANGE_PERIOD}d_low'
        return self._store_result(metric_name, low)
    
    def calculate_price_mid_range_high(self) -> Optional[float]:
        """Calculate mid-range high."""
        if 'High' not in self.history.columns or len(self.history) < self.MID_RANGE_PERIOD:
            return None
        high = self.history['High'].tail(self.MID_RANGE_PERIOD).max()
        metric_name = f'price_{self.MID_RANGE_PERIOD}d_high'
        return self._store_result(metric_name, high)
    
    def calculate_price_mid_range_low(self) -> Optional[float]:
        """Calculate mid-range low."""
        if 'Low' not in self.history.columns or len(self.history) < self.MID_RANGE_PERIOD:
            return None
        low = self.history['Low'].tail(self.MID_RANGE_PERIOD).min()
        metric_name = f'price_{self.MID_RANGE_PERIOD}d_low'
        return self._store_result(metric_name, low)
    
    def calculate_volume_group(self) -> Dict[str, Any]:
        """Calculate all volume metrics."""
        return self._collect_new_results([
            self.calculate_avg_volume_total,
            self.calculate_avg_volume_recent,
            self.calculate_latest_volume,
            self.calculate_volume_trend
        ])
    
    def calculate_avg_volume_total(self) -> Optional[float]:
        """Calculate average volume over entire period."""
        if 'Volume' not in self.history.columns or self.history.empty:
            return None
        return self._store_result('avg_volume_total', self.history['Volume'].mean())
    
    def calculate_avg_volume_recent(self) -> Optional[float]:
        """Calculate average volume over recent period."""
        if 'Volume' not in self.history.columns or len(self.history) < self.VOLUME_MA_PERIOD:
            return None
        avg_vol = self.history['Volume'].tail(self.VOLUME_MA_PERIOD).mean()
        metric_name = f'avg_volume_{self.VOLUME_MA_PERIOD}d'
        return self._store_result(metric_name, avg_vol)
    
    def calculate_latest_volume(self) -> Optional[float]:
        """Get latest volume."""
        if 'Volume' not in self.history.columns or self.history.empty:
            return None
        return self._store_result('latest_volume', self.history['Volume'].iloc[-1])
    
    def calculate_volume_trend(self) -> Optional[float]:
        """Calculate volume trend."""
        if 'Volume' not in self.history.columns or len(self.history) < self.VOLUME_MA_PERIOD:
            return None
        latest = self.history['Volume'].iloc[-1]
        avg = self.history['Volume'].tail(self.VOLUME_MA_PERIOD).mean()
        if avg == 0 or pd.isna(avg):
            return None
        metric_name = f'volume_trend_{self.VOLUME_MA_PERIOD}d'
        return self._store_result(metric_name, latest / avg)
    
    def calculate_all(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute all or selected technical calculations."""
        if self.history.empty:
            return {}
        group_methods = {
            'returns': self.calculate_returns_group,
            'volatility': self.calculate_volatility_group,
            'moving_averages': self.calculate_moving_averages_group,
            'momentum': self.calculate_momentum_group,
            'performance': self.calculate_performance_group,
            'price_ranges': self.calculate_price_ranges_group,
            'volume': self.calculate_volume_group,
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
