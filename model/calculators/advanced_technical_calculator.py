import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional, List, Any

from model.calculators.technical_calculator import TechnicalCalculator


class AdvancedTechnicalCalculator(TechnicalCalculator):
    """Calculates advanced technical and statistical metrics."""
    
    VOLATILITY_WINDOW = 30
    VOLATILITY_ANNUALIZATION = 252
    DRAWDOWN_THRESHOLD = 0.001
    STRESS_PRICE_THRESHOLD = 0.999
    TAIL_QUANTILE = 0.05
    MIN_SAMPLES_SKEWNESS = 3
    MIN_SAMPLES_KURTOSIS = 4
    MIN_SAMPLES_TAIL = 20
    STOCHASTIC_PERIOD = 14
    STOCHASTIC_SIGNAL = 3
    CCI_PERIOD = 20
    CCI_CONSTANT = 0.015
    TREND_WINDOW = 21
    MIN_SAMPLES_HALF_LIFE = 30
    VARIANCE_RATIO_PERIOD = 5
    
    def __init__(self, history: pd.DataFrame):
        self.history = history.copy()
        self.calculations: Dict[str, Any] = {}
        self._aligned_data = self._align_ohlc_data()
        self._drawdown_cache: Optional[Dict[str, pd.Series]] = None
        self._stochastic_cache: Optional[Dict[str, pd.Series]] = None
    
    def _align_ohlc_data(self) -> Optional[pd.DataFrame]:
        """Align High, Low, Close data to common index."""
        required_cols = ['Close', 'High', 'Low']
        if not all(col in self.history.columns for col in required_cols):
            return None
        
        close = self.prices
        high = self.history['High'].dropna()
        low = self.history['Low'].dropna()
        
        common_idx = close.index.intersection(high.index).intersection(low.index)
        if len(common_idx) == 0:
            return None

        return pd.DataFrame({
            'Close': close.loc[common_idx],
            'High': high.loc[common_idx],
            'Low': low.loc[common_idx]
        })
    
    def _get_drawdown_data(self) -> Optional[Dict[str, pd.Series]]:
        """Compute and cache drawdown series."""
        if self._drawdown_cache is None:
            prices = self.prices
            if prices is None or len(prices) < 2:
                return None
            
            rolling_max = prices.cummax()
            safe_max = rolling_max.replace(0, np.nan)
            drawdown = (prices / safe_max - 1).fillna(0)
            
            self._drawdown_cache = {
                'prices': prices,
                'rolling_max': rolling_max,
                'drawdown': drawdown
            }
        return self._drawdown_cache
    
    def _get_stochastic_data(self) -> Optional[Dict[str, pd.Series]]:
        """Compute and cache stochastic oscillator lines."""
        if self._stochastic_cache is not None:
            return self._stochastic_cache
        
        if self._aligned_data is None:
            return None
        
        ohlc = self._aligned_data
        min_samples = self.STOCHASTIC_PERIOD + self.STOCHASTIC_SIGNAL
        
        if len(ohlc) < min_samples:
            return None
        
        try:
            high_n = ohlc['High'].rolling(window=self.STOCHASTIC_PERIOD).max()
            low_n = ohlc['Low'].rolling(window=self.STOCHASTIC_PERIOD).min()
            close = ohlc['Close']
            
            price_range = high_n - low_n
            price_range = price_range.replace(0, np.nan)
            
            k_line = ((close - low_n) / price_range * 100).dropna()
            d_line = k_line.rolling(window=self.STOCHASTIC_SIGNAL).mean().dropna()
            
            self._stochastic_cache = {
                'k_line': k_line,
                'd_line': d_line
            }
            return self._stochastic_cache
        except Exception:
            return None
    
    def _require_drawdown_data(self) -> Optional[Dict[str, pd.Series]]:
        return self._get_drawdown_data()
    
    def _require_stochastic_data(self, require_k: bool = True, require_d: bool = False) -> Optional[Dict[str, pd.Series]]:
        data = self._get_stochastic_data()
        if data is None:
            return None
        if require_k and data.get('k_line', pd.Series()).empty:
            return None
        if require_d and data.get('d_line', pd.Series()).empty:
            return None
        return data
    
    def _require_aligned_data(self, min_len: int = 0) -> Optional[pd.DataFrame]:
        data = self._aligned_data
        if data is None or len(data) < min_len:
            return None
        return data
    
    def calculate_return_skewness(self) -> Optional[float]:
        """Calculate return skewness. Positive = right tail longer, Negative = left tail longer."""
        returns = self._require_returns(self.MIN_SAMPLES_SKEWNESS)
        if returns is None:
            return None
        
        skew = returns.skew()
        return self._store_result('return_skewness', skew)
    
    def calculate_return_kurtosis(self) -> Optional[float]:
        """Calculate return kurtosis. Positive = heavier tails, Negative = lighter tails."""
        returns = self._require_returns(self.MIN_SAMPLES_KURTOSIS)
        if returns is None:
            return None
        
        kurt = returns.kurtosis()
        return self._store_result('return_kurtosis', kurt)
    
    def calculate_left_tail_frequency(self) -> Optional[float]:
        """Calculate frequency of extreme negative returns."""
        returns = self._require_returns(self.MIN_SAMPLES_TAIL)
        if returns is None:
            return None
        
        threshold = returns.quantile(self.TAIL_QUANTILE)
        freq = (returns < threshold).mean()
        return self._store_result(f'left_tail_frequency_{int(self.TAIL_QUANTILE * 100)}pct', freq)
    
    def calculate_max_drawdown_duration(self) -> Optional[int]:
        """Calculate maximum consecutive days in drawdown."""
        data = self._require_drawdown_data()
        if data is None:
            return None
        
        drawdown = data['drawdown']
        below = drawdown < -self.DRAWDOWN_THRESHOLD
        max_duration = duration = 0
        for flag in below:
            if pd.isna(flag):
                continue
            duration = duration + 1 if flag else 0
            max_duration = max(max_duration, duration)
        
        result = self._store_result('max_drawdown_duration_days', max_duration)
        return int(result) if result is not None else None
    
    def calculate_max_drawdown_recovery(self) -> Optional[int]:
        """Calculate maximum time to recover from drawdown."""
        data = self._require_drawdown_data()
        if data is None:
            return None
        
        drawdown = data['drawdown']
        recovery_start = None
        max_recovery = 0
        
        for i, dd in enumerate(drawdown):
            if not np.isfinite(dd):
                continue
            if dd < -self.DRAWDOWN_THRESHOLD:
                if recovery_start is None:
                    recovery_start = i
            else:
                if recovery_start is not None:
                    max_recovery = max(max_recovery, i - recovery_start)
                    recovery_start = None
        
        result = self._store_result('max_drawdown_recovery_time_days', max_recovery)
        return int(result) if result is not None else None
    
    def calculate_max_days_to_new_high(self) -> Optional[int]:
        """Calculate maximum days below previous high."""
        data = self._require_drawdown_data()
        if data is None:
            return None
        
        prices = data['prices']
        rolling_max = data['rolling_max']
        
        days_since_high = 0
        max_days = 0
        
        for price, peak in zip(prices, rolling_max):
            if not np.isfinite(price) or not np.isfinite(peak) or peak == 0:
                continue
            if price >= peak * self.STRESS_PRICE_THRESHOLD:
                if days_since_high > 0:
                    max_days = max(max_days, days_since_high)
                days_since_high = 0
            else:
                days_since_high += 1
        
        result = self._store_result('max_days_to_new_high_after_stress', max_days)
        return int(result) if result is not None else None
    
    def calculate_trend_break(self) -> Optional[int]:
        """Calculate trend break indicator. Returns 1 if price above recent low, 0 otherwise."""
        if self._require_prices(self.TREND_WINDOW) is None:
            return None
        prices = self.prices
        
        recent_window = prices.iloc[-self.TREND_WINDOW:]
        recent_min = recent_window.min()
        if recent_min <= 0:
            return None
        
        breakout = int(prices.iloc[-1] > recent_min * 1.001)
        self.calculations[f'trend_break_{self.TREND_WINDOW}d'] = breakout
        return breakout
    
    def calculate_trend_half_life(self) -> Optional[float]:
        """Calculate mean-reversion half-life using AR(1) regression on log prices."""
        log_prices = self._require_log_prices(self.MIN_SAMPLES_HALF_LIFE)
        
        if log_prices is None:
            return None
        
        try:
            y = log_prices.iloc[1:].values
            x = log_prices.iloc[:-1].values
            
            if len(y) != len(x) or len(y) < 10:
                return None
            
            x_const = sm.add_constant(x)
            model = sm.OLS(y, x_const).fit()
            beta = model.params[1]
            
            if 0 < beta < 1:
                half_life = -np.log(2) / np.log(beta)
                if 1 <= half_life <= 504:
                    return self._store_result('trend_half_life_days', half_life)
            
            return None
        except Exception:
            return None
    
    def calculate_stochastic_k(self) -> Optional[float]:
        """Calculate current Stochastic %K value."""
        data = self._require_stochastic_data(require_k=True, require_d=False)
        if data is None:
            return None
        
        k_val = data['k_line'].iloc[-1]
        return self._store_result(f'stochastic_{self.STOCHASTIC_PERIOD}_k_current', k_val)
    
    def calculate_stochastic_d(self) -> Optional[float]:
        """Calculate current Stochastic %D value."""
        data = self._require_stochastic_data(require_k=False, require_d=True)
        if data is None:
            return None
        
        d_val = data['d_line'].iloc[-1]
        return self._store_result(f'stochastic_{self.STOCHASTIC_PERIOD}_{self.STOCHASTIC_SIGNAL}_d_current', d_val)
    
    def calculate_stochastic_diff(self) -> Optional[float]:
        """Calculate Stochastic %K - %D difference."""
        data = self._require_stochastic_data(require_k=True, require_d=True)
        if data is None:
            return None
        
        k_d_diff = data['k_line'].iloc[-1] - data['d_line'].iloc[-1]
        return self._store_result(f'stochastic_{self.STOCHASTIC_PERIOD}_k_d_difference', k_d_diff)
    
    def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate all drawdown metrics."""
        return self._collect_new_results([
            self.calculate_max_drawdown_duration,
            self.calculate_max_drawdown_recovery,
            self.calculate_max_days_to_new_high
        ])
    
    def calculate_return_distribution(self) -> Dict[str, Any]:
        """Calculate all return distribution statistics."""
        return self._collect_new_results([
            self.calculate_return_skewness,
            self.calculate_return_kurtosis,
            self.calculate_left_tail_frequency
        ])
    
    def calculate_volatility_regime(self) -> Dict[str, Any]:
        """Calculate volatility regime classification and conditional returns."""
        returns = self._require_returns(self.VOLATILITY_WINDOW + 1)
        results = {}
        
        if returns is None:
            return results
        
        rolling_vol = returns.rolling(self.VOLATILITY_WINDOW).std() * np.sqrt(self.VOLATILITY_ANNUALIZATION)
        rolling_vol_clean = rolling_vol.dropna()
        
        if len(rolling_vol_clean) < 3:
            return results
        
        low_threshold = rolling_vol_clean.quantile(0.33)
        high_threshold = rolling_vol_clean.quantile(0.66)
        
        last_vol = rolling_vol_clean.iloc[-1]
        if last_vol < low_threshold:
            current_regime = 0
        elif last_vol < high_threshold:
            current_regime = 1
        else:
            current_regime = 2
        
        regimes = pd.Series(0, index=rolling_vol_clean.index)
        regimes[rolling_vol_clean >= low_threshold] = 1
        regimes[rolling_vol_clean >= high_threshold] = 2
        
        returns_aligned = returns.reindex(rolling_vol_clean.index)
        regime_returns = returns_aligned[regimes == current_regime].dropna()
        
        if not regime_returns.empty:
            mean_ret = regime_returns.mean()
            regime_return_key = f'regime_{self.VOLATILITY_WINDOW}d_conditional_mean_return'
            volatility_regime_key = f'volatility_regime_{self.VOLATILITY_WINDOW}d'
            self._store_result(regime_return_key, mean_ret)
            self.calculations[volatility_regime_key] = int(current_regime)
            results[regime_return_key] = self.calculations.get(regime_return_key)
            results[volatility_regime_key] = current_regime
        
        return results
    
    def calculate_autocorrelation(self) -> Dict[str, Any]:
        """Calculate return autocorrelation at lag 1."""
        returns = self._require_returns(3)
        results = {}
        
        if returns is None:
            return results
        
        autocorr = returns.autocorr(lag=1)
        result = self._store_result('return_autocorrelation_lag1', autocorr)
        if result is not None:
            results['return_autocorrelation_lag1'] = result
        
        return results
    
    def calculate_trend_metrics(self) -> Dict[str, Any]:
        """Calculate all trend metrics."""
        return self._collect_new_results([
            self.calculate_trend_break,
            self.calculate_trend_half_life
        ])
    
    def calculate_variance_ratio(self) -> Dict[str, Any]:
        """Calculate variance ratio test statistic."""
        k = self.VARIANCE_RATIO_PERIOD
        results = {}
        
        returns = self._require_returns(k * 3)
        if returns is None:
            return results
        
        try:
            k_returns = returns.rolling(k).sum().dropna()
            
            if len(k_returns) < k:
                return results
            
            var_1d = returns.var()
            var_kd = k_returns.var()
            
            if var_1d > 1e-12:
                vr = var_kd / (k * var_1d)
                variance_ratio_key = f'variance_ratio_{self.VARIANCE_RATIO_PERIOD}d'
                result = self._store_result(variance_ratio_key, vr)
                if result is not None:
                    results[variance_ratio_key] = result
        except Exception:
            pass
        
        return results
    
    def calculate_stochastic(self) -> Dict[str, Any]:
        """Calculate all Stochastic Oscillator metrics."""
        return self._collect_new_results([
            self.calculate_stochastic_k,
            self.calculate_stochastic_d,
            self.calculate_stochastic_diff
        ])
    
    def calculate_cci(self) -> Dict[str, Any]:
        """Calculate Commodity Channel Index."""
        results = {}
        
        ohlc = self._require_aligned_data(self.CCI_PERIOD)
        if ohlc is None:
            return results
        
        try:
            tp = (ohlc['High'] + ohlc['Low'] + ohlc['Close']) / 3
            tp_sma = tp.rolling(self.CCI_PERIOD).mean()
            
            def calc_mad(x):
                return np.abs(x - x.mean()).mean()
            
            mad = tp.rolling(self.CCI_PERIOD).apply(calc_mad, raw=False)
            mad = mad.replace(0, np.nan)
            
            cci = (tp - tp_sma) / (self.CCI_CONSTANT * mad)
            cci_clean = cci.dropna()
            
            if not cci_clean.empty:
                cci_val = cci_clean.iloc[-1]
                cci_key = f'cci_{self.CCI_PERIOD}_current'
                result = self._store_result(cci_key, cci_val)
                if result is not None:
                    results[cci_key] = result
        except Exception:
            pass
        
        return results
    
    def calculate_all(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute all or selected advanced technical calculations."""
        if self.history.empty or len(self.history) < 3:
            return {}
        
        group_methods = {
            'drawdown': self.calculate_drawdown_metrics,
            'distribution': self.calculate_return_distribution,
            'volatility_regime': self.calculate_volatility_regime,
            'autocorrelation': self.calculate_autocorrelation,
            'trend': self.calculate_trend_metrics,
            'variance_ratio': self.calculate_variance_ratio,
            'stochastic': self.calculate_stochastic,
            'cci': self.calculate_cci,
        }
        
        individual_methods = {
            'return_skewness': self.calculate_return_skewness,
            'return_kurtosis': self.calculate_return_kurtosis,
            'left_tail_frequency': self.calculate_left_tail_frequency,
            'max_drawdown_duration': self.calculate_max_drawdown_duration,
            'max_drawdown_recovery': self.calculate_max_drawdown_recovery,
            'max_days_to_new_high': self.calculate_max_days_to_new_high,
            'trend_break': self.calculate_trend_break,
            'trend_half_life': self.calculate_trend_half_life,
            'stochastic_k': self.calculate_stochastic_k,
            'stochastic_d': self.calculate_stochastic_d,
            'stochastic_diff': self.calculate_stochastic_diff,
        }
        
        if metrics is None:
            for method in group_methods.values():
                method()
        else:
            for metric_name in metrics:
                if metric_name in group_methods:
                    group_methods[metric_name]()
                elif metric_name in individual_methods:
                    individual_methods[metric_name]()
        
        return self.calculations.copy()
