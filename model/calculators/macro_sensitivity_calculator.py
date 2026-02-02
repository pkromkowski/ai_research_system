import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional, List, Any

from model.calculators.calculator_base import CalculatorBase


class MacroSensitivityCalculator(CalculatorBase):
    """Calculates stock sensitivity to macroeconomic indicators."""

    ROLLING_WINDOW = 63
    TAIL_QUANTILE = 0.05
    REGIME_LOW_QUANTILE = 0.33
    REGIME_HIGH_QUANTILE = 0.66
    MIN_SAMPLES_REGRESSION = 30

    MACRO_TREATMENT = {
        'GS2': 'level_change',
        'GS10': 'level_change',
        'GS30': 'level_change',
        'DFF': 'level_change',
        'T10Y2Y': 'level_change',
        'CPIAUCSL': 'pct_change',
        'INDPRO': 'pct_change',
        'PAYEMS': 'pct_change',
        'GDPC1': 'pct_change',
        'UNRATE': 'level_change',
        'UMCSENT': 'level_change',
    }
    
    def __init__(self, history: pd.DataFrame):
        """
        Initialize calculator with price history.
        
        Args:
            history: DataFrame with 'Close' column and DatetimeIndex
        """
        self.history = history.copy()
        self.calculations: Dict[str, Any] = {}
        
        self.macro_aligned: Dict[str, pd.Series] = {}
        self.macro_returns: Dict[str, pd.Series] = {}
        self._macro_data_ready = False
    
    def _require_macro_ready(self) -> bool:
        """Check if macro data is ready and sufficient."""
        series = self.returns
        return self._macro_data_ready and series is not None and len(series) >= self.MIN_SAMPLES_REGRESSION

    def _align_with_factor(self, factor_name: str, use_returns: bool = True):
        """Align stock series with macro factor."""
        if not self._macro_data_ready:
            return None, None
        series = self.macro_returns.get(factor_name) if use_returns else self.macro_aligned.get(factor_name)
        if series is None:
            return None, None
        try:
            stock_series = self.returns
            if stock_series is None:
                return None, None
            stock_aligned, factor_aligned = stock_series.align(series, join='inner')
            return stock_aligned, factor_aligned
        except Exception:
            return None, None

    def _regime_mask_and_suffix(self, factor_aligned: pd.Series, regime: str = 'high'):
        """Generate regime mask and metric suffix."""
        if regime == 'high':
            threshold = factor_aligned.quantile(self.REGIME_HIGH_QUANTILE)
            mask = factor_aligned > threshold
            suffix = f'p{int(self.REGIME_HIGH_QUANTILE * 100)}'
        else:
            threshold = factor_aligned.quantile(self.REGIME_LOW_QUANTILE)
            mask = factor_aligned < threshold
            suffix = f'p{int(self.REGIME_LOW_QUANTILE * 100)}'
        return mask.astype(bool), suffix
    
    def calculate_all(self, macro_data: Dict[str, pd.Series] = None,
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute all or selected macro sensitivity calculations.
        
        Args:
            macro_data: Dictionary of {factor_name: pd.Series} with macro data
            metrics: Optional list of metric groups to calculate. If None, calculates all.
                     Valid options: ['rolling_betas', 'conditional_performance',
                                    'regime_drawdowns', 'correlation_metrics',
                                    'tail_risk', 'multifactor_regression']
        
        Returns:
            Dictionary of all calculated metrics
        """
        if not macro_data or len(macro_data) == 0:
            return {}
        
        if self.history.empty or len(self.history) < self.ROLLING_WINDOW:
            return {}
        
        if not self._setup_macro_data(macro_data):
            return {}
        
        group_methods = {
            'rolling_betas': self.calculate_rolling_betas,
            'conditional_performance': self.calculate_conditional_performance,
            'regime_drawdowns': self.calculate_regime_drawdowns,
            'correlation_metrics': self.calculate_correlation_metrics,
            'tail_risk': self.calculate_tail_risk,
            'multifactor_regression': self.calculate_multifactor_regression,
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
    
    def _setup_macro_data(self, macro_data: Dict[str, pd.Series]) -> bool:
        """
        Align and prepare macro data with proper transformations.
        
        Returns:
            True if setup successful, False otherwise
        """
        if not self._has_close():
            return False

        stock_returns = self.returns
        if stock_returns is None or len(stock_returns) < self.MIN_SAMPLES_REGRESSION:
            return False
        
        macro_aligned = {}
        macro_returns = {}
        
        for factor_name, factor_series in macro_data.items():
            if not isinstance(factor_series, pd.Series):
                continue
            
            # Forward fill to align with daily stock data
            aligned = factor_series.reindex(stock_returns.index, method='ffill').dropna()
            if len(aligned) == 0:
                continue
            
            macro_aligned[factor_name] = aligned
            
            # Determine transformation based on series type
            treatment = self.MACRO_TREATMENT.get(factor_name, 'pct_change')
            
            if treatment == 'level_change':
                # Treasury rates, Fed Funds, UNRATE, UMCSENT: use level changes
                transformed = aligned.diff().dropna()
            else:  # treatment == 'pct_change'
                # CPI, Industrial Production, Payroll, GDP: use percentage changes
                transformed = aligned.pct_change().dropna()
            
            if len(transformed) > 0:
                macro_returns[factor_name] = transformed
        
        if len(macro_aligned) == 0:
            return False
        
        # stock returns are derived from ReturnsMixin; store macro data
        self.macro_aligned = macro_aligned
        self.macro_returns = macro_returns
        self._macro_data_ready = True
        return True
    
    def calculate_rolling_beta(self, factor_name: str) -> Optional[float]:
        """
        Calculate rolling beta vs a specific macro factor.
        
        Beta = Cov(stock, factor) / Var(factor)
        
        Args:
            factor_name: Name of the macro factor
        
        Returns:
            Rolling beta value, or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=True)
        if stock_aligned is None or len(stock_aligned) < self.ROLLING_WINDOW:
            return None
        
        try:
            
            rolling_cov = stock_aligned.rolling(self.ROLLING_WINDOW).cov(factor_aligned)
            rolling_var = factor_aligned.rolling(self.ROLLING_WINDOW).var()
            
            # Avoid division by zero
            if rolling_var.iloc[-1] == 0 or pd.isna(rolling_var.iloc[-1]):
                return None
            
            beta = rolling_cov.iloc[-1] / rolling_var.iloc[-1]
            
            metric_name = f'macro_rolling_beta_{self.ROLLING_WINDOW}d_vs_{factor_name}'
            return self._store_result(metric_name, beta)
        except Exception:
            return None
    
    def calculate_regime_performance(self, factor_name: str, regime: str = 'high') -> Optional[float]:
        """
        Calculate average stock performance in a macro regime.
        
        Args:
            factor_name: Name of the macro factor
            regime: 'high' or 'low'
        
        Returns:
            Mean return in the regime, or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=False)
        if stock_aligned is None or stock_aligned.empty or factor_aligned is None or factor_aligned.empty:
            return None
        
        try:
            mask, suffix = self._regime_mask_and_suffix(factor_aligned, regime)
            if not mask.any():
                return None
            
            mean_return = stock_aligned[mask].mean()
            metric_name = f'macro_performance_{regime}_{suffix}_{factor_name}_regime'
            return self._store_result(metric_name, mean_return)
        except Exception:
            return None
    
    def calculate_regime_volatility(self, factor_name: str, regime: str = 'high') -> Optional[float]:
        """
        Calculate stock volatility in a macro regime.
        
        Args:
            factor_name: Name of the macro factor
            regime: 'high' or 'low'
        
        Returns:
            Volatility (std dev) in the regime, or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=False)
        if stock_aligned is None or stock_aligned.empty or factor_aligned is None or factor_aligned.empty:
            return None
        
        try:
            mask, suffix = self._regime_mask_and_suffix(factor_aligned, regime)
            if not mask.any() or mask.sum() < 2:  # Need at least 2 for std
                return None
            
            volatility = stock_aligned[mask].std()
            metric_name = f'macro_volatility_{regime}_{suffix}_{factor_name}_regime'
            return self._store_result(metric_name, volatility)
        except Exception:
            return None
    
    def calculate_regime_drawdown(self, factor_name: str, regime: str = 'high') -> Optional[float]:
        """
        Calculate max drawdown during a macro regime.
        
        Uses cumulative returns to properly calculate drawdown.
        
        Args:
            factor_name: Name of the macro factor
            regime: 'high' or 'low'
        
        Returns:
            Maximum drawdown (negative value), or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=False)
        if stock_aligned is None or stock_aligned.empty or factor_aligned is None or factor_aligned.empty:
            return None
        
        try:
            mask, suffix = self._regime_mask_and_suffix(factor_aligned, regime)
            if not mask.any():
                return None
            
            # Get returns in this regime and calculate cumulative returns
            regime_returns = stock_aligned[mask]
            
            # Convert to cumulative wealth (1 + r1) * (1 + r2) * ...
            cumulative_wealth = (1 + regime_returns).cumprod()
            
            # Calculate drawdown from cumulative wealth
            rolling_max = cumulative_wealth.cummax()
            drawdown = (cumulative_wealth / rolling_max - 1).min()
            
            metric_name = f'macro_max_drawdown_{regime}_{suffix}_{factor_name}_regime'
            return self._store_result(metric_name, drawdown)
        except Exception:
            return None
    
    def calculate_correlation(self, factor_name: str) -> Optional[float]:
        """
        Calculate correlation between stock returns and macro factor changes.
        
        Args:
            factor_name: Name of the macro factor
        
        Returns:
            Correlation coefficient, or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=True)
        if stock_aligned is None or len(stock_aligned) < self.MIN_SAMPLES_REGRESSION:
            return None
        
        try:
            correlation = stock_aligned.corr(factor_aligned)
            metric_name = f'macro_correlation_vs_{factor_name}'
            return self._store_result(metric_name, correlation)
        except Exception:
            return None
    
    def calculate_rolling_correlation(self, factor_name: str) -> Optional[float]:
        """
        Calculate rolling correlation between stock returns and macro factor.
        
        Args:
            factor_name: Name of the macro factor
        
        Returns:
            Current rolling correlation, or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=True)
        if stock_aligned is None or len(stock_aligned) < self.ROLLING_WINDOW:
            return None
        
        try:
            rolling_corr = stock_aligned.rolling(self.ROLLING_WINDOW).corr(factor_aligned)
            current_corr = rolling_corr.iloc[-1]
            
            metric_name = f'macro_rolling_correlation_{self.ROLLING_WINDOW}d_vs_{factor_name}'
            return self._store_result(metric_name, current_corr)
        except Exception:
            return None
    
    def calculate_tail_risk(self, factor_name: str) -> Optional[float]:
        """
        Calculate worst stock return during extreme macro conditions.
        
        Uses TAIL_QUANTILE to define extreme conditions (both tails).
        
        Args:
            factor_name: Name of the macro factor
        
        Returns:
            Minimum return during extreme macro conditions, or None if unavailable
        """
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=True)
        if stock_aligned is None or len(stock_aligned) < self.MIN_SAMPLES_REGRESSION:
            return None
        
        try:
            low_thresh = factor_aligned.quantile(self.TAIL_QUANTILE)
            high_thresh = factor_aligned.quantile(1 - self.TAIL_QUANTILE)
            
            extreme_mask = (factor_aligned < low_thresh) | (factor_aligned > high_thresh)
            
            if not extreme_mask.any():
                return None
            
            tail_return = stock_aligned[extreme_mask].min()
            metric_name = f'macro_tail_risk_{int(self.TAIL_QUANTILE * 100)}pct_{factor_name}'
            return self._store_result(metric_name, tail_return)
        except Exception:
            return None
    
    def calculate_multifactor_beta(self, factor_name: str) -> Optional[float]:
        """
        Calculate beta for a specific factor from multi-factor regression.
        
        Runs OLS regression of stock returns on all available macro factors
        and returns the coefficient for the specified factor.
        
        Args:
            factor_name: Name of the macro factor
        
        Returns:
            Multi-factor beta coefficient, or None if unavailable
        """
        if not self._require_macro_ready():
            return None
        if factor_name not in self.macro_returns:
            return None
        
        if len(self.macro_returns) < 2:
            return None
        
        try:
            # Align all series
            aligned_dfs = [self.returns.rename('Stock')]
            for name, series in self.macro_returns.items():
                aligned_dfs.append(series.rename(name))
            
            aligned_df = pd.concat(aligned_dfs, axis=1).dropna()
            
            if len(aligned_df) < self.MIN_SAMPLES_REGRESSION:
                return None
            
            y = aligned_df['Stock']
            X = aligned_df[list(self.macro_returns.keys())]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            
            coef = model.params.get(factor_name)
            if coef is None:
                return None
            
            metric_name = f'macro_multifactor_beta_{factor_name}'
            return self._store_result(metric_name, coef)
        except Exception:
            return None
    
    def calculate_rolling_betas(self, factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate rolling betas for all or specified macro factors.
        
        Args:
            factors: Optional list of factor names. If None, calculates for all.
        
        Returns:
            Dictionary of rolling beta metrics
        """
        if not self._macro_data_ready:
            return {}
        
        if factors is None:
            factors = list(self.macro_returns.keys())
        
        def _runner():
            for factor_name in factors:
                self.calculate_rolling_beta(factor_name)
        
        return self._collect_new_results([_runner])
    
    def calculate_conditional_performance(self, factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate performance and volatility in high/low macro regimes.
        
        Args:
            factors: Optional list of factor names. If None, calculates for all.
        
        Returns:
            Dictionary of conditional performance metrics
        """
        if not self._macro_data_ready:
            return {}
        
        if factors is None:
            factors = list(self.macro_aligned.keys())
        
        def _runner():
            for factor_name in factors:
                for regime in ['high', 'low']:
                    self.calculate_regime_performance(factor_name, regime)
                    self.calculate_regime_volatility(factor_name, regime)
        
        return self._collect_new_results([_runner])
    
    def calculate_regime_drawdowns(self, factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate max drawdowns in macro regimes.
        
        Args:
            factors: Optional list of factor names. If None, calculates for all.
        
        Returns:
            Dictionary of regime drawdown metrics
        """
        if not self._macro_data_ready:
            return {}
        
        if factors is None:
            factors = list(self.macro_aligned.keys())
        
        def _runner():
            for factor_name in factors:
                for regime in ['high', 'low']:
                    self.calculate_regime_drawdown(factor_name, regime)
        
        return self._collect_new_results([_runner])
    
    def calculate_correlation_metrics(self, factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate correlation metrics vs macro factors.
        
        Args:
            factors: Optional list of factor names. If None, calculates for all.
        
        Returns:
            Dictionary of correlation metrics
        """
        if not self._macro_data_ready:
            return {}
        
        if factors is None:
            factors = list(self.macro_returns.keys())
        
        def _runner():
            for factor_name in factors:
                self.calculate_correlation(factor_name)
                self.calculate_rolling_correlation(factor_name)
        
        return self._collect_new_results([_runner])
    
    def calculate_tail_risk(self, factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate tail risk in extreme macro conditions.
        
        Args:
            factors: Optional list of factor names. If None, calculates for all.
        
        Returns:
            Dictionary of tail risk metrics
        """
        if not self._macro_data_ready:
            return {}
        
        if factors is None:
            factors = list(self.macro_returns.keys())
        
        def _runner():
            for factor_name in factors:
                self._calculate_tail_risk_for_factor(factor_name)
        
        return self._collect_new_results([_runner])
    
    def _calculate_tail_risk_for_factor(self, factor_name: str) -> Optional[float]:
        """Internal helper to avoid name collision with group method."""
        stock_aligned, factor_aligned = self._align_with_factor(factor_name, use_returns=True)
        if stock_aligned is None or len(stock_aligned) < self.MIN_SAMPLES_REGRESSION:
            return None
        
        try:
            low_thresh = factor_aligned.quantile(self.TAIL_QUANTILE)
            high_thresh = factor_aligned.quantile(1 - self.TAIL_QUANTILE)
            
            extreme_mask = (factor_aligned < low_thresh) | (factor_aligned > high_thresh)
            
            if not extreme_mask.any():
                return None
            
            tail_return = stock_aligned[extreme_mask].min()
            metric_name = f'macro_tail_risk_{int(self.TAIL_QUANTILE * 100)}pct_{factor_name}'
            return self._store_result(metric_name, tail_return)
        except Exception:
            return None
    
    def calculate_multifactor_regression(self) -> Dict[str, Any]:
        """
        Calculate multi-factor regression betas for all factors.
        
        Returns:
            Dictionary of multi-factor beta metrics
        """
        if not self._macro_data_ready or len(self.macro_returns) < 2:
            return {}
        
        def _runner():
            try:
                # Align all series
                aligned_dfs = [self.returns.rename('Stock')]
                for name, series in self.macro_returns.items():
                    aligned_dfs.append(series.rename(name))
                
                aligned_df = pd.concat(aligned_dfs, axis=1).dropna()
                
                if len(aligned_df) < self.MIN_SAMPLES_REGRESSION:
                    return
                
                y = aligned_df['Stock']
                X = aligned_df[list(self.macro_returns.keys())]
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                
                for factor_name in self.macro_returns.keys():
                    coef = model.params.get(factor_name)
                    if coef is not None:
                        metric_name = f'macro_multifactor_beta_{factor_name}'
                        self._store_result(metric_name, coef)
                
                # Also store R-squared
                self._store_result('macro_multifactor_r_squared', model.rsquared)
            except Exception:
                pass
        
        return self._collect_new_results([_runner])
