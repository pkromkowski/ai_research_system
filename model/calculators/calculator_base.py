import numpy as np
import pandas as pd
from typing import Optional, List, Any, Dict


class CalculatorBase:
    """Base class providing cached price/return data and result helpers.
    
    Requires subclass to have `history` DataFrame with 'Close' column.
    """

    @property
    def returns(self) -> Optional[pd.Series]:
        """Daily percent-change returns (tz-normalized, cleaned)."""
        cache = getattr(self, "_returns_cache", None)
        if cache is not None:
            return cache

        prices = self.prices
        if prices is None:
            setattr(self, "_returns_cache", None)
            return None

        series = prices.pct_change().dropna()
        if series.empty:
            setattr(self, "_returns_cache", None)
            return None

        setattr(self, "_returns_cache", series)
        return series

    @property
    def prices(self) -> Optional[pd.Series]:
        """Close price series (cleaned, tz-naive)."""
        cache = getattr(self, "_prices_cache", None)
        if cache is not None:
            return cache

        hist = getattr(self, "history", None)
        if hist is None or 'Close' not in hist.columns:
            setattr(self, "_prices_cache", None)
            return None

        series = hist['Close'].dropna()
        if series.empty:
            setattr(self, "_prices_cache", None)
            return None

        if series.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_localize(None)

        setattr(self, "_prices_cache", series)
        return series

    @property
    def log_returns(self) -> Optional[pd.Series]:
        """Log returns."""
        cache = getattr(self, "_log_returns_cache", None)
        if cache is not None:
            return cache

        prices = self.prices
        if prices is None:
            setattr(self, "_log_returns_cache", None)
            return None

        series = np.log(prices / prices.shift(1)).dropna()
        if series.empty:
            setattr(self, "_log_returns_cache", None)
            return None

        setattr(self, "_log_returns_cache", series)
        return series

    @property
    def log_prices(self) -> Optional[pd.Series]:
        """Log of prices."""
        cache = getattr(self, "_log_prices_cache", None)
        if cache is not None:
            return cache

        prices = self.prices
        if prices is None:
            setattr(self, "_log_prices_cache", None)
            return None

        series = np.log(prices.where(prices > 0, np.nan)).dropna()
        if series.empty:
            setattr(self, "_log_prices_cache", None)
            return None

        setattr(self, "_log_prices_cache", series)
        return series

    def _require_returns(self, min_len: int) -> Optional[pd.Series]:
        """Returns if length >= min_len, else None."""
        series = self.returns
        if series is None or len(series) < min_len:
            return None
        return series

    def _require_prices(self, min_len: int) -> Optional[pd.Series]:
        """Prices if length >= min_len, else None."""
        series = self.prices
        if series is None or len(series) < min_len:
            return None
        return series

    def _require_log_prices(self, min_len: int) -> Optional[pd.Series]:
        """Log prices if length >= min_len, else None."""
        series = self.log_prices
        if series is None or len(series) < min_len:
            return None
        return series

    def _require_log_returns(self, min_len: int) -> Optional[pd.Series]:
        """Log returns if length >= min_len, else None."""
        series = self.log_returns
        if series is None or len(series) < min_len:
            return None
        return series

    def _has_volume(self) -> bool:
        """Check if Volume column exists."""
        hist = getattr(self, 'history', None)
        if hist is None:
            return False
        return 'Volume' in hist.columns and not hist.empty

    def _has_close(self) -> bool:
        """Check if Close column exists."""
        hist = getattr(self, 'history', None)
        if hist is None:
            return False
        return 'Close' in hist.columns and not hist.empty

    def _clean_result(self, value: Any) -> Optional[float]:
        """Clean result (None/NaN/inf -> None)."""
        if value is None or pd.isna(value) or not np.isfinite(value):
            return None
        return float(value)

    def _store_result(self, key: str, value: Any) -> Optional[float]:
        """Store cleaned result in calculations dict."""
        cleaned = self._clean_result(value)
        if cleaned is not None:
            if not hasattr(self, 'calculations'):
                self.calculations = {}
            self.calculations[key] = cleaned
        return cleaned

    def _collect_new_results(self, callables: List[Any]) -> Dict[str, Any]:
        """Execute callables and return newly-added calculations."""
        if not hasattr(self, 'calculations'):
            self.calculations = {}
        before = set(self.calculations.keys())
        for fn in callables:
            try:
                fn()
            except Exception:
                pass
        after = set(self.calculations.keys())
        new_keys = after - before
        return {k: self.calculations[k] for k in new_keys}