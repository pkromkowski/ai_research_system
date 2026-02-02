import logging
import pandas as pd
from typing import Any, Dict, List, Optional

try:
    from fredapi import Fred as _Fred
except Exception:
    _Fred = None

from model.core.config import FredConfig

logger = logging.getLogger(__name__)


class FredDataProvider:
    """Provides macroeconomic data from FRED API with caching and alignment."""

    def __init__(self, config: Optional[FredConfig] = None, api_key: Optional[str] = None, fred_client: Optional[Any] = None):
        """Initialize provider.

        Args:
            config: FredConfig instance (default creates new instance)
            api_key: API key override
            fred_client: Pre-initialized FRED client for testing
        """
        self.config = config or FredConfig()
        self.api_key = api_key or self.config.api_key
        self._fred: Optional[Any] = fred_client
        if self._fred is None and not self.api_key:
            raise ValueError("FRED_API_KEY not configured and no fred_client provided. Set FRED_API_KEY environment variable or pass a FredConfig")

        self.observation_start = self.config.observation_start
        self.default_indicators = self.config.default_indicators
        self.categories = self.config.categories
        self._cache: Dict[str, pd.Series] = {}
    
    @property
    def fred(self) -> Any:
        """Lazy-load FRED API client."""
        if self._fred is None:
            if _Fred is None:
                raise ImportError("fredapi is required for FredDataProvider; install it or pass a `fred_client` to the constructor")
            self._fred = _Fred(api_key=self.api_key)
        return self._fred
    
    def get_series(self, 
                   series_id: str, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, 
                   frequency: Optional[str] = None) -> pd.Series:
        """Fetch a single FRED series with caching.

        Args:
            series_id: FRED series identifier
            start_date: Observation start date
            end_date: Observation end date
            frequency: Data frequency override

        Returns:
            pd.Series with datetime index
        """
        if start_date is None:
            start_date = self.observation_start
        
        cache_key = f"{series_id}_{start_date}_{end_date}_{frequency}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            kwargs = {'series_id': series_id, 'observation_start': start_date}
            if end_date:
                kwargs['observation_end'] = end_date
            if frequency:
                kwargs['frequency'] = frequency
            
            series = self.fred.get_series(**kwargs)
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                logger.debug("Could not cast series index to datetime for %s", series_id)

            self._cache[cache_key] = series
            logger.debug("Fetched FRED series: %s (%s observations)", series_id, len(series))
            return series
            
        except Exception as e:
            logger.error("Failed to fetch FRED series %s: %s", series_id, e)
            raise ValueError("Failed to fetch %s: %s" % (series_id, str(e)))
    
    def get_multiple_series(self, 
                           series_ids: List[str],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           frequency: Optional[str] = None) -> Dict[str, pd.Series]:
        """Fetch multiple FRED series.

        Args:
            series_ids: List of FRED series identifiers
            start_date: Observation start date
            end_date: Observation end date
            frequency: Data frequency override

        Returns:
            Dictionary mapping series_id to pd.Series
        """
        result = {}
        for series_id in series_ids:
            try:
                result[series_id] = self.get_series(series_id, start_date, end_date, frequency)
            except ValueError as e:
                logger.warning(f"Skipping series {series_id}: {e}")
                continue
        
        logger.info(f"Fetched {len(result)}/{len(series_ids)} FRED series")
        return result
    
    def get_macro_data(self, 
                      indicators: Optional[Dict[str, str]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, pd.Series]:
        """Fetch standard macro indicators.

        Args:
            indicators: Dict of {series_id: description}
            start_date: Observation start date
            end_date: Observation end date

        Returns:
            Dictionary mapping series_id to pd.Series
        """
        if indicators is None:
            indicators = self.default_indicators
        
        series_ids = list(indicators.keys())
        return self.get_multiple_series(series_ids, start_date, end_date)
    
    def get_category_data(self, 
                         category: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, pd.Series]:
        """Fetch macro data for a specific category.

        Args:
            category: Category name from config
            start_date: Observation start date
            end_date: Observation end date

        Returns:
            Dictionary mapping series_id to pd.Series
        """
        if category not in self.categories:
            available = list(self.categories.keys())
            raise ValueError(f"Unknown category '{category}'. Available: {available}")

        series_ids = self.categories[category]
        indicators = {sid: self.default_indicators.get(sid, sid) for sid in series_ids}
        return self.get_macro_data(indicators, start_date, end_date)
    
    def align_to_stock_dates(self, 
                            macro_data: Dict[str, pd.Series],
                            stock_dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Align macro data to stock trading dates using forward-fill.

        Args:
            macro_data: Dictionary of FRED series
            stock_dates: DatetimeIndex from stock history

        Returns:
            Dictionary of aligned series
        """
        # Normalize to a timezone-naive DatetimeIndex for matching
        stock_dates = pd.DatetimeIndex(stock_dates)
        if stock_dates.tz is not None:
            stock_dates = stock_dates.tz_localize(None)
        stock_dates_naive = stock_dates
        
        aligned = {}
        for series_id, series in macro_data.items():
            try:
                aligned_series = series.reindex(stock_dates_naive).ffill().dropna()
                if len(aligned_series) > 0:
                    aligned[series_id] = aligned_series
                else:
                    logger.debug(f"Series {series_id} has no overlapping data with stock dates")
            except Exception as e:
                logger.warning(f"Could not align series {series_id}: {e}")
                continue
        
        logger.debug(f"Aligned {len(aligned)}/{len(macro_data)} series to stock dates")
        return aligned
    
    def clear_cache(self) -> None:
        """Clear the series cache."""
        count = len(self._cache)
        self._cache.clear()
        logger.debug("Cleared %s cached series from FredDataProvider", count)


_default_provider: Optional[FredDataProvider] = None


def get_macro_data_for_stock(stock_history: pd.DataFrame,
                            indicators: Optional[Dict[str, str]] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            api_key: Optional[str] = None,
                            fred_config: Optional[FredConfig] = None) -> Dict[str, pd.Series]:
    """Fetch and align macro data for a stock.

    Args:
        stock_history: Stock history DataFrame with DatetimeIndex
        indicators: Dict of {series_id: description}
        start_date: Observation start date
        end_date: Observation end date
        api_key: FRED API key override
        fred_config: FredConfig instance

    Returns:
        Dictionary of macro series aligned to stock trading dates
    """
    global _default_provider

    if fred_config is not None:
        provider = FredDataProvider(config=fred_config)
    elif api_key:
        provider = FredDataProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = FredDataProvider()
        provider = _default_provider

    macro_data = provider.get_macro_data(indicators, start_date, end_date)
    aligned = provider.align_to_stock_dates(macro_data, stock_history.index)
    return aligned
