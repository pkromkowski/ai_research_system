import logging
import pandas as pd
import yfinance as yf
from typing import Any, Callable, Dict, Optional
    
logger = logging.getLogger(__name__)


class IndexProvider:
    """
    Discovers appropriate market and sector indices for a stock.
    Selects broad index based on market cap (SPY, QQQ, or IWM).
    Dynamically finds sector ETF or falls back to curated mapping.
    Provides price history aligned to stock's trading dates.
    """

    DEFAULT_PERIOD_LENGTH: str = "1y"
    LARGE_CAP_THRESHOLD: float = 100_000_000_000
    MID_CAP_THRESHOLD: float = 10_000_000_000
    BROAD_INDEX_MAPPING: Dict[str, Dict[str, Any]] = {
        'large_cap': {
            'symbol': 'SPY',
            'name': 'SPDR S&P 500 ETF Trust',
            'description': 'Large-cap equities (S&P 500)',
        },
        'mid_cap': {
            'symbol': 'QQQ',
            'name': 'Invesco QQQ Trust',
            'description': 'Tech-heavy large-cap equities (NASDAQ-100)',
        },
        'small_cap': {
            'symbol': 'IWM',
            'name': 'iShares Russell 2000 ETF',
            'description': 'Small-cap equities (Russell 2000)',
        }
    }

    SECTOR_ETF_MAPPING: Dict[str, str] = {
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Information Technology': 'XLK',
        'Technology': 'XLK',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Industrials': 'XLI',
        'Basic Materials': 'XLB',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
    }
    
    def __init__(self, ticker: str, period_length: Optional[str] = None, ticker_factory: Optional[Callable[[str], Any]] = None):
        """
        Initialize with a stock ticker and period.

        Args:
            ticker: Stock ticker symbol
            period_length: Period for historical data
            ticker_factory: Factory callable for creating ticker objects (for testing)
        """
        self.ticker = ticker.upper()
        self.period_length = period_length or self.DEFAULT_PERIOD_LENGTH
        self._ticker_factory: Callable[[str], Any] = ticker_factory or (lambda t: yf.Ticker(t))
        self._stock: Optional[Any] = None
        self._info: Optional[Dict[str, Any]] = None
        self._stock_history: Optional[pd.DataFrame] = None
        self._broad_index: Optional[Dict[str, Any]] = None
        self._sector_index: Optional[Dict[str, Any]] = None
    
    @property
    def stock(self) -> Any:
        """Lazy-load ticker object."""
        if self._stock is None:
            try:
                self._stock = self._ticker_factory(self.ticker)
            except Exception as e:
                logger.warning("Could not create ticker object for %s: %s", self.ticker, e)
                self._stock = None
        return self._stock
    
    @property
    def info(self) -> Dict[str, Any]:
        """Lazy-load stock info."""
        if self._info is None:
            try:
                ticker = self.stock
                self._info = ticker.info if ticker is not None else {}
            except Exception as e:
                logger.warning("Could not fetch info for %s: %s", self.ticker, e)
                self._info = {}
        return self._info
    
    def _get_market_cap_tier(self, market_cap: Optional[float]) -> str:
        """Determine market cap tier: 'large_cap', 'mid_cap', or 'small_cap'."""
        if market_cap is None:
            return 'large_cap'  # Default
        
        if market_cap >= self.LARGE_CAP_THRESHOLD:
            return 'large_cap'
        elif market_cap >= self.MID_CAP_THRESHOLD:
            return 'mid_cap'
        else:
            return 'small_cap'
    
    def get_broad_index(self) -> Dict[str, Any]:
        """Get broad market index based on market cap (SPY, QQQ, or IWM)."""
        if self._broad_index is not None:
            return self._broad_index
        
        market_cap = self.info.get('marketCap')
        tier = self._get_market_cap_tier(market_cap)
        selection_method = 'market_cap_based' if market_cap else 'default'
        
        tier_info = self.BROAD_INDEX_MAPPING[tier]
        
        self._broad_index = {
            'symbol': tier_info['symbol'],
            'name': tier_info['name'],
            'description': tier_info['description'],
            'tier': tier,
            'selection_method': selection_method,
        }
        
        logger.debug("Selected broad index %s for %s (tier=%s, method=%s)", tier_info['symbol'], self.ticker, tier, selection_method)
        
        return self._broad_index
    
    def get_sector_index(self) -> Dict[str, Any]:
        """Get sector index ETF using dynamic discovery or fallback mapping."""
        
        sector = self.info.get('sector')

        if not sector:
            logger.warning("Sector not available for %s", self.ticker)
            self._sector_index = {
                'symbol': None,
                'name': None,
                'description': 'No sector information available',
                'sector': None,
                'selection_method': 'none',
            }
            return self._sector_index

        dynamic_etf = self._find_sector_etf_dynamic(sector)
        if dynamic_etf:
            self._sector_index = {
                'symbol': dynamic_etf,
                'name': self._get_etf_name(dynamic_etf),
                'description': f'{sector} sector ETF',
                'sector': sector,
                'selection_method': 'dynamic',
            }
            logger.debug(f"Found sector ETF dynamically: {dynamic_etf} for {sector}")
            return self._sector_index

        mapped_etf = self.SECTOR_ETF_MAPPING.get(sector)
        if mapped_etf:
            self._sector_index = {
                'symbol': mapped_etf,
                'name': self._get_etf_name(mapped_etf),
                'description': f'{sector} sector ETF',
                'sector': sector,
                'selection_method': 'mapped',
            }
            logger.debug(f"Found sector ETF from mapping: {mapped_etf} for {sector}")
            return self._sector_index

        logger.warning(f"No sector ETF found for {self.ticker} in sector {sector}")
        self._sector_index = {
            'symbol': None,
            'name': None,
            'description': f'No ETF found for {sector} sector',
            'sector': sector,
            'selection_method': 'none',
        }
        return self._sector_index
    
    def _find_sector_etf_dynamic(self, sector: str) -> Optional[str]:
        """Find sector ETF using yfinance Screener."""
        try:
            screener = yf.Screener()
            screener.add_query_filter('sector', sector, match_operator='match')
            screener.add_query_filter('type', 'ETF', match_operator='match')

            data = screener.run()

            if data.empty or 'Symbol' not in data.columns:
                return None

            if 'Market Cap' in data.columns:
                data = data.sort_values('Market Cap', ascending=False)

            symbols = data['Symbol'].tolist()
            return symbols[0] if symbols else None

        except Exception as e:
            logger.debug("Error finding sector ETF dynamically for %s: %s", sector, e)
            return None
    
    def _get_etf_name(self, ticker: str) -> Optional[str]:
        """Get ETF name."""
        try:
            etf = self._ticker_factory(ticker)
            etf_info = etf.info if etf is not None else {}
            return etf_info.get('longName') or etf_info.get('shortName')
        except Exception as e:
            logger.debug("Could not fetch name for ETF %s: %s", ticker, e)
            return None
    
    def get_indices(self) -> Dict[str, Any]:
        """Get both broad and sector indices."""
        return {
            'broad_index': self.get_broad_index(),
            'sector_index': self.get_sector_index(),
            'stock': self.ticker,
        }

    def get_stock_history(self, period: Optional[str] = None) -> pd.DataFrame:
        """Get stock price history (lazy-loaded and cached)."""
        if self._stock_history is None:
            try:
                use_period = period or self.period_length
                ticker = self.stock
                self._stock_history = ticker.history(period=use_period) if ticker is not None else pd.DataFrame()
            except Exception as e:
                logger.warning("Could not fetch history for %s: %s", self.ticker, e)
                self._stock_history = pd.DataFrame()
        return self._stock_history
    
    def get_index_prices(self, stock_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fetch index price data aligned to stock's timeline."""
        stock_history = stock_history if stock_history is not None else self.get_stock_history()

        if stock_history is None or len(stock_history.index) == 0:
            logger.warning("No stock history for timeline alignment: %s", self.ticker)
            return {
                'broad_index_history': None,
                'sector_index_history': None,
                'broad_index_symbol': None,
                'sector_index_symbol': None,
            }

        stock_index = stock_history.index

        broad_info = self.get_broad_index()
        sector_info = self.get_sector_index()

        result: Dict[str, Any] = {
            'broad_index_history': None,
            'sector_index_history': None,
            'broad_index_symbol': broad_info.get('symbol'),
            'sector_index_symbol': sector_info.get('symbol'),
        }

        broad_symbol = broad_info.get('symbol')
        if broad_symbol:
            result['broad_index_history'] = self._fetch_aligned_history(
                broad_symbol, stock_index, 'broad'
            )

        sector_symbol = sector_info.get('symbol')
        if sector_symbol:
            result['sector_index_history'] = self._fetch_aligned_history(
                sector_symbol, stock_index, 'sector'
            )

        logger.info(
            "Fetched index prices for %s: broad=%s (%s), sector=%s (%s)",
            self.ticker,
            broad_symbol,
            'ok' if result['broad_index_history'] is not None else 'failed',
            sector_symbol,
            'ok' if result['sector_index_history'] is not None else 'failed',
        )

        return result

    def _fetch_aligned_history(self,
                               symbol: str,
                               target_index: pd.DatetimeIndex,
                               index_type: str) -> Optional[pd.DataFrame]:
        """Fetch and align index history to target dates."""
        try:
            etf = self._ticker_factory(symbol)
            history = etf.history(period=self.period_length) if etf is not None else pd.DataFrame()

            if history.empty:
                logger.warning("Empty history for %s index %s", index_type, symbol)
                return None

            target_index = pd.DatetimeIndex(target_index)
            aligned = history.reindex(target_index, method='ffill')
            logger.debug("Fetched %s index %s: %s points", index_type, symbol, len(aligned))
            return aligned

        except Exception as e:
            logger.warning("Could not fetch %s index %s: %s", index_type, symbol, e)
            return None
