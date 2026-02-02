import logging
import pandas as pd
import yfinance as yf
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PeerDiscoveryProvider:
    """
    Discovers peer companies using yfinance Screener.
    
    Strategy: Industry + market cap match → sector + broader cap → industry leaders.
    Provides price history for peers, aligned to stock's trading dates.
    """

    DEFAULT_PERIOD_LENGTH: str = "1y"
    DEFAULT_NUM_PEERS: int = 10
    INDUSTRY_CAP_RANGE: float = 0.5
    SECTOR_CAP_RANGE: float = 1.0

    def __init__(self, ticker: str, period_length: Optional[str] = None, ticker_factory: Optional[Callable[[str], Any]] = None):
        """
        Initialize with a stock ticker and period.
        
        Args:
            ticker: Stock ticker symbol
            period_length: Period for historical data
            ticker_factory: Factory returning object with `.info` and `.history()`
        """
        self.ticker = ticker.upper()
        self.period_length = period_length or self.DEFAULT_PERIOD_LENGTH
        self._ticker_factory: Callable[[str], Any] = ticker_factory or (lambda t: yf.Ticker(t))
        self._stock: Optional[Any] = None
        self._info: Optional[Dict[str, Any]] = None
        self._stock_history: Optional[pd.DataFrame] = None
        self._cached_peers: Optional[Dict[str, Any]] = None

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

    def get_peers(self,
                  num_peers: Optional[int] = None,
                  include_broader: bool = True) -> Dict[str, Any]:
        """
        Get peer companies for the stock.
        
        Args:
            num_peers: Number of peers to return
            include_broader: If insufficient peers found, broaden search
        
        Returns:
            Dict with 'peers', 'industry', 'sector', 'strategy', 'count'
        """
        num_peers = num_peers or self.DEFAULT_NUM_PEERS

        if self._cached_peers is not None and self._cached_peers.get('count', 0) >= num_peers:
            cached = self._cached_peers.copy()
            cached['peers'] = cached['peers'][:num_peers]
            cached['count'] = len(cached['peers'])
            return cached

        industry = self.info.get('industry')
        sector = self.info.get('sector')
        market_cap = self.info.get('marketCap')

        if not industry or not market_cap:
            logger.warning("Missing data for peer discovery: %s (industry=%s, market_cap=%s)", self.ticker, bool(industry), bool(market_cap))
            return {
                'peers': [],
                'industry': industry,
                'sector': sector,
                'strategy': 'insufficient_data',
                'count': 0,
            }

        peers: List[str] = []
        strategy = 'no_peers_found'

        industry_peers = self._find_peers_by_filter(
            filter_type='industry',
            filter_value=industry,
            market_cap=market_cap,
            cap_range=self.INDUSTRY_CAP_RANGE,
        )
        peers.extend(industry_peers)

        if len(peers) >= num_peers:
            strategy = 'industry_market_cap_match'
        elif include_broader:
            sector_peers = self._find_peers_by_filter(
                filter_type='sector',
                filter_value=sector,
                market_cap=market_cap,
                cap_range=self.SECTOR_CAP_RANGE,
            )
            new_peers = [p for p in sector_peers if p not in peers]
            peers.extend(new_peers)

            if len(peers) >= num_peers:
                strategy = 'sector_broader_cap_match'
            else:
                leaders = self._find_industry_leaders(industry, limit=num_peers)
                new_peers = [p for p in leaders if p not in peers]
                peers.extend(new_peers)
                strategy = 'industry_leaders_fallback' if peers else 'no_peers_found'
        else:
            strategy = 'industry_market_cap_match' if peers else 'no_peers_found'

        self._cached_peers = {
            'peers': peers,
            'industry': industry,
            'sector': sector,
            'strategy': strategy,
            'count': len(peers),
        }

        result = self._cached_peers.copy()
        result['peers'] = peers[:num_peers]
        result['count'] = len(result['peers'])

        logger.debug("Discovered %s peers for %s (strategy=%s)", result['count'], self.ticker, strategy)

        return result

    def _find_peers_by_filter(self,
                              filter_type: str,
                              filter_value: Optional[str],
                              market_cap: float,
                              cap_range: float) -> List[str]:
        """
        Find peers using filter with market cap range.
        
        Args:
            filter_type: 'industry' or 'sector'
            filter_value: Value to filter by
            market_cap: Market cap of target stock
            cap_range: Range multiplier
        
        Returns:
            List of peer tickers sorted by market cap proximity
        """
        if not filter_value:
            return []

        try:
            min_cap = market_cap * (1 - cap_range)
            max_cap = market_cap * (1 + cap_range)

            try:
                filter_query = yf.EquityQuery('EQ', [filter_type, filter_value])
            except ValueError:
                logger.debug("Skipping %s filter (invalid value format): %s", filter_type, filter_value)
                return []

            cap_query = yf.EquityQuery('BTWN', ['intradaymarketcap', min_cap, max_cap])
            region_query = yf.EquityQuery('EQ', ['region', 'us'])

            combined_query = yf.EquityQuery('AND', [filter_query, cap_query, region_query])

            result = yf.screener.screen(
                combined_query,
                sortField='intradaymarketcap',
                sortAsc=False,
                size=50,
            )

            if not result or 'quotes' not in result:
                return []

            quotes = result['quotes']

            peers = []
            for q in quotes:
                symbol = q.get('symbol', '')
                if symbol == self.ticker or '.' in symbol:
                    continue
                peers.append({
                    'symbol': symbol,
                    'cap_diff': abs(q.get('marketCap', 0) - market_cap),
                })

            peers.sort(key=lambda x: x['cap_diff'])
            peer_symbols = [p['symbol'] for p in peers]

            logger.debug("Found %s peers by %s=%s (cap_range=±%s%%)", len(peer_symbols), filter_type, filter_value, int(cap_range*100))
            return peer_symbols

        except Exception as e:
            logger.warning("Error finding peers by %s: %s", filter_type, e)
            return []

    def _find_industry_leaders(self, industry: str, limit: int) -> List[str]:
        """
        Find top companies in industry by market cap.
        
        Args:
            industry: Industry name
            limit: Maximum number to return
        
        Returns:
            List of peer tickers sorted by market cap descending
        """
        try:
            try:
                industry_query = yf.EquityQuery('EQ', ['industry', industry])
            except ValueError as e:
                logger.debug("Skipping industry leaders (invalid value format): %s", industry)
                return []

            region_query = yf.EquityQuery('EQ', ['region', 'us'])
            combined_query = yf.EquityQuery('AND', [industry_query, region_query])

            result = yf.screener.screen(
                combined_query,
                sortField='intradaymarketcap',
                sortAsc=False,
                size=limit + 5,
            )

            if not result or 'quotes' not in result:
                return []

            quotes = result['quotes']

            peers = []
            for q in quotes:
                symbol = q.get('symbol', '')
                if symbol == self.ticker or '.' in symbol:
                    continue
                peers.append(symbol)
                if len(peers) >= limit:
                    break

            logger.debug("Found %s industry leaders in %s", len(peers), industry)
            return peers

        except Exception as e:
            logger.warning("Error finding industry leaders: %s", e)
            return []

    def get_stock_history(self, period: Optional[str] = None) -> pd.DataFrame:
        """
        Get the stock's price history.

        Args:
            period: Period override for history lookup

        Returns:
            DataFrame with OHLCV data
        """
        if self._stock_history is None:
            try:
                use_period = period or self.period_length
                ticker = self.stock
                self._stock_history = ticker.history(period=use_period) if ticker is not None else pd.DataFrame()
            except Exception as e:
                logger.warning("Could not fetch history for %s: %s", self.ticker, e)
                self._stock_history = pd.DataFrame()
        return self._stock_history

    def get_peers_stock_data(self,
                             peer_tickers: Optional[List[str]] = None,
                             num_peers: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Fetch historical price data for peers, aligned to stock's timeline.
        
        Args:
            peer_tickers: List of peer tickers (discovers if None)
            num_peers: Number of peers to discover if peer_tickers not provided
        
        Returns:
            Dict with {ticker: pd.Series of closing prices}
        """
        num_peers = num_peers or self.DEFAULT_NUM_PEERS

        if peer_tickers is None:
            peer_result = self.get_peers(num_peers=num_peers)
            peer_tickers = peer_result.get('peers', [])

        if not peer_tickers:
            logger.warning("No peer tickers to fetch for %s", self.ticker)
            return {}

        stock_history = self.get_stock_history()
        if stock_history.empty:
            logger.warning("No stock history for timeline alignment: %s", self.ticker)
            return {}

        stock_index = stock_history.index
        peer_data: Dict[str, pd.Series] = {}

        for ticker in peer_tickers:
            aligned_prices = self._fetch_aligned_prices(ticker, stock_index)
            if aligned_prices is not None:
                peer_data[ticker] = aligned_prices

        logger.info("Fetched peer stock data for %s: %s/%s peers", self.ticker, len(peer_data), len(peer_tickers))
        return peer_data

    def _fetch_aligned_prices(self,
                              ticker: str,
                              target_index: pd.DatetimeIndex) -> Optional[pd.Series]:
        """
        Fetch and align peer prices to target dates.
        
        Args:
            ticker: Peer ticker symbol
            target_index: DatetimeIndex to align to
        
        Returns:
            Aligned Series of closing prices or None on error
        """
        try:
            peer = self._ticker_factory(ticker)
            peer_history = peer.history(period=self.period_length) if peer is not None else pd.DataFrame()

            if peer_history.empty or 'Close' not in peer_history.columns:
                logger.debug("Empty history for peer %s", ticker)
                return None

            target_index = pd.DatetimeIndex(target_index)
            aligned_prices = peer_history['Close'].reindex(target_index, method='ffill')
            aligned_prices = aligned_prices.dropna()

            if len(aligned_prices) == 0:
                logger.debug("No overlapping dates for peer %s", ticker)
                return None

            logger.debug("Fetched peer data: %s (%s points)", ticker, len(aligned_prices))
            return aligned_prices

        except Exception as e:
            logger.debug("Could not fetch data for peer %s: %s", ticker, e)
            return None
