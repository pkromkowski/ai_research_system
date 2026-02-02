import logging
import pandas as pd
import yfinance as yf
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StockDataProvider:
    """Provides stock data from yfinance including prices, financials, dividends, splits, and news."""
    DEFAULT_PERIOD_LENGTH: str = "1y"
    DEFAULT_FINANCIAL_FREQ: str = "quarterly"
    DEFAULT_NEWS_COUNT: int = 10
    BACKGROUND_FIELDS: frozenset = frozenset({
        "symbol",
        "longName",
        "shortName",
        "displayName",
        "sector",
        "industry",
        "country",
        "website",
        "phone",
        "longBusinessSummary",
        "address1",
        "address2",
        "city",
        "state",
        "zip",
        "exchange",
        "fullExchangeName",
        "fullTimeEmployees",
        "ipoExpectedDate",
        "companyOfficers",
        "language",
        "currency",
        "quoteType",
        "market",
    })
    METRIC_FIELDS: frozenset = frozenset({
        # Price data
        "currentPrice",
        "regularMarketPrice",
        "previousClose",
        "open",
        "dayHigh",
        "dayLow",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "allTimeHigh",
        "allTimeLow",
        "fiftyDayAverage",
        "twoHundredDayAverage",
        # Volume data
        "volume",
        "regularMarketVolume",
        "averageVolume",
        "averageVolume10days",
        "averageDailyVolume10Day",
        "averageDailyVolume3Month",
        # Market cap & shares
        "marketCap",
        "enterpriseValue",
        "sharesOutstanding",
        "floatShares",
        "impliedSharesOutstanding",
        "sharesShort",
        "shortRatio",
        "shortPercentOfFloat",
        # Valuation ratios
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "trailingPegRatio",
        "bookValue",
        "revenuePerShare",
        # Margins
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "ebitdaMargins",
        # Returns & ratios
        "returnOnAssets",
        "returnOnEquity",
        "currentRatio",
        "quickRatio",
        "debtToEquity",
        # Cash & debt
        "totalCash",
        "totalCashPerShare",
        "totalDebt",
        "operatingCashflow",
        "freeCashflow",
        # Growth & EPS
        "revenueGrowth",
        "epsTrailingTwelveMonths",
        "epsCurrentYear",
        "epsForward",
        "priceEpsCurrentYear",
        # Risk & ownership
        "beta",
        "SandP52WeekChange",
        "52WeekChange",
        "fiftyTwoWeekChangePercent",
        "heldPercentInsiders",
        "heldPercentInstitutions",
        # Dividends
        "trailingAnnualDividendRate",
        "trailingAnnualDividendYield",
        "payoutRatio",
        # Risk scores
        "overallRisk",
        "auditRisk",
        "boardRisk",
        "compensationRisk",
        "shareHolderRightsRisk",
    })
    
    def __init__(
        self,
        ticker: str,
        period_length: Optional[str] = None,
        financial_freq: Optional[str] = None,
        ticker_factory: Optional[Callable[[str], Any]] = None,
    ):
        """Initialize with ticker and optional data preferences. ticker_factory is for testing."""
        self.ticker = ticker.upper()
        self.period_length = period_length or self.DEFAULT_PERIOD_LENGTH
        self.financial_freq = financial_freq or self.DEFAULT_FINANCIAL_FREQ
        self._ticker_factory: Callable[[str], Any] = ticker_factory or (lambda t: yf.Ticker(t))
        self._stock: Optional[Any] = None
        self._info: Optional[Dict[str, Any]] = None
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
        """Lazy-load stock info, returns empty dict on failure."""
        if self._info is None:
            try:
                ticker = self.stock
                self._info = ticker.info if ticker is not None else {}
            except Exception as e:
                logger.warning("Could not fetch info for %s: %s", self.ticker, e)
                self._info = {}
        return self._info
    def history(self, *args, **kwargs) -> pd.DataFrame:
        """Get historical price data."""
        if "period" not in kwargs and len(args) == 0:
            kwargs["period"] = self.period_length
        return self.stock.history(*args, **kwargs)
    def get_info(self) -> Dict[str, Any]:
        """Get stock information."""
        return self.info
    def get_income_stmt(self, freq: Optional[str] = None) -> pd.DataFrame:
        """Get income statement."""
        return self.stock.get_income_stmt(freq=freq or self.financial_freq)
    def get_balance_sheet(self, freq: Optional[str] = None) -> pd.DataFrame:
        """Get balance sheet."""
        return self.stock.get_balance_sheet(freq=freq or self.financial_freq)
    def get_cashflow(self, freq: Optional[str] = None) -> pd.DataFrame:
        """Get cash flow statement."""
        return self.stock.get_cashflow(freq=freq or self.financial_freq)
    def get_actions(self, period: Optional[str] = None) -> pd.DataFrame:
        """Get dividends and stock splits."""
        if period:
            return self.stock.get_actions(period=period)
        return self.stock.get_actions()
    def get_news(self, count: Optional[int] = None, tab: str = "news") -> List[Dict[str, Any]]:
        """Get news articles, returns cleaned article dictionaries."""
        if count is None:
            count = self.DEFAULT_NEWS_COUNT
        try:
            raw_news = self.stock.get_news(count=count, tab=tab)
        except Exception as e:
            logger.warning("Could not fetch news for %s: %s", self.ticker, e)
            return []
        cleaned_news = []
        for article in raw_news:
            try:
                content = article.get("content", {})
                cleaned_article = {
                    "title": content.get("title"),
                    "summary": content.get("summary"),
                    "source": content.get("provider", {}).get("displayName"),
                    "published_time": content.get("pubDate"),
                    "link": content.get("clickThroughUrl", {}).get("url"),
                }
                if cleaned_article["title"] and cleaned_article["link"]:
                    cleaned_news.append(cleaned_article)
            except Exception as e:
                logger.debug("Skipping malformed news article: %s", e)
                continue
        return cleaned_news
    def get_stock_background(self) -> Dict[str, Any]:
        """Get qualitative background information (non-None values only)."""
        return {
            k: v for k, v in self.info.items()
            if k in self.BACKGROUND_FIELDS and v is not None
        }
    def get_stock_metrics(self) -> Dict[str, Any]:
        """Get quantitative metrics and ratios (non-None values only)."""
        return {
            k: v for k, v in self.info.items()
            if k in self.METRIC_FIELDS and v is not None
        }
    def get_actions_calculations(self) -> Dict[str, Any]:
        """Calculate dividend and split metrics from actions data."""
        try:
            actions = self.get_actions()
            if actions.empty:
                return {}
            calculations: Dict[str, Any] = {}
            if "Dividends" in actions.columns:
                dividends = actions.loc[actions["Dividends"] > 0, "Dividends"]
                if len(dividends) > 0:
                    calculations["total_dividends_paid"] = float(dividends.sum())
                    calculations["latest_dividend"] = float(dividends.iloc[0])
                    calculations["dividend_frequency"] = len(dividends)
                    calculations["average_dividend_per_payment"] = float(dividends.mean())
            if "Stock Splits" in actions.columns:
                splits = actions.loc[actions["Stock Splits"] != 1.0, "Stock Splits"]
                if len(splits) > 0:
                    calculations["total_split_events"] = len(splits)
                    calculations["latest_split_ratio"] = float(splits.iloc[0])
            return calculations
        except Exception as e:
            logger.debug("Could not calculate actions for %s: %s", self.ticker, e)
            return {}
    def clear_cache(self) -> None:
        """Clear cached ticker and info."""
        self._stock = None
        self._info = None
        logger.debug("Cleared StockDataProvider cache for %s", self.ticker)
