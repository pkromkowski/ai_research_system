import logging
import pandas as pd
import yfinance as yf
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ResearchDataProvider:
    """Provides research and fundamentals data from yfinance including analyst data, insider transactions, and sector/industry information."""
    
    def __init__(self, ticker: str, ticker_factory: Optional[Callable[[str], Any]] = None):
        """Initialize with stock ticker.

        Args:
            ticker: Stock ticker symbol
            ticker_factory: Optional factory for testing. Defaults to yf.Ticker.
        """
        self.ticker = ticker.upper()
        self._ticker_factory: Callable[[str], Any] = ticker_factory or (lambda t: yf.Ticker(t))
        self._stock: Optional[Any] = None
        self._info: Optional[Dict[str, Any]] = None
        self._sector: Optional[Any] = None
        self._industry: Optional[Any] = None

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
        """Lazy-load and cache stock info."""
        if self._info is None:
            try:
                ticker = self.stock
                self._info = ticker.info if ticker is not None else {}
            except Exception as e:
                logger.warning("Could not fetch info for %s: %s", self.ticker, e)
                self._info = {}
        return self._info

    @property
    def sector(self) -> Optional[yf.Sector]:
        """Lazy-load sector object."""
        if self._sector is None:
            sector_name = self.info.get('sector')
            if sector_name:
                try:
                    self._sector = yf.Sector(sector_name)
                except Exception as e:
                    logger.debug("Could not load sector %s: %s", sector_name, e)
        return self._sector

    @property
    def industry(self) -> Optional[yf.Industry]:
        """Lazy-load industry object."""
        if self._industry is None:
            industry_name = self.info.get('industry')
            if industry_name:
                try:
                    self._industry = yf.Industry(industry_name)
                except Exception as e:
                    logger.debug("Could not load industry %s: %s", industry_name, e)
        return self._industry

    def _safe_get_attribute(self, attr_name: str, source: str = "stock") -> pd.DataFrame:
        """Safely get a DataFrame attribute from stock, sector, or industry."""
        try:
            if source == "stock":
                obj = self.stock
            elif source == "sector":
                obj = self.sector
                if obj is None:
                    return pd.DataFrame()
            elif source == "industry":
                obj = self.industry
                if obj is None:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
            result = getattr(obj, attr_name, None)
            if result is None:
                return pd.DataFrame()
            if callable(result):
                result = result()

            if isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, pd.Series):
                return result.to_frame()
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.debug("Could not fetch %s.%s for %s: %s", source, attr_name, self.ticker, e)
            return pd.DataFrame()

    def get_sec_filings(self) -> pd.DataFrame:
        """Get SEC filings (10-Q, 10-K, etc.)."""
        return self._safe_get_attribute("get_sec_filings", "stock")
    
    def get_recommendations(self) -> pd.DataFrame:
        """Get analyst recommendations."""
        return self._safe_get_attribute("recommendations", "stock")
    
    def get_upgrades_downgrades(self) -> pd.DataFrame:
        """Get analyst upgrades and downgrades history."""
        return self._safe_get_attribute("upgrades_downgrades", "stock")
    
    def get_analyst_price_targets(self) -> pd.DataFrame:
        """Get analyst price targets."""
        return self._safe_get_attribute("analyst_price_targets", "stock")
    
    def get_earnings_estimate(self) -> pd.DataFrame:
        """Get earnings estimate data."""
        return self._safe_get_attribute("earnings_estimate", "stock")
    
    def get_revenue_estimate(self) -> pd.DataFrame:
        """Get revenue estimate data."""
        return self._safe_get_attribute("revenue_estimate", "stock")
    
    def get_earnings_history(self) -> pd.DataFrame:
        """Get historical earnings data."""
        return self._safe_get_attribute("earnings_history", "stock")
    
    def get_eps_trend(self) -> pd.DataFrame:
        """Get EPS trend data."""
        return self._safe_get_attribute("eps_trend", "stock")
    
    def get_eps_revisions(self) -> pd.DataFrame:
        """Get EPS revisions data."""
        return self._safe_get_attribute("eps_revisions", "stock")
    
    def get_growth_estimates(self) -> pd.DataFrame:
        """Get growth estimates."""
        return self._safe_get_attribute("growth_estimates", "stock")

    def get_funds_data(self) -> pd.DataFrame:
        """Get fund holdings data."""
        return self._safe_get_attribute("funds_data", "stock")
    
    def get_insider_purchases(self) -> pd.DataFrame:
        """Get insider purchase transactions."""
        return self._safe_get_attribute("insider_purchases", "stock")
    
    def get_insider_transactions(self) -> pd.DataFrame:
        """Get all insider transactions."""
        return self._safe_get_attribute("insider_transactions", "stock")
    
    def get_insider_roster_holders(self) -> pd.DataFrame:
        """Get insider roster holders."""
        return self._safe_get_attribute("insider_roster_holders", "stock")
    
    def get_major_holders(self) -> pd.DataFrame:
        """Get major shareholders."""
        return self._safe_get_attribute("major_holders", "stock")
    
    def get_institutional_holders(self) -> pd.DataFrame:
        """Get institutional shareholders."""
        return self._safe_get_attribute("institutional_holders", "stock")
    
    def get_mutualfund_holders(self) -> pd.DataFrame:
        """Get mutual fund holders."""
        return self._safe_get_attribute("mutualfund_holders", "stock")

    def get_sector_overview(self) -> pd.DataFrame:
        """Get sector overview data."""
        return self._safe_get_attribute("overview", "sector")
    
    def get_sector_research_reports(self) -> pd.DataFrame:
        """Get sector research reports."""
        return self._safe_get_attribute("research_reports", "sector")
    
    def get_sector_top_companies(self) -> pd.DataFrame:
        """Get top companies in the sector."""
        return self._safe_get_attribute("top_companies", "sector")
    
    def get_sector_top_etfs(self) -> pd.DataFrame:
        """Get top ETFs in the sector."""
        return self._safe_get_attribute("top_etfs", "sector")
    
    def get_sector_top_mutual_funds(self) -> pd.DataFrame:
        """Get top mutual funds in the sector."""
        return self._safe_get_attribute("top_mutual_funds", "sector")

    def get_industry_overview(self) -> pd.DataFrame:
        """Get industry overview data."""
        return self._safe_get_attribute("overview", "industry")
    
    def get_industry_research_reports(self) -> pd.DataFrame:
        """Get industry research reports."""
        return self._safe_get_attribute("research_reports", "industry")
    
    def get_industry_top_companies(self) -> pd.DataFrame:
        """Get top companies in the industry."""
        return self._safe_get_attribute("top_companies", "industry")
    
    def get_industry_top_growth_companies(self) -> pd.DataFrame:
        """Get top growth companies in the industry."""
        return self._safe_get_attribute("top_growth_companies", "industry")
    
    def get_industry_top_performing_companies(self) -> pd.DataFrame:
        """Get top performing companies in the industry."""
        return self._safe_get_attribute("top_performing_companies", "industry")

    def get_all_analyst_data(self) -> Dict[str, pd.DataFrame]:
        """Get all analyst-related data in one call."""
        return {
            "recommendations": self.get_recommendations(),
            "upgrades_downgrades": self.get_upgrades_downgrades(),
            "price_targets": self.get_analyst_price_targets(),
            "earnings_estimate": self.get_earnings_estimate(),
            "revenue_estimate": self.get_revenue_estimate(),
            "earnings_history": self.get_earnings_history(),
            "eps_trend": self.get_eps_trend(),
            "eps_revisions": self.get_eps_revisions(),
            "growth_estimates": self.get_growth_estimates(),
        }
    
    def get_all_ownership_data(self) -> Dict[str, pd.DataFrame]:
        """Get all ownership-related data in one call."""
        return {
            "major_holders": self.get_major_holders(),
            "institutional_holders": self.get_institutional_holders(),
            "mutualfund_holders": self.get_mutualfund_holders(),
            "insider_purchases": self.get_insider_purchases(),
            "insider_transactions": self.get_insider_transactions(),
            "insider_roster": self.get_insider_roster_holders(),
            "funds_data": self.get_funds_data(),
        }
    
    def get_all_sector_data(self) -> Dict[str, pd.DataFrame]:
        """Get all sector-related data in one call."""
        return {
            "overview": self.get_sector_overview(),
            "research_reports": self.get_sector_research_reports(),
            "top_companies": self.get_sector_top_companies(),
            "top_etfs": self.get_sector_top_etfs(),
            "top_mutual_funds": self.get_sector_top_mutual_funds(),
        }
    
    def get_all_industry_data(self) -> Dict[str, pd.DataFrame]:
        """Get all industry-related data in one call."""
        return {
            "overview": self.get_industry_overview(),
            "research_reports": self.get_industry_research_reports(),
            "top_companies": self.get_industry_top_companies(),
            "top_growth_companies": self.get_industry_top_growth_companies(),
            "top_performing_companies": self.get_industry_top_performing_companies(),
        }

    def clear_cache(self) -> None:
        """Clear cached objects (stock, info, sector, industry)."""
        self._stock = None
        self._info = None
        self._sector = None
        self._industry = None
        logger.debug("Cleared ResearchDataProvider caches for %s", self.ticker)
