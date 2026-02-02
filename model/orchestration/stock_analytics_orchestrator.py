# --- stock analytics orchestrator ---
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Callable

from model.core.config import AppConfig
from model.core.types import ThesisQuantitativeContext

from model.services.index_provider import IndexProvider
from model.services.ten_k_orchestrator import TenKOrchestrator
from model.services.stock_data_provider import StockDataProvider
from model.services.fred_provider import get_macro_data_for_stock
from model.services.research_data_provider import ResearchDataProvider
from model.services.peer_discovery_provider import PeerDiscoveryProvider
from model.services.perplexity_research_provider import PerplexityResearchProvider

from model.calculators.financial_calculator import FinancialCalculator
from model.calculators.technical_calculator import TechnicalCalculator
from model.calculators.peer_intelligence_calculator import PeerIntelligenceCalculator
from model.calculators.macro_sensitivity_calculator import MacroSensitivityCalculator
from model.calculators.advanced_technical_calculator import AdvancedTechnicalCalculator
from model.calculators.volume_positioning_calculator import VolumePositioningCalculator

logger = logging.getLogger(__name__)


class StockAnalyticsOrchestrator:
    """
    Modular stock analytics orchestrator that delegates calculations to specialized calculators.
    Also manages 10-K filing download, processing, and storage.
    
    Data Access:
        Access data through public provider attributes:
        - runner.stock_data_provider.get_info()
        - runner.stock_data_provider.get_income_stmt()
        - runner.research_data_provider.get_recommendations()
        - runner.peer_discovery_provider.get_peers()
        - runner.index_provider.get_indices()
        - runner.perplexity_provider.get_recent_news()
        - runner.ten_k_orchestrator.sync_and_process()
    
    Calculations:
        Use calculator methods for computed metrics:
        - runner.get_technical_metrics()
        - runner.get_financial_metrics()
        - runner.get_all_metrics()
    """
    
    def __init__(
        self,
        ticker: str,
        period_length: str = "1y",
        financial_freq: str = "quarterly",
        config: Optional[AppConfig] = None,
        ten_k_orchestrator_factory: Optional[Callable[[str, Optional[AppConfig]], TenKOrchestrator]] = None,
        perplexity_provider_factory: Optional[Callable[[str], PerplexityResearchProvider]] = None,
    ):
        """Initialize runner with configuration and optional test-friendly factories.

        Args:
            ticker: Stock ticker symbol
            period_length: Period for historical data (default: "1y")
            financial_freq: Frequency for financial statements (default: "quarterly")
            config: Optional AppConfig instance
            ten_k_orchestrator_factory: Optional factory to create TenKOrchestrator for tests
            perplexity_provider_factory: Optional factory to create PerplexityResearchProvider for tests
        """
        self.ticker = ticker.upper()
        self.period_length = period_length
        self.financial_freq = financial_freq
        self.config = config or AppConfig.from_env()

        logger.debug("Initializing StockAnalyticsOrchestrator", extra={"ticker": self.ticker, "period_length": period_length, "financial_freq": financial_freq})

        # --- public providers ---
        self.stock_data_provider = StockDataProvider(self.ticker, period_length, financial_freq)
        self.research_data_provider = ResearchDataProvider(self.ticker)
        self.peer_discovery_provider = PeerDiscoveryProvider(self.ticker, period_length)
        self.index_provider = IndexProvider(self.ticker, period_length)

        # --- factories ---
        self._ten_k_orchestrator_factory = ten_k_orchestrator_factory or (lambda t, cfg: TenKOrchestrator(t, config=cfg))
        self._perplexity_provider_factory = perplexity_provider_factory or (lambda t: PerplexityResearchProvider(t))

        # --- lazy caches ---
        self._ten_k_orchestrator: Optional[TenKOrchestrator] = None
        self._perplexity_provider: Optional[PerplexityResearchProvider] = None
    
    # --- lazy-loaded provider properties ---

    @property
    def ten_k_orchestrator(self) -> TenKOrchestrator:
        """
        10-K filing orchestrator (lazy-loaded).
        
        Use for SEC filing management:
            runner.ten_k_orchestrator.sync_and_process()
            runner.ten_k_orchestrator.get_latest_extraction()
        """
        if self._ten_k_orchestrator is None:
            logger.debug("Creating TenKOrchestrator", extra={"ticker": self.ticker})
            self._ten_k_orchestrator = self._ten_k_orchestrator_factory(self.ticker, self.config.filing)
        return self._ten_k_orchestrator
    
    @property
    def perplexity_provider(self) -> PerplexityResearchProvider:
        """Perplexity research provider (lazy-loaded)."""
        if self._perplexity_provider is None:
            self._perplexity_provider = self._perplexity_provider_factory(self.ticker)
        return self._perplexity_provider
    
    # --- calculator methods ---
    
    def get_technical_metrics(self) -> dict:
        """
        Calculate technical metrics.
        
        Returns:
            Dictionary with 25+ technical metrics
        """
        logger.debug("Calculating technical metrics", extra={"ticker": self.ticker})
        history = self.stock_data_provider.history()
        if history.empty:
            logger.warning("No history data for technical metrics", extra={"ticker": self.ticker})
            return {}
        
        calculator = TechnicalCalculator(history)
        result = calculator.calculate_all()
        logger.debug(
            "Technical metrics calculated",
            extra={"ticker": self.ticker, "metric_count": len(result)}
        )
        return result
    
    def get_advanced_technical_metrics(self) -> dict:
        """
        Calculate advanced technical metrics.
        
        Returns:
            Dictionary with 15+ advanced metrics (drawdowns, distributions, regimes)
        """
        logger.debug("Calculating advanced technical metrics", extra={"ticker": self.ticker})
        history = self.stock_data_provider.history()
        if history.empty:
            logger.warning("No history data for advanced technical metrics", extra={"ticker": self.ticker})
            return {}
        
        calculator = AdvancedTechnicalCalculator(history)
        result = calculator.calculate_all()
        logger.debug(
            "Advanced technical metrics calculated",
            extra={"ticker": self.ticker, "metric_count": len(result)}
        )
        return result
    
    def get_financial_metrics(self) -> dict:
        """
        Calculate fundamental financial metrics.
        
        Returns:
            Dictionary with 40+ financial metrics
        """
        logger.debug("Calculating financial metrics", extra={"ticker": self.ticker})
        income = self.stock_data_provider.get_income_stmt()
        balance = self.stock_data_provider.get_balance_sheet()
        cf = self.stock_data_provider.get_cashflow()
        
        if income.empty or balance.empty or cf.empty:
            logger.warning(
                "Incomplete financial data",
                extra={
                    "ticker": self.ticker,
                    "has_income": not income.empty,
                    "has_balance": not balance.empty,
                    "has_cashflow": not cf.empty
                }
            )
            return {}
        
        calculator = FinancialCalculator(income, balance, cf, freq=self.financial_freq)
        result = calculator.calculate_all()
        logger.debug(
            "Financial metrics calculated",
            extra={"ticker": self.ticker, "metric_count": len(result)}
        )
        return result
    
    def get_volume_positioning_metrics(self, 
                                       sector_index_history: pd.DataFrame = None,
                                       broad_index_history: pd.DataFrame = None) -> dict:
        """
        Calculate volume and positioning metrics.
        
        Args:
            sector_index_history: Optional sector index prices (auto-fetched if None)
            broad_index_history: Optional broad market index prices (auto-fetched if None)
        
        Returns:
            Dictionary with 8+ volume and positioning metrics
        """
        logger.debug("Calculating volume and positioning metrics", extra={"ticker": self.ticker})
        history = self.stock_data_provider.history()
        if history.empty:
            logger.warning("No history data for volume metrics", extra={"ticker": self.ticker})
            return {}
        
        if sector_index_history is None or broad_index_history is None:
            data = self.index_provider.get_index_prices()
            sector_index_history = sector_index_history or data.get('sector_index_history')
            broad_index_history = broad_index_history or data.get('broad_index_history')
        
        calculator = VolumePositioningCalculator(history)
        result = calculator.calculate_all(sector_index_history=sector_index_history, broad_index_history=broad_index_history)
        logger.debug(
            "Volume and positioning metrics calculated",
            extra={"ticker": self.ticker, "metric_count": len(result)}
        )
        return result
    
    def get_peer_metrics(self, peer_prices: dict = None, window: int = 63) -> dict:
        """
        Calculate peer intelligence metrics.
        
        Args:
            peer_prices: Dictionary of {ticker: pd.Series} (auto-fetched if None)
            window: Rolling window size
        
        Returns:
            Dictionary with 9+ peer intelligence metrics
        """
        logger.debug("Calculating peer intelligence metrics", extra={"ticker": self.ticker})
        history = self.stock_data_provider.history()
        if history.empty:
            logger.warning("No history data for peer metrics", extra={"ticker": self.ticker})
            return {}
        
        # Auto-fetch peer prices if not provided
        if peer_prices is None:
            peer_prices = self.peer_discovery_provider.get_peers_stock_data()
        
        calculator = PeerIntelligenceCalculator(history)
        # Set both class and instance attribute to preserve backward-compatible behavior
        try:
            type(calculator).ROLLING_WINDOW = window
        except Exception:
            pass
        calculator.ROLLING_WINDOW = window
        result = calculator.calculate_all(peer_prices)
        logger.debug(
            "Peer metrics calculated",
            extra={"ticker": self.ticker, "metric_count": len(result)}
        )
        return result
    
    def get_macro_metrics(self, macro_data: dict = None) -> dict:
        """
        Calculate macro sensitivity metrics.
        
        Args:
            macro_data: Dictionary of {factor_name: pd.Series} (auto-fetched from FRED if None)
        
        Returns:
            Dictionary with 40+ macro sensitivity metrics
        """
        logger.debug("Calculating macro sensitivity metrics", extra={"ticker": self.ticker})
        history = self.stock_data_provider.history()
        if history.empty:
            logger.warning("No history data for macro metrics", extra={"ticker": self.ticker})
            return {}
        
        # Auto-fetch from FRED if not provided
        if macro_data is None:
            try:
                macro_data = get_macro_data_for_stock(history)
                logger.info(
                    "FRED macro data fetched",
                    extra={"ticker": self.ticker, "indicator_count": len(macro_data)}
                )
            except Exception as e:
                logger.error(
                    "Error fetching FRED data",
                    extra={"ticker": self.ticker, "error": str(e)},
                    exc_info=True
                )
                macro_data = {}
        
        calculator = MacroSensitivityCalculator(history)
        result = calculator.calculate_all(macro_data)
        logger.debug(
            "Macro sensitivity metrics calculated",
            extra={"ticker": self.ticker, "metric_count": len(result)}
        )
        return result
    
    # --- orchestration ---
    
    def get_all_metrics(self, 
                       request: dict = None,
                       sector_index_history: pd.DataFrame = None,
                       broad_index_history: pd.DataFrame = None,
                       peer_prices: dict = None,
                       macro_data: dict = None) -> dict:
        """
        Calculate all available metrics based on request configuration.
        
        Args:
            request: Dict specifying which metrics to calculate:
                - include_volume: Include volume/positioning metrics (default: False)
                - include_peer: Include peer intelligence metrics (default: False)
                - include_macro: Include macro sensitivity metrics (default: False)
            sector_index_history: Optional sector index prices (auto-fetched if needed)
            broad_index_history: Optional broad market index prices (auto-fetched if needed)
            peer_prices: Optional peer price data (auto-fetched if needed)
            macro_data: Optional macro indicator data (auto-fetched from FRED if needed)
        
        Returns:
            Dict with all calculated metrics
        """
        if request is None:
            request = {'ticker': self.ticker}
        
        logger.info(
            "Starting metrics calculation",
            extra={"ticker": self.ticker, "request_keys": list(request.keys())}
        )
        
        response = {'ticker': self.ticker, 'metrics': {}}
        errors = []
        
        try:
            # Always calculate core metrics
            logger.debug("Calculating core metrics", extra={"ticker": self.ticker})
            tech_metrics = self.get_technical_metrics()
            adv_metrics = self.get_advanced_technical_metrics()
            fin_metrics = self.get_financial_metrics()
            
            # Combine core metrics
            all_metrics = {}
            all_metrics.update(tech_metrics)
            all_metrics.update(adv_metrics)
            all_metrics.update(fin_metrics)
            logger.debug(
                "Core metrics calculated",
                extra={"ticker": self.ticker, "metric_count": len(all_metrics)}
            )
            
            # Volume/positioning metrics
            if request.get('include_volume') or request.get('include_peer'):
                logger.debug("Calculating volume metrics", extra={"ticker": self.ticker})
                try:
                    vol_metrics = self.get_volume_positioning_metrics(
                        sector_index_history, broad_index_history
                    )
                    all_metrics.update(vol_metrics)
                except Exception as e:
                    errors.append("Volume metrics failed: %s" % (str(e),))
                    logger.warning("Volume metrics failed: %s", e, extra={"ticker": self.ticker})
            
            # Peer intelligence metrics
            if request.get('include_peer'):
                logger.debug("Calculating peer metrics", extra={"ticker": self.ticker})
                try:
                    peer_metrics = self.get_peer_metrics(peer_prices)
                    all_metrics.update(peer_metrics)
                except Exception as e:
                    errors.append("Peer metrics failed: %s" % (str(e),))
                    logger.warning("Peer metrics failed: %s", e, extra={"ticker": self.ticker})
            
            # Macro sensitivity metrics
            if request.get('include_macro'):
                logger.debug("Calculating macro sensitivity metrics", extra={"ticker": self.ticker})
                try:
                    macro_metrics = self.get_macro_metrics(macro_data)
                    all_metrics.update(macro_metrics)
                except Exception as e:
                    errors.append("Macro metrics failed: %s" % (str(e),))
                    logger.warning("Macro metrics failed: %s", e, extra={"ticker": self.ticker})
            
            response['metrics'] = all_metrics
            response['total_metrics'] = len(all_metrics)
            response['errors'] = errors if errors else None
            
            logger.info(
                "Metrics calculation completed",
                extra={
                    "ticker": self.ticker,
                    "total_metrics": len(all_metrics),
                    "error_count": len(errors)
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Error during metrics calculation",
                extra={"ticker": self.ticker, "error": str(e)},
                exc_info=True
            )
            response['errors'] = [str(e)]
            return response
    
    
    # --- thesis validation context ---

    def get_thesis_context(self, include_peer_metrics: bool = True) -> ThesisQuantitativeContext:
        """Get curated quantitative context for thesis validation agents.

        Compiles trend-focused metrics useful for thesis validation without overwhelming
        the agent with raw outputs.
        """
        logger.info("Building thesis quantitative context", extra={"ticker": self.ticker, "include_peers": include_peer_metrics})

        context = ThesisQuantitativeContext(stock_ticker=self.ticker, data_as_of=datetime.now().strftime("%Y-%m-%d"))

        try:
            # Get financial statements and basic data
            income = self.stock_data_provider.get_income_stmt()
            balance = self.stock_data_provider.get_balance_sheet()
            cf = self.stock_data_provider.get_cashflow()
            info = self.stock_data_provider.get_info()
            history = self.stock_data_provider.history()

            # --- growth trajectory ---
            if not income.empty and not balance.empty and not cf.empty:
                fin_calc = FinancialCalculator(income, balance, cf, freq=self.financial_freq)
                context.revenue_growth_yoy = fin_calc.calculate_revenue_growth_yoy()
                context.earnings_growth_yoy = fin_calc.calculate_earnings_growth_yoy()
                context.revenue_cagr_3yr = fin_calc.calculate_revenue_cagr(3)
                context.fcf_growth_yoy = fin_calc.calculate_fcf_growth_yoy()

                # Margins and capital efficiency
                context.gross_margin_current = fin_calc.calculate_gross_margin()
                context.gross_margin_yoy_change = fin_calc.calculate_gross_margin_yoy_change()
                context.operating_margin_current = fin_calc.calculate_operating_margin()
                context.operating_margin_yoy_change = fin_calc.calculate_operating_margin_yoy_change()
                context.roe_current = fin_calc.calculate_return_on_equity()
                context.roe_vs_prior_year_change = fin_calc.calculate_roe_yoy_change()
                context.debt_to_equity_current = fin_calc.calculate_debt_to_equity()
                context.debt_to_equity_yoy_change = fin_calc.calculate_debt_to_equity_yoy_change()

            # --- valuation context ---
            if info:
                context.pe_current = info.get('trailingPE')
                context.forward_pe = info.get('forwardPE')
                context.pe_5yr_avg = None

                fifty_two_week_high = info.get('fiftyTwoWeekHigh')
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if fifty_two_week_high and current_price:
                    context.price_vs_52w_high_pct = current_price / fifty_two_week_high

            # --- analyst sentiment ---
            try:
                recommendations = self.research_data_provider.get_recommendations()
                price_targets = self.research_data_provider.get_analyst_price_targets()
                eps_revisions = self.research_data_provider.get_eps_revisions()

                if info:
                    context.analyst_recommendation_mean = info.get('recommendationMean')

                if info:
                    target_mean = info.get('targetMeanPrice')
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if target_mean and current_price and current_price > 0:
                        context.analyst_target_price_upside = (target_mean - current_price) / current_price

                if not eps_revisions.empty:
                    context.analyst_revision_direction = self._determine_revision_direction(eps_revisions)

            except Exception as e:
                logger.debug("Could not fetch analyst data: %s", e)

            # --- peer context ---
            if include_peer_metrics:
                try:
                    peer_prices = self.peer_discovery_provider.get_peers_stock_data()
                    if peer_prices and not history.empty:
                        peer_calc = PeerIntelligenceCalculator(history)
                        peer_metrics = peer_calc.calculate_all(peer_prices)
                        context.excess_return_vs_peers_1yr = peer_metrics.get('rolling_excess_return_mean_63d')
                        context.outperformance_frequency = peer_metrics.get('outperformance_frequency')
                except Exception as e:
                    logger.debug("Could not fetch peer metrics: %s", e)

            logger.info("Thesis context built successfully", extra={"ticker": self.ticker, "metrics_populated": sum(1 for v in context.__dict__.values() if v is not None)})

        except Exception as e:
            logger.error("Error building thesis context", extra={"ticker": self.ticker, "error": str(e)}, exc_info=True)

        return context
    
    def _determine_revision_direction(self, eps_revisions: pd.DataFrame) -> str:
        """
        Determine overall direction of EPS revisions.
        
        Args:
            eps_revisions: DataFrame with EPS revision data
            
        Returns:
            "Up", "Down", "Mixed", or "Stable"
        """
        try:
            # yfinance eps_revisions typically has 'upLast7days', 'upLast30days', etc.
            up_cols = [c for c in eps_revisions.columns if 'up' in c.lower()]
            down_cols = [c for c in eps_revisions.columns if 'down' in c.lower()]
            
            total_up = 0
            total_down = 0
            
            for col in up_cols:
                val = eps_revisions[col].iloc[0] if len(eps_revisions) > 0 else 0
                if pd.notna(val):
                    total_up += val
            
            for col in down_cols:
                val = eps_revisions[col].iloc[0] if len(eps_revisions) > 0 else 0
                if pd.notna(val):
                    total_down += val
            
            if total_up == 0 and total_down == 0:
                return "Stable"
            elif total_up > total_down * 2:
                return "Up"
            elif total_down > total_up * 2:
                return "Down"
            else:
                return "Mixed"
        except Exception:
            return "Unknown"