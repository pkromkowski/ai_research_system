import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    import requests as _requests
except Exception:
    _requests = None

try:
    from perplexity import Perplexity as _Perplexity
except Exception:
    _Perplexity = None

from model.core.config import PerplexityConfig
from model.core.types import ResearchResult
from model.prompts.perplexity_research_prompts import (
    BULL_BEAR_CASES_PROMPT,
    COMPETITIVE_LANDSCAPE_PROMPT,
    EARNINGS_ANALYSIS_PROMPT,
    ESG_SUSTAINABILITY_PROMPT,
    INSTITUTIONAL_OWNERSHIP_PROMPT,
    MACRO_SENSITIVITY_PROMPT,
    MANAGEMENT_GOVERNANCE_PROMPT,
    PROMPT_REGISTRY,
    RECENT_NEWS_PROMPT,
    RISK_FACTORS_PROMPT,
    SHORT_INTEREST_SENTIMENT_PROMPT,
    SUPPLY_CHAIN_RELATIONSHIPS_PROMPT,
    VALUATION_ANALYST_PROMPT,
)

logger = logging.getLogger(__name__)


class PerplexityResearchProvider:
    """Investment research provider using Perplexity AI's search-grounded models."""

    DEFAULT_MODEL: str = "sonar-pro"
    REASONING_MODEL: str = "sonar-reasoning-pro"
    DEFAULT_TEMPERATURE: float = 0.2
    DEFAULT_MAX_TOKENS: int = 4096
    DEFAULT_NEWS_DAYS: int = 7
    API_BASE_URL: str = "https://api.perplexity.ai"
    REQUEST_TIMEOUT: int = 120

    MODELS: Dict[str, str] = {
        "sonar": "Fast search for quick factual queries",
        "sonar-pro": "Complex queries with follow-up support",
        "sonar-reasoning-pro": "Multi-step analysis with chain of thought",
        "sonar-deep-research": "Exhaustive research with comprehensive reports",
    }
    
    def __init__(self, ticker: str, api_key: Optional[str] = None, config: Optional["PerplexityConfig"] = None, client: Optional[Any] = None):
        """Initialize Perplexity research provider.

        Args:
            ticker: Stock ticker symbol
            api_key: Perplexity API key (overrides config)
            config: Optional PerplexityConfig instance
            client: Optional pre-initialized client for testing

        Raises:
            ValueError: If API key not found
        """
        self.ticker = ticker.upper()
        cfg = config or PerplexityConfig()
        self.api_key = api_key or cfg.api_key

        if not self.api_key and client is None:
            raise ValueError(
                "PERPLEXITY_API_KEY not found. Set it in .env or pass api_key/config."
            )

        self._client: Any = client
        self._client_type: Optional[str] = "injected" if client is not None else None
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Simple prompt formatter used for demos and tests.

        For production, use richer templating and safety checks.
        """
        try:
            return template.format(**kwargs)
        except Exception:
            # Fallback to a simple representation
            return str(template)
    
    @property
    def client(self) -> Any:
        """Lazy-load the API client."""
        if self._client is None:
            self._init_client()
        return self._client

    def _init_client(self) -> None:
        """Initialize the Perplexity client. Tries SDK first, falls back to requests."""
        if _Perplexity is not None:
            try:
                self._client = _Perplexity(api_key=self.api_key)
                self._client_type = "perplexity_sdk"
                logger.debug("Initialized Perplexity SDK client")
                return
            except Exception as e:
                logger.debug("Perplexity SDK init failed: %s", e)

        if _requests is not None:
            self._client_type = "requests"
            logger.debug("Using requests-based client")
            return

        raise ImportError("No HTTP client available. Install requests or the Perplexity SDK (perplexityai).")

    def _query(self, 
               prompt: str, 
               model: Optional[str] = None,
               temperature: Optional[float] = None, 
               max_tokens: Optional[int] = None) -> ResearchResult:
        """Query Perplexity API.

        Args:
            prompt: The research query
            model: Model to use
            temperature: Response randomness
            max_tokens: Maximum response length

        Returns:
            ResearchResult with content and citations
        """
        model = model or self.DEFAULT_MODEL
        temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        messages = [{"role": "user", "content": prompt}]
        _ = self.client

        if self._client_type == "perplexity_sdk":
            return self._query_sdk(prompt, model, messages, temperature, max_tokens)
        else:
            return self._query_requests(prompt, model, messages, temperature, max_tokens)

    def _query_sdk(self, 
                   prompt: str, 
                   model: str, 
                   messages: List[Dict[str, str]],
                   temperature: float, 
                   max_tokens: int) -> ResearchResult:
        """Query using official Perplexity SDK."""
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content if response.choices else ""
        citations = getattr(response, 'citations', []) or []

        return ResearchResult(
            query=prompt,
            content=content,
            citations=citations,
            model_used=model,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

    def _query_requests(self, 
                        prompt: str, 
                        model: str, 
                        messages: List[Dict[str, str]],
                        temperature: float, 
                        max_tokens: int) -> ResearchResult:
        """Query using HTTP requests."""
        if _requests is None:
            raise ImportError("requests is required for the requests-based Perplexity client")

        url = f"{self.API_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = _requests.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=self.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        citations = data.get("citations", [])

        return ResearchResult(
            query=prompt,
            content=content,
            citations=citations,
            model_used=model,
            raw_response=data,
        )

    def get_recent_news(self, days: Optional[int] = None) -> ResearchResult:
        """Get recent news and developments.

        Args:
            days: How many days back to search

        Returns:
            ResearchResult with news summary and citations
        """
        days = days or self.DEFAULT_NEWS_DAYS
        prompt = self.format_prompt(
            RECENT_NEWS_PROMPT,
            ticker_or_company=self.ticker,
            days=days,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_earnings_analysis(self) -> ResearchResult:
        """Get analysis of most recent earnings and forward guidance.

        Returns:
            ResearchResult with earnings analysis and citations
        """
        prompt = self.format_prompt(
            EARNINGS_ANALYSIS_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_competitive_landscape(self) -> ResearchResult:
        """Analyze competitive positioning and recent developments.

        Returns:
            ResearchResult with competitive analysis and citations
        """
        prompt = self.format_prompt(
            COMPETITIVE_LANDSCAPE_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_risk_factors(self) -> ResearchResult:
        """Research current and emerging risk factors.

        Returns:
            ResearchResult with risk analysis and citations
        """
        prompt = self.format_prompt(
            RISK_FACTORS_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.REASONING_MODEL)

    def get_bull_bear_cases(self) -> ResearchResult:
        """Get balanced bull and bear investment cases with price targets.

        Returns:
            ResearchResult with bull/bear analysis and citations
        """
        prompt = self.format_prompt(
            BULL_BEAR_CASES_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.REASONING_MODEL)

    def get_management_governance(self) -> ResearchResult:
        """Research management team and corporate governance.

        Returns:
            ResearchResult with management/governance info and citations
        """
        prompt = self.format_prompt(
            MANAGEMENT_GOVERNANCE_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_valuation_analyst_sentiment(self) -> ResearchResult:
        """Get current valuation metrics and analyst sentiment.

        Returns:
            ResearchResult with valuation data, analyst ratings, and price targets
        """
        prompt = self.format_prompt(
            VALUATION_ANALYST_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_institutional_ownership(self) -> ResearchResult:
        """Research institutional ownership and recent fund flows.

        Returns:
            ResearchResult with ownership data from 13F filings
        """
        prompt = self.format_prompt(
            INSTITUTIONAL_OWNERSHIP_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_short_interest_sentiment(self) -> ResearchResult:
        """Get short interest data and market sentiment indicators.

        Returns:
            ResearchResult with short interest, options flow, and sentiment data
        """
        prompt = self.format_prompt(
            SHORT_INTEREST_SENTIMENT_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_esg_profile(self) -> ResearchResult:
        """Research ESG (Environmental, Social, Governance) profile.

        Returns:
            ResearchResult with ESG ratings, controversies, and sustainability data
        """
        prompt = self.format_prompt(
            ESG_SUSTAINABILITY_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_supply_chain_relationships(self) -> ResearchResult:
        """Analyze supply chain and key business relationships.

        Returns:
            ResearchResult with customer/supplier concentration and partnership data
        """
        prompt = self.format_prompt(
            SUPPLY_CHAIN_RELATIONSHIPS_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.DEFAULT_MODEL)

    def get_macro_sensitivity(self) -> ResearchResult:
        """Analyze macroeconomic sensitivity and scenario impacts.

        Returns:
            ResearchResult with economic sensitivity analysis
        """
        prompt = self.format_prompt(
            MACRO_SENSITIVITY_PROMPT,
            ticker_or_company=self.ticker,
        )
        return self._query(prompt, model=self.REASONING_MODEL)

    def get_comprehensive_research(self, 
                                    include_extended: bool = False) -> Dict[str, Any]:
        """Run comprehensive research combining multiple methods.

        Args:
            include_extended: Include additional research sections

        Returns:
            Dictionary with all research sections and metadata
        """
        results: Dict[str, Any] = {
            "ticker": self.ticker,
            "timestamp": datetime.now().isoformat(),
            "research_type": "extended" if include_extended else "core",
            "sections": {},
        }

        core_sections: List[tuple[str, Callable[[], ResearchResult]]] = [
            ("recent_news", self.get_recent_news),
            ("earnings", self.get_earnings_analysis),
            ("competitive_landscape", self.get_competitive_landscape),
            ("risk_factors", self.get_risk_factors),
            ("bull_bear_cases", self.get_bull_bear_cases),
            ("management_governance", self.get_management_governance),
        ]

        extended_sections: List[tuple[str, Callable[[], ResearchResult]]] = [
            ("valuation_sentiment", self.get_valuation_analyst_sentiment),
            ("institutional_ownership", self.get_institutional_ownership),
            ("short_interest_sentiment", self.get_short_interest_sentiment),
            ("esg_profile", self.get_esg_profile),
            ("supply_chain", self.get_supply_chain_relationships),
            ("macro_sensitivity", self.get_macro_sensitivity),
        ]

        sections_to_run = core_sections + extended_sections if include_extended else core_sections

        for name, method in sections_to_run:
            try:
                result = method()
                results["sections"][name] = result.to_dict()
                logger.debug("Completed research section: %s", name)
            except Exception as e:
                logger.warning("Failed research section %s: %s", name, e)
                results["sections"][name] = {"error": str(e)}

        sections_data = results["sections"]
        results["sections_completed"] = sum(1 for s in sections_data.values() if "error" not in s)
        results["sections_failed"] = sum(1 for s in sections_data.values() if "error" in s)

        logger.info(
            "Comprehensive research for %s: %s/%s sections completed",
            self.ticker,
            results['sections_completed'],
            len(sections_to_run),
        )

        return results

    def get_custom_research(self, 
                            prompt_name: str, 
                            **kwargs: Any) -> ResearchResult:
        """Run a specific research prompt by name.

        Args:
            prompt_name: Name of the prompt from PROMPT_REGISTRY
            **kwargs: Additional format parameters for the prompt

        Returns:
            ResearchResult with research findings

        Raises:
            KeyError: If prompt_name not found in registry
        """
        if prompt_name not in PROMPT_REGISTRY:
            available = list(PROMPT_REGISTRY.keys())
            raise KeyError("Unknown prompt: %s. Available: %s" % (prompt_name, available))
        
        prompt_info = PROMPT_REGISTRY[prompt_name]
        prompt = self.format_prompt(
            prompt_info["template"],
            ticker_or_company=self.ticker,
            **kwargs,
        )

