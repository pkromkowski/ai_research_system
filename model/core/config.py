import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

load_dotenv()


class EnvConfig:
    """Small helper for reading and casting environment variables.

    Usage: EnvConfig.get('FRED_API_KEY', cast=str, aliases=['FRED_KEY'])
    """

    @staticmethod
    def get(name: str, default: Any = None, cast: Optional[Callable] = None, aliases: Optional[list] = None):
        aliases = aliases or []
        for key in (name, *aliases):
            val = os.getenv(key)
            if val is not None:
                if cast is not None:
                    try:
                        return cast(val)
                    except Exception as exc:  # keep error explicit
                        raise ValueError(f"Invalid value for {key}: {exc}")
                return val
        return default


# Default macro economic indicators
DEFAULT_MACRO_INDICATORS = {
    'GDPC1': 'Real Gross Domestic Product',
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
    'UNRATE': 'Unemployment Rate',
    'GS2': '2-Year Treasury Constant Maturity Rate',
    'GS10': '10-Year Treasury Constant Maturity Rate',
    'GS30': '30-Year Treasury Constant Maturity Rate',
    'T10Y2Y': '10-Year minus 2-Year Treasury Spread',
    'DFF': 'Effective Federal Funds Rate',
    'PAYEMS': 'Total Nonfarm Payroll',
    'INDPRO': 'Industrial Production Index',
    'UMCSENT': 'University of Michigan Inflation Expectation',
}

# Macro indicators grouped by category
MACRO_CATEGORIES = {
    'growth': ['GDPC1', 'PAYEMS', 'INDPRO'],
    'inflation': ['CPIAUCSL', 'UMCSENT'],
    'rates': ['GS2', 'GS10', 'GS30', 'DFF', 'T10Y2Y'],
}

# Debug flag
DEBUG = EnvConfig.get('DEBUG', default='False', cast=lambda v: v.lower() == 'true')


@dataclass
class BaseConfig:
    """Mixin-like helper for dataclasses that load from envs and validate."""

    @classmethod
    def _env(cls, name: str, default: Any = None, cast: Optional[Callable] = None, aliases: Optional[list] = None):
        return EnvConfig.get(name, default=default, cast=cast, aliases=aliases)

    def validate(self, required: bool = True):
        """Default no-op; override in subclasses with required flag."""
        return None


@dataclass
class FredConfig(BaseConfig):
    api_key: Optional[str] = None
    observation_start: str = '2006-01-01'
    default_indicators: Optional[Dict[str, str]] = None
    categories: Optional[Dict[str, list]] = None

    def __post_init__(self):
        self.api_key = self.api_key or self._env('FRED_API_KEY')
        # Respect explicit constructor values: only consult env var when using the dataclass default
        if self.observation_start == FredConfig.observation_start:
            self.observation_start = self._env('FRED_OBSERVATION_START', default=self.observation_start)
        self.default_indicators = self.default_indicators or DEFAULT_MACRO_INDICATORS
        self.categories = self.categories or MACRO_CATEGORIES

    def validate(self, required: bool = True):
        if required and not self.api_key:
            raise ValueError('FRED_API_KEY not set in environment')
        if not isinstance(self.observation_start, str):
            raise ValueError('observation_start must be an ISO date string')


@dataclass
class PerplexityConfig(BaseConfig):
    api_key: Optional[str] = None
    api_base_url: str = "https://api.perplexity.ai"

    def __post_init__(self):
        self.api_key = self.api_key or self._env('PERPLEXITY_API_KEY')

    def validate(self, required: bool = True):
        if required and not self.api_key:
            raise ValueError('PERPLEXITY_API_KEY not set in environment')


@dataclass
class FilingConfig(BaseConfig):
    base_dir: Path = None
    max_retries: int = 3
    timeout_seconds: int = 30

    def __post_init__(self):
        base = self._env('TEN_K_BASE_DIR', default='version2/documents', aliases=['10K_BASE_DIR'])
        self.base_dir = Path(base) if self.base_dir is None else Path(self.base_dir)

        retries = self._env('TEN_K_MAX_RETRIES', default=None, cast=int, aliases=['10K_MAX_RETRIES'])
        if retries is not None:
            self.max_retries = retries
        timeout = self._env('TEN_K_TIMEOUT_SECONDS', default=None, cast=int, aliases=['10K_TIMEOUT_SECONDS'])
        if timeout is not None:
            self.timeout_seconds = timeout

    def validate(self, required: bool = True) -> None:
        if not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)
        if self.max_retries < 1:
            raise ValueError('max_retries must be >= 1')
        if self.timeout_seconds < 1:
            raise ValueError('timeout_seconds must be >= 1')

    def get_raw_dir(self, ticker: str) -> Path:
        return self.base_dir / 'raw' / ticker.upper()

    def get_extractions_dir(self, ticker: str) -> Path:
        return self.base_dir / 'extractions' / ticker.upper()

    def get_metadata_dir(self, ticker: str) -> Path:
        return self.base_dir / 'metadata' / ticker.upper()


@dataclass
class ClaudeConfig(BaseConfig):
    api_key: Optional[str] = None
    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 3000
    temperature: float = 0.0

    def __post_init__(self):
        self.api_key = self.api_key or self._env('ANTHROPIC_API_KEY', aliases=['CLAUDE_API_KEY'])
        self.model = self._env('CLAUDE_MODEL', default=self.model)
        tokens = self._env('CLAUDE_MAX_TOKENS', default=None, cast=int)
        if tokens is not None:
            self.max_tokens = tokens
        temp = self._env('CLAUDE_TEMPERATURE', default=None, cast=float)
        if temp is not None:
            self.temperature = temp

    def validate(self, required: bool = True) -> None:
        if required and not self.api_key:
            raise ValueError('ANTHROPIC_API_KEY not set. Set via environment or ClaudeConfig.api_key')
        if self.max_tokens < 100:
            raise ValueError('max_tokens must be >= 100')
        if not 0 <= self.temperature <= 2:
            raise ValueError('temperature must be between 0 and 2')


@dataclass
class AnalyticsConfig(BaseConfig):
    period: str = '1y'
    financial_frequency: str = 'quarterly'

    def __post_init__(self):
        self.period = self._env('ANALYTICS_PERIOD', default=self.period)
        self.financial_frequency = self._env('ANALYTICS_FINANCIAL_FREQUENCY', default=self.financial_frequency)

    def validate(self, required: bool = True) -> None:
        valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        if self.period not in valid_periods:
            raise ValueError(f'period must be one of {valid_periods}')
        valid_freq = ['annual', 'quarterly']
        if self.financial_frequency not in valid_freq:
            raise ValueError(f'financial_frequency must be one of {valid_freq}')


class AppConfig:
    """Central application configuration container.

    Access sub-configs as attributes (e.g., `AppConfig.fred`).
    Use `AppConfig.from_env()` for a validated instance reflecting the
    current environment.
    """

    filing: FilingConfig = FilingConfig()
    claude: ClaudeConfig = ClaudeConfig()
    fred: FredConfig = FredConfig()
    perplexity: PerplexityConfig = PerplexityConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()

    @staticmethod
    def validate_all(strict: bool = False) -> None:
        AppConfig.filing.validate()
        AppConfig.claude.validate(required=strict)
        AppConfig.fred.validate(required=strict)
        AppConfig.perplexity.validate(required=strict)
        AppConfig.analytics.validate()

    @staticmethod
    def check_availability() -> Dict[str, Dict[str, Any]]:
        """Return availability map for each config.

        For each named sub-config return a dict with keys:
        - available: bool
        - reason: Optional[str] explaining failure when available is False
        """
        results: Dict[str, Dict[str, Any]] = {}
        # Construct fresh instances so availability reflects current environment
        configs = {
            'filing': FilingConfig(),
            'claude': ClaudeConfig(),
            'fred': FredConfig(),
            'perplexity': PerplexityConfig(),
            'analytics': AnalyticsConfig(),
        }
        for name, cfg in configs.items():
            try:
                cfg.validate(required=True)
                results[name] = {'available': True, 'reason': None}
            except Exception as e:
                results[name] = {'available': False, 'reason': str(e)}
        return results

    @staticmethod
    def from_env(strict: bool = False) -> 'AppConfig':
        config = AppConfig()
        config.validate_all(strict=strict)
        return config

