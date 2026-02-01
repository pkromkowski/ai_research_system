from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Centralized node-type constants for reuse across agents
NODE_TYPE_ASSUMPTION = "ASSUMPTION"
NODE_TYPE_DRIVER = "DRIVER"
NODE_TYPE_OUTCOME = "OUTCOME"


# -- Stock Analytics Metrics -- 
@dataclass
class ThesisQuantitativeContext:
    """
    Curated quantitative metrics from Stage 1 Stock Analytics to provide 
    context for thesis validation agents.
    
    These metrics are SUPPLEMENTARY context, not the primary basis for analysis.
    Agents should use these to ground their research, but shouldn't over-weight 
    them versus their own independent analysis.
    
    Focus: Long-term fundamental investing metrics showing TRENDS and TRAJECTORY,
    not static point-in-time values.
    """
    stock_ticker: str
    data_as_of: str  # When data was fetched
    
    # --- Growth trajectory ---
    # These show how the business is evolving, not just where it is
    revenue_growth_yoy: Optional[float] = None  # YoY revenue growth rate
    earnings_growth_yoy: Optional[float] = None  # YoY earnings growth rate
    revenue_cagr_3yr: Optional[float] = None  # 3-year compound annual growth
    fcf_growth_yoy: Optional[float] = None  # Free cash flow growth
    
    # --- Margin trends ---
    # Change vs prior year - shows improvement or deterioration
    gross_margin_current: Optional[float] = None  # Current gross margin
    gross_margin_yoy_change: Optional[float] = None  # Change vs year ago (pp)
    operating_margin_current: Optional[float] = None  # Current operating margin
    operating_margin_yoy_change: Optional[float] = None  # Change vs year ago (pp)
    
    # --- Valuation context ---
    # Current vs historical - shows if expensive or cheap vs its own history
    pe_current: Optional[float] = None  # Current P/E ratio
    pe_5yr_avg: Optional[float] = None  # 5-year average P/E
    forward_pe: Optional[float] = None  # Forward P/E from estimates
    price_vs_52w_high_pct: Optional[float] = None  # Current price as % of 52-week high
    
    # --- Capital efficiency ---
    # Shows improving or declining returns on capital
    roe_current: Optional[float] = None  # Current return on equity
    roe_vs_prior_year_change: Optional[float] = None  # ROE change vs prior year (pp)
    debt_to_equity_current: Optional[float] = None  # Current D/E ratio
    debt_to_equity_yoy_change: Optional[float] = None  # D/E change vs year ago
    
    # --- Analyst sentiment ---
    # External perspective on the stock
    analyst_recommendation_mean: Optional[float] = None  # 1-5 scale (1=Strong Buy)
    analyst_target_price_upside: Optional[float] = None  # % upside to mean target
    analyst_revision_direction: Optional[str] = None  # "Up", "Down", "Mixed", "Stable"
    
    # --- Peer context ---
    # Performance relative to comparable companies
    excess_return_vs_peers_1yr: Optional[float] = None  # Outperformance vs peer median
    outperformance_frequency: Optional[float] = None  # % of periods beating peers
    
    def to_prompt_context(self) -> str:
        """
        Format context for inclusion in thesis validation prompts.
        
        Returns concise, relevant summary that won't overwhelm the agent.
        """
        sections = []
        
        sections.append(f"QUANTITATIVE CONTEXT FOR {self.stock_ticker} (as of {self.data_as_of})")
        sections.append("=" * 50)
        sections.append("NOTE: These metrics provide supplementary context only. ")
        sections.append("Your independent research and analysis should take precedence.")
        sections.append("")
        
        # Growth
        growth_items = []
        if self.revenue_growth_yoy is not None:
            growth_items.append(f"Revenue Growth (YoY): {self.revenue_growth_yoy:+.1%}")
        if self.earnings_growth_yoy is not None:
            growth_items.append(f"Earnings Growth (YoY): {self.earnings_growth_yoy:+.1%}")
        if self.revenue_cagr_3yr is not None:
            growth_items.append(f"Revenue CAGR (3yr): {self.revenue_cagr_3yr:+.1%}")
        if self.fcf_growth_yoy is not None:
            growth_items.append(f"FCF Growth (YoY): {self.fcf_growth_yoy:+.1%}")
        if growth_items:
            sections.append("GROWTH TRAJECTORY:")
            sections.extend([f"  • {item}" for item in growth_items])
            sections.append("")
        
        # Margins
        margin_items = []
        if self.gross_margin_current is not None:
            margin_str = f"Gross Margin: {self.gross_margin_current:.1%}"
            if self.gross_margin_yoy_change is not None:
                margin_str += f" ({self.gross_margin_yoy_change:+.1f}pp vs year ago)"
            margin_items.append(margin_str)
        if self.operating_margin_current is not None:
            margin_str = f"Operating Margin: {self.operating_margin_current:.1%}"
            if self.operating_margin_yoy_change is not None:
                margin_str += f" ({self.operating_margin_yoy_change:+.1f}pp vs year ago)"
            margin_items.append(margin_str)
        if margin_items:
            sections.append("MARGIN TRENDS:")
            sections.extend([f"  • {item}" for item in margin_items])
            sections.append("")
        
        # Valuation
        valuation_items = []
        if self.pe_current is not None and self.pe_5yr_avg is not None:
            valuation_items.append(f"P/E: {self.pe_current:.1f}x (5yr avg: {self.pe_5yr_avg:.1f}x)")
        elif self.pe_current is not None:
            valuation_items.append(f"P/E: {self.pe_current:.1f}x")
        if self.forward_pe is not None:
            valuation_items.append(f"Forward P/E: {self.forward_pe:.1f}x")
        if self.price_vs_52w_high_pct is not None:
            valuation_items.append(f"Price vs 52w High: {self.price_vs_52w_high_pct:.1%}")
        if valuation_items:
            sections.append("VALUATION CONTEXT:")
            sections.extend([f"  • {item}" for item in valuation_items])
            sections.append("")
        
        # Capital efficiency
        efficiency_items = []
        if self.roe_current is not None:
            roe_str = f"ROE: {self.roe_current:.1%}"
            if self.roe_vs_prior_year_change is not None:
                roe_str += f" ({self.roe_vs_prior_year_change:+.1f}pp vs year ago)"
            efficiency_items.append(roe_str)
        if self.debt_to_equity_current is not None:
            de_str = f"Debt/Equity: {self.debt_to_equity_current:.2f}x"
            if self.debt_to_equity_yoy_change is not None:
                de_str += f" ({self.debt_to_equity_yoy_change:+.2f} vs year ago)"
            efficiency_items.append(de_str)
        if efficiency_items:
            sections.append("CAPITAL EFFICIENCY:")
            sections.extend([f"  • {item}" for item in efficiency_items])
            sections.append("")
        
        # Analyst sentiment
        sentiment_items = []
        if self.analyst_recommendation_mean is not None:
            rec_label = self._recommendation_label(self.analyst_recommendation_mean)
            sentiment_items.append(f"Analyst Rating: {rec_label} ({self.analyst_recommendation_mean:.2f})")
        if self.analyst_target_price_upside is not None:
            sentiment_items.append(f"Target Price Upside: {self.analyst_target_price_upside:+.1%}")
        if self.analyst_revision_direction:
            sentiment_items.append(f"Recent Revisions: {self.analyst_revision_direction}")
        if sentiment_items:
            sections.append("ANALYST SENTIMENT:")
            sections.extend([f"  • {item}" for item in sentiment_items])
            sections.append("")
        
        # Peer performance
        peer_items = []
        if self.excess_return_vs_peers_1yr is not None:
            peer_items.append(f"Excess Return vs Peers (1yr): {self.excess_return_vs_peers_1yr:+.1%}")
        if self.outperformance_frequency is not None:
            peer_items.append(f"Outperformance Frequency: {self.outperformance_frequency:.0%}")
        if peer_items:
            sections.append("PEER COMPARISON:")
            sections.extend([f"  • {item}" for item in peer_items])
        
        return "\n".join(sections)
    
    def _recommendation_label(self, mean: float) -> str:
        """Convert 1-5 scale to label."""
        if mean <= 1.5:
            return "Strong Buy"
        elif mean <= 2.5:
            return "Buy"
        elif mean <= 3.5:
            return "Hold"
        elif mean <= 4.5:
            return "Sell"
        else:
            return "Strong Sell"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None and not k.startswith('_')
        }
    

# --- NDG ---
@dataclass
class NDGNode:
    """Node in Narrative Decomposition Graph"""
    id: str  # Unique identifier
    claim: str  # The actual claim text
    node_type: str  # "ASSUMPTION" | "DRIVER" | "OUTCOME"
    dependencies: List[str]  # IDs of nodes this depends on (parent nodes)
    
    # 2.3 Assumption Localization & Ownership
    control: str  # "Company" | "Industry" | "Macro" | "Exogenous"
    nature: str  # "Structural" | "Cyclical" | "Execution"
    time_sensitivity: str  # "Short" | "Medium" | "Long"
    
    # 2.4 Evidence Mapping
    evidence_sources: List[Dict] = field(default_factory=list)  # List of {"type": "Quantitative/Qualitative/External", "description": str, "freshness": str}
    evidence_strength: float = 0.0  # 0-1: how well supported
    contradicting_evidence: List[str] = field(default_factory=list)  # Evidence that challenges this claim
    
    # 2.5 Confidence Attribution
    confidence: float = 0.0  # 0-1: analyst confidence in this specific claim
    confidence_basis: str = ""  # Why this confidence level (certainty markers)
    
    # Note: default_factory ensures list fields are non-null and avoids the need
    # for ad-hoc __post_init__ initializers used previously. (Cleaner & safer.)

@dataclass
class NDGEdge:
    """Edge in causal graph"""
    source_id: str  # Node that causes
    target_id: str  # Node that is affected
    relationship: str  # "CAUSES" | "ENABLES" | "MODERATES"
    strength: float = 0.5  # 0-1: how strong is this causal link

@dataclass
class FeedbackLoop:
    """Metadata describing a detected feedback loop (strongly connected component)."""
    id: str
    nodes: List[str]
    avg_edge_strength: float
    avg_evidence: float
    control_mix: Dict[str, int]
    reinforcing: bool = False


@dataclass
class FragilityMetrics:
    """Structural fragility diagnostics (2.6)

    Note: `fragility_components` provides the per-axis contributions used to
    compute `fragility_score`. This supports explainability and sensitivity
    testing in downstream tools/UI.
    """
    assumption_load: int  # Total number of assumptions
    avg_assumptions_per_outcome: float  # How many assumptions feed each outcome
    single_point_failures: List[str]  # Node IDs that, if broken, collapse the thesis
    path_concentration: Dict[str, int]  # {node_id: number of paths through it}
    max_graph_depth: int  # Longest path from assumption to outcome
    fragility_score: float  # 0-1: overall structural brittleness
    fragility_components: Optional[Dict[str, float]] = None  # Breakdown by component (optional)
    feedback_loops: List[FeedbackLoop] = field(default_factory=list)  # Detected feedback loops
    critical_low_evidence_nodes: List[str] = field(default_factory=list)  # Nodes with high importance but low evidence

@dataclass
class NDGOutput:
    """Narrative Decomposition Graph output"""
    stock_ticker: str
    thesis_text: str
    nodes: List[NDGNode]
    edges: List[NDGEdge]
    fragility_metrics: FragilityMetrics
    total_confidence: float  # Should sum to 1.0 across all nodes
    confidence_sum: float = 0.0  # Actual sum of confidences (for downstream checks)
    confidence_consistent: bool = True  # True if confidence_sum within tolerance
    summary_text: str = ""  # Human-readable summary for UI display
    version: int = 1  # For versioning (2.7)
    # Extracted metrics and raw claims (populated by NDG parsing). These are
    # optional and provide a canonical, machine-readable set of assumptions
    # that downstream agents (like CRE) should consume directly.
    extracted_metrics: Optional[Dict[str, Any]] = None
    extracted_claims: Optional[List[Dict[str, Any]]] = None
    created_at: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# --- Red Team Agent ---
@dataclass
class HistoricalAnalog:
    """Historical case where similar assumption failed (3.2)"""
    case_name: str  # e.g., "Peloton demand elasticity (2021-2022)"
    assumption_type: str  # What belief failed
    failure_mode: str  # How it failed (from taxonomy)
    context: str  # Brief description of what happened
    relevance_score: float = 0.0  # 0-1: how similar to current case
    relevance_reasoning: str = ""  # Why this analog matters
    year: int = None  # When it happened

@dataclass
class FailureMode:
    """Classified failure mechanism (3.3)

    Fields added to support flexible taxonomy:
    - taxonomy_match: True if category is from the provided taxonomy, False if an alternative label
    - category_confidence: Model confidence (0.0-1.0) that the selected category is appropriate
    - alternative_category: Optional label when taxonomy_match is False
    """
    category: str  # Primary label (may be taxonomy or alternative)
    description: str  # How the failure manifested
    early_warnings: Optional[List[str]] = None  # Observable indicators
    taxonomy_match: bool = True
    category_confidence: float = 1.0
    alternative_category: Optional[str] = None
    
    def __post_init__(self):
        if self.early_warnings is None:
            self.early_warnings = []

@dataclass
class RelevanceScoring:
    """Relevance dimensions for historical analog (3.4)"""
    business_model_similarity: float = 0.0  # 0-1
    competitive_structure: float = 0.0  # 0-1
    balance_sheet_flexibility: float = 0.0  # 0-1
    regulatory_environment: float = 0.0  # 0-1
    cycle_position: float = 0.0  # 0-1
    overall_relevance: float = 0.0  # 0-1: weighted average
    justification: str = ""  # Explanation of scoring

@dataclass
class RedTeamChallenge:
    """Single adversarial challenge to an assumption (3.5)"""
    node_id: str  # ID of assumption being challenged
    assumption_text: str  # The claim being challenged
    historical_precedent: HistoricalAnalog  # Similar case that failed
    failure_mechanism: FailureMode  # How it could break
    relevance: RelevanceScoring  # Why this matters now
    early_warning_indicators: List[str]  # What to watch for
    challenge_text: str  # Concise, neutral critique
    severity: str = "medium"  # "high" | "medium" | "low"
    score_breakdown: Optional[Dict[str, float]] = None  # Component contributions to challenge priority score
    inputs_used: Optional[Dict[str, Any]] = None  # Raw inputs used to compute scores

    def __post_init__(self):
        if self.score_breakdown is None:
            self.score_breakdown = {}
        if self.inputs_used is None:
            self.inputs_used = {}

@dataclass
class RedTeamOutput:
    """AI Red Team with Memory output (3.7)"""
    stock_ticker: str
    challenges: List[RedTeamChallenge]  # Prioritized challenges
    unresponded_challenges: Optional[List[str]] = None  # Challenge IDs with no response
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    summary_text: str = ""  # Human-readable summary for UI display
    challenged_node_ids: Optional[List[str]] = None  # Node IDs that were challenged
    node_score_breakdowns: Optional[Dict[str, Dict[str, float]]] = None  # Per-node breakdowns for explainability
    created_at: str = None
    
    def __post_init__(self):
        if self.unresponded_challenges is None:
            self.unresponded_challenges = []
        if self.challenged_node_ids is None:
            self.challenged_node_ids = []
        if self.node_score_breakdowns is None:
            self.node_score_breakdowns = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# --- CRE ---
@dataclass
class Scenario:
    """A counterfactual world"""
    name: str
    description: str
    impact: str
    stressed_assumptions: Dict  # Override base assumptions
    plausibility_weight: float = 0.5  # 0-1: how likely is this scenario
    detailed_reasoning: str = ""  # AI explanation of why this scenario matters
    # Optional scenario-specific factor weight adjustments (factor -> multiplier)
    factor_weight_overrides: Optional[Dict[str, float]] = None

@dataclass
class ValuationResult:
    """Output of financial model under a scenario"""
    scenario_name: str
    valuation_change: float  # -0.25 = -25%
    outcome_tier: str  # "SURVIVES" | "IMPAIRED" | "BROKEN"
    narrative_consistent: bool
    margin_of_safety: float
    plausibility_weight: float = 0.5
    detailed_reasoning: str = ""  # Why this scenario leads to this outcome
    # Factorized valuation diagnostics (optional)
    factor_contributions: Optional[Dict[str, float]] = None  # factor -> contribution score (-1..1)
    metric_to_factor_mapping: Optional[Dict[str, Dict[str, float]]] = None  # metric -> {factor: coeff}

@dataclass
class CREOutput:
    """Counterfactual Research Engine output

    Notes:
    - `structural_survival_rate` indicates the fraction of evaluated scenarios where
      the thesis avoided outright failure (i.e., SURVIVES or IMPAIRED). This is a
      structural diagnostic, not an investment recommendation.
    - `structural_fragility_score` captures the CRE's fragility assessment on a 0-1
      scale; it reflects structural brittleness of the thesis in scenario space.
    """
    structural_survival_rate: float  # 0-1: fraction of scenarios where thesis avoids outright failure
    dominant_failure_modes: List[str]
    tail_loss_percentile: float  # Loss at configured percentile (default 5th)
    structural_fragility_score: float  # 0-1: inverse of robustness (structural fragility)
    scenario_results: List[ValuationResult]
    # Additional metadata
    weighted_survival_rate: Optional[float] = None  # Plausibility-weighted survival
    impaired_scenarios: Optional[List[str]] = None  # Scenarios that weakened but didn't break thesis
    summary_text: Optional[str] = None  # Generated summary of key contradictions
    summary_structured: Optional[Dict[str, Any]] = None  # Structured summary JSON (CRE_SUMMARY_SCHEMA)
    # Provenance & factorization diagnostics
    defaults_applied: Optional[List[str]] = None
    inferred_metrics: Optional[List[str]] = None
    factor_scores: Optional[Dict[str, float]] = None  # aggregated factor contributions across scenarios
    metric_factor_mapping: Optional[Dict[str, Dict[str, float]]] = None  # mapping of metrics -> factors

@dataclass
class CREScenarioSet:
    """CRE scenario-only output used by orchestrator before valuation."""
    stock_ticker: str
    scenarios: List[Scenario]
    rejected_scenarios: List[str]
    base_metrics: Dict[str, Any]
    bounds: Dict[str, Any]
    generated_raw: Optional[List[Dict[str, Any]]] = None
    summary_text: Optional[str] = None
    total_duration_ms: Optional[float] = None

@dataclass
class CREResult:
    """Container holding both scenario set (pre-valuation) and evaluated CRE output."""
    scenario_set: CREScenarioSet
    cre_output: CREOutput
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class CREGenerationResult:
    """Output of CRE generation stage (pre-valuation).

    Contains the scenario set produced by CRE, the canonical claims the CRE used,
    and any minimal defaults that were applied to the base metrics.
    """
    scenario_set: CREScenarioSet
    claims: Optional[List[str]] = None
    defaults_applied: Optional[List[str]] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class FTResult:
    """Result returned by FinancialTranslation.run

    Contains the evaluated CREOutput and the structured summary produced by the
    translation step. FT intentionally does not mutate the original CREGenerationResult
    but returns a separate structured summary payload along with the final CREOutput.
    """
    scenario_set: CREScenarioSet
    cre_output: CREOutput
    structured_summary: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()



@dataclass
class BaseAssumptions:
    """Analyst-provided base case assumptions"""
    revenue_growth: List[float]  # 5-year growth rates
    gross_margin: float
    operating_margin: float
    net_retention: float
    wacc: float
    terminal_multiple: float


@dataclass
class ThesisInput:
    """Analyst thesis submission"""
    stock_ticker: str
    narrative: str
    base_assumptions: BaseAssumptions
    stated_confidence: float  # 0-1
    submission_date: str = None

    def __post_init__(self):
        if not self.submission_date:
            self.submission_date = datetime.now().isoformat()


@dataclass
class ThesisValidityOutput:
    """Output of rule-based thesis validity evaluation"""
    stock_ticker: str
    status: str  # "Valid" | "Fragile" | "Broken"
    reasons: List[str]
    dominant_failure_modes: List[str]
    required_conditions: List[str]
    survival_rate: float
    fragility_score: float
    tail_loss: float
    high_severity_challenges: int
    summary_text: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()



@dataclass
class EvidenceQualityScore:
    """Credibility and relevance scoring (4.3)"""
    source_reliability: float  # 0-1: How trustworthy is the source
    recurrence_score: float  # 0-1: One-off vs repeated signal
    forward_relevance: float  # 0-1: How much does this matter for future
    data_quality: float  # 0-1: Precision and completeness
    overall_weight: float = 0.0  # Composite score
    
    def __post_init__(self):
        # Weighted average: source most important, then recurrence
        weights = [0.35, 0.30, 0.20, 0.15]
        scores = [self.source_reliability, self.recurrence_score, 
                 self.forward_relevance, self.data_quality]
        self.overall_weight = sum(w * s for w, s in zip(weights, scores))


@dataclass
class ContradictionTracking:
    """Accumulation and persistence tracking (4.4)"""
    node_id: str
    contradiction_count: int = 0  # How many contradicting evidence items
    support_count: int = 0  # How many supporting evidence items
    first_contradiction_date: str = None
    last_contradiction_date: str = None
    persistence_score: float = 0.0  # 0-1: How persistent is the contradiction
    unresolved: bool = True  # Has management/data addressed the issue?
    contradiction_severity: str = "low"  # "low" | "medium" | "high"



@dataclass
class AssumptionHealthState:
    """Current health of an assumption (4.6)"""
    node_id: str
    state: str  # "Intact" | "Under Stress" | "Impaired" | "Broken"
    previous_state: str = "Intact"
    state_changed_at: str = None
    justification: str = ""  # Why this state
    evidence_summary: str = ""  # Brief summary of evidence
    contradiction_score: float = 0.0  # 0-1: overall contradiction level
    
    def __post_init__(self):
        if not self.state_changed_at:
            self.state_changed_at = datetime.now().isoformat()


# --- IHLE ---

@dataclass
class AssumptionDecayRate:
    """Decay rate for individual assumption"""
    node_id: str
    assumption_text: str
    decay_rate: float  # 0-1, higher = faster decay
    decay_slope: str  # "Stable" | "Linear" | "Accelerating" | "Decelerating"
    contradiction_frequency: int  # Count of contradictions
    contradiction_severity: str  # "low" | "medium" | "high"
    structural_vs_cyclical: str  # "Structural" | "Cyclical" | "Mixed"
    management_acknowledged: bool
    days_since_first_contradiction: int
    evidence_persistence: float  # From DEH


@dataclass
class PathDecayScore:
    """Decay aggregated along causal path"""
    path_id: str
    path_description: str  # Human-readable path
    node_ids: List[str]  # Nodes in path
    path_decay_score: float  # 0-1, weighted by importance
    bottleneck_node: str  # Most critical node in path
    redundancy_factor: float  # 0-1, lower = more fragile
    estimated_path_half_life_months: float


@dataclass
class RegimeSensitivity:
    """Macro/industry regime adjustments"""
    regime_tags: List[str]  # From NDG (e.g., "Macro", "Industry Cycle")
    current_regime_state: str  # "Stable" | "Transitioning" | "Unstable"
    regime_alignment: float  # 0-1, thesis compatibility with regime
    adjustment_factor: float  # Multiplier for half-life (0.5-2.0)
    adjustment_reasoning: str


@dataclass
class MonitoringCadence:
    """Recommended review frequency"""
    half_life_months: float
    recommended_frequency: str  # "Continuous" | "Monthly" | "Quarterly" | "Semi-Annual"
    next_review_date: str  # ISO format
    priority_level: str  # "Critical" | "High" | "Medium" | "Low"
    review_justification: str


@dataclass
class HalfLifeEstimate:
    """Final half-life calculation"""
    estimated_half_life_months: float
    confidence_band_low: float  # Lower bound in months
    confidence_band_high: float  # Upper bound in months
    confidence_width: float  # Band width as % of estimate
    primary_decay_drivers: List[str]  # Top 3 assumptions driving decay
    decay_trend: str  # "Stable" | "Gradual" | "Accelerating"
    time_to_first_broken: float  # Months until first assumption breaks (projected)
    regime_adjusted: bool


@dataclass
class IHLEOutput:
    """Complete Idea Half-Life Estimator output"""
    stock_ticker: str
    ndg_version: str
    deh_timestamp: str  # When DEH data was captured
    
    # Core estimates
    half_life_estimate: HalfLifeEstimate
    decay_rates: List[AssumptionDecayRate]
    path_scores: List[PathDecayScore]
    
    # Adjustments
    regime_sensitivity: RegimeSensitivity
    
    # Recommendations
    monitoring_cadence: MonitoringCadence
    
    # Summary metrics
    total_assumptions: int
    assumptions_decaying: int
    fastest_decay_rate: float
    slowest_decay_rate: float
    
    # Feedback signals
    cre_scenario_weights_update: Dict[str, float]  # Updated scenario probabilities
    red_team_relevance_boost: List[str]  # Node IDs with increased challenge priority
    
    # Additional metadata
    summary_text: Optional[str] = None  # Generated summary of durability analysis
# --- CCE ---

@dataclass
class ConvictionRecord:
    """Pre-decision conviction capture"""
    analyst_id: str
    stock_ticker: str
    thesis_id: str
    timestamp: str  # ISO format
    stated_conviction: float  # 0-1 scale
    conviction_percentile: int  # 0-100, where analyst ranks this vs own history
    qualitative_confidence: str  # "Very High" | "High" | "Moderate" | "Low"
    primary_conviction_drivers: List[str]  # Node IDs from NDG


@dataclass
class ConvictionDecomposition:
    """Confidence allocation across NDG structure"""
    node_id: str
    assumption_text: str
    allocated_confidence: float  # Portion of total conviction
    structural_importance: float  # From NDG fragility
    analyst_emphasis: float  # Analyst's stated emphasis
    evidence_strength: float  # From NDG
    confidence_concentration: float  # % of total conviction in this node


@dataclass
class OutcomeAttribution:
    """Post-fact outcome mapping to assumptions"""
    node_id: str
    assumption_text: str
    original_conviction: float
    actual_outcome: str  # "Validated" | "Invalidated" | "Indeterminate"
    outcome_driver: str  # "Assumption Correct" | "Assumption Wrong" | "External Event" | "Luck"
    contribution_to_result: float  # -1 to 1
    reasoning: str


@dataclass
class CalibrationMetrics:
    """Analyst calibration scoring"""
    analyst_id: str
    time_period: str  # e.g., "Last 12 months"
    sample_size: int  # Number of theses evaluated
    
    # Core calibration metrics
    overconfidence_score: float  # 0-1, how often stated conviction > reality
    underconfidence_score: float  # 0-1, how often stated conviction < reality
    brier_score: float  # Calibration error (lower = better)
    
    # Breakdown by confidence level
    high_confidence_accuracy: float  # Success rate when conviction > 0.7
    medium_confidence_accuracy: float  # Success rate when conviction 0.4-0.7
    low_confidence_accuracy: float  # Success rate when conviction < 0.4
    
    # Trend
    recent_improvement: bool  # Whether calibration improving
    trend_direction: str  # "Improving" | "Stable" | "Degrading"


@dataclass
class ContextualCalibration:
    """Domain-aware calibration adjustment"""
    context_type: str  # "Sector" | "Business Model" | "Regime" | "Thesis Type"
    context_value: str  # e.g., "SaaS", "Cyclical", "Growth"
    sample_size: int
    
    calibration_adjustment: float  # Multiplier (0.5-1.5)
    context_accuracy: float  # Historical accuracy in this context
    context_overconfidence: float  # Tendency in this context
    
    reasoning: str


@dataclass
class ConvictionAdjustment:
    """Forward-looking conviction scaling"""
    original_conviction: float
    
    # Adjustment factors
    analyst_calibration_factor: float  # From CalibrationMetrics
    thesis_fragility_factor: float  # From CRE + NDG
    half_life_factor: float  # From IHLE
    context_factor: float  # From ContextualCalibration
    
    # Combined adjustment
    combined_adjustment_factor: float  # Product of all factors
    calibrated_conviction: float  # original * combined_adjustment
    
    # Position sizing guidance
    suggested_position_size_factor: float  # 0.5-2.0x of baseline
    
    adjustment_reasoning: str


@dataclass
class AnalystDevelopmentSignal:
    """Long-term learning feedback"""
    analyst_id: str
    time_period: str
    
    # Performance patterns
    persistent_overconfidence_areas: Optional[List[str]] = None  # Contexts where consistently overconfident
    persistent_underconfidence_areas: Optional[List[str]] = None  # Contexts where consistently underconfident
    
    # Improvement tracking
    calibration_trajectory: str = "Stable"  # "Improving" | "Stable" | "Declining"
    strongest_domains: Optional[List[str]] = None  # Contexts with best calibration
    weakest_domains: Optional[List[str]] = None  # Contexts with worst calibration
    
    # Actionable insights
    development_priorities: Optional[List[str]] = None
    recent_blind_spots: Optional[List[str]] = None
    
    # Privacy preserved
    peer_comparison_percentile: int = 50  # Optional, 0-100
    
    def __post_init__(self):
        if self.persistent_overconfidence_areas is None:
            self.persistent_overconfidence_areas = []
        if self.persistent_underconfidence_areas is None:
            self.persistent_underconfidence_areas = []
        if self.strongest_domains is None:
            self.strongest_domains = []
        if self.weakest_domains is None:
            self.weakest_domains = []
        if self.development_priorities is None:
            self.development_priorities = []
        if self.recent_blind_spots is None:
            self.recent_blind_spots = []

@dataclass
class CCEOutput:
    """Complete Conviction Calibration Engine output"""
    stock_ticker: str
    analyst_id: str
    timestamp: str
    
    # Pre-decision state
    conviction_record: ConvictionRecord
    conviction_decomposition: List[ConvictionDecomposition]
    
    # Historical performance (if available)
    calibration_metrics: CalibrationMetrics
    contextual_calibrations: List[ContextualCalibration]
    
    # Forward adjustment
    conviction_adjustment: ConvictionAdjustment
    
    # Development feedback
    development_signal: AnalystDevelopmentSignal
    
    # Summary metrics
    original_conviction: float
    calibrated_conviction: float
    adjustment_magnitude: float  # Absolute change
    position_size_recommendation: str  # "Reduce" | "Maintain" | "Increase"
    
    # Integration with other modules
    cre_survival_rate: float  # From CRE
    ndg_fragility: float  # From NDG
    deh_contradiction_rate: float  # From DEH
    ihle_half_life_months: float  # From IHLE


@dataclass
class FinalThesisReport:
    """Final output for PM / Investment Committee"""
    stock: str
    submission_date: str
    narrative: str
    survival_rate: float
    fragility_score: float
    dominant_failure_modes: List[str]
    calibrated_confidence: float
    suggested_position_size_factor: float
    half_life_months: float
    key_risks: List[str]
    actionable_notes: List[str]
    detailed_components: Dict
    quantitative_context: Optional[ThesisQuantitativeContext] = None  # Stage 1 metrics
