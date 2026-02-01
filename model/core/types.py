from datetime import datetime
from dataclasses import dataclass,field
from typing import Optional, List, Dict, Any

# Centralized node-type constants for reuse across agents
NODE_TYPE_ASSUMPTION = "ASSUMPTION"
NODE_TYPE_DRIVER = "DRIVER"
NODE_TYPE_OUTCOME = "OUTCOME"


# --- Perplexity Research
@dataclass
class ResearchResult:
    """Container for research results with citations"""
    query: str
    content: str
    citations: List[str] = field(default_factory=list)
    model_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "content": self.content,
            "citations": self.citations,
            "model_used": self.model_used,
            "timestamp": self.timestamp,
        }
    

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
    directionality: str = ""  # Optional: causal expression like 'X → Y'
    
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
    factor_weight_overrides: Optional[Dict[str, float]] = None

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


# --- FT ---
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
    factor_contributions: Optional[Dict[str, float]] = None  # factor -> contribution score (-1..1)
    metric_to_factor_mapping: Optional[Dict[str, Dict[str, float]]] = None  # metric -> {factor: coeff}

@dataclass
class CREOutput:
    """Counterfactual Research Engine output

    Notes:
    - Aggregated, thesis-level diagnostics (survival fractions, fragility proxies,
      tail loss) are computed by the Thesis Validity Evaluation stage (the
      `ThesisValidityEvaluator`) from `scenario_results`. The Financial Translation
      stage intentionally returns per-scenario `ValuationResult`s only; the
      evaluator synthesizes these into authoritative diagnostics for downstream
      reporting.
    """
    # Per-scenario results (primary output from FT)
    scenario_results: List[ValuationResult]

    # Thesis-level diagnostics are optional here and should be computed by the
    # Aggregation & Diagnostics stage. These fields are kept for backward
    # compatibility but are optional and may be None until aggregated.
    structural_survival_rate: Optional[float] = None  # DEPRECATED: prefer `scenario_survival_fraction` from aggregator
    scenario_survival_fraction: Optional[float] = None  # 0-1: fraction of evaluated scenarios where thesis avoids outright failure
    dominant_failure_modes: Optional[List[str]] = None
    tail_loss_percentile: Optional[float] = None  # Loss at configured percentile (default 5th)
    structural_fragility_score: Optional[float] = None  # DEPRECATED: prefer `raw_fragility_proxy` computed by aggregator
    raw_fragility_proxy: Optional[float] = None  # 0-1: heuristic fragility proxy

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


# --- Thesis Validity Evaluation ---
@dataclass
class ThesisValidityOutput:
    """Output of rule-based thesis validity evaluation

    Fields:
      - `key_contradictions`: concise list of NDG + Red Team contradictions relevant to the thesis
      - `required_conditions`: actions needed to improve status toward Valid

    Note: This is a deterministic, rule-based output used to drive downstream
    Half-Life Estimation and final reporting. It intentionally remains simple
    and transparent for auditability.
    """
    stock_ticker: str
    status: str  # "Valid" | "Fragile" | "Broken"
    reasons: List[str]
    dominant_failure_modes: List[str]
    required_conditions: List[str]
    key_contradictions: Optional[List[str]] = None
    survival_rate: float = 0.0
    weighted_survival_rate: float = 0.0
    fragility_score: float = 0.0
    tail_loss: float = 0.0
    impaired_scenarios: Optional[List[str]] = None
    high_severity_challenges: int = 0
    summary_text: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# --- Idea Half-Life Estimator ---
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
    analysis_timestamp: str  # When the IHLE analysis was performed
    
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
    
    # Feedback signals (pipeline annotations; generation gated via `EMIT_PIPELINE_SIGNALS`)
    cre_scenario_weights_update: Dict[str, float]  # Updated scenario probabilities (annotation only)
    red_team_relevance_boost: List[str]  # Node IDs with increased challenge priority (annotation only)
    half_life_signals: Optional[dict] = None  # Aggregated signals (e.g., rapidly decaying nodes, suggested scenario weights)

    # Additional metadata
    summary_text: Optional[str] = None  # Generated summary of durability analysis
    # Per-node contribution breakdown for auditability
    node_contributions: Optional[List[dict]] = None
    # Monte Carlo distribution summary (optional)
    monte_carlo: Optional[dict] = None
    # Sensitivity analysis summary (optional)
    sensitivity_analysis: Optional[dict] = None


# --- Final Thesis Report ---
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
