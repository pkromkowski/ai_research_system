import random
import logging
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

from model.thesis_agents.llm_helper import LLMHelperMixin
from model.prompts.thesis_validation_prompts import IHLE_ADJUST_REGIME_PROMPT
from model.core.types import (
    CT_NODE_OUTCOME, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH,
    CONTROL_COMPANY, CONTROL_INDUSTRY, CONTROL_MACRO, CONTROL_EXOGENOUS,
    NATURE_STRUCTURAL, NATURE_CYCLICAL, NATURE_EXECUTION,
    NDGOutput, RedTeamOutput, AssumptionDecayRate, PathDecayScore, RegimeSensitivity,
    MonitoringCadence, HalfLifeEstimate, IHLEOutput,
)

logger = logging.getLogger(__name__)


class IdeaHalfLifeEstimator(LLMHelperMixin):
    """
    Estimates thesis half-life based on assumption decay rates.

    Converts node-level contradictions and metadata into per-assumption decay rates,
    aggregates them along causal paths, and produces a time-based half-life estimate
    with uncertainty bands. Supports optional Monte Carlo and sensitivity analysis.
    """
    # ============================================================================
    # POLICY PARAMETER: Strategy-Dependent Baseline (Cannot Be Derived)
    # ============================================================================
    # Baseline half-life is a STRATEGY-SCOPED normalization choice.
    # - 24 months = typical long-only equity horizon
    # - Tactical strategies may use 3-6 months
    # - Macro strategies may use 12-18 months
    # 
    # This is NOT an empirical claim - it's a strategic time horizon choice.
    # Cannot be derived without knowing investor strategy/mandate.
    # ============================================================================
    BASELINE_HALF_LIFE_MONTHS: float = 24.0  # Policy: Long-only equity baseline

    # Emit pipeline signals (CRE/RedTeam annotations) - disabled by default
    EMIT_PIPELINE_SIGNALS: bool = False

    # Severity persistence mapping - treat as a first-class calibration surface
    SEVERITY_PERSISTENCE_MAP: Dict[str, float] = {
        'low': 0.25,
        'medium': 0.55,
        'high': 0.80
    }

    # State name constants
    STATE_NAME_INTACT: str = "Intact"
    STATE_NAME_STRESS: str = "Under Stress"

    # Cadence priority labels
    CADENCE_PRIORITY_CRITICAL: str = "Critical"
    CADENCE_PRIORITY_HIGH: str = "High"
    CADENCE_PRIORITY_MEDIUM: str = "Medium"
    CADENCE_PRIORITY_LOW: str = "Low"

    # Per-step LLM configuration
    MAX_TOKENS_REGIME: int = 400
    LLM_TEMPERATURE: Optional[float] = None

    # Toggle for using smooth (sigmoid) transforms instead of hard thresholds
    USE_SMOOTH_TRANSFORMS: bool = True

    # Monte Carlo / Sensitivity defaults (class-level tunables)
    MONTE_CARLO_SAMPLES: int = 100
    # Sensible defaults to sample high-leverage IHLE constants (enables MC by default)
    MONTE_CARLO_PARAM_RANGES: Optional[Dict[str, Tuple[float, float]]] = {
        # unresolved contradiction boost: modest range
        'UNRESOLVED_CONTRADICTION_BOOST': (1.0, 1.5),
        # nature modifiers
        'DECAY_MODIFIER_STRUCTURAL': (0.6, 0.9),
        'DECAY_MODIFIER_CYCLICAL': (1.1, 1.5),
    }
    MONTE_CARLO_SKIP_REGIME: bool = True
    SENSITIVITY_PARAM_RANGES: Optional[Dict[str, Tuple[float, float]]] = None

    # Optional RNG seed for reproducible Monte Carlo draws (None => non-deterministic)
    RNG_SEED: Optional[int] = None

    # Decay slope labels (configurable)
    SLOPE_LABEL_STABLE: str = "Stable"
    SLOPE_LABEL_LINEAR: str = "Linear"
    SLOPE_LABEL_ACCELERATING: str = "Accelerating"
    SLOPE_LABEL_DECELERATING: str = "Decelerating"

    # Trend classification thresholds
    ACCELERATING_SHARE_THRESHOLD: float = 0.5

    # Regime state fallback
    REGIME_STATE_UNKNOWN: str = "Unknown"
    REGIME_STATE_NA: str = "N/A"
    
    # Decay rate thresholds
    DECAY_ACTIVELY_DECAYING_THRESHOLD: float = 0.10  # > 10% = actively decaying
    DECAY_RAPID_THRESHOLD: float = 0.50              # > 50% = rapid decay
    DECAY_BREAK_THRESHOLD: float = 0.80              # > 80% = assumption considered broken
    
    # Assumption nature decay modifiers
    DECAY_MODIFIER_STRUCTURAL: float = 0.80      # Structural decays slower
    DECAY_MODIFIER_CYCLICAL: float = 1.30        # Cyclical decays faster
    DECAY_MODIFIER_EXECUTION: float = 1.00       # Baseline
    DECAY_MODIFIER_MIXED: float = 1.00
    
    # Assumption control decay modifiers
    CONTROL_MODIFIER_COMPANY: float = 0.90       # Company-controlled slightly slower
    CONTROL_MODIFIER_INDUSTRY: float = 1.10      # Industry more volatile
    CONTROL_MODIFIER_MACRO: float = 1.20         # Macro most volatile
    CONTROL_MODIFIER_EXOGENOUS: float = 1.30     # External factors fastest decay
    
    # Severity multipliers for decay calculation
    SEVERITY_MULTIPLIER_LOW: float = 0.70
    SEVERITY_MULTIPLIER_MEDIUM: float = 1.00
    SEVERITY_MULTIPLIER_HIGH: float = 1.40
    
    # Severity label constants (match types.py)
    SEVERITY_LABEL_LOW: str = "LOW"
    SEVERITY_LABEL_MEDIUM: str = "MEDIUM"
    SEVERITY_LABEL_HIGH: str = "HIGH"
    
    # Time-based decay acceleration thresholds (days)
    TIME_ACCELERATION_THRESHOLD_HIGH: int = 90   # > 90 days = 1.3x acceleration
    TIME_ACCELERATION_THRESHOLD_MEDIUM: int = 30  # > 30 days = 1.15x acceleration
    TIME_ACCELERATION_MULTIPLIER_HIGH: float = 1.30
    TIME_ACCELERATION_MULTIPLIER_MEDIUM: float = 1.15
    
    # Unresolved contradiction boost
    UNRESOLVED_CONTRADICTION_BOOST: float = 1.20
    
    # Management acknowledgment discount
    MANAGEMENT_ACKNOWLEDGED_DISCOUNT: float = 0.85
    
    # Confidence band calculations
    MIN_ASSUMPTIONS_FOR_NARROW_BAND: int = 3
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 2: OPERATIONAL POLICY - Monitoring Cadence
    # ═══════════════════════════════════════════════════════════════════════════
    # These are operational constraints, not analytical parameters.
    # They should live in a monitoring policy layer, not estimator core.
    # Acceptable to keep for v1, but recognize as org-specific choices.
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Cadence thresholds (months) - POLICY: operational constraints
    CADENCE_CONTINUOUS_THRESHOLD: float = 6.0    # < 6 months = continuous
    CADENCE_MONTHLY_THRESHOLD: float = 12.0      # 6-12 months = monthly
    CADENCE_QUARTERLY_THRESHOLD: float = 18.0    # 12-18 months = quarterly
    # > 18 months = semi-annual
    
    # Cadence review intervals (days)
    REVIEW_INTERVAL_CONTINUOUS: int = 7
    REVIEW_INTERVAL_MONTHLY: int = 30
    REVIEW_INTERVAL_QUARTERLY: int = 90
    REVIEW_INTERVAL_SEMI_ANNUAL: int = 180
    
    # Decay slope classification thresholds
    SLOPE_ACCELERATING_CONTRADICTIONS_HIGH: int = 3
    SLOPE_ACCELERATING_DAYS_HIGH: int = 60
    SLOPE_ACCELERATING_CONTRADICTIONS_LOW: int = 2
    SLOPE_ACCELERATING_DAYS_LOW: int = 30
    SLOPE_DECELERATING_CONTRADICTIONS: int = 1
    SLOPE_DECELERATING_DAYS: int = 90
    
    def __init__(self, stock_ticker: str):
        """
        Initialize IHLE engine.

        Args:
            stock_ticker: Stock ticker symbol (required)
            client: Optional injected Anthropic client (preferred for testing)
            api_key: Optional API key used for lazy client initialization
        """
        if not stock_ticker:
            raise ValueError("stock_ticker is required - IHLE needs company context")
        self.stock_ticker = stock_ticker

    def ingest_thesis_state(
        self,
        ndg: NDGOutput,
        red_team: Optional[RedTeamOutput] = None
    ) -> Dict[str, Dict]:
        """
        Aggregate per-node health signals used for decay estimation.

        Args:
            ndg: NDG structure with node/edge definitions
            red_team: Optional Red Team output with node-level challenges

        Returns:
            Mapping node_id -> state dictionary with contradiction and persistence metrics
        """
        challenge_map: Dict[str, List] = {}
        if red_team:
            for c in red_team.challenges:
                challenge_map.setdefault(c.node_id, []).append(c)
        
        thesis_state: Dict[str, Dict] = {}
        for node in ndg.nodes:
            if node.node_type == CT_NODE_OUTCOME:
                continue

            contradiction_count = 0
            support_count = 0
            contradiction_severity = self.SEVERITY_LABEL_LOW
            unresolved = False
            persistence_score = 0.0
            contradiction_score = 0.0

            challenges = challenge_map.get(node.id, [])
            if challenges:
                contradiction_count = len(challenges)
                highest = max(challenges, key=lambda x: {SEVERITY_LOW: 1, SEVERITY_MEDIUM: 2, SEVERITY_HIGH: 3}[x.severity])
                contradiction_severity = highest.severity
                unresolved = True
                persistence_score = min(1.0, self.SEVERITY_PERSISTENCE_MAP.get(contradiction_severity, 0.3) * contradiction_count)
                contradiction_score = persistence_score

            # Calculate days since first contradiction from timestamps
            days_since_first = 0
            if hasattr(node, 'first_seen_date') and node.first_seen_date:
                try:
                    first_seen = node.first_seen_date if isinstance(node.first_seen_date, datetime) else datetime.fromisoformat(str(node.first_seen_date))
                    days_since_first = (datetime.now() - first_seen).days
                except (ValueError, AttributeError):
                    days_since_first = 0
            elif challenges:
                # Fallback to earliest challenge timestamp if available
                earliest_challenge = None
                for c in challenges:
                    if hasattr(c, 'timestamp') and c.timestamp:
                        try:
                            challenge_time = c.timestamp if isinstance(c.timestamp, datetime) else datetime.fromisoformat(str(c.timestamp))
                            if earliest_challenge is None or challenge_time < earliest_challenge:
                                earliest_challenge = challenge_time
                        except (ValueError, AttributeError):
                            continue
                if earliest_challenge:
                    days_since_first = (datetime.now() - earliest_challenge).days

            thesis_state[node.id] = {
                "node": node,
                "health_state": (self.STATE_NAME_STRESS if persistence_score >= self.SEVERITY_MULTIPLIER_MEDIUM else self.STATE_NAME_INTACT),
                "contradiction_score": contradiction_score,
                "persistence_score": persistence_score,
                "contradiction_count": contradiction_count,
                "support_count": support_count,
                "contradiction_severity": contradiction_severity,
                "unresolved": unresolved,
                "days_since_first": days_since_first
            }
        return thesis_state

    def _get_severity_multiplier(self, severity: str) -> float:
        """Return decay multiplier for contradiction severity level."""
        multipliers = {
            self.SEVERITY_LABEL_LOW: self.SEVERITY_MULTIPLIER_LOW,
            self.SEVERITY_LABEL_MEDIUM: self.SEVERITY_MULTIPLIER_MEDIUM,
            self.SEVERITY_LABEL_HIGH: self.SEVERITY_MULTIPLIER_HIGH
        }
        return multipliers.get(severity, self.SEVERITY_MULTIPLIER_MEDIUM)

    def _get_nature_modifier(self, nature: str) -> float:
        """Return decay modifier for assumption nature (STRUCTURAL/CYCLICAL/etc)."""
        modifiers = {
            NATURE_STRUCTURAL: self.DECAY_MODIFIER_STRUCTURAL,
            NATURE_CYCLICAL: self.DECAY_MODIFIER_CYCLICAL,
            NATURE_EXECUTION: self.DECAY_MODIFIER_EXECUTION,
            "Mixed": self.DECAY_MODIFIER_MIXED
        }
        return modifiers.get(nature, self.DECAY_MODIFIER_MIXED)

    def _get_control_modifier(self, control: str) -> float:
        """Return decay modifier for assumption control type (COMPANY/INDUSTRY/MACRO/etc)."""
        modifiers = {
            CONTROL_COMPANY: self.CONTROL_MODIFIER_COMPANY,
            CONTROL_INDUSTRY: self.CONTROL_MODIFIER_INDUSTRY,
            CONTROL_MACRO: self.CONTROL_MODIFIER_MACRO,
            CONTROL_EXOGENOUS: self.CONTROL_MODIFIER_EXOGENOUS
        }
        return modifiers.get(control, 1.0)  # Default to neutral for mixed/unknown control

    def _classify_decay_slope(self, contradiction_count: int, decay_rate: float) -> str:
        """
        Classify decay trajectory as Stable/Linear/Accelerating/Decelerating.

        Args:
            contradiction_count: Number of contradictions for this assumption
            decay_rate: Calculated decay rate (0-1)

        Returns:
            Slope classification label
        """
        if decay_rate < self.DECAY_ACTIVELY_DECAYING_THRESHOLD:
            return self.SLOPE_LABEL_STABLE

        if contradiction_count >= self.SLOPE_ACCELERATING_CONTRADICTIONS_HIGH:
            return self.SLOPE_LABEL_ACCELERATING

        if contradiction_count <= self.SLOPE_DECELERATING_CONTRADICTIONS:
            return self.SLOPE_LABEL_DECELERATING

        return self.SLOPE_LABEL_LINEAR
    
    def estimate_decay_rates(self, thesis_state: Dict[str, Dict], ndg: NDGOutput) -> List[AssumptionDecayRate]:
        """
        Estimate decay rate (0-1) for each tracked assumption.

        Args:
            thesis_state: Per-node aggregated state from ingest_thesis_state
            ndg: NDG structure for classification metadata

        Returns:
            List of AssumptionDecayRate records
        """
        decay_rates = []
        for node_id, state in thesis_state.items():
            node = state['node']
            total_signals = state['contradiction_count'] + state['support_count']

            if total_signals == 0:
                base_decay = 0.0
            else:
                contradiction_ratio = state['contradiction_count'] / total_signals
                severity_multiplier = self._get_severity_multiplier(state['contradiction_severity'])
                base_decay = contradiction_ratio * severity_multiplier
                if state['unresolved']:
                    base_decay *= self.UNRESOLVED_CONTRADICTION_BOOST

            nature_classification = getattr(node, 'nature', 'Mixed')
            nature_modifier = self._get_nature_modifier(nature_classification)
            control_type = getattr(node, 'control', 'Mixed')
            control_modifier = self._get_control_modifier(control_type)
            combined_modifier = (nature_modifier + control_modifier) / 2
            adjusted_decay = min(1.0, base_decay * combined_modifier)

            decay_slope = self._classify_decay_slope(state['contradiction_count'], adjusted_decay)
            management_acknowledged = getattr(node, 'management_acknowledged', False)
            if management_acknowledged:
                adjusted_decay *= self.MANAGEMENT_ACKNOWLEDGED_DISCOUNT

            decay_rates.append(AssumptionDecayRate(
                node_id=node_id,
                assumption_text=node.claim,
                decay_rate=adjusted_decay,
                decay_slope=decay_slope,
                contradiction_frequency=state['contradiction_count'],
                contradiction_severity=state['contradiction_severity'],
                structural_vs_cyclical=nature_classification,
                management_acknowledged=management_acknowledged,
                days_since_first_contradiction=state['days_since_first'],
                evidence_persistence=state['persistence_score']
            ))
        return decay_rates

    def _extract_causal_paths(self, ndg: NDGOutput) -> List[Dict]:
        """
        Extract all causal paths from drivers to outcomes.

        Returns:
            List of dicts with 'nodes' key containing node_id list
        """
        outcome_nodes = [n for n in ndg.nodes if n.node_type == CT_NODE_OUTCOME]
        if not outcome_nodes:
            return [{'nodes': [n.id for n in ndg.nodes if n.node_type != CT_NODE_OUTCOME]}]

        paths = []
        for outcome in outcome_nodes:
            path_nodes = []
            visited = set()

            def trace_back(node_id: str):
                if node_id in visited:
                    return  # Prevent infinite recursion on cycles
                visited.add(node_id)
                if node_id not in path_nodes:
                    path_nodes.append(node_id)
                parents = [e.source_id for e in ndg.edges if e.target_id == node_id]
                for parent in parents:
                    trace_back(parent)

            trace_back(outcome.id)
            if len(path_nodes) > 1:
                paths.append({'nodes': path_nodes})

        return paths if paths else [{'nodes': [n.id for n in ndg.nodes]}]

    def _describe_path(self, node_ids: List[str], ndg: NDGOutput) -> str:
        """Generate human-readable path description from node claims."""
        node_map = {n.id: n for n in ndg.nodes}
        claims = [
            node_map[nid].claim[:40] 
            for nid in node_ids 
            if nid in node_map
        ]
        return " → ".join(claims[:3])  # First 3 nodes
    
    def aggregate_along_paths(self, decay_rates: List[AssumptionDecayRate], ndg: NDGOutput) -> List[PathDecayScore]:
        """
        Aggregate decay along causal paths using bottleneck-weighted method.

        Args:
            decay_rates: Per-assumption decay rates
            ndg: NDG causal structure

        Returns:
            List of PathDecayScore records
        """
        decay_map = {d.node_id: d for d in decay_rates}
        paths = self._extract_causal_paths(ndg)
        path_scores = []

        for path_id, path_data in enumerate(paths):
            node_ids = path_data['nodes']
            path_decays = [decay_map[nid].decay_rate for nid in node_ids if nid in decay_map]
            if not path_decays:
                continue

            max_decay = max(path_decays)
            avg_decay = sum(path_decays) / len(path_decays)
            path_decay = max_decay  # Use bottleneck rate directly

            bottleneck_idx = path_decays.index(max_decay)
            bottleneck = node_ids[bottleneck_idx] if bottleneck_idx < len(node_ids) else node_ids[0]
            redundancy = 0.9 if len(paths) > 1 else 1.0  # Modest discount for redundancy

            if path_decay > 0:
                path_half_life = self.BASELINE_HALF_LIFE_MONTHS * (1 - path_decay) * redundancy
            else:
                path_half_life = self.BASELINE_HALF_LIFE_MONTHS

            path_scores.append(PathDecayScore(
                path_id=f"path_{path_id}",
                path_description=self._describe_path(node_ids, ndg),
                node_ids=node_ids,
                path_decay_score=path_decay,
                bottleneck_node=bottleneck,
                redundancy_factor=redundancy,
                estimated_path_half_life_months=path_half_life
            ))
        
        return path_scores
    
    def calculate_half_life(self, path_scores: List[PathDecayScore], decay_rates: List[AssumptionDecayRate], ndg: NDGOutput) -> HalfLifeEstimate:
        """
        Convert path and per-node decay into half-life estimate with confidence bands.

        Args:
            path_scores: Aggregated PathDecayScore values
            decay_rates: Per-assumption decay rates
            ndg: NDG structure

        Returns:
            HalfLifeEstimate with point estimate and diagnostics
        """
        if not path_scores:
            if decay_rates:
                avg_decay = sum(d.decay_rate for d in decay_rates) / len(decay_rates)
                estimated_months = self.BASELINE_HALF_LIFE_MONTHS * (1 - avg_decay)
            else:
                estimated_months = self.BASELINE_HALF_LIFE_MONTHS
        else:
            path_half_lives = [p.estimated_path_half_life_months for p in path_scores]
            estimated_months = sum(path_half_lives) / len(path_half_lives)


        sorted_decays = sorted(decay_rates, key=lambda d: d.decay_rate, reverse=True)
        primary_drivers = [d.node_id for d in sorted_decays[:3]]

        accelerating_count = sum(1 for d in decay_rates if d.decay_slope == self.SLOPE_LABEL_ACCELERATING)
        if decay_rates and accelerating_count >= len(decay_rates) * self.ACCELERATING_SHARE_THRESHOLD:
            decay_trend = self.SLOPE_LABEL_ACCELERATING
        elif decay_rates and all(d.decay_slope == self.SLOPE_LABEL_STABLE for d in decay_rates):
            decay_trend = self.SLOPE_LABEL_STABLE
        else:
            decay_trend = "Gradual"

        time_to_first_broken = self._project_time_to_break(decay_rates, estimated_months)

        return HalfLifeEstimate(
            estimated_half_life_months=estimated_months,
            primary_decay_drivers=primary_drivers,
            decay_trend=decay_trend,
            time_to_first_broken=time_to_first_broken,
            regime_adjusted=False
        )

    def adjust_for_regime(self, half_life: HalfLifeEstimate, ndg: NDGOutput, current_macro_context: Optional[str] = None) -> Tuple[HalfLifeEstimate, RegimeSensitivity]:
        """Adjust half-life for macro/industry regime exposure using structured LLM output.

        The method calls a structured LLM tool (schema-based) to obtain an
        `adjustment_factor`, `regime_state`, and `alignment`. It applies the
        returned multiplicative `adjustment_factor` directly to the half-life
        estimate (no clamping), and returns both the adjusted estimate and a
        `RegimeSensitivity` record containing the LLM metadata.

        Args:
            half_life: Unadjusted `HalfLifeEstimate`
            ndg: NDG structure with regime-related nodes
            current_macro_context: Optional macro/industry context to include in prompt

        Returns:
            (adjusted_half_life, RegimeSensitivity)
        """
        # Extract regime-dependent nodes
        regime_nodes = [
            n for n in ndg.nodes 
            if hasattr(n, 'control') and n.control in ['Macro', 'Industry']
        ]
        regime_tags = [n.control for n in regime_nodes]
        
        if not regime_tags:
            # No regime sensitivity
            return half_life, RegimeSensitivity(
                regime_tags=[],
                current_regime_state=self.REGIME_STATE_NA,
                regime_alignment=1.0,
                adjustment_factor=1.0,
                adjustment_reasoning="No macro or industry dependencies identified"
            )
        
        # Assess current regime state
        prompt = self.format_prompt(
            IHLE_ADJUST_REGIME_PROMPT,
            stock_ticker=self.stock_ticker,
            regime_tags=', '.join(set(regime_tags)),
            current_macro_context=current_macro_context or 'No specific context provided'
        )

        # Use structured tool-use call to get validated fields
        try:
            from model.prompts.thesis_validation_schemas import IHLE_ADJUST_REGIME_SCHEMA
            regime_data = self._call_llm_structured(
                prompt,
                IHLE_ADJUST_REGIME_SCHEMA,
                max_tokens=self.MAX_TOKENS_REGIME,
                temperature=self.LLM_TEMPERATURE,
            )
        except Exception as e:
            logger.warning("Structured regime assessment failed: %s; applying conservative fallback adjustment", e)
            # Conservative fallback: reduce half-life by 15% when regime risk exists but can't be assessed
            regime_data = {
                'adjustment_factor': 0.85,
                'regime_state': self.REGIME_STATE_UNKNOWN,
                'alignment': 1.0,
                'reasoning': f'LLM assessment failed - applied conservative 15% reduction due to known regime exposure: {e}'
            }

        # Extract adjustment values (no clamping per instruction)
        adjustment_factor = regime_data.get('adjustment_factor', 1.0)
        regime_state = regime_data.get('regime_state', self.REGIME_STATE_UNKNOWN)
        alignment = regime_data.get('alignment', 1.0)
        reasoning = regime_data.get('reasoning', '')

        # Apply adjustment directly (no clamping)
        adjusted_half_life = HalfLifeEstimate(
            estimated_half_life_months=half_life.estimated_half_life_months * adjustment_factor,
            # Returns estimate only
            primary_decay_drivers=half_life.primary_decay_drivers,
            decay_trend=half_life.decay_trend,
            time_to_first_broken=half_life.time_to_first_broken * adjustment_factor,
            regime_adjusted=True
        )

        regime_sensitivity = RegimeSensitivity(
            regime_tags=list(set(regime_tags)),
            current_regime_state=regime_state,
            regime_alignment=alignment,
            adjustment_factor=adjustment_factor,
            adjustment_reasoning=reasoning
        )

        return adjusted_half_life, regime_sensitivity
    
    def recommend_cadence(self, half_life: HalfLifeEstimate) -> MonitoringCadence:
        """
        Recommend review frequency based on half-life estimate.

        Args:
            half_life: Current half-life estimate

        Returns:
            MonitoringCadence recommendation
        """
        months = half_life.estimated_half_life_months

        if months < self.CADENCE_CONTINUOUS_THRESHOLD:
            frequency = "Continuous"
            priority = self.CADENCE_PRIORITY_CRITICAL
            days_to_next = self.REVIEW_INTERVAL_CONTINUOUS
        elif months < self.CADENCE_MONTHLY_THRESHOLD:
            frequency = "Monthly"
            priority = self.CADENCE_PRIORITY_HIGH
            days_to_next = self.REVIEW_INTERVAL_MONTHLY
        elif months < self.CADENCE_QUARTERLY_THRESHOLD:
            frequency = "Quarterly"
            priority = self.CADENCE_PRIORITY_MEDIUM
            days_to_next = self.REVIEW_INTERVAL_QUARTERLY
        else:
            frequency = "Semi-Annual"
            priority = self.CADENCE_PRIORITY_LOW
            days_to_next = self.REVIEW_INTERVAL_SEMI_ANNUAL

        next_review = datetime.now() + timedelta(days=days_to_next)
        next_review_str = next_review.strftime("%Y-%m-%d")

        justification = f"Half-life of {months:.1f} months suggests {frequency.lower()} monitoring. Decay trend is {half_life.decay_trend.lower()}. "
        if half_life.time_to_first_broken < months * 0.5:
            justification += "Projected assumption break within half-life window - increase monitoring."

        return MonitoringCadence(
            half_life_months=months,
            recommended_frequency=frequency,
            next_review_date=next_review_str,
            priority_level=priority,
            review_justification=justification
        )

    def _generate_cre_feedback(self, decay_rates: List[AssumptionDecayRate]) -> Dict[str, float]:
        """
        Generate updated scenario weights for CRE based on decay patterns.

        Returns:
            Mapping of node_id -> weight boost
        """
        feedback = {}
        for decay in decay_rates:
            if decay.decay_rate > self.DECAY_ACTIVELY_DECAYING_THRESHOLD:
                feedback[decay.node_id] = 1.0 + decay.decay_rate
        return feedback

    def _generate_red_team_feedback(self, decay_rates: List[AssumptionDecayRate]) -> List[str]:
        """
        Generate red team relevance boosts for rapidly decaying assumptions.

        Returns:
            List of node_ids requiring increased scrutiny
        """
        return [d.node_id for d in decay_rates if d.decay_rate > self.DECAY_RAPID_THRESHOLD]

    def _generate_pipeline_signals(self, decay_rates: List[AssumptionDecayRate], emit_pipeline_signals: Optional[bool]) -> Tuple[Dict[str, float], List[str], Optional[Dict]]:
        """
        Generate pipeline feedback signals for CRE and Red Team.

        Args:
            decay_rates: Per-assumption decay rates
            emit_pipeline_signals: Whether to emit signals (None uses class default)

        Returns:
            Tuple of (cre_updates, red_team_boosts, half_life_signals)
        """
        effective_emit = self.EMIT_PIPELINE_SIGNALS if emit_pipeline_signals is None else emit_pipeline_signals

        if effective_emit:
            cre_updates = self._generate_cre_feedback(decay_rates)
            red_team_boosts = self._generate_red_team_feedback(decay_rates)
            half_life_signals = {
                'rapidly_decaying_nodes': red_team_boosts,
                'suggested_scenario_weights': cre_updates
            }
        else:
            cre_updates = {}
            red_team_boosts = []
            half_life_signals = None

        return cre_updates, red_team_boosts, half_life_signals

    def _compute_decay_stats(self, decay_rates: List[AssumptionDecayRate]) -> Tuple[int, float, float]:
        """
        Compute decay statistics.

        Returns:
            Tuple of (decaying_count, fastest_decay, slowest_decay)
        """
        if not decay_rates:
            return 0, 0.0, 0.0
        decaying_count = sum(1 for d in decay_rates if d.decay_rate > self.DECAY_ACTIVELY_DECAYING_THRESHOLD)
        values = [d.decay_rate for d in decay_rates]
        return decaying_count, max(values), min(values)

    def _compute_node_contributions(self, decay_rates: List[AssumptionDecayRate]) -> List[dict]:
        """
        Compute per-node contribution to overall decay.

        Returns:
            List of dicts with node_id, assumption_text, decay_rate, contribution_pct, rank
        """
        contributions = []
        total = sum(d.decay_rate for d in decay_rates) if decay_rates else 0.0
        for idx, d in enumerate(sorted(decay_rates, key=lambda x: x.decay_rate, reverse=True)):
            contribution_pct = (d.decay_rate / total * 100.0) if total > 0 else 0.0
            contributions.append({
                'node_id': d.node_id,
                'assumption_text': d.assumption_text,
                'decay_rate': d.decay_rate,
                'contribution_pct': contribution_pct,
                'rank': idx + 1
            })
        return contributions

    def run_monte_carlo(self, ndg: NDGOutput, samples: int = 100, param_ranges: Optional[Dict[str, Tuple[float, float]]] = None, current_macro_context: Optional[str] = None, skip_regime: bool = True) -> dict:
        """
        Run Monte Carlo study over IHLE parameters.

        Args:
            ndg: NDG causal structure
            samples: Number of Monte Carlo draws
            param_ranges: Map of parameter name -> (low, high) sampling range
            current_macro_context: Optional macro context
            skip_regime: If True, skip LLM regime adjustment

        Returns:
            Dict with median, p05, p95, mean, stdev and raw samples
        """
        samples_list = []
        param_ranges = param_ranges or {}
        
        # Compute thesis state once - it doesn't depend on IHLE parameters
        thesis_state = self.ingest_thesis_state(ndg)
        
        decay_rates = self.estimate_decay_rates(thesis_state, ndg)
        path_scores = self.aggregate_along_paths(decay_rates, ndg)
        half_life = self.calculate_half_life(path_scores, decay_rates, ndg)
        adjusted_half_life = half_life
        if not skip_regime:
            try:
                adjusted_half_life, _ = self.adjust_for_regime(adjusted_half_life, ndg, current_macro_context)
            except Exception as e:
                logger.warning("Regime adjustment failed during MC baseline: %s", e)
        baseline_months = adjusted_half_life.estimated_half_life_months

        original_values = {k: getattr(self, k) for k in param_ranges.keys()}
        rng = random.Random(self.RNG_SEED)

        for i in range(samples):
            for k, (low, high) in param_ranges.items():
                sampled = rng.uniform(low, high)
                setattr(self, k, sampled)

            try:
                # Reuse precomputed thesis_state
                decay_rates_s = self.estimate_decay_rates(thesis_state, ndg)
                path_scores_s = self.aggregate_along_paths(decay_rates_s, ndg)
                half_life_s = self.calculate_half_life(path_scores_s, decay_rates_s, ndg)
                adjusted_s = half_life_s
                if not skip_regime:
                    try:
                        adjusted_s, _ = self.adjust_for_regime(adjusted_s, ndg, current_macro_context)
                    except Exception as e:
                        logger.warning("Regime adjustment failed during MC sample: %s", e)
                samples_list.append(adjusted_s.estimated_half_life_months)
            except Exception as e:
                logger.warning("Monte Carlo run failed for sample %d: %s", i, e)

        for k, v in original_values.items():
            setattr(self, k, v)

        if not samples_list:
            return {'median': baseline_months, 'p05': baseline_months, 'p95': baseline_months, 'samples': []}

        median = statistics.median(samples_list)
        p05 = sorted(samples_list)[max(0, int(0.05 * len(samples_list)) - 1)]
        p95 = sorted(samples_list)[min(len(samples_list) - 1, int(0.95 * len(samples_list)))]
        mean = statistics.mean(samples_list)
        stdev = statistics.pstdev(samples_list)

        return {'median': median, 'p05': p05, 'p95': p95, 'mean': mean, 'stdev': stdev, 'samples': samples_list}

    def run_sensitivity_analysis(self, ndg: NDGOutput, param_ranges: Optional[Dict[str, Tuple[float, float]]] = None, current_macro_context: Optional[str] = None, skip_regime: bool = True) -> Dict[str, Any]:
        """
        Run one-factor sensitivity analysis across parameter ranges.

        Args:
            ndg: NDG causal structure
            param_ranges: Map of parameter name -> (low, high) bounds
            current_macro_context: Optional macro context
            skip_regime: If True, skip LLM regime adjustment

        Returns:
            Dict with sensitivity_table and ranked impact
        """
        param_ranges = param_ranges or {}
        
        # Compute thesis state once - it doesn't depend on IHLE parameters
        thesis_state = self.ingest_thesis_state(ndg)
        
        decay_rates = self.estimate_decay_rates(thesis_state, ndg)
        path_scores = self.aggregate_along_paths(decay_rates, ndg)
        half_life = self.calculate_half_life(path_scores, decay_rates, ndg)
        adjusted_half_life = half_life
        if not skip_regime:
            try:
                adjusted_half_life, _ = self.adjust_for_regime(adjusted_half_life, ndg, current_macro_context)
            except Exception as e:
                logger.warning("Regime adjustment failed during sensitivity baseline: %s", e)
        baseline_months = adjusted_half_life.estimated_half_life_months

        results = []
        for param, (low, high) in param_ranges.items():
            orig = getattr(self, param)
            setattr(self, param, low)
            low_est = None
            try:
                # Reuse precomputed thesis_state
                decay_rates_l = self.estimate_decay_rates(thesis_state, ndg)
                path_scores_l = self.aggregate_along_paths(decay_rates_l, ndg)
                half_life_l = self.calculate_half_life(path_scores_l, decay_rates_l, ndg)
                adjusted_l = half_life_l
                if not skip_regime:
                    try:
                        adjusted_l, _ = self.adjust_for_regime(adjusted_l, ndg, current_macro_context)
                    except Exception as e:
                        logger.warning("Regime adjustment failed during sensitivity low: %s", e)
                low_est = adjusted_l.estimated_half_life_months
            except Exception as e:
                logger.warning("Sensitivity low evaluation failed for %s: %s", param, e)
            setattr(self, param, high)
            high_est = None
            try:
                # Reuse precomputed thesis_state
                decay_rates_h = self.estimate_decay_rates(thesis_state, ndg)
                path_scores_h = self.aggregate_along_paths(decay_rates_h, ndg)
                half_life_h = self.calculate_half_life(path_scores_h, decay_rates_h, ndg)
                adjusted_h = half_life_h
                if not skip_regime:
                    try:
                        adjusted_h, _ = self.adjust_for_regime(adjusted_h, ndg, current_macro_context)
                    except Exception as e:
                        logger.warning("Regime adjustment failed during sensitivity high: %s", e)
                high_est = adjusted_h.estimated_half_life_months
            except Exception as e:
                logger.warning("Sensitivity high evaluation failed for %s: %s", param, e)
            setattr(self, param, orig)

            results.append({
                'parameter': param,
                'baseline': baseline_months,
                'low_value': low,
                'low_estimate': low_est,
                'high_value': high,
                'high_estimate': high_est,
                'pct_change_low': ((low_est - baseline_months) / baseline_months * 100.0) if baseline_months else 0.0,
                'pct_change_high': ((high_est - baseline_months) / baseline_months * 100.0) if baseline_months else 0.0
            })

        # Rank by max absolute percent change
        ranked = sorted(results, key=lambda r: max(abs(r['pct_change_low']), abs(r['pct_change_high'])), reverse=True)
        return {'sensitivity_table': results, 'ranked': ranked}

    def _run_optional_analyses(self, ndg: NDGOutput, current_macro_context: Optional[str]) -> Tuple[Optional[dict], Optional[Dict[str, Any]]]:
        """
        Run optional Monte Carlo and sensitivity analyses if configured.

        Args:
            ndg: NDG causal structure
            current_macro_context: Optional macro state description

        Returns:
            Tuple of (monte_carlo_results, sensitivity_results)
        """
        monte_carlo_res = None
        sensitivity_res = None

        if self.MONTE_CARLO_PARAM_RANGES:
            try:
                monte_carlo_res = self.run_monte_carlo(
                    ndg,
                    samples=self.MONTE_CARLO_SAMPLES,
                    param_ranges=self.MONTE_CARLO_PARAM_RANGES,
                    current_macro_context=current_macro_context,
                    skip_regime=self.MONTE_CARLO_SKIP_REGIME
                )
            except Exception as e:
                logger.warning("Monte Carlo analysis failed: %s", e)

        if self.SENSITIVITY_PARAM_RANGES:
            try:
                sensitivity_res = self.run_sensitivity_analysis(
                    ndg,
                    param_ranges=self.SENSITIVITY_PARAM_RANGES,
                    current_macro_context=current_macro_context,
                    skip_regime=self.MONTE_CARLO_SKIP_REGIME
                )
            except Exception as e:
                logger.warning("Sensitivity analysis failed: %s", e)

        return monte_carlo_res, sensitivity_res

    def _project_time_to_break(self, decay_rates: List[AssumptionDecayRate], default_months: float) -> float:
        """
        Project months until first assumption breaks.

        Returns:
            Estimated months to first break
        """
        return default_months


    def _build_summary_text(self, half_life: HalfLifeEstimate, decay_rates: List[AssumptionDecayRate], decaying_count: int, cadence: MonitoringCadence, node_contributions: List[dict]) -> str:
        """Compact, human-readable summary used in IHLE outputs."""
        lines = [
            f"Thesis half-life: {half_life.estimated_half_life_months:.1f} months."
        ]
        if decaying_count > 0:
            lines.append(f"{decaying_count}/{len(decay_rates)} assumptions actively decaying.")
        else:
            lines.append("All assumptions stable.")
        if half_life.decay_trend == self.SLOPE_LABEL_ACCELERATING:
            lines.append("Decay is accelerating - thesis durability declining.")
        if node_contributions:
            top = node_contributions[:3]
            contribs = ", ".join([f"{c['node_id']} ({c['contribution_pct']:.0f}%)" for c in top])
            lines.append(f"Top contributors: {contribs}.")
        lines.append(f"Recommended: {cadence.recommended_frequency} review (next: {cadence.next_review_date}).")
        return " ".join(lines)

    def _generate_pipeline_signals(self, decay_rates: List[AssumptionDecayRate], emit_pipeline_signals: Optional[bool]) -> Tuple[Dict[str, float], List[str], Optional[Dict]]:
        """Generate pipeline feedback signals for CRE and Red Team.
        
        Args:
            decay_rates: Per-assumption decay rates
            emit_pipeline_signals: Whether to emit signals (None uses class default)
            
        Returns:
            Tuple of (cre_updates, red_team_boosts, half_life_signals)
        """
        effective_emit = self.EMIT_PIPELINE_SIGNALS if emit_pipeline_signals is None else emit_pipeline_signals
        
        if effective_emit:
            cre_updates = self._generate_cre_feedback(decay_rates)
            red_team_boosts = self._generate_red_team_feedback(decay_rates)
            half_life_signals = {
                'rapidly_decaying_nodes': red_team_boosts,
                'suggested_scenario_weights': cre_updates
            }
        else:
            cre_updates = {}
            red_team_boosts = []
            half_life_signals = None
            
        return cre_updates, red_team_boosts, half_life_signals

    def _run_optional_analyses(self, ndg: NDGOutput, current_macro_context: Optional[str]) -> Tuple[Optional[dict], Optional[Dict[str, Any]]]:
        """Run optional Monte Carlo and sensitivity analyses if configured.
        
        Args:
            ndg: NDG causal structure
            current_macro_context: Optional macro state description
            
        Returns:
            Tuple of (monte_carlo_results, sensitivity_results)
        """
        monte_carlo_res = None
        sensitivity_res = None

        if self.MONTE_CARLO_PARAM_RANGES:
            try:
                monte_carlo_res = self.run_monte_carlo(
                    ndg,
                    samples=self.MONTE_CARLO_SAMPLES,
                    param_ranges=self.MONTE_CARLO_PARAM_RANGES,
                    current_macro_context=current_macro_context,
                    skip_regime=self.MONTE_CARLO_SKIP_REGIME
                )
            except Exception as e:
                logger.warning("Monte Carlo analysis failed: %s", e)

        if self.SENSITIVITY_PARAM_RANGES:
            try:
                sensitivity_res = self.run_sensitivity_analysis(
                    ndg,
                    param_ranges=self.SENSITIVITY_PARAM_RANGES,
                    current_macro_context=current_macro_context,
                    skip_regime=self.MONTE_CARLO_SKIP_REGIME
                )
            except Exception as e:
                logger.warning("Sensitivity analysis failed: %s", e)
                
        return monte_carlo_res, sensitivity_res

    def run(
        self,
        ndg: NDGOutput,
        current_macro_context: Optional[str] = None,
        emit_pipeline_signals: Optional[bool] = None
    ) -> IHLEOutput:
        """
        Run the IHLE pipeline and return structured results.

        This method performs the full, auditable IHLE analysis and returns an
        `IHLEOutput` containing the point estimate, per-assumption diagnostics,
        node-level contribution breakdown, monitoring cadence recommendation,
        and optional Monte Carlo and sensitivity analysis summaries (attached
        when the corresponding class-level parameter ranges are configured).

        Args:
            ndg: NDG causal structure
            current_macro_context: Optional macro state description
            emit_pipeline_signals: If True, emit pipeline annotations for CRE/RedTeam.
                If None, uses the class-level `EMIT_PIPELINE_SIGNALS` default.

        Returns:
            `IHLEOutput` with the analysis results and optional numerical studies.
        """        
        logger.info(f"Starting IHLE analysis for {self.stock_ticker}")
        
        thesis_state = self.ingest_thesis_state(ndg)
        decay_rates = self.estimate_decay_rates(thesis_state, ndg)
        path_scores = self.aggregate_along_paths(decay_rates, ndg)
        half_life = self.calculate_half_life(path_scores, decay_rates, ndg)
        adjusted_half_life = half_life
        
        try:
            adjusted_half_life, regime_sensitivity = self.adjust_for_regime(adjusted_half_life, ndg, current_macro_context)
        except Exception as e:
            logger.warning("Regime adjustment failed: %s", e)
            regime_sensitivity = RegimeSensitivity(
                regime_tags=[],
                current_regime_state=self.REGIME_STATE_UNKNOWN,
                regime_alignment=1.0,
                adjustment_factor=1.0,
                adjustment_reasoning="regime adjustment failed"
            )
        
        cadence = self.recommend_cadence(adjusted_half_life)
        cre_updates, red_team_boosts, half_life_signals = self._generate_pipeline_signals(decay_rates, emit_pipeline_signals)
        decaying_count, fastest_decay, slowest_decay = self._compute_decay_stats(decay_rates)
        node_contributions = self._compute_node_contributions(decay_rates)
        monte_carlo_res, sensitivity_res = self._run_optional_analyses(ndg, current_macro_context)
        summary_text = self._build_summary_text(adjusted_half_life, decay_rates, decaying_count, cadence, node_contributions)
        
        logger.info(
            "IHLE complete: %.1f months; trend=%s; decaying=%d/%d",
            adjusted_half_life.estimated_half_life_months,
            adjusted_half_life.decay_trend,
            decaying_count,
            len(decay_rates),
        )

        return IHLEOutput(
            stock_ticker=self.stock_ticker,
            ndg_version=ndg.version,
            analysis_timestamp=datetime.now().isoformat(),
            half_life_estimate=adjusted_half_life,
            decay_rates=decay_rates,
            path_scores=path_scores,
            regime_sensitivity=regime_sensitivity,
            monitoring_cadence=cadence,
            total_assumptions=len(decay_rates),
            assumptions_decaying=decaying_count,
            fastest_decay_rate=fastest_decay,
            slowest_decay_rate=slowest_decay,
            cre_scenario_weights_update=cre_updates,
            red_team_relevance_boost=red_team_boosts,
            summary_text=summary_text,
            node_contributions=node_contributions,
            monte_carlo=monte_carlo_res,
            sensitivity_analysis=sensitivity_res
        )
    
