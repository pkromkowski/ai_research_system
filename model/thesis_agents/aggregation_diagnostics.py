import logging
from typing import Optional, Dict, Any, List

from model.core.types import (
    OUTCOME_TIER_BROKEN, OUTCOME_TIER_IMPAIRED, OUTCOME_TIER_SURVIVES,
    ThesisValidityOutput, RedTeamOutput, ValuationResult, FTOutput, IHLEOutput, 
    AggregationDiagnosticsOutput
)

logger = logging.getLogger(__name__)


class AggregationDiagnostics:
    """Final aggregation and diagnostics stage.

    Responsibilities:
      - Compute FT-level aggregates (survival fraction, tail loss, raw fragility proxy)
      - Consolidate IHLE signals and pipeline annotations
      - Produce a concise diagnostics output used by the orchestrator and final report
      - Return diagnostics only; this class does NOT mutate upstream objects such
        as `FTOutput` or `RedTeamOutput`.

    This class intentionally avoids making prescriptive pipeline routing decisions;
    it returns diagnostics and suggestions which the orchestrator or caller can use.

    Aggregation & Diagnostics (outline):
      Purpose
        Produce a PM- and IC-readable research summary.
      Inputs
        - Validity status
        - Fragility metrics
        - Half-life estimates
        - Scenario diagnostics
      Process
        - Normalization and ranking
        - Cross-thesis comparability scoring
        - Presentation formatting
      Outputs
        - Research diagnostic packet
        - Comparable scores across ideas
        - Clear articulation of “why this could fail”
      Feeds Into
        → Investment decision (human)
        → Disconfirming Evidence Monitoring (optional - v2)
        → Conviction & Position Management (optional - v2)
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # POLICY PARAMETERS: Comparability Scoring
    # ═══════════════════════════════════════════════════════════════════════════
    # These parameters define how thesis strength is normalized for cross-idea comparison.
    # They are strategy-dependent, horizon-dependent, and investor-specific.
    # 
    # COMPARABILITY_BASELINE_HALF_LIFE_MONTHS (default: 24.0)
    #   - Reference horizon for normalizing time-decay sensitivity
    #   - For long-only equity: 24 months = typical holding period
    #   - For event-driven: might be 6-12 months
    #   - For value investing: might be 36-48 months
    #   - POLICY CHOICE: Set based on investment strategy and typical horizon
    #   - NOT derivable from data; reflects investor time preference
    # 
    # COMPARABILITY_MISSING_HALF_LIFE_FACTOR (default: 0.5)
    #   - Penalty applied when IHLE signal is unavailable
    #   - This is a CONSERVATISM CHOICE, not a statistical inference
    #   - Missing half-life ≠ weak signal; it represents absence of decay analysis
    #   - Options:
    #     * 1.0 = neutral (ignore missing data)
    #     * 0.5 = moderate conservatism (default)
    #     * 0.0 = maximum conservatism (require IHLE for any score)
    #   - POLICY CHOICE: Set based on risk tolerance and data quality standards
    #   - NOT derivable; no empirical distribution exists for "unknown decay"
    # ═══════════════════════════════════════════════════════════════════════════
    
    COMPARABILITY_BASELINE_HALF_LIFE_MONTHS: float = 24.0  # Long-only equity horizon
    COMPARABILITY_MISSING_HALF_LIFE_FACTOR: float = 0.5  # Moderate conservatism
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRESENTATION PARAMETERS: Human-Readable Output Limits
    # ═══════════════════════════════════════════════════════════════════════════
    # These control cognitive load in IC decks and research summaries.
    # Limits are presentation choices, not analytical constraints.
    # ═══════════════════════════════════════════════════════════════════════════
    
    MAX_DISPLAYED_RED_CHALLENGES: int = 5  # Top N red team challenges in summary
    MAX_DISPLAYED_DECAY_DRIVERS: int = 3   # Top N decay drivers in summary

    def __init__(self, stock_ticker: str):
        if not stock_ticker:
            raise ValueError("stock_ticker is required")
        self.stock_ticker = stock_ticker

    def _compute_scenario_counts(self, ft_output: FTOutput) -> Dict[str, Any]:
        """Extract scenario outcome counts and lists from FT output."""
        scenario_results = getattr(ft_output, 'scenario_results', []) or []
        n = len(scenario_results)
        broken_count = sum(1 for r in scenario_results if r.outcome_tier == OUTCOME_TIER_BROKEN)
        impaired_count = sum(1 for r in scenario_results if r.outcome_tier == OUTCOME_TIER_IMPAIRED)
        survives_count = sum(1 for r in scenario_results if r.outcome_tier == OUTCOME_TIER_SURVIVES)
        impaired_scenarios = [r.scenario_name for r in scenario_results if r.outcome_tier == OUTCOME_TIER_IMPAIRED]
        broken_scenarios = [r.scenario_name for r in scenario_results if r.outcome_tier == OUTCOME_TIER_BROKEN]
        return {
            'total_scenarios': n,
            'broken_count': broken_count,
            'impaired_count': impaired_count,
            'survives_count': survives_count,
            'impaired_scenarios': impaired_scenarios,
            'broken_scenarios': broken_scenarios,
            'scenario_results': scenario_results
        }

    def _extract_ihle_signals(self, ihle_output: Optional[IHLEOutput]) -> Dict[str, Any]:
        """Extract half-life, decay drivers, and monitoring cadence from IHLE output."""
        if not ihle_output:
            return {
                'ihle_half_life_months': None,
                'primary_decay_drivers': [],
                'recommended_cadence': None
            }
        half_life_estimate = getattr(ihle_output, 'half_life_estimate', None)
        return {
            'ihle_half_life_months': getattr(half_life_estimate, 'estimated_half_life_months', None) if half_life_estimate else None,
            'primary_decay_drivers': getattr(half_life_estimate, 'primary_decay_drivers', []) if half_life_estimate else [],
            'recommended_cadence': getattr(ihle_output, 'monitoring_cadence', None)
        }

    def _extract_red_team_challenges(self, red_team_output: RedTeamOutput) -> List[str]:
        """Extract top N Red Team challenge texts for presentation."""
        challenges = getattr(red_team_output, 'challenges', []) or []
        return [c.challenge_text for c in challenges if hasattr(c, 'challenge_text')][:self.MAX_DISPLAYED_RED_CHALLENGES]

    def _build_summary_text(
        self,
        scenario_survival_fraction: Optional[float],
        tail_loss: Optional[float],
        ihle_half_life_months: Optional[float],
        primary_decay_drivers: List[str]
    ) -> str:
        """Build concise summary from key metrics."""
        summary_lines = []
        if scenario_survival_fraction is not None:
            summary_lines.append(f"Scenario survival fraction: {scenario_survival_fraction:.2f}")
        if tail_loss is not None:
            summary_lines.append(f"Tail loss: {tail_loss:.1%}")
        if ihle_half_life_months is not None:
            summary_lines.append(f"IHLE half-life: {ihle_half_life_months:.1f} months")
        if primary_decay_drivers:
            top = primary_decay_drivers[:self.MAX_DISPLAYED_DECAY_DRIVERS]
            summary_lines.append(f"Primary drivers: {', '.join(top)}")
        return "; ".join(summary_lines) if summary_lines else "No diagnostics available"

    def _build_failure_articulation(
        self,
        validity_output: Optional[ThesisValidityOutput],
        top_red_challenges: List[str],
        primary_decay_drivers: List[str]
    ) -> Optional[str]:
        """Build human-readable failure articulation from validity, Red Team, and IHLE outputs."""
        failure_reasons = []
        if validity_output and getattr(validity_output, 'dominant_failure_modes', None):
            failure_reasons.extend(validity_output.dominant_failure_modes[:self.MAX_DISPLAYED_DECAY_DRIVERS])
        if top_red_challenges:
            failure_reasons.extend(top_red_challenges[:self.MAX_DISPLAYED_DECAY_DRIVERS])
        if primary_decay_drivers:
            failure_reasons.extend(primary_decay_drivers[:self.MAX_DISPLAYED_DECAY_DRIVERS])
        failure_reasons = list(dict.fromkeys(failure_reasons))
        return "; ".join(failure_reasons) if failure_reasons else None

    def generate_comparability_score(self, validity_output: ThesisValidityOutput, ihle_output: Optional[IHLEOutput]) -> float:
        """Compute cross-thesis comparability score (0..1) from survival rate and half-life.
        
        DESIGN CHOICES (documented for transparency):
        
        1. LINEAR HALF-LIFE SCALING
           - Formula: min(1.0, half_life / baseline)
           - Linear monotonic scaling chosen for INTERPRETABILITY
           - Properties: monotonic ✓, interpretable ✓, stable ✓
           - Non-linear alternatives (log, sigmoid) would:
             * Increase cognitive load
             * Add tuning degrees of freedom without calibration data
             * Not improve correctness in absence of historical outcomes
           - Keep linear unless you have empirical portfolio validation data
        
        2. MULTIPLICATIVE AGGREGATION 
           - Formula: survival × half_life_factor
           - Semantics: Joint-necessity model (BOTH dimensions required for strength)
           - This is NOT an independence assumption
           - Alternatives:
             * Weighted average → allows time to compensate for fragility (undesirable)
             * Max/min → too coarse
             * Product → thesis strong only if it survives scenarios AND persists over time
           - This is a design choice, not a mathematical error
        
        3. CLIPPING TO [0,1]
           - Formula: max(0.0, min(1.0, ...))
           - This is CORRECT and should NOT be removed
           - Rationale: Score is defined as 0-1 comparability metric by contract
           - Values >1 would violate semantic meaning
           - If unbounded scores are needed, define a different metric
        """
        survival = getattr(validity_output, 'survival_rate', 0.0) if validity_output else 0.0
        half_life = ihle_output.half_life_estimate.estimated_half_life_months if ihle_output else None
        
        # Linear scaling of half-life (see documentation above)
        half_life_factor = min(1.0, (half_life / self.COMPARABILITY_BASELINE_HALF_LIFE_MONTHS)) if half_life is not None else self.COMPARABILITY_MISSING_HALF_LIFE_FACTOR
        
        # Multiplicative aggregation: joint-necessity model (see documentation above)
        score = survival * half_life_factor
        
        # Clipping to [0,1]: preserve score domain semantics (see documentation above)
        return max(0.0, min(1.0, score))

    def _rank_by_downside_risk(self, scenario_results: List[ValuationResult]) -> List[str]:
        """Rank scenarios by downside risk (ordinal ranking only)."""
        if not scenario_results:
            return []
        scores = {}
        for r in scenario_results:
            weight = getattr(r, 'plausibility_weight', 1.0) or 1.0
            val = getattr(r, 'valuation_change', 0.0) or 0.0
            risk_raw = max(0.0, -val) * weight
            scores[r.scenario_name] = risk_raw
        # Return ranked list of scenario names (ordinal - no normalization)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked]

    def _format_research_packet(
        self,
        summary_text: str,
        ihle_half_life_months: Optional[float],
        scenario_survival_fraction: Optional[float],
        tail_loss_percentile: Optional[float],
        primary_decay_drivers: List[str],
        top_red_challenges: List[str],
        failure_articulation: Optional[str],
        scenario_ranking: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Format research diagnostic packet for PM/IC consumption."""
        return {
            'summary': summary_text,
            'half_life_months': ihle_half_life_months,
            'survival_fraction': scenario_survival_fraction,
            'tail_loss_percentile': tail_loss_percentile,
            'primary_drivers': primary_decay_drivers,
            'top_red_challenges': top_red_challenges,
            'failure_articulation': failure_articulation,
            'scenario_ranking': scenario_ranking
        }

    def run(
        self,
        ihle_output: IHLEOutput,
        ft_output: FTOutput,
        red_team_output: RedTeamOutput,
        validity_output: ThesisValidityOutput,
    ) -> AggregationDiagnosticsOutput:
        """Compute aggregation diagnostics and return structured output."""
        logger.info("Running aggregation diagnostics for %s", self.stock_ticker)
        scenario_counts = self._compute_scenario_counts(ft_output)
        ihle_signals = self._extract_ihle_signals(ihle_output)
        top_red_challenges = self._extract_red_team_challenges(red_team_output)
        scenario_survival_fraction = getattr(validity_output, 'survival_rate', None)
        weighted_survival_rate = getattr(validity_output, 'weighted_survival_rate', None) if validity_output else None
        tail_loss_percentile = getattr(validity_output, 'tail_loss', None) if validity_output else None
        raw_fragility_proxy = getattr(validity_output, 'fragility_score', None) if validity_output else None
        summary_text = self._build_summary_text(
            scenario_survival_fraction,
            tail_loss_percentile,
            ihle_signals['ihle_half_life_months'],
            ihle_signals['primary_decay_drivers']
        )
        failure_articulation = self._build_failure_articulation(
            validity_output,
            top_red_challenges,
            ihle_signals['primary_decay_drivers']
        )
        
        try:
            comparable = self.generate_comparability_score(validity_output, ihle_output)
            comparable_scores = {'comparability_score': comparable}
        except Exception as e:
            logger.debug("Comparability score failed: %s", e, exc_info=True)
            comparable_scores = None
        
        try:
            scenario_ranking = self._rank_by_downside_risk(scenario_counts['scenario_results'])
        except Exception as e:
            logger.debug("Scenario ranking failed: %s", e, exc_info=True)
            scenario_ranking = None
        
        try:
            research_packet = self._format_research_packet(
                summary_text,
                ihle_signals['ihle_half_life_months'],
                scenario_survival_fraction,
                tail_loss_percentile,
                ihle_signals['primary_decay_drivers'],
                top_red_challenges,
                failure_articulation,
                scenario_ranking
            )
        except Exception as e:
            logger.debug("Research packet formatting failed: %s", e, exc_info=True)
            research_packet = None
        
        logger.debug("Aggregation diagnostics complete for %s: %s", self.stock_ticker, summary_text)
        
        return AggregationDiagnosticsOutput(
            stock_ticker=self.stock_ticker,
            total_scenarios=scenario_counts['total_scenarios'],
            broken_count=scenario_counts['broken_count'],
            impaired_count=scenario_counts['impaired_count'],
            survives_count=scenario_counts['survives_count'],
            scenario_survival_fraction=scenario_survival_fraction,
            weighted_survival_rate=weighted_survival_rate,
            tail_loss_percentile=tail_loss_percentile,
            raw_fragility_proxy=raw_fragility_proxy,
            impaired_scenarios=scenario_counts['impaired_scenarios'],
            broken_scenarios=scenario_counts['broken_scenarios'],
            ihle_half_life_months=ihle_signals['ihle_half_life_months'],
            primary_decay_drivers=ihle_signals['primary_decay_drivers'],
            recommended_cadence=ihle_signals['recommended_cadence'],
            top_red_challenges=top_red_challenges,
            summary_text=summary_text,
            failure_articulation=failure_articulation,
            scenario_ranking=scenario_ranking,
            comparable_scores=comparable_scores,
            research_packet=research_packet
        )
