import math
import logging
from typing import List, Dict, Any

from model.core.types import (
    THESIS_STATUS_VALID, THESIS_STATUS_FRAGILE, THESIS_STATUS_BROKEN,
    OUTCOME_TIER_BROKEN, OUTCOME_TIER_SURVIVES, OUTCOME_TIER_IMPAIRED, 
    FTOutput, RedTeamOutput, NDGOutput, ThesisValidityOutput, ValuationResult
)

logger = logging.getLogger(__name__)


class ThesisValidityEvaluator:
    """
    Rule-based evaluator synthesizing Financial Translation, Red Team, and NDG outputs
    into a thesis status: Valid, Fragile, or Broken.

    This evaluator is the authoritative integration point for thesis-level diagnostics,
    combining scenario valuation results (economic authority), epistemic challenges
    (Red Team), and structural fragility (NDG) into a single verdict.

    Design Principles:
    - Red Team severity is epistemic (relevance + evidence weakness), not economic
    - High-severity challenges can prevent Valid status but cannot soften Broken verdicts
    - Economic metrics (survival, tail loss, fragility) are the primary drivers
    - Classification rules are transparent and threshold-based for auditability

    Thresholds (configurable via class constants):
    - SURVIVAL_MAJORITY_THRESHOLD: Minimum survival rate for Valid status
    - MAX_FRAGILITY_FOR_VALID: Maximum fragility score for Valid status
    - MAX_TAIL_LOSS_FOR_VALID: Maximum acceptable tail loss for Valid status

    Output:
    - ThesisValidityOutput with status, reasons, contradictions, and required conditions
    """
    # Decision policy thresholds (not epistemic claims)
    # Survival interpretation: Fraction of plausible futures in which thesis remains intact
    
    # VALID policy: Thesis survives in majority of plausible futures (>50%)
    SURVIVAL_MAJORITY_THRESHOLD: float = 0.5
    
    # FRAGILE policy: Thesis survives in meaningful minority of futures (>30%)
    SURVIVAL_MEANINGFUL_MINORITY_THRESHOLD: float = 0.3
    
    # ============================================================================
    # POLICY PARAMETERS: Fragility Risk Tolerance (Cannot Be Derived)
    # ============================================================================
    # Fragility = probability-weighted structural failure rate (0-1)
    # These thresholds define RISK TOLERANCE for structural brittleness.
    # - 30% fragility for VALID = accepting ~1-in-3 chance of cascading failure
    # - 70% fragility for FRAGILE = elevated risk but not certain failure
    # 
    # Cannot be derived without:
    # - Historical thesis failure rates by fragility score
    # - Portfolio-level risk budget constraints
    # - Strategy-specific loss tolerance
    # 
    # V2 Calibration: Backtest against thesis outcomes to find optimal cutoffs
    # ============================================================================
    
    # Fragility policy: Low structural failure risk for VALID (<30%)
    MAX_FRAGILITY_FOR_VALID: float = 0.3
    
    # Fragility policy: Elevated but manageable structural risk for FRAGILE (<70%)
    MAX_FRAGILITY_FOR_FRAGILE: float = 0.7
    
    # ============================================================================
    # POLICY PARAMETERS: Tail Loss Tolerance (Cannot Be Derived)
    # ============================================================================
    # Tail loss = 5th percentile downside in scenario distribution
    # These thresholds define LOSS TOLERANCE for worst-case scenarios.
    # - 15% tail loss for VALID = modest downside acceptable
    # - 30% tail loss for FRAGILE = material but not catastrophic
    # 
    # Strategy-dependent:
    # - Growth portfolios: may accept -20% to -30% tail loss
    # - Value/defensive: may require -10% to -15% tail loss
    # - Tactical: may use tighter stops (-5% to -10%)
    # 
    # Cannot be derived without knowing portfolio risk budget and strategy mandate.
    # V2 Calibration: Fit to portfolio drawdown constraints and strategy type.
    # ============================================================================
    
    # Tail loss policy: Modest downside for VALID (>-15%)
    MAX_TAIL_LOSS_FOR_VALID: float = -0.15
    
    # Tail loss policy: Material but not catastrophic downside for FRAGILE (>-30%)
    MAX_TAIL_LOSS_FOR_FRAGILE: float = -0.30
    
    # Red team policy: VALID requires all high-severity challenges covered by scenarios
    # Coverage logic: high-severity challenges must be stress-tested, not counted

    def __init__(self, stock_ticker: str):
        if not stock_ticker:
            raise ValueError("stock_ticker is required")
        self.stock_ticker = stock_ticker

    def _compute_tail_loss(self, valuations: List[float], percentile: int = 5) -> float:
        """
        Compute loss at given percentile (default 5th percentile for tail risk).
        
        POLICY PARAMETER: percentile = 5 is financial convention (Value-at-Risk)
        - 5th percentile = ~1-in-20 worst case (standard VaR measure)
        - Alternative: 1st percentile (CVaR), 10th percentile (less conservative)
        - Cannot be derived without risk appetite specification
        
        ALGORITHMIC CHOICE: Uses floor() for percentile index calculation
        - Simple indexing without interpolation between values
        - Alternative: linear interpolation (numpy.percentile default)
        - Floor chosen for simplicity and determinism with small sample sizes.

        Args:
            valuations: List of valuation changes from scenario results
            percentile: Percentile to compute (0-100), default 5 for tail risk

        Returns:
            Valuation change at specified percentile. Negative values indicate losses,
            positive values indicate gains. Returns 0.0 if valuations list is empty.
        """
        if not valuations:
            return 0.0
        sorted_vals = sorted(valuations)
        k = max(0, min(len(sorted_vals) - 1, int(math.floor((percentile / 100.0) * len(sorted_vals)))))
        return sorted_vals[k]

    def _compute_fragility_score(self, results: List[ValuationResult], tail_loss: float, loss_cap: float = 1.0) -> float:
        """
        Compute fragility as probability-weighted structural failure measure.

        Principled definition: Expected probability mass in irrecoverable loss states.
        fragility = Sum P(scenario_i) * I(outcome_i == BROKEN) * severity_penalty

        Where:
        - P(scenario_i) = plausibility_weight / Sum weights (normalized probability)
        - I(outcome_i == BROKEN) = 1 if BROKEN, 0 otherwise
        - severity_penalty = min(1, |tail_loss| / loss_cap) (monotonic tail severity)

        Args:
            results: List of scenario valuation results with plausibility weights
            tail_loss: Tail loss percentile value (negative for losses)
            loss_cap: Maximum expected loss for normalization (default 1.0 = 100%)

        Returns:
            Fragility score between 0-1 where higher indicates more fragility.
            Now interpretable as probability-weighted structural failure rate.
        """
        if not results:
            return 0.0

        # Compute normalized probabilities
        total_weight = sum(r.plausibility_weight for r in results)
        if total_weight == 0:
            return 0.0

        # Probability-weighted broken rate
        broken_probability = sum(
            r.plausibility_weight / total_weight
            for r in results
            if r.outcome_tier == 'BROKEN'
        )

        # Tail severity as monotonic penalty (optional refinement)
        tail_severity = min(1.0, abs(min(0.0, tail_loss)) / loss_cap)

        # Fragility = broken probability * severity penalty
        fragility = broken_probability * tail_severity
        
        # NOTE: Fragility can theoretically exceed 1.0 if both:
        # - High broken probability (>50% scenarios fail)
        # - Severe tail loss (>100% of loss_cap)
        # We preserve this information rather than clipping to 1.0.
        # Interpretation: >1.0 = compound structural failure (cascading + extreme loss)
        return fragility

    def _aggregate_from_scenario_results(self, results: List[ValuationResult]) -> Dict[str, Any]:
        """
        Aggregate scenario valuation results into thesis-level diagnostic metrics.

        Computes survival rates (unweighted and plausibility-weighted), tail loss,
        fragility proxy, and identifies dominant failure modes.

        Args:
            results: List of ValuationResult from Financial Translation

        Returns:
            Dictionary containing:
            - scenario_survival_fraction: Fraction of scenarios that did not break (0-1)
            - weighted_survival_rate: Plausibility-weighted survival fraction (0-1)
            - tail_loss_percentile: Loss at 5th percentile (negative for losses)
            - raw_fragility_proxy: Heuristic fragility score (0-1)
            - dominant_failure_modes: Top 3 broken scenario names by loss magnitude
            - impaired_scenarios: List of scenario names with IMPAIRED outcome

        Note:
            Returns zeros and empty lists if results is empty.
        """
        diagnostics: Dict[str, Any] = {}
        if not results:
            diagnostics.update(
                scenario_survival_fraction=0.0,
                weighted_survival_rate=0.0,
                tail_loss_percentile=0.0,
                raw_fragility_proxy=0.0,
                dominant_failure_modes=[],
                impaired_scenarios=[],
            )
            return diagnostics

        survival_count = sum(1 for r in results if r.outcome_tier == OUTCOME_TIER_SURVIVES)
        impaired_count = sum(1 for r in results if r.outcome_tier == OUTCOME_TIER_IMPAIRED)
        broken_count = sum(1 for r in results if r.outcome_tier == OUTCOME_TIER_BROKEN)

        n = len(results)
        scenario_survival_fraction = (survival_count + impaired_count) / n if n else 0.0

        weighted_survival = sum(r.plausibility_weight for r in results if r.outcome_tier == OUTCOME_TIER_SURVIVES)
        total_weight = sum(r.plausibility_weight for r in results)
        weighted_survival_rate = weighted_survival / total_weight if total_weight > 0 else 0.0

        valuations = [r.valuation_change for r in results]
        tail_loss = self._compute_tail_loss(valuations)

        fragility = self._compute_fragility_score(results, tail_loss)

        failure_modes = [r.scenario_name for r in sorted([r for r in results if r.outcome_tier == OUTCOME_TIER_BROKEN], key=lambda x: x.valuation_change)[:3]]
        impaired_scenarios = [r.scenario_name for r in results if r.outcome_tier == OUTCOME_TIER_IMPAIRED]

        diagnostics.update(
            scenario_survival_fraction=scenario_survival_fraction,
            weighted_survival_rate=weighted_survival_rate,
            tail_loss_percentile=tail_loss,
            raw_fragility_proxy=fragility,
            dominant_failure_modes=failure_modes,
            impaired_scenarios=impaired_scenarios,
        )

        return diagnostics

    def _check_uncovered_high_severity_challenges(
        self,
        red_team: 'RedTeamOutput',
        scenario_results: List['ValuationResult']
    ) -> tuple[bool, List[str]]:
        """
        Check if high-severity challenges are covered by stress-test scenarios.
        
        Principled approach: Coverage is a constraint satisfaction problem, not a tally.
        A thesis cannot be VALID if there are high-severity challenges that haven't
        been stress-tested through scenarios.
        
        Coverage heuristic:
        - High-severity challenges should have corresponding BROKEN or IMPAIRED scenarios
        - If multiple high-severity challenges exist without economic confirmation,
          they are considered uncovered
        
        Args:
            red_team: Red team output with challenges
            scenario_results: List of scenario valuation results
        
        Returns:
            Tuple of (has_uncovered, uncovered_reasons)
        """
        if not red_team or not hasattr(red_team, 'challenges'):
            return False, []
        
        high_severity_challenges = [
            ch for ch in red_team.challenges
            if ch.severity == 'HIGH'
        ]
        
        if not high_severity_challenges:
            return False, []
        
        # Check if scenarios provided stress-test coverage
        has_broken_or_impaired = any(
            r.outcome_tier in ['BROKEN', 'IMPAIRED']
            for r in scenario_results
        )
        
        # If multiple high-severity challenges exist but no scenarios broke/impaired,
        # the challenges lack coverage
        if len(high_severity_challenges) > 0 and not has_broken_or_impaired:
            uncovered_reasons = [
                f"High-severity challenge on '{ch.node_id}' lacks stress-test coverage"
                for ch in high_severity_challenges[:3]
            ]
            return True, uncovered_reasons
        
        return False, []

    def _collect_failure_signals(
        self,
        survival: float,
        fragility: float,
        tail_loss: float,
        uncovered_challenges: List[str]
    ) -> tuple[List[str], bool]:
        """
        Collect strong failure signals and determine if Red Team requires CRE coverage.

        Checks economic metrics against Fragile-tier thresholds (more stringent than Valid).
        Red Team coverage is required when high-severity challenge count exceeds threshold.

        Args:
            survival: Scenario survival fraction (0-1)
            fragility: Raw fragility proxy score (0-1)
            tail_loss: Tail loss percentile (negative for losses)
            uncovered_challenges: List of uncovered challenge descriptions

        Returns:
            Tuple of:
            - List of failure reason strings (human-readable)
            - Boolean indicating if Red Team coverage is required
        """
        reasons: List[str] = []

        if survival < self.SURVIVAL_MEANINGFUL_MINORITY_THRESHOLD:
            reasons.append(f"Low survival rate ({survival:.2f})")
        if fragility >= self.MAX_FRAGILITY_FOR_FRAGILE:
            reasons.append(f"High fragility ({fragility:.2f})")
        if tail_loss <= self.MAX_TAIL_LOSS_FOR_FRAGILE:
            reasons.append(f"Severe tail loss ({tail_loss:.2f})")

        has_uncovered = len(uncovered_challenges) > 0
        if has_uncovered:
            reasons.extend(uncovered_challenges)

        return reasons, has_uncovered

    def _classify_thesis_status(
        self,
        survival: float,
        fragility: float,
        tail_loss: float,
        red_team_coverage_required: bool,
        reasons: List[str]
    ) -> tuple[str, List[str]]:
        """
        Classify thesis as Valid, Fragile, or Broken using threshold-based rules.

        Classification Logic:
        - Valid: All economic metrics pass Valid thresholds AND Red Team does not
          require coverage with economic confirmation
        - Fragile: Borderline survival, excess fragility, or excess tail risk
        - Broken: All other cases (multiple failure signals below thresholds)

        Red Team can prevent Valid status but cannot soften Broken verdicts.
        Economic metrics are the primary drivers.

        Args:
            survival: Scenario survival fraction (0-1)
            fragility: Raw fragility proxy score (0-1)
            tail_loss: Tail loss percentile (negative for losses)
            red_team_coverage_required: Whether Red Team requires CRE coverage
            reasons: Existing failure reasons from _collect_failure_signals

        Returns:
            Tuple of:
            - Status string: "VALID", "FRAGILE", or "BROKEN"
            - Updated reasons list with classification rationale
        """
        meets_valid_thresholds = (
            survival >= self.SURVIVAL_MAJORITY_THRESHOLD
            and fragility <= self.MAX_FRAGILITY_FOR_VALID
            and tail_loss >= self.MAX_TAIL_LOSS_FOR_VALID
        )

        red_team_with_economic_confirmation = (
            red_team_coverage_required
            and (survival < self.SURVIVAL_MEANINGFUL_MINORITY_THRESHOLD or tail_loss <= self.MAX_TAIL_LOSS_FOR_FRAGILE)
        )

        borderline_survival = self.SURVIVAL_MEANINGFUL_MINORITY_THRESHOLD <= survival < self.SURVIVAL_MAJORITY_THRESHOLD
        excess_fragility = fragility > self.MAX_FRAGILITY_FOR_VALID
        excess_tail_risk = tail_loss < self.MAX_TAIL_LOSS_FOR_VALID

        if meets_valid_thresholds and not red_team_with_economic_confirmation:
            status = THESIS_STATUS_VALID
            reasons.insert(0, "Meets validity thresholds")
        elif borderline_survival or excess_fragility or excess_tail_risk:
            status = THESIS_STATUS_FRAGILE
            if not reasons:
                reasons.append("Borderline thresholds or elevated fragility")
        else:
            status = THESIS_STATUS_BROKEN
            if not reasons:
                reasons.append("Multiple failure signals below thresholds")

        return status, reasons

    def _build_contradictions_list(self, ndg: NDGOutput, red_team: RedTeamOutput) -> List[str]:
        """
        Build list of key contradictions from NDG nodes and Red Team challenges.

        Extracts contradictions from two sources:
        1. NDG nodes with explicit contradicting evidence or low confidence (<0.4)
        2. Top 3 Red Team challenges by severity

        Args:
            ndg: NDG output containing thesis graph nodes
            red_team: Red Team output containing epistemic challenges

        Returns:
            List of human-readable contradiction strings. Empty list if none found.
        """
        contradictions: List[str] = []

        for node in ndg.nodes:
            if getattr(node, 'contradicting_evidence', None):
                contradictions.append(
                    f"NDG node {node.id}: {node.claim} (contradicting evidence: {len(node.contradicting_evidence)})"
                )
            elif getattr(node, 'confidence', None) is not None and node.confidence < 0.4:
                contradictions.append(f"NDG node {node.id}: low confidence ({node.confidence:.2f})")

        if getattr(red_team, 'challenges', None):
            for ch in red_team.challenges[:3]:
                contradictions.append(f"Red Team: {ch.challenge_text} (severity={ch.severity})")

        return contradictions

    def _build_required_conditions(
        self,
        status: str,
        survival: float,
        fragility: float,
        tail_loss: float,
    ) -> List[str]:
        """
        Build list of required conditions to achieve Valid status.

        For non-Valid theses, identifies which metrics fail Valid thresholds
        and provides actionable guidance.

        Args:
            status: Current thesis status ("Valid", "Fragile", or "Broken")
            survival: Scenario survival fraction (0-1)
            fragility: Raw fragility proxy score (0-1)
            tail_loss: Tail loss percentile (negative for losses)
            uncovered_challenges: List of uncovered challenge descriptions

        Returns:
            List of required condition strings. Empty for Valid theses.
        """
        required_conditions: List[str] = []

        if status != THESIS_STATUS_VALID:
            if survival < self.SURVIVAL_MAJORITY_THRESHOLD:
                required_conditions.append(f"Increase survival rate to >= {self.SURVIVAL_MAJORITY_THRESHOLD:.0%}")
            if fragility > self.MAX_FRAGILITY_FOR_VALID:
                required_conditions.append(f"Reduce fragility score to <= {self.MAX_FRAGILITY_FOR_VALID:.2f}")
            if tail_loss < self.MAX_TAIL_LOSS_FOR_VALID:
                required_conditions.append(f"Limit tail loss to no more than {abs(self.MAX_TAIL_LOSS_FOR_VALID):.0%}")
                required_conditions.append("Resolve or mitigate high-severity Red Team challenges")

        return required_conditions

    def _create_summary_text(
        self,
        status: str,
        survival: float,
        fragility: float,
        tail_loss: float,
        contradictions_count: int
    ) -> str:
        """
        Create concise summary text with key diagnostic metrics.

        Args:
            status: Thesis status ("VALID", "FRAGILE", or "BROKEN")
            survival: Scenario survival fraction (0-1)
            fragility: Raw fragility proxy score (0-1)
            tail_loss: Tail loss percentile (negative for losses)
            uncovered_challenges: List of uncovered challenge descriptions
            contradictions_count: Number of contradictions found

        Returns:
            Single-line summary string with all key metrics.
        """
        return (
            f"Status: {status}; survival={survival:.2f}; fragility={fragility:.2f}; "
            f"tail_loss={tail_loss:.2f}; contradictions={contradictions_count}"
        )

    def run(self, ft: FTOutput, red_team: RedTeamOutput, ndg: NDGOutput) -> ThesisValidityOutput:
        """
        Execute rule-based thesis validity evaluation.

        Orchestrates the complete evaluation pipeline:
        1. Aggregate scenario results into diagnostic metrics
        2. Collect failure signals and Red Team coverage requirements
        3. Classify thesis status (Valid/Fragile/Broken)
        4. Build contradictions list from NDG and Red Team
        5. Build required conditions for improvement
        6. Create summary text

        Args:
            ft: Financial Translation output with scenario valuation results
            red_team: Red Team output with epistemic challenges
            ndg: Narrative Decomposition Graph output with thesis structure

        Returns:
            ThesisValidityOutput containing status, reasons, contradictions,
            required conditions, and all diagnostic metrics.
        """
        diagnostics = self._aggregate_from_scenario_results(ft.scenario_results)
        survival = diagnostics['scenario_survival_fraction']
        weighted_survival = diagnostics['weighted_survival_rate']
        fragility = diagnostics['raw_fragility_proxy']
        tail_loss = diagnostics['tail_loss_percentile']
        impaired_scenarios = diagnostics['impaired_scenarios']
        has_uncovered, uncovered_reasons = self._check_uncovered_high_severity_challenges(red_team, ft.scenario_results)

        reasons, red_team_coverage_required = self._collect_failure_signals(survival, fragility, tail_loss, uncovered_reasons)
        status, reasons = self._classify_thesis_status(survival, fragility, tail_loss, red_team_coverage_required, reasons)
        contradictions = self._build_contradictions_list(ndg, red_team)
        required_conditions = self._build_required_conditions(status, survival, fragility, tail_loss)
        summary = self._create_summary_text(status, survival, fragility, tail_loss, len(contradictions))

        dominant_failures = ft.dominant_failure_modes if ft.dominant_failure_modes else []

        return ThesisValidityOutput(
            stock_ticker=self.stock_ticker,
            status=status,
            reasons=reasons,
            dominant_failure_modes=dominant_failures,
            required_conditions=required_conditions,
            key_contradictions=contradictions if contradictions else None,
            survival_rate=survival,
            weighted_survival_rate=weighted_survival,
            fragility_score=fragility,
            tail_loss=tail_loss,
            impaired_scenarios=impaired_scenarios,
            high_severity_challenges=len([ch for ch in (red_team.challenges if red_team and hasattr(red_team, "challenges") else []) if ch.severity == "HIGH"]),
            summary_text=summary
        )
