import random
import logging
from typing import List, Dict, Any, Optional

from model.thesis_agents.llm_helper import LLMHelperMixin
from model.core.types import (
    CT_NODE_OUTCOME, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH,
    NDGOutput, NDGNode, HistoricalAnalog, FailureMode, RelevanceScoring, 
    RedTeamChallenge, RedTeamOutput
)
from model.prompts.thesis_validation_prompts import (
    RTA_RETRIEVE_ANALOGS_PROMPT, RTA_MAP_FAILURE_MODE_PROMPT,
    RTA_SCORE_RELEVANCE_PROMPT, RTA_SYNTHESIZE_CHALLENGE_PROMPT
)
from model.prompts.thesis_validation_schemas import (
    RTA_RETRIEVE_ANALOGS_SCHEMA, RTA_MAP_FAILURE_MODE_SCHEMA,
    RTA_SCORE_RELEVANCE_SCHEMA, RTA_SYNTHESIZE_CHALLENGE_SCHEMA
)

logger = logging.getLogger(__name__)


class AIRedTeamWithMemory(LLMHelperMixin):
    """
    AI Red Team that surfaces plausible failure mechanisms through historical analogs.

    Uses structured LLM calls to retrieve historical precedents, map failure modes,
    score relevance, and synthesize falsifiable challenges. Employs soft scoring
    with sigmoid transforms for stable, tunable risk assessment.

    Attributes:
        stock_ticker: Company ticker symbol for context
    """    
    # Token limits per step
    MAX_TOKENS_ANALOGS: int = 2000
    MAX_TOKENS_FAILURE_MODE: int = 800
    MAX_TOKENS_RELEVANCE: int = 1000
    MAX_TOKENS_CHALLENGE: int = 600
    
    # Temperature settings
    # Higher for analog retrieval (creativity in recall), lower for classification
    TEMPERATURE_ANALOGS: float = 0.3
    TEMPERATURE_FAILURE_MODE: float = 0.0
    TEMPERATURE_RELEVANCE: float = 0.0
    TEMPERATURE_CHALLENGE: float = 0.2
    
    # Failure mode categories - guidance for LLM classification
    # These are not constraints on analyst thesis - they help structure the analysis
    # The LLM can identify failures outside these categories if appropriate
    FAILURE_MODE_CATEGORIES: List[str] = [
        "Demand Elasticity",
        "Switching Costs",
        "Margin Durability",
        "Competitive Response",
        "Management Signal"
    ]
    
    # Default pipeline parameters
    MAX_CHALLENGES: int = 5
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RISK SCORING THRESHOLDS  # POLICY GATES)
    # ═══════════════════════════════════════════════════════════════════════════════
    # These act as gates: which nodes considered, which flagged, which escalate.
    # 
    # PROBLEM: They assume score calibration that likely does NOT exist.
    #          Underlying scores are LLM outputs, not calibrated probabilities.
    #          They implicitly define workload size and alert rates.
    #
    # CURRENT: Absolute thresholds in [0,1]
    # V2 ALTERNATIVES:
    #   - Quantile-based: "High confidence = top X% of confidence scores per run"
    #   - Capacity-driven: "Choose thresholds so expected challenges ≈ N per company"
    #   - Post-hoc calibration: If historical adjudications exist, tune for precision/recall
    #
    # These should eventually be derived from operational constraints or empirical data.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    HIGH_CONFIDENCE_THRESHOLD: float = 0.2   # POLICY: What qualifies as "high confidence"
    WEAK_EVIDENCE_THRESHOLD: float = 0.5     # POLICY: What qualifies as "weak evidence"
    MIN_CHALLENGE_PRIORITY: float = 0.15     # POLICY: Minimum risk score to surface
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RISK CONTRIBUTION WEIGHTS  # ADDITIVE UTILITY MODEL)
    # ═══════════════════════════════════════════════════════════════════════════════
    # Risk score = gap + confidence + weak_evidence (linear additive model)
    # 
    # PROBLEMS:
    #   - Relative importance is implicit and undocumented
    #   - Coefficients do not clearly sum to a budget
    #   - Interactions ignored (e.g., high gap + low evidence)
    #
    # INTERPRETATION: This is a weighted HEURISTIC UTILITY function, not probability.
    #
    # V2 ALTERNATIVES:
    #   - Budgeted contribution: Each component has max share (e.g., gap ≤ 40%, conf ≤ 40%, evidence ≤ 20%)
    #   - Latent-risk model: Combine via Bayesian or logistic aggregation
    #   - Decision-driven: Learn weights from historical usefulness
    #
    # For V1: Recognize as heuristic scoring, not probabilistic evidence combination.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    RISK_CONTRIBUTION_MAX_GAP: float = 0.5                    # Max contribution from gap component
    RISK_CONTRIBUTION_CONFIDENCE_MULTIPLIER: float = 0.5      # Scale factor for confidence component
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SEVERITY DETERMINATION LOGIC  # ASYMMETRIC AND/OR)
    # ═══════════════════════════════════════════════════════════════════════════════
    # Severity classification mixes AND logic for HIGH, OR logic for MEDIUM:
    #   HIGH:   relevance >= 0.7 AND evidence <= 0.4 (both conditions required)
    #   MEDIUM: relevance >= 0.5 OR evidence <= 0.5  (either condition sufficient)
    #   LOW:    Everything else (catch-all)
    #
    # HIDDEN POLICY: "High severity = high relevance + weak evidence"
    #                "Medium severity = almost everything else" (becomes default)
    #
    # PROBLEMS:
    #   - Asymmetry is undocumented
    #   - MEDIUM becomes catch-all category
    #   - Thresholds interact in non-obvious ways
    #
    # V2 ALTERNATIVES:
    #   - Ordinal scoring: Severity = discretized continuous risk score
    #   - Decision matrix: Explicit table mapping (relevance, evidence) → severity
    #   - Loss-based: Assign severity based on expected cost of being wrong
    #
    # For V1: Keep logic but document asymmetry explicitly.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    HIGH_SEVERITY_RELEVANCE_THRESHOLD: float = 0.7   # HIGH requires: high relevance AND...
    HIGH_SEVERITY_EVIDENCE_THRESHOLD: float = 0.4    # ...weak evidence (both conditions)
    MEDIUM_SEVERITY_RELEVANCE_THRESHOLD: float = 0.5 # MEDIUM requires: moderate relevance OR...
    MEDIUM_SEVERITY_EVIDENCE_THRESHOLD: float = 0.5  # ...moderate evidence (either condition)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RELEVANCE DIMENSION WEIGHTS  # SECTOR/STRATEGY DEPENDENT)
    # ═══════════════════════════════════════════════════════════════════════════════
    # Hierarchy: Business Model > Competitive > Balance Sheet > Regulatory > Cycle
    # 
    # This is NORMATIVE and SECTOR-DEPENDENT, not universal truth:
    #   - SaaS: Business model dominates (unit economics, retention)
    #   - Financials: Balance sheet dominates (capital, liquidity)
    #   - Biotech: Regulatory dominates (FDA approval risk)
    #   - Commodities: Cycle dominates (macro sensitivity)
    #
    # IMPLICIT WORLDVIEW: Growth investor lens (business model weighted high)
    #
    # V2 ALTERNATIVES:
    #   - Sector-specific weight vectors (configurable by industry)
    #   - Learning-to-rank: Learn weights that predict material risks historically
    #   - Scenario-conditioned: Regulatory shocks → overweight regulatory
    #
    # For V1: Document as growth-investor baseline, should be configurable.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Relevance dimension weights (must sum to 1.0) - GROWTH INVESTOR BASELINE
    RELEVANCE_WEIGHT_BUSINESS_MODEL: float = 0.30  # Dominant for SaaS/tech
    RELEVANCE_WEIGHT_COMPETITIVE: float = 0.25     # Market structure
    RELEVANCE_WEIGHT_BALANCE_SHEET: float = 0.20   # Financial stability
    RELEVANCE_WEIGHT_REGULATORY: float = 0.15      # Policy risk
    RELEVANCE_WEIGHT_CYCLE: float = 0.10           # Macro sensitivity

    # Node-type constants and skip list
    NODE_TYPE_OUTCOME: str = CT_NODE_OUTCOME
    SKIP_NODE_TYPES: List[str] = [NODE_TYPE_OUTCOME]

    # ═══════════════════════════════════════════════════════════════════════════════
    # SIGMOID STEEPNESS PARAMETERS  # NEAR-BINARY BEHAVIOR)
    # ═══════════════════════════════════════════════════════════════════════════════
    # k = 10 produces NEAR-BINARY behavior (sharp transitions at thresholds)
    # 
    # This effectively turns continuous inputs into threshold detectors:
    #   - Small changes near threshold → large score changes
    #   - Large changes far from threshold → negligible score changes
    #
    # IMPLICIT POLICY: "We care about crossing boundaries, not magnitude"
    #
    # CENTERING: Gap centered at 0 assumes symmetric costs for +/- deviations
    #
    # V2 ALTERNATIVES:
    #   - Lower k (e.g., 3-5): Preserve ranking information, smoother gradients
    #   - Piecewise linear: More interpretable
    #   - Empirical tuning: Choose k to maximize score dispersion or predictive power
    #
    # For V1: Document quasi-discrete interpretation.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Sigmoid steepness parameters (k=10 → quasi-discrete, k=3 → smooth)
    SIGMOID_K_GAP: float = 10.0         # High k = near-binary gap detection
    SIGMOID_K_CONFIDENCE: float = 10.0  # High k = sharp confidence thresholds
    SIGMOID_K_EVIDENCE: float = 10.0    # High k = sharp evidence thresholds

    # ═══════════════════════════════════════════════════════════════════════════════
    # SATURATION AND DISCOUNTS  # RISK AVERSION & ASYMMETRY)
    # ═══════════════════════════════════════════════════════════════════════════════
    # These mechanisms dampen contributions and encode implicit beliefs:
    #
    # CONFIDENCE_SATURATION: Caps confidence contribution (behavioral safeguard?)
    # UPSIDE_SCENARIO_DISCOUNT: Penalizes upside scenarios (downside bias)
    # WEAK_EVIDENCE cap: Limits low-evidence contribution
    #
    # KEY QUESTION: Are these...
    #   A) Behavioral safeguards against LLM overconfidence?
    #   B) Substantive beliefs about downside vs upside asymmetry?
    #
    # IMPLICIT STANCE: Conservatism and pessimism (downside risks weighted higher)
    #
    # V2 ALTERNATIVES:
    #   - Explicit priors: "Downside risks have higher transferability than upside"
    #   - Nonlinear saturation: Logistic or exponential damping
    #   - Calibration to human review: Tune so reviewers aren't overwhelmed
    #
    # For V1: Document philosophical stance (conservative, asymmetric).
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Contribution caps and discounts - CONSERVATIVE/ASYMMETRIC POLICY
    RISK_CONTRIBUTION_WEAK_EVIDENCE: float = 0.2    # Cap for low-evidence scenarios
    LOW_CATEGORY_CONFIDENCE_THRESHOLD: float = 0.5  # Confidence damping threshold
    UPSIDE_SCENARIO_DISCOUNT: float = 0.3           # Downside bias: upside less transferable
    CONFIDENCE_SATURATION: float = 0.4              # Behavioral safeguard vs LLM overconfidence

    def __init__(self, stock_ticker: str):
        """
        Initialize Red Team engine.
        
        Args:
            stock_ticker: Ticker symbol for company-specific analysis
        
        Raises:
            ValueError: If stock_ticker is empty or None
        """
        if not stock_ticker:
            raise ValueError("stock_ticker is required")
        self.stock_ticker = stock_ticker

        # Validate relevance weights sum to 1.0 (within tolerance)
        self._validate_relevance_weights()
    
    def _validate_relevance_weights(self) -> None:
        """
        Validate relevance dimension weights sum to 1.0.
        
        Raises:
            ValueError: If weights don't sum to 1.0 within tolerance
        """
        weights_sum = sum(self._get_relevance_weights())
        if abs(weights_sum - 1.0) > 0.001:
            raise ValueError(
                f"Relevance weights must sum to 1.0, got {weights_sum:.4f}. "
                f"Check RELEVANCE_WEIGHT_* class variables."
            )

    def _sigmoid(self, x: float, threshold: float = 0.0, k: float = 1.0) -> float:
        """Sigmoid scaled to 0-1 centered at threshold with steepness k."""
        import math
        z = k * (x - threshold)
        return 1.0 / (1.0 + math.exp(-z))

    def _compute_node_risk(self, node: NDGNode, ndg: NDGOutput, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compute epistemic risk score based on confidence-evidence gaps and evidence weakness.

        Args:
            node: Node to evaluate
            ndg: Graph context (unused but kept for API consistency)
            params: Parameter overrides for sensitivity analysis

        Returns:
            Dict with challenge_priority_score, breakdown, and inputs_used
        """
        p = params or {}
        gap_k = p.get('SIGMOID_K_GAP', self.SIGMOID_K_GAP)
        conf_k = p.get('SIGMOID_K_CONFIDENCE', self.SIGMOID_K_CONFIDENCE)
        ev_k = p.get('SIGMOID_K_EVIDENCE', self.SIGMOID_K_EVIDENCE)

        max_gap_contrib = p.get('RISK_CONTRIBUTION_MAX_GAP', self.RISK_CONTRIBUTION_MAX_GAP)
        confidence_mult = p.get('RISK_CONTRIBUTION_CONFIDENCE_MULTIPLIER', self.RISK_CONTRIBUTION_CONFIDENCE_MULTIPLIER)
        weak_evidence_cap = p.get('RISK_CONTRIBUTION_WEAK_EVIDENCE', self.RISK_CONTRIBUTION_WEAK_EVIDENCE)

        high_conf_threshold = p.get('HIGH_CONFIDENCE_THRESHOLD', self.HIGH_CONFIDENCE_THRESHOLD)
        weak_evidence_threshold = p.get('WEAK_EVIDENCE_THRESHOLD', self.WEAK_EVIDENCE_THRESHOLD)

        breakdown: Dict[str, float] = {}
        inputs_used: Dict[str, Any] = {
            'confidence': node.confidence,
            'evidence_strength': node.evidence_strength,
            'node_id': node.id
        }

        gap = max(0.0, node.confidence - node.evidence_strength)
        gap_score = self._sigmoid(gap, threshold=0.0, k=gap_k)
        gap_contribution = min(max_gap_contrib, gap_score * max_gap_contrib)
        breakdown['gap_contribution'] = gap_contribution
        inputs_used['confidence_evidence_gap'] = gap

        confidence_score = self._sigmoid(node.confidence, threshold=high_conf_threshold, k=conf_k)
        saturation_multiplier = 1.0 - (confidence_score * self.CONFIDENCE_SATURATION)
        confidence_contribution = confidence_score * confidence_mult * saturation_multiplier
        breakdown['confidence_contribution'] = confidence_contribution
        breakdown['confidence_saturation_multiplier'] = saturation_multiplier
        inputs_used['confidence_score'] = confidence_score
        inputs_used['confidence_saturation_applied'] = saturation_multiplier

        evidence_weakness = 1.0 - self._sigmoid(node.evidence_strength, threshold=weak_evidence_threshold, k=ev_k)
        weak_evidence_contribution = evidence_weakness * weak_evidence_cap
        breakdown['weak_evidence_contribution'] = weak_evidence_contribution
        inputs_used['evidence_weakness'] = evidence_weakness

        challenge_priority_score = gap_contribution + confidence_contribution + weak_evidence_contribution
        breakdown['total'] = challenge_priority_score

        return {'challenge_priority_score': challenge_priority_score, 'breakdown': breakdown, 'inputs_used': inputs_used}
    
    def _get_relevance_weights(self) -> List[float]:
        """
        Get relevance dimension weights.
        
        Returns:
            List of weights: business_model, competitive, balance_sheet, regulatory, cycle
        """
        return [
            self.RELEVANCE_WEIGHT_BUSINESS_MODEL,
            self.RELEVANCE_WEIGHT_COMPETITIVE,
            self.RELEVANCE_WEIGHT_BALANCE_SHEET,
            self.RELEVANCE_WEIGHT_REGULATORY,
            self.RELEVANCE_WEIGHT_CYCLE
        ]
    
    def prioritize_assumptions(self, ndg: NDGOutput) -> List[NDGNode]:
        """
        Identify assumptions for adversarial scrutiny based on epistemic risk.
        
        Scores nodes on confidence-evidence gaps, high confidence flags, and evidence weakness.
        Applies confidence saturation to prevent structural priors from dominating rankings.
        
        Args:
            ndg: Narrative Decomposition Graph output
            
        Returns:
            Prioritized list of nodes sorted by challenge priority score
        """
        high_priority_nodes = []
        
        for node in ndg.nodes:
            if node.node_type in self.SKIP_NODE_TYPES:
                continue

            computed = self._compute_node_risk(node, ndg)
            challenge_priority_score = computed['challenge_priority_score']
            breakdown = computed['breakdown']
            inputs_used = computed['inputs_used']

            challenge_reasons = []
            if breakdown['gap_contribution'] > 0:
                challenge_reasons.append(
                    f"confidence-evidence gap ({node.confidence:.2f} conf, {node.evidence_strength:.2f} ev)"
                )
            if breakdown['confidence_contribution'] > 0.0:
                challenge_reasons.append(f"high confidence ({node.confidence:.2f})")
            if breakdown['weak_evidence_contribution'] > 0.0:
                challenge_reasons.append(f"weak evidence ({node.evidence_strength:.2f})")

            node.challenge_priority_score = challenge_priority_score
            node.challenge_reasons = challenge_reasons
            node.score_breakdown = breakdown
            node.inputs_used = inputs_used

            if challenge_priority_score >= self.MIN_CHALLENGE_PRIORITY:
                high_priority_nodes.append(node)
        
        high_priority_nodes.sort(key=lambda n: n.challenge_priority_score, reverse=True)
        
        try:
            self._last_run_stats = {
                'total_nodes_evaluated': len([n for n in ndg.nodes if n.node_type not in self.SKIP_NODE_TYPES]),
                'high_priority_nodes_identified': len(high_priority_nodes)
            }
        except Exception:
            pass
        
        return high_priority_nodes
    
    def retrieve_historical_analogs(
        self,
        node: NDGNode,
        company_context: str
    ) -> List[HistoricalAnalog]:
        """
        Retrieve historical precedents where similar assumptions failed.

        Args:
            node: Assumption node being evaluated
            company_context: Company description

        Returns:
            List of historical analogs ordered by estimated relevance
        """        
        prompt = self.format_prompt(
            RTA_RETRIEVE_ANALOGS_PROMPT,
            stock_ticker=self.stock_ticker,
            company_context=company_context,
            assumption_claim=node.claim,
            node_type=node.node_type,
            control=node.control,
            nature=node.nature,
            evidence_strength=node.evidence_strength
        )

        result = self._call_llm_structured(
            prompt,
            RTA_RETRIEVE_ANALOGS_SCHEMA,
            max_tokens=self.MAX_TOKENS_ANALOGS,
            temperature=self.TEMPERATURE_ANALOGS
        )

        analogs_data = result.get('analogs', [])
        if not isinstance(analogs_data, list):
            analogs_data = []

        analogs = [
            HistoricalAnalog(
                case_name=a['case_name'],
                assumption_type=a['assumption_type'],
                failure_mode=a['failure_mode'],
                context=a['context'],
                year=a.get('year'),
                relevance_reasoning=a.get('relevance_reasoning', '')
            )
            for a in analogs_data
        ]

        logger.debug("Retrieved %d analogs for node %s", len(analogs), node.id)
        return analogs
    
    def map_failure_mode(self, analog: HistoricalAnalog) -> FailureMode:
        """
        Map historical analog to structured failure mechanism.

        Args:
            analog: Historical case to analyze

        Returns:
            FailureMode with category, description, and early warning indicators
        """        
        prompt = self.format_prompt(
            RTA_MAP_FAILURE_MODE_PROMPT,
            case_name=analog.case_name,
            context=analog.context,
            failure_mode=analog.failure_mode,
            failure_modes_list=', '.join(self.FAILURE_MODE_CATEGORIES)
        )

        failure_data = self._call_llm_structured(
            prompt,
            RTA_MAP_FAILURE_MODE_SCHEMA,
            max_tokens=self.MAX_TOKENS_FAILURE_MODE,
            temperature=self.TEMPERATURE_FAILURE_MODE
        )

        return FailureMode(
            category=failure_data['category'],
            description=failure_data['description'],
            early_warnings=failure_data.get('early_warnings', []),
            taxonomy_match=bool(failure_data.get('taxonomy_match', True)),
            category_confidence=float(failure_data.get('category_confidence', 1.0)),
            alternative_category=failure_data.get('alternative_category'),
            is_downside_transferable=bool(failure_data.get('is_downside_transferable', True))
        )
    
    def score_relevance(
        self,
        analog: HistoricalAnalog,
        node: NDGNode,
        company_context: str
    ) -> RelevanceScoring:
        """
        Score relevance of historical analog to current assumption.

        Evaluates similarity across business model, competitive structure,
        balance sheet, regulatory environment, and cycle position.

        Args:
            analog: Historical analog being compared
            node: Current assumption node
            company_context: Company description

        Returns:
            RelevanceScoring with per-dimension scores and overall relevance
        """        
        prompt = self.format_prompt(
            RTA_SCORE_RELEVANCE_PROMPT,
            stock_ticker=self.stock_ticker,
            company_context=company_context,
            assumption_claim=node.claim,
            control=node.control,
            nature=node.nature,
            case_name=analog.case_name,
            analog_context=analog.context,
            failure_mode=analog.failure_mode
        )

        rel_data = self._call_llm_structured(
            prompt,
            RTA_SCORE_RELEVANCE_SCHEMA,
            max_tokens=self.MAX_TOKENS_RELEVANCE,
            temperature=self.TEMPERATURE_RELEVANCE
        )
        
        weights = self._get_relevance_weights()
        scores = [
            rel_data['business_model_similarity'],
            rel_data['competitive_structure'],
            rel_data['balance_sheet_flexibility'],
            rel_data['regulatory_environment'],
            rel_data['cycle_position']
        ]
        computed_overall = sum(w * s for w, s in zip(weights, scores))
        
        llm_overall = rel_data.get('overall_relevance')
        if llm_overall is not None:
            overall = llm_overall
            if abs(llm_overall - computed_overall) > 0.15:
                logger.debug(
                    f"Relevance score discrepancy for {node.id}: "
                    f"LLM={llm_overall:.2f}, computed={computed_overall:.2f}"
                )
        else:
            overall = computed_overall
        
        justification = "\n".join([
            f"Business Model: {rel_data.get('business_model_reasoning', '')}",
            f"Competition: {rel_data.get('competitive_structure_reasoning', '')}",
            f"Balance Sheet: {rel_data.get('balance_sheet_flexibility_reasoning', '')}",
            f"Regulatory: {rel_data.get('regulatory_environment_reasoning', '')}",
            f"Cycle: {rel_data.get('cycle_position_reasoning', '')}",
            f"Overall: {rel_data.get('overall_reasoning', '')}"
        ])
        
        node.relevance_info = {
            'business_model': rel_data['business_model_similarity'],
            'competitive': rel_data['competitive_structure'],
            'balance_sheet': rel_data['balance_sheet_flexibility'],
            'regulatory': rel_data['regulatory_environment'],
            'cycle': rel_data['cycle_position'],
            'overall': overall
        }

        return RelevanceScoring(
            business_model_similarity=rel_data['business_model_similarity'],
            competitive_structure=rel_data['competitive_structure'],
            balance_sheet_flexibility=rel_data['balance_sheet_flexibility'],
            regulatory_environment=rel_data['regulatory_environment'],
            cycle_position=rel_data['cycle_position'],
            overall_relevance=overall,
            justification=justification
        )

    def run_sensitivity_report(self, ndg: NDGOutput, params: Optional[Dict[str, float]] = None, n: int = 100) -> Dict[str, Any]:
        """
        Monte Carlo sensitivity analysis of scoring parameters.

        Args:
            ndg: Graph output to evaluate
            params: Fractional std devs for parameter perturbations
            n: Number of Monte Carlo samples

        Returns:
            Per-node statistics showing ranking stability under parameter perturbations
        """        
        perturb_defaults = {
            'RISK_CONTRIBUTION_MAX_GAP': 0.1,
            'RISK_CONTRIBUTION_CONFIDENCE_MULTIPLIER': 0.1,
            'RISK_CONTRIBUTION_WEAK_EVIDENCE': 0.1,
            'WEAK_EVIDENCE_THRESHOLD': 0.05,
            'HIGH_CONFIDENCE_THRESHOLD': 0.05
        }
        perturb = perturb_defaults.copy()
        if params:
            perturb.update(params)

        base_nodes = self.prioritize_assumptions(ndg)
        base_top_ids = [n.id for n in base_nodes[:self.MAX_CHALLENGES]]

        node_stats: Dict[str, Dict[str, Any]] = {n.id: {'times_in_top': 0, 'times_above_threshold': 0} for n in ndg.nodes}

        for _ in range(n):
            sampled = {}
            for name, frac in perturb.items():
                nominal = getattr(self, name)
                std = abs(nominal) * frac
                sampled_val = random.gauss(nominal, std)
                if 'THRESHOLD' in name:
                    sampled[name] = max(0.0, sampled_val)
                else:
                    sampled[name] = sampled_val

            sampled_high = []
            for node in ndg.nodes:
                computed = self._compute_node_risk(node, ndg, params=sampled)
                if computed['challenge_priority_score'] >= self.MIN_CHALLENGE_PRIORITY:
                    sampled_high.append(node)
                    node_stats[node.id]['times_above_threshold'] += 1

            sampled_top_ids = [n.id for n in sorted(sampled_high, key=lambda x: x.challenge_priority_score, reverse=True)][:self.MAX_CHALLENGES]
            for nid in sampled_top_ids:
                node_stats[nid]['times_in_top'] += 1

        report = {
            'n_samples': n,
            'base_top_ids': base_top_ids,
            'node_stats': {
                nid: {
                    'times_in_top': stats['times_in_top'],
                    'times_above_threshold': stats['times_above_threshold'],
                    'percent_in_top': stats['times_in_top']/n,
                    'percent_above_threshold': stats['times_above_threshold']/n
                } for nid, stats in node_stats.items()
            }
        }
        return report

    def _determine_severity(
        self,
        relevance_score: float,
        evidence_strength: float
    ) -> str:
        """
        Determine challenge severity from relevance and evidence strength.

        Args:
            relevance_score: Overall relevance (0-1)
            evidence_strength: Evidence strength (0-1)

        Returns:
            Severity level: HIGH, MEDIUM, or LOW
        """
        if (relevance_score >= self.HIGH_SEVERITY_RELEVANCE_THRESHOLD and 
            evidence_strength < self.HIGH_SEVERITY_EVIDENCE_THRESHOLD):
            return SEVERITY_HIGH
        
        if (relevance_score >= self.MEDIUM_SEVERITY_RELEVANCE_THRESHOLD or 
            evidence_strength < self.MEDIUM_SEVERITY_EVIDENCE_THRESHOLD):
            return SEVERITY_MEDIUM
        
        return SEVERITY_LOW
    
    def synthesize_challenge(
        self,
        node: NDGNode,
        analog: HistoricalAnalog,
        failure_mode: FailureMode,
        relevance: RelevanceScoring
    ) -> RedTeamChallenge:
        """
        Generate falsifiable challenge from historical analog and failure mode.
        
        Applies asymmetric transferability discount (0.3x) for upside scenario failures.
        Severity based on effective relevance and evidence strength (epistemic only).

        Args:
            node: Assumption node being challenged
            analog: Historical precedent
            failure_mode: Mapped failure mechanism
            relevance: Relevance scoring

        Returns:
            RedTeamChallenge with severity, monitoring indicators, and suspected_value_exposure=UNKNOWN
        """
        early_warnings_text = (
            chr(10).join(f"- {w}" for w in failure_mode.early_warnings)
            if failure_mode.early_warnings
            else "- No specific early warnings identified"
        )
        
        prompt = self.format_prompt(
            RTA_SYNTHESIZE_CHALLENGE_PROMPT,
            assumption_claim=node.claim,
            evidence_strength=node.evidence_strength,
            confidence=node.confidence,
            case_name=analog.case_name,
            analog_context=analog.context,
            failure_description=failure_mode.description,
            relevance_score=f"{relevance.overall_relevance:.2f}",
            relevance_justification=relevance.justification,
            early_warnings=early_warnings_text
        )

        synth_data = self._call_llm_structured(
            prompt,
            RTA_SYNTHESIZE_CHALLENGE_SCHEMA,
            max_tokens=self.MAX_TOKENS_CHALLENGE,
            temperature=self.TEMPERATURE_CHALLENGE
        )

        challenge_text = synth_data.get('challenge_text', '').strip()
        monitor_list = synth_data.get('monitor_list', []) or []

        early_warnings = monitor_list if monitor_list else (failure_mode.early_warnings or [])

        effective_relevance = (
            relevance.overall_relevance if failure_mode.is_downside_transferable
            else relevance.overall_relevance * self.UPSIDE_SCENARIO_DISCOUNT
        )

        severity = self._determine_severity(
            effective_relevance, 
            node.evidence_strength
        )

        cat_conf = getattr(failure_mode, 'category_confidence', 1.0)
        taxonomy_match = getattr(failure_mode, 'taxonomy_match', True)
        if cat_conf < self.LOW_CATEGORY_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Low category confidence detected",
                extra={"node_id": node.id, "category_confidence": cat_conf}
            )
            if severity == 'high':
                severity = 'medium'
            elif severity == 'medium':
                severity = 'low'

        analog.relevance_score = relevance.overall_relevance
        analog.relevance_reasoning = relevance.justification

        logger.debug(
            "Synthesized challenge",
            extra={
                "node_id": node.id,
                "analog": analog.case_name,
                "relevance": f"{relevance.overall_relevance:.2f}",
                "category_confidence": cat_conf,
                "taxonomy_match": taxonomy_match
            }
        )

        score_breakdown = getattr(node, 'score_breakdown', None)
        inputs_used = getattr(node, 'inputs_used', {})
        inputs_used['failure_mode_category_confidence'] = cat_conf
        inputs_used['failure_mode_taxonomy_match'] = taxonomy_match

        return RedTeamChallenge(
            node_id=node.id,
            assumption_text=node.claim,
            historical_precedent=analog,
            failure_mechanism=failure_mode,
            relevance=relevance,
            early_warning_indicators=early_warnings,
            challenge_text=challenge_text,
            severity=severity,
            suspected_value_exposure="UNKNOWN",
            score_breakdown=score_breakdown,
            inputs_used=inputs_used
        )

    def _validate_and_normalize_ndg(self, ndg: NDGOutput) -> NDGOutput:
        """
        Validate and normalize NDG inputs.

        Args:
            ndg: Graph output to validate

        Returns:
            Normalized NDG with confidence/evidence clamped to [0,1]

        Raises:
            ValueError: If nodes list is missing
        """
        if not hasattr(ndg, 'nodes') or not isinstance(ndg.nodes, list):
            raise ValueError("ndg must have a 'nodes' list")
        fm = getattr(ndg, 'fragility_metrics', None)
        if fm is None:
            class _FM:  # small local fallback container
                pass
            fm = _FM()
            fm.single_point_failures = []
            fm.path_concentration = {}
            ndg.fragility_metrics = fm
        else:
            if not hasattr(fm, 'single_point_failures') or fm.single_point_failures is None:
                fm.single_point_failures = []
            if not hasattr(fm, 'path_concentration') or fm.path_concentration is None:
                fm.path_concentration = {}

        for node in ndg.nodes:
            c = getattr(node, 'confidence', 0.0)
            try:
                node.confidence = max(0.0, min(1.0, float(c or 0.0)))
            except Exception:
                node.confidence = 0.0

            s = getattr(node, 'evidence_strength', 0.0)
            try:
                node.evidence_strength = max(0.0, min(1.0, float(s or 0.0)))
            except Exception:
                node.evidence_strength = 0.0

            if not getattr(node, 'node_type', None):
                node.node_type = 'ASSUMPTION'

        return ndg

    def _process_node_challenges(
        self,
        nodes_to_challenge: List[NDGNode],
        company_context: str
    ) -> tuple[List[RedTeamChallenge], Dict[str, int]]:
        """
        Generate challenges for high-priority nodes.

        Args:
            nodes_to_challenge: Nodes to challenge
            company_context: Company description

        Returns:
            Tuple of challenges and severity counts
        """
        challenges = []
        severity_counts = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 0, SEVERITY_LOW: 0}

        for i, node in enumerate(nodes_to_challenge, 1):
            try:
                analogs = self.retrieve_historical_analogs(node, company_context)
                if not analogs:
                    logger.warning(
                        f"[{i}/{len(nodes_to_challenge)}] No analogs found for node {node.id}, skipping"
                    )
                    continue

                best_analog = analogs[0]
                failure_mode = self.map_failure_mode(best_analog)
                relevance = self.score_relevance(best_analog, node, company_context)
                challenge = self.synthesize_challenge(node, best_analog, failure_mode, relevance)

                logger.info(
                    f"[{i}/{len(nodes_to_challenge)}] Node {node.id} challenged — "
                    f"analog: {best_analog.case_name}; relevance: {relevance.overall_relevance:.2f}; "
                    f"severity: {challenge.severity}"
                )

                challenges.append(challenge)
                severity_counts[challenge.severity] += 1

            except Exception as exc:
                logger.warning(
                    f"[{i}/{len(nodes_to_challenge)}] Exception while processing node "
                    f"{getattr(node, 'id', '<unknown>')}: {exc}"
                )
                continue

        return challenges, severity_counts

    def _build_summary_text(
        self,
        num_challenges: int,
        severity_counts: Dict[str, int],
        num_high_priority: int
    ) -> str:
        """
        Generate summary text for output.

        Args:
            num_challenges: Challenges generated
            severity_counts: Challenges by severity
            num_high_priority: High-priority nodes identified

        Returns:
            Summary string
        """
        return (
            f"Red Team: challenges={num_challenges} "
            f"(high={severity_counts[SEVERITY_HIGH]}, medium={severity_counts[SEVERITY_MEDIUM]}, "
            f"low={severity_counts[SEVERITY_LOW]}); high_priority_nodes={num_high_priority}"
        )

    def run(self, ndg: NDGOutput, company_context: str) -> RedTeamOutput:
        """
        Execute Red Team pipeline: prioritize assumptions, retrieve analogs, map failures,
        score relevance, and synthesize challenges.

        Args:
            ndg: Narrative decomposition graph
            company_context: Company description

        Returns:
            RedTeamOutput with epistemic challenges (severity is epistemic, not economic)

        Raises:
            ValueError: If company_context is empty
        """
        if not company_context:
            raise ValueError("company_context is required for Red Team analysis")

        ndg = self._validate_and_normalize_ndg(ndg)
        logger.info(f"Starting Red Team pipeline for {self.stock_ticker}")

        high_priority_nodes = self.prioritize_assumptions(ndg)
        logger.info(
            f"Identified {len(high_priority_nodes)} high-priority assumptions "
            f"(threshold: {self.MIN_CHALLENGE_PRIORITY})"
        )

        if not high_priority_nodes:
            logger.info("No nodes exceeded challenge-priority threshold.")
        else:
            for node in high_priority_nodes[:3]:
                logger.info(
                    f"Top priority: {node.claim[:80]}... (score: {node.challenge_priority_score:.2f})"
                )

        nodes_to_challenge = high_priority_nodes[:self.MAX_CHALLENGES]
        challenges, severity_counts = self._process_node_challenges(nodes_to_challenge, company_context)

        logger.info(
            f"Red Team generated {len(challenges)} challenges "
            f"(high={severity_counts[SEVERITY_HIGH]}, medium={severity_counts[SEVERITY_MEDIUM]}, "
            f"low={severity_counts[SEVERITY_LOW]})"
        )

        summary_text = self._build_summary_text(
            len(challenges),
            severity_counts,
            len(high_priority_nodes)
        )

        return RedTeamOutput(
            stock_ticker=self.stock_ticker,
            challenges=challenges,
            high_severity_count=severity_counts[SEVERITY_HIGH],
            medium_severity_count=severity_counts[SEVERITY_MEDIUM],
            low_severity_count=severity_counts[SEVERITY_LOW],
            summary_text=summary_text,
            challenged_node_ids=[n.id for n in nodes_to_challenge],
            node_score_breakdowns={n.id: getattr(n, 'score_breakdown', {}) for n in nodes_to_challenge}
        )
