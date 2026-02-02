import json
import copy
import logging
from typing import List, Dict, Optional, Any

from model.thesis_agents.llm_helper import LLMHelperMixin
from model.core.types import (
    CT_NODE_ASSUMPTION, CT_NODE_DRIVER, 
    ThesisQuantitativeContext, NDGOutput, RedTeamOutput, 
    Scenario, CREScenarioSet, CREGenerationResult,
)
from model.prompts.thesis_validation_prompts import CRE_BOUND_ASSUMPTIONS_PROMPT, CRE_GENERATE_SCENARIOS_PROMPT
from model.prompts.thesis_validation_schemas import CRE_BOUND_ASSUMPTIONS_SCHEMA, CRE_GENERATE_SCENARIOS_SCHEMA

logger = logging.getLogger(__name__)


class CounterfactualResearchEngine(LLMHelperMixin):
    """
    Generate bounded counterfactual scenarios from NDG and Red Team outputs.

    Extracts canonical assumptions, assigns empirical bounds via historical analogs,
    generates scenarios, and augments with Red Team-focused and extreme variants.

    Plausibility weights are scenario-level heuristics for ordering within a set.
    They are not calibrated probabilities and must not be aggregated without explicit
    documentation of the aggregation method.

    Attributes:
        stock_ticker: Company ticker symbol for context
    """
    MAX_TOKENS_BOUND: int = 3000
    MAX_TOKENS_SCENARIO_GENERATION: int = 4000
    SCENARIO_GENERATION_TEMPERATURE: float = 0.3
    FORECAST_YEARS: int = 5
    SUGGESTED_SCENARIO_TEMPLATES: List[str] = [
        "Demand Slowdown",
        "Competitive Pressure",
        "Margin Compression",
        "Regulatory Friction",
        "Capital Intensity Change",
        "Upside Acceleration"
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TAIL DISTANCE POLICY
    # ═══════════════════════════════════════════════════════════════════════════════
    # Controls how far "extreme" scenarios deviate from base case.
    # 
    # EXTREME_TAIL_DISTANCE (default: 1.5)
    #   - Multiplier for stressed assumption values in extreme variants
    #   - 1.5 = "50% more extreme than base scenario"
    #   - This is a POLICY CHOICE encoding tail severity assumption
    #   - NOT derived from data; defines what "extreme" means
    # 
    # Future V2: Replace with percentile-based deviations
    #   - Base case = 50th percentile
    #   - Adverse = 20th percentile  
    #   - Extreme = 5th percentile
    #   - Would eliminate multiplier ambiguity and align with risk language
    # ═══════════════════════════════════════════════════════════════════════════════
    
    GENERATE_EXTREME_VARIANTS: bool = True
    EXTREME_TAIL_DISTANCE: float = 1.5  # How far extreme scenarios deviate from base

    def __init__(self, stock_ticker: str):
        """
        Initialize CRE engine.

        Args:
            stock_ticker: Ticker symbol for company-specific analysis

        Raises:
            ValueError: If stock_ticker is empty or None
        """
        if not stock_ticker:
            raise ValueError("stock_ticker is required")
        self.stock_ticker = stock_ticker

    def _extract_canonical_assumptions(self, ndg: NDGOutput, red_team: Optional[RedTeamOutput]) -> tuple[Dict[str, Any], List[str]]:
        """
        Extract canonical metrics and claims from NDG output.

        Prefers ndg.extracted_metrics/claims, falls back to node claims. Red Team
        challenges are routed separately via priority_claims for focused generation.

        Args:
            ndg: NDG output
            red_team: Optional Red Team output (unused but kept for API consistency)

        Returns:
            Tuple of (metrics dict, thesis claims list)
        """
        metrics = getattr(ndg, 'extracted_metrics', {}) or {}
        claims = getattr(ndg, 'extracted_claims', None)
        if claims is None:
            claims = [n.claim for n in ndg.nodes if n.node_type in (CT_NODE_ASSUMPTION, CT_NODE_DRIVER)]

        return metrics, claims

    def _derive_priority_claims(self, ndg: NDGOutput, red_team: Optional[RedTeamOutput]) -> Optional[List[str]]:
        """
        Derive prioritized claims from Red Team challenges and NDG single-point failures.

        Args:
            ndg: NDG output
            red_team: Optional Red Team output

        Returns:
            List of unique prioritized claim texts, or None if no priority claims identified
        """
        claims: List[str] = []
        if red_team and red_team.challenges:
            for c in red_team.challenges:
                if c.challenge_text and c.challenge_text not in claims:
                    claims.append(c.challenge_text)

        spofs = getattr(ndg, 'fragility_metrics', None)
        if spofs and getattr(spofs, 'single_point_failures', None):
            id_to_claim = {n.id: n.claim for n in ndg.nodes}
            for nid in spofs.single_point_failures:
                claim_text = id_to_claim.get(nid)
                if claim_text and claim_text not in claims:
                    claims.append(claim_text)

        return claims or None

    def bound_assumptions(
        self,
        metrics: Dict,
        claims: List[str],
        company_context: str,
        quantitative_context: Optional[ThesisQuantitativeContext] = None
    ) -> Dict:
        """
        Assign empirical bounds to assumptions using historical analogs.

        Args:
            metrics: Extracted quantitative metrics from thesis
            claims: Qualitative claims from thesis
            company_context: Company description
            quantitative_context: Optional Stage 1 quantitative context

        Returns:
            Dict with 'bounds' mapping metrics to empirical min/max ranges with justifications
        
        Raises:
            ValueError: If company_context is empty
        """
        if not company_context:
            raise ValueError("company_context is required for accurate bounds")
        
        quant_context_str = ""
        if quantitative_context:
            quant_context_str = quantitative_context.to_prompt_context()
        
        prompt = self.format_prompt(
            CRE_BOUND_ASSUMPTIONS_PROMPT,
            stock_ticker=self.stock_ticker,
            company_context=company_context,
            quantitative_context=quant_context_str,
            metrics_json=json.dumps(metrics, indent=2),
            claims_json=json.dumps(claims, indent=2)
        )
        
        result = self._call_llm_structured(
            prompt,
            CRE_BOUND_ASSUMPTIONS_SCHEMA,
            max_tokens=self.MAX_TOKENS_BOUND,
        )
        
        return result

    def _generate_and_augment_scenarios(
        self,
        claims: List[str],
        metrics: Dict,
        bounds: Dict,
        ndg: NDGOutput,
        red_team: Optional[RedTeamOutput],
        quantitative_context: Optional[ThesisQuantitativeContext]
    ) -> List[Scenario]:
        """
        Generate scenarios and augment with Red Team challenges if provided.

        Args:
            claims: Canonical claims from NDG
            metrics: Extracted metrics
            bounds: Empirical bounds from bound_assumptions
            ndg: NDG output for priority claims derivation
            red_team: Optional Red Team output for augmentation
            quantitative_context: Optional quantitative context

        Returns:
            List of scenarios including base scenarios and Red Team augmentations
        """
        priority_claims = self._derive_priority_claims(ndg, red_team)
        
        scenarios = self.generate_scenarios(
            claims,
            metrics,
            bounds,
            quantitative_context,
            priority_claims=priority_claims,
            bias_toward_downside=bool(priority_claims)
        )
        
        if red_team and red_team.challenges:
            scenarios = self._augment_scenarios_with_red_team(scenarios, red_team, ndg, bounds)
        
        return scenarios

    def generate_scenarios(
        self,
        claims: List[str],
        metrics: Dict,
        bounds: Dict,
        quantitative_context: Optional[ThesisQuantitativeContext] = None,
        priority_claims: Optional[List[str]] = None,
        bias_toward_downside: bool = False
    ) -> List[Scenario]:
        """
        Generate plausible counterfactual scenarios with bounded variations.

        Focuses generation on priority_claims when provided (Red Team challenges and
        NDG single-point failures). When bias_toward_downside is True, emphasizes
        downside impacts.

        Args:
            claims: Canonical claims from thesis
            metrics: Extracted metrics
            bounds: Empirical bounds for metrics
            quantitative_context: Optional Stage 1 quantitative context
            priority_claims: Optional prioritized claims for focused generation
            bias_toward_downside: Whether to emphasize downside scenarios

        Returns:
            List of scenarios with structured stressed assumptions and RELATIVE plausibility weights.
            
            IMPORTANT: Plausibility weights are RELATIVE ordering within this scenario set.
            They are NOT calibrated probabilities and should NOT be aggregated as probabilities.
            Downstream evaluator is responsible for normalization and importance assignment.

        Note:
            Plausibility weights are heuristics for relative ordering within the generated set.
            They are not calibrated probabilities and must not be aggregated without explicit
            documentation.
        """
        quant_context_str = ""
        if quantitative_context:
            quant_context_str = quantitative_context.to_prompt_context()

        prompt = self.format_prompt(
            CRE_GENERATE_SCENARIOS_PROMPT,
            stock_ticker=self.stock_ticker,
            scenario_templates_json=json.dumps(self.SUGGESTED_SCENARIO_TEMPLATES, indent=2),
            quantitative_context=quant_context_str,
            claims_json=json.dumps(claims, indent=2),
            metrics_json=json.dumps(metrics, indent=2),
            bounds_json=json.dumps(bounds.get('bounds', {}), indent=2),
            priority_claims_json=json.dumps(priority_claims or [], indent=2),
            downside_bias=bias_toward_downside
        )

        result = self._call_llm_structured(
            prompt,
            CRE_GENERATE_SCENARIOS_SCHEMA,
            max_tokens=self.MAX_TOKENS_SCENARIO_GENERATION,
            temperature=self.SCENARIO_GENERATION_TEMPERATURE,
        )
        scenarios_data = result.get("scenarios", [])

        scenarios = []
        for s in scenarios_data:
            scenarios.append(Scenario(
                name=s.get("name", "Unnamed Scenario"),
                description=s.get("description", ""),
                impact=s.get("impact", ""),
                stressed_assumptions=s.get("stressed_assumptions", {}),
                plausibility_weight=s.get("plausibility_weight", 0.5),
                detailed_reasoning=s.get("justification", "")
            ))
            logger.debug(f"Generated scenario: {s.get('name', 'Unnamed Scenario')} (plausibility: {s.get('plausibility_weight', 0.5):.1%})")

        return scenarios

    def _augment_scenarios_with_red_team(self, scenarios: List[Scenario], red_team: RedTeamOutput, ndg: NDGOutput, bounds: Dict) -> List[Scenario]:
        """
        Augment scenarios with Red Team-focused and amplified extreme variants.

        Extreme variants are clipped to empirical bounds to maintain plausibility.

        Args:
            scenarios: Base scenarios from generate_scenarios
            red_team: Red Team output with challenges
            ndg: NDG output (unused but kept for API consistency)
            bounds: Empirical bounds from bound_assumptions for clipping

        Returns:
            List of scenarios including originals, Red Team-focused, and extreme variants
        """
        augmented = list(scenarios)

        for idx, ch in enumerate(red_team.challenges):
            if not ch.challenge_text:
                continue
            try:
                prompt = self.format_prompt(
                    CRE_GENERATE_SCENARIOS_PROMPT,
                    stock_ticker=self.stock_ticker,
                    scenario_templates_json=json.dumps(self.SUGGESTED_SCENARIO_TEMPLATES, indent=2),
                    quantitative_context="",
                    claims_json=json.dumps([ch.challenge_text], indent=2),
                    metrics_json=json.dumps({}, indent=2),
                    bounds_json=json.dumps({}, indent=2),
                    priority_claims_json=json.dumps([ch.challenge_text], indent=2),
                    downside_bias=True
                )

                result = self._call_llm_structured(
                    prompt,
                    CRE_GENERATE_SCENARIOS_SCHEMA,
                    max_tokens=self.MAX_TOKENS_SCENARIO_GENERATION,
                    temperature=self.SCENARIO_GENERATION_TEMPERATURE,
                )

                # Return all generated scenarios, let downstream handle pruning
                # Rationale: Different theses generate different risk densities;
                #            evaluator/plausibility weighting should determine relevance
                for s_idx, s in enumerate(result.get('scenarios', [])):
                    augmented.append(Scenario(
                        name=f"RT-{idx+1}.{s_idx+1}: {s.get('name', 'RedTeam')}",
                        description=s.get('description', ch.challenge_text),
                        impact=s.get('impact', ''),
                        stressed_assumptions=s.get('stressed_assumptions', {}),
                        plausibility_weight=s.get('plausibility_weight', 0.25),
                        detailed_reasoning=s.get('justification', '')
                    ))
            except Exception:
                logger.debug(f"Focused scenario generation for Red Team challenge failed: {ch.challenge_text}")
                continue

        extreme_variants = []
        for s in augmented:
            if not s.stressed_assumptions:
                continue
            new_stressed = copy.deepcopy(s.stressed_assumptions)
            changed = False

            for key, val in list(new_stressed.items()):
                if val is None:
                    continue
                try:
                    if isinstance(val, (int, float)):
                        if self.GENERATE_EXTREME_VARIANTS:
                            # Apply tail distance uniformly
                            if key.endswith('_factor'):
                                # For factors: move away from 1.0
                                if val < 1.0:
                                    new_stressed[key] = val / self.EXTREME_TAIL_DISTANCE
                                else:
                                    new_stressed[key] = val * self.EXTREME_TAIL_DISTANCE
                            else:
                                # For absolute values: scale up
                                new_stressed[key] = val * self.EXTREME_TAIL_DISTANCE
                            changed = True
                    else:
                        new_stressed[key] = val
                    
                    if isinstance(new_stressed[key], (int, float)) and 'bounds' in bounds and key in bounds['bounds']:
                        bound_info = bounds['bounds'][key]
                        min_v = bound_info.get('min')
                        max_v = bound_info.get('max')
                        if min_v is not None:
                            new_stressed[key] = max(min_v, new_stressed[key])
                        if max_v is not None:
                            new_stressed[key] = min(max_v, new_stressed[key])
                except Exception:
                    continue

            if changed and new_stressed != s.stressed_assumptions:
                extreme_variants.append(Scenario(
                    name=f"{s.name} - Extreme Variant",
                    description=f"Extreme-tilted variant focused on: {s.description}",
                    impact=s.impact,
                    stressed_assumptions=new_stressed,
                    plausibility_weight=s.plausibility_weight * 0.5,  # Extreme discount only
                    detailed_reasoning=(s.detailed_reasoning or "")
                ))

        augmented.extend(extreme_variants)
        logger.info(f"Augmented scenarios with {len(augmented)-len(scenarios)} Red Team focused scenarios and extreme variants")
        return augmented

    def _apply_defaults_to_metrics(self, metrics: Dict) -> tuple[Dict[str, Any], List[str]]:
        """
        Apply minimal defaults to provided metrics.

        Args:
            metrics: Extracted metrics from thesis

        Returns:
            Tuple of (base_metrics with defaults applied, list of applied defaults)
        """
        base_metrics = dict(metrics)
        defaults_applied: List[str] = []

        if not any(k for k in base_metrics.keys() if 'growth' in k or 'cagr' in k or 'revenue' in k):
            base_metrics['revenue_growth'] = None
            defaults_applied.append('revenue_growth')

        if 'forecast_years' not in base_metrics:
            base_metrics['forecast_years'] = self.FORECAST_YEARS
            defaults_applied.append('forecast_years')

        return base_metrics, defaults_applied

    def _prepare_scenario_set(self, scenarios: List[Scenario], base_metrics: Dict[str, Any], bounds: Dict) -> CREScenarioSet:
        """
        Prepare final scenario set with scenarios and metadata.

        Args:
            scenarios: Generated scenarios
            base_metrics: Base metrics with defaults applied
            bounds: Empirical bounds for metrics

        Returns:
            CREScenarioSet with minimal summary (counts only, no diagnostics)
        """
        summary_text = f"CRE: scenarios={len(scenarios)}; generated_raw={len(scenarios)}"
        return CREScenarioSet(
            stock_ticker=self.stock_ticker,
            scenarios=scenarios,
            rejected_scenarios=[],
            base_metrics=base_metrics,
            bounds=bounds,
            generated_raw=[s.__dict__ for s in scenarios],
            summary_text=summary_text,
            total_duration_ms=None
        )
    
    def run(
        self, 
        ndg: NDGOutput,
        red_team: Optional[RedTeamOutput],
        company_context: str,
        quantitative_context: Optional[ThesisQuantitativeContext],
    ) -> CREGenerationResult:
        """
        Execute scenario generation pipeline.

        Extracts canonical assumptions, assigns empirical bounds, generates scenarios,
        and augments with Red Team challenges and extreme variants.

        Args:
            ndg: Narrative decomposition graph
            red_team: Optional Red Team output for challenge augmentation
            company_context: Company description
            quantitative_context: Optional Stage 1 quantitative context

        Returns:
            CREGenerationResult with scenario set, canonical claims, and applied defaults

        Raises:
            ValueError: If company_context or ndg is missing
        """
        if not company_context:
            raise ValueError("company_context is required for accurate analysis")
        if ndg is None:
            raise ValueError("ndg (NarrativeDecompositionGraph output) is required")
        
        logger.info(f"Starting CRE generation for {self.stock_ticker}")

        metrics, claims = self._extract_canonical_assumptions(ndg, red_team)
        logger.info(f"Using NDG-extracted assumptions: {len(claims)} claims, metrics keys: {list(metrics.keys())}")
        
        bounds = self.bound_assumptions(metrics, claims, company_context, quantitative_context)
        scenarios = self._generate_and_augment_scenarios(claims, metrics, bounds, ndg, red_team, quantitative_context)
        
        base_metrics, defaults_applied = self._apply_defaults_to_metrics(metrics)
        scenario_set = self._prepare_scenario_set(scenarios, base_metrics, bounds)

        logger.info(f"CRE generation complete (scenarios={len(scenario_set.scenarios)})")
       
        return CREGenerationResult(
            scenario_set=scenario_set,
            claims=claims,
            defaults_applied=defaults_applied
        )
