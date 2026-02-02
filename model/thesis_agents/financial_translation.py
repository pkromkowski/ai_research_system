import copy
import json
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from model.thesis_agents.llm_helper import LLMHelperMixin
from model.core.types import (
    OUTCOME_TIER_SURVIVES, OUTCOME_TIER_IMPAIRED, OUTCOME_TIER_BROKEN,
    Scenario, CREScenarioSet, CREGenerationResult, ValuationResult, FTOutput, FTResult,
    
)
from model.prompts.thesis_validation_schemas import FT_GENERATE_REASONING_SCHEMA, FT_SUMMARY_SCHEMA, METRICS_BATCH_CLASSIFICATION_SCHEMA
from model.prompts.thesis_validation_prompts import FT_GENERATE_REASONING_PROMPT, FT_SUMMARY_PROMPT, METRICS_BATCH_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class FinancialTranslation(LLMHelperMixin):
    """
    Deterministic financial translation layer.

    Maps CRE scenarios to valuations using factor-based scoring, validates scenario
    plausibility, and generates structured executive summaries.
    """
    MAX_TOKENS_REASONING: int = 500
    MAX_TOKENS_SUMMARY: int = 500
    REASONING_TEMPERATURE: float = 0.2
    SUMMARY_TEMPERATURE: Optional[float] = None
    MAX_WORKERS: int = 4
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DECISION TIER THRESHOLDS
    # ═══════════════════════════════════════════════════════════════════════════════
    # These are PM-facing heuristics for categorizing scenario outcomes.
    # They are NOT calibrated to valuation theory or empirical drawdowns.
    # They are strategy-dependent decision boundaries.
    #
    # DECISION_TOLERANCE_THRESHOLD: -10%
    #   - "Thesis still works" = valuation impact < 10% drawdown
    #   - For growth investors: -10% might be too strict
    #   - For value investors: -10% might be too loose
    #
    # CAPITAL_IMPAIRMENT_THRESHOLD: -40%
    #   - "Permanent capital loss risk" = valuation impact > 40% drawdown
    #   - This is a material decision boundary for position sizing
    #
    # DO NOT treat these as economic truth - they are operational heuristics.
    # V2 calibration: Should be derived from realized PnL / drawdown data.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    DECISION_TOLERANCE_THRESHOLD: float = -0.10
    CAPITAL_IMPAIRMENT_THRESHOLD: float = -0.40
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SCENARIO BOUNDS
    # ═══════════════════════════════════════════════════════════════════════════════
    # Defensive programming to prevent pathological scenario explosions.
    # These are NOT modeling choices - they are operational constraints.
    # Should log when clipping occurs for diagnostics.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    MIN_VALUATION_CHANGE: float = -0.80  # Prevent -1000% scenarios
    MAX_VALUATION_CHANGE: float = 1.50   # Prevent +10000% scenarios
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MARGIN OF SAFETY
    # ═══════════════════════════════════════════════════════════════════════════════
    # Strategy-specific baseline margin of safety.
    # Should NEVER be baked into core translation logic.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    MARGIN_OF_SAFETY_BASELINE: float = 0.20
    #      - Sensitivity folded into sign (+ / –) and presence/absence
    #      - FACTOR_WEIGHTS alone controls importance
    #      - Cuts parameter count in half, removes silent interactions
    # ═══════════════════════════════════════════════════════════════════════════════
    
    FACTOR_WEIGHTS: Dict[str, float] = {
        "revenue_scale": 0.25,
        "revenue_durability": 0.15,
        "unit_economics": 0.20,
        "capital_intensity": 0.10,
        "optionality": 0.10,
        "competitive_intensity": 0.10,
        "sales_efficiency": 0.05
    }
    # ═══════════════════════════════════════════════════════════════════════════════
    # METRIC-TO-FACTOR MAPPING
    # ═══════════════════════════════════════════════════════════════════════════════
    # These coefficients encode CAUSAL BELIEFS, not statistical relationships.
    # They represent "how much does this metric move this factor?"
    #
    # Current representation: Numeric coefficients (false precision)
    # V2 improvement: Replace with structural graph:
    #   - Direction: {positive, negative}
    #   - Strength: {weak: 0.3, medium: 0.6, strong: 1.0}
    #   - Map categories → numeric weights once, downstream
    #
    # Benefits of V2 approach:
    #   - Makes beliefs explicit and reviewable by PMs
    #   - Reduces false precision
    #   - Separates economic logic from numeric calibration
    #
    # For v1: Keep numeric form but recognize as placeholders for structural model.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    FACTOR_MAPPING: Dict[str, Dict[str, float]] = {
        "growth": {"revenue_scale": 1.0, "revenue_durability": 0.2},
        "cagr": {"revenue_scale": 1.0},
        "revenue": {"revenue_scale": 0.8},
        "gross_margin": {"unit_economics": 0.8},
        "operating_margin": {"unit_economics": 0.6},
        "fcf_margin": {"unit_economics": 1.0},
        "retention": {"revenue_durability": 1.0},
        "net_retention": {"revenue_durability": 1.0},
        "cac": {"unit_economics": -0.8, "sales_efficiency": -0.6},
        "payback": {"unit_economics": -0.6},
        "wacc": {"capital_intensity": -1.0},
        "discount_rate": {"capital_intensity": -1.0},
        "terminal_multiple": {"optionality": 0.8},
        "exit_multiple": {"optionality": 0.8},
        "gpu": {"capital_intensity": -0.5},
        "pricing": {"optionality": 0.4},
        "sales_efficiency": {"sales_efficiency": 1.0}
    }

    def __init__(self, stock_ticker: str):
        """
        Initialize Financial Translation engine.

        Args:
            stock_ticker: Company ticker symbol

        Raises:
            ValueError: If stock_ticker is empty
        """
        if not stock_ticker:
            raise ValueError("stock_ticker is required")
        self.stock_ticker = stock_ticker
        self._metric_classification_cache: Dict[str, Dict[str, Any]] = {}

    def _evaluate_and_aggregate(self, scenario_set: CREScenarioSet, defaults_applied: Optional[List[str]] = None) -> FTOutput:
        """
        Evaluate scenario set and aggregate factor diagnostics.

        Args:
            scenario_set: CRE scenario set to evaluate
            defaults_applied: List of defaulted metrics

        Returns:
            FTOutput with scenario results and aggregated diagnostics
        """
        cre_output = self.evaluate_scenario_set(scenario_set, defaulted_metrics=defaults_applied)
        cre_output.generated_raw = scenario_set.generated_raw
        cre_output.generated_count = len(scenario_set.scenarios)
        cre_output.defaults_applied = defaults_applied

        factor_to_sum: Dict[str, float] = {}
        weight_sum = 0.0
        metric_map_agg: Dict[str, Dict[str, float]] = {}
        inferred_set = set()
        for r in cre_output.scenario_results:
            if r.factor_contributions:
                for f, v in r.factor_contributions.items():
                    factor_to_sum[f] = factor_to_sum.get(f, 0.0) + v * r.plausibility_weight
                weight_sum += r.plausibility_weight
            if r.metric_to_factor_mapping:
                for m, mapping in r.metric_to_factor_mapping.items():
                    metric_map_agg.setdefault(m, {})
                    for fac, coeff in mapping.items():
                        metric_map_agg[m][fac] = metric_map_agg[m].get(fac, 0.0) + coeff
                    if mapping and not all(abs(c) < 1e-6 for c in mapping.values()):
                        inferred_set.add(m)

        if weight_sum > 0:
            cre_output.factor_scores = {f: factor_to_sum[f] / weight_sum for f in factor_to_sum}
        else:
            cre_output.factor_scores = None

        cre_output.metric_factor_mapping = {m: metric_map_agg[m] for m in metric_map_agg}
        cre_output.inferred_metrics = list(inferred_set) if inferred_set else None

        return cre_output

    def evaluate_scenario_set(self, scenario_set: CREScenarioSet, defaulted_metrics: Optional[List[str]] = None) -> FTOutput:
        """
        Evaluate CRE scenario set.

        Args:
            scenario_set: Scenario set to evaluate
            defaulted_metrics: List of defaulted metrics

        Returns:
            FTOutput with scenario results
        """
        results, rejected = self.evaluate_scenarios(
            scenario_set.scenarios,
            scenario_set.base_metrics,
            scenario_set.bounds,
            defaulted_metrics=defaulted_metrics,
        )

        return FTOutput(
            scenario_results=results,
            summary_text=scenario_set.summary_text,
            impaired_scenarios=[r.scenario_name for r in results if r.outcome_tier == OUTCOME_TIER_IMPAIRED][:3]
        )

    def evaluate_scenarios(
        self,
        scenarios: List[Scenario],
        base_metrics: Dict[str, Any],
        bounds: Dict,
        defaulted_metrics: Optional[List[str]] = None,
    ) -> tuple[List[ValuationResult], List[str]]:
        """
        Evaluate multiple scenarios concurrently.

        Args:
            scenarios: Scenarios to evaluate
            base_metrics: Base case metrics
            bounds: Empirical bounds
            defaulted_metrics: List of defaulted metrics

        Returns:
            Tuple of (valuation results, rejected scenario names)
        """
        results: List[ValuationResult] = []
        rejected_scenarios: List[str] = []

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_scenario = {
                executor.submit(self._evaluate_single_scenario, scenario, base_metrics, bounds, defaulted_metrics): scenario
                for scenario in scenarios
            }

            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                    if result is None:
                        rejected_scenarios.append(scenario.name)
                    else:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Scenario {scenario.name} failed: {e}")
                    rejected_scenarios.append(scenario.name)

        return results, rejected_scenarios

    def _evaluate_single_scenario(
        self,
        scenario: Scenario,
        base_metrics: Dict[str, Any],
        bounds: Dict,
        defaulted_metrics: Optional[List[str]] = None,
    ) -> Optional[ValuationResult]:
        """
        Evaluate single scenario with factor-based valuation.

        Args:
            scenario: Scenario to evaluate
            base_metrics: Base case metrics
            bounds: Empirical bounds
            defaulted_metrics: List of defaulted metrics

        Returns:
            ValuationResult or None if invalid
        """
        if not self.validate_scenario(scenario, base_metrics, bounds):
            return None

        stressed_metrics = self.stress_assumptions(base_metrics, scenario)
        valuation_change, factor_contributions, metric_map, inferred_metrics = self._factor_valuation(
            stressed_metrics,
            base_metrics,
            defaulted_metrics=defaulted_metrics,
            factor_weight_overrides=getattr(scenario, 'factor_weight_overrides', None)
        )
        outcome_tier = self._classify_outcome(valuation_change)
        margin_of_safety = max(0, self.MARGIN_OF_SAFETY_BASELINE + valuation_change)
        detailed_reasoning = self.generate_detailed_reasoning(
            scenario, stressed_metrics, base_metrics, valuation_change, outcome_tier, factor_contributions, metric_map
        )

        return ValuationResult(
            scenario_name=scenario.name,
            valuation_change=valuation_change,
            outcome_tier=outcome_tier,
            narrative_consistent=(outcome_tier == OUTCOME_TIER_SURVIVES),
            margin_of_safety=margin_of_safety,
            plausibility_weight=scenario.plausibility_weight,
            detailed_reasoning=detailed_reasoning,
            factor_contributions=factor_contributions,
            metric_to_factor_mapping=metric_map
        )
    
    def validate_scenario(self, scenario: Scenario, base_metrics: Dict, bounds: Dict) -> bool:
        """
        Check scenario for metric bound violations.

        Args:
            scenario: Scenario to validate
            base_metrics: Base case metrics
            bounds: Empirical bounds

        Returns:
            True if valid
        """
        stressed = scenario.stressed_assumptions
        bounds_dict = bounds.get('bounds', {})

        for key, value in stressed.items():
            if key.endswith("_factor"):
                metric_name = key.replace("_factor", "")
                if metric_name in bounds_dict:
                    bound = bounds_dict[metric_name]
                    base_val = base_metrics.get(metric_name)
                    if isinstance(base_val, list):
                        base_val = base_val[0] if base_val else None
                    if base_val is not None:
                        stressed_val = base_val * value
                        if stressed_val < bound.get('min', float('-inf')) or \
                            stressed_val > bound.get('max', float('inf')):
                            logger.debug(f"Scenario {scenario.name} rejected: {metric_name} stress {stressed_val} out of bounds")
                            return False

        return True

    def stress_assumptions(self, base_metrics: Dict[str, Any], scenario: Scenario) -> Dict[str, Any]:
        """
        Apply scenario shocks to base assumptions.

        Args:
            base_metrics: Base case metrics
            scenario: Scenario with stressed_assumptions

        Returns:
            Stressed metrics
        """
        stressed = copy.deepcopy(base_metrics)

        for key, value in scenario.stressed_assumptions.items():
            if value is None:
                continue

            if key.endswith('_override'):
                metric_name = key.replace('_override', '')
                stressed[metric_name] = value

            elif key.endswith('_factor'):
                metric_name = key.replace('_factor', '')
                if metric_name in stressed and stressed[metric_name] is not None:
                    base_val = stressed[metric_name]
                    if isinstance(base_val, list):
                        stressed[metric_name] = [v * value for v in base_val]
                    else:
                        stressed[metric_name] = base_val * value

            else:
                if key in stressed and stressed[key] is not None:
                    base_val = stressed[key]
                    if isinstance(base_val, list):
                        stressed[key] = [v + value for v in base_val]
                    else:
                        stressed[key] = base_val + value
                else:
                    stressed[key] = value

        return stressed
    
    def generate_detailed_reasoning(
        self,
        scenario: Scenario,
        stressed_metrics: Dict[str, Any],
        base_metrics: Dict[str, Any],
        valuation_change: float,
        outcome_tier: str,
        factor_contributions: Optional[Dict[str, float]] = None,
        metric_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """
        Generate LLM explanation for scenario outcome.

        Args:
            scenario: Scenario being evaluated
            stressed_metrics: Stressed metrics
            base_metrics: Base case metrics
            valuation_change: Computed valuation change
            outcome_tier: Classification (SURVIVES/IMPAIRED/BROKEN)
            factor_contributions: Factor contributions
            metric_map: Metric-to-factor mapping

        Returns:
            Detailed reasoning string
        """
        def format_metrics_summary(metrics: Dict, label: str) -> str:
            lines = [f"{label}:"]
            for key, val in sorted(metrics.items()):
                if val is None:
                    continue
                if isinstance(val, float):
                    if any(x in key for x in ['margin', 'retention', 'growth', 'rate']):
                        lines.append(f"  - {key}: {val:.1%}")
                    elif abs(val) < 2:
                        lines.append(f"  - {key}: {val:.2f}")
                    else:
                        lines.append(f"  - {key}: {val:.1f}")
                elif isinstance(val, list):
                    if all(isinstance(v, (int, float)) for v in val):
                        if 'growth' in key:
                            lines.append(f"  - {key}: {[f'{v:.1%}' for v in val]}")
                        else:
                            lines.append(f"  - {key}: {val}")
                    else:
                        lines.append(f"  - {key}: {val}")
                else:
                    lines.append(f"  - {key}: {val}")
            return "\n".join(lines)

        base_summary = format_metrics_summary(base_metrics, "Base Case Assumptions")
        stressed_summary = format_metrics_summary(stressed_metrics, "Stressed Assumptions")

        prompt = self.format_prompt(
            FT_GENERATE_REASONING_PROMPT,
            stock_ticker=self.stock_ticker,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            base_summary=base_summary,
            stressed_summary=stressed_summary,
            factor_contributions_json=json.dumps(factor_contributions or {}, indent=2),
            metric_map_json=json.dumps(metric_map or {}, indent=2),
            valuation_change=f"{valuation_change:+.1%}",
            outcome_tier=outcome_tier
        )

        result = self._call_llm_structured(
            prompt,
            FT_GENERATE_REASONING_SCHEMA,
            max_tokens=self.MAX_TOKENS_REASONING,
            temperature=self.REASONING_TEMPERATURE,
        )

        explanation = result.get('explanation', '').strip()
        key_drivers = result.get('key_drivers', [])
        hist = result.get('historical_precedent', '')
        related_factors = result.get('related_factors', [])
        factor_expl = result.get('factor_explanation', '')

        summary_lines = [explanation]
        if key_drivers:
            summary_lines.append(f"Key drivers: {', '.join(key_drivers)}")
        if related_factors:
            summary_lines.append(f"Related factors: {', '.join(related_factors)}")
        if factor_expl:
            summary_lines.append(f"Factor explanation: {factor_expl}")
        if hist:
            summary_lines.append(f"Historical precedent: {hist}")

        return "\n".join(summary_lines)
    
    def _generate_structured_summary(self, cre_output: FTOutput, claims: List[str], scenario_set: CREScenarioSet) -> Optional[Dict[str, Any]]:
        """
        Generate structured executive summary via LLM.

        Args:
            cre_output: FT output with scenario results
            claims: Thesis claims
            scenario_set: Original scenario set

        Returns:
            Structured summary dict or None on failure
        """
        try:
            claims_json = json.dumps(claims if claims else [s.description for s in scenario_set.scenarios], indent=2)
            results_json = json.dumps([
                {
                    'scenario_name': r.scenario_name,
                    'outcome_tier': r.outcome_tier,
                    'valuation_change': r.valuation_change,
                    'plausibility_weight': r.plausibility_weight
                }
                for r in cre_output.scenario_results
            ], indent=2)

            prompt = self.format_prompt(
                FT_SUMMARY_PROMPT,
                stock_ticker=self.stock_ticker,
                claims_json=claims_json,
                results_json=results_json,
                defaults_json=json.dumps(cre_output.defaults_applied or [], indent=2),
                factor_scores_json=json.dumps(cre_output.factor_scores or {}, indent=2),
                metric_map_json=json.dumps(cre_output.metric_factor_mapping or {}, indent=2),
            )

            summary_struct = self._call_llm_structured(
                prompt,
                FT_SUMMARY_SCHEMA,
                max_tokens=self.MAX_TOKENS_SUMMARY,
                temperature=self.SUMMARY_TEMPERATURE or 0.0
            )

            return summary_struct
        except Exception as e:
            logger.warning(f"FT summary generation failed: {e}")
            return None

    def _factor_valuation(self, stressed_metrics: Dict[str, Any], base_metrics: Dict[str, Any], defaulted_metrics: Optional[List[str]] = None, factor_weight_overrides: Optional[Dict[str, float]] = None) -> tuple[float, Dict[str, float], Dict[str, Dict[str, float]], List[str]]:
        """
        Compute valuation change using factor deltas and weights.

        Args:
            stressed_metrics: Stressed metrics
            base_metrics: Base case metrics
            defaulted_metrics: List of defaulted metrics
            factor_weight_overrides: Scenario-specific weight adjustments

        Returns:
            Tuple of (valuation_change, factor_deltas, metric_to_factor_map, inferred_metrics)
        """
        factor_deltas, metric_map, inferred_metrics = self._compute_factor_deltas(base_metrics, stressed_metrics, defaulted_metrics=defaulted_metrics)

        # Factor aggregation assumes independence (no correlation structure)
        # Multiple metrics mapping to same factor stack linearly, which may:
        # - Overweight factors with dense metric coverage
        # - Ignore correlation between factors (e.g., revenue_scale <-> unit_economics)
        # Mitigation: factor_deltas clipped to [-1, 1], no additional scaling applied

        total_weight = sum(self.FACTOR_WEIGHTS.values()) or 1.0
        valuation = 0.0
        for factor, delta in factor_deltas.items():
            base_w = self.FACTOR_WEIGHTS.get(factor, 0.0) / total_weight
            override = 1.0
            if factor_weight_overrides and factor in factor_weight_overrides:
                try:
                    override_raw = float(factor_weight_overrides[factor])
                    override = max(0.5, min(2.0, override_raw))
                except Exception:
                    override = 1.0
            effective_w = base_w * override
            valuation += delta * effective_w

        valuation_change = max(self.MIN_VALUATION_CHANGE, min(self.MAX_VALUATION_CHANGE, valuation))
        return valuation_change, factor_deltas, metric_map, inferred_metrics
    
    def _compute_factor_deltas(self, base_metrics: Dict[str, Any], stressed_metrics: Dict[str, Any], defaulted_metrics: Optional[List[str]] = None) -> tuple[Dict[str, float], Dict[str, Dict[str, float]], List[str]]:
        """
        Map metric deltas to factor deltas using FACTOR_MAPPING.

        Args:
            base_metrics: Base case metrics
            stressed_metrics: Stressed metrics
            defaulted_metrics: List of defaulted metrics

        Returns:
            Tuple of (factor_deltas, metric_to_factor_map, inferred_metrics)
        """
        factor_deltas: Dict[str, float] = {f: 0.0 for f in self.FACTOR_WEIGHTS.keys()}
        metric_map: Dict[str, Dict[str, float]] = {}
        inferred_metrics: List[str] = []
        defaulted_metrics = defaulted_metrics or []
        all_metrics = set(base_metrics.keys()) | set(stressed_metrics.keys())

        unmatched_metrics: List[tuple[str, Any, float]] = []

        for metric in all_metrics:
            base_val = base_metrics.get(metric)
            stressed_val = stressed_metrics.get(metric)
            delta = self._compute_metric_delta(metric, stressed_val, base_val)
            if delta is None:
                continue

            matched = False
            for key, factor_dict in self.FACTOR_MAPPING.items():
                if self._metric_matches_key(metric, key):
                    matched = True
                    metric_map.setdefault(metric, {})
                    for factor, coeff in factor_dict.items():
                        sensitivity = 1.0 
                        contrib = coeff * delta * sensitivity
                        if metric in defaulted_metrics:
                            contrib *= (1.0 - self.SYNTHETIC_METRIC_PENALTY)
                        factor_deltas[factor] = factor_deltas.get(factor, 0.0) + contrib
                        metric_map[metric][factor] = metric_map[metric].get(factor, 0.0) + coeff

            if not matched:
                cached = self._metric_classification_cache.get(metric)
                if cached:
                    if cached.get('factor_influences'):
                        inferred_metrics.append(metric)
                        metric_map.setdefault(metric, {})
                        for factor, coeff in cached['factor_influences'].items():
                            try:
                                coeff_f = float(coeff)
                            except Exception:
                                coeff_f = 0.0
                            sensitivity = 1.0
                            contrib = coeff_f * delta * sensitivity
                            if metric in defaulted_metrics:
                                contrib *= (1.0 - self.SYNTHETIC_METRIC_PENALTY)
                            factor_deltas[factor] = factor_deltas.get(factor, 0.0) + contrib
                            metric_map[metric][factor] = metric_map[metric].get(factor, 0.0) + coeff_f
                    else:
                        metric_map.setdefault(metric, {})
                else:
                    unmatched_metrics.append((metric, stressed_val, delta))

        if unmatched_metrics:
            try:
                batch_classifications = self._classify_metrics_batch(
                    [(m, v) for m, v, _ in unmatched_metrics],
                    company_context=self.stock_ticker
                )

                for metric, stressed_val, delta in unmatched_metrics:
                    classification = batch_classifications.get(metric)
                    if classification:
                        self._metric_classification_cache[metric] = classification

                        if classification.get('factor_influences'):
                            inferred_metrics.append(metric)
                            metric_map.setdefault(metric, {})
                            for factor, coeff in classification['factor_influences'].items():
                                try:
                                    coeff_f = float(coeff)
                                except Exception:
                                    coeff_f = 0.0
                                sensitivity = 1.0 
                                contrib = coeff_f * delta * sensitivity
                                if metric in defaulted_metrics:
                                    contrib *= (1.0 - self.SYNTHETIC_METRIC_PENALTY)
                                factor_deltas[factor] = factor_deltas.get(factor, 0.0) + contrib
                                metric_map[metric][factor] = metric_map[metric].get(factor, 0.0) + coeff_f
                        else:
                            metric_map.setdefault(metric, {})
                    else:
                        metric_map.setdefault(metric, {})
            except Exception as e:
                logger.warning(f"Batch metric classification failed: {e}")
                for metric, _, _ in unmatched_metrics:
                    metric_map.setdefault(metric, {})

        for f in factor_deltas:
            factor_deltas[f] = max(-1.0, min(1.0, factor_deltas[f]))

        factor_concentration = {f: 0 for f in self.FACTOR_WEIGHTS.keys()}
        for metric, factors in metric_map.items():
            for factor in factors.keys():
                if factor in factor_concentration:
                    factor_concentration[factor] += 1

        metric_map['__factor_concentration__'] = factor_concentration

        return factor_deltas, metric_map, inferred_metrics

    def _classify_metrics_batch(self, metrics: List[tuple[str, Any]], company_context: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Batch classify multiple metrics into factor influences.

        Args:
            metrics: List of (metric_name, metric_value) tuples
            company_context: Company context

        Returns:
            Dict mapping metric names to classification results
        """
        if not metrics:
            return {}
        
        try:
            metrics_json = json.dumps([
                {"metric": m, "value": json.dumps(v) if v is not None else ""}
                for m, v in metrics
            ], indent=2)
            
            prompt = self.format_prompt(
                METRICS_BATCH_CLASSIFICATION_PROMPT,
                company_context=company_context or "",
                metrics_json=metrics_json
            )
            
            result = self._call_llm_structured(
                prompt,
                METRICS_BATCH_CLASSIFICATION_SCHEMA,
                max_tokens=200 * len(metrics),  # Scale token limit with batch size
                temperature=0.0
            )
            
            # Convert array result to dict keyed by metric name
            classifications = {}
            if result and 'classifications' in result:
                for item in result['classifications']:
                    metric_name = item.get('metric')
                    if metric_name:
                        classifications[metric_name] = item
            
            return classifications

        except Exception as e:
            logger.warning(f"Batch metric classification failed: {e}")
            return {}

    def _compute_metric_delta(self, metric_name: str, stressed_val: Any, base_val: Any) -> Optional[float]:
        """
        Compute normalized metric delta.

        Args:
            metric_name: Metric name
            stressed_val: Stressed value
            base_val: Base value

        Returns:
            Delta (positive=improvement, negative=deterioration) or None
        """
        if base_val is None and stressed_val is None:
            return None

        if isinstance(base_val, list) or isinstance(stressed_val, list) or 'growth' in metric_name or 'cagr' in metric_name:
            base_list = base_val if isinstance(base_val, list) else [base_val] if base_val is not None else None
            stressed_list = stressed_val if isinstance(stressed_val, list) else [stressed_val] if stressed_val is not None else None
            return self._delta_growth_series(base_list, stressed_list)

        if any(x in metric_name for x in ['margin', 'rate', 'wacc', 'retention', 'payback']):
            return self._delta_margin_like(metric_name, stressed_val, base_val)

        return self._delta_relative(stressed_val, base_val)

    def _delta_growth_series(self, base_list: Optional[List[float]], stressed_list: Optional[List[float]]) -> Optional[float]:
        """Compute delta for growth series by comparing compounded growth."""
        if not (base_list and stressed_list):
            return None

        eps = 1e-9
        base_prod = 1.0
        for g in base_list:
            base_prod *= (1 + (g if g is not None else 0))
        stressed_prod = 1.0
        for g in stressed_list:
            stressed_prod *= (1 + (g if g is not None else 0))
        return stressed_prod / (base_prod + eps) - 1.0

    def _delta_margin_like(self, metric_name: str, stressed_val: Any, base_val: Any) -> Optional[float]:
        """Compute absolute delta for margin-like metrics."""
        if base_val is None or stressed_val is None:
            return None
        try:
            return float(stressed_val) - float(base_val)
        except Exception:
            return None

    def _delta_relative(self, stressed_val: Any, base_val: Any) -> Optional[float]:
        """Compute relative delta for general metrics."""
        if base_val is None or stressed_val is None:
            return None
        eps = 1e-9
        try:
            return (float(stressed_val) - float(base_val)) / (abs(float(base_val)) + eps)
        except Exception:
            return None

    def _metric_matches_key(self, metric: str, key: str) -> bool:
        """Check if metric matches FACTOR_MAPPING key."""
        m = metric.lower()
        k = key.lower()
        return m == k or m.startswith(k) or k in m.split('_') or k in m

    def _classify_outcome(self, valuation_change: float) -> str:
        """
        Classify outcome tier from valuation change.

        Args:
            valuation_change: Computed valuation change

        Returns:
            Outcome tier: SURVIVES, IMPAIRED, or BROKEN
        """
        scale = 1.0
        factor_score = valuation_change / scale
        if factor_score >= self.DECISION_TOLERANCE_THRESHOLD:
            return OUTCOME_TIER_SURVIVES
        elif factor_score >= self.CAPITAL_IMPAIRMENT_THRESHOLD:
            return OUTCOME_TIER_IMPAIRED
        else:
            return OUTCOME_TIER_BROKEN
        
    def run(self, cre_generation: CREGenerationResult) -> FTResult:
        """
        Execute financial translation pipeline.

        Evaluates scenarios, computes valuations, and generates structured summary.

        Args:
            cre_generation: CRE generation result with scenarios and metadata

        Returns:
            FTResult with valuations and executive summary

        Raises:
            ValueError: If cre_generation is None
        """
        if cre_generation is None:
            raise ValueError("cre_generation is required")
        cre_output = self._evaluate_and_aggregate(cre_generation.scenario_set, cre_generation.defaults_applied)
        summary_struct = self._generate_structured_summary(cre_output, cre_generation.claims or [], cre_generation.scenario_set)
        return FTResult(scenario_set=cre_generation.scenario_set, ft_output=cre_output, structured_summary=summary_struct)
