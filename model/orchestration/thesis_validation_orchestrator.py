import logging
from typing import Optional
from datetime import datetime

from model.core.types import (
    FinalThesisReport,
    ThesisQuantitativeContext,
    NDGOutput,
    RedTeamOutput,
    CREGenerationResult,
    FTResult,
    FTOutput,
    ThesisValidityOutput,
    IHLEOutput,
    AggregationDiagnosticsOutput,
)
from model.thesis_agents.red_team import AIRedTeamWithMemory
from model.thesis_agents.half_life_estimator import IdeaHalfLifeEstimator
from model.thesis_agents.financial_translation import FinancialTranslation
from model.thesis_agents.thesis_validity_evaluator import ThesisValidityEvaluator
from model.thesis_agents.narrative_decomposition import NarrativeDecompositionGraph
from model.thesis_agents.counterfactual_research import CounterfactualResearchEngine
from model.thesis_agents.aggregation_diagnostics import AggregationDiagnostics

logger = logging.getLogger(__name__)


class ThesisValidationOrchestrator:
    """
    Orchestrates end-to-end thesis validation pipeline.

    Pipeline stages:
    1. NDG: Narrative decomposition and fragility analysis
    2. Red Team: Adversarial challenge generation
    3. CRE: Counterfactual scenario generation
    4. FT: Financial translation and valuation
    5. Validity: Rule-based thesis status evaluation
    6. IHLE: Half-life estimation and monitoring cadence
    7. Aggregation: Final diagnostics and comparability scores

    Accepts optional Stage 1 quantitative context from StockAnalyticsOrchestrator.
    """
    
    def __init__(self, stock_ticker: str):
        self.stock_ticker = stock_ticker
        logger.debug("Initializing ThesisValidationOrchestrator", extra={"ticker": stock_ticker})
        self.ndg = NarrativeDecompositionGraph(stock_ticker=stock_ticker)
        self.red_team = AIRedTeamWithMemory(stock_ticker=stock_ticker)
        self.cre = CounterfactualResearchEngine(stock_ticker=stock_ticker)
        self.ft = FinancialTranslation(stock_ticker=stock_ticker)
        self.ihle = IdeaHalfLifeEstimator(stock_ticker=stock_ticker)
        self.validity_evaluator = ThesisValidityEvaluator(stock_ticker=stock_ticker)
        self.aggregator = AggregationDiagnostics(stock_ticker=stock_ticker)
    
    def run(
        self,
        thesis_narrative: str,
        company_context: str,
        quantitative_context: Optional[ThesisQuantitativeContext] = None
    ) -> FinalThesisReport:
        """
        Execute complete thesis validation pipeline.

        Args:
            thesis_narrative: Investment thesis narrative
            company_context: Company description
            quantitative_context: Optional Stage 1 metrics from StockAnalyticsOrchestrator

        Returns:
            FinalThesisReport with validation results and diagnostics

        Raises:
            RuntimeError: If any required pipeline stage fails
        """

        logger.info("Running thesis validation pipeline", extra={"ticker": self.stock_ticker})
        if quantitative_context:
            logger.info("Loaded Stage 1 context", extra={"data_as_of": quantitative_context.data_as_of})

        logger.info("[1/7] Running NDG...")
        try:
            ndg_output: NDGOutput = self.ndg.run(
                thesis_narrative=thesis_narrative,
                company_context=company_context,
                quantitative_context=quantitative_context,
            )
            logger.info("NDG complete", extra={"nodes": len(ndg_output.nodes), "edges": len(ndg_output.edges)})
        except Exception as e:
            logger.exception("NDG failed")
            raise RuntimeError(f"NDG failed: {e}") from e

        logger.info("[2/7] Running Red Team...")
        try:
            red_team_output: RedTeamOutput = self.red_team.run(ndg=ndg_output, company_context=company_context)
            logger.info("Red Team complete", extra={"challenges": len(red_team_output.challenges)})
        except Exception as e:
            logger.exception("Red Team failed")
            raise RuntimeError(f"Red Team failed: {e}") from e

        logger.info("[3/7] Running CRE...")
        try:
            cre_generation: CREGenerationResult = self.cre.run(
                ndg=ndg_output,
                red_team=red_team_output,
                company_context=company_context,
                quantitative_context=quantitative_context,
            )
            logger.info("CRE complete", extra={"scenarios": len(cre_generation.scenario_set.scenarios)})
        except Exception as e:
            logger.exception("CRE failed")
            raise RuntimeError(f"CRE failed: {e}") from e

        logger.info("[4/7] Running FT...")
        try:
            ft_result: FTResult = self.ft.run(cre_generation)
            ft_output: FTOutput = ft_result.cre_output
            logger.info("FT complete", extra={"valuations": len(ft_output.scenario_results)})
        except Exception as e:
            logger.exception("FT failed")
            raise RuntimeError(f"FT failed: {e}") from e

        logger.info("[5/7] Running Validity Evaluator...")
        try:
            validity_output: ThesisValidityOutput = self.validity_evaluator.run(ft_output, red_team_output, ndg_output)
            logger.info("Validity complete", extra={"status": validity_output.status})
        except Exception as e:
            logger.exception("Validity Evaluator failed")
            raise RuntimeError(f"Validity Evaluator failed: {e}") from e

        logger.info("[6/7] Running IHLE...")
        try:
            ihle_output: IHLEOutput = self.ihle.run(ndg=ndg_output, current_macro_context=None)
            logger.info("IHLE complete", extra={"half_life_months": ihle_output.half_life_estimate.estimated_half_life_months})
        except Exception as e:
            logger.exception("IHLE failed")
            ihle_output = None

        logger.info("[7/7] Running Aggregation...")
        try:
            aggregation_output: AggregationDiagnosticsOutput = self.aggregator.run(
                ihle_output=ihle_output,
                ft_output=ft_output,
                red_team_output=red_team_output,
                validity_output=validity_output,
            )
            logger.info("Aggregation complete")
        except Exception as e:
            logger.warning("Aggregation failed: %s", e)
            aggregation_output = None

        report = FinalThesisReport(
            stock=self.stock_ticker,
            submission_date=datetime.now().strftime("%Y-%m-%d"),
            narrative=thesis_narrative,
            survival_rate=validity_output.survival_rate,
            fragility_score=ndg_output.fragility_metrics.fragility_score,
            dominant_failure_modes=(validity_output.dominant_failure_modes or []),
            suggested_position_size_factor=1.0,
            half_life_months=(
                ihle_output.half_life_estimate.estimated_half_life_months
                if ihle_output else self.ihle.BASELINE_HALF_LIFE_MONTHS
            ),
            key_risks=[c.challenge_text for c in red_team_output.challenges[:5]],
            actionable_notes=[c.challenge_text for c in red_team_output.challenges[:3]],
            detailed_components={
                "ndg": ndg_output,
                "ft": ft_output,
                "red_team": red_team_output,
                "ihle": ihle_output,
                "validity": validity_output,
                "aggregation": aggregation_output,
            },
            quantitative_context=quantitative_context,
        )
        logger.info("Pipeline completed")
        return report
