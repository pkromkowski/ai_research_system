"""Shared test fixtures for thesis validation tests."""
import pytest
from unittest.mock import MagicMock

from model.core.types import (
    CT_NODE_ASSUMPTION, CT_NODE_DRIVER, CT_NODE_OUTCOME, 
    NDGNode, NDGEdge, FragilityMetrics, NDGOutput,
    RedTeamChallenge, RedTeamOutput,
    Scenario, CREScenarioSet, CREGenerationResult,
    FTOutput, FTResult, ValuationResult,
    ThesisValidityOutput,
    ThesisQuantitativeContext,
    HistoricalAnalog, FailureMode, RelevanceScoring,
    IHLEOutput, HalfLifeEstimate, MonitoringCadence, RegimeSensitivity,
    AssumptionDecayRate, PathDecayScore, AggregationDiagnosticsOutput
)


@pytest.fixture
def sample_thesis_narrative():
    """Sample investment thesis text."""
    return """
    Company XYZ is a leading SaaS provider in the enterprise software market.
    The company has demonstrated strong revenue growth of 40% YoY and is expanding
    into adjacent markets. Management has a track record of execution and the market
    opportunity is large. The stock should reach $200 in the next 18 months as
    the company scales and improves margins.
    """


@pytest.fixture
def sample_company_context():
    """Sample company context."""
    return """
    XYZ Corp (Ticker: XYZ) is an enterprise software company founded in 2015.
    The company provides cloud-based solutions for mid-market businesses.
    Current market cap: $5B, Revenue: $500M (TTM), Employees: 1000.
    Sector: Technology, Industry: Software - Application.
    """


@pytest.fixture
def sample_quantitative_context():
    """Sample quantitative context."""
    return ThesisQuantitativeContext(
        stock_ticker='XYZ',
        data_as_of='2026-02-01',
        revenue_growth_yoy=0.40,
        earnings_growth_yoy=0.35,
        gross_margin_current=0.75,
        operating_margin_current=0.20,
        pe_current=45.0,
        forward_pe=35.0
    )


@pytest.fixture
def sample_ndg_nodes():
    """Sample NDG nodes."""
    return [
        NDGNode(
            id='node_1',
            claim='Strong revenue growth of 40% YoY',
            node_type=CT_NODE_DRIVER,
            dependencies=[],
            control='Company',
            nature='Execution',
            time_sensitivity='Short',
            confidence=0.4,
            evidence_strength=0.7
        ),
        NDGNode(
            id='node_2',
            claim='Market opportunity is large',
            node_type=CT_NODE_ASSUMPTION,
            dependencies=[],
            control='Industry',
            nature='Structural',
            time_sensitivity='Long',
            confidence=0.3,
            evidence_strength=0.6
        ),
        NDGNode(
            id='node_3',
            claim='Stock reaches $200 in 18 months',
            node_type=CT_NODE_OUTCOME,
            dependencies=['node_1', 'node_2'],
            control='Macro',
            nature='Cyclical',
            time_sensitivity='Medium',
            confidence=0.3,
            evidence_strength=0.5
        )
    ]


@pytest.fixture
def sample_ndg_edges():
    """Sample NDG edges."""
    return [
        NDGEdge(source_id='node_1', target_id='node_3', relationship='ENABLES'),
        NDGEdge(source_id='node_2', target_id='node_1', relationship='REQUIRES')
    ]


@pytest.fixture
def sample_fragility_metrics():
    """Sample fragility metrics."""
    return FragilityMetrics(
        assumption_load=2,
        avg_assumptions_per_outcome=2.0,
        single_point_failures=['node_2'],
        path_concentration={'node_1': 2, 'node_2': 3, 'node_3': 1},
        max_graph_depth=2,
        fragility_score=0.45,
        feedback_loops=[]
    )


@pytest.fixture
def sample_ndg_output(sample_ndg_nodes, sample_ndg_edges, sample_fragility_metrics):
    """Sample NDG output."""
    return NDGOutput(
        stock_ticker='XYZ',
        thesis_text='Sample thesis',
        nodes=sample_ndg_nodes,
        edges=sample_ndg_edges,
        fragility_metrics=sample_fragility_metrics,
        total_confidence=1.0,
        confidence_sum=1.0,
        confidence_consistent=True,
        summary_text='Fragility: 0.45; nodes: 3; avg_evidence: 0.60'
    )


@pytest.fixture
def sample_red_team_challenges():
    """Sample red team challenges."""
    return [
        RedTeamChallenge(
            node_id='node_1',
            assumption_text='Strong revenue growth of 40% YoY',
            historical_precedent=HistoricalAnalog(
                case_name='Similar SaaS slowdown',
                assumption_type='Growth persistence',
                failure_mode='Market saturation',
                context='Previous company saw growth decline after market saturation',
                relevance_score=0.8,
                relevance_reasoning='Similar market dynamics'
            ),
            failure_mechanism=FailureMode(
                category='Demand shock',
                description='Market saturation reduces growth',
                early_warnings=['Slowing user growth', 'Increased churn'],
                taxonomy_match=True,
                category_confidence=0.8,
                is_downside_transferable=True
            ),
            relevance=RelevanceScoring(
                business_model_similarity=0.8,
                competitive_structure=0.7,
                balance_sheet_flexibility=0.6,
                regulatory_environment=0.5,
                cycle_position=0.7,
                overall_relevance=0.8,
                justification='Strong market similarities'
            ),
            early_warning_indicators=['Slowing user growth', 'Increased churn'],
            challenge_text='Revenue growth may slow due to market saturation',
            severity='high',
            suspected_value_exposure='HIGH'
        ),
        RedTeamChallenge(
            node_id='node_2',
            assumption_text='Market opportunity is large',
            historical_precedent=HistoricalAnalog(
                case_name='TAM overestimation case',
                assumption_type='Market sizing',
                failure_mode='TAM overestimation',
                context='Previous companies overestimated addressable market',
                relevance_score=0.6,
                relevance_reasoning='Common market sizing error'
            ),
            failure_mechanism=FailureMode(
                category='Market sizing error',
                description='Addressable market smaller than expected',
                early_warnings=['Market penetration plateau', 'Competitor saturation'],
                taxonomy_match=True,
                category_confidence=0.7,
                is_downside_transferable=True
            ),
            relevance=RelevanceScoring(
                business_model_similarity=0.6,
                competitive_structure=0.6,
                balance_sheet_flexibility=0.5,
                regulatory_environment=0.5,
                cycle_position=0.6,
                overall_relevance=0.6,
                justification='Moderate market similarities'
            ),
            early_warning_indicators=['Market penetration plateau', 'Competitor saturation'],
            challenge_text='Market opportunity may be overestimated',
            severity='medium',
            suspected_value_exposure='MEDIUM'
        )
    ]


@pytest.fixture
def sample_red_team_output(sample_red_team_challenges):
    """Sample red team output."""
    return RedTeamOutput(
        stock_ticker='XYZ',
        challenges=sample_red_team_challenges,
        high_severity_count=1,
        medium_severity_count=1,
        low_severity_count=0,
        summary_text='Generated 2 challenges'
    )


@pytest.fixture
def sample_scenarios():
    """Sample scenarios."""
    return [
        Scenario(
            name='Base Case',
            description='Base case scenario with moderate growth',
            impact='20% upside',
            stressed_assumptions={'revenue_growth': 0.40, 'margin': 0.20},
            plausibility_weight=0.5
        ),
        Scenario(
            name='Bull Case',
            description='Bull case scenario with strong growth',
            impact='50% upside',
            stressed_assumptions={'revenue_growth': 0.50, 'margin': 0.25},
            plausibility_weight=0.3
        ),
        Scenario(
            name='Bear Case',
            description='Bear case scenario with weak growth',
            impact='20% downside',
            stressed_assumptions={'revenue_growth': 0.20, 'margin': 0.15},
            plausibility_weight=0.2
        )
    ]


@pytest.fixture
def sample_valuation_results():
    """Sample valuation results."""
    return [
        ValuationResult(
            scenario_name='Base Case',
            valuation_change=0.20,
            outcome_tier='SURVIVES',
            narrative_consistent=True,
            margin_of_safety=0.15,
            plausibility_weight=0.5
        ),
        ValuationResult(
            scenario_name='Bull Case',
            valuation_change=0.67,
            outcome_tier='SURVIVES',
            narrative_consistent=True,
            margin_of_safety=0.40,
            plausibility_weight=0.3
        ),
        ValuationResult(
            scenario_name='Bear Case',
            valuation_change=-0.20,
            outcome_tier='IMPAIRED',
            narrative_consistent=False,
            margin_of_safety=-0.05,
            plausibility_weight=0.2
        )
    ]


@pytest.fixture
def sample_ft_output(sample_valuation_results):
    """Sample financial translation output."""
    return FTOutput(
        scenario_results=sample_valuation_results
    )


@pytest.fixture
def sample_validity_output():
    """Sample validity output."""
    return ThesisValidityOutput(
        stock_ticker='XYZ',
        status='Valid',
        reasons=['Strong scenario survival', 'Low fragility'],
        dominant_failure_modes=[],
        required_conditions=['Maintain revenue growth', 'Expand margins'],
        key_contradictions=['Market saturation risk'],
        survival_rate=0.67,
        fragility_score=0.45
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM structured response."""
    return {
        'claims': [
            {
                'claim': 'Strong revenue growth',
                'type': 'DRIVER',
                'dependencies': []
            }
        ],
        'metrics': {'revenue_growth': 0.40}
    }


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock = MagicMock()
    mock.messages.create.return_value = MagicMock(
        content=[MagicMock(input={'claims': [], 'metrics': {}})]
    )
    return mock


# --- Integration fixtures for orchestration integration test ---
@pytest.fixture
def sample_cre_scenario_set(sample_scenarios):
    """CREScenarioSet with provided scenarios."""
    return CREScenarioSet(
        stock_ticker='XYZ',
        scenarios=sample_scenarios,
        rejected_scenarios=[],
        base_metrics={'revenue': 0.40},
        bounds={},
        generated_raw=[],
        summary_text='CRE generated scenarios',
        total_duration_ms=12.3
    )

@pytest.fixture
def sample_cre_generation_result(sample_cre_scenario_set):
    """CREGenerationResult using the scenario set."""
    return CREGenerationResult(
        scenario_set=sample_cre_scenario_set,
        claims=['revenue, margin'],
        defaults_applied=[]
    )

@pytest.fixture
def sample_ft_result(sample_cre_scenario_set, sample_ft_output):
    """FTResult with embedded cre_output for backward compat with orchestrator."""
    ft_result = FTResult(scenario_set=sample_cre_scenario_set, ft_output=sample_ft_output)
    # Orchestrator expects attribute `cre_output` on the FTResult; keep both for compatibility
    setattr(ft_result, 'cre_output', sample_ft_output)
    return ft_result

@pytest.fixture
def sample_ihle_output():
    """IHLEOutput with a minimal, realistic payload."""
    half_life = HalfLifeEstimate(
        estimated_half_life_months=12.0,
        primary_decay_drivers=['node_1'],
        decay_trend='Stable',
        time_to_first_broken=6.0,
        regime_adjusted=False
    )
    regime = RegimeSensitivity(
        regime_tags=['Macro'],
        current_regime_state='Stable',
        regime_alignment=1.0,
        adjustment_factor=1.0,
        adjustment_reasoning='No regime adjustments'
    )
    cadence = MonitoringCadence(
        half_life_months=12.0,
        recommended_frequency='Quarterly',
        next_review_date='2026-03-01',
        priority_level='Medium',
        review_justification='Standard monitoring'
    )
    return IHLEOutput(
        stock_ticker='XYZ',
        ndg_version='v1',
        analysis_timestamp='2026-02-01T00:00:00Z',
        half_life_estimate=half_life,
        decay_rates=[],
        path_scores=[],
        regime_sensitivity=regime,
        monitoring_cadence=cadence,
        total_assumptions=3,
        assumptions_decaying=1,
        fastest_decay_rate=0.20,
        slowest_decay_rate=0.01,
        cre_scenario_weights_update={},
        red_team_relevance_boost=[],
        half_life_signals=None,
        summary_text='Sample IHLE summary',
        node_contributions=[],
        monte_carlo=None,
        sensitivity_analysis=None
    )

@pytest.fixture
def sample_aggregation_output():
    """AggregationDiagnosticsOutput with basic diagnostics."""
    return AggregationDiagnosticsOutput(
        stock_ticker='XYZ',
        total_scenarios=3,
        broken_count=1,
        impaired_count=1,
        survives_count=1,
        scenario_survival_fraction=0.33,
        weighted_survival_rate=0.50,
        tail_loss_percentile=-0.20,
        raw_fragility_proxy=0.45,
        impaired_scenarios=['Bear Case'],
        broken_scenarios=['Broken Case'],
        ihle_half_life_months=12.0,
        primary_decay_drivers=['node_1'],
        recommended_cadence=None,
        top_red_challenges=[],
        summary_text='Aggregated summary',
        failure_articulation=None,
        scenario_ranking=['Bull Case','Base Case','Bear Case'],
        comparable_scores={},
        research_packet={}
    )
