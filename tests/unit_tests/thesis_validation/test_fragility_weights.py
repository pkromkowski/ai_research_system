import pytest

from model.thesis_agents.narrative_decomposition import NarrativeDecompositionGraph
from model.core.types import NDGNode, NDGEdge, NODE_TYPE_ASSUMPTION, NODE_TYPE_DRIVER, NODE_TYPE_OUTCOME


def make_node(id, claim, node_type, control="Company", nature="Structural", time_sensitivity="Medium", evidence_strength=1.0, confidence=0.0):
    return NDGNode(
        id=id,
        claim=claim,
        node_type=node_type,
        dependencies=[],
        control=control,
        nature=nature,
        time_sensitivity=time_sensitivity,
        evidence_sources=[],
        evidence_strength=evidence_strength,
        contradicting_evidence=[],
        confidence=confidence,
        confidence_basis=""
    )


def test_weighted_assumption_load_affects_component():
    ndg = NarrativeDecompositionGraph(stock_ticker="TEST")

    # Scenario A: two company-controlled structural assumptions
    n1 = make_node('a1', 'assump1', NODE_TYPE_ASSUMPTION, control='Company', nature='Structural')
    n2 = make_node('a2', 'assump2', NODE_TYPE_ASSUMPTION, control='Company', nature='Structural')
    outcome = make_node('o1', 'out', NODE_TYPE_OUTCOME, control='Company', nature='Structural')
    nodes = [n1, n2, outcome]
    edges = [NDGEdge(source_id='a1', target_id='o1'), NDGEdge(source_id='a2', target_id='o1')]

    fm_company = ndg.compute_fragility(nodes, edges)

    # Scenario B: same topology but macro-controlled assumptions (higher risk weight)
    m1 = make_node('b1', 'assump1', NODE_TYPE_ASSUMPTION, control='Macro', nature='Cyclical')
    m2 = make_node('b2', 'assump2', NODE_TYPE_ASSUMPTION, control='Macro', nature='Cyclical')
    nodes_b = [m1, m2, outcome]
    edges_b = [NDGEdge(source_id='b1', target_id='o1'), NDGEdge(source_id='b2', target_id='o1')]

    fm_macro = ndg.compute_fragility(nodes_b, edges_b)

    assert fm_company.fragility_components['assumption_load_weighted'] < fm_macro.fragility_components['assumption_load_weighted']


def test_spof_detection_with_control_weighting():
    ndg = NarrativeDecompositionGraph(stock_ticker="TEST")

    # Setup: assumption -> driver -> outcome
    a = make_node('a', 'assump', NODE_TYPE_ASSUMPTION, control='Company')

    # Driver with same evidence but different control
    d_company = make_node('d1', 'driver', NODE_TYPE_DRIVER, control='Company', evidence_strength=0.4)
    d_exog = make_node('d2', 'driver', NODE_TYPE_DRIVER, control='Exogenous', evidence_strength=0.4)

    o = make_node('o', 'outcome', NODE_TYPE_OUTCOME, control='Company')

    nodes1 = [a, d_company, o]
    edges1 = [NDGEdge(source_id='a', target_id='d1'), NDGEdge(source_id='d1', target_id='o')]
    fm1 = ndg.compute_fragility(nodes1, edges1)

    nodes2 = [a, d_exog, o]
    edges2 = [NDGEdge(source_id='a', target_id='d2'), NDGEdge(source_id='d2', target_id='o')]
    fm2 = ndg.compute_fragility(nodes2, edges2)

    # d_company should not be flagged as SPOF (company control reduces risk)
    assert 'd1' not in fm1.single_point_failures
    # d_exog should be flagged
    assert 'd2' in fm2.single_point_failures
