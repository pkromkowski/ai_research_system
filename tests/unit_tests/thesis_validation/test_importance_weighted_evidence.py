from model.thesis_agents.narrative_decomposition import NarrativeDecompositionGraph
from model.core.types import NDGNode, NDGEdge, NODE_TYPE_ASSUMPTION, NODE_TYPE_DRIVER, NODE_TYPE_OUTCOME


def make_node(id, claim, node_type, control="Company", nature="Structural", time_sensitivity="Medium", evidence_strength=1.0):
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
        confidence=0.0,
        confidence_basis=""
    )


def test_core_driver_low_evidence_flagged_despite_high_avg():
    ndg = NarrativeDecompositionGraph(stock_ticker="TEST")

    # One core driver low evidence
    core = make_node('d_core', 'core driver', NODE_TYPE_DRIVER, control='Company', nature='Structural', evidence_strength=0.2)
    # Ten peripheral assumptions high evidence
    periph = [make_node(f'a{i}', f'assump {i}', NODE_TYPE_ASSUMPTION, control='Company', nature='Structural', evidence_strength=0.9) for i in range(10)]
    outcome = make_node('o', 'outcome', NODE_TYPE_OUTCOME)

    nodes = [core] + periph + [outcome]
    edges = [NDGEdge(source_id=core.id, target_id=outcome.id)] + [NDGEdge(source_id=p.id, target_id=outcome.id) for p in periph]

    fm = ndg.compute_fragility(nodes, edges)

    # Average unweighted evidence might look OK but core should be flagged as critical low
    assert 'd_core' in fm.critical_low_evidence_nodes


def test_importance_weighted_evidence_reduces_false_safety():
    ndg = NarrativeDecompositionGraph(stock_ticker="TEST")

    core = make_node('d_core', 'core driver', NODE_TYPE_DRIVER, control='Macro', nature='Cyclical', evidence_strength=0.2)
    periph = [make_node(f'a{i}', f'assump {i}', NODE_TYPE_ASSUMPTION, control='Company', nature='Structural', evidence_strength=0.9) for i in range(10)]
    outcome = make_node('o', 'outcome', NODE_TYPE_OUTCOME)

    nodes = [core] + periph + [outcome]
    edges = [NDGEdge(source_id=core.id, target_id=outcome.id)] + [NDGEdge(source_id=p.id, target_id=outcome.id) for p in periph]

    fm = ndg.compute_fragility(nodes, edges)

    assert fm.fragility_components['avg_importance_weighted_evidence'] < 0.6
