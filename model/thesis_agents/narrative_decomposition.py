import json
import logging
import math
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any

from model.thesis_agents.pipeline_helpers import LLMHelperMixin
from model.core.types import (
    NDGNode, NDGEdge, FragilityMetrics, FeedbackLoop, NDGOutput, ThesisQuantitativeContext, ThesisInput,
    NODE_TYPE_ASSUMPTION as CT_NODE_ASSUMPTION,
    NODE_TYPE_DRIVER as CT_NODE_DRIVER,
    NODE_TYPE_OUTCOME as CT_NODE_OUTCOME,
)
from model.prompts.thesis_validation_prompts import (
    NDG_PARSE_THESIS_PROMPT,
    NDG_BUILD_DAG_PROMPT,
    NDG_CLASSIFY_ASSUMPTIONS_PROMPT,
    NDG_MAP_EVIDENCE_PROMPT,
    NDG_DISTRIBUTE_CONFIDENCE_PROMPT,
)
from model.prompts.output_schemas import (
    NDG_PARSE_THESIS_SCHEMA,
    NDG_BUILD_DAG_SCHEMA,
    NDG_CLASSIFY_ASSUMPTIONS_SCHEMA,
    NDG_MAP_EVIDENCE_SCHEMA,
    NDG_DISTRIBUTE_CONFIDENCE_SCHEMA
)

logger = logging.getLogger(__name__)


class NarrativeDecompositionGraph(LLMHelperMixin):
    """
    Decomposes an investment thesis into a structured causal graph (DAG) and
    produces compact, explainable diagnostics useful for thesis validation.

    Key characteristics:
    - Uses structured tool-use LLM calls exclusively to ensure deterministic,
      machine-readable outputs (no free-text fallbacks).
    - Does not write persistent intermediate artifacts; operations are
      side-effect free other than returned dataclasses.
    - Graph traversals use memoized algorithms for performance on larger graphs.

    Output: `NDGOutput` containing nodes, edges, `FragilityMetrics` (with a
    per-component breakdown), and `total_confidence`.
    """
    
    # Token limits per step (sized for expected output complexity)
    MAX_TOKENS_PARSE: int = 2000
    MAX_TOKENS_DAG: int = 3000
    MAX_TOKENS_CLASSIFY: int = 2000
    MAX_TOKENS_EVIDENCE: int = 3000
    MAX_TOKENS_CONFIDENCE: int = 2000
    
    # Temperature settings (0.0 for deterministic, higher for evidence search)
    TEMPERATURE_PARSE: float = 0.0
    TEMPERATURE_DAG: float = 0.0
    TEMPERATURE_CLASSIFY: float = 0.0
    TEMPERATURE_EVIDENCE: float = 0.1  # Slightly higher for evidence exploration
    TEMPERATURE_CONFIDENCE: float = 0.0
    
    # Confidence validation
    # Tolerance for how far confidence sum can deviate from 1.0
    CONFIDENCE_SUM_TOLERANCE: float = 0.15

    # Automatic normalization configuration
    # When True, NDG will normalize node confidences (if inconsistent) automatically
    NORMALIZE_CONFIDENCES_ON_RED_TEAM: bool = False
    NORMALIZE_METHOD: str = 'proportional'
    
    # Evidence quality thresholds
    # Nodes with evidence_strength below this are flagged as weak
    EVIDENCE_STRENGTH_WEAK_THRESHOLD: float = 0.4
    
    # Single-Point-of-Failure (SPOF) detection
    # Node must have at least this many paths through it to be considered for SPOF
    SPOF_MIN_PATH_COUNT: int = 2
    # Evidence strength must be below this for high-traffic node to be SPOF
    SPOF_EVIDENCE_THRESHOLD: float = 0.5
    
    # Fragility scoring configuration
    DEPTH_PENALTY_THRESHOLD: int = 5
    DEPTH_PENALTY_PER_LAYER: float = 0.1

    # Maximum contribution of each component to fragility score (sum to 1.0)
    MAX_FRAGILITY_ASSUMPTION_LOAD: float = 0.3
    MAX_FRAGILITY_GRAPH_DEPTH: float = 0.3
    MAX_FRAGILITY_SPOF: float = 0.2
    MAX_FRAGILITY_EVIDENCE: float = 0.2

    # Feedback loop handling thresholds (stage 1+2)
    LOOP_EDGE_STRENGTH_THRESHOLD: float = 0.6
    LOOP_EVIDENCE_THRESHOLD: float = 0.6
    LOOP_REINFORCEMENT_BONUS_MAX: float = 0.1
    LOOP_WEAKNESS_PENALTY_MAX: float = 0.1

    # Weight maps to adjust fragility by controllability/nature/time (conservative defaults)
    CONTROL_WEIGHTS: Dict[str, float] = {
        "Company": 0.6,
        "Industry": 0.9,
        "Macro": 1.0,
        "Exogenous": 1.2
    }
    NATURE_WEIGHTS: Dict[str, float] = {
        "Structural": 0.8,
        "Cyclical": 1.0,
        "Execution": 1.05
    }
    TIME_WEIGHTS: Dict[str, float] = {
        "Short": 0.9,
        "Medium": 1.0,
        "Long": 1.05
    }

    # Importance thresholds for critical evidence detection
    CRITICAL_IMPORTANCE_THRESHOLD: float = 0.5  # normalized importance (0-1)
    CRITICAL_EVIDENCE_THRESHOLD: float = 0.5  # evidence strength below which node is critical
    
    # Assumption load scaling factor (per outcome)
    ASSUMPTION_LOAD_SCALE_FACTOR: float = 0.05
    
    # SPOF contribution per failure
    SPOF_CONTRIBUTION_PER_FAILURE: float = 0.1

    # Node type constants (instance-level override allowed via constructor)
    NODE_TYPE_ASSUMPTION: str = CT_NODE_ASSUMPTION
    NODE_TYPE_DRIVER: str = CT_NODE_DRIVER
    NODE_TYPE_OUTCOME: str = CT_NODE_OUTCOME

    def __init__(self, stock_ticker: str):
        """
        Initialize NDG engine.

        Args:
            stock_ticker: Company ticker symbol for context.
        """
        if not stock_ticker:
            raise ValueError("stock_ticker is required ")

        self.stock_ticker = stock_ticker

        logger.info(f"Initialized NarrativeDecompositionGraph for {stock_ticker}")


    # --- 2.1 Thesis parsing & claim identification ---
    def parse_thesis(self, thesis_narrative: str) -> Dict[str, Any]:
        """Parse an analyst thesis into structured claims and optional metrics.

        Normalizes legacy field names (e.g., ``time_horizon`` -> ``time_sensitivity``)
        and ensures each claim includes a ``directionality`` key.

        Returns a dict with keys ``claims`` and ``metrics`` suitable for downstream use.
        """
        prompt = self.format_prompt(
            NDG_PARSE_THESIS_PROMPT,
            stock_ticker=self.stock_ticker,
            thesis_narrative=thesis_narrative
        )

        result = self._call_llm_structured(
            prompt,
            NDG_PARSE_THESIS_SCHEMA,
            max_tokens=self.MAX_TOKENS_PARSE,
            temperature=self.TEMPERATURE_PARSE,
        )

        claims = result.get("claims", [])
        metrics = result.get("metrics", {}) or {}

        # Normalize parser outputs for downstream consistency
        for c in claims:
            # time_horizon legacy name -> time_sensitivity
            if 'time_horizon' in c and 'time_sensitivity' not in c:
                c['time_sensitivity'] = c.pop('time_horizon')
            # Ensure directionality field exists (may be empty)
            if 'directionality' not in c:
                c['directionality'] = ''

        logger.info(f"Parsed thesis: {len(claims)} claims; metrics keys: {list(metrics.keys())}")
        return {
            "claims": claims,
            "metrics": metrics
        }

    
    # --- 2.2 Causal graph construction ---
    def build_dag(self, claims: List[Dict]) -> Tuple[List[NDGNode], List[NDGEdge]]:
        """Construct a DAG from parsed claims.

        The LLM returns node and edge records; this method validates acyclicity
        and converts records into `NDGNode`/`NDGEdge` dataclasses. Parsed fields
        such as ``directionality`` and ``time_sensitivity`` are preserved on
        nodes when available.

        Returns a tuple: (nodes, edges).
        """
        prompt = self.format_prompt(
            NDG_BUILD_DAG_PROMPT,
            stock_ticker=self.stock_ticker,
            claims_json=json.dumps(claims, indent=2)
        )
        
        graph_data = self._call_llm_structured(
            prompt,
            NDG_BUILD_DAG_SCHEMA,
            max_tokens=self.MAX_TOKENS_DAG,
            temperature=self.TEMPERATURE_DAG,
        )

        if not self._is_dag(graph_data['edges']):
            logger.warning("Cycles detected in graph - resolved weakest edges")
            graph_data = self._break_cycles(graph_data)

        # preserve parsed claim fields
        claim_map = {c['claim']: c for c in claims}

        nodes = [
            NDGNode(
                id=n['id'],
                claim=n['claim'],
                node_type=n['node_type'],
                dependencies=n['dependencies'],
                control="",  # Set in step 2.3
                nature="",
                time_sensitivity=n.get('time_sensitivity', claim_map.get(n['claim'], {}).get('time_sensitivity', '')),
                directionality=n.get('directionality', claim_map.get(n['claim'], {}).get('directionality', ''))
            )
            for n in graph_data['nodes']
        ]
        
        edges = [
            NDGEdge(
                source_id=e['source_id'],
                target_id=e['target_id'],
                relationship=e['relationship'],
                strength=e['strength']
            )
            for e in graph_data['edges']
        ]
        
        return nodes, edges
    
    def _is_dag(self, edges: List[Dict]) -> bool:
        """
        Check if graph is acyclic using DFS cycle detection.
        
        Args:
            edges: List of edge dictionaries with source_id and target_id
            
        Returns:
            True if graph is a valid DAG (no cycles), False otherwise
        """
        graph = defaultdict(list)
        nodes_set = set()
        for edge in edges:
            graph[edge['source_id']].append(edge['target_id'])
            nodes_set.add(edge['source_id'])
            nodes_set.add(edge['target_id'])
        
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes (sources and targets)
        for node in nodes_set:
            if node not in visited:
                if has_cycle(node):
                    return False
        return True
    
    def _break_cycles(self, graph_data: Dict) -> Dict:
        """
        Remove weakest edge in each cycle to restore DAG property.
        
        Simple heuristic: sort edges by strength ascending and remove
        the weakest edges until the graph becomes acyclic.
        
        Args:
            graph_data: Dictionary with 'nodes' and 'edges' keys
            
        Returns:
            Modified graph_data with cycles broken
        """
        edges_sorted = sorted(graph_data['edges'], key=lambda e: e['strength'])
        
        for edge in edges_sorted:
            test_edges = [e for e in graph_data['edges'] if e != edge]
            if self._is_dag(test_edges):
                edge_desc = f"{edge['source_id']} → {edge['target_id']}"
                logger.info(f"Removed cycle edge: {edge_desc}")
                graph_data['edges'] = test_edges
                return graph_data
        
        return graph_data
    
    # --- 2.3 Assumption localization & ownership ---
    def classify_assumptions(self, nodes: List[NDGNode]) -> List[NDGNode]:
        """Classify each node by ``control``, ``nature``, and ``time_sensitivity``.

        The method updates nodes in-place and returns the modified list.
        """
        nodes_json = json.dumps(
            [{'id': n.id, 'claim': n.claim, 'type': n.node_type} for n in nodes], 
            indent=2
        )
        prompt = self.format_prompt(
            NDG_CLASSIFY_ASSUMPTIONS_PROMPT,
            stock_ticker=self.stock_ticker,
            nodes_json=nodes_json
        )
        
        classifications = self._call_llm_structured(
            prompt,
            NDG_CLASSIFY_ASSUMPTIONS_SCHEMA,
            max_tokens=self.MAX_TOKENS_CLASSIFY,
            temperature=self.TEMPERATURE_CLASSIFY,
        )

        # Apply classifications to nodes
        class_map = {c['id']: c for c in classifications['classifications']}
        for node in nodes:
            if node.id in class_map:
                c = class_map[node.id]
                node.control = c['control']
                node.nature = c['nature']
                node.time_sensitivity = c['time_sensitivity']

        ambiguous = classifications.get('ambiguous_nodes', [])
        if ambiguous:
            logger.warning(f"Ambiguous classifications: {ambiguous}")

        logger.info("Classified assumptions")
        return nodes
    
    # --- 2.4 EVIDENCE MAPPING ---
    def map_evidence(
        self, 
        nodes: List[NDGNode], 
        company_context: str,
        quantitative_context: Optional[ThesisQuantitativeContext] = None
    ) -> List[NDGNode]:
        """Attach supporting and contradicting evidence to each node.

        Each node receives ``evidence_sources``, ``contradicting_evidence``, and
        an ``evidence_strength`` score (0-1) returned by the structured LLM call.
        """
        nodes_json = json.dumps(
            [{'id': n.id, 'claim': n.claim, 'type': n.node_type, 'control': n.control} for n in nodes], 
            indent=2
        )
        
        quant_context_str = ""
        if quantitative_context:
            quant_context_str = quantitative_context.to_prompt_context()
        
        prompt = self.format_prompt(
            NDG_MAP_EVIDENCE_PROMPT,
            stock_ticker=self.stock_ticker,
            company_context=company_context,
            quantitative_context=quant_context_str,
            nodes_json=nodes_json
        )
        
        evidence_data = self._call_llm_structured(
            prompt,
            NDG_MAP_EVIDENCE_SCHEMA,
            max_tokens=self.MAX_TOKENS_EVIDENCE,
            temperature=self.TEMPERATURE_EVIDENCE,
        )

        evidence_map = {e['id']: e for e in evidence_data['evidence_map']}
        weak_nodes = []
        total_evidence = 0.0

        for node in nodes:
            if node.id in evidence_map:
                e = evidence_map[node.id]
                node.evidence_sources = e.get('supporting_evidence', [])
                node.contradicting_evidence = e.get('contradicting_evidence', [])
                node.evidence_strength = e['evidence_strength']
                total_evidence += node.evidence_strength
                if node.evidence_strength < self.EVIDENCE_STRENGTH_WEAK_THRESHOLD:
                    weak_nodes.append(node.id)

        avg = total_evidence / len(nodes) if nodes else 0.0
        logger.info(f"Mapped evidence (avg strength: {avg:.2f}, weak_nodes={len(weak_nodes)})")
        return nodes
    
    # --- 2.5 CONFIDENCE ATTRIBUTION ---
    def distribute_confidence(
        self, 
        thesis_narrative: str, 
        nodes: List[NDGNode],
        normalize: bool = False,
        normalize_method: str = 'proportional'
    ) -> List[NDGNode]:
        """Allocate a confidence value (0-1) and short basis for each node.

        The LLM returns a confidence distribution and an optional ``total_confidence``
        value; the method updates nodes in-place and returns the node list.
        """ 
        nodes_json = json.dumps([{
            'id': n.id, 
            'claim': n.claim, 
            'evidence_strength': n.evidence_strength
        } for n in nodes], indent=2)
        
        prompt = self.format_prompt(
            NDG_DISTRIBUTE_CONFIDENCE_PROMPT,
            stock_ticker=self.stock_ticker,
            thesis_narrative=thesis_narrative,
            nodes_json=nodes_json,
        )
        
        conf_data = self._call_llm_structured(
            prompt,
            NDG_DISTRIBUTE_CONFIDENCE_SCHEMA,
            max_tokens=self.MAX_TOKENS_CONFIDENCE,
            temperature=self.TEMPERATURE_CONFIDENCE,
        )

        conf_map = {c['id']: c for c in conf_data['confidence_distribution']}
        for node in nodes:
            if node.id in conf_map:
                c = conf_map[node.id]
                node.confidence = c['confidence']
                node.confidence_basis = c['basis']

        total = conf_data.get('total_confidence', sum(n.confidence for n in nodes))
        # Record raw sum and whether it's within tolerance
        confidence_sum = total
        confidence_consistent = abs(confidence_sum - 1.0) <= self.CONFIDENCE_SUM_TOLERANCE

        if not confidence_consistent:
            logger.warning(f"Confidence sum is {confidence_sum:.2f}, expected ~1.0 (tolerance {self.CONFIDENCE_SUM_TOLERANCE})")

            if normalize:
                logger.info("Normalizing confidences using method: %s", normalize_method)
                norm_info = self.normalize_confidences(nodes, method=normalize_method)
                confidence_sum = norm_info.get('new_sum', sum(n.confidence for n in nodes))
                confidence_consistent = abs(confidence_sum - 1.0) <= self.CONFIDENCE_SUM_TOLERANCE
                logger.info("Normalization complete (new sum: %.3f)", confidence_sum)

        overconf_warnings = conf_data.get('high_confidence_low_evidence', [])
        if overconf_warnings:
            logger.warning(f"Overconfidence detected in {len(overconf_warnings)} nodes")

        self._last_confidence_sum = confidence_sum
        self._last_confidence_consistent = confidence_consistent

        logger.info("Distributed confidence across nodes")
        return nodes

    def normalize_confidences(self, nodes: List[NDGNode], method: str = 'proportional') -> Dict[str, Any]:
        """Normalize node confidence values in-place.

        Supported methods:
        - 'proportional' (default): divide each confidence by the total sum
        - 'clamp_proportional': clamp negatives to 0 then proportionalize

        Returns a dict with keys: 'method', 'original_sum', 'new_sum', 'adjustments'
        """
        original_sum = sum(max(0.0, n.confidence) for n in nodes)
        adjustments: Dict[str, Tuple[float, float]] = {}

        if original_sum == 0:
            # No confident signals; evenly distribute
            per = 1.0 / len(nodes) if nodes else 0.0
            for n in nodes:
                old = n.confidence
                n.confidence = per
                adjustments[n.id] = (old, n.confidence)
            new_sum = sum(n.confidence for n in nodes) if nodes else 0.0
            return {'method': method, 'original_sum': original_sum, 'new_sum': new_sum, 'adjustments': adjustments}

        if method == 'proportional' or method == 'clamp_proportional':
            if method == 'clamp_proportional':
                for n in nodes:
                    if n.confidence < 0:
                        n.confidence = 0.0
            for n in nodes:
                old = n.confidence
                n.confidence = n.confidence / original_sum
                adjustments[n.id] = (old, n.confidence)
            new_sum = sum(n.confidence for n in nodes)
            return {'method': method, 'original_sum': original_sum, 'new_sum': new_sum, 'adjustments': adjustments}

        # Unknown method: no-op
        return {'method': method, 'original_sum': original_sum, 'new_sum': original_sum, 'adjustments': adjustments}

    # --- Helper utilities for `run()` simplification ---
    def _warn_ambiguous_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and log ambiguous claims returned by the parser."""
        ambiguous = [c for c in claims if c.get('claim', '').startswith('AMBIGUOUS:')]
        if ambiguous:
            logger.warning(f"{len(ambiguous)} ambiguous claims flagged for review")
        return ambiguous

    def _compute_avg_evidence(self, nodes: List[NDGNode]) -> float:
        """Return average evidence_strength across provided nodes (0-1)."""
        if not nodes:
            return 0.0
        return sum(getattr(n, 'evidence_strength', 0.0) for n in nodes) / len(nodes)

    def _compute_total_confidence(self, nodes: List[NDGNode]) -> float:
        """Sum node confidences."""
        return sum(getattr(n, 'confidence', 0.0) for n in nodes)

    def _ensure_confidence_consistency(self, nodes: List[NDGNode]) -> Tuple[float, bool]:
        """Ensure confidence sum is within tolerance; optionally normalize and update metadata.

        Returns (total_confidence, is_consistent).
        """
        total_conf = self._compute_total_confidence(nodes)
        # Prefer last seen metadata when available
        previous_sum = getattr(self, '_last_confidence_sum', total_conf)
        is_consistent = getattr(self, '_last_confidence_consistent', abs(previous_sum - 1.0) <= self.CONFIDENCE_SUM_TOLERANCE)

        # If current total differs from previous metadata, use the computed total as truth
        if abs(total_conf - previous_sum) > 1e-9:
            previous_sum = total_conf
            is_consistent = abs(previous_sum - 1.0) <= self.CONFIDENCE_SUM_TOLERANCE

        # Optionally normalize if inconsistent and configured
        if not is_consistent and self.NORMALIZE_CONFIDENCES_ON_RED_TEAM:
            logger.info("NDG configured to normalize confidences on run; normalizing now using method: %s", self.NORMALIZE_METHOD)
            norm_info = self.normalize_confidences(nodes, method=self.NORMALIZE_METHOD)
            total_conf = norm_info.get('new_sum', self._compute_total_confidence(nodes))
            is_consistent = abs(total_conf - 1.0) <= self.CONFIDENCE_SUM_TOLERANCE
            self._last_confidence_sum = total_conf
            self._last_confidence_consistent = is_consistent
            logger.info("NDG normalization complete (new sum: %.3f)", total_conf)
        else:
            # update metadata if not present
            self._last_confidence_sum = previous_sum
            self._last_confidence_consistent = is_consistent

        return total_conf, is_consistent

    def _format_summary(self, fragility: FragilityMetrics, nodes: List[NDGNode], avg_evidence: float) -> str:
        """Create a compact summary string for NDGOutput."""
        return f"Fragility: {fragility.fragility_score:.2f}; nodes: {len(nodes)}; avg_evidence: {avg_evidence:.2f}"

    def _prepare_inputs(self, thesis_narrative: Optional[str], company_context: Optional[str], thesis_input: Optional[ThesisInput]) -> Tuple[str, str]:
        """Return validated inputs, preferring `thesis_input` when provided."""
        if thesis_input is not None:
            thesis_narrative = thesis_input.narrative

        if not thesis_narrative:
            raise ValueError("thesis_narrative is required for NDG analysis")
        if not company_context:
            raise ValueError("company_context is required for NDG analysis")
        return thesis_narrative, company_context

    def _build_graph_and_classify(self, claims: List[Dict[str, Any]]) -> Tuple[List[NDGNode], List[NDGEdge]]:
        """Build DAG and classify assumptions in one helper."""
        ambiguous = self._warn_ambiguous_claims(claims)
        nodes, edges = self.build_dag(claims)
        logger.info(f"Built graph: {len(nodes)} nodes, {len(edges)} edges")
        nodes = self.classify_assumptions(nodes)
        logger.info("Classified assumptions")
        return nodes, edges

    def _enrich_nodes(self, nodes: List[NDGNode], company_context: str, quantitative_context: Optional[ThesisQuantitativeContext], thesis_narrative: str) -> List[NDGNode]:
        """Map evidence and distribute confidence for node enrichment."""
        nodes = self.map_evidence(nodes, company_context, quantitative_context)
        avg_evidence = self._compute_avg_evidence(nodes)
        logger.debug(f"Evidence mapped (avg strength: {avg_evidence:.2f})")
        nodes = self.distribute_confidence(thesis_narrative, nodes)
        return nodes

    # --- 2.6 FRAGILITY & CONCENTRATION DIAGNOSTICS ---
    def compute_fragility(
        self, 
        nodes: List[NDGNode], 
        edges: List[NDGEdge]
    ) -> FragilityMetrics:
        """
        Compute structural fragility metrics and return a `FragilityMetrics`
        dataclass that includes both the overall `fragility_score` (0-1) and a
        per-component breakdown (`fragility_components`) for explainability.

        Components evaluated:
        - assumption_load (normalized by number of outcomes)
        - graph_depth (penalized when deeper than a threshold)
        - single_point_failures (count × contribution)
        - evidence_weakness (inverse of average evidence strength)

        Args:
            nodes: Fully enriched nodes
            edges: Graph edges

        Returns:
            `FragilityMetrics` with diagnostics and `fragility_components`.
        """       
        # Build adjacency structures
        node_map = {n.id: n for n in nodes}
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        
        for edge in edges:
            outgoing[edge.source_id].append(edge.target_id)
            incoming[edge.target_id].append(edge.source_id)
        
        # Categorize nodes by type
        assumptions = [n for n in nodes if n.node_type == self.NODE_TYPE_ASSUMPTION]
        drivers = [n for n in nodes if n.node_type == self.NODE_TYPE_DRIVER]
        outcomes = [n for n in nodes if n.node_type == self.NODE_TYPE_OUTCOME]
        
        # Track node counts
        # 1. Assumption load (total number of assumptions)
        assumption_load = len(assumptions)
        
        # 2. Average assumptions per outcome (trace paths)
        def count_assumptions_for_outcome(outcome_id: str, visited: Set = None) -> int:
            if visited is None:
                visited = set()
            if outcome_id in visited:
                return 0
            visited.add(outcome_id)
            
            node = node_map[outcome_id]
            if node.node_type == self.NODE_TYPE_ASSUMPTION:
                return 1
            
            count = 0
            for parent_id in incoming[outcome_id]:
                count += count_assumptions_for_outcome(parent_id, visited.copy())
            return count
        
        assumptions_per_outcome = []
        for outcome in outcomes:
            count = count_assumptions_for_outcome(outcome.id)
            assumptions_per_outcome.append(count)
        
        avg_assumptions_per_outcome = (
            sum(assumptions_per_outcome) / len(assumptions_per_outcome) 
            if assumptions_per_outcome else 0.0
        )
        
        # 3. Single-point failures (nodes with low evidence but high path concentration)
        def count_paths_through_node(node_id: str) -> int:
            """Count how many paths from assumptions to outcomes pass through this node."""
            upstream_memo: Dict[str, int] = {}
            downstream_memo: Dict[str, int] = {}

            def count_upstream(nid: str) -> int:
                if nid in upstream_memo:
                    return upstream_memo[nid]
                node = node_map[nid]
                if node.node_type == self.NODE_TYPE_ASSUMPTION:
                    upstream_memo[nid] = 1
                    return 1
                total = 0
                for parent in incoming[nid]:
                    total += count_upstream(parent)
                upstream_memo[nid] = total
                return total

            def count_downstream(nid: str) -> int:
                if nid in downstream_memo:
                    return downstream_memo[nid]
                node = node_map[nid]
                if node.node_type == self.NODE_TYPE_OUTCOME:
                    downstream_memo[nid] = 1
                    return 1
                total = 0
                for child in outgoing[nid]:
                    total += count_downstream(child)
                downstream_memo[nid] = total
                return total

            upstream = count_upstream(node_id)
            downstream = count_downstream(node_id)
            return upstream * downstream
        
        # --- Detect feedback loops (SCCs) ---
        # Tarjan's algorithm for SCCs
        node_ids = list(node_map.keys())
        index = 0
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        onstack: Set[str] = set()
        sccs: List[List[str]] = []

        def strongconnect(v: str):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            onstack.add(v)

            for w in outgoing.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in onstack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    onstack.remove(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)

        for nid in node_ids:
            if nid not in indices:
                strongconnect(nid)

        loops = [s for s in sccs if len(s) > 1]

        feedback_loops: List[dict] = []
        loop_bonus_total = 0.0
        loop_penalty_total = 0.0

        # Build a fast set for edges per (src,target) strength lookup
        edge_strength_map = {(e.source_id, e.target_id): e.strength for e in edges}

        for i, loop in enumerate(loops):
            # internal edge strength average
            internal_edges = [edge_strength_map.get((a, b), 0.0) for a in loop for b in loop if a != b]
            avg_edge_strength = sum(internal_edges) / len(internal_edges) if internal_edges else 0.0
            avg_evidence = sum(node_map[nid].evidence_strength for nid in loop) / len(loop)
            control_counts: Dict[str, int] = {}
            for nid in loop:
                ctrl = getattr(node_map[nid], 'control', '') or 'Unknown'
                control_counts[ctrl] = control_counts.get(ctrl, 0) + 1

            # Determine reinforcement vs weakness
            reinforcing = (
                avg_edge_strength >= self.LOOP_EDGE_STRENGTH_THRESHOLD and
                avg_evidence >= self.LOOP_EVIDENCE_THRESHOLD and
                (control_counts.get('Exogenous', 0) / len(loop)) < 0.5
            )

            # Compute conservative bonus/penalty
            if reinforcing:
                factor = max(0.0, ((avg_edge_strength - self.LOOP_EDGE_STRENGTH_THRESHOLD) + (avg_evidence - self.LOOP_EVIDENCE_THRESHOLD)) / (2 * (1.0 - self.LOOP_EDGE_STRENGTH_THRESHOLD)))
                bonus = self.LOOP_REINFORCEMENT_BONUS_MAX * min(1.0, factor)
                loop_bonus_total += bonus
            else:
                # loop weakness when both edge and evidence are below thresholds
                if avg_edge_strength < self.LOOP_EDGE_STRENGTH_THRESHOLD and avg_evidence < self.LOOP_EVIDENCE_THRESHOLD:
                    factor = max(0.0, ((self.LOOP_EDGE_STRENGTH_THRESHOLD - avg_edge_strength) + (self.LOOP_EVIDENCE_THRESHOLD - avg_evidence)) / (2 * self.LOOP_EDGE_STRENGTH_THRESHOLD))
                    penalty = self.LOOP_WEAKNESS_PENALTY_MAX * min(1.0, factor)
                    loop_penalty_total += penalty
                else:
                    penalty = 0.0
                    bonus = 0.0

            feedback_loops.append({
                'id': f'loop_{i+1}',
                'nodes': loop,
                'avg_edge_strength': avg_edge_strength,
                'avg_evidence': avg_evidence,
                'control_mix': control_counts,
                'reinforcing': reinforcing
            })

        # Build condensation graph (SCCs as nodes) for path/depth calculations
        scc_map: Dict[str, int] = {}
        for idx, comp in enumerate(sccs):
            for nid in comp:
                scc_map[nid] = idx

        scc_outgoing: Dict[int, Set[int]] = defaultdict(set)
        scc_incoming: Dict[int, Set[int]] = defaultdict(set)
        for e in edges:
            s = scc_map.get(e.source_id)
            t = scc_map.get(e.target_id)
            if s is not None and t is not None and s != t:
                scc_outgoing[s].add(t)
                scc_incoming[t].add(s)

        # Memoized upstream/downstream counts per SCC
        scc_upstream_memo: Dict[int, int] = {}
        scc_downstream_memo: Dict[int, int] = {}

        def count_upstream_scc(sid: int) -> int:
            if sid in scc_upstream_memo:
                return scc_upstream_memo[sid]
            # If any node in SCC is an assumption, count as 1
            comp_nodes = sccs[sid]
            if any(node_map[nid].node_type == self.NODE_TYPE_ASSUMPTION for nid in comp_nodes):
                scc_upstream_memo[sid] = 1
                return 1
            total = 0
            for parent in scc_incoming.get(sid, []):
                total += count_upstream_scc(parent)
            scc_upstream_memo[sid] = total
            return total

        def count_downstream_scc(sid: int) -> int:
            if sid in scc_downstream_memo:
                return scc_downstream_memo[sid]
            comp_nodes = sccs[sid]
            if any(node_map[nid].node_type == self.NODE_TYPE_OUTCOME for nid in comp_nodes):
                scc_downstream_memo[sid] = 1
                return 1
            total = 0
            for child in scc_outgoing.get(sid, []):
                total += count_downstream_scc(child)
            scc_downstream_memo[sid] = total
            return total

        path_concentration = {}
        single_point_failures = []
        spof_details = []

        for node in nodes:
            sid = scc_map.get(node.id)
            upstream = count_upstream_scc(sid)
            downstream = count_downstream_scc(sid)
            path_count = upstream * downstream
            path_concentration[node.id] = path_count

            if (path_count >= self.SPOF_MIN_PATH_COUNT and node.evidence_strength < self.SPOF_EVIDENCE_THRESHOLD):
                single_point_failures.append(node.id)
                spof_details.append({
                    'node_id': node.id,
                    'claim': node.claim,
                    'path_count': path_count,
                    'evidence_strength': node.evidence_strength
                })
        
        # 4. Max graph depth
        # Use memoization to compute max depth from node efficiently
        depth_memo: Dict[str, int] = {}
        def max_depth_from_node(node_id: str) -> int:
            if node_id in depth_memo:
                return depth_memo[node_id]
            if not outgoing[node_id]:
                depth_memo[node_id] = 1
                return 1
            max_child_depth = 0
            for child_id in outgoing[node_id]:
                child_depth = max_depth_from_node(child_id)
                max_child_depth = max(max_child_depth, child_depth)
            depth_memo[node_id] = 1 + max_child_depth
            return depth_memo[node_id]
        
        max_graph_depth = 0
        for assumption in assumptions:
            depth = max_depth_from_node(assumption.id)
            max_graph_depth = max(max_graph_depth, depth)
        
        # 5. Overall fragility score
        # Penalize: many assumptions, deep graph, single-point failures, low evidence
        fragility_score = 0.0
        fragility_components = {}
        
        # Component 1: Assumption load (weighted by control/nature/time, normalized by number of outcomes)
        weighted_assumption_load = 0.0
        per_node_weights: Dict[str, float] = {}
        for n in assumptions:
            cw = self.CONTROL_WEIGHTS.get(getattr(n, 'control', ''), 1.0)
            nw = self.NATURE_WEIGHTS.get(getattr(n, 'nature', ''), 1.0)
            tw = self.TIME_WEIGHTS.get(getattr(n, 'time_sensitivity', ''), 1.0)
            weight = (cw * nw * tw) ** (1.0 / 3.0)
            per_node_weights[n.id] = weight
            weighted_assumption_load += weight

        if outcomes:
            load_component = min(
                self.MAX_FRAGILITY_ASSUMPTION_LOAD,
                (weighted_assumption_load / len(outcomes)) * self.ASSUMPTION_LOAD_SCALE_FACTOR
            )
        else:
            load_component = 0.0
        fragility_score += load_component
        fragility_components['assumption_load'] = load_component
        fragility_components['assumption_load_weighted'] = weighted_assumption_load

        # Component 2: Graph depth (penalize above threshold)
        if max_graph_depth > self.DEPTH_PENALTY_THRESHOLD:
            depth_component = min(
                self.MAX_FRAGILITY_GRAPH_DEPTH,
                (max_graph_depth - self.DEPTH_PENALTY_THRESHOLD) * self.DEPTH_PENALTY_PER_LAYER
            )
        else:
            depth_component = 0.0
        fragility_score += depth_component
        fragility_components['graph_depth'] = depth_component

        # Component 3: Single-point failures (consider control/nature multipliers for thresholding)
        spof_hits = 0
        for node_id, pc in path_concentration.items():
            node = node_map[node_id]
            cw = self.CONTROL_WEIGHTS.get(getattr(node, 'control', ''), 1.0)
            nw = self.NATURE_WEIGHTS.get(getattr(node, 'nature', ''), 1.0)
            node_risk_multiplier = cw * nw
            threshold_adj = self.SPOF_EVIDENCE_THRESHOLD * node_risk_multiplier
            if (pc >= self.SPOF_MIN_PATH_COUNT and node.evidence_strength < threshold_adj):
                spof_hits += 1
                single_point_failures.append(node_id)
                spof_details.append({
                    'node_id': node.id,
                    'claim': node.claim,
                    'path_count': pc,
                    'evidence_strength': node.evidence_strength,
                    'control_weight': cw,
                    'nature_weight': nw,
                    'threshold_adj': threshold_adj
                })
        spof_component = min(self.MAX_FRAGILITY_SPOF, spof_hits * self.SPOF_CONTRIBUTION_PER_FAILURE)
        fragility_score += spof_component
        fragility_components['single_point_failures'] = spof_component

        # Component 4: Evidence strength (importance-weighted across nodes)
        # Use path_concentration-derived importance to weight evidence by structural impact
        max_pc = max(path_concentration.values()) if path_concentration else 1
        importance_norm: Dict[str, float] = {nid: (pc / max_pc if max_pc else 0.0) for nid, pc in path_concentration.items()}

        total_weighted_evidence = 0.0
        total_weights = 0.0
        critical_low_nodes: List[str] = []
        for n in nodes:
            base_w = per_node_weights.get(n.id, 1.0)
            imp = importance_norm.get(n.id, 0.0)
            importance_factor = 1.0 + imp  # ranges from 1.0 .. 2.0
            combined_w = base_w * importance_factor
            total_weighted_evidence += n.evidence_strength * combined_w
            total_weights += combined_w

            # detect critical low evidence nodes (high importance + low evidence)
            if imp >= self.CRITICAL_IMPORTANCE_THRESHOLD and n.evidence_strength < self.CRITICAL_EVIDENCE_THRESHOLD:
                critical_low_nodes.append(n.id)

        avg_importance_weighted_evidence = total_weighted_evidence / total_weights if total_weights else 0.0
        evidence_component = (1.0 - avg_importance_weighted_evidence) * self.MAX_FRAGILITY_EVIDENCE
        fragility_score += evidence_component
        fragility_components['evidence_weakness'] = evidence_component
        fragility_components['avg_importance_weighted_evidence'] = avg_importance_weighted_evidence
        fragility_components['critical_low_nodes'] = critical_low_nodes
        
        # Apply loop-based adjustments
        fragility_components['feedback_loops_bonus'] = loop_bonus_total
        fragility_components['feedback_loops_penalty'] = loop_penalty_total

        # convert feedback loop dicts into FeedbackLoop objects for structured output
        loops_objs: List[FeedbackLoop] = [
            FeedbackLoop(
                id=l['id'],
                nodes=l['nodes'],
                avg_edge_strength=l['avg_edge_strength'],
                avg_evidence=l['avg_evidence'],
                control_mix=l['control_mix'],
                reinforcing=l['reinforcing']
            ) for l in feedback_loops
        ]

        # adjust fragility score conservatively
        fragility_score = fragility_score - loop_bonus_total + loop_penalty_total
        fragility_score = min(1.0, max(0.0, fragility_score))
        fragility_components['total'] = fragility_score

        logger.info(f"Computed fragility: {fragility_score:.2f}; loops={len(loops_objs)}; critical_low={len(critical_low_nodes)}")
        return FragilityMetrics(
            assumption_load=assumption_load,
            avg_assumptions_per_outcome=avg_assumptions_per_outcome,
            single_point_failures=single_point_failures,
            path_concentration=path_concentration,
            max_graph_depth=max_graph_depth,
            fragility_score=fragility_score,
            fragility_components=fragility_components,
            feedback_loops=loops_objs,
            critical_low_evidence_nodes=critical_low_nodes
        )
    
    # --- 2.7 MAIN PIPELINE ---
    def run(
        self, 
        thesis_narrative: Optional[str] = None, 
        company_context: Optional[str] = None, 
        quantitative_context: Optional[ThesisQuantitativeContext] = None,
        thesis_input: Optional[ThesisInput] = None
    ) -> NDGOutput:
        """
        Execute the full NDG pipeline and return structured diagnostics.

        Supports either a raw thesis narrative string (legacy) or a `ThesisInput`
        dataclass object (preferred when available). When `thesis_input` is
        provided its `.narrative` field is used as the thesis text (caller may
        provide `company_context` and `quantitative_context` separately).

        Steps performed:
        1. Parse thesis → extract claims
        2. Build DAG → structure causal relationships
        3. Classify assumptions → ownership and control
        4. Map evidence → supporting/contradicting
        5. Distribute confidence → attribution across nodes
        6. Compute fragility → structural brittleness diagnostics

        Args:
            thesis_narrative: Natural language investment thesis (legacy input)
            company_context: Required company description for context
            quantitative_context: Optional Stage 1 quantitative context from
                                  StockAnalyticsOrchestrator.get_thesis_context().
                                  This is supplementary and should not be
                                  over-weighted relative to independent analysis.
            thesis_input: Optional `ThesisInput` object (preferred) whose
                          `.narrative` will be used if provided.

        Returns:
            `NDGOutput` containing graph, `FragilityMetrics` (with breakdown),
            and `total_confidence`.

        Raises:
            ValueError: If `company_context` is empty or None or if no thesis is provided
        """
        thesis_narrative, company_context = self._prepare_inputs(thesis_narrative, company_context, thesis_input)

        logger.info(f"Starting NDG pipeline for {self.stock_ticker}")
        
        if quantitative_context:
            logger.info(f"Quantitative context loaded (data as of: {quantitative_context.data_as_of})")
        
        # Step 1: Parse thesis (claims + optional metrics)
        parsed = self.parse_thesis(thesis_narrative)
        claims = parsed.get('claims', [])
        parsed_metrics = parsed.get('metrics', {}) or {}
        logger.info(f"Parsed thesis: {len(claims)} claims; metrics keys: {list(parsed_metrics.keys())}")

        # Build and classify graph
        nodes, edges = self._build_graph_and_classify(claims)

        # Enrich nodes (evidence mapping + confidence distribution)
        nodes = self._enrich_nodes(nodes, company_context, quantitative_context, thesis_narrative)

        # Diagnostics & summaries delegated to helpers
        avg_evidence = self._compute_avg_evidence(nodes)
        total_conf, confidence_consistent = self._ensure_confidence_consistency(nodes)
        fragility = self.compute_fragility(nodes, edges)
        logger.info(f"Fragility score: {fragility.fragility_score:.2f}")

        summary_text = self._format_summary(fragility, nodes, avg_evidence)

        confidence_sum = getattr(self, '_last_confidence_sum', total_conf)
        confidence_consistent = getattr(self, '_last_confidence_consistent', confidence_consistent)

        return NDGOutput(
            stock_ticker=self.stock_ticker,
            thesis_text=thesis_narrative,
            nodes=nodes,
            edges=edges,
            fragility_metrics=fragility,
            total_confidence=total_conf,
            confidence_sum=confidence_sum,
            confidence_consistent=confidence_consistent,
            summary_text=summary_text,
            extracted_metrics=parsed_metrics,
            extracted_claims=claims
        )
