"""Tests for NarrativeDecompositionGraph agent."""
import pytest
from unittest.mock import patch

from model.thesis_agents.narrative_decomposition import NarrativeDecompositionGraph


class TestNarrativeDecompositionGraph:
    """Test suite for NarrativeDecompositionGraph."""
    
    def test_initialization(self):
        """Test agent initialization."""
        ndg = NarrativeDecompositionGraph('AAPL')
        assert ndg.stock_ticker == 'AAPL'
        assert ndg.MAX_TOKENS_PARSE > 0
        assert ndg.TEMPERATURE_PARSE >= 0.0
    
    def test_prepare_inputs_with_narrative(self):
        """Test input preparation with narrative string."""
        ndg = NarrativeDecompositionGraph('AAPL')
        
        thesis, company = ndg._prepare_inputs(
            'Sample thesis',
            'Sample company',
            None
        )
        
        assert thesis == 'Sample thesis'
        assert company == 'Sample company'
    
    def test_prepare_inputs_missing_thesis(self):
        """Test error when thesis is missing."""
        ndg = NarrativeDecompositionGraph('AAPL')
        
        with pytest.raises(ValueError, match='thesis_narrative is required'):
            ndg._prepare_inputs(None, 'Company', None)
    
    def test_prepare_inputs_missing_company(self):
        """Test error when company context is missing."""
        ndg = NarrativeDecompositionGraph('AAPL')
        
        with pytest.raises(ValueError, match='company_context is required'):
            ndg._prepare_inputs('Thesis', None, None)
    
    def test_compute_avg_evidence(self, sample_ndg_nodes):
        """Test average evidence calculation."""
        ndg = NarrativeDecompositionGraph('AAPL')
        avg = ndg._compute_avg_evidence(sample_ndg_nodes)
        
        assert isinstance(avg, float)
        assert 0.0 <= avg <= 1.0
    
    def test_compute_total_confidence(self, sample_ndg_nodes):
        """Test total confidence calculation."""
        ndg = NarrativeDecompositionGraph('AAPL')
        total = ndg._compute_total_confidence(sample_ndg_nodes)
        
        assert isinstance(total, float)
        assert total > 0.0
    
    def test_is_dag_valid(self):
        """Test DAG validation with valid edges."""
        ndg = NarrativeDecompositionGraph('AAPL')
        
        edges = [
            {'source_id': 'a', 'target_id': 'b'},
            {'source_id': 'b', 'target_id': 'c'}
        ]
        
        assert ndg._is_dag(edges) is True
    
    def test_is_dag_with_cycle(self):
        """Test DAG validation with cycle."""
        ndg = NarrativeDecompositionGraph('AAPL')
        
        edges = [
            {'source_id': 'a', 'target_id': 'b'},
            {'source_id': 'b', 'target_id': 'a'}
        ]
        
        assert ndg._is_dag(edges) is False
    
    @patch.object(NarrativeDecompositionGraph, '_call_llm_structured')
    def test_parse_thesis(self, mock_llm, mock_llm_response):
        """Test thesis parsing."""
        mock_llm.return_value = mock_llm_response
        
        ndg = NarrativeDecompositionGraph('AAPL')
        result = ndg.parse_thesis('Sample thesis narrative')
        
        assert 'claims' in result
        assert 'metrics' in result
        assert isinstance(result['claims'], list)
        mock_llm.assert_called_once()
    
    @patch.object(NarrativeDecompositionGraph, '_call_llm_structured')
    def test_build_dag(self, mock_llm):
        """Test DAG building."""
        mock_llm.return_value = {
            'nodes': [{'id': 'n1', 'claim': 'Test', 'node_type': 'ASSUMPTION', 'dependencies': []}],
            'edges': [{'source_id': 'n1', 'target_id': 'n2', 'relationship': 'ENABLES', 'strength': 0.8}]
        }
        
        ndg = NarrativeDecompositionGraph('AAPL')
        claims = [{'claim': 'Test claim', 'type': 'ASSUMPTION'}]
        
        nodes, edges = ndg.build_dag(claims)
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
    
    @patch.object(NarrativeDecompositionGraph, '_call_llm_structured')
    def test_classify_assumptions(self, mock_llm, sample_ndg_nodes):
        """Test assumption classification."""
        mock_llm.return_value = {
            'classifications': [
                {
                    'id': 'node_1',
                    'control': 'Company',
                    'nature': 'Execution',
                    'time_sensitivity': 'Short'
                }
            ]
        }
        
        ndg = NarrativeDecompositionGraph('AAPL')
        result = ndg.classify_assumptions(sample_ndg_nodes[:1])
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_normalize_confidences_proportional(self, sample_ndg_nodes):
        """Test proportional confidence normalization."""
        ndg = NarrativeDecompositionGraph('AAPL')
        result = ndg.normalize_confidences(sample_ndg_nodes, method='proportional')
        
        # normalize_confidences returns a dict with adjustments
        assert isinstance(result, dict)
        assert 'adjustments' in result
        assert 'method' in result
        assert 'original_sum' in result
        assert 'new_sum' in result
    
    def test_compute_fragility(self, sample_ndg_nodes, sample_ndg_edges):
        """Test fragility computation."""
        ndg = NarrativeDecompositionGraph('AAPL')
        fragility = ndg.compute_fragility(sample_ndg_nodes, sample_ndg_edges)
        
        assert hasattr(fragility, 'fragility_score')
        assert isinstance(fragility.fragility_score, float)
        assert 0.0 <= fragility.fragility_score <= 1.0
    
    def test_format_summary(self, sample_fragility_metrics, sample_ndg_nodes):
        """Test summary formatting."""
        ndg = NarrativeDecompositionGraph('AAPL')
        summary = ndg._format_summary(sample_fragility_metrics, sample_ndg_nodes, 0.65)
        
        assert isinstance(summary, str)
        assert '0.45' in summary
        assert 'nodes: 3' in summary
