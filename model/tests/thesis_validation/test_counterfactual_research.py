"""Tests for CounterfactualResearchEngine agent."""
import pytest
from unittest.mock import patch

from model.thesis_agents.counterfactual_research import CounterfactualResearchEngine


class TestCounterfactualResearchEngine:
    """Test suite for CounterfactualResearchEngine."""
    
    def test_initialization(self):
        """Test agent initialization."""
        cre = CounterfactualResearchEngine('AAPL')
        assert cre.stock_ticker == 'AAPL'
        assert cre.MAX_TOKENS_BOUND > 0
    
    def test_extract_canonical_assumptions(self, sample_ndg_output):
        """Test canonical assumption extraction."""
        cre = CounterfactualResearchEngine('AAPL')
        metrics, claims = cre._extract_canonical_assumptions(sample_ndg_output, None)
        
        assert isinstance(metrics, dict)
        assert isinstance(claims, list)
    
    def test_derive_priority_claims(self, sample_ndg_output, sample_red_team_output):
        """Test priority claim derivation."""
        cre = CounterfactualResearchEngine('AAPL')
        priority = cre._derive_priority_claims(sample_ndg_output, sample_red_team_output)
        
        assert priority is None or isinstance(priority, list)
    
    @patch.object(CounterfactualResearchEngine, '_call_llm_structured')
    def test_bound_assumptions(self, mock_llm):
        """Test assumption bounding."""
        mock_llm.return_value = {
            'bounds': {
                'revenue_growth': {'min': 0.2, 'max': 0.6, 'base': 0.4}
            }
        }
        
        cre = CounterfactualResearchEngine('AAPL')
        metrics = {'revenue_growth': 0.40}
        claims = ['Strong revenue growth']
        
        result = cre.bound_assumptions(
            metrics, claims, 'Company context', None
        )
        
        assert isinstance(result, dict)
    
    @patch.object(CounterfactualResearchEngine, '_call_llm_structured')
    @patch.object(CounterfactualResearchEngine, 'bound_assumptions')
    def test_run(self, mock_bound, mock_llm, sample_ndg_output):
        """Test full CRE execution."""
        mock_bound.return_value = {'revenue_growth': {'min': 0.2, 'max': 0.6}}
        mock_llm.return_value = {
            'scenarios': [
                {'scenario_id': 'base', 'name': 'Base', 'probability': 0.5}
            ]
        }
        
        cre = CounterfactualResearchEngine('AAPL')
        result = cre.run(
            ndg=sample_ndg_output,
            red_team=None,
            company_context='Test company',
            quantitative_context=None
        )
        
        assert result is not None
        assert hasattr(result, 'scenario_set')
    
    def test_run_missing_company_context(self, sample_ndg_output):
        """Test error with missing company context."""
        cre = CounterfactualResearchEngine('AAPL')
        
        with pytest.raises(ValueError, match='company_context is required'):
            cre.run(
                ndg=sample_ndg_output,
                red_team=None,
                company_context='',
                quantitative_context=None
            )
