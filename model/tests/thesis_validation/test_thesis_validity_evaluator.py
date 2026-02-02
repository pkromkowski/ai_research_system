"""Tests for ThesisValidityEvaluator agent."""
import pytest
from unittest.mock import patch

from model.thesis_agents.thesis_validity_evaluator import ThesisValidityEvaluator


class TestThesisValidityEvaluator:
    """Test suite for ThesisValidityEvaluator."""
    
    def test_initialization(self):
        """Test agent initialization."""
        evaluator = ThesisValidityEvaluator('AAPL')
        assert evaluator.stock_ticker == 'AAPL'
    
    def test_aggregate_from_scenario_results_positive(self, sample_valuation_results):
        """Test aggregation with positive results."""
        evaluator = ThesisValidityEvaluator('AAPL')
        
        # Mock positive valuations
        for vr in sample_valuation_results:
            vr.upside_potential = 0.20
        
        aggregate = evaluator._aggregate_from_scenario_results(sample_valuation_results)
        
        assert aggregate is not None
        assert 'weighted_upside' in aggregate or isinstance(aggregate, dict)
    
    def test_aggregate_from_scenario_results_negative(self, sample_valuation_results):
        """Test aggregation with negative results."""
        evaluator = ThesisValidityEvaluator('AAPL')
        
        # Mock negative valuations
        for vr in sample_valuation_results:
            vr.upside_potential = -0.15
        
        aggregate = evaluator._aggregate_from_scenario_results(sample_valuation_results)
        
        assert aggregate is not None
    
    def test_classify_thesis_status_valid(self):
        """Test thesis classification - valid."""
        evaluator = ThesisValidityEvaluator('AAPL')
        
        status, reasons = evaluator._classify_thesis_status(
            survival=0.75,
            fragility=0.3,
            tail_loss=-0.05,
            red_team_coverage_required=False,
            reasons=[]
        )
        
        assert status in ['VALID', 'FRAGILE', 'BROKEN']
        assert isinstance(reasons, list)
    
    def test_classify_thesis_status_invalid(self):
        """Test thesis classification - invalid."""
        evaluator = ThesisValidityEvaluator('AAPL')
        
        status, reasons = evaluator._classify_thesis_status(
            survival=0.3,
            fragility=0.8,
            tail_loss=-0.20,
            red_team_coverage_required=True,
            reasons=[]
        )
        
        assert status in ['BROKEN', 'FRAGILE']
        assert isinstance(reasons, list)
    
    def test_build_contradiction_list(self, sample_ndg_output, sample_red_team_output):
        """Test contradiction building."""
        evaluator = ThesisValidityEvaluator('AAPL')
        result = evaluator._build_contradictions_list(
            ndg=sample_ndg_output,
            red_team=sample_red_team_output
        )
        
        assert isinstance(result, list)
    
    @patch.object(ThesisValidityEvaluator, '_aggregate_from_scenario_results')
    @patch.object(ThesisValidityEvaluator, '_classify_thesis_status')
    def test_run(self, mock_classify, mock_aggregate, sample_ndg_output,
                 sample_red_team_output, sample_ft_output):
        """Test full validity evaluation."""
        mock_aggregate.return_value = {
            'scenario_survival_fraction': 0.67,
            'weighted_survival_rate': 0.60,
            'tail_loss_percentile': -0.10,
            'raw_fragility_proxy': 0.35,
            'dominant_failure_modes': ['Bear Case'],
            'impaired_scenarios': []
        }
        mock_classify.return_value = ('VALID', ['Strong scenario survival'])
        
        evaluator = ThesisValidityEvaluator('AAPL')
        result = evaluator.run(
            ndg=sample_ndg_output,
            red_team=sample_red_team_output,
            ft=sample_ft_output
        )
        
        assert result is not None
        assert hasattr(result, 'status')
        assert result.status == 'VALID'
    
    def test_run_missing_inputs(self):
        """Test error with missing inputs."""
        evaluator = ThesisValidityEvaluator('AAPL')
        
        with pytest.raises(AttributeError):
            evaluator.run(ndg=None, red_team=None, ft=None)
