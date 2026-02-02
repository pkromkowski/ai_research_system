"""Tests for ThesisValidationOrchestrator."""
import pytest
from unittest.mock import Mock, patch

from model.orchestration.thesis_validation_orchestrator import ThesisValidationOrchestrator


class TestThesisValidationOrchestrator:
    """Test suite for ThesisValidationOrchestrator."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ThesisValidationOrchestrator('AAPL')
        assert orchestrator.stock_ticker == 'AAPL'
        
        # Verify all agents initialized
        assert hasattr(orchestrator, 'ndg')
        assert hasattr(orchestrator, 'red_team')
        assert hasattr(orchestrator, 'cre')
        assert hasattr(orchestrator, 'ft')
        assert hasattr(orchestrator, 'validity_evaluator')
        assert hasattr(orchestrator, 'ihle')
        assert hasattr(orchestrator, 'aggregator')
    
    def test_run_requires_inputs(self):
        """Test that run() method requires proper inputs."""
        orchestrator = ThesisValidationOrchestrator('AAPL')
        
        # Verify run() method exists and is callable
        assert callable(orchestrator.run)
    
    def test_run_full_pipeline(self, sample_ndg_output, sample_red_team_output, sample_cre_generation_result, sample_ft_result, sample_validity_output, sample_ihle_output, sample_aggregation_output, monkeypatch):
        """Test full orchestration pipeline using realistic fixtures and monkeypatching."""
        # Helper factory returning a callable class that produces an object with run() -> return_value
        def _factory(return_value):
            def _constructor(*args, **kwargs):
                m = Mock()
                m.run.return_value = return_value
                return m
            return _constructor

        # Monkeypatch agent classes in the orchestrator module
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.NarrativeDecompositionGraph', _factory(sample_ndg_output))
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.AIRedTeamWithMemory', _factory(sample_red_team_output))
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.CounterfactualResearchEngine', _factory(sample_cre_generation_result))
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.FinancialTranslation', _factory(sample_ft_result))
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.ThesisValidityEvaluator', _factory(sample_validity_output))
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.IdeaHalfLifeEstimator', _factory(sample_ihle_output))
        monkeypatch.setattr('model.orchestration.thesis_validation_orchestrator.AggregationDiagnostics', _factory(sample_aggregation_output))

        # Run orchestrator
        orchestrator = ThesisValidationOrchestrator('AAPL')
        result = orchestrator.run(
            thesis_narrative='Test thesis',
            company_context='Test company',
            quantitative_context=None
        )

        # Verify the returned report is consistent with fixtures
        assert result is not None
        assert result.stock == 'AAPL'
        assert abs(result.survival_rate - sample_validity_output.survival_rate) < 1e-6
        assert result.half_life_months == sample_ihle_output.half_life_estimate.estimated_half_life_months
        assert result.detailed_components['ndg'] == sample_ndg_output
        assert result.detailed_components['ft'] == sample_ft_result.cre_output
        assert result.detailed_components['validity'] == sample_validity_output
        assert result.detailed_components['ihle'] == sample_ihle_output
        assert result.detailed_components['aggregation'] == sample_aggregation_output
    
    def test_error_handling_ndg_failure(self):
        """Test error handling when NDG stage fails."""
        orchestrator = ThesisValidationOrchestrator('AAPL')
        
        with patch.object(orchestrator.ndg, 'run', side_effect=Exception('NDG error')):
            with pytest.raises(Exception, match='NDG error'):
                orchestrator.run(
                    thesis_narrative='Test thesis',
                    company_context='Test company'
                )
