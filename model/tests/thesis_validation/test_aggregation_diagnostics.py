"""Tests for AggregationDiagnostics agent."""
from unittest.mock import Mock

from model.thesis_agents.aggregation_diagnostics import AggregationDiagnostics


class TestAggregationDiagnostics:
    """Test suite for AggregationDiagnostics."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agg = AggregationDiagnostics('AAPL')
        assert agg.stock_ticker == 'AAPL'
    
    def test_run(self, sample_ndg_output, sample_red_team_output,
                 sample_ft_output, sample_validity_output):
        """Test full aggregation."""
        agg = AggregationDiagnostics('AAPL')
        
        # Mock IHLE output
        ihle_output = Mock()
        ihle_output.half_life_estimate = Mock(
            estimated_half_life_months=6.0,
            primary_decay_drivers=['Driver 1', 'Driver 2', 'Driver 3']
        )
        ihle_output.monitoring_cadence = Mock(recommended_frequency='monthly')
        
        result = agg.run(
            ihle_output=ihle_output,
            ft_output=sample_ft_output,
            red_team_output=sample_red_team_output,
            validity_output=sample_validity_output
        )
        
        assert result is not None
        assert hasattr(result, 'stock_ticker')
        assert hasattr(result, 'summary_text')
    
    def test_run_missing_inputs(self):
        """Test handling of missing inputs."""
        agg = AggregationDiagnostics('AAPL')
        
        # Should handle None inputs gracefully
        result = agg.run(ihle_output=None, ft_output=None, red_team_output=None, validity_output=None)
        assert result is not None
        assert hasattr(result, 'stock_ticker')
    
    def test_key_metrics_structure(self, sample_ndg_output, sample_red_team_output,
                                    sample_ft_output, sample_validity_output):
        """Test key metrics output structure."""
        agg = AggregationDiagnostics('AAPL')
        ihle_output = Mock()
        ihle_output.half_life_estimate = Mock(
            estimated_half_life_months=6.0,
            primary_decay_drivers=['Driver 1', 'Driver 2']
        )
        ihle_output.monitoring_cadence = Mock(recommended_frequency='monthly')
        
        result = agg.run(
            ihle_output=ihle_output,
            ft_output=sample_ft_output,
            red_team_output=sample_red_team_output,
            validity_output=sample_validity_output
        )
        
        assert hasattr(result, 'summary_text')
        assert isinstance(result.summary_text, (str, type(None)))
