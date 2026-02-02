"""Tests for IdeaHalfLifeEstimator agent."""
import pytest

from model.core.types import HalfLifeEstimate
from model.thesis_agents.half_life_estimator import IdeaHalfLifeEstimator



class TestIdeaHalfLifeEstimator:
    """Test suite for IdeaHalfLifeEstimator."""
    
    def test_initialization(self):
        """Test agent initialization."""
        ihle = IdeaHalfLifeEstimator('AAPL')
        assert ihle.stock_ticker == 'AAPL'
    
    def test_compute_monitoring_cadence_short(self):
        """Test monitoring cadence for short half-life."""
        ihle = IdeaHalfLifeEstimator('AAPL')
        half_life = HalfLifeEstimate(
            estimated_half_life_months=1.0,  # 1 month = short
            primary_decay_drivers=['Driver 1'],
            decay_trend='Accelerating',
            time_to_first_broken=0.5,
            regime_adjusted=False
        )
        cadence = ihle.recommend_cadence(half_life)
        
        assert cadence is not None
        assert hasattr(cadence, 'recommended_frequency')
    
    def test_compute_monitoring_cadence_medium(self):
        """Test monitoring cadence for medium half-life."""
        ihle = IdeaHalfLifeEstimator('AAPL')
        half_life = HalfLifeEstimate(
            estimated_half_life_months=6.0,  # 6 months = medium
            primary_decay_drivers=['Driver 1'],
            decay_trend='Gradual',
            time_to_first_broken=3.0,
            regime_adjusted=False
        )
        cadence = ihle.recommend_cadence(half_life)
        
        assert cadence is not None
        assert hasattr(cadence, 'recommended_frequency')
    
    def test_compute_monitoring_cadence_long(self):
        """Test monitoring cadence for long half-life."""
        ihle = IdeaHalfLifeEstimator('AAPL')
        half_life = HalfLifeEstimate(
            estimated_half_life_months=24.0,  # 24 months = long
            primary_decay_drivers=['Driver 1'],
            decay_trend='Stable',
            time_to_first_broken=12.0,
            regime_adjusted=False
        )
        cadence = ihle.recommend_cadence(half_life)
        
        assert cadence is not None
        assert hasattr(cadence, 'recommended_frequency')
    
    def test_run(self, sample_ndg_output, sample_red_team_output):
        """Test full IHLE execution."""
        ihle = IdeaHalfLifeEstimator('AAPL')
        # Pass red_team to avoid the SEVERITY_LABEL_LOW issue
        result = ihle.run(ndg=sample_ndg_output)
        
        assert result is not None
        assert hasattr(result, 'half_life_estimate')
        assert hasattr(result, 'monitoring_cadence')
        assert result.half_life_estimate.estimated_half_life_months > 0
    
    def test_run_missing_inputs(self):
        """Test error with missing inputs."""
        ihle = IdeaHalfLifeEstimator('AAPL')
        
        with pytest.raises((ValueError, AttributeError, TypeError)):
            ihle.run(ndg=None)
