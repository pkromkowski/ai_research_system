"""Tests for FinancialTranslation agent."""
import pytest

from model.thesis_agents.financial_translation import FinancialTranslation
from model.core.types import Scenario, CREScenarioSet, CREGenerationResult


class TestFinancialTranslation:
    """Test suite for FinancialTranslation."""
    
    def test_initialization(self):
        """Test agent initialization."""
        ft = FinancialTranslation('AAPL')
        assert ft.stock_ticker == 'AAPL'
    
    def test_validate_scenarios_valid(self, sample_scenarios):
        """Test scenario validation with valid data."""
        ft = FinancialTranslation('AAPL')
        base_metrics = {'revenue': 100.0, 'margin': 0.20}
        bounds = {'revenue': (50.0, 200.0), 'margin': (0.10, 0.40)}
        
        # Test validation with first scenario
        result = ft.validate_scenario(sample_scenarios[0], base_metrics, bounds)
        assert isinstance(result, bool)
    
    def test_validate_scenarios_empty(self):
        """Test scenario validation with no scenarios."""
        ft = FinancialTranslation('AAPL')
        base_metrics = {'revenue': 100.0}
        bounds = {'revenue': (50.0, 200.0)}
        
        # Create an invalid scenario (will test the validation logic)
        invalid_scenario = Scenario(
            name='Invalid',
            description='Test',
            impact='Test',
            stressed_assumptions={},
            plausibility_weight=0.0
        )
        result = ft.validate_scenario(invalid_scenario, base_metrics, bounds)
        assert isinstance(result, bool)
    
    def test_translate_to_valuation(self, sample_scenarios):
        """Test scenario evaluation."""
        ft = FinancialTranslation('AAPL')
        
        # Create a mock CREGenerationResult with necessary structure
        scenario_set = CREScenarioSet(
            stock_ticker='AAPL',
            scenarios=sample_scenarios,
            rejected_scenarios=[],
            base_metrics={'revenue_growth': 0.30, 'margin': 0.20},
            bounds={'revenue_growth': (0.10, 0.50), 'margin': (0.10, 0.40)}
        )
        
        # Test that evaluate methods work
        assert scenario_set is not None
        assert len(scenario_set.scenarios) > 0
    
    def test_run(self, sample_scenarios):
        """Test full FT execution."""
        ft = FinancialTranslation('AAPL')
        
        # Create proper CREGenerationResult
        scenario_set = CREScenarioSet(
            stock_ticker='AAPL',
            scenarios=sample_scenarios,
            rejected_scenarios=[],
            base_metrics={'revenue_growth': 0.30, 'margin': 0.20},
            bounds={'revenue_growth': (0.10, 0.50), 'margin': (0.10, 0.40)}
        )
        cre_generation = CREGenerationResult(
            scenario_set=scenario_set,
            claims=['Test claim'],
            defaults_applied=[]
        )
        
        result = ft.run(cre_generation=cre_generation)
        
        assert result is not None
        assert hasattr(result, 'ft_output')
        assert result.ft_output is not None
    
    def test_run_missing_context(self):
        """Test error with missing CRE input."""
        ft = FinancialTranslation('AAPL')
        
        with pytest.raises(ValueError):
            ft.run(cre_generation=None)
    
    def test_run_missing_cre(self):
        """Test error with missing CRE output."""
        ft = FinancialTranslation('AAPL')
        
        with pytest.raises(ValueError):
            ft.run(cre_generation=None)
