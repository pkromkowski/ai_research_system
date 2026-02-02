"""Tests for AdvancedTechnicalCalculator."""
import pandas as pd

from model.calculators.advanced_technical_calculator import AdvancedTechnicalCalculator


class TestAdvancedTechnicalCalculator:
    """Test suite for AdvancedTechnicalCalculator."""
    
    def test_initialization(self, sample_price_history):
        """Test calculator initialization."""
        calc = AdvancedTechnicalCalculator(sample_price_history)
        assert calc.history.equals(sample_price_history)
        assert isinstance(calc.calculations, dict)
    
    def test_drawdown_metrics(self, sample_price_history):
        """Test drawdown metric calculations."""
        calc = AdvancedTechnicalCalculator(sample_price_history)
        result = calc.calculate_drawdown_metrics()
        
        assert isinstance(result, dict)
        if 'max_drawdown' in result:
            assert result['max_drawdown'] <= 0
    
    def test_distribution_metrics(self, sample_price_history):
        """Test distribution metric calculations."""
        calc = AdvancedTechnicalCalculator(sample_price_history)
        result = calc.calculate_return_distribution()
        
        assert isinstance(result, dict)
        if 'return_skewness' in result:
            assert isinstance(result['return_skewness'], (int, float, type(None)))
    
    def test_regime_metrics(self, sample_price_history):
        """Test regime metric calculations."""
        calc = AdvancedTechnicalCalculator(sample_price_history)
        result = calc.calculate_volatility_regime()
        
        assert isinstance(result, dict)
    
    def test_stochastic_metrics(self, sample_price_history):
        """Test stochastic oscillator calculations."""
        calc = AdvancedTechnicalCalculator(sample_price_history)
        result = calc.calculate_stochastic()
        
        assert isinstance(result, dict)
    
    def test_calculate_all(self, sample_price_history):
        """Test calculate_all method."""
        calc = AdvancedTechnicalCalculator(sample_price_history)
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_empty_history(self):
        """Test calculator with empty history."""
        calc = AdvancedTechnicalCalculator(pd.DataFrame())
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) == 0
