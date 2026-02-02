"""Tests for MacroSensitivityCalculator."""
import pandas as pd

from model.calculators.macro_sensitivity_calculator import MacroSensitivityCalculator


class TestMacroSensitivityCalculator:
    """Test suite for MacroSensitivityCalculator."""
    
    def test_initialization(self, sample_price_history):
        """Test calculator initialization."""
        calc = MacroSensitivityCalculator(sample_price_history)
        assert calc.history.equals(sample_price_history)
        assert isinstance(calc.calculations, dict)
    
    def test_calculate_all(self, sample_price_history, sample_macro_data):
        """Test calculate_all method."""
        calc = MacroSensitivityCalculator(sample_price_history)
        result = calc.calculate_all(sample_macro_data)
        
        assert isinstance(result, dict)
    
    def test_empty_history(self):
        """Test calculator with empty history."""
        calc = MacroSensitivityCalculator(pd.DataFrame())
        result = calc.calculate_all({})
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_no_macro_data(self, sample_price_history):
        """Test calculator without macro data."""
        calc = MacroSensitivityCalculator(sample_price_history)
        result = calc.calculate_all({})
        
        assert isinstance(result, dict)
        assert len(result) == 0
