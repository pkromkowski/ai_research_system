"""Tests for VolumePositioningCalculator."""
import pandas as pd

from model.calculators.volume_positioning_calculator import VolumePositioningCalculator


class TestVolumePositioningCalculator:
    """Test suite for VolumePositioningCalculator."""
    
    def test_initialization(self, sample_price_history):
        """Test calculator initialization."""
        calc = VolumePositioningCalculator(sample_price_history)
        assert calc.history.equals(sample_price_history)
        assert isinstance(calc.calculations, dict)
    
    def test_volume_metrics(self, sample_price_history):
        """Test volume metric calculations."""
        calc = VolumePositioningCalculator(sample_price_history)
        result = calc.calculate_volume_group()
        
        assert isinstance(result, dict)
        if 'volume_surge' in result:
            assert isinstance(result['volume_surge'], (int, float))
    
    def test_positioning_vs_sector(self, sample_price_history, sample_index_prices):
        """Test positioning metrics vs sector index."""
        calc = VolumePositioningCalculator(sample_price_history)
        result = calc.calculate_positioning_vs_benchmark('sector')
        
        assert isinstance(result, dict)
        if 'beta_sector' in result:
            assert isinstance(result['beta_sector'], (int, float, type(None)))
    
    def test_positioning_vs_broad(self, sample_price_history, sample_index_prices):
        """Test positioning metrics vs broad market index."""
        calc = VolumePositioningCalculator(sample_price_history)
        result = calc.calculate_positioning_vs_benchmark('broad')
        
        assert isinstance(result, dict)
        if 'beta_broad' in result:
            assert isinstance(result['beta_broad'], (int, float, type(None)))
    
    def test_calculate_all(self, sample_price_history, sample_index_prices):
        """Test calculate_all method."""
        calc = VolumePositioningCalculator(sample_price_history)
        result = calc.calculate_all(
            sector_index_history=sample_index_prices,
            broad_index_history=sample_index_prices
        )
        
        assert isinstance(result, dict)
    
    def test_empty_history(self):
        """Test calculator with empty history."""
        calc = VolumePositioningCalculator(pd.DataFrame())
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_no_indices(self, sample_price_history):
        """Test calculator without index data."""
        calc = VolumePositioningCalculator(sample_price_history)
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
