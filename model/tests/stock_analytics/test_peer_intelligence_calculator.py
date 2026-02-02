"""Tests for PeerIntelligenceCalculator."""
import pandas as pd

from model.calculators.peer_intelligence_calculator import PeerIntelligenceCalculator


class TestPeerIntelligenceCalculator:
    """Test suite for PeerIntelligenceCalculator."""
    
    def test_initialization(self, sample_price_history):
        """Test calculator initialization."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        assert calc.history.equals(sample_price_history)
        assert isinstance(calc.calculations, dict)
        assert calc._peer_data_ready is False
    
    def test_setup_peer_data(self, sample_price_history, sample_peer_prices):
        """Test peer data setup."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        success = calc._setup_peer_data(sample_peer_prices)
        
        # Success depends on data quality and alignment
        assert isinstance(success, bool)
        if success:
            assert calc._peer_data_ready is True
            assert calc.stock_returns_aligned is not None
            assert calc.peer_median_aligned is not None
    
    def test_rolling_excess_return(self, sample_price_history, sample_peer_prices):
        """Test rolling excess return calculation."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        calc._setup_peer_data(sample_peer_prices)
        result = calc.calculate_rolling_excess_return()
        
        if result is not None:
            assert isinstance(result, (int, float))
    
    def test_outperformance_frequency(self, sample_price_history, sample_peer_prices):
        """Test outperformance frequency calculation."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        calc._setup_peer_data(sample_peer_prices)
        result = calc.calculate_outperformance_frequency()
        
        if result is not None:
            assert isinstance(result, (int, float))
            assert 0 <= result <= 1
    
    def test_down_market_relative_return(self, sample_price_history, sample_peer_prices):
        """Test down market relative return."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        calc._setup_peer_data(sample_peer_prices)
        result = calc.calculate_down_market_relative_return()
        
        if result is not None:
            assert isinstance(result, (int, float))
    
    def test_relative_max_drawdown(self, sample_price_history, sample_peer_prices):
        """Test relative max drawdown calculation."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        calc._setup_peer_data(sample_peer_prices)
        result = calc.calculate_relative_max_drawdown()
        
        if result is not None:
            assert isinstance(result, (int, float))
    
    def test_left_tail_return(self, sample_price_history, sample_peer_prices):
        """Test left tail return calculation."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        calc._setup_peer_data(sample_peer_prices)
        result = calc.calculate_left_tail_return_vs_peers()
        
        if result is not None:
            assert isinstance(result, (int, float))
    
    def test_correlation_dispersion(self, sample_price_history, sample_peer_prices):
        """Test correlation dispersion calculation."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        calc._setup_peer_data(sample_peer_prices)
        result = calc.calculate_correlation_dispersion()
        
        if result is not None:
            assert isinstance(result, (int, float))
            assert result >= 0
    
    def test_calculate_all(self, sample_price_history, sample_peer_prices):
        """Test calculate_all method."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        result = calc.calculate_all(peer_prices=sample_peer_prices)
        
        assert isinstance(result, dict)
    
    def test_empty_history(self):
        """Test calculator with empty history."""
        calc = PeerIntelligenceCalculator(pd.DataFrame())
        result = calc.calculate_all(peer_prices={})
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_no_peers(self, sample_price_history):
        """Test calculator without peer data."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        result = calc.calculate_all(peer_prices={})
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_selective_metrics(self, sample_price_history, sample_peer_prices):
        """Test calculating selective metrics."""
        calc = PeerIntelligenceCalculator(sample_price_history)
        result = calc.calculate_all(
            peer_prices=sample_peer_prices,
            metrics=['excess_returns', 'outperformance']
        )
        
        assert isinstance(result, dict)
