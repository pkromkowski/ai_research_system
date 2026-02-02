"""Tests for TechnicalCalculator."""
import numpy as np
import pandas as pd

from model.calculators.technical_calculator import TechnicalCalculator


class TestTechnicalCalculator:
    """Test suite for TechnicalCalculator."""
    
    def test_initialization(self, sample_price_history):
        """Test calculator initialization."""
        calc = TechnicalCalculator(sample_price_history)
        assert calc.history.equals(sample_price_history)
        assert isinstance(calc.calculations, dict)
        assert len(calc.calculations) == 0
    
    def test_returns_calculation(self, sample_price_history):
        """Test basic returns calculation."""
        calc = TechnicalCalculator(sample_price_history)
        returns = calc.returns
        
        assert returns is not None
        assert len(returns) == len(sample_price_history) - 1
        assert isinstance(returns, pd.Series)
    
    def test_volatility_calculation(self, sample_price_history):
        """Test volatility metrics."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_volatility_group()
        
        assert 'historical_volatility_daily' in result or 'rolling_volatility_20d_latest' in result
        if 'historical_volatility_daily' in result:
            assert isinstance(result['historical_volatility_daily'], float)
            assert result['historical_volatility_daily'] > 0
    
    def test_moving_averages(self, sample_price_history):
        """Test moving average calculations."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_moving_averages_group()
        
        assert 'sma_50_current' in result or 'sma_20_current' in result
        assert 'ema_20_current' in result or 'current_price' in result
        if 'sma_50_current' in result:
            assert isinstance(result['sma_50_current'], float)
    
    def test_momentum_indicators(self, sample_price_history):
        """Test momentum indicator calculations."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_momentum_group()
        
        assert 'rsi_current' in result or 'momentum_20d' in result
        if 'macd' in result:
            assert isinstance(result['macd'], (float, type(None)))
        
        rsi = result.get('rsi_current')
        if rsi is not None:
            assert 0 <= rsi <= 100
    
    def test_performance_metrics(self, sample_price_history):
        """Test performance metric calculations."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_performance_group()
        
        assert 'sharpe_ratio' in result or 'max_drawdown' in result
        
        if 'max_drawdown' in result:
            assert result['max_drawdown'] <= 0
    
    def test_calculate_all(self, sample_price_history):
        """Test calculate_all method."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check for some expected keys from different groups
        expected_keys = ['historical_volatility_daily', 'sma_20_current', 'rsi_current', 'current_price']
        found_keys = [k for k in expected_keys if k in result]
        assert len(found_keys) > 0
    
    def test_empty_history(self):
        """Test calculator with empty history."""
        calc = TechnicalCalculator(pd.DataFrame())
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_insufficient_data(self):
        """Test calculator with insufficient data."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D')
        history = pd.DataFrame({
            'Close': np.random.randn(10) + 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        
        calc = TechnicalCalculator(history)
        result = calc.calculate_moving_averages_group()
        
        assert 'sma_200_current' not in result or result['sma_200_current'] is None
    
    def test_volume_metrics(self, sample_price_history):
        """Test volume-based metrics."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_volume_group()
        
        assert isinstance(result, dict)
        if 'volume_sma_20' in result:
            assert result['volume_sma_20'] > 0
    
    def test_price_ranges(self, sample_price_history):
        """Test price range calculations."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_price_ranges_group()
        
        assert isinstance(result, dict)
        if 'price_52w_high' in result:
            assert result['price_52w_high'] > 0
    
    def test_selective_metrics(self, sample_price_history):
        """Test calculating selective metric groups."""
        calc = TechnicalCalculator(sample_price_history)
        result = calc.calculate_all(metrics=['volatility', 'returns'])
        
        assert isinstance(result, dict)
        assert 'historical_volatility_daily' in result or 'daily_return_mean' in result or len(result) > 0
