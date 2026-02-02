"""Tests for IndexProvider."""
from unittest.mock import Mock, patch, MagicMock

from model.services.index_provider import IndexProvider


class TestIndexProvider:
    """Test suite for IndexProvider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = IndexProvider('AAPL')
        assert provider.ticker == 'AAPL'
        assert provider.period_length == '1y'
    
    @patch('model.services.index_provider.yf.Ticker')
    def test_get_indices(self, mock_ticker):
        """Test index retrieval."""
        mock_stock = MagicMock()
        mock_stock.info = {'sector': 'Technology'}
        mock_ticker.return_value = mock_stock
        
        provider = IndexProvider('AAPL')
        result = provider.get_indices()
        
        assert isinstance(result, dict)
        assert 'sector' in result or 'sector_index' in result
    
    @patch('model.services.index_provider.yf.download')
    def test_get_index_prices(self, mock_download):
        """Test index price data retrieval."""
        mock_download.return_value = Mock()
        
        provider = IndexProvider('AAPL')
        result = provider.get_index_prices()
        
        assert isinstance(result, dict)
