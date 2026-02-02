"""Tests for ResearchDataProvider."""
from unittest.mock import Mock, MagicMock, patch

from model.services.research_data_provider import ResearchDataProvider


class TestResearchDataProvider:
    """Test suite for ResearchDataProvider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = ResearchDataProvider('AAPL')
        assert provider.ticker == 'AAPL'
    
    @patch('model.services.research_data_provider.yf.Ticker')
    def test_get_recommendations(self, mock_ticker):
        """Test recommendations retrieval."""
        mock_stock = MagicMock()
        mock_stock.recommendations = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = ResearchDataProvider('AAPL')
        result = provider.get_recommendations()
        
        assert result is not None
    
    @patch('model.services.research_data_provider.yf.Ticker')
    def test_get_analyst_price_targets(self, mock_ticker):
        """Test analyst price targets retrieval."""
        mock_stock = MagicMock()
        mock_stock.analyst_price_target = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = ResearchDataProvider('AAPL')
        result = provider.get_analyst_price_targets()
        
        assert result is not None
    
    @patch('model.services.research_data_provider.yf.Ticker')
    def test_get_eps_revisions(self, mock_ticker):
        """Test EPS revisions retrieval."""
        mock_stock = MagicMock()
        mock_stock.eps_revisions = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = ResearchDataProvider('AAPL')
        result = provider.get_eps_revisions()
        
        assert result is not None
