"""Tests for PeerDiscoveryProvider."""
from unittest.mock import Mock, patch, MagicMock

from model.services.peer_discovery_provider import PeerDiscoveryProvider


class TestPeerDiscoveryProvider:
    """Test suite for PeerDiscoveryProvider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = PeerDiscoveryProvider('AAPL')
        assert provider.ticker == 'AAPL'
        assert provider.period_length == '1y'
    
    @patch('model.services.peer_discovery_provider.yf.Ticker')
    def test_get_peers(self, mock_ticker):
        """Test peer ticker retrieval."""
        mock_stock = MagicMock()
        mock_stock.info = {'sector': 'Technology', 'industry': 'Consumer Electronics'}
        mock_ticker.return_value = mock_stock
        
        provider = PeerDiscoveryProvider('AAPL')
        result = provider.get_peers()
        
        assert isinstance(result, dict)
        assert 'peers' in result
        assert isinstance(result['peers'], list)
    
    @patch('model.services.peer_discovery_provider.yf.download')
    def test_get_peers_stock_data(self, mock_download):
        """Test peer stock data retrieval."""
        mock_download.return_value = Mock()
        
        provider = PeerDiscoveryProvider('AAPL')
        result = provider.get_peers_stock_data(peer_tickers=['MSFT', 'GOOGL'])
        
        assert isinstance(result, dict)
