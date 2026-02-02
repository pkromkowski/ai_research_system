"""Tests for StockDataProvider."""
from unittest.mock import Mock, patch, MagicMock

from model.services.stock_data_provider import StockDataProvider


class TestStockDataProvider:
    """Test suite for StockDataProvider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = StockDataProvider('AAPL')
        assert provider.ticker == 'AAPL'
        assert provider.period_length == '1y'
        assert provider.financial_freq == 'quarterly'
    
    def test_initialization_with_params(self):
        """Test provider initialization with custom parameters."""
        provider = StockDataProvider('MSFT', period_length='2y', financial_freq='yearly')
        assert provider.ticker == 'MSFT'
        assert provider.period_length == '2y'
        assert provider.financial_freq == 'yearly'
    
    @patch('model.services.stock_data_provider.yf.Ticker')
    def test_history(self, mock_ticker):
        """Test history data retrieval."""
        mock_stock = MagicMock()
        mock_stock.history.return_value = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = StockDataProvider('AAPL')
        result = provider.history()
        
        mock_stock.history.assert_called_once()
        assert result is not None
    
    @patch('model.services.stock_data_provider.yf.Ticker')
    def test_get_info(self, mock_ticker):
        """Test info data retrieval."""
        mock_stock = MagicMock()
        mock_stock.info = {'sector': 'Technology'}
        mock_ticker.return_value = mock_stock
        
        provider = StockDataProvider('AAPL')
        result = provider.get_info()
        
        assert isinstance(result, dict)
        assert result == {'sector': 'Technology'}
    
    @patch('model.services.stock_data_provider.yf.Ticker')
    def test_get_income_stmt(self, mock_ticker):
        """Test income statement retrieval."""
        mock_stock = MagicMock()
        mock_stock.quarterly_income_stmt = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = StockDataProvider('AAPL', financial_freq='quarterly')
        result = provider.get_income_stmt()
        
        assert result is not None
    
    @patch('model.services.stock_data_provider.yf.Ticker')
    def test_get_balance_sheet(self, mock_ticker):
        """Test balance sheet retrieval."""
        mock_stock = MagicMock()
        mock_stock.quarterly_balance_sheet = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = StockDataProvider('AAPL', financial_freq='quarterly')
        result = provider.get_balance_sheet()
        
        assert result is not None
    
    @patch('model.services.stock_data_provider.yf.Ticker')
    def test_get_cashflow(self, mock_ticker):
        """Test cash flow statement retrieval."""
        mock_stock = MagicMock()
        mock_stock.quarterly_cashflow = Mock()
        mock_ticker.return_value = mock_stock
        
        provider = StockDataProvider('AAPL', financial_freq='quarterly')
        result = provider.get_cashflow()
        
        assert result is not None
    
    def test_ticker_normalization(self):
        """Test ticker symbol normalization."""
        provider = StockDataProvider('aapl')
        assert provider.ticker == 'AAPL'
