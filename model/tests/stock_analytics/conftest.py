"""Shared test fixtures for stock analytics tests."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock


@pytest.fixture
def sample_price_history():
    """Generate sample price history DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)


@pytest.fixture
def sample_income_stmt():
    """Generate sample income statement DataFrame."""
    periods = pd.date_range(end=datetime.now(), periods=8, freq='Q')
    return pd.DataFrame({
        period: {
            'TotalRevenue': 1000000 * (1.05 ** i),
            'CostOfRevenue': 600000 * (1.05 ** i),
            'GrossProfit': 400000 * (1.05 ** i),
            'OperatingIncome': 200000 * (1.05 ** i),
            'EBITDA': 250000 * (1.05 ** i),
            'NetIncome': 150000 * (1.05 ** i),
            'InterestExpense': 10000
        } for i, period in enumerate(periods)
    })


@pytest.fixture
def sample_balance_sheet():
    """Generate sample balance sheet DataFrame."""
    periods = pd.date_range(end=datetime.now(), periods=8, freq='Q')
    return pd.DataFrame({
        period: {
            'TotalAssets': 5000000 * (1.03 ** i),
            'CurrentAssets': 2000000 * (1.03 ** i),
            'CashAndCashEquivalents': 500000 * (1.03 ** i),
            'Inventory': 300000 * (1.03 ** i),
            'AccountsReceivable': 400000 * (1.03 ** i),
            'TotalLiabilities': 3000000 * (1.03 ** i),
            'CurrentLiabilities': 1000000 * (1.03 ** i),
            'TotalDebt': 1500000 * (1.03 ** i),
            'StockholdersEquity': 2000000 * (1.03 ** i)
        } for i, period in enumerate(periods)
    })


@pytest.fixture
def sample_cashflow():
    """Generate sample cash flow statement DataFrame."""
    periods = pd.date_range(end=datetime.now(), periods=8, freq='Q')
    return pd.DataFrame({
        period: {
            'OperatingCashFlow': 200000 * (1.05 ** i),
            'CapitalExpenditure': -50000 * (1.05 ** i),
            'FreeCashFlow': 150000 * (1.05 ** i)
        } for i, period in enumerate(periods)
    })


@pytest.fixture
def sample_peer_prices():
    """Generate sample peer price data."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(43)
    
    peers = {}
    for ticker in ['PEER1', 'PEER2', 'PEER3']:
        base_price = 100 + np.random.randint(-20, 20)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()
        peers[ticker] = pd.Series(prices, index=dates)
    
    return peers


@pytest.fixture
def sample_index_prices():
    """Generate sample index price data."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(44)
    
    base_price = 1000
    returns = np.random.normal(0.0005, 0.01, len(dates))
    prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Close': prices
    }, index=dates)


@pytest.fixture
def sample_macro_data():
    """Generate sample macro indicator data."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(45)
    
    return {
        'GDP': pd.Series(np.random.normal(2.5, 0.5, len(dates)), index=dates),
        'INFLATION': pd.Series(np.random.normal(2.0, 0.3, len(dates)), index=dates),
        'RATES': pd.Series(np.random.normal(3.0, 0.5, len(dates)), index=dates)
    }


@pytest.fixture
def mock_stock_data_provider():
    """Create mock StockDataProvider."""
    mock = Mock()
    mock.ticker = 'AAPL'
    mock.history.return_value = pd.DataFrame()
    mock.get_income_stmt.return_value = pd.DataFrame()
    mock.get_balance_sheet.return_value = pd.DataFrame()
    mock.get_cashflow.return_value = pd.DataFrame()
    mock.get_info.return_value = {}
    return mock


@pytest.fixture
def mock_index_provider():
    """Create mock IndexProvider."""
    mock = Mock()
    mock.get_index_prices.return_value = {
        'sector_index_history': pd.DataFrame(),
        'broad_index_history': pd.DataFrame()
    }
    mock.get_indices.return_value = {}
    return mock


@pytest.fixture
def mock_peer_discovery_provider():
    """Create mock PeerDiscoveryProvider."""
    mock = Mock()
    mock.get_peers.return_value = []
    mock.get_peers_stock_data.return_value = {}
    return mock
