"""Tests for StockAnalyticsOrchestrator (integration-style tests).

These tests create lightweight, deterministic provider objects to exercise
orchestrator plumbing without reaching external services.
"""
import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np
import importlib

# Import the orchestrator lazily inside the test to allow light stubbing of prompts/providers



class TestStockAnalyticsOrchestrator:
    """Test suite for StockAnalyticsOrchestrator."""

    @pytest.fixture
    def sample_price_history(self):
        """Minimal price history DataFrame with daily OHLCV columns."""
        idx = pd.date_range(end=pd.Timestamp("2026-02-01"), periods=300, freq='D')
        close = np.linspace(100, 120, len(idx))
        high = close * 1.01
        low = close * 0.99
        open_ = close * 0.995
        volume = np.random.randint(1000, 5000, size=len(idx))
        df = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=idx)
        return df

    @pytest.fixture
    def sample_income_stmt(self):
        # Rows are metrics, columns are periods (most recent first)
        cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30"), pd.Timestamp("2025-06-30"), pd.Timestamp("2025-03-31")]
        data = {
            'TotalRevenue': [130, 120, 110, 100],
            'GrossProfit': [65, 60, 55, 50],
            'OperatingIncome': [26, 24, 22, 20],
            'EBITDA': [30, 28, 26, 24],
            'NetIncome': [20, 18, 16, 14]
        }
        df = pd.DataFrame(data, index=cols).T
        return df

    @pytest.fixture
    def sample_balance_sheet(self):
        cols = [pd.Timestamp("2025-12-31")]
        data = {
            'TotalAssets': [500],
            'AccountsReceivable': [50]
        }
        df = pd.DataFrame(data, index=cols).T
        return df

    @pytest.fixture
    def sample_cashflow(self):
        cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30"), pd.Timestamp("2025-06-30"), pd.Timestamp("2025-03-31")]
        data = {
            'OperatingCashFlow': [10, 9, 11, 12],
            'CapitalExpenditure': [-2, -2, -2, -3],
            'FreeCashFlow': [8, 7, 9, 9]
        }
        df = pd.DataFrame(data, index=cols).T
        return df

    def test_orchestrator_integration_smoke(self, monkeypatch, sample_price_history, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Integration-style smoke test that monkeypatches providers with deterministic data."""
        # Provide a safe fallback for the prompts helper (Perplexity provider imports it at module import time)
        prompts = importlib.import_module('model.prompts')
        if not hasattr(prompts, 'format_prompt'):
            setattr(prompts, 'format_prompt', lambda *args, **kwargs: "")

        # Import orchestrator after we've ensured prompt helpers exist
        from model.orchestration.stock_analytics_orchestrator import StockAnalyticsOrchestrator

        # Replace StockDataProvider so orchestrator gets deterministic history and statements
        def _stock_data_provider_factory(ticker, period_length, freq):
            svc = Mock()
            svc.history.return_value = sample_price_history
            svc.get_income_stmt.return_value = sample_income_stmt
            svc.get_balance_sheet.return_value = sample_balance_sheet
            svc.get_cashflow.return_value = sample_cashflow
            return svc

        # Replace IndexProvider
        def _index_provider_factory(ticker, period_length):
            svc = Mock()
            # Provide sector and broad index history for volume metrics
            svc.get_index_prices.return_value = {
                'sector_index_history': sample_price_history * 0.9,
                'broad_index_history': sample_price_history * 1.02
            }
            return svc

        # Replace PeerDiscoveryProvider
        def _peer_discovery_factory(ticker, period_length):
            svc = Mock()
            # simple peer series - same dates (use 'Close' column)
            peer_series = { 'AAA': sample_price_history['Close'], 'BBB': sample_price_history['Close'] * 0.95 }
            svc.get_peers_stock_data.return_value = peer_series
            return svc

        # Replace ResearchDataProvider and TenK/Perplexity factories (not exercised deeply here)
        monkeypatch.setattr('model.orchestration.stock_analytics_orchestrator.StockDataProvider', _stock_data_provider_factory)
        monkeypatch.setattr('model.orchestration.stock_analytics_orchestrator.IndexProvider', _index_provider_factory)
        monkeypatch.setattr('model.orchestration.stock_analytics_orchestrator.PeerDiscoveryProvider', _peer_discovery_factory)

        orchestrator = StockAnalyticsOrchestrator('XYZ', ten_k_orchestrator_factory=lambda t, cfg: Mock(), perplexity_provider_factory=lambda t: Mock())

        tech = orchestrator.get_technical_metrics()
        assert isinstance(tech, dict)
        assert tech  # should have metrics

        adv = orchestrator.get_advanced_technical_metrics()
        assert isinstance(adv, dict)

        fin = orchestrator.get_financial_metrics()
        assert isinstance(fin, dict)
        assert fin  # must not be empty with sample statements

        vol = orchestrator.get_volume_positioning_metrics()
        assert isinstance(vol, dict)

        peers = orchestrator.get_peer_metrics()
        assert isinstance(peers, dict)
        assert peers
