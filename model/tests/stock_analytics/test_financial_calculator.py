"""Tests for FinancialCalculator."""
import pandas as pd

from model.calculators.financial_calculator import FinancialCalculator


class TestFinancialCalculator:
    """Test suite for FinancialCalculator."""
    
    def test_initialization(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test calculator initialization."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        
        assert calc.income_stmt.equals(sample_income_stmt)
        assert calc.balance_sheet.equals(sample_balance_sheet)
        assert calc.cashflow.equals(sample_cashflow)
        assert calc.freq == 'quarterly'
        assert calc.suffix == 'q'
    
    def test_profitability_margins(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test profitability margin calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_profitability_margins()
        
        assert 'gross_margin_q' in result
        assert 'operating_margin_q' in result
        assert 'net_profit_margin_q' in result
        
        assert 0 < result['gross_margin_q'] < 1
        assert result['operating_margin_q'] < result['gross_margin_q']
    
    def test_ttm_metrics(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test trailing twelve month metrics."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_ttm_metrics()
        
        assert 'ttm_revenue' in result
        assert 'ttm_net_income' in result
        assert result['ttm_revenue'] > 0
        assert result['ttm_net_income'] > 0
    
    def test_growth_metrics(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test growth metric calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_growth_metrics()
        
        assert 'revenue_growth_q' in result or 'revenue_growth_yoy' in result
        
        if 'revenue_growth_yoy' in result:
            assert isinstance(result['revenue_growth_yoy'], (int, float))
    
    def test_liquidity_ratios(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test liquidity ratio calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_liquidity_ratios()
        
        assert 'current_ratio_q' in result
        assert 'quick_ratio_q' in result
        assert result['current_ratio_q'] > 0
        assert result['quick_ratio_q'] > 0
    
    def test_leverage_ratios(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test leverage ratio calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_leverage_ratios()
        
        assert 'debt_to_equity_q' in result
        assert 'debt_to_assets_q' in result
        assert result['debt_to_equity_q'] >= 0
    
    def test_cashflow_metrics(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test cash flow metric calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_cashflow_metrics()
        
        assert 'free_cash_flow_q' in result
        assert isinstance(result['free_cash_flow_q'], (int, float))
    
    def test_return_metrics(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test return metric calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_return_metrics()
        
        assert 'return_on_equity_q' in result or 'return_on_assets_q' in result
    
    def test_efficiency_ratios(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test efficiency ratio calculations."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_efficiency_ratios()
        
        assert isinstance(result, dict)
        if 'asset_turnover_q' in result:
            assert result['asset_turnover_q'] > 0
    
    def test_calculate_all(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test calculate_all method."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) > 10
        
        expected_keys = ['gross_margin_q', 'current_ratio_q', 'ttm_revenue']
        found_keys = [k for k in expected_keys if k in result]
        assert len(found_keys) > 0
    
    def test_empty_statements(self):
        """Test calculator with empty statements."""
        calc = FinancialCalculator(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        result = calc.calculate_all()
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_yearly_frequency(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test calculator with yearly frequency."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow, freq='yearly')
        
        assert calc.freq == 'yearly'
        assert calc.suffix == 'annual'
        
        result = calc.calculate_profitability_margins()
        if result:
            assert any('_annual' in key for key in result.keys())
    
    def test_selective_metrics(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test calculating selective metric groups."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_all(metrics=['profitability', 'liquidity'])
        
        assert isinstance(result, dict)
        assert 'gross_margin_q' in result
        assert 'current_ratio_q' in result
    
    def test_quality_metrics(self, sample_income_stmt, sample_balance_sheet, sample_cashflow):
        """Test earnings quality metrics."""
        calc = FinancialCalculator(sample_income_stmt, sample_balance_sheet, sample_cashflow)
        result = calc.calculate_quality_metrics()
        
        assert isinstance(result, dict)
