import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

from model.calculators.calculator_base import CalculatorBase


class FinancialCalculator(CalculatorBase):
    """Calculates financial metrics from income statements, balance sheets, and cash flow statements.
    
    Supports quarterly, yearly, and trailing twelve month data.
    Metric names use dynamic suffixes based on frequency (e.g., 'gross_margin_q').
    """
    DAYS_PER_YEAR = 365
    MIN_PERIODS_GROWTH = 2
    MIN_PERIODS_YOY = 5
    QUARTERS_PER_YEAR = 4
    
    def __init__(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, 
                 cashflow: pd.DataFrame, freq: str = 'quarterly'):
        """Initialize calculator with financial statements.
        
        Args:
            income_stmt: Income statement DataFrame (rows=metrics, cols=periods)
            balance_sheet: Balance sheet DataFrame (rows=metrics, cols=periods)
            cashflow: Cash flow statement DataFrame (rows=metrics, cols=periods)
            freq: Data frequency - 'quarterly', 'yearly', or 'trailing'F
        """
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.cashflow = cashflow
        self.freq = freq
        self.calculations: Dict[str, Any] = {}
        
        self._suffixes = {
            'trailing': 'ttm',
            'yearly': 'annual',
            'quarterly': 'q'
        }
        self.suffix = self._suffixes.get(freq, 'q')
        
        self._validate_inputs()
        self._aligned_periods = self._get_aligned_periods()
    
    def _validate_inputs(self) -> None:
        """Validate input DataFrames have expected structure."""
        for name, df in [('income_stmt', self.income_stmt), 
                         ('balance_sheet', self.balance_sheet),
                         ('cashflow', self.cashflow)]:
            if df is None:
                setattr(self, name, pd.DataFrame())
    
    def _get_aligned_periods(self) -> List:
        """Get common periods across all statements for cross-statement calculations."""
        if self.income_stmt.empty or self.balance_sheet.empty or self.cashflow.empty:
            return []
        
        income_periods = set(self.income_stmt.columns)
        balance_periods = set(self.balance_sheet.columns)
        cashflow_periods = set(self.cashflow.columns)
        
        common = income_periods & balance_periods & cashflow_periods
        if not common:
            return []
        
        return sorted(list(common), reverse=True)
    
    def _safe_get(self, df: pd.DataFrame, row: str, col, default: float = None) -> Optional[float]:
        """Safely get a value from DataFrame with validation."""
        if df.empty or row not in df.index or col not in df.columns:
            return default
        val = df.loc[row, col]
        if pd.isna(val) or (isinstance(val, float) and not np.isfinite(val)):
            return default
        return float(val)
    
    def _safe_divide(self, numerator: Optional[float], denominator: Optional[float], 
                     allow_negative_denom: bool = False) -> Optional[float]:
        """Safely divide two values with proper checks."""
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        if not allow_negative_denom and denominator < 0:
            return None
        result = numerator / denominator
        if not np.isfinite(result):
            return None
        return result
    
    def _has_income(self) -> bool:
        """True if income statement is present."""
        return not self.income_stmt.empty
    
    def _has_balance(self) -> bool:
        """True if balance sheet is present."""
        return not self.balance_sheet.empty
    
    def _has_cashflow(self) -> bool:
        """True if cashflow statement is present."""
        return not self.cashflow.empty
    
    def _require_sheets(self, *names: str) -> bool:
        """Return True if all named sheet attributes are non-empty."""
        for name in names:
            df = getattr(self, name, None)
            if df is None or getattr(df, 'empty', True):
                return False
        return True
    
    def _latest_col(self, df: pd.DataFrame):
        """Return the most recent column from a DataFrame, or None if empty."""
        return df.columns[0] if not df.empty else None
    
    def _require_income_cols(self, min_cols: int) -> bool:
        """Return True if income statement has at least min_cols columns."""
        return self._has_income() and len(self.income_stmt.columns) >= min_cols
    
    def _get_ttm_value(self, df: pd.DataFrame, row: str) -> Optional[float]:
        """Get TTM value with proper handling based on frequency.
        
        - quarterly: Sum last 4 quarters
        - yearly: Use latest year value (already annual)
        - trailing: Use the single trailing value directly
        """
        if df.empty or row not in df.index:
            return None
        
        if self.freq == 'trailing':
            return self._safe_get(df, row, df.columns[0])
        
        elif self.freq == 'yearly':
            return self._safe_get(df, row, df.columns[0])
        
        else:
            if len(df.columns) < self.QUARTERS_PER_YEAR:
                return None
            
            values = pd.to_numeric(df.loc[row].iloc[:self.QUARTERS_PER_YEAR], errors='coerce')
            if values.isna().any():
                return None
            return float(values.sum())
    
    def _get_annualized_revenue(self) -> Optional[float]:
        """Get annualized revenue based on frequency."""
        if self.freq == 'quarterly':
            return self._get_ttm_value(self.income_stmt, 'TotalRevenue')
        else:
            return self._safe_get(self.income_stmt, 'TotalRevenue', 
                                  self.income_stmt.columns[0] if not self.income_stmt.empty else None)
    
    def calculate_gross_margin(self) -> Optional[float]:
        """Calculate gross margin for the latest period."""
        if not self._has_income():
            return None
        
        latest_col = self._latest_col(self.income_stmt)
        revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_col)
        if revenue is None or revenue <= 0:
            return None
        
        gross_profit = self._safe_get(self.income_stmt, 'GrossProfit', latest_col)
        margin = self._safe_divide(gross_profit, revenue)
        return self._store_result(f'gross_margin_{self.suffix}', margin)
    
    def calculate_operating_margin(self) -> Optional[float]:
        """Calculate operating margin for the latest period."""
        if not self._has_income():
            return None
        
        latest_col = self._latest_col(self.income_stmt)
        revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_col)
        if revenue is None or revenue <= 0:
            return None
        
        operating_income = self._safe_get(self.income_stmt, 'OperatingIncome', latest_col)
        margin = self._safe_divide(operating_income, revenue)
        return self._store_result(f'operating_margin_{self.suffix}', margin)
    
    def calculate_ebitda_margin(self) -> Optional[float]:
        """Calculate EBITDA margin for the latest period."""
        if not self._has_income():
            return None
        
        latest_col = self._latest_col(self.income_stmt)
        revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_col)
        if revenue is None or revenue <= 0:
            return None
        
        ebitda = self._safe_get(self.income_stmt, 'EBITDA', latest_col)
        margin = self._safe_divide(ebitda, revenue)
        return self._store_result(f'ebitda_margin_{self.suffix}', margin)
    
    def calculate_net_profit_margin(self) -> Optional[float]:
        """Calculate net profit margin for the latest period."""
        if not self._has_income():
            return None
        
        latest_col = self._latest_col(self.income_stmt)
        revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_col)
        if revenue is None or revenue <= 0:
            return None
        
        net_income = self._safe_get(self.income_stmt, 'NetIncome', latest_col)
        margin = self._safe_divide(net_income, revenue)
        return self._store_result(f'net_profit_margin_{self.suffix}', margin)
    
    def calculate_profitability_margins(self) -> Dict[str, Any]:
        """Calculate all profitability margins for the latest period."""
        return self._collect_new_results([
            self.calculate_gross_margin,
            self.calculate_operating_margin,
            self.calculate_ebitda_margin,
            self.calculate_net_profit_margin
        ])
    
    def calculate_ttm_revenue(self) -> Optional[float]:
        """Calculate trailing twelve month revenue."""
        if not self._has_income():
            return None
        
        ttm_revenue = self._get_ttm_value(self.income_stmt, 'TotalRevenue')
        if ttm_revenue is not None and ttm_revenue > 0:
            return self._store_result('ttm_revenue', ttm_revenue)
        return None
    
    def calculate_ttm_net_income(self) -> Optional[float]:
        """Calculate trailing twelve month net income."""
        if not self._has_income():
            return None
        
        ttm_ni = self._get_ttm_value(self.income_stmt, 'NetIncome')
        return self._store_result('ttm_net_income', ttm_ni)
    
    def calculate_ttm_ebitda(self) -> Optional[float]:
        """Calculate trailing twelve month EBITDA."""
        if not self._has_income():
            return None
        
        ttm_ebitda = self._get_ttm_value(self.income_stmt, 'EBITDA')
        return self._store_result('ttm_ebitda', ttm_ebitda)
    
    def calculate_ttm_net_margin(self) -> Optional[float]:
        """Calculate trailing twelve month net margin."""
        ttm_revenue = self.calculations.get('ttm_revenue') or self.calculate_ttm_revenue()
        ttm_ni = self.calculations.get('ttm_net_income') or self.calculate_ttm_net_income()
        
        if ttm_revenue and ttm_revenue > 0 and ttm_ni is not None:
            margin = self._safe_divide(ttm_ni, ttm_revenue)
            return self._store_result('ttm_net_margin', margin)
        return None
    
    def calculate_ttm_ebitda_margin(self) -> Optional[float]:
        """Calculate trailing twelve month EBITDA margin."""
        ttm_revenue = self.calculations.get('ttm_revenue') or self.calculate_ttm_revenue()
        ttm_ebitda = self.calculations.get('ttm_ebitda') or self.calculate_ttm_ebitda()
        
        if ttm_revenue and ttm_revenue > 0 and ttm_ebitda is not None:
            margin = self._safe_divide(ttm_ebitda, ttm_revenue)
            return self._store_result('ttm_ebitda_margin', margin)
        return None
    
    def calculate_ttm_operating_cashflow(self) -> Optional[float]:
        """Calculate trailing twelve month operating cash flow."""
        if not self._has_cashflow():
            return None
        
        ttm_ocf = self._get_ttm_value(self.cashflow, 'OperatingCashFlow')
        return self._store_result('ttm_operating_cashflow', ttm_ocf)
    
    def calculate_ttm_free_cashflow(self) -> Optional[float]:
        """Calculate trailing twelve month free cash flow."""
        if not self._has_cashflow():
            return None
        
        ttm_ocf = self.calculations.get('ttm_operating_cashflow') or self.calculate_ttm_operating_cashflow()
        ttm_capex = self._get_ttm_value(self.cashflow, 'CapitalExpenditure')
        
        if ttm_ocf is not None and ttm_capex is not None:
            ttm_fcf = ttm_ocf + ttm_capex if ttm_capex < 0 else ttm_ocf - ttm_capex
            return self._store_result('ttm_free_cashflow', ttm_fcf)
        return None
    
    def calculate_ttm_metrics(self) -> Dict[str, Any]:
        """Calculate all trailing twelve month metrics."""
        return self._collect_new_results([
            self.calculate_ttm_revenue,
            self.calculate_ttm_net_income,
            self.calculate_ttm_ebitda,
            self.calculate_ttm_net_margin,
            self.calculate_ttm_ebitda_margin,
            self.calculate_ttm_operating_cashflow,
            self.calculate_ttm_free_cashflow
        ])
    
    def calculate_revenue_growth(self) -> Optional[float]:
        """Calculate period-over-period revenue growth."""
        if self.freq == 'trailing' or not self._require_income_cols(self.MIN_PERIODS_GROWTH):
            return None
        if 'TotalRevenue' not in self.income_stmt.index:
            return None
        
        latest = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[0])
        prior = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[1])
        
        if latest is None or prior is None:
            return None
        
        growth = self._safe_divide(latest - prior, prior)
        return self._store_result(f'revenue_growth_{self.suffix}', growth)
    
    def calculate_revenue_growth_yoy(self) -> Optional[float]:
        """Calculate year-over-year revenue growth (quarterly data only)."""
        if self.freq != 'quarterly' or not self._require_income_cols(self.MIN_PERIODS_YOY):
            return None
        if 'TotalRevenue' not in self.income_stmt.index:
            return None
        
        latest = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[0])
        year_ago = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[4])
        
        if latest is None or year_ago is None:
            return None
        
        growth = self._safe_divide(latest - year_ago, year_ago)
        return self._store_result('revenue_growth_yoy', growth)
    
    def calculate_earnings_growth(self) -> Optional[float]:
        """Calculate period-over-period earnings growth."""
        if self.freq == 'trailing' or not self._require_income_cols(self.MIN_PERIODS_GROWTH):
            return None
        if 'NetIncome' not in self.income_stmt.index:
            return None
        
        latest = self._safe_get(self.income_stmt, 'NetIncome', self.income_stmt.columns[0])
        prior = self._safe_get(self.income_stmt, 'NetIncome', self.income_stmt.columns[1])
        
        if latest is None or prior is None:
            return None
        
        growth = self._safe_divide(latest - prior, abs(prior) if prior else None)
        return self._store_result(f'earnings_growth_{self.suffix}', growth)
    
    def calculate_earnings_growth_yoy(self) -> Optional[float]:
        """Calculate year-over-year earnings growth (quarterly data only)."""
        if self.freq != 'quarterly' or not self._require_income_cols(self.MIN_PERIODS_YOY):
            return None
        if 'NetIncome' not in self.income_stmt.index:
            return None
        
        latest = self._safe_get(self.income_stmt, 'NetIncome', self.income_stmt.columns[0])
        year_ago = self._safe_get(self.income_stmt, 'NetIncome', self.income_stmt.columns[4])
        
        if latest is None or year_ago is None:
            return None
        
        growth = self._safe_divide(latest - year_ago, abs(year_ago) if year_ago else None)
        return self._store_result('earnings_growth_yoy', growth)
    
    def calculate_growth_metrics(self) -> Dict[str, Any]:
        """Calculate all growth metrics."""
        return self._collect_new_results([
            self.calculate_revenue_growth,
            self.calculate_revenue_growth_yoy,
            self.calculate_earnings_growth,
            self.calculate_earnings_growth_yoy
        ])
    
    def calculate_asset_turnover(self) -> Optional[float]:
        """Calculate asset turnover ratio."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_bs_col)
        annualized_revenue = self._get_annualized_revenue()
        
        if total_assets and total_assets > 0 and annualized_revenue:
            turnover = self._safe_divide(annualized_revenue, total_assets)
            return self._store_result(f'asset_turnover_{self.suffix}', turnover)
        return None
    
    def calculate_days_sales_outstanding(self) -> Optional[float]:
        """Calculate days sales outstanding (DSO)."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        receivables = self._safe_get(self.balance_sheet, 'AccountsReceivable', latest_bs_col)
        annualized_revenue = self._get_annualized_revenue()
        
        if receivables and annualized_revenue and annualized_revenue > 0:
            daily_revenue = annualized_revenue / self.DAYS_PER_YEAR
            dso = self._safe_divide(receivables, daily_revenue)
            return self._store_result('days_sales_outstanding', dso)
        return None
    
    def calculate_inventory_turnover(self) -> Optional[float]:
        """Calculate inventory turnover ratio."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        inventory = self._safe_get(self.balance_sheet, 'Inventory', latest_bs_col)
        cogs = self._get_ttm_value(self.income_stmt, 'CostOfRevenue')
        
        if inventory and inventory > 0 and cogs:
            turnover = self._safe_divide(cogs, inventory)
            return self._store_result('inventory_turnover', turnover)
        return None
    
    def calculate_efficiency_ratios(self) -> Dict[str, Any]:
        """Calculate all efficiency and turnover ratios."""
        return self._collect_new_results([
            self.calculate_asset_turnover,
            self.calculate_days_sales_outstanding,
            self.calculate_inventory_turnover
        ])
    
    def calculate_current_ratio(self) -> Optional[float]:
        """Calculate current ratio."""
        if not self._has_balance():
            return None
        
        latest_col = self._latest_col(self.balance_sheet)
        current_assets = self._safe_get(self.balance_sheet, 'CurrentAssets', latest_col)
        current_liabilities = self._safe_get(self.balance_sheet, 'CurrentLiabilities', latest_col)
        
        if current_liabilities and current_liabilities > 0 and current_assets:
            ratio = self._safe_divide(current_assets, current_liabilities)
            return self._store_result(f'current_ratio_{self.suffix}', ratio)
        return None
    
    def calculate_quick_ratio(self) -> Optional[float]:
        """Calculate quick ratio (acid test)."""
        if not self._has_balance():
            return None
        
        latest_col = self._latest_col(self.balance_sheet)
        current_assets = self._safe_get(self.balance_sheet, 'CurrentAssets', latest_col)
        current_liabilities = self._safe_get(self.balance_sheet, 'CurrentLiabilities', latest_col)
        inventory = self._safe_get(self.balance_sheet, 'Inventory', latest_col) or 0
        
        if current_liabilities and current_liabilities > 0 and current_assets:
            quick_assets = current_assets - inventory
            ratio = self._safe_divide(quick_assets, current_liabilities)
            return self._store_result(f'quick_ratio_{self.suffix}', ratio)
        return None
    
    def calculate_cash_ratio(self) -> Optional[float]:
        """Calculate cash ratio."""
        if not self._has_balance():
            return None
        
        latest_col = self._latest_col(self.balance_sheet)
        cash = self._safe_get(self.balance_sheet, 'CashAndCashEquivalents', latest_col)
        current_liabilities = self._safe_get(self.balance_sheet, 'CurrentLiabilities', latest_col)
        
        if current_liabilities and current_liabilities > 0 and cash:
            ratio = self._safe_divide(cash, current_liabilities)
            return self._store_result(f'cash_ratio_{self.suffix}', ratio)
        return None
    
    def calculate_liquidity_ratios(self) -> Dict[str, Any]:
        """Calculate all liquidity ratios."""
        return self._collect_new_results([
            self.calculate_current_ratio,
            self.calculate_quick_ratio,
            self.calculate_cash_ratio
        ])
    
    def calculate_debt_to_equity(self) -> Optional[float]:
        """Calculate debt to equity ratio."""
        if not self._has_balance():
            return None
        
        latest_col = self._latest_col(self.balance_sheet)
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_col)
        equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', latest_col)
        
        total_debt = self._safe_get(self.balance_sheet, 'TotalDebt', latest_col)
        if total_debt is None and total_assets and equity:
            total_debt = total_assets - equity
        
        if total_debt is not None and equity and equity > 0:
            ratio = self._safe_divide(total_debt, equity)
            return self._store_result(f'debt_to_equity_{self.suffix}', ratio)
        return None
    
    def calculate_debt_to_assets(self) -> Optional[float]:
        """Calculate debt to assets ratio."""
        if not self._has_balance():
            return None
        
        latest_col = self._latest_col(self.balance_sheet)
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_col)
        equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', latest_col)
        
        total_debt = self._safe_get(self.balance_sheet, 'TotalDebt', latest_col)
        if total_debt is None and total_assets and equity:
            total_debt = total_assets - equity
        
        if total_debt is not None and total_assets and total_assets > 0:
            ratio = self._safe_divide(total_debt, total_assets)
            return self._store_result(f'debt_to_assets_{self.suffix}', ratio)
        return None
    
    def calculate_equity_multiplier(self) -> Optional[float]:
        """Calculate equity multiplier (financial leverage)."""
        if not self._has_balance():
            return None
        
        latest_col = self._latest_col(self.balance_sheet)
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_col)
        equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', latest_col)
        
        if total_assets and equity and equity > 0:
            ratio = self._safe_divide(total_assets, equity)
            return self._store_result(f'equity_multiplier_{self.suffix}', ratio)
        return None
    
    def calculate_interest_coverage_ratio(self) -> Optional[float]:
        """Calculate interest coverage ratio."""
        if not self._has_income():
            return None
        
        ebit = self._get_ttm_value(self.income_stmt, 'OperatingIncome')
        interest_expense = self._get_ttm_value(self.income_stmt, 'InterestExpense')
        
        if ebit and interest_expense and interest_expense != 0:
            ratio = self._safe_divide(ebit, abs(interest_expense))
            return self._store_result('interest_coverage_ratio', ratio)
        return None
    
    def calculate_leverage_ratios(self) -> Dict[str, Any]:
        """Calculate all leverage and solvency ratios."""
        return self._collect_new_results([
            self.calculate_debt_to_equity,
            self.calculate_debt_to_assets,
            self.calculate_equity_multiplier,
            self.calculate_interest_coverage_ratio
        ])
    
    def calculate_ocf_to_net_income(self) -> Optional[float]:
        """Calculate operating cash flow to net income ratio."""
        if not self._require_sheets('cashflow', 'income_stmt'):
            return None
        
        latest_cf_col = self._latest_col(self.cashflow)
        latest_income_col = self._latest_col(self.income_stmt)
        
        ocf = self._safe_get(self.cashflow, 'OperatingCashFlow', latest_cf_col)
        net_income = self._safe_get(self.income_stmt, 'NetIncome', latest_income_col)
        
        if ocf is not None and net_income and net_income != 0:
            ratio = self._safe_divide(ocf, net_income, allow_negative_denom=True)
            return self._store_result(f'ocf_to_net_income_{self.suffix}', ratio)
        return None
    
    def calculate_free_cash_flow(self) -> Optional[float]:
        """Calculate free cash flow for the latest period."""
        if not self._has_cashflow():
            return None
        
        latest_col = self._latest_col(self.cashflow)
        ocf = self._safe_get(self.cashflow, 'OperatingCashFlow', latest_col)
        capex = self._safe_get(self.cashflow, 'CapitalExpenditure', latest_col)
        
        if ocf is not None and capex is not None:
            fcf = ocf + capex if capex < 0 else ocf - capex
            return self._store_result(f'free_cash_flow_{self.suffix}', fcf)
        return None
    
    def calculate_fcf_margin(self) -> Optional[float]:
        """Calculate free cash flow margin."""
        fcf = self.calculations.get(f'free_cash_flow_{self.suffix}') or self.calculate_free_cash_flow()
        
        if fcf is not None and self._has_income():
            latest_income_col = self._latest_col(self.income_stmt)
            revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_income_col)
            if revenue and revenue > 0:
                margin = self._safe_divide(fcf, revenue)
                return self._store_result(f'fcf_margin_{self.suffix}', margin)
        return None
    
    def calculate_capex_to_revenue(self) -> Optional[float]:
        """Calculate capital expenditure to revenue ratio."""
        if not self._require_sheets('cashflow', 'income_stmt'):
            return None
        
        latest_cf_col = self._latest_col(self.cashflow)
        latest_income_col = self._latest_col(self.income_stmt)
        
        capex = self._safe_get(self.cashflow, 'CapitalExpenditure', latest_cf_col)
        revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_income_col)
        
        if capex is not None and revenue and revenue > 0:
            ratio = self._safe_divide(abs(capex), revenue)
            return self._store_result(f'capex_to_revenue_{self.suffix}', ratio)
        return None
    
    def calculate_cashflow_metrics(self) -> Dict[str, Any]:
        """Calculate all cash flow metrics for the latest period."""
        return self._collect_new_results([
            self.calculate_ocf_to_net_income,
            self.calculate_free_cash_flow,
            self.calculate_fcf_margin,
            self.calculate_capex_to_revenue
        ])
    
    def calculate_return_on_assets(self) -> Optional[float]:
        """Calculate return on assets."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        annualized_ni = self._get_ttm_value(self.income_stmt, 'NetIncome')
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_bs_col)
        
        if annualized_ni is not None and total_assets and total_assets > 0:
            roa = self._safe_divide(annualized_ni, total_assets)
            return self._store_result(f'return_on_assets_{self.suffix}', roa)
        return None
    
    def calculate_return_on_equity(self) -> Optional[float]:
        """Calculate return on equity."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        annualized_ni = self._get_ttm_value(self.income_stmt, 'NetIncome')
        equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', latest_bs_col)
        
        if annualized_ni is not None and equity and equity > 0:
            roe = self._safe_divide(annualized_ni, equity)
            return self._store_result(f'return_on_equity_{self.suffix}', roe)
        return None
    
    def calculate_roic(self) -> Optional[float]:
        """Calculate return on invested capital."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        annualized_oi = self._get_ttm_value(self.income_stmt, 'OperatingIncome')
        
        equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', latest_bs_col)
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_bs_col)
        total_debt = self._safe_get(self.balance_sheet, 'TotalDebt', latest_bs_col)
        cash = self._safe_get(self.balance_sheet, 'CashAndCashEquivalents', latest_bs_col) or 0
        
        if annualized_oi is not None and equity:
            if total_debt is None:
                total_debt = (total_assets - equity) if total_assets else 0
            
            invested_capital = equity + (total_debt or 0) - cash
            if invested_capital > 0:
                roic = self._safe_divide(annualized_oi, invested_capital)
                return self._store_result(f'roic_{self.suffix}', roic)
        return None
    
    def calculate_return_metrics(self) -> Dict[str, Any]:
        """Calculate all return on capital metrics."""
        return self._collect_new_results([
            self.calculate_return_on_assets,
            self.calculate_return_on_equity,
            self.calculate_roic
        ])
    
    def calculate_operating_leverage(self) -> Optional[float]:
        """Calculate operating leverage."""
        if not self._has_income():
            return None
        
        latest_col = self._latest_col(self.income_stmt)
        ebitda = self._safe_get(self.income_stmt, 'EBITDA', latest_col)
        oi = self._safe_get(self.income_stmt, 'OperatingIncome', latest_col)
        
        if ebitda and oi and oi != 0:
            ratio = self._safe_divide(ebitda, oi, allow_negative_denom=True)
            return self._store_result('operating_leverage', ratio)
        return None
    
    def calculate_accruals_ratio(self) -> Optional[float]:
        """Calculate accruals ratio (earnings quality indicator)."""
        if not self._require_sheets('income_stmt', 'cashflow', 'balance_sheet'):
            return None
        
        latest_income_col = self._latest_col(self.income_stmt)
        latest_cf_col = self._latest_col(self.cashflow)
        latest_bs_col = self._latest_col(self.balance_sheet)
        
        ni = self._safe_get(self.income_stmt, 'NetIncome', latest_income_col)
        ocf = self._safe_get(self.cashflow, 'OperatingCashFlow', latest_cf_col)
        total_assets = self._safe_get(self.balance_sheet, 'TotalAssets', latest_bs_col)
        
        if ni is not None and ocf is not None and total_assets and total_assets > 0:
            accruals = self._safe_divide(ni - ocf, total_assets)
            return self._store_result('accruals_ratio', accruals)
        return None
    
    def calculate_fcf_to_ni_ratio(self) -> Optional[float]:
        """Calculate free cash flow to net income ratio."""
        fcf = self.calculations.get(f'free_cash_flow_{self.suffix}') or self.calculate_free_cash_flow()
        
        if fcf is not None and self._has_income():
            latest_col = self._latest_col(self.income_stmt)
            ni = self._safe_get(self.income_stmt, 'NetIncome', latest_col)
            if ni and ni != 0:
                ratio = self._safe_divide(fcf, ni, allow_negative_denom=True)
                return self._store_result('fcf_to_ni_ratio', ratio)
        return None
    
    def calculate_working_capital_pct_revenue(self) -> Optional[float]:
        """Calculate working capital as percentage of revenue."""
        if not self._require_sheets('balance_sheet', 'income_stmt'):
            return None
        
        latest_bs_col = self._latest_col(self.balance_sheet)
        latest_income_col = self._latest_col(self.income_stmt)
        
        ca = self._safe_get(self.balance_sheet, 'CurrentAssets', latest_bs_col)
        cl = self._safe_get(self.balance_sheet, 'CurrentLiabilities', latest_bs_col)
        revenue = self._safe_get(self.income_stmt, 'TotalRevenue', latest_income_col)
        
        if ca is not None and cl is not None and revenue and revenue > 0:
            wc = ca - cl
            ratio = self._safe_divide(wc, revenue)
            return self._store_result('working_capital_pct_revenue', ratio)
        return None
    
    def calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate all earnings quality and financial strength metrics."""
        return self._collect_new_results([
            self.calculate_operating_leverage,
            self.calculate_accruals_ratio,
            self.calculate_fcf_to_ni_ratio,
            self.calculate_working_capital_pct_revenue
        ])
    
    def calculate_gross_margin_yoy_change(self) -> Optional[float]:
        """Calculate year-over-year change in gross margin (percentage points)."""
        if self.freq == 'trailing' or not self._has_income():
            return None
        
        n_cols = len(self.income_stmt.columns)
        periods_back = 4 if self.freq == 'quarterly' else 1
        
        if n_cols <= periods_back:
            return None
        
        current_col = self.income_stmt.columns[0]
        prior_col = self.income_stmt.columns[periods_back]
        
        current_revenue = self._safe_get(self.income_stmt, 'TotalRevenue', current_col)
        current_gross = self._safe_get(self.income_stmt, 'GrossProfit', current_col)
        prior_revenue = self._safe_get(self.income_stmt, 'TotalRevenue', prior_col)
        prior_gross = self._safe_get(self.income_stmt, 'GrossProfit', prior_col)
        
        if not all([current_revenue, current_gross, prior_revenue, prior_gross]):
            return None
        if current_revenue <= 0 or prior_revenue <= 0:
            return None
        
        current_margin = current_gross / current_revenue
        prior_margin = prior_gross / prior_revenue
        change = current_margin - prior_margin
        
        return self._store_result('gross_margin_yoy_change', change * 100)
    
    def calculate_operating_margin_yoy_change(self) -> Optional[float]:
        """Calculate year-over-year change in operating margin (percentage points)."""
        if self.freq == 'trailing' or not self._has_income():
            return None
        
        n_cols = len(self.income_stmt.columns)
        periods_back = 4 if self.freq == 'quarterly' else 1
        
        if n_cols <= periods_back:
            return None
        
        current_col = self.income_stmt.columns[0]
        prior_col = self.income_stmt.columns[periods_back]
        
        current_revenue = self._safe_get(self.income_stmt, 'TotalRevenue', current_col)
        current_op_income = self._safe_get(self.income_stmt, 'OperatingIncome', current_col)
        prior_revenue = self._safe_get(self.income_stmt, 'TotalRevenue', prior_col)
        prior_op_income = self._safe_get(self.income_stmt, 'OperatingIncome', prior_col)
        
        if not all([current_revenue, current_op_income is not None, 
                   prior_revenue, prior_op_income is not None]):
            return None
        if current_revenue <= 0 or prior_revenue <= 0:
            return None
        
        current_margin = current_op_income / current_revenue
        prior_margin = prior_op_income / prior_revenue
        change = current_margin - prior_margin
        
        return self._store_result('operating_margin_yoy_change', change * 100)
    
    def calculate_roe_yoy_change(self) -> Optional[float]:
        """Calculate year-over-year change in return on equity (percentage points)."""
        if self.freq == 'trailing' or not self._require_sheets('income_stmt', 'balance_sheet'):
            return None
        
        n_inc_cols = len(self.income_stmt.columns)
        n_bs_cols = len(self.balance_sheet.columns)
        periods_back = 4 if self.freq == 'quarterly' else 1
        
        if n_inc_cols <= periods_back or n_bs_cols <= periods_back:
            return None
        
        current_ni = self._safe_get(self.income_stmt, 'NetIncome', self.income_stmt.columns[0])
        current_equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', self.balance_sheet.columns[0])
        
        prior_ni = self._safe_get(self.income_stmt, 'NetIncome', self.income_stmt.columns[periods_back])
        prior_equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', self.balance_sheet.columns[periods_back])
        
        if not all([current_ni is not None, current_equity, prior_ni is not None, prior_equity]):
            return None
        if current_equity <= 0 or prior_equity <= 0:
            return None
        
        current_roe = current_ni / current_equity
        prior_roe = prior_ni / prior_equity
        change = current_roe - prior_roe
        
        return self._store_result('roe_yoy_change', change * 100)
    
    def calculate_debt_to_equity_yoy_change(self) -> Optional[float]:
        """Calculate year-over-year change in debt-to-equity ratio."""
        if self.freq == 'trailing' or not self._has_balance():
            return None
        
        n_cols = len(self.balance_sheet.columns)
        periods_back = 4 if self.freq == 'quarterly' else 1
        
        if n_cols <= periods_back:
            return None
        
        current_col = self.balance_sheet.columns[0]
        prior_col = self.balance_sheet.columns[periods_back]
        
        current_debt = self._safe_get(self.balance_sheet, 'TotalDebt', current_col)
        current_equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', current_col)
        prior_debt = self._safe_get(self.balance_sheet, 'TotalDebt', prior_col)
        prior_equity = self._safe_get(self.balance_sheet, 'StockholdersEquity', prior_col)
        
        if not all([current_debt is not None, current_equity, prior_debt is not None, prior_equity]):
            return None
        if current_equity <= 0 or prior_equity <= 0:
            return None
        
        current_de = current_debt / current_equity
        prior_de = prior_debt / prior_equity
        change = current_de - prior_de
        
        return self._store_result('debt_to_equity_yoy_change', change)
    
    def calculate_fcf_growth_yoy(self) -> Optional[float]:
        """Calculate year-over-year free cash flow growth rate."""
        if self.freq == 'trailing' or not self._has_cashflow():
            return None
        
        n_cols = len(self.cashflow.columns)
        periods_back = 4 if self.freq == 'quarterly' else 1
        
        if n_cols <= periods_back:
            return None
        
        current_col = self.cashflow.columns[0]
        prior_col = self.cashflow.columns[periods_back]
        
        current_ocf = self._safe_get(self.cashflow, 'OperatingCashFlow', current_col)
        current_capex = self._safe_get(self.cashflow, 'CapitalExpenditure', current_col)
        prior_ocf = self._safe_get(self.cashflow, 'OperatingCashFlow', prior_col)
        prior_capex = self._safe_get(self.cashflow, 'CapitalExpenditure', prior_col)
        
        if not all([current_ocf is not None, current_capex is not None,
                   prior_ocf is not None, prior_capex is not None]):
            return None
        
        current_fcf = current_ocf + current_capex if current_capex < 0 else current_ocf - abs(current_capex)
        prior_fcf = prior_ocf + prior_capex if prior_capex < 0 else prior_ocf - abs(prior_capex)
        
        if prior_fcf == 0:
            return None
        
        growth = (current_fcf - prior_fcf) / abs(prior_fcf)
        return self._store_result('fcf_growth_yoy', growth)
    
    def calculate_revenue_cagr(self, years: int = 3) -> Optional[float]:
        """Calculate compound annual growth rate for revenue over specified years."""
        if not self._has_income() or years < 1:
            return None
        
        n_cols = len(self.income_stmt.columns)
        
        if self.freq == 'quarterly':
            required_quarters = years * 4 + 1
            if n_cols < required_quarters:
                return None
            
            current_ttm_values = []
            for i in range(4):
                val = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[i])
                if val is not None:
                    current_ttm_values.append(val)
            
            prior_ttm_values = []
            start_idx = years * 4
            for i in range(start_idx, min(start_idx + 4, n_cols)):
                val = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[i])
                if val is not None:
                    prior_ttm_values.append(val)
            
            if len(current_ttm_values) != 4 or len(prior_ttm_values) < 4:
                return None
            
            current_revenue = sum(current_ttm_values)
            prior_revenue = sum(prior_ttm_values[:4])
        
        else:
            if n_cols < years + 1:
                return None
            
            current_revenue = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[0])
            prior_revenue = self._safe_get(self.income_stmt, 'TotalRevenue', self.income_stmt.columns[years])
        
        if not current_revenue or not prior_revenue or prior_revenue <= 0:
            return None
        
        try:
            cagr = (current_revenue / prior_revenue) ** (1.0 / years) - 1
            key = f'revenue_cagr_{years}yr'
            return self._store_result(key, cagr)
        except (ValueError, ZeroDivisionError):
            return None
    
    def calculate_thesis_context_metrics(self) -> Dict[str, Any]:
        """Calculate metrics specifically useful for thesis validation context."""
        return self._collect_new_results([
            self.calculate_gross_margin_yoy_change,
            self.calculate_operating_margin_yoy_change,
            self.calculate_roe_yoy_change,
            self.calculate_debt_to_equity_yoy_change,
            self.calculate_fcf_growth_yoy,
            lambda: self.calculate_revenue_cagr(3)
        ])
    
    def calculate_all(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute all or selected financial calculations.
        
        Args:
            metrics: Optional list of metric names or groups to calculate.
                     Groups: 'profitability', 'ttm', 'growth', 'efficiency', 
                     'liquidity', 'leverage', 'cashflow', 'returns', 'quality'
                     If None, calculates all metrics.
        
        Returns:
            Dictionary of all calculated metrics
        """
        if not self._require_sheets('income_stmt', 'balance_sheet', 'cashflow'):
            return {}
        
        group_methods = {
            'profitability': self.calculate_profitability_margins,
            'ttm': self.calculate_ttm_metrics,
            'growth': self.calculate_growth_metrics,
            'efficiency': self.calculate_efficiency_ratios,
            'liquidity': self.calculate_liquidity_ratios,
            'leverage': self.calculate_leverage_ratios,
            'cashflow': self.calculate_cashflow_metrics,
            'returns': self.calculate_return_metrics,
            'quality': self.calculate_quality_metrics,
        }
        
        individual_methods = {
            'gross_margin': self.calculate_gross_margin,
            'operating_margin': self.calculate_operating_margin,
            'ebitda_margin': self.calculate_ebitda_margin,
            'net_profit_margin': self.calculate_net_profit_margin,
            'ttm_revenue': self.calculate_ttm_revenue,
            'ttm_net_income': self.calculate_ttm_net_income,
            'ttm_ebitda': self.calculate_ttm_ebitda,
            'ttm_net_margin': self.calculate_ttm_net_margin,
            'ttm_ebitda_margin': self.calculate_ttm_ebitda_margin,
            'ttm_operating_cashflow': self.calculate_ttm_operating_cashflow,
            'ttm_free_cashflow': self.calculate_ttm_free_cashflow,
            'revenue_growth': self.calculate_revenue_growth,
            'revenue_growth_yoy': self.calculate_revenue_growth_yoy,
            'earnings_growth': self.calculate_earnings_growth,
            'earnings_growth_yoy': self.calculate_earnings_growth_yoy,
            'asset_turnover': self.calculate_asset_turnover,
            'days_sales_outstanding': self.calculate_days_sales_outstanding,
            'inventory_turnover': self.calculate_inventory_turnover,
            'current_ratio': self.calculate_current_ratio,
            'quick_ratio': self.calculate_quick_ratio,
            'cash_ratio': self.calculate_cash_ratio,
            'debt_to_equity': self.calculate_debt_to_equity,
            'debt_to_assets': self.calculate_debt_to_assets,
            'equity_multiplier': self.calculate_equity_multiplier,
            'interest_coverage_ratio': self.calculate_interest_coverage_ratio,
            'ocf_to_net_income': self.calculate_ocf_to_net_income,
            'free_cash_flow': self.calculate_free_cash_flow,
            'fcf_margin': self.calculate_fcf_margin,
            'capex_to_revenue': self.calculate_capex_to_revenue,
            'return_on_assets': self.calculate_return_on_assets,
            'return_on_equity': self.calculate_return_on_equity,
            'roic': self.calculate_roic,
            'operating_leverage': self.calculate_operating_leverage,
            'accruals_ratio': self.calculate_accruals_ratio,
            'fcf_to_ni_ratio': self.calculate_fcf_to_ni_ratio,
            'working_capital_pct_revenue': self.calculate_working_capital_pct_revenue,
        }
        
        if metrics is None:
            for method in group_methods.values():
                method()
        else:
            for metric_name in metrics:
                if metric_name in group_methods:
                    group_methods[metric_name]()
                elif metric_name in individual_methods:
                    individual_methods[metric_name]()
        
        return self.calculations.copy()
