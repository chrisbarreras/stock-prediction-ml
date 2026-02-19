"""Shared pytest fixtures with synthetic test data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_price_data():
    """Create synthetic price data mimicking real price_data.pkl format.

    Returns Dict[str, DataFrame] with 3 tickers, 500 trading days each.
    """
    tickers = ['TEST_A', 'TEST_B', 'TEST_C']
    price_data = {}

    np.random.seed(42)
    dates = pd.bdate_range('2015-01-01', periods=500)

    for ticker in tickers:
        base_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.01, len(dates)),
            'High': prices * np.random.uniform(1.00, 1.03, len(dates)),
            'Low': prices * np.random.uniform(0.97, 1.00, len(dates)),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1_000_000, 50_000_000, len(dates)),
        }, index=dates)

        price_data[ticker] = df

    return price_data


@pytest.fixture
def sample_financial_data():
    """Create synthetic financial data mimicking real financial_data.pkl format.

    Returns List[Dict] with quarterly_income and quarterly_balance DataFrames
    where metrics are rows and quarter dates are columns.
    """
    tickers = ['TEST_A', 'TEST_B', 'TEST_C']
    financial_data = []

    np.random.seed(42)
    quarters = pd.to_datetime([
        '2016-03-31', '2016-06-30', '2016-09-30', '2016-12-31',
        '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31',
    ])

    for ticker in tickers:
        base_revenue = np.random.uniform(5e9, 50e9)

        income_data = {
            'Total Revenue': base_revenue * np.random.uniform(0.9, 1.1, len(quarters)),
            'Cost of Revenue': base_revenue * np.random.uniform(0.4, 0.7, len(quarters)),
            'Gross Profit': base_revenue * np.random.uniform(0.3, 0.6, len(quarters)),
            'Operating Income': base_revenue * np.random.uniform(0.1, 0.3, len(quarters)),
            'Net Income': base_revenue * np.random.uniform(0.05, 0.2, len(quarters)),
            'EPS Basic': np.random.uniform(1.0, 5.0, len(quarters)),
            'EPS Diluted': np.random.uniform(0.9, 4.8, len(quarters)),
            'R&D Expense': base_revenue * np.random.uniform(0.05, 0.15, len(quarters)),
            'SGA Expense': base_revenue * np.random.uniform(0.1, 0.2, len(quarters)),
            'Interest Expense': base_revenue * np.random.uniform(0.01, 0.05, len(quarters)),
            'Income Tax': base_revenue * np.random.uniform(0.02, 0.08, len(quarters)),
        }
        quarterly_income = pd.DataFrame(income_data, index=quarters).T
        quarterly_income.columns = quarters

        balance_data = {
            'Total Assets': base_revenue * np.random.uniform(3, 8, len(quarters)),
            'Current Assets': base_revenue * np.random.uniform(1, 3, len(quarters)),
            'Total Liabilities': base_revenue * np.random.uniform(1.5, 5, len(quarters)),
            'Current Liabilities': base_revenue * np.random.uniform(0.5, 2, len(quarters)),
            'Long Term Debt': base_revenue * np.random.uniform(0.5, 2, len(quarters)),
            'Short Term Debt': base_revenue * np.random.uniform(0.1, 0.5, len(quarters)),
            'Stockholders Equity': base_revenue * np.random.uniform(1, 3, len(quarters)),
            'Cash': base_revenue * np.random.uniform(0.2, 1, len(quarters)),
            'Shares Outstanding': np.random.uniform(1e9, 5e9, len(quarters)),
            'Operating Cash Flow': base_revenue * np.random.uniform(0.1, 0.3, len(quarters)),
            'Capital Expenditures': base_revenue * np.random.uniform(0.05, 0.15, len(quarters)),
        }
        quarterly_balance = pd.DataFrame(balance_data, index=quarters).T
        quarterly_balance.columns = quarters

        financial_data.append({
            'ticker': ticker,
            'quarterly_income': quarterly_income,
            'quarterly_balance': quarterly_balance,
            'info': {'symbol': ticker},
        })

    return financial_data


@pytest.fixture
def sample_sector_data():
    """Create synthetic sector data mimicking real sector_data.pkl format."""
    return {
        'TEST_A': {'sector': 'Information Technology', 'sub_industry': 'Software', 'company': 'Test A Inc'},
        'TEST_B': {'sector': 'Financials', 'sub_industry': 'Banks', 'company': 'Test B Corp'},
        'TEST_C': {'sector': 'Information Technology', 'sub_industry': 'Hardware', 'company': 'Test C Ltd'},
    }


@pytest.fixture
def sample_macro_data():
    """Create synthetic macro data mimicking real macro_data.pkl format."""
    dates = pd.date_range('2015-01-01', periods=1000, freq='D')
    return {
        'GS10': pd.Series(np.random.uniform(1.5, 3.5, len(dates)), index=dates),
        'VIXCLS': pd.Series(np.random.uniform(10, 35, len(dates)), index=dates),
        'UNRATE': pd.Series(np.random.uniform(3.5, 6.0, len(dates)), index=dates),
        'GDP': pd.Series(np.random.uniform(18000, 22000, len(dates)), index=dates),
        'CPIAUCSL': pd.Series(np.random.uniform(250, 300, len(dates)), index=dates),
    }


@pytest.fixture
def sample_processed_dataset():
    """Create a small processed dataset mimicking processed_dataset.csv."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'ticker': np.random.choice(['TEST_A', 'TEST_B', 'TEST_C'], n),
        'date': pd.date_range('2016-03-31', periods=n, freq='QE'),
        'quarter_price': np.random.uniform(50, 200, n),
        'revenue': np.random.uniform(1e9, 50e9, n),
        'revenue_growth': np.random.uniform(-0.2, 0.3, n),
        'profit_margin': np.random.uniform(-0.1, 0.3, n),
        'operating_margin': np.random.uniform(0.05, 0.4, n),
        'net_income': np.random.uniform(1e8, 5e9, n),
        'roa': np.random.uniform(0.01, 0.1, n),
        'roe': np.random.uniform(0.05, 0.3, n),
        'debt_to_assets': np.random.uniform(0.1, 0.6, n),
        'current_ratio': np.random.uniform(0.8, 3.0, n),
        'target_return': np.random.uniform(-0.3, 0.5, n),
    })
