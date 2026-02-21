"""Tests for feature engineering helper functions."""

import numpy as np
import pandas as pd

from src.features import (
    safe_loc, safe_divide, safe_growth, calculate_return,
    calculate_excess_return, select_features,
)


class TestSafeLoc:
    """Tests for safe_loc function."""

    def test_valid_lookup(self):
        df = pd.DataFrame(
            {'2024-03-31': [100, 200]},
            index=['Revenue', 'Net Income'],
        )
        assert safe_loc(df, 'Revenue', '2024-03-31') == 100.0

    def test_missing_metric(self):
        df = pd.DataFrame(
            {'2024-03-31': [100]},
            index=['Revenue'],
        )
        assert np.isnan(safe_loc(df, 'Nonexistent', '2024-03-31'))

    def test_missing_quarter(self):
        df = pd.DataFrame(
            {'2024-03-31': [100]},
            index=['Revenue'],
        )
        assert np.isnan(safe_loc(df, 'Revenue', '2099-01-01'))

    def test_nan_value(self):
        df = pd.DataFrame(
            {'2024-03-31': [np.nan]},
            index=['Revenue'],
        )
        assert np.isnan(safe_loc(df, 'Revenue', '2024-03-31'))

    def test_returns_float(self):
        df = pd.DataFrame(
            {'2024-03-31': [42]},
            index=['Revenue'],
        )
        result = safe_loc(df, 'Revenue', '2024-03-31')
        assert isinstance(result, float)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert np.isnan(safe_loc(df, 'Revenue', '2024-03-31'))


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self):
        assert safe_divide(10, 5) == 2.0

    def test_zero_denominator(self):
        assert np.isnan(safe_divide(10, 0))

    def test_nan_numerator(self):
        assert np.isnan(safe_divide(np.nan, 5))

    def test_nan_denominator(self):
        assert np.isnan(safe_divide(10, np.nan))

    def test_both_nan(self):
        assert np.isnan(safe_divide(np.nan, np.nan))

    def test_negative_values(self):
        assert safe_divide(-10, 5) == -2.0

    def test_fractional_result(self):
        assert abs(safe_divide(1, 3) - 0.3333333) < 0.001


class TestSafeGrowth:
    """Tests for safe_growth function."""

    def test_positive_growth(self):
        assert safe_growth(150, 100) == 0.5

    def test_negative_growth(self):
        assert safe_growth(80, 100) == -0.2

    def test_zero_previous(self):
        assert np.isnan(safe_growth(100, 0))

    def test_nan_current(self):
        assert np.isnan(safe_growth(np.nan, 100))

    def test_nan_previous(self):
        assert np.isnan(safe_growth(100, np.nan))

    def test_negative_to_positive(self):
        result = safe_growth(50, -100)
        assert abs(result - 1.5) < 0.001

    def test_no_change(self):
        assert safe_growth(100, 100) == 0.0


class TestCalculateReturn:
    """Tests for calculate_return function."""

    def test_positive_return(self):
        assert calculate_return(100, 110) == 0.1

    def test_negative_return(self):
        assert calculate_return(100, 90) == -0.1

    def test_zero_current_price(self):
        assert np.isnan(calculate_return(0, 100))

    def test_nan_inputs(self):
        assert np.isnan(calculate_return(np.nan, 100))
        assert np.isnan(calculate_return(100, np.nan))

    def test_no_change(self):
        assert calculate_return(100, 100) == 0.0


class TestCalculateExcessReturn:
    """Tests for calculate_excess_return function."""

    def test_positive_excess(self):
        assert abs(calculate_excess_return(0.15, 0.05) - 0.10) < 1e-10

    def test_negative_excess(self):
        assert calculate_excess_return(0.02, 0.08) == -0.06

    def test_zero_excess(self):
        assert calculate_excess_return(0.05, 0.05) == 0.0

    def test_nan_stock_return(self):
        assert np.isnan(calculate_excess_return(np.nan, 0.05))

    def test_nan_benchmark(self):
        assert np.isnan(calculate_excess_return(0.10, np.nan))

    def test_both_nan(self):
        assert np.isnan(calculate_excess_return(np.nan, np.nan))

    def test_negative_both(self):
        result = calculate_excess_return(-0.10, -0.05)
        assert abs(result - (-0.05)) < 1e-10


class TestSelectFeatures:
    """Tests for select_features function."""

    def test_drops_highly_correlated(self):
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.01, 100)  # nearly identical
        x3 = np.random.normal(0, 1, 100)  # independent
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        result, dropped = select_features(df, corr_threshold=0.95)
        assert len(dropped) == 1
        assert 'x3' in result.columns
        assert result.shape[1] == 2

    def test_keeps_uncorrelated(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.normal(0, 1, 100),
            'b': np.random.normal(0, 1, 100),
            'c': np.random.normal(0, 1, 100),
        })
        result, dropped = select_features(df, corr_threshold=0.95)
        assert len(dropped) == 0
        assert result.shape[1] == 3

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        result, dropped = select_features(df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(dropped, list)

    def test_custom_threshold(self):
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.3, 100)  # corr ~0.95
        df = pd.DataFrame({'x1': x1, 'x2': x2})
        # With strict threshold, should drop
        _, dropped_strict = select_features(df, corr_threshold=0.8)
        assert len(dropped_strict) >= 1


class TestFeatureCreation:
    """Tests for the feature creation pipeline using synthetic data."""

    def test_price_data_format(self, sample_price_data):
        for ticker, df in sample_price_data.items():
            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.index, pd.DatetimeIndex)
            assert 'Adj Close' in df.columns
            assert 'Volume' in df.columns
            assert len(df) > 0

    def test_financial_data_format(self, sample_financial_data):
        for company in sample_financial_data:
            assert 'ticker' in company
            assert 'quarterly_income' in company
            assert 'quarterly_balance' in company
            income = company['quarterly_income']
            assert 'Total Revenue' in income.index
            assert 'Net Income' in income.index

    def test_financial_data_metrics_as_rows(self, sample_financial_data):
        """Verify metrics are rows and quarters are columns."""
        company = sample_financial_data[0]
        income = company['quarterly_income']
        assert income.shape[0] == 11  # 11 income metrics
        assert income.shape[1] == 8   # 8 quarters
        assert isinstance(income.columns[0], pd.Timestamp)

    def test_sector_data_format(self, sample_sector_data):
        for ticker, info in sample_sector_data.items():
            assert 'sector' in info
            assert 'sub_industry' in info
            assert isinstance(info['sector'], str)

    def test_macro_data_format(self, sample_macro_data):
        assert 'GS10' in sample_macro_data
        assert 'VIXCLS' in sample_macro_data
        for series_id, data in sample_macro_data.items():
            assert isinstance(data, pd.Series)
            assert isinstance(data.index, pd.DatetimeIndex)
            assert len(data) > 0
