"""Tests for data validation — verify data quality and format requirements."""

import numpy as np
import pandas as pd

from src.features import validate_dataset


class TestDatasetValidation:
    """Tests for processed dataset quality."""

    def test_has_required_columns(self, sample_processed_dataset):
        required = ['ticker', 'date', 'target_return']
        for col in required:
            assert col in sample_processed_dataset.columns, f"Missing column: {col}"

    def test_no_nan_in_target(self, sample_processed_dataset):
        assert not sample_processed_dataset['target_return'].isna().any(), \
            "target_return contains NaN values"

    def test_no_infinite_values(self, sample_processed_dataset):
        numeric_cols = sample_processed_dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(sample_processed_dataset[col]).any(), \
                f"Column {col} contains infinite values"

    def test_target_return_range(self, sample_processed_dataset):
        """Target returns should be between -100% and +100%."""
        assert (sample_processed_dataset['target_return'] >= -1.0).all(), \
            "target_return below -100%"
        assert (sample_processed_dataset['target_return'] <= 1.0).all(), \
            "target_return above +100%"

    def test_positive_revenue(self, sample_processed_dataset):
        if 'revenue' in sample_processed_dataset.columns:
            assert (sample_processed_dataset['revenue'] > 0).all(), \
                "Revenue should be positive"

    def test_minimum_sample_count(self, sample_processed_dataset):
        assert len(sample_processed_dataset) >= 10, \
            f"Too few samples: {len(sample_processed_dataset)}"

    def test_multiple_companies(self, sample_processed_dataset):
        assert sample_processed_dataset['ticker'].nunique() >= 2, \
            "Need at least 2 companies"

    def test_profit_margin_range(self, sample_processed_dataset):
        """Profit margin should be reasonable (between -10 and 1)."""
        if 'profit_margin' in sample_processed_dataset.columns:
            margins = sample_processed_dataset['profit_margin']
            assert (margins >= -10).all(), "Profit margin too low"
            assert (margins <= 1).all(), "Profit margin above 100%"

    def test_validate_dataset_function(self, sample_processed_dataset):
        results = validate_dataset(sample_processed_dataset)
        assert results['is_valid']
        assert results['has_target']
        assert results['has_ticker']
        assert results['no_nan_target']
        assert results['no_inf']
        assert results['sample_count'] > 0
        assert results['feature_count'] > 0

    def test_validate_dataset_missing_target(self):
        bad_dataset = pd.DataFrame({'ticker': ['A'], 'value': [1]})
        results = validate_dataset(bad_dataset)
        assert not results['is_valid']
        assert not results['has_target']


class TestPriceDataValidation:
    """Tests for price data format."""

    def test_is_dict(self, sample_price_data):
        assert isinstance(sample_price_data, dict)

    def test_values_are_dataframes(self, sample_price_data):
        for ticker, df in sample_price_data.items():
            assert isinstance(df, pd.DataFrame), f"{ticker} is not a DataFrame"

    def test_datetime_index(self, sample_price_data):
        for ticker, df in sample_price_data.items():
            assert isinstance(df.index, pd.DatetimeIndex), \
                f"{ticker} does not have DatetimeIndex"

    def test_required_columns(self, sample_price_data):
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for ticker, df in sample_price_data.items():
            for col in required:
                assert col in df.columns, f"{ticker} missing column: {col}"

    def test_prices_positive(self, sample_price_data):
        for ticker, df in sample_price_data.items():
            assert (df['Close'] > 0).all(), f"{ticker} has non-positive close prices"

    def test_volume_non_negative(self, sample_price_data):
        for ticker, df in sample_price_data.items():
            assert (df['Volume'] >= 0).all(), f"{ticker} has negative volume"

    def test_high_gte_low(self, sample_price_data):
        for ticker, df in sample_price_data.items():
            assert (df['High'] >= df['Low']).all(), \
                f"{ticker} has High < Low"


class TestSectorDataValidation:
    """Tests for sector data format."""

    def test_is_dict(self, sample_sector_data):
        assert isinstance(sample_sector_data, dict)

    def test_entries_have_sector(self, sample_sector_data):
        for ticker, info in sample_sector_data.items():
            assert 'sector' in info, f"{ticker} missing sector"
            assert len(info['sector']) > 0, f"{ticker} has empty sector"

    def test_entries_have_sub_industry(self, sample_sector_data):
        for ticker, info in sample_sector_data.items():
            assert 'sub_industry' in info, f"{ticker} missing sub_industry"


class TestMacroDataValidation:
    """Tests for macro data format."""

    def test_is_dict(self, sample_macro_data):
        assert isinstance(sample_macro_data, dict)

    def test_expected_indicators(self, sample_macro_data):
        expected = ['GS10', 'VIXCLS', 'UNRATE', 'GDP', 'CPIAUCSL']
        for indicator in expected:
            assert indicator in sample_macro_data, f"Missing indicator: {indicator}"

    def test_values_are_series(self, sample_macro_data):
        for name, series in sample_macro_data.items():
            assert isinstance(series, pd.Series), f"{name} is not a Series"

    def test_no_nan_values(self, sample_macro_data):
        for name, series in sample_macro_data.items():
            assert not series.isna().any(), f"{name} contains NaN values"
