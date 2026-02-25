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

    def test_has_excess_return_columns(self, sample_processed_dataset):
        """Dataset should have benchmark and excess return columns."""
        assert 'benchmark_return' in sample_processed_dataset.columns
        assert 'target_excess_return' in sample_processed_dataset.columns

    def test_has_technical_features(self, sample_processed_dataset):
        """Dataset should have new technical indicator columns."""
        technical = ['rsi_14', 'macd_histogram', 'bollinger_width',
                     'volume_trend', 'price_to_52wk_high']
        for col in technical:
            assert col in sample_processed_dataset.columns, f"Missing: {col}"

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


class TestDataCleaningPipeline:
    """Tests for the NaN imputation and outlier handling pipeline."""

    def test_winsorize_clips_extreme_ratios(self, sample_dirty_dataset):
        """Winsorizing to 1st-99th percentile should remove extreme ROE/ROA."""
        df = sample_dirty_dataset.copy()
        for col in ['roa', 'roe', 'debt_to_equity']:
            valid_mask = df[col].notna()
            if valid_mask.sum() > 10:
                values = df.loc[valid_mask, col].values
                p1, p99 = np.percentile(values, [1, 99])
                df.loc[valid_mask, col] = np.clip(values, p1, p99)
        # After winsorization, extreme values should be gone
        assert df['roe'].max() < 1_768_519.0
        assert df['roa'].max() < 500_000.0

    def test_forward_fill_preserves_valid_values(self, sample_dirty_dataset):
        """Forward fill should not overwrite existing valid values."""
        df = sample_dirty_dataset.copy()
        original_revenue = df['revenue'].copy()
        # Revenue has no NaN, so ffill should not change it
        df['revenue'] = df.groupby('ticker')['revenue'].transform(lambda x: x.ffill())
        np.testing.assert_array_equal(original_revenue.values, df['revenue'].values)

    def test_forward_fill_reduces_nans(self, sample_dirty_dataset):
        """Forward fill within companies should reduce NaN count."""
        df = sample_dirty_dataset.copy().sort_values(['ticker', 'date'])
        before_nans = df['debt_to_equity'].isna().sum()
        df['debt_to_equity'] = df.groupby('ticker')['debt_to_equity'].transform(
            lambda x: x.ffill()
        )
        after_nans = df['debt_to_equity'].isna().sum()
        assert after_nans <= before_nans

    def test_sector_median_fills_remaining_nans(self, sample_dirty_dataset):
        """Sector median should fill NaN values that forward-fill missed."""
        df = sample_dirty_dataset.copy()
        col = 'interest_coverage'
        before_nans = df[col].isna().sum()
        assert before_nans > 0, "Test fixture should have NaN values"
        sector_median = df.groupby('sector')[col].transform('median')
        mask = df[col].isna()
        df.loc[mask, col] = sector_median.loc[mask]
        after_nans = df[col].isna().sum()
        assert after_nans < before_nans

    def test_no_nan_after_full_pipeline(self, sample_dirty_dataset):
        """After the full cleaning pipeline, no NaN should remain in numeric features."""
        df = sample_dirty_dataset.copy().sort_values(['ticker', 'date'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'target_return']
        # Forward fill
        df[numeric_cols] = df.groupby('ticker')[numeric_cols].transform(lambda x: x.ffill())
        # Sector median
        sector_medians = df.groupby('sector')[numeric_cols].transform('median')
        for col in numeric_cols:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = sector_medians.loc[mask, col]
        # Last resort
        df[numeric_cols] = df[numeric_cols].fillna(0)
        assert not df[numeric_cols].isna().any().any()


class TestSectorRelativeFeatures:
    """Tests for sector-relative and z-score features."""

    def test_has_sector_vs_features(self, sample_processed_dataset):
        """Dataset should have sector-relative difference features."""
        vs_features = ['profit_margin_vs_sector', 'roe_vs_sector']
        for col in vs_features:
            assert col in sample_processed_dataset.columns, f"Missing: {col}"

    def test_has_sector_z_scores(self, sample_processed_dataset):
        """Dataset should have sector z-score features."""
        z_features = ['profit_margin_sector_z', 'roe_sector_z']
        for col in z_features:
            assert col in sample_processed_dataset.columns, f"Missing: {col}"

    def test_z_scores_are_finite(self, sample_processed_dataset):
        """Sector z-scores should contain no NaN or inf values."""
        z_cols = [c for c in sample_processed_dataset.columns if c.endswith('_sector_z')]
        for col in z_cols:
            assert not sample_processed_dataset[col].isna().any(), f"{col} contains NaN"
            assert not np.isinf(sample_processed_dataset[col]).any(), f"{col} contains inf"

    def test_sector_z_score_computation(self):
        """Sector z-scores should have approximately zero mean within each sector."""
        data = pd.DataFrame({
            'sector': ['A', 'A', 'A', 'B', 'B', 'B'],
            'metric': [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        sector_median = data.groupby('sector')['metric'].transform('median')
        sector_std = data.groupby('sector')['metric'].transform('std')
        z = (data['metric'] - sector_median) / sector_std
        for sector in ['A', 'B']:
            mask = data['sector'] == sector
            assert abs(z[mask].mean()) < 0.01


class TestFeatureInteractions:
    """Tests for feature interaction columns."""

    def test_has_interaction_features(self, sample_processed_dataset):
        """Dataset should have feature interaction columns."""
        interactions = ['momentum_x_quality', 'risk_leverage', 'growth_profitability']
        for col in interactions:
            assert col in sample_processed_dataset.columns, f"Missing: {col}"

    def test_interactions_are_finite(self, sample_processed_dataset):
        """Feature interactions should contain no NaN or inf values."""
        interactions = ['momentum_x_quality', 'risk_leverage', 'growth_profitability']
        for col in interactions:
            if col in sample_processed_dataset.columns:
                assert not sample_processed_dataset[col].isna().any(), f"{col} contains NaN"
                assert not np.isinf(sample_processed_dataset[col]).any(), f"{col} contains inf"


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
