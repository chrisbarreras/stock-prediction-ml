"""Tests for model training and validation pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class TestModelPipeline:
    """Tests that the model training pipeline works end-to-end."""

    @pytest.fixture
    def trained_model(self, sample_processed_dataset):
        """Train a small XGBoost model on synthetic data."""
        dataset = sample_processed_dataset
        feature_cols = [c for c in dataset.columns
                        if c not in ['ticker', 'date', 'sector', 'target_return']]

        X = dataset[feature_cols].copy()
        y = dataset['target_return'].copy()

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror',
        )
        model.fit(X_train, y_train, verbose=False)

        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }

    def test_model_produces_predictions(self, trained_model):
        predictions = trained_model['model'].predict(trained_model['X_test'])
        assert len(predictions) == len(trained_model['X_test'])
        assert not np.isnan(predictions).any()

    def test_predictions_are_finite(self, trained_model):
        predictions = trained_model['model'].predict(trained_model['X_test'])
        assert np.all(np.isfinite(predictions))

    def test_feature_importances_exist(self, trained_model):
        importances = trained_model['model'].feature_importances_
        assert len(importances) == len(trained_model['feature_columns'])
        assert np.all(importances >= 0)

    def test_feature_importances_sum_to_one(self, trained_model):
        importances = trained_model['model'].feature_importances_
        assert abs(importances.sum() - 1.0) < 0.01

    def test_scaler_transform_shape(self, trained_model):
        scaler = trained_model['scaler']
        X_test = trained_model['X_test']
        transformed = scaler.transform(X_test)
        assert transformed.shape == X_test.shape

    def test_results_dict_structure(self, trained_model):
        """Verify the results dict has the keys notebook 04 expects."""
        assert 'model' in trained_model
        assert 'scaler' in trained_model
        assert 'feature_columns' in trained_model
        assert isinstance(trained_model['feature_columns'], list)
        assert len(trained_model['feature_columns']) > 0


class TestModelMetrics:
    """Tests for model metric calculations."""

    def test_rmse_is_positive(self, sample_processed_dataset):
        y_true = sample_processed_dataset['target_return'].values
        y_pred = y_true + np.random.normal(0, 0.05, len(y_true))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse > 0

    def test_direction_accuracy_calculation(self):
        y_true = np.array([0.1, -0.05, 0.2, -0.1, 0.05])
        y_pred = np.array([0.05, -0.02, 0.1, 0.01, 0.03])
        correct = np.sum(np.sign(y_true) == np.sign(y_pred))
        accuracy = correct / len(y_true)
        assert 0 <= accuracy <= 1
        assert accuracy == 0.8  # 4 out of 5 correct

    def test_direction_accuracy_bounds(self):
        """Direction accuracy must be between 0 and 1."""
        y_true = np.random.uniform(-0.5, 0.5, 100)
        y_pred = np.random.uniform(-0.5, 0.5, 100)
        correct = np.sum(np.sign(y_true) == np.sign(y_pred))
        accuracy = correct / len(y_true)
        assert 0 <= accuracy <= 1

    def test_spearman_rank_correlation(self):
        """Spearman rank correlation should be between -1 and 1."""
        from scipy.stats import spearmanr
        y_true = np.array([0.1, -0.05, 0.2, -0.1, 0.05, 0.15, -0.02])
        y_pred = np.array([0.05, -0.02, 0.15, 0.01, 0.03, 0.1, -0.01])
        rho, pval = spearmanr(y_true, y_pred)
        assert -1 <= rho <= 1
        assert rho > 0  # predictions should have positive rank correlation
