"""Feature engineering helper functions for stock prediction model."""

import numpy as np
import pandas as pd


def safe_loc(df, metric, quarter):
    """Safely get value from DataFrame using .loc[metric, quarter].

    Args:
        df: DataFrame with metrics as rows, dates as columns.
        metric: Row label to look up.
        quarter: Column label (date) to look up.

    Returns:
        Float value or np.nan if not found.
    """
    try:
        if metric in df.index and quarter in df.columns:
            val = df.loc[metric, quarter]
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return np.nan


def safe_divide(a, b):
    """Safely divide two numbers, returning NaN for zero/NaN denominators.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        Float result or np.nan.
    """
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def safe_growth(current, previous):
    """Calculate growth rate between two values.

    Args:
        current: Current period value.
        previous: Previous period value.

    Returns:
        Growth rate as float, or np.nan if calculation not possible.
    """
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return (current - previous) / abs(previous)


def calculate_return(current_price, future_price):
    """Calculate simple return between two prices.

    Args:
        current_price: Price at start of period.
        future_price: Price at end of period.

    Returns:
        Return as float, or np.nan if inputs invalid.
    """
    if pd.isna(current_price) or pd.isna(future_price) or current_price == 0:
        return np.nan
    return (future_price - current_price) / current_price


def validate_dataset(dataset):
    """Validate a processed dataset meets quality requirements.

    Args:
        dataset: pandas DataFrame with features and target_return.

    Returns:
        Dict with validation results.
    """
    results = {
        'has_target': 'target_return' in dataset.columns,
        'has_ticker': 'ticker' in dataset.columns,
        'has_date': 'date' in dataset.columns,
        'no_nan_target': not dataset['target_return'].isna().any() if 'target_return' in dataset.columns else False,
        'sample_count': len(dataset),
        'feature_count': len([c for c in dataset.columns if c not in ['ticker', 'date', 'sector', 'target_return']]),
        'no_inf': not np.isinf(dataset.select_dtypes(include=[np.number])).any().any(),
    }
    results['is_valid'] = all([
        results['has_target'],
        results['has_ticker'],
        results['no_nan_target'],
        results['sample_count'] > 0,
        results['no_inf'],
    ])
    return results
