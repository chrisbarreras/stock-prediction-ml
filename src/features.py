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


def calculate_excess_return(stock_return, benchmark_return):
    """Calculate excess return (stock return minus benchmark).

    Args:
        stock_return: Individual stock return for the period.
        benchmark_return: Benchmark (e.g. S&P 500) return for same period.

    Returns:
        Excess return as float, or np.nan if inputs invalid.
    """
    if pd.isna(stock_return) or pd.isna(benchmark_return):
        return np.nan
    return stock_return - benchmark_return


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index.

    Args:
        prices: pd.Series of closing prices (needs at least period+1 values).
        period: Lookback period (default 14).

    Returns:
        RSI value (0-100) or np.nan if insufficient data.
    """
    if not isinstance(prices, pd.Series) or len(prices) < period + 1:
        return np.nan
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.iloc[-period:].mean()
    avg_loss = loss.iloc[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd_histogram(prices, fast=12, slow=26, signal=9):
    """Calculate MACD histogram value.

    Args:
        prices: pd.Series of closing prices (needs at least slow+signal values).
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        MACD histogram value as float, or np.nan if insufficient data.
    """
    if not isinstance(prices, pd.Series) or len(prices) < slow + signal:
        return np.nan
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(histogram.iloc[-1])


def calculate_bollinger_width(prices, period=20, num_std=2):
    """Calculate Bollinger Band width.

    Args:
        prices: pd.Series of closing prices (needs at least period values).
        period: Moving average period (default 20).
        num_std: Number of standard deviations (default 2).

    Returns:
        Band width as float (upper - lower) / middle, or np.nan.
    """
    if not isinstance(prices, pd.Series) or len(prices) < period:
        return np.nan
    window = prices.iloc[-period:]
    middle = window.mean()
    if middle == 0:
        return np.nan
    std = window.std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return float((upper - lower) / middle)


def calculate_volume_trend(volumes, period=20):
    """Calculate volume trend as current volume / period-day average.

    Args:
        volumes: pd.Series of volume data (needs at least period values).
        period: Lookback period (default 20).

    Returns:
        Volume ratio as float, or np.nan if insufficient data.
    """
    if not isinstance(volumes, pd.Series) or len(volumes) < period:
        return np.nan
    avg_volume = volumes.iloc[-period:].mean()
    if avg_volume == 0:
        return np.nan
    return float(volumes.iloc[-1] / avg_volume)


def calculate_price_to_52wk_high(prices):
    """Calculate current price relative to 52-week high.

    Args:
        prices: pd.Series of closing prices (needs at least 252 trading days).

    Returns:
        Ratio (0 to 1) as float, or np.nan if insufficient data.
    """
    if not isinstance(prices, pd.Series) or len(prices) < 252:
        return np.nan
    high_52wk = prices.iloc[-252:].max()
    if high_52wk == 0:
        return np.nan
    return float(prices.iloc[-1] / high_52wk)


def select_features(X, corr_threshold=0.95):
    """Remove highly correlated features from a DataFrame.

    Iterates through the correlation matrix and drops one feature
    from each pair with correlation above the threshold.

    Args:
        X: DataFrame of features.
        corr_threshold: Maximum allowed absolute correlation (default 0.95).

    Returns:
        Tuple of (filtered DataFrame, list of dropped column names).
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = []
    for col in upper.columns:
        if any(upper[col] > corr_threshold):
            to_drop.append(col)
    return X.drop(columns=to_drop), to_drop


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
