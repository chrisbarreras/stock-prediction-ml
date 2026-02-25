"""Download S&P 500 (SPY) benchmark price data via yfinance."""

import pickle
import os
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("Downloading SPY benchmark data...")
spy = yf.Ticker('SPY')
spy_prices = spy.history(start='2005-01-01')

if spy_prices is not None and len(spy_prices) > 0:
    # Strip timezone for compatibility
    spy_prices.index = spy_prices.index.tz_localize(None)
    print(f"Downloaded {len(spy_prices)} days of SPY data")
    print(f"Date range: {spy_prices.index.min().date()} to {spy_prices.index.max().date()}")

    output_path = os.path.join(DATA_DIR, 'spy_prices.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(spy_prices, f)
    print(f"Saved to {output_path}")
else:
    print("ERROR: SPY download returned no data")
