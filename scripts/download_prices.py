"""Download S&P 500 price data using yfinance."""
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import time
import shutil
import requests
from io import StringIO

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
START_DATE = '2005-01-01'

# Step 1: Get S&P 500 ticker list from Wikipedia
print('Fetching S&P 500 ticker list from Wikipedia...')
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0 (stock-prediction-ml project)'}
response = requests.get(url, headers=headers)
response.raise_for_status()
tables = pd.read_html(StringIO(response.text))
sp500_df = tables[0]
sp500_tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
print(f'Found {len(sp500_tickers)} tickers.\n')

# Step 2: Download price data using Ticker.history()
price_data = {}
failed_tickers = []

total = len(sp500_tickers)
print(f'Downloading {total} tickers...\n')

for i, ticker in enumerate(sp500_tickers):
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=START_DATE, auto_adjust=False)
        if len(df) > 100:
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
            price_data[ticker] = df[cols].copy()
        else:
            failed_tickers.append(ticker)
    except Exception as e:
        failed_tickers.append(ticker)

    if (i + 1) % 25 == 0:
        print(f'  Progress: {i + 1}/{total} ({len(price_data)} successful, {len(failed_tickers)} failed)')

    time.sleep(0.3)

print(f'\nFirst pass: {len(price_data)} successful, {len(failed_tickers)} failed')

# Step 3: Retry failed tickers
if failed_tickers:
    print(f'\nRetrying {len(failed_tickers)} failed tickers with longer delay...')
    still_failed = []
    for ticker in failed_tickers:
        try:
            time.sleep(2)
            t = yf.Ticker(ticker)
            df = t.history(start=START_DATE, auto_adjust=False)
            if len(df) > 100:
                df.index = df.index.tz_localize(None) if df.index.tz else df.index
                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
                price_data[ticker] = df[cols].copy()
                print(f'  Recovered: {ticker}')
            else:
                still_failed.append(ticker)
        except Exception:
            still_failed.append(ticker)

    print(f'Recovered: {len(failed_tickers) - len(still_failed)}')
    if still_failed:
        print(f'Still failed ({len(still_failed)}): {still_failed}')

print(f'\n=== Final: {len(price_data)} tickers ===')

# Step 4: Backup and save
old_path = os.path.join(DATA_DIR, 'price_data.pkl')
backup_path = os.path.join(DATA_DIR, 'price_data_kaggle_backup.pkl')

if os.path.exists(old_path) and not os.path.exists(backup_path):
    shutil.copy2(old_path, backup_path)
    print(f'\nBacked up old price data to {backup_path}')

with open(old_path, 'wb') as f:
    pickle.dump(price_data, f)

print(f'\nSaved price_data.pkl: {len(price_data)} tickers')
print(f'File size: {os.path.getsize(old_path) / 1024 / 1024:.1f} MB')

# Verify
sample = list(price_data.keys())[0]
print(f'\nSample ({sample}):')
print(f'  Date range: {price_data[sample].index.min()} to {price_data[sample].index.max()}')
print(f'  Columns: {list(price_data[sample].columns)}')
print(f'  Rows: {len(price_data[sample])}')
print('\nDone!')
