"""Download macroeconomic indicators from FRED API.

Downloads 5 indicators (10yr Treasury, VIX, unemployment, GDP, CPI),
forward-fills to daily frequency, and saves as macro_data.pkl.

Requires FRED_API_KEY in .env file (free at https://fred.stlouisfed.org/).

Usage:
    python scripts/download_macro.py
"""

import pandas as pd
import pickle
import os
from fredapi import Fred
from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
START_DATE = '2005-01-01'

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
FRED_API_KEY = os.getenv('FRED_API_KEY')

if not FRED_API_KEY:
    raise ValueError('FRED_API_KEY not found in .env file. Add: FRED_API_KEY=your_key\n'
                     'Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html')

fred = Fred(api_key=FRED_API_KEY)
print('FRED API connected successfully.')
print(f'Download range: {START_DATE} to present\n')

# Define indicators
INDICATORS = {
    'GS10': '10-Year Treasury Constant Maturity Rate (daily)',
    'VIXCLS': 'CBOE Volatility Index (daily)',
    'UNRATE': 'Unemployment Rate (monthly)',
    'GDP': 'Gross Domestic Product (quarterly)',
    'CPIAUCSL': 'Consumer Price Index (monthly)',
}

print('Indicators to download:')
for series_id, desc in INDICATORS.items():
    print(f'  {series_id}: {desc}')

# Download
print(f'\n=== Downloading ===')
macro_data = {}

for series_id, desc in INDICATORS.items():
    try:
        data = fred.get_series(series_id, observation_start=START_DATE)
        data = data.dropna()
        macro_data[series_id] = data
        print(f'  {series_id}: {len(data)} observations '
              f'({data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")})')
    except Exception as e:
        print(f'  {series_id}: FAILED - {e}')

print(f'\nDownloaded {len(macro_data)}/{len(INDICATORS)} indicators')

if len(macro_data) < 3:
    raise ValueError('Insufficient macro data - check API key and network connection')

# Forward-fill to daily frequency
print(f'\n=== Resampling to daily frequency ===')
macro_daily = {}

for series_id, data in macro_data.items():
    daily = data.resample('D').ffill()
    macro_daily[series_id] = daily
    print(f'  {series_id}: {len(data)} raw -> {len(daily)} daily observations')

# Save
output_path = os.path.join(DATA_DIR, 'macro_data.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(macro_daily, f)

print(f'\nSaved macro_data.pkl')
print(f'File size: {os.path.getsize(output_path) / 1024:.1f} KB')
print(f'Indicators: {list(macro_daily.keys())}')

# Sample values at COVID period
sample_date = pd.Timestamp('2020-03-15')
print(f'\nSample values at {sample_date.strftime("%Y-%m-%d")} (COVID):')
for series_id, data in macro_daily.items():
    val = data.asof(sample_date)
    print(f'  {series_id}: {val:.2f}')

# Verify
for series_id, data in macro_daily.items():
    assert isinstance(data, pd.Series), f'{series_id}: Expected Series'
    assert isinstance(data.index, pd.DatetimeIndex), f'{series_id}: Expected DatetimeIndex'

print('\nVerification passed! Done.')
