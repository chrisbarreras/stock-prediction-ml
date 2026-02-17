"""Download S&P 500 sector classifications from Wikipedia."""
import pandas as pd
import pickle
import os
import requests
from io import StringIO

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Scrape Wikipedia
print('Fetching S&P 500 sector data from Wikipedia...')
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0 (stock-prediction-ml project)'}
response = requests.get(url, headers=headers)
response.raise_for_status()
tables = pd.read_html(StringIO(response.text))
sp500_df = tables[0]

print(f'Total companies: {len(sp500_df)}')
print(f'Unique sectors: {sp500_df["GICS Sector"].nunique()}')

# Build sector dictionary
sector_data = {}
for _, row in sp500_df.iterrows():
    ticker_wiki = str(row['Symbol']).strip()
    ticker_yf = ticker_wiki.replace('.', '-')

    entry = {
        'sector': row['GICS Sector'],
        'sub_industry': row['GICS Sub-Industry'],
        'company': row.get('Security', '')
    }

    sector_data[ticker_wiki] = entry
    if ticker_yf != ticker_wiki:
        sector_data[ticker_yf] = entry

# Cross-check with price data
price_path = os.path.join(DATA_DIR, 'price_data.pkl')
if os.path.exists(price_path):
    with open(price_path, 'rb') as f:
        price_data = pickle.load(f)
    matched = set(price_data.keys()) & set(sector_data.keys())
    missing = set(price_data.keys()) - set(sector_data.keys())
    print(f'\nPrice data tickers: {len(price_data)}')
    print(f'Matched with sector data: {len(matched)}')
    if missing:
        print(f'Missing sector data for: {sorted(missing)[:10]}')

# Save
output_path = os.path.join(DATA_DIR, 'sector_data.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(sector_data, f)

print(f'\nSaved sector_data.pkl: {len(sector_data)} entries')
print(f'File size: {os.path.getsize(output_path) / 1024:.1f} KB')

# Sector distribution
print('\nSector Distribution:')
for sector, count in sp500_df['GICS Sector'].value_counts().items():
    print(f'  {sector}: {count}')

print('\nDone!')
