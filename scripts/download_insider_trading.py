"""Download Form 4 insider trading data from SEC EDGAR.

For each S&P 500 company, fetches Form 4 filings and aggregates net
insider purchases vs. disposals by quarter.  Saves insider_trading.pkl.

Data sources (all free, no API key):
  https://www.sec.gov/files/company_tickers.json  — ticker → CIK map
  https://data.sec.gov/submissions/CIK{cik:010d}.json  — filing list
  https://www.sec.gov/Archives/edgar/data/...  — Form 4 XML

Runtime: ~30-90 min first run (rate-limited to ≤10 req/sec).
Results are cached per company; re-running is instant if cache exists.

Usage:
    python scripts/download_insider_trading.py
"""

import io
import json
import os
import pickle
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_DIR  = os.path.join(DATA_DIR, 'insider_trading_cache')
START_DATE = '2005-01-01'
REQ_DELAY  = 0.12   # seconds between requests — keeps us ≤9 req/sec (SEC limit: 10)

HEADERS = {
    'User-Agent': 'StockPredictionML research@example.com',
    'Accept-Encoding': 'gzip, deflate',
}

# Transaction codes that represent genuine open-market activity
BUY_CODES  = {'P'}   # open-market purchase
SELL_CODES = {'S'}   # open-market sale
# A=award/grant, D=disposition (non-sale), G=gift, F=tax-withholding, M=exercise → ignored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> requests.Response:
    """GET with retry and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(REQ_DELAY)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                print('  Rate-limited; sleeping 60s ...')
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f'  Retry {attempt + 1} after error: {e}')
            time.sleep(5)
    raise RuntimeError(f'Failed after {retries} retries: {url}')


def fetch_cik_map(tickers: list[str]) -> dict[str, int]:
    """Return {ticker: cik_int} for tickers that exist in the SEC mapping."""
    print('Fetching SEC ticker→CIK map ...')
    resp = _get('https://www.sec.gov/files/company_tickers.json')
    raw = resp.json()
    # SEC format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {}
    for entry in raw.values():
        t = entry.get('ticker', '').upper()
        if t in tickers:
            mapping[t] = int(entry['cik_str'])
    print(f'  Matched {len(mapping)}/{len(tickers)} tickers to CIKs')
    return mapping


def get_form4_filings(cik: int, start_date: str) -> list[dict]:
    """
    Return list of {date, accession_no, primary_doc} for Form 4 filings
    after start_date, fetching paginated submissions if needed.
    """
    filings = []
    url = f'https://data.sec.gov/submissions/CIK{cik:010d}.json'
    data = _get(url).json()

    def _extract(recent: dict):
        forms   = recent.get('form', [])
        dates   = recent.get('filingDate', [])
        accnos  = recent.get('accessionNumber', [])
        docs    = recent.get('primaryDocument', [])
        for form, date, accno, doc in zip(forms, dates, accnos, docs):
            if form == '4' and date >= start_date:
                filings.append({'date': date, 'accession_no': accno, 'primary_doc': doc})

    _extract(data.get('filings', {}).get('recent', {}))

    # Paginated older filings
    for page_file in data.get('filings', {}).get('files', []):
        page_url = 'https://data.sec.gov/submissions/' + page_file['name']
        page_data = _get(page_url).json()
        _extract(page_data)

    return filings


def parse_form4_xml(cik: int, accession_no: str, primary_doc: str) -> list[dict]:
    """
    Download a Form 4 XML and return a list of transactions:
      [{'date': 'YYYY-MM-DD', 'code': 'P'/'S', 'shares': float, 'price': float}, ...]
    Returns [] on any parse error.
    """
    # Accession numbers stored as "0001234567-23-012345" but URL uses no dashes
    accno_clean = accession_no.replace('-', '')
    url = (f'https://www.sec.gov/Archives/edgar/data/{cik}/'
           f'{accno_clean}/{primary_doc}')
    try:
        resp = _get(url)
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    transactions = []
    # nonDerivativeTransaction elements
    for txn in root.iter('nonDerivativeTransaction'):
        try:
            code_el    = txn.find('.//transactionCode')
            shares_el  = txn.find('.//transactionShares/value')
            price_el   = txn.find('.//transactionPricePerShare/value')
            date_el    = txn.find('.//transactionDate/value')
            if code_el is None or shares_el is None or date_el is None:
                continue
            code   = (code_el.text or '').strip().upper()
            shares = float(shares_el.text or 0)
            price  = float(price_el.text or 0) if price_el is not None else 0.0
            date   = (date_el.text or '').strip()
            if code and shares > 0 and date:
                transactions.append({'date': date, 'code': code,
                                     'shares': shares, 'price': price})
        except (ValueError, AttributeError):
            continue
    return transactions


def aggregate_to_quarters(transactions: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-transaction records into quarterly rows.

    For each calendar quarter-end, count open-market buys (P) and sells (S),
    compute:
      net_direction  = (n_buys - n_sells) / (n_buys + n_sells + 1)  ∈ (-1, +1)
      insider_buy_value = sum(buy_shares × price)  in dollars

    Returns DataFrame indexed by quarter-end timestamps.
    """
    if not transactions:
        return pd.DataFrame(columns=['net_direction', 'insider_buy_value'])

    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['quarter_end'] = df['date'] + pd.offsets.QuarterEnd(0)

    rows = []
    for qend, group in df.groupby('quarter_end'):
        buys  = group[group['code'].isin(BUY_CODES)]
        sells = group[group['code'].isin(SELL_CODES)]
        n_b = len(buys)
        n_s = len(sells)
        net_dir    = (n_b - n_s) / (n_b + n_s + 1)
        buy_value  = float((buys['shares'] * buys['price']).sum())
        rows.append({'quarter_end': qend, 'net_direction': net_dir,
                     'insider_buy_value': buy_value})

    if not rows:
        return pd.DataFrame(columns=['net_direction', 'insider_buy_value'])

    result = pd.DataFrame(rows).set_index('quarter_end').sort_index()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load ticker list
    tickers_path = os.path.join(DATA_DIR, 'tickers.pkl')
    if not os.path.exists(tickers_path):
        print('ERROR: tickers.pkl not found. Run notebook 01 first.')
        sys.exit(1)
    with open(tickers_path, 'rb') as f:
        tickers = [t.upper() for t in pickle.load(f)]
    print(f'=== Form 4 Insider Trading Download ===')
    print(f'Tickers: {len(tickers)} | Start date: {START_DATE}\n')

    # CIK mapping
    cik_map = fetch_cik_map(tickers)
    if not cik_map:
        print('ERROR: Could not build CIK map.')
        sys.exit(1)

    # Process each company
    insider_trading: dict[str, pd.DataFrame] = {}
    skipped = 0
    failed  = 0

    for idx, ticker in enumerate(sorted(cik_map.keys()), 1):
        cache_path = os.path.join(CACHE_DIR, f'{ticker}.pkl')

        # Use cache if available
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                insider_trading[ticker] = pickle.load(f)
            continue

        cik = cik_map[ticker]
        prefix = f'[{idx:3d}/{len(cik_map)}] {ticker}'
        try:
            filings = get_form4_filings(cik, START_DATE)
            if not filings:
                print(f'{prefix}: 0 filings')
                df_ticker = pd.DataFrame(columns=['net_direction', 'insider_buy_value'])
            else:
                all_txns = []
                for filing in filings:
                    txns = parse_form4_xml(cik, filing['accession_no'], filing['primary_doc'])
                    all_txns.extend(txns)
                df_ticker = aggregate_to_quarters(all_txns)
                n_qtrs = len(df_ticker)
                n_buys  = sum(1 for t in all_txns if t['code'] in BUY_CODES)
                n_sells = sum(1 for t in all_txns if t['code'] in SELL_CODES)
                print(f'{prefix}: {len(filings)} filings, '
                      f'{n_buys}B/{n_sells}S transactions, {n_qtrs} quarters')

            insider_trading[ticker] = df_ticker
            with open(cache_path, 'wb') as f:
                pickle.dump(df_ticker, f)

        except Exception as e:
            print(f'{prefix}: ERROR — {e}')
            failed += 1
            insider_trading[ticker] = pd.DataFrame(
                columns=['net_direction', 'insider_buy_value'])

    # Save final output
    output_path = os.path.join(DATA_DIR, 'insider_trading.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(insider_trading, f)

    size_kb = os.path.getsize(output_path) / 1024
    n_with_data = sum(1 for df in insider_trading.values() if len(df) > 0)
    print(f'\nSaved insider_trading.pkl ({size_kb:.1f} KB)')
    print(f'Companies with data: {n_with_data}/{len(insider_trading)}')
    if failed:
        print(f'Failed downloads: {failed}')

    # Spot-check Apple
    if 'AAPL' in insider_trading and len(insider_trading['AAPL']) > 0:
        df_aapl = insider_trading['AAPL']
        print(f'\nAAPL sample (last 4 quarters):')
        print(df_aapl.tail(4).to_string())

    print('\nDone!')


if __name__ == '__main__':
    main()
