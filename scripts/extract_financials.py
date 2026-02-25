"""Extract quarterly financial data from SEC EDGAR XBRL filings.

Parses companyfacts JSON files and maps S&P 500 tickers to CIK numbers
via Wikipedia. Requires SEC EDGAR bulk data already downloaded to:
  data/raw/kaggle/sec_edgar/companyfacts/

Usage:
    python scripts/extract_financials.py
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import requests
from io import StringIO
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SEC_DIR = os.path.join(DATA_DIR, 'raw', 'kaggle', 'sec_edgar')
FACTS_DIR = os.path.join(SEC_DIR, 'companyfacts')

# --- XBRL Tag Mapping ---

INCOME_TAGS = {
    'Total Revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'Revenues',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
    ],
    'Cost of Revenue': [
        'CostOfGoodsAndServicesSold',
        'CostOfRevenue',
        'CostOfGoodsSold',
    ],
    'Gross Profit': ['GrossProfit'],
    'Operating Income': ['OperatingIncomeLoss'],
    'Net Income': ['NetIncomeLoss'],
    'EPS Basic': ['EarningsPerShareBasic'],
    'EPS Diluted': ['EarningsPerShareDiluted'],
    'R&D Expense': ['ResearchAndDevelopmentExpense'],
    'SGA Expense': [
        'SellingGeneralAndAdministrativeExpense',
        'SellingAndMarketingExpense',
    ],
    'Interest Expense': [
        'InterestExpense',
        'InterestExpenseDebt',
    ],
    'Income Tax': ['IncomeTaxExpenseBenefit'],
}

BALANCE_TAGS = {
    'Total Assets': ['Assets'],
    'Current Assets': ['AssetsCurrent'],
    'Total Liabilities': ['Liabilities', 'LiabilitiesAndStockholdersEquity'],
    'Current Liabilities': ['LiabilitiesCurrent'],
    'Long Term Debt': ['LongTermDebtNoncurrent', 'LongTermDebt'],
    'Short Term Debt': ['ShortTermBorrowings', 'CommercialPaper'],
    'Stockholders Equity': ['StockholdersEquity'],
    'Cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments'],
    'Shares Outstanding': ['CommonStockSharesOutstanding'],
}

CASHFLOW_TAGS = {
    'Operating Cash Flow': [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ],
    'Capital Expenditures': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets',
    ],
}


# --- Parser Functions ---

def extract_quarterly_values(facts, tag_names, is_instant=False):
    """Extract single-quarter values from SEC XBRL data."""
    gaap = facts.get('us-gaap', {})

    for tag in tag_names:
        if tag not in gaap:
            continue

        units = gaap[tag].get('units', {})
        if 'USD' in units:
            entries = units['USD']
        elif 'shares' in units:
            entries = units['shares']
        elif 'USD/shares' in units:
            entries = units['USD/shares']
        else:
            continue

        quarterly = [e for e in entries if e.get('form') == '10-Q']
        if not quarterly:
            continue

        values = {}
        for entry in quarterly:
            end_date = pd.Timestamp(entry['end'])

            if is_instant:
                values[end_date] = entry['val']
            else:
                start = entry.get('start')
                if start:
                    days = (datetime.strptime(entry['end'], '%Y-%m-%d') -
                            datetime.strptime(start, '%Y-%m-%d')).days
                    if 85 <= days <= 100:
                        values[end_date] = entry['val']

        if values:
            return values, tag

    return {}, None


def load_company_financials(ticker, cik, facts_dir):
    """Load and parse SEC financial data for one company."""
    json_path = os.path.join(facts_dir, f'CIK{cik}.json')
    with open(json_path) as f:
        data = json.load(f)

    facts = data['facts']

    income_data = {}
    for metric_name, tag_list in INCOME_TAGS.items():
        values, _ = extract_quarterly_values(facts, tag_list, is_instant=False)
        if values:
            income_data[metric_name] = values

    balance_data = {}
    for metric_name, tag_list in BALANCE_TAGS.items():
        values, _ = extract_quarterly_values(facts, tag_list, is_instant=True)
        if values:
            balance_data[metric_name] = values

    cashflow_data = {}
    for metric_name, tag_list in CASHFLOW_TAGS.items():
        values, _ = extract_quarterly_values(facts, tag_list, is_instant=False)
        if values:
            cashflow_data[metric_name] = values

    if income_data:
        income_df = pd.DataFrame(income_data)
        income_df.index = pd.to_datetime(income_df.index)
        income_df = income_df.sort_index()
        income_df = income_df.T
    else:
        income_df = pd.DataFrame()

    all_balance = {**balance_data, **cashflow_data}
    if all_balance:
        balance_df = pd.DataFrame(all_balance)
        balance_df.index = pd.to_datetime(balance_df.index)
        balance_df = balance_df.sort_index()
        balance_df = balance_df.T
    else:
        balance_df = pd.DataFrame()

    return {
        'ticker': ticker,
        'quarterly_income': income_df,
        'quarterly_balance': balance_df,
        'info': {'symbol': ticker},
    }


# --- Main ---

print(f'SEC data dir: {os.path.abspath(FACTS_DIR)}')
print(f'SEC files found: {len(os.listdir(FACTS_DIR))}')

# Load existing price data
price_path = os.path.join(DATA_DIR, 'price_data.pkl')
with open(price_path, 'rb') as f:
    price_data = pickle.load(f)
price_tickers = set(price_data.keys())
print(f'Price data: {len(price_tickers)} companies')

# Get S&P 500 CIK mapping from Wikipedia
print('\nFetching S&P 500 CIK mapping from Wikipedia...')
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0 (stock-prediction-ml project)'}
response = requests.get(url, headers=headers)
response.raise_for_status()
tables = pd.read_html(StringIO(response.text))
sp500_df = tables[0]

ticker_to_cik = {}
for _, row in sp500_df.iterrows():
    ticker_wiki = str(row['Symbol']).strip()
    ticker_yf = ticker_wiki.replace('.', '-')
    try:
        cik = str(int(row['CIK'])).zfill(10)
    except (ValueError, TypeError):
        continue
    json_path = os.path.join(FACTS_DIR, f'CIK{cik}.json')
    if os.path.exists(json_path):
        if ticker_yf in price_tickers:
            ticker_to_cik[ticker_yf] = cik
        elif ticker_wiki in price_tickers:
            ticker_to_cik[ticker_wiki] = cik

print(f'S&P 500 tickers with SEC EDGAR + price data: {len(ticker_to_cik)}')

# Extract financial data
print(f'\n=== Parsing {len(ticker_to_cik)} companies ===')
print('(quality thresholds: income_metrics >= 4, quarters >= 8)\n')

financial_data = []
skipped = []
errors = []

for i, (ticker, cik) in enumerate(sorted(ticker_to_cik.items()), 1):
    try:
        company = load_company_financials(ticker, cik, FACTS_DIR)

        income_metrics = len(company['quarterly_income'])
        balance_metrics = len(company['quarterly_balance'])
        quarters = company['quarterly_income'].shape[1] if not company['quarterly_income'].empty else 0

        if income_metrics >= 4 and quarters >= 8:
            financial_data.append(company)
            if i % 50 == 0 or i == len(ticker_to_cik):
                print(f'  [{i}/{len(ticker_to_cik)}] {ticker}: '
                      f'{income_metrics} income + {balance_metrics} balance, {quarters} quarters')
        else:
            skipped.append(f'{ticker} ({income_metrics} income, {quarters} quarters)')

    except Exception as e:
        errors.append(f'{ticker}: {e}')

print(f'\n--- Results ---')
print(f'Loaded: {len(financial_data)} companies')
print(f'Skipped: {len(skipped)} (insufficient data)')
print(f'Errors: {len(errors)}')

if skipped:
    print(f'\nSkipped: {skipped[:20]}')
    if len(skipped) > 20:
        print(f'  ... and {len(skipped) - 20} more')
if errors:
    print(f'\nErrors: {errors[:10]}')

# Data quality report
print(f'\n=== Data Quality Report ===')

all_income_metrics = set()
all_balance_metrics = set()
for company in financial_data:
    all_income_metrics.update(company['quarterly_income'].index.tolist())
    all_balance_metrics.update(company['quarterly_balance'].index.tolist())

print(f'\nMetric Coverage (% of companies):')
for metric in sorted(all_income_metrics):
    count = sum(1 for c in financial_data if metric in c['quarterly_income'].index)
    pct = count / len(financial_data) * 100
    print(f'  {metric:<30s} {count:>3}/{len(financial_data)} ({pct:.0f}%)')
for metric in sorted(all_balance_metrics):
    count = sum(1 for c in financial_data if metric in c['quarterly_balance'].index)
    pct = count / len(financial_data) * 100
    print(f'  {metric:<30s} {count:>3}/{len(financial_data)} ({pct:.0f}%)')

quarters_list = [c['quarterly_income'].shape[1] for c in financial_data
                 if not c['quarterly_income'].empty]
print(f'\nQuarters per Company:')
print(f'  Min: {min(quarters_list)}, Max: {max(quarters_list)}, '
      f'Mean: {np.mean(quarters_list):.1f}, Median: {np.median(quarters_list):.0f}')

# Save
with open(os.path.join(DATA_DIR, 'financial_data.pkl'), 'wb') as f:
    pickle.dump(financial_data, f)

final_tickers = [c['ticker'] for c in financial_data]
with open(os.path.join(DATA_DIR, 'tickers.pkl'), 'wb') as f:
    pickle.dump(final_tickers, f)

print(f'\n=== Saved ===')
print(f'  financial_data.pkl - {len(financial_data)} companies')
print(f'  tickers.pkl - {len(final_tickers)} tickers')

# Verify
sample = financial_data[0]
print(f'\nVerification ({sample["ticker"]}):')
print(f'  Income: {sample["quarterly_income"].shape} (metrics x quarters)')
print(f'  Balance: {sample["quarterly_balance"].shape}')
print(f'\nDone!')
