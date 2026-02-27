"""Download Fama-French portfolio factor returns from Kenneth French's data library.

Downloads the FF5 factors (SMB, HML, RMW, CMA) and Momentum factor (MOM),
forward-fills to daily frequency, and saves as ff_factors.pkl.

No API key required — data is publicly available.

Usage:
    python scripts/download_ff_factors.py
"""

import io
import os
import sys
import zipfile
import pickle

import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
START_DATE = '2005-01-01'

# Kenneth French Data Library (stable URLs)
FF5_URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
           'F-F_Research_Data_5_Factors_2x3_CSV.zip')
MOM_URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
           'F-F_Momentum_Factor_CSV.zip')


def _is_yyyymm(token):
    """Return True if token is a valid YYYYMM integer (e.g. 200301)."""
    try:
        v = int(str(token).strip())
        year, month = v // 100, v % 100
        return 1926 <= year <= 2099 and 1 <= month <= 12
    except (ValueError, TypeError):
        return False


def download_and_parse(url):
    """Download a Fama-French ZIP and return a monthly DataFrame (values in decimal)."""
    print(f'  Fetching {url.split("/")[-1]} ...')
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f'  ERROR downloading: {e}')
        return None

    # Extract the single CSV file from the ZIP
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        fname = z.namelist()[0]
        raw = z.read(fname).decode('utf-8', errors='replace')

    lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # Locate the header line: the line immediately before the first YYYYMM data row
    header_line_idx = None
    first_data_idx = None
    for i, line in enumerate(lines):
        parts = line.replace(',', ' ').split()
        if parts and _is_yyyymm(parts[0]):
            first_data_idx = i
            # The header is the last non-empty line before this
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    header_line_idx = j
                    break
            break

    if first_data_idx is None or header_line_idx is None:
        print('  ERROR: Could not locate data in file.')
        return None

    # Parse header (may be whitespace or comma separated)
    header_text = lines[header_line_idx].strip()
    if ',' in header_text:
        col_names = [c.strip() for c in header_text.split(',') if c.strip()]
    else:
        col_names = header_text.split()

    # Collect monthly data rows until the first blank or non-YYYYMM line
    data_rows = []
    for line in lines[first_data_idx:]:
        stripped = line.strip()
        if not stripped:
            break   # end of monthly section (annual section follows)
        parts = stripped.replace(',', ' ').split()
        if _is_yyyymm(parts[0]):
            data_rows.append(parts)
        else:
            break

    if not data_rows:
        print('  ERROR: No data rows parsed.')
        return None

    # Build DataFrame
    records = []
    for row in data_rows:
        try:
            yyyymm = row[0]
            date = pd.Timestamp(yyyymm[:4] + '-' + yyyymm[4:6]) + pd.offsets.MonthEnd(0)
            values = [float(v) for v in row[1:]]
            records.append([date] + values)
        except (ValueError, IndexError):
            continue

    n_cols = min(len(col_names), max(len(r) - 1 for r in records))
    df = pd.DataFrame(
        [[r[0]] + r[1: n_cols + 1] for r in records],
        columns=['date'] + col_names[:n_cols],
    ).set_index('date')

    df = df / 100.0  # percent → decimal
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print('=== Fama-French Factor Download ===')
    print(f'Start date: {START_DATE}\n')

    # --- FF5 factors ---
    print('Downloading FF5 factors (SMB, HML, RMW, CMA)...')
    df5 = download_and_parse(FF5_URL)
    if df5 is None:
        print('FATAL: Could not download FF5 data.')
        sys.exit(1)
    print(f'  Parsed {len(df5)} monthly rows | columns: {list(df5.columns)}')
    print(f'  Date range: {df5.index.min().date()} to {df5.index.max().date()}')

    # --- Momentum factor ---
    print('\nDownloading Momentum factor (MOM)...')
    df_mom = download_and_parse(MOM_URL)
    if df_mom is not None:
        # Momentum column is named 'Mom' or similar
        mom_col = next(
            (c for c in df_mom.columns if c.lower() in ('mom', 'umd', 'wml', 'pr1yr')),
            None,
        )
        if mom_col:
            df5['MOM'] = df_mom[mom_col].reindex(df5.index)
            print(f'  Added MOM (source column: {mom_col})')
        else:
            print(f'  WARNING: momentum column not identified in {list(df_mom.columns)}')
    else:
        print('  WARNING: Momentum download failed — skipping MOM factor.')

    # --- Filter to project start date ---
    df5 = df5[df5.index >= pd.Timestamp(START_DATE)]
    print(f'\nFiltered to {START_DATE}: {len(df5)} monthly observations')

    # --- Build daily forward-filled series (matches macro_data format) ---
    factors_to_save = ['SMB', 'HML', 'RMW', 'CMA', 'MOM']
    ff_factors = {}
    print('\n=== Resampling to daily frequency ===')
    for col in factors_to_save:
        if col in df5.columns:
            series = df5[col].dropna()
            daily = series.resample('D').ffill()
            ff_factors[col] = daily
            print(f'  {col}: {len(series)} monthly -> {len(daily)} daily observations')
        else:
            print(f'  {col}: not available, skipping')

    if not ff_factors:
        print('ERROR: No factors to save.')
        sys.exit(1)

    # --- Save ---
    output_path = os.path.join(DATA_DIR, 'ff_factors.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(ff_factors, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'\nSaved ff_factors.pkl ({size_kb:.1f} KB)')
    print(f'Factors saved: {list(ff_factors.keys())}')

    # --- Spot-check ---
    print(f'\nSample values at 2020-03-31 (COVID quarter end):')
    ref = pd.Timestamp('2020-03-31')
    for name, series in ff_factors.items():
        val = series.asof(ref)
        tag = f'{val:+.4f}' if pd.notna(val) else 'N/A'
        print(f'  {name}: {tag}')

    print('\nDone!')


if __name__ == '__main__':
    main()
