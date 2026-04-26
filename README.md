# Stock Return Prediction with XGBoost

[![CI](https://github.com/Thomas-J-Barreras-Consulting/stock-prediction-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Thomas-J-Barreras-Consulting/stock-prediction-ml/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Thomas-J-Barreras-Consulting/stock-prediction-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/Thomas-J-Barreras-Consulting/stock-prediction-ml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting quarterly S&P 500 stock returns using financial fundamentals, technical indicators, and macroeconomic data.

![Portfolio Strategy Performance vs. Buy & Hold](results/prediction_analysis.png)

## Overview

This project builds an XGBoost regression model to predict next-quarter excess stock returns (vs S&P 500 benchmark) based on features derived from real SEC EDGAR filings, technical price indicators, FRED macroeconomic data, and sector-relative metrics. The model is trained on ~488 S&P 500 companies using quarterly data from 2008-2026, with a temporal train/test split to prevent data leakage. Hyperparameters are tuned via Optuna Bayesian optimization with expanding-window time-series cross-validation.

## Machine Learning Pipeline

```mermaid
flowchart LR
    A[yfinance<br/>prices] --> F[Feature<br/>Engineering]
    B[SEC EDGAR<br/>XBRL filings] --> F
    C[FRED<br/>macro data] --> F
    D[Wikipedia<br/>GICS sectors] --> F
    E[Fama-French<br/>factors] --> F
    F --> G[Optuna +<br/>Time-Series CV]
    G --> H[XGBoost<br/>model]
    H --> I[Direction accuracy<br/>R2 / overfit ratio]
```

## Results

| Metric | Value |
|--------|-------|
| Direction Accuracy | 53.0% (calibrated) |
| Overfit Ratio | 0.85 |
| Test R2 | -0.016 |
| Features | 21 (after automated selection) |
| Trees | 51 (Optuna-selected) |
| Dataset | ~12,150 samples, 488 companies |

## Features

The model uses features across 9 categories, automatically selected from candidates via correlation filtering:

**Profitability** - revenue, revenue_growth, profit_margin, operating_margin, net_income, net_income_growth, operating_income_growth

**Per-Share** - eps_diluted, eps_growth

**Expense Ratios** - rd_ratio, sga_ratio, tax_rate

**Balance Sheet** - total_assets, debt_to_assets, debt_to_equity, current_ratio, cash_ratio, equity_ratio

**Returns & Efficiency** - roa, roe, asset_turnover, interest_coverage

**Cash Flow & Valuation** - operating_cash_flow, free_cash_flow, fcf_margin, market_cap, pe_ratio, price_to_book, quarter_price

**Technical Indicators** - ma_50_ratio, ma_200_ratio, momentum_3m, volatility, rsi_14, macd_histogram, bollinger_width, volume_trend, price_to_52wk_high

**Macro** - gs10 (10yr Treasury), vix, unemployment, gdp, cpi

**Sector-Relative & Interactions** - sector z-scores, sector differences, momentum x quality, risk x leverage, growth x profitability

## Project Structure

```
stock-prediction-ml/
├── scripts/                             # Data collection (run these first)
│   ├── download_prices.py               # S&P 500 price data via yfinance
│   ├── download_spy.py                  # SPY benchmark prices via yfinance
│   ├── download_sectors.py              # Sector classifications from Wikipedia
│   ├── download_macro.py                # FRED macroeconomic indicators
│   ├── download_ff_factors.py           # Fama-French 5 factors + momentum
│   ├── download_sec_edgar.py            # SEC EDGAR bulk XBRL data (~1.5 GB compressed)
│   └── extract_financials.py            # Parse SEC EDGAR quarterly filings
├── notebooks/
│   ├── 01_data_collection.ipynb         # Run all data collection scripts
│   ├── 02_feature_engineering.ipynb      # Transform raw data into ML features
│   ├── 03_model_training.ipynb          # Train XGBoost model (Google Colab)
│   └── 04_analysis.ipynb               # Evaluate results and visualizations
├── data/
│   ├── raw/kaggle/sec_edgar/            # SEC EDGAR XBRL filings
│   └── processed_dataset.csv           # Final ML-ready dataset
├── src/
│   ├── __init__.py
│   └── features.py                      # Importable feature engineering functions
├── tests/
│   ├── conftest.py                      # Shared pytest fixtures (synthetic data)
│   ├── test_data_validation.py          # Data format and range validation
│   ├── test_feature_engineering.py      # Feature function unit tests
│   └── test_model_validation.py         # Model output structure tests
├── models/
│   └── model_results.pkl                # Trained model, metrics, and predictions
├── results/                             # Generated charts and visualizations
├── .github/workflows/ci.yml             # GitHub Actions CI pipeline
├── requirements.txt
└── requirements-dev.txt                 # Test dependencies (pytest, flake8)
```

## Data Sources

- **Stock Prices**: [Yahoo Finance](https://finance.yahoo.com/) via yfinance - Daily OHLCV for ~500 S&P 500 companies (2005-present)
- **Financial Statements**: [SEC EDGAR](https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip) - Real quarterly filings (10-Q) parsed from XBRL
- **Benchmark Prices**: [Yahoo Finance](https://finance.yahoo.com/) via yfinance - S&P 500 (SPY) daily prices for excess return calculation
- **Macroeconomic Data**: [FRED](https://fred.stlouisfed.org/) - 10yr Treasury, VIX, unemployment, GDP, CPI
- **Fama-French Factors**: [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) - FF5 factors (SMB, HML, RMW, CMA) plus momentum (MOM)
- **Sector Classifications**: [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) - GICS sector and sub-industry

## Methodology

1. **Data Collection** - Scripts download daily stock prices via yfinance, parse quarterly financial statements from SEC EDGAR XBRL data, fetch macroeconomic indicators from FRED, and scrape sector classifications from Wikipedia.

2. **Feature Engineering** - Computed features including profitability ratios, growth metrics, leverage ratios, efficiency metrics, valuation multiples, technical indicators (RSI, MACD, Bollinger Bands, momentum, volatility), macroeconomic context, sector-relative z-scores, and feature interactions. Target is excess return (stock return minus S&P 500 return).

   ![Correlation Heatmap](results/correlation_heatmap.png)

3. **Feature Selection** - Automated pipeline removes zero-variance features, highly correlated features (>0.95 threshold), and features with weak target correlation (<0.03 threshold).

   ![Feature Importance](results/feature_importance.png)

4. **Model Training** - Temporal train/test split ensures the model is only evaluated on future data. Expanding-window time-series cross-validation with 5 folds. Hyperparameter tuning via Optuna Bayesian optimization (150 trials), with `n_estimators` tuned directly by Optuna rather than early stopping (more reliable on noisy financial data). Huber loss objective for robustness to outliers.

5. **Prediction Calibration** - Huber loss with heavy regularization makes the regressor very conservative; raw predictions cluster tightly near zero (std ≈ 0.004 vs actual return std ≈ 0.14), so noise pushes most predictions to the wrong side of the zero line and direction accuracy collapses to 43% (worse than random). A variance-matching affine transform — `(pred - pred_mean) × (train_std / pred_std) + train_mean` — rescales predictions to match the training-set distribution. This preserves rank order (Spearman correlation unchanged) but unsquashes magnitudes across the zero line, lifting direction accuracy from 43% → 53%. The tradeoff is that R² gets *more* negative (squared errors grow with prediction magnitude).

   ![Calibration Diagnostics](results/calibration_diagnostics.png)

6. **Analysis** - Evaluated using RMSE, R2, MAE, overfit ratio, Spearman rank correlation, and directional accuracy. Includes Optuna-tuned logistic regression classifier for direction prediction. The hero chart at the top of this README converts the model's rank ordering into investable performance: it plots SPY benchmark vs three conviction tiers (all predicted-outperform names, top quartile, top decile, all selected within each quarter to avoid lookahead bias).

## Reproducing Results

End-to-end runbook from a clean checkout.

### Prerequisites

- **Python 3.12+**
- **FRED API key** — free at [fred.stlouisfed.org](https://fred.stlouisfed.org/). Save as `FRED_API_KEY=...` in a `.env` file at the project root.
- **SEC EDGAR bulk data** — fetched automatically by `scripts/download_sec_edgar.py` (~1.5 GB compressed, ~15 GB extracted, ~13,000 JSON files). Optionally set `SEC_USER_AGENT="Your Name your@email.com"` to identify SEC requests with your contact info; otherwise a generic project-repo identifier is used.
- **Google Colab account** (*optional*) — convenient for portable training; local Jupyter execution is also supported (notebook 03 auto-detects).

> **Note**: `data/` and `models/` artifacts are not checked into git. Each step below must be re-run per machine.

### 1. Environment setup (~2 min)

```bash
git clone https://github.com/Thomas-J-Barreras-Consulting/stock-prediction-ml.git
cd stock-prediction-ml

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Data collection (~20–25 min)

Run [notebook 01](notebooks/01_data_collection.ipynb) top-to-bottom, or invoke the scripts individually:

```bash
python scripts/download_prices.py       # S&P 500 daily OHLCV    -> data/price_data.pkl
python scripts/download_spy.py          # SPY benchmark prices   -> data/spy_data.pkl
python scripts/download_sectors.py      # GICS sectors           -> data/sector_data.pkl
python scripts/download_macro.py        # FRED indicators        -> data/macro_data.pkl
python scripts/download_ff_factors.py   # Fama-French factors    -> data/ff_factors.pkl
python scripts/download_sec_edgar.py    # SEC EDGAR bulk dump    -> data/raw/kaggle/sec_edgar/companyfacts/
python scripts/extract_financials.py    # SEC EDGAR XBRL filings -> data/financial_data.pkl
```

### 3. Feature engineering (~1-3 min, local)

Run [notebook 02](notebooks/02_feature_engineering.ipynb) top-to-bottom. Produces `data/processed_dataset.csv` (~12,150 quarterly samples × ~50 features).

### 4. Model training (~1-30 min)

Notebook 03 auto-detects whether it's running in Colab or locally and resolves data/model paths accordingly:

- **Local Jupyter**: `jupyter notebook notebooks/03_model_training.ipynb` and run all cells. Reads from `data/processed_dataset.csv`, writes to `models/model_results.pkl`.
- **Google Colab**: upload the notebook, mount Drive at `/content/drive/MyDrive/stock_prediction_data/`, place `processed_dataset.csv` there, and run all cells. The notebook auto-installs `xgboost` and `optuna`.

Note: training currently runs on **CPU** (no `tree_method='gpu_hist'` set). With ~10k samples × 23 features × 150 Optuna trials, CPU is competitive — GPU overhead would likely make it slower at this scale.

### 5. Analysis (~1-2 min, local)

Run [notebook 04](notebooks/04_analysis.ipynb) to regenerate the charts in [results/](results/) and evaluate direction accuracy, R², overfit ratio, and feature importances.

## Limitations

- Negative test R2 indicates the model doesn't generalize well on unseen future periods — this is typical for stock prediction
- No sentiment or alternative data sources
- Single model architecture (XGBoost only)
- Quarterly prediction horizon limits sample count per company

## Potential Improvements

- Experiment with LSTM or transformer models for time series
- Add sentiment analysis from earnings calls or news
- Ensemble methods combining multiple model architectures
- Shorter prediction horizons (monthly) for more training samples

## CI Pipeline

The repo treats research code as production code. Every push runs a multi-job [GitHubActions pipeline](https://github.com/Thomas-J-Barreras-Consulting/stock-prediction-ml/actions) defined by [ci.yml](.github/workflows/ci.yml):

```mermaid
flowchart LR
    A[Push or PR<br/>to master] --> B[Lint &amp; Types<br/>black, isort,<br/>flake8, mypy]
    A --> C[Security Audit<br/>pip-audit]
    A --> D[Tests &amp; Coverage<br/>pytest, codecov]
    A --> E[Docker Build]
    B --> F[CI Status<br/>all jobs required]
    C --> F
    D --> F
    E --> F
```

| Job | Tools | Purpose |
|---|---|---|
| **Lint & Type Check** | black, isort, flake8, mypy | Enforce formatting (line length 120), import order, style, and static types |
| **Security Audit** | pip-audit | Scan dependencies for known CVEs (`--strict`) |
| **Tests & Coverage** | pytest, codecov | 110 unit tests on synthetic data, coverage uploaded to Codecov |
| **Docker Build** | docker | Verify the [Dockerfile](Dockerfile) builds cleanly on every commit |

### Testing Strategy

The 110 tests aren't "happy path" smoke tests — they're designed as guardrails for an evolving ML pipeline:

| Category | Examples | Why it matters |
|---|---|---|
| **Pure helpers** | `safe_divide`, `safe_loc`, `safe_growth` exercised against NaN, zero, negative bases, missing keys, empty DataFrames | Financial data is messy; defensive helpers are the foundation everything else builds on |
| **Data contracts** | Shape and value ranges of every external source — yfinance OHLCV (DatetimeIndex, High ≥ Low, prices > 0), SEC EDGAR financial DataFrames, FRED indicator dict, Wikipedia GICS sector dict | Catches upstream API/format changes before bad data reaches the model |
| **Cleaning pipeline** | End-to-end: winsorize → forward-fill → sector-median → `fillna(0)`. Verified against `sample_dirty_dataset` — a fixture built from real issues hit when scaling to 438 companies (45% NaN in `debt_to_equity`/`interest_coverage`, ROE outliers in the millions, negative debt-to-equity) | Pipeline correctness is verified against *observed* production data quality, not theoretical edge cases |
| **Model & metrics** | XGBoost trains on synthetic data; predictions finite, importances sum to 1; direction-accuracy formula verified manually (4/5 = 0.8); RSI bounded [0,100] with overbought/oversold thresholds; sector z-scores have ≈0 mean within sector | Math correctness, not just "doesn't crash" |
| **Inter-notebook contract** | `test_results_dict_structure` verifies notebook 03's saved `model_results.pkl` has the keys notebook 04 expects | Guards the Colab → local notebook handoff, a typically unmonitored gap in ML notebook projects |

All fixtures are synthetic ([tests/conftest.py](tests/conftest.py)) — no real data files, no network calls, no API keys needed. The full suite completes in seconds, so every push is fully gated.

Run the full local check before pushing:

```bash
pip install -r requirements-dev.txt
black --check --line-length 120 src/ tests/ scripts/
isort --check-only --profile black --line-length 120 src/ tests/ scripts/
flake8 src/ tests/ --max-line-length 120 --ignore E501,W503
mypy src/ --ignore-missing-imports
pytest tests/ -v --cov=src
```

## Tech Stack

- **Language**: Python 3.12+
- **ML**: XGBoost, Optuna, scikit-learn, scipy
- **Data**: pandas, NumPy, yfinance, fredapi
- **Visualization**: matplotlib, seaborn
- **Quality**: pytest, black, isort, flake8, mypy, pip-audit
- **Infra**: GitHub Actions (CI), Docker, Google Colab (optional), Codecov
