# Stock Return Prediction with XGBoost

Predicting quarterly S&P 500 stock returns using financial fundamentals, technical indicators, and macroeconomic data.

## Overview

This project builds an XGBoost regression model to predict next-quarter excess stock returns (vs S&P 500 benchmark) based on features derived from real SEC EDGAR filings, technical price indicators, FRED macroeconomic data, and sector-relative metrics. The model is trained on ~436 S&P 500 companies using quarterly data from 2005-2026, with a temporal train/test split to prevent data leakage. Hyperparameters are tuned via Optuna Bayesian optimization with expanding-window time-series cross-validation.

## Results

| Metric | Value |
|--------|-------|
| Direction Accuracy | 54.7% (calibrated) |
| Overfit Ratio | 0.87 |
| Test R2 | -0.018 |
| Features | 15 (after automated selection) |
| Trees | 145 (early stopping) |
| Dataset | ~10,700 samples, ~436 companies |

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
- **Sector Classifications**: [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) - GICS sector and sub-industry

## Methodology

1. **Data Collection** - Scripts download daily stock prices via yfinance, parse quarterly financial statements from SEC EDGAR XBRL data, fetch macroeconomic indicators from FRED, and scrape sector classifications from Wikipedia.

2. **Feature Engineering** - Computed features including profitability ratios, growth metrics, leverage ratios, efficiency metrics, valuation multiples, technical indicators (RSI, MACD, Bollinger Bands, momentum, volatility), macroeconomic context, sector-relative z-scores, and feature interactions. Target is excess return (stock return minus S&P 500 return).

3. **Feature Selection** - Automated pipeline removes zero-variance features, highly correlated features (>0.95 threshold), and features with weak target correlation (<0.03 threshold).

4. **Model Training** - Temporal train/test split ensures the model is only evaluated on future data. Expanding-window time-series cross-validation with 5 folds. Hyperparameter tuning via Optuna Bayesian optimization (150 trials) with early stopping. Huber loss objective for robustness to outliers.

5. **Analysis** - Evaluated using RMSE, R2, MAE, overfit ratio, Spearman rank correlation, and directional accuracy. Includes Optuna-tuned binary classifier for direction prediction.

## Setup

```bash
# Clone repository
git clone https://github.com/chrisbarreras/stock-prediction-ml.git
cd stock-prediction-ml

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Data Collection

A FRED API key is required for macro data (free at https://fred.stlouisfed.org/). SEC EDGAR bulk data must be downloaded separately ([companyfacts.zip](https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip)).

Run notebook 01 to execute all data collection scripts, or run them individually:

```bash
python scripts/download_prices.py       # S&P 500 prices via yfinance (~5 min)
python scripts/download_spy.py          # SPY benchmark prices
python scripts/download_sectors.py      # Sector classifications from Wikipedia
python scripts/download_macro.py        # FRED macro data (requires FRED_API_KEY in .env)
python scripts/extract_financials.py    # Parse SEC EDGAR quarterly filings
```

### Model Pipeline

Run notebook 02 (feature engineering) locally, notebook 03 (model training) on Google Colab with GPU, and notebook 04 (analysis) locally.

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

## Testing

110 automated tests covering feature engineering, data validation, and model output structure. All tests use synthetic data and run without real data files.

```bash
pip install -r requirements-dev.txt
pytest
```

CI runs automatically on push/PR via GitHub Actions.

## Tech Stack

- Python 3.13
- XGBoost, Optuna, scikit-learn, scipy
- yfinance, fredapi
- pandas, NumPy
- matplotlib, seaborn
- pytest, flake8
- GitHub Actions (CI)
- Google Colab (model training)
