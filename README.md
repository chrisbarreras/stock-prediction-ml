# Stock Return Prediction with XGBoost

Predicting quarterly S&P 500 stock returns using financial fundamentals, technical indicators, and macroeconomic data.

## Overview

This project builds an XGBoost regression model to predict next-quarter stock returns based on 37 features (selected from 42 candidates) derived from real SEC EDGAR filings, technical price indicators, FRED macroeconomic data, and sector-relative metrics. The model is trained on 45 S&P 500 companies using quarterly data from 2009-2026, with a temporal train/test split to prevent data leakage.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 0.1420 |
| Test R2 | -0.0124 |
| Direction Accuracy | 60.4% |
| CV RMSE | 0.1325 (+/- 0.0251) |
| Overfit Ratio | 0.88 |
| Temporal Split | 995 train / 235 test |

## Features

The model uses 37 features (after automated selection from 42 candidates) across 8 categories:

**Profitability** - revenue, revenue_growth, profit_margin, operating_margin, net_income, net_income_growth, operating_income_growth

**Per-Share** - eps_diluted, eps_growth

**Expense Ratios** - rd_ratio, sga_ratio, tax_rate

**Balance Sheet** - total_assets, debt_to_assets, debt_to_equity, current_ratio, cash_ratio, equity_ratio

**Returns & Efficiency** - roa, roe, asset_turnover, interest_coverage

**Cash Flow & Valuation** - operating_cash_flow, free_cash_flow, fcf_margin, market_cap, pe_ratio, price_to_book, quarter_price

**Technical Indicators** - ma_50_ratio, ma_200_ratio, momentum_3m, volatility

**Macro & Sector** - gs10 (10yr Treasury), vix, unemployment, gdp, cpi, profit_margin_vs_sector, operating_margin_vs_sector, roe_vs_sector, revenue_growth_vs_sector, debt_to_equity_vs_sector, pe_ratio_vs_sector

## Project Structure

```
stock-prediction-ml/
├── notebooks/
│   ├── 01_data_collection.ipynb       # Load Kaggle price data (legacy)
│   ├── 01b_real_data_collection.ipynb  # Parse SEC EDGAR financial data
│   ├── 01c_yfinance_prices.ipynb      # Download extended price data (2009-present)
│   ├── 01d_sector_data.ipynb          # S&P 500 sector classifications
│   ├── 01e_fred_macro.ipynb           # FRED macroeconomic indicators
│   ├── 02_feature_engineering.ipynb    # Transform raw data into 42 ML features
│   ├── 03_model_training.ipynb        # Train XGBoost model (Google Colab)
│   └── 04_analysis.ipynb              # Evaluate results and visualizations
├── scripts/
│   ├── download_prices.py             # yfinance price download script
│   ├── download_sectors.py            # Wikipedia sector scraper script
│   └── download_spy.py               # SPY benchmark data download script
├── data/
│   ├── raw/kaggle/                    # SEC EDGAR filings + Kaggle price data
│   └── processed_dataset.csv          # Final ML-ready dataset (1,230 samples)
├── src/
│   ├── __init__.py
│   └── features.py                    # Importable feature engineering functions
├── tests/
│   ├── conftest.py                    # Shared pytest fixtures (synthetic data)
│   ├── test_data_validation.py        # Data format and range validation
│   ├── test_feature_engineering.py    # Feature function unit tests
│   └── test_model_validation.py       # Model output structure tests
├── models/
│   └── model_results.pkl              # Trained model, metrics, and predictions
├── results/
│   ├── data_exploration.png           # Dataset distributions
│   ├── feature_importance.png         # XGBoost feature importance
│   ├── feature_importance_analysis.png
│   ├── prediction_analysis.png        # Predicted vs actual returns
│   ├── correlation_heatmap.png        # Feature correlations
│   └── model_results.png             # Model performance plots
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI pipeline
├── requirements.txt
└── requirements-dev.txt               # Test dependencies (pytest, flake8)
```

## Data Sources

- **Stock Prices**: [Yahoo Finance](https://finance.yahoo.com/) via yfinance - Daily OHLCV for 502 S&P 500 companies (2009-2026)
- **Financial Statements**: [SEC EDGAR](https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip) - Real quarterly filings (10-Q) parsed from XBRL
- **Benchmark Prices**: [Yahoo Finance](https://finance.yahoo.com/) via yfinance - S&P 500 (SPY) daily prices for benchmark returns
- **Macroeconomic Data**: [FRED](https://fred.stlouisfed.org/) - 10yr Treasury, VIX, unemployment, GDP, CPI
- **Sector Classifications**: [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) - GICS sector and sub-industry

### Companies Analyzed

45 S&P 500 companies across sectors including technology, finance, healthcare, consumer, and industrials.

## Methodology

1. **Data Collection** - Downloaded 17 years of daily stock prices via yfinance for 502 S&P 500 companies. Parsed real quarterly financial statements from SEC EDGAR bulk XBRL data. Downloaded macroeconomic indicators from FRED, sector classifications from Wikipedia, and S&P 500 (SPY) benchmark prices.

2. **Feature Engineering** - Computed 42 candidate features including profitability ratios, growth metrics, leverage ratios, efficiency metrics, valuation multiples, technical indicators (moving average ratios, momentum, volatility), macroeconomic context, and sector-relative comparisons. Matched stock prices to quarter-end dates and computed 90-day forward returns as the prediction target.

3. **Feature Selection** - Automated pipeline removes zero-variance features and highly correlated features (>0.95 correlation threshold), reducing from 42 to 37 features.

4. **Model Training** - Temporal train/test split (cutoff: 2023-07-28) ensures the model is only evaluated on future data it hasn't seen. Expanding-window time-series cross-validation with 5 folds. Hyperparameter tuning via RandomizedSearchCV (50 iterations) over 8 XGBoost parameters.

5. **Analysis** - Evaluated model using RMSE, R2, MAE, overfit ratio, cross-validation RMSE, and directional accuracy. Analyzed feature importance and prediction patterns across 45 companies.

## Setup

```bash
# Clone repository
git clone https://github.com/chrisbarreras/stock-prediction-ml.git
cd stock-prediction-ml

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

Run notebooks 01b, 01c, 01d, 01e, 02, and 04 locally. Notebook 03 was designed for Google Colab with GPU support. A FRED API key is required for notebook 01e (free at https://fred.stlouisfed.org/).

## Limitations

- 1,230 samples from 45 companies (limited by SEC filing completeness)
- Negative test R2 indicates the model doesn't generalize well on unseen future periods
- No sentiment or alternative data sources
- Single model architecture (XGBoost only)

## Potential Improvements

- Experiment with LSTM or transformer models for time series
- Add sentiment analysis from earnings calls or news
- Expand dataset with more companies and longer history
- Ensemble methods combining multiple model architectures

## Testing

75 automated tests covering feature engineering, data validation, and model output structure. All tests use synthetic data and run without real data files.

```bash
pip install -r requirements-dev.txt
pytest
```

CI runs automatically on push/PR via GitHub Actions.

## Tech Stack

- Python 3.12
- XGBoost
- yfinance, fredapi
- pandas, NumPy, scikit-learn
- matplotlib, seaborn
- pytest, flake8
- GitHub Actions (CI)
- Google Colab (model training)
