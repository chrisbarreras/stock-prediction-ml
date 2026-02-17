# Stock Return Prediction with XGBoost

Predicting quarterly S&P 500 stock returns using financial fundamentals, technical indicators, and macroeconomic data.

## Overview

This project builds an XGBoost regression model to predict next-quarter stock returns based on 42 features derived from real SEC EDGAR filings, technical price indicators, FRED macroeconomic data, and sector-relative metrics. The model is trained on 45 S&P 500 companies using quarterly data from 2009-2026.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 0.1202 |
| Test R2 | 0.2665 |
| Direction Accuracy | 72.4% |
| Train/Test Split | 984/246 |

## Features

The model uses 42 features across 8 categories:

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
│   └── download_sectors.py            # Wikipedia sector scraper script
├── data/
│   ├── raw/kaggle/                    # SEC EDGAR filings + Kaggle price data
│   └── processed_dataset.csv          # Final ML-ready dataset (1,230 samples)
├── models/
│   └── model_results.pkl              # Trained model and metrics
├── results/
│   ├── data_exploration.png           # Dataset distributions
│   ├── feature_importance.png         # XGBoost feature importance
│   ├── feature_importance_analysis.png
│   ├── prediction_analysis.png        # Predicted vs actual returns
│   ├── correlation_heatmap.png        # Feature correlations
│   └── model_results.png             # Model performance plots
└── requirements.txt
```

## Data Sources

- **Stock Prices**: [Yahoo Finance](https://finance.yahoo.com/) via yfinance - Daily OHLCV for 502 S&P 500 companies (2009-2026)
- **Financial Statements**: [SEC EDGAR](https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip) - Real quarterly filings (10-Q) parsed from XBRL
- **Macroeconomic Data**: [FRED](https://fred.stlouisfed.org/) - 10yr Treasury, VIX, unemployment, GDP, CPI
- **Sector Classifications**: [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) - GICS sector and sub-industry

### Companies Analyzed

45 S&P 500 companies across sectors including technology, finance, healthcare, consumer, and industrials.

## Methodology

1. **Data Collection** - Downloaded 17 years of daily stock prices via yfinance for 502 S&P 500 companies. Parsed real quarterly financial statements from SEC EDGAR bulk XBRL data. Downloaded macroeconomic indicators from FRED and sector classifications from Wikipedia.

2. **Feature Engineering** - Computed 42 features including profitability ratios, growth metrics, leverage ratios, efficiency metrics, valuation multiples, technical indicators (moving average ratios, momentum, volatility), macroeconomic context, and sector-relative comparisons. Matched stock prices to quarter-end dates and computed 90-day forward returns as the prediction target.

3. **Model Training** - Trained XGBoost regressor with max_depth=4, 200 estimators, and learning rate of 0.05. Used 80/20 train/test split with 5-fold cross-validation.

4. **Analysis** - Evaluated model using RMSE, R2, MAE, cross-validation RMSE, and directional accuracy. Analyzed feature importance and prediction patterns across 45 companies.

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
- No sentiment or alternative data sources
- Single model architecture (XGBoost only)

## Potential Improvements

- Experiment with LSTM or transformer models for time series
- Add sentiment analysis from earnings calls or news
- Implement time-series cross-validation for more realistic evaluation
- Use excess returns (vs S&P 500) as target variable

## Tech Stack

- Python 3.12
- XGBoost
- yfinance, fredapi
- pandas, NumPy, scikit-learn
- matplotlib, seaborn
- Google Colab (model training)
