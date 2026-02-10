# Stock Return Prediction with XGBoost

Predicting quarterly S&P 500 stock returns using financial fundamentals and machine learning.

## Overview

This project builds an XGBoost regression model to predict next-quarter stock returns based on 26 financial features derived from real SEC EDGAR filings. The model is trained on 32 S&P 500 companies using quarterly data from 2013-2018.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 0.1316 |
| Test R2 | 0.1839 |
| CV RMSE | 0.1296 (+/- 0.0077) |
| Direction Accuracy | 79.5% |

## Features

The model uses 26 financial features across 6 categories:

**Profitability** - revenue, revenue_growth, profit_margin, operating_margin, net_income, net_income_growth, operating_income_growth

**Per-Share** - eps_diluted, eps_growth

**Expense Ratios** - rd_ratio, sga_ratio, tax_rate

**Balance Sheet** - total_assets, debt_to_assets, debt_to_equity, current_ratio, cash_ratio, equity_ratio

**Returns & Efficiency** - roa, roe, asset_turnover, interest_coverage

**Cash Flow & Valuation** - market_cap, pe_ratio, price_to_book, quarter_price

## Project Structure

```
stock-prediction-ml/
├── notebooks/
│   ├── 01_data_collection.ipynb       # Load Kaggle price data
│   ├── 01b_real_data_collection.ipynb  # Parse SEC EDGAR financial data
│   ├── 02_feature_engineering.ipynb    # Transform raw data into 26 ML features
│   ├── 03_model_training.ipynb        # Train XGBoost model (Google Colab)
│   └── 04_analysis.ipynb              # Evaluate results and visualizations
├── data/
│   ├── raw/kaggle/                    # S&P 500 price data + SEC EDGAR filings
│   └── processed_dataset.csv          # Final ML-ready dataset (192 samples)
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

- **Stock Prices**: [Kaggle S&P 500 Dataset](https://www.kaggle.com/datasets/camnugent/sandp500) - Real historical daily prices (2013-2018)
- **Financial Statements**: [SEC EDGAR](https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip) - Real quarterly filings (10-Q) parsed from XBRL

### Companies Analyzed

32 S&P 500 companies across sectors including technology, finance, healthcare, consumer, and industrials.

## Methodology

1. **Data Collection** - Loaded 5 years of daily stock prices from Kaggle. Parsed real quarterly financial statements from SEC EDGAR bulk XBRL data, extracting income statement, balance sheet, and cash flow metrics with tag fallback logic.

2. **Feature Engineering** - Computed 26 derived features including profitability ratios, growth metrics, leverage ratios, efficiency metrics, and valuation multiples. Matched stock prices to quarter-end dates and computed 90-day forward returns as the prediction target.

3. **Model Training** - Trained XGBoost regressor with max_depth=4, 200 estimators, and learning rate of 0.05. Used 80/20 train/test split with 5-fold cross-validation.

4. **Analysis** - Evaluated model using RMSE, R2, MAE, cross-validation RMSE, and directional accuracy. Analyzed feature importance and prediction patterns.

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

Run notebooks 01, 01b, 02, and 04 locally. Notebook 03 was designed for Google Colab with GPU support.

## Limitations

- Limited to 2013-2018 period due to Kaggle price data coverage
- 192 samples from 32 companies (companies with incomplete SEC filings excluded)
- No technical indicators (purely fundamental analysis)

## Potential Improvements

- Add technical indicators (RSI, MACD, moving averages)
- Extend price data beyond 2013-2018 for more samples
- Experiment with LSTM or transformer models for time series
- Add sector-relative features for cross-company comparison

## Tech Stack

- Python 3.12
- XGBoost
- pandas, NumPy, scikit-learn
- matplotlib, seaborn
- Google Colab (model training)
