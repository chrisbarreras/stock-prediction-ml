# Stock Return Prediction with XGBoost

Predicting quarterly S&P 500 stock returns using financial fundamentals and machine learning.

## Overview

This project builds an XGBoost regression model to predict next-quarter stock returns based on company financial metrics including revenue growth, profit margins, and balance sheet ratios. The model is trained on 10 major S&P 500 companies over a 2-year period (2016-2018).

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | See notebook 04 |
| Test R2 | See notebook 04 |
| Direction Accuracy | See notebook 04 |

## Features

The model uses 8 financial features:

- **Revenue** - Total quarterly revenue
- **Revenue Growth** - Quarter-over-quarter revenue change
- **Profit Margin** - Net income / revenue
- **Gross Margin** - Gross profit / revenue
- **Net Income** - Bottom line earnings
- **Total Assets** - Company size indicator
- **Debt-to-Assets** - Leverage ratio
- **Quarter Price** - Stock price at quarter end

## Project Structure

```
stock-prediction-ml/
├── notebooks/
│   ├── 01_data_collection.ipynb       # Load Kaggle price data + synthetic financials
│   ├── 02_feature_engineering.ipynb    # Transform raw data into ML features
│   ├── 03_model_training.ipynb        # Train XGBoost model (Google Colab)
│   └── 04_analysis.ipynb              # Evaluate results and visualizations
├── data/
│   ├── raw/kaggle/                    # S&P 500 historical price data
│   └── processed_dataset.csv          # Final ML-ready dataset (60 samples)
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
- **Financial Statements**: Synthetically generated quarterly data with realistic patterns

### Companies Analyzed

AAPL, MSFT, GOOGL, AMZN, FB, INTC, NVDA, V, JPM, UNH

## Methodology

1. **Data Collection** - Loaded 5 years of daily stock prices from Kaggle for 10 S&P 500 companies. Generated synthetic quarterly financial statements.

2. **Feature Engineering** - Calculated financial ratios and growth metrics for each company-quarter. Matched stock prices to quarter-end dates and computed 90-day forward returns as the prediction target.

3. **Model Training** - Trained XGBoost regressor with max_depth=3, 100 estimators, and learning rate of 0.1. Used 80/20 train/test split.

4. **Analysis** - Evaluated model using RMSE, R2, MAE, and directional accuracy. Analyzed feature importance and prediction patterns.

## Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/stock-prediction-ml.git
cd stock-prediction-ml

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

Run notebooks 01, 02, and 04 locally. Notebook 03 was designed for Google Colab with GPU support.

## Limitations

- Small dataset (60 samples from 10 companies)
- Synthetic financial data (not real SEC filings)
- Limited to 2016-2018 period due to data overlap
- Simple feature set without technical indicators

## Potential Improvements

- Use real financial statement data from SEC EDGAR API
- Add technical indicators (RSI, MACD, moving averages)
- Include more companies and longer time periods
- Experiment with LSTM or transformer models for time series
- Add cross-validation for more robust evaluation

## Tech Stack

- Python 3.12
- XGBoost
- pandas, NumPy, scikit-learn
- matplotlib, seaborn
- Google Colab (model training)
