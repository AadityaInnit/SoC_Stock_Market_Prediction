# Stock Price Forecasting with LSTM

This project implements stock market forecasting with an LSTM model using Python, yfinance, pandas, NumPy, matplotlib, scikit-learn, and TensorFlow. The notebook prompts the user for a stock ticker, start date, end date, and data interval at runtime, then downloads historical market data with `yfinance.download()` using `start`, `end`, and `interval` parameters.

## Features

- Downloads historical stock data from Yahoo Finance.
- Uses user-defined runtime inputs for ticker, start date, end date, and interval.
- Visualizes adjusted close price over time.
- Computes technical indicators including MACD, RSI, SMA, and Bollinger Bands.
- Normalizes features and creates 60-step sliding windows for LSTM training.
- Trains an LSTM model with TensorFlow/Keras for time-series prediction.
- Predicts the next 15 time units and evaluates performance using RMSE, MAE, and R-squared.

## Technical Indicators

### MACD
The Moving Average Convergence Divergence indicator measures momentum by comparing short-term and long-term exponential moving averages. In this notebook, MACD is calculated as EMA(12) minus EMA(26), along with a 9-period signal line for trend confirmation.

### RSI
The Relative Strength Index measures the speed and magnitude of price changes on a 0 to 100 scale. Values above 70 are often interpreted as overbought conditions, while values below 30 are often interpreted as oversold conditions.

### Bollinger Bands
Bollinger Bands place upper and lower bands around a moving average using rolling standard deviation. They are useful for observing volatility expansion and contraction around the price series.

## Project Structure

- `stock_price_forecasting_lstm.ipynb` - Main notebook implementation.
- Exported CSV files - processed data, evaluation metrics, and forecast output.
- Plot outputs - generated from notebook cells and optionally saved as images.

## Installation

Install the required libraries:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow seaborn notebook
```

## How to Run

1. Open the notebook in Jupyter Notebook or JupyterLab.
2. Run the import and setup cells.
3. When prompted, enter:
   - Stock ticker, for example `AAPL`
   - Start date, for example `2022-01-01`
   - End date, for example `2026-06-30`
   - Interval, for example `1d`
4. Run the remaining cells in order.
5. Review the charts, metrics, and 15-step forecast output.

## Preprocessing and Modeling

The workflow preserves chronological order, scales features with MinMaxScaler, and converts the time series into sliding windows of 60 observations before model training. This follows standard TensorFlow guidance for time-series forecasting pipelines using windowed input data.[cite:9]

The model uses stacked LSTM layers with dropout regularization and early stopping. The target variable is the adjusted closing price, and the forecast is generated recursively for the next 15 periods.

## Evaluation

The notebook computes RMSE, MAE, and the coefficient of determination using the following formula:

\[
R^2 = 1 - rac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - ar{y})^2}
\]

Higher R-squared values generally indicate that the model explains more variance in the observed data.

## GitHub Submission

GitHub’s documented flow is to create a new repository from the **New repository** menu, assign a repository name and visibility, and then upload or push the notebook, README, and generated files.

### Web upload method

1. Sign in to GitHub.
2. Click the `+` button and choose **New repository**.
3. Enter the repository name and optional description, choose visibility, and create the repository.
4. Open the repository and choose **Upload files** to add the notebook, README, CSV outputs, and any saved visualizations.
5. Add a commit message and commit the changes.

### Git method

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/your-repository-name.git
git push -u origin main
```

## Notes

- Intraday intervals may have Yahoo Finance availability limits depending on date range and interval selection.[cite:17][cite:33]
- Forecast quality depends on data quantity, interval choice, market volatility, and model hyperparameters.
- The notebook is intended for educational and research use.
