# Stock Price Forecasting with LSTM

This project implements stock market forecasting with an LSTM model using Python, yfinance, pandas, NumPy, matplotlib, scikit-learn, and TensorFlow. The notebook prompts the user for a stock ticker, start date, end date, and data interval at runtime, then downloads historical market data with `yfinance.download()` using `start`, `end`, and `interval` parameters.

## Features

- Downloads historical stock data from Yahoo Finance.
- Uses user-defined runtime inputs for ticker, start date, end date, and interval.
- Visualizes adjusted close price over time.
- Computes technical indicators including MACD, RSI, SMA, and Bollinger Bands.
- Standardizes features and creates 30-step sliding windows for LSTM training.
- Trains an LSTM model with TensorFlow/Keras to predict the next-step **Return**, then reconstructs price from it.
- Predicts the next 15 time units and evaluates performance using RMSE, MAE, and R-squared (train and test).

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
   - Start date, for example `2018-01-01` (see "Data Quantity" below for why this should be several years back, not just one)
   - End date, for example `2026-06-30`
   - Interval, for example `1d`
4. Run the remaining cells in order.
5. Review the charts, metrics, and 15-step forecast output.

## Preprocessing and Modeling

The workflow preserves chronological order, standardizes features with `StandardScaler` (fit on the training split only), and converts the time series into sliding windows of 30 observations before model training. This follows standard TensorFlow guidance for time-series forecasting pipelines using windowed input data.[cite:9]

The model uses a single LSTM(32) layer with dropout, `EarlyStopping`, and `ReduceLROnPlateau`. **The target variable is the next-step Return (percentage change in Adj_Close), not the raw price level** -- absolute prices are reconstructed afterwards as `previous_close * (1 + predicted_return)`. The forecast is generated recursively for the next 15 periods by compounding predicted returns onto the last known price.

### Why predict returns instead of price?

An earlier version of this notebook predicted the price level directly using a scaler fit on the training data. That approach has a structural flaw: once the stock price moves beyond the min/max range seen during training (which happens quickly for any trending stock), the model has to extrapolate into scaled values it never saw during training, both as an input and as a target. In practice this produced a large train/test gap and a **negative test R^2** (worse than just predicting the mean). Predicting returns sidesteps this because returns are roughly stationary -- their scale doesn't depend on the absolute price level, so the model only needs to learn repeatable, bounded dynamics instead of extrapolating an unbounded price trend.

### Data quantity

A 30-step window needs enough history to produce a reasonable number of training sequences. As a rule of thumb, the notebook warns if fewer than `window_size * 6` rows are available after indicator warm-up. For daily data this means **at least 2-3 years of history**, and ideally more (5+ years) so the model sees multiple market regimes. A single year of daily data (~250 rows) leaves only ~100-150 training sequences, which is not enough to reliably train even a small LSTM -- this was a major contributor to the original poor results (Train RMSE 8.5 vs. Test RMSE 15.5).

### Model size vs. data size

The original model used two stacked LSTM(64) layers (~50k+ parameters) trained on as few as ~124 sequences -- badly overparameterized for the data available, which shows up as a large train/test performance gap (overfitting). The current model uses a single LSTM(32) layer sized to match the amount of data this workflow typically has available. If you supply a much larger date range, increasing the LSTM units or adding a second layer back is reasonable.

## Evaluation

The notebook computes RMSE, MAE, and the coefficient of determination (reported for both train and test sets) using the following formula:

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

Higher R-squared values generally indicate that the model explains more variance in the observed data. A large gap between train and test R^2 indicates overfitting; consider more history, fewer LSTM units, or more dropout.

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
