import pandas as pd
import yfinance as yf
from datetime import datetime

SENTIMENT_CSV = 'sentiment_features.csv'
OUTPUT_CSV = 'merged_features.csv'
TICKER_COL = 'tickers'
DATE_COL = 'date'

# --- LOAD SENTIMENT DATA ---
sentiment_df = pd.read_csv(SENTIMENT_CSV)
sentiment_df[DATE_COL] = pd.to_datetime(sentiment_df[DATE_COL])

# --- FETCH PRICE DATA ---
all_tickers = sentiment_df[TICKER_COL].unique().tolist()
all_dates = sentiment_df[DATE_COL].unique()

# Download price data for all tickers
price_data = []
for ticker in all_tickers:
    yf_ticker = yf.Ticker(ticker)
    # Download daily OHLCV for the date range in sentiment_df
    start = sentiment_df[DATE_COL].min().strftime('%Y-%m-%d')
    end = (sentiment_df[DATE_COL].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    hist = yf_ticker.history(start=start, end=end)
    if hist.empty:
        continue
    hist = hist.reset_index()
    hist['date'] = hist['Date'].dt.date
    hist['tickers'] = ticker
    price_data.append(hist[["date", "tickers", "Open", "High", "Low", "Close", "Volume"]])

if not price_data:
    print('No price data found for tickers.')
    exit()

price_df = pd.concat(price_data, ignore_index=True)
price_df['date'] = pd.to_datetime(price_df['date'])

# --- MERGE DATASETS ---
merged = pd.merge(sentiment_df, price_df, how='inner', left_on=['tickers', 'date'], right_on=['tickers', 'date'])

# --- EXPORT ---
merged.to_csv(OUTPUT_CSV, index=False)
print(f'Merged dataset saved to {OUTPUT_CSV}') 