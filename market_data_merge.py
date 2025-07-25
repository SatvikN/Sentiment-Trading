import pandas as pd
import yfinance as yf
from datetime import datetime

SENTIMENT_DAILY_CSV = 'sentiment_features_daily.csv'
SENTIMENT_HOURLY_CSV = 'sentiment_features_hourly.csv'
OUTPUT_DAILY_CSV = 'merged_features_daily.csv'
OUTPUT_HOURLY_CSV = 'merged_features_hourly.csv'
TICKER_COL = 'tickers'
DATE_COL = 'date'
HOUR_COL = 'hour'

# --- USER SELECTION ---
USE_HOURLY = False  # Set to True for hourly, False for daily

sentiment_csv = SENTIMENT_HOURLY_CSV if USE_HOURLY else SENTIMENT_DAILY_CSV
output_csv = OUTPUT_HOURLY_CSV if USE_HOURLY else OUTPUT_DAILY_CSV

# --- LOAD SENTIMENT DATA ---
sentiment_df = pd.read_csv(sentiment_csv)
sentiment_df[DATE_COL] = pd.to_datetime(sentiment_df[DATE_COL])
if USE_HOURLY and HOUR_COL in sentiment_df:
    sentiment_df[HOUR_COL] = sentiment_df[HOUR_COL].astype(int)

# --- FETCH PRICE DATA ---
all_tickers = sentiment_df[TICKER_COL].unique().tolist()

price_data = []
for ticker in all_tickers:
    yf_ticker = yf.Ticker(ticker)
    start = sentiment_df[DATE_COL].min().strftime('%Y-%m-%d')
    end = (sentiment_df[DATE_COL].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    hist = yf_ticker.history(start=start, end=end, interval='1h' if USE_HOURLY else '1d')
    if hist.empty:
        continue
    hist = hist.reset_index()
    if USE_HOURLY:
        hist['date'] = hist['Datetime'].dt.date
        hist['hour'] = hist['Datetime'].dt.hour
        cols = ["date", "hour", "tickers", "Open", "High", "Low", "Close", "Volume"]
        hist['tickers'] = ticker
        price_data.append(hist[cols])
    else:
        hist['date'] = hist['Date'].dt.date
        hist['tickers'] = ticker
        cols = ["date", "tickers", "Open", "High", "Low", "Close", "Volume"]
        price_data.append(hist[cols])

if not price_data:
    print('No price data found for tickers.')
    exit()

price_df = pd.concat(price_data, ignore_index=True)
price_df['date'] = pd.to_datetime(price_df['date'])
if USE_HOURLY and HOUR_COL in price_df:
    price_df[HOUR_COL] = price_df[HOUR_COL].astype(int)

# --- MERGE DATASETS ---
if USE_HOURLY:
    merged = pd.merge(sentiment_df, price_df, how='left', on=['tickers', 'date', 'hour'])
    # Forward-fill missing price data within each ticker
    merged = merged.sort_values(['tickers', 'date', 'hour'])
    merged = merged.groupby('tickers').apply(lambda g: g.ffill()).reset_index(drop=True)
else:
    merged = pd.merge(sentiment_df, price_df, how='left', on=['tickers', 'date'])
    merged = merged.sort_values(['tickers', 'date'])
    merged = merged.groupby('tickers').apply(lambda g: g.ffill()).reset_index(drop=True)

# --- EXPORT ---
merged.to_csv(output_csv, index=False)
print(f'Merged dataset saved to {output_csv}') 