import pymongo
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
MONGO_URI = 'mongodb://localhost:27017/'  # Or your MongoDB Atlas URI
DB_NAME = 'reddit_sentiment'
COLLECTION_NAME = 'posts'
OUTPUT_CSV = 'sentiment_features.csv'

# --- MONGODB SETUP ---
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- LOAD DATA ---
posts = list(collection.find({'sentiment': {'$exists': True}}))
if not posts:
    print('No posts with sentiment found.')
    exit()

# --- PREPARE DATAFRAME ---
df = pd.DataFrame(posts)
# Flatten sentiment dict
sentiment_df = pd.json_normalize(df['sentiment'])
df = pd.concat([df, sentiment_df], axis=1)

# Convert created_utc to date and hour
if 'created_utc' in df:
    df['datetime'] = pd.to_datetime(df['created_utc'])
else:
    df['datetime'] = pd.to_datetime(df['created'])
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# Explode tickers (one row per ticker per post)
df = df.explode('tickers')

# --- AGGREGATE FEATURES (DAILY) ---
daily_grouped = df.groupby(['tickers', 'date']).agg(
    avg_sentiment=('compound', 'mean'),
    sentiment_volatility=('compound', 'std'),
    post_volume=('id', 'count')
).reset_index()

daily_grouped = daily_grouped.sort_values(['tickers', 'date'])
daily_grouped['sentiment_change'] = daily_grouped.groupby('tickers')['avg_sentiment'].diff()
# Rolling 3-day average and momentum
for ticker, group in daily_grouped.groupby('tickers'):
    idx = group.index
    daily_grouped.loc[idx, 'rolling_avg_sentiment'] = group['avg_sentiment'].rolling(window=3, min_periods=1).mean().values
    daily_grouped.loc[idx, 'momentum_3d'] = group['avg_sentiment'].diff(periods=3).values

daily_grouped.to_csv('sentiment_features_daily.csv', index=False)
print('Daily feature CSV saved to sentiment_features_daily.csv')

# --- AGGREGATE FEATURES (HOURLY) ---
hourly_grouped = df.groupby(['tickers', 'date', 'hour']).agg(
    avg_sentiment=('compound', 'mean'),
    sentiment_volatility=('compound', 'std'),
    post_volume=('id', 'count')
).reset_index()

hourly_grouped = hourly_grouped.sort_values(['tickers', 'date', 'hour'])
hourly_grouped['sentiment_change'] = hourly_grouped.groupby('tickers')['avg_sentiment'].diff()
# Rolling 3-hour average and momentum
for ticker, group in hourly_grouped.groupby('tickers'):
    idx = group.index
    hourly_grouped.loc[idx, 'rolling_avg_sentiment'] = group['avg_sentiment'].rolling(window=3, min_periods=1).mean().values
    hourly_grouped.loc[idx, 'momentum_3h'] = group['avg_sentiment'].diff(periods=3).values

hourly_grouped.to_csv('sentiment_features_hourly.csv', index=False)
print('Hourly feature CSV saved to sentiment_features_hourly.csv') 