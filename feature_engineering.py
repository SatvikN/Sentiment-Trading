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

# Convert created_utc to date
if 'created_utc' in df:
    df['date'] = pd.to_datetime(df['created_utc']).dt.date
else:
    df['date'] = pd.to_datetime(df['created']).dt.date

# Explode tickers (one row per ticker per post)
df = df.explode('tickers')

# --- AGGREGATE FEATURES ---
grouped = df.groupby(['tickers', 'date']).agg(
    avg_sentiment=('compound', 'mean'),
    sentiment_volatility=('compound', 'std'),
    post_volume=('id', 'count')
).reset_index()

# --- SENTIMENT CHANGE ---
grouped = grouped.sort_values(['tickers', 'date'])
grouped['sentiment_change'] = grouped.groupby('tickers')['avg_sentiment'].diff()

# --- EXPORT TO CSV ---
grouped.to_csv(OUTPUT_CSV, index=False)
print(f'Feature CSV saved to {OUTPUT_CSV}') 