import pymongo
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from bson.objectid import ObjectId

# --- CONFIGURATION ---
MONGO_URI = 'mongodb://localhost:27017/'  # Or your MongoDB Atlas URI
DB_NAME = 'reddit_sentiment'
COLLECTION_NAME = 'posts'

# --- NLTK SETUP ---
nltk.download('vader_lexicon')

# --- TEXT PREPROCESSING ---
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()
    return text

# --- MONGODB SETUP ---
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- VADER SETUP ---
sia = SentimentIntensityAnalyzer()

# --- PROCESS POSTS ---
def analyze_and_update_sentiment():
    posts = list(collection.find({'sentiment': {'$exists': False}}))
    print(f"Found {len(posts)} posts to analyze.")
    for post in posts:
        text = f"{post.get('title', '')} {post.get('selftext', '')}"
        cleaned = clean_text(text)
        sentiment = sia.polarity_scores(cleaned)
        # Add sentiment to post
        collection.update_one({'_id': post['_id']}, {'$set': {'sentiment': sentiment}})
    print("Sentiment analysis complete.")

if __name__ == '__main__':
    analyze_and_update_sentiment() 