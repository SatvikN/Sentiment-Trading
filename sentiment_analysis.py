import pymongo
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from bson.objectid import ObjectId
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords

# --- CONFIGURATION ---
MONGO_URI = 'mongodb://localhost:27017/'  # Or your MongoDB Atlas URI
DB_NAME = 'reddit_sentiment'
COLLECTION_NAME = 'posts'

# --- NLTK SETUP ---
nltk.download('vader_lexicon')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# --- TEXT PREPROCESSING ---
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

# --- FINBERT SETUP ---
finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# --- FINBERT SCORING ---
def finbert_score(text):
    inputs = finbert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
    return {
        'finbert_positive': scores[2],
        'finbert_neutral': scores[1],
        'finbert_negative': scores[0]
    }

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
        vader_sentiment = sia.polarity_scores(cleaned)
        finbert_sentiment = finbert_score(cleaned)
        # Add both sentiment scores to post
        collection.update_one({'_id': post['_id']}, {'$set': {'sentiment': vader_sentiment, 'finbert_sentiment': finbert_sentiment}})
    print("Sentiment analysis complete.")

if __name__ == '__main__':
    analyze_and_update_sentiment() 