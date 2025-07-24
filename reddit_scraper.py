import praw
import pymongo
import pandas as pd
from datetime import datetime, timedelta
import re
import config

# --- CONFIGURATION ---
REDDIT_CLIENT_ID = config.client_id
REDDIT_CLIENT_SECRET = config.client_secret
REDDIT_USER_AGENT = 'Sentiment Analysis v1.0'
MONGO_URI = 'mongodb://localhost:27017/'  # Or your MongoDB Atlas URI
SUBREDDITS = ['wallstreetbets', 'investing', 'stocks']
TICKERS = ['AAPL', 'TSLA', 'GME', 'AMC', 'NVDA', 'MSFT', 'SPY']
DB_NAME = 'reddit_sentiment'
COLLECTION_NAME = 'posts'

# --- SETUP ---
# Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# MongoDB
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- SCRAPING FUNCTION ---
def fetch_posts(subreddit_name, tickers, hours=24):
    subreddit = reddit.subreddit(subreddit_name)
    time_filter = 'day' if hours == 24 else 'all'
    posts = []
    for submission in subreddit.top(time_filter=time_filter, limit=1000):
        created_utc = datetime.utcfromtimestamp(submission.created_utc)
        if created_utc < datetime.utcnow() - timedelta(hours=hours):
            continue
        title = submission.title
        selftext = submission.selftext or ''
        text = f"{title} {selftext}".upper()
        mentioned = [ticker for ticker in tickers if re.search(rf'\b{ticker}\b', text)]
        if mentioned:
            post_data = {
                'id': submission.id,
                'subreddit': subreddit_name,
                'title': title,
                'selftext': selftext,
                'created_utc': created_utc,
                'tickers': mentioned,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'permalink': submission.permalink,
                'url': submission.url
            }
            posts.append(post_data)
    return posts

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    all_posts = []
    for sub in SUBREDDITS:
        posts = fetch_posts(sub, TICKERS, hours=24)
        if posts:
            collection.insert_many(posts)
            print(f"Inserted {len(posts)} posts from r/{sub}")
        all_posts.extend(posts)
    print(f"Total posts inserted: {len(all_posts)}") 