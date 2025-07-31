import praw
import yfinance as yf
import time
from datetime import datetime
from sentiment_analysis import clean_text, sia, finbert_score

# --- CONFIGURATION ---
REDDIT_CLIENT_ID = 'YOUR_CLIENT_ID'
REDDIT_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
REDDIT_USER_AGENT = 'Sentiment Analysis v1.0'
SUBREDDITS = ['wallstreetbets', 'investing', 'stocks']
TICKERS = ['AAPL', 'TSLA', 'GME', 'AMC', 'NVDA', 'MSFT', 'SPY']

# --- REDDIT SETUP ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# --- IN-MEMORY BUFFER ---
sentiment_buffer = {ticker: [] for ticker in TICKERS}

# --- STREAM REDDIT POSTS ---
def stream_reddit():
    for submission in reddit.subreddit('+'.join(SUBREDDITS)).stream.submissions(skip_existing=True):
        text = f"{submission.title} {submission.selftext}".upper()
        mentioned = [ticker for ticker in TICKERS if f" {ticker} " in f" {text} "]
        if mentioned:
            cleaned = clean_text(text)
            vader_sent = sia.polarity_scores(cleaned)
            finbert_sent = finbert_score(cleaned)
            for ticker in mentioned:
                sentiment_buffer[ticker].append({
                    'timestamp': datetime.utcnow(),
                    'vader': vader_sent,
                    'finbert': finbert_sent,
                    'text': submission.title
                })
                print(f"[Reddit] {ticker} | VADER: {vader_sent['compound']:.3f} | FinBERT: {finbert_sent} | {submission.title}")

# --- FETCH LIVE MARKET DATA ---
def fetch_live_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1d', interval='1m')
        if not data.empty:
            price = data['Close'].iloc[-1]
            return price
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
    return None

# --- SYNCHRONIZE AND PRINT ---
def sync_and_print():
    while True:
        for ticker in TICKERS:
            if sentiment_buffer[ticker]:
                latest_sent = sentiment_buffer[ticker][-1]
                price = fetch_live_price(ticker)
                print(f"[Sync] {ticker} | Price: {price} | Sentiment: {latest_sent['vader']['compound']:.3f} | Time: {latest_sent['timestamp']}")
        time.sleep(60)

if __name__ == '__main__':
    import threading
    reddit_thread = threading.Thread(target=stream_reddit, daemon=True)
    reddit_thread.start()
    sync_and_print()
