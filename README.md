# Sentiment Trading Reddit Scraper (MVP)

## Overview
This script collects Reddit posts from r/wallstreetbets, r/investing, and r/stocks from the last 24 hours, filters for posts mentioning selected stock tickers, and stores them in MongoDB.

## Setup

### 1. Reddit API Credentials
- Go to https://www.reddit.com/prefs/apps
- Click "create another app"
- Set type to "script"
- Note your `client_id`, `client_secret`, and set a `user_agent` (e.g., 'sentiment-trader/0.1 by YOUR_USERNAME')
- Fill these in the placeholders in `reddit_scraper.py`

### 2. MongoDB
- Install MongoDB locally or use [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- Update the `MONGO_URI` in `reddit_scraper.py` if needed

### 3. Install Requirements
```
pip install -r requirements.txt
```

### 4. Run the Scraper
```
python reddit_scraper.py
```

## Customization
- Edit `SUBREDDITS` and `TICKERS` in `reddit_scraper.py` to change what you collect.

## Next Steps
- Expand to include comments
- Add sentiment analysis pipeline 