import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title='Sentiment Trading Dashboard', layout='wide')

# --- CONFIGURATION ---
TICKERS = ['AAPL', 'TSLA', 'GME', 'AMC', 'NVDA', 'MSFT', 'SPY']
SENTIMENT_CSV = 'sentiment_features_daily.csv'
TRADE_LOG = 'trade_log.txt'
MERGED_CSV = 'merged_features_daily.csv'

# --- LOAD DATA ---
@st.cache_data
def load_sentiment():
    return pd.read_csv(SENTIMENT_CSV)

@st.cache_data
def load_trades():
    try:
        df = pd.read_csv(TRADE_LOG, names=['datetime', 'ticker', 'action', 'price', 'position'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception:
        return pd.DataFrame(columns=['datetime', 'ticker', 'action', 'price', 'position'])

@st.cache_data
def load_merged():
    return pd.read_csv(MERGED_CSV)

sentiment_df = load_sentiment()
trade_df = load_trades()
merged_df = load_merged()

# --- SIDEBAR ---
st.sidebar.title('Controls')
ticker = st.sidebar.selectbox('Select Ticker', TICKERS)
days = st.sidebar.slider('Days to Display', 7, 90, 30)
end_date = datetime.now().date()
start_date = end_date - timedelta(days=days)

# --- SENTIMENT VISUALIZATION ---
st.header(f'Sentiment & Price for {ticker}')
ticker_sent = sentiment_df[sentiment_df['tickers'] == ticker]
ticker_sent['date'] = pd.to_datetime(ticker_sent['date'])
ticker_sent = ticker_sent[(ticker_sent['date'] >= pd.to_datetime(start_date)) & (ticker_sent['date'] <= pd.to_datetime(end_date))]

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ticker_sent['date'], ticker_sent['avg_sentiment'], color='tab:blue', label='Avg Sentiment')
ax1.set_ylabel('Avg Sentiment', color='tab:blue')
ax2 = ax1.twinx()
if 'Close' in ticker_sent:
    ax2.plot(ticker_sent['date'], ticker_sent['Close'], color='tab:orange', label='Close Price')
    ax2.set_ylabel('Close Price', color='tab:orange')
fig.tight_layout()
st.pyplot(fig)

# --- RECENT TRADES ---
st.header('Recent Trades')
recent_trades = trade_df[trade_df['ticker'] == ticker].sort_values('datetime', ascending=False).head(20)
st.dataframe(recent_trades)

# --- PORTFOLIO STATUS ---
st.header('Portfolio Status')
if not trade_df.empty:
    latest_positions = trade_df.groupby('ticker').tail(1).set_index('ticker')['position']
    st.write('Current Positions:')
    st.write(latest_positions)
else:
    st.write('No trades yet.')

# --- PERFORMANCE METRICS ---
st.header('Performance Metrics')
if not trade_df.empty:
    # Simple P&L calculation
    pnl = 0
    for t in TICKERS:
        trades = trade_df[trade_df['ticker'] == t]
        if not trades.empty:
            last_pos = trades.iloc[-1]['position']
            last_price = float(trades.iloc[-1]['price'])
            pnl += last_pos * last_price
    st.metric('Estimated Portfolio Value', f'${pnl:,.2f}')
else:
    st.write('No performance data yet.')

# --- SENTIMENT VS. PRICE CORRELATION ---
st.header('Sentiment vs. Price Correlation')
ticker_merged = merged_df[merged_df['tickers'] == ticker]
if 'avg_sentiment' in ticker_merged and 'Close' in ticker_merged:
    corr = ticker_merged['avg_sentiment'].corr(ticker_merged['Close'])
    st.write(f'Correlation: {corr:.2f}')
    st.line_chart(ticker_merged[['avg_sentiment', 'Close']].set_index(ticker_merged['date']))
else:
    st.write('Not enough data for correlation.')

# --- ALERTS (Stub) ---
if not trade_df.empty and trade_df['action'].str.contains('sell').any():
    st.warning('Recent sell action detected!') 