import alpaca_trade_api as tradeapi
import pandas as pd
import time
import config

# --- CONFIGURATION ---
API_KEY = config.alpaca_key
API_SECRET = config.alpaca_secret
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
SIGNALS_CSV = 'ml_predictions.csv'  # Or backtest_results_ml.csv
ORDER_QTY = 1  # Number of shares per trade

# --- ALPACA API SETUP ---
api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')

# --- LOAD SIGNALS ---
signals = pd.read_csv(SIGNALS_CSV)
signals['date'] = pd.to_datetime(signals['date'])

# --- FILTER FOR TODAY'S SIGNALS ---
today = pd.Timestamp.now().normalize()
today_signals = signals[signals['date'] == today]

# --- PLACE ORDERS ---
for _, row in today_signals.iterrows():
    ticker = row['tickers']
    action = row.get('ml_action', 0)  # 1=buy, -1=sell, 0=hold
    if action == 1:
        side = 'buy'
    elif action == -1:
        side = 'sell'
    else:
        print(f"{ticker}: Hold signal, no order placed.")
        continue
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=ORDER_QTY,
            side=side,
            type='market',
            time_in_force='day'
        )
        print(f"Order placed: {side.upper()} {ORDER_QTY} {ticker}")
    except Exception as e:
        print(f"Error placing order for {ticker}: {e}")
    time.sleep(1)  # Avoid rate limits

print("Alpaca paper trading script complete.") 