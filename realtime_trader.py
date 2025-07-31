import alpaca_trade_api as tradeapi
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
ALPACA_API_KEY = 'YOUR_ALPACA_API_KEY'
ALPACA_SECRET_KEY = 'YOUR_ALPACA_SECRET_KEY'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
TICKERS = ['AAPL', 'TSLA', 'GME', 'AMC', 'NVDA', 'MSFT', 'SPY']
MAX_POSITION_SIZE = 10  # shares
STOP_LOSS_PCT = 0.05

# --- ALPACA SETUP ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# --- TRADING LOGIC ---
def get_signal(ticker):
    # Stub: randomly choose action (replace with real-time model output)
    return random.choice(['buy', 'sell', 'hold'])

positions = {ticker: 0 for ticker in TICKERS}
buy_prices = {ticker: None for ticker in TICKERS}

while True:
    for ticker in TICKERS:
        signal = get_signal(ticker)
        last_price = float(api.get_last_trade(ticker).price)
        position = positions[ticker]
        # --- RISK MANAGEMENT: STOP LOSS ---
        if position > 0 and buy_prices[ticker] is not None:
            if last_price < buy_prices[ticker] * (1 - STOP_LOSS_PCT):
                print(f"[STOP LOSS] Selling {position} shares of {ticker} at {last_price}")
                api.submit_order(symbol=ticker, qty=position, side='sell', type='market', time_in_force='gtc')
                positions[ticker] = 0
                buy_prices[ticker] = None
                continue
        # --- SIGNAL EXECUTION ---
        if signal == 'buy' and position < MAX_POSITION_SIZE:
            qty = MAX_POSITION_SIZE - position
            print(f"[TRADE] Buying {qty} shares of {ticker} at {last_price}")
            api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='gtc')
            positions[ticker] += qty
            buy_prices[ticker] = last_price
        elif signal == 'sell' and position > 0:
            print(f"[TRADE] Selling {position} shares of {ticker} at {last_price}")
            api.submit_order(symbol=ticker, qty=position, side='sell', type='market', time_in_force='gtc')
            positions[ticker] = 0
            buy_prices[ticker] = None
        else:
            print(f"[HOLD] {ticker} at {last_price}")
        # Log action
        with open('trade_log.txt', 'a') as f:
            f.write(f"{datetime.utcnow()}, {ticker}, {signal}, {last_price}, {positions[ticker]}\n")
    time.sleep(60)
