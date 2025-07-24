import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = 'merged_features.csv'
INITIAL_CASH = 10000
POSITION_SIZE = 1  # Number of shares per trade
RISK_FREE_RATE = 0.0  # For Sharpe ratio

# --- STRATEGY FUNCTION TEMPLATE ---
def rule_based_strategy(row):
    """Simple rule: buy if avg_sentiment > 0.2, sell if < -0.2, else hold."""
    if row['avg_sentiment'] > 0.2:
        return 'buy'
    elif row['avg_sentiment'] < -0.2:
        return 'sell'
    else:
        return 'hold'

# --- BACKTEST ENGINE ---
def backtest(df, strategy_func):
    df = df.sort_values(['tickers', 'date']).reset_index(drop=True)
    results = []
    for ticker in df['tickers'].unique():
        tdf = df[df['tickers'] == ticker].reset_index(drop=True)
        cash = INITIAL_CASH
        position = 0
        portfolio_values = []
        actions = []
        trade_dates = []
        trade_prices = []
        wins = 0
        trades = 0
        last_buy_price = None
        for i, row in tdf.iterrows():
            price = row['Close']
            action = strategy_func(row)
            # Execute action
            if action == 'buy' and cash >= price:
                position += POSITION_SIZE
                cash -= price * POSITION_SIZE
                trade_dates.append(row['date'])
                trade_prices.append(price)
                last_buy_price = price
                trades += 1
            elif action == 'sell' and position > 0:
                cash += price * position
                if last_buy_price is not None and price > last_buy_price:
                    wins += 1
                position = 0
                trade_dates.append(row['date'])
                trade_prices.append(price)
                trades += 1
                last_buy_price = None
            portfolio_value = cash + position * price
            portfolio_values.append(portfolio_value)
            actions.append(action)
        tdf['portfolio_value'] = portfolio_values
        tdf['action'] = actions
        tdf['trade_marker'] = [d in trade_dates for d in tdf['date']]
        tdf['trade_price'] = [p if d in trade_dates else np.nan for d, p in zip(tdf['date'], tdf['Close'])]
        # Metrics
        tdf['returns'] = tdf['portfolio_value'].pct_change().fillna(0)
        total_return = (tdf['portfolio_value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH
        sharpe = (tdf['returns'].mean() - RISK_FREE_RATE) / (tdf['returns'].std() + 1e-9) * np.sqrt(252) if tdf['returns'].std() > 0 else 0
        running_max = tdf['portfolio_value'].cummax()
        drawdown = (tdf['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        win_rate = wins / trades if trades > 0 else 0
        print(f"\nTicker: {ticker}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Win Rate: {win_rate:.2%}")
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(tdf['date'], tdf['portfolio_value'], label='Equity Curve')
        plt.scatter([d for d in trade_dates], [p for p in trade_prices], marker='o', color='red', label='Trade')
        plt.title(f"Backtest: {ticker}")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        results.append(tdf)
    return pd.concat(results)

# --- MAIN ---
if __name__ == '__main__':
    df = pd.read_csv(INPUT_CSV)
    result_df = backtest(df, rule_based_strategy)
    # Save results if needed
    result_df.to_csv('backtest_results.csv', index=False) 