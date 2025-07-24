import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

INPUT_CSV = 'merged_features.csv'
ML_PREDICTIONS_CSV = 'ml_predictions.csv'
INITIAL_CASH = 10000
POSITION_SIZE = 1  # Number of shares per trade
RISK_FREE_RATE = 0.0  # For Sharpe ratio

# --- STRATEGY FUNCTION TEMPLATE ---
def rule_based_strategy(row):
    if row['avg_sentiment'] > 0.2:
        return 'buy'
    elif row['avg_sentiment'] < -0.2:
        return 'sell'
    else:
        return 'hold'

def ml_strategy(row):
    # Use ML model predictions as actions (expects 'ml_action' column)
    if 'ml_action' in row and not pd.isnull(row['ml_action']):
        if row['ml_action'] == 1:
            return 'buy'
        elif row['ml_action'] == -1:
            return 'sell'
        else:
            return 'hold'
    else:
        return 'hold'

# --- BACKTEST ENGINE ---
def backtest(df, strategy_func, label='Strategy'):
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
        tdf[f'portfolio_value_{label}'] = portfolio_values
        tdf[f'action_{label}'] = actions
        tdf[f'trade_marker_{label}'] = [d in trade_dates for d in tdf['date']]
        tdf[f'trade_price_{label}'] = [p if d in trade_dates else np.nan for d, p in zip(tdf['date'], tdf['Close'])]
        tdf['returns'] = pd.Series(portfolio_values).pct_change().fillna(0)
        total_return = (portfolio_values[-1] - INITIAL_CASH) / INITIAL_CASH
        sharpe = (tdf['returns'].mean() - RISK_FREE_RATE) / (tdf['returns'].std() + 1e-9) * np.sqrt(252) if tdf['returns'].std() > 0 else 0
        running_max = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - running_max) / running_max
        max_drawdown = drawdown.min()
        win_rate = wins / trades if trades > 0 else 0
        print(f"\n[{label}] Ticker: {ticker}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Win Rate: {win_rate:.2%}")
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(tdf['date'], tdf[f'portfolio_value_{label}'], label=f'Equity Curve ({label})')
        plt.scatter([d for d in trade_dates], [p for p in trade_prices], marker='o', color='red', label='Trade')
        plt.title(f"Backtest: {ticker} ({label})")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        results.append(tdf)
    return pd.concat(results)

# --- ML PREDICTION GENERATION ---
def generate_ml_predictions(df):
    from sklearn.ensemble import RandomForestClassifier
    features = ['avg_sentiment', 'sentiment_volatility', 'post_volume', 'sentiment_change']
    # Use only rows with non-null label
    df = df.copy()
    df['label'] = df.groupby('tickers')['Close'].transform(lambda x: (x.shift(-1) - x).apply(lambda y: 1 if y > 0.5 else (-1 if y < -0.5 else 0)))
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    X = df[features]
    y = df['label']
    # Train/test split (time-based)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X)
    df['ml_action'] = y_pred
    df.to_csv(ML_PREDICTIONS_CSV, index=False)
    print(f"ML predictions saved to {ML_PREDICTIONS_CSV}")
    return df

# --- MAIN ---
if __name__ == '__main__':
    df = pd.read_csv(INPUT_CSV)
    # Rule-based backtest
    print("\n=== Rule-Based Strategy ===")
    rule_results = backtest(df, rule_based_strategy, label='Rule')
    # ML-based backtest
    print("\n=== ML-Based Strategy ===")
    ml_df = generate_ml_predictions(df)
    ml_results = backtest(ml_df, ml_strategy, label='ML')
    # Save results
    rule_results.to_csv('backtest_results_rule.csv', index=False)
    ml_results.to_csv('backtest_results_ml.csv', index=False) 