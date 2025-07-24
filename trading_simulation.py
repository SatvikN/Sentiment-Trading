import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = 'merged_features.csv'
INITIAL_CASH = 10000
POSITION_SIZE = 1  # Number of shares per trade (for simplicity)

# --- LOAD DATA ---
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(['tickers', 'date'])

results = []
for ticker in df['tickers'].unique():
    tdf = df[df['tickers'] == ticker].reset_index(drop=True)
    cash = INITIAL_CASH
    position = 0
    portfolio_values = []
    actions = []
    for i, row in tdf.iterrows():
        price = row['Close']
        sentiment = row['avg_sentiment']
        # Simple rule: buy if sentiment > 0.2, sell if < -0.2, else hold
        if sentiment > 0.2 and cash >= price:
            position += POSITION_SIZE
            cash -= price * POSITION_SIZE
            action = 'buy'
        elif sentiment < -0.2 and position > 0:
            cash += price * position
            position = 0
            action = 'sell'
        else:
            action = 'hold'
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)
        actions.append(action)
    tdf['portfolio_value'] = portfolio_values
    tdf['action'] = actions
    results.append(tdf)

final_df = pd.concat(results)

# --- PLOT RESULTS ---
for ticker in final_df['tickers'].unique():
    tdf = final_df[final_df['tickers'] == ticker]
    plt.plot(tdf['date'], tdf['portfolio_value'], label=ticker)
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Rule-Based Trading Simulation')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- PRINT FINAL PORTFOLIO VALUES ---
print(final_df.groupby('tickers').tail(1)[['tickers', 'portfolio_value']]) 