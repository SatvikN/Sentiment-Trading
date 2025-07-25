import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

INPUT_CSV = 'merged_features.csv'

# --- LOAD DATA ---
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(['tickers', 'date'])

# --- CREATE LABEL: Next-day price movement (1=up, -1=down, 0=hold) ---
def get_label(prices):
    diff = prices.shift(-1) - prices
    return diff.apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))

df['label'] = df.groupby('tickers')['Close'].transform(get_label)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# --- FEATURES ---
features = ['avg_sentiment', 'sentiment_volatility', 'post_volume', 'sentiment_change']
X = df[features]
y = df['label']

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- RANDOM FOREST ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("Feature importances:", dict(zip(features, rf.feature_importances_)))

# --- XGBOOST ---
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print("Feature importances:", dict(zip(features, xgb.feature_importances_)))

# --- RULE-BASED STRATEGY ---
def rule_based_strategy(row):
    if row['avg_sentiment'] > 0.2:
        return 1
    elif row['avg_sentiment'] < -0.2:
        return -1
    else:
        return 0

df['rule_pred'] = df.apply(rule_based_strategy, axis=1)

# --- BUY-AND-HOLD BENCHMARK ---
def buy_and_hold_benchmark(df):
    # Always predict 'buy' (1) for the first test period, then hold
    preds = np.zeros(len(df))
    preds[0] = 1
    return preds.astype(int)

df['buyhold_pred'] = buy_and_hold_benchmark(df)

# --- OUTPUT PREDICTIONS FOR BACKTESTING ---
pred_df = df[['tickers', 'date', 'Close', 'label', 'rule_pred']].copy()
pred_df['rf_pred'] = np.nan
pred_df['xgb_pred'] = np.nan
pred_df.loc[X_test.index, 'rf_pred'] = y_pred_rf
pred_df.loc[X_test.index, 'xgb_pred'] = y_pred_xgb
pred_df['buyhold_pred'] = df['buyhold_pred']
pred_df.to_csv('ml_baseline_predictions.csv', index=False)
print('Predictions saved to ml_baseline_predictions.csv')

# --- ACCURACY REPORTS ---
print("\nRule-Based Strategy Accuracy:", accuracy_score(df.loc[X_test.index, 'label'], df.loc[X_test.index, 'rule_pred']))
print("Buy-and-Hold Benchmark Accuracy:", accuracy_score(df.loc[X_test.index, 'label'], df.loc[X_test.index, 'buyhold_pred']))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb)) 