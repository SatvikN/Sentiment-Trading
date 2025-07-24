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