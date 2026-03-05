# train_xgb.py
import os, joblib
import yfinance as yf
import pandas as pd, numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator as StochIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'return_1', 'return_3', 'log_ret', 'ema8', 'ema21', 'ema50',
    'rsi14', 'macd', 'macd_sig', 'atr14', 'vol_10', 'stoch_k', 'stoch_d'
]

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS",
    "MARUTI.NS", "SBIN.NS", "AXISBANK.NS", "BAJAJ_AUTO.NS", "SUNPHARMA.NS",
    "ADANIENT.NS", "HCLTECH.NS", "M&M.NS", "BAJAJFINSV.NS"
]

def compute_features(df):
    df = df.copy()
    close = pd.Series(df['Close'].values.ravel(), index=df.index).astype(float)
    high = pd.Series(df['High'].values.ravel(), index=df.index).astype(float)
    low = pd.Series(df['Low'].values.ravel(), index=df.index).astype(float)
    df['return_1'] = close.pct_change(1)
    df['return_3'] = close.pct_change(3)
    df['log_ret'] = np.log(close / close.shift(1))
    df['ema8'] = EMAIndicator(close, window=8).ema_indicator()
    df['ema21'] = EMAIndicator(close, window=21).ema_indicator()
    df['ema50'] = EMAIndicator(close, window=50).ema_indicator()
    df['rsi14'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['macd'] = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    df['atr14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['vol_10'] = df['log_ret'].rolling(10).std()
    stoch = StochIndicator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    return df

frames = []
for sym in SYMBOLS:
    # use larger period to ensure rolling windows work
    df = yf.download(sym, period="365d", interval="1d", progress=False)
    print(f"Fetched {sym} -> {df.shape}")
    if df.empty: continue
    feat_df = compute_features(df)
    feat_df['target'] = (feat_df['Close'].shift(-1) > feat_df['Close']).astype(int)
    feat_df = feat_df.dropna()
    frames.append(feat_df)

if len(frames) == 0:
    raise RuntimeError("No data fetched. Check network or symbols.")

data = pd.concat(frames, axis=0).reset_index(drop=True)
print("Combined data shape:", data.shape)

X = data[FEATURES].fillna(0)
y = data['target'].values

# Save medians to fill missing values at prediction time
medians = X.median().to_dict()

# fit scaler on training X (after fill with medians)
X_filled = X.fillna(pd.Series(medians))
scaler = StandardScaler().fit(X_filled.values)
X_scaled = scaler.transform(X_filled.values)

model = XGBClassifier(n_estimators=200, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)

joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES, 'medians': medians}, os.path.join(MODEL_DIR, "xgb_model_v2.pkl"))
print("Saved XGBoost model to models/xgb_model_v2.pkl")
