# train_lstm.py
import os, joblib
import yfinance as yf
import pandas as pd, numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator as StochIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from fundamentals import fetch_fundamentals
from features_monthly import monthly_mood

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_FEATURES = [
    'return_1', 'return_3', 'log_ret', 'ema8', 'ema21', 'ema50',
    'rsi14', 'macd', 'macd_sig', 'atr14', 'vol_10', 'stoch_k', 'stoch_d'
]

# optional additional features
FUND_FEATURES = ['trailingPE', 'marketCap', 'profitMargins', 'totalDebt']
MONTHLY_FEATURES = ['monthly_return', 'monthly_vol', 'momentum', 'mood_score']

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS"
]

LOOKBACK = 30   # 30-candle sequences (balanced)
EPOCHS = 30
BATCH = 64

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

def create_sequences(X, y, steps=LOOKBACK):
    xs, ys = [], []
    for i in range(len(X) - steps):
        xs.append(X[i:i+steps])
        ys.append(y[i+steps])
    return np.array(xs), np.array(ys)



def attach_monthly(df):
    try:
        s = df['Close'].astype(float)
        mood = monthly_mood(pd.DataFrame({'Close': s}, index=df.index))
        # forward-fill monthly mood values to the daily index
        mood = mood.reindex(df.index, method='ffill')
        for c in mood.columns:
            df[c] = mood[c].values
    except Exception:
        for c in MONTHLY_FEATURES:
            df[c] = np.nan
    return df


def attach_fundamentals(df, yf_sym):
    try:
        info = fetch_fundamentals(yf_sym)
        for k in FUND_FEATURES:
            df[k] = info.get(k)
    except Exception:
        for k in FUND_FEATURES:
            df[k] = np.nan
    return df


def build_dataset(symbols, include_fundamentals=False, include_monthly=False):
    frames = []
    for sym in symbols:
        df = yf.download(sym, period="365d", interval="1d", progress=False)
        print("Fetch", sym, df.shape)
        if df.empty:
            continue

        if include_monthly:
            df = attach_monthly(df)

        if include_fundamentals:
            df = attach_fundamentals(df, sym)

        feat = compute_features(df).dropna()
        feat['target'] = (feat['Close'].shift(-1) > feat['Close']).astype(int)
        feat = feat.dropna()
        frames.append(feat)

    if len(frames) == 0:
        raise RuntimeError("No data fetched for LSTM. Check network or symbols.")

    data = pd.concat(frames, axis=0).reset_index(drop=True)
    print("LSTM combined:", data.shape)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-fundamentals', action='store_true')
    parser.add_argument('--include-monthly', action='store_true')
    args = parser.parse_args()

    data = build_dataset(SYMBOLS, include_fundamentals=args.include_fundamentals, include_monthly=args.include_monthly)

    # build final feature list based on flags
    FEATURES = list(BASE_FEATURES)
    if args.include_fundamentals:
        FEATURES += FUND_FEATURES
    if args.include_monthly:
        FEATURES += MONTHLY_FEATURES

# fill missing features with medians (computed per-column)
feature_df = data[FEATURES].copy()
medians = feature_df.median().to_dict()
feature_df = feature_df.fillna(pd.Series(medians))

scaler = StandardScaler().fit(feature_df.values)
X_all = scaler.transform(feature_df.values)
y_all = data['target'].values

# reshape back to dataframe-like for sequence building
X_df = pd.DataFrame(X_all, columns=FEATURES)

X_seq, y_seq = create_sequences(X_df.values, y_all, steps=LOOKBACK)
print("X_seq shape:", X_seq.shape)

# split train/test (time-series safe split)
split = int(0.8 * len(X_seq))
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

# build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mc = ModelCheckpoint(os.path.join(MODEL_DIR, 'lstm_model.h5'), save_best_only=True, monitor='val_loss')

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH, callbacks=[es, mc])

# save scaler & medians
joblib.dump({'scaler': scaler, 'features': FEATURES, 'medians': medians, 'lookback': LOOKBACK}, os.path.join(MODEL_DIR, 'lstm_meta.pkl'))
print("Saved LSTM model and metadata.")
