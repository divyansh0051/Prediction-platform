"""Train HMM regime detection models per ticker.
Saves models to models/hmm_<TICKER>.pkl
"""
import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM


def map_symbol(tv_symbol):
    s = str(tv_symbol).strip().upper()
    if ':' in s:
        s = s.split(':',1)[1]
    s = s.replace('/', '.')
    if not s.endswith('.NS'):
        s = s + '.NS'
    return s


def compute_simple_features(df):
    df = df.copy()
    df['close'] = df['Close'].astype(float)
    df['ret1'] = df['close'].pct_change(1)
    df['ret5'] = df['close'].pct_change(5)
    df['vol20'] = df['ret1'].rolling(20).std()
    df = df.dropna()
    return df[['ret1','ret5','vol20']]


def train_hmm_for_ticker(ticker, n_components=3):
    yf_sym = map_symbol(ticker)
    print(f"Fetching {yf_sym} for HMM training...")
    df = yf.download(yf_sym, period='720d', interval='1d', progress=False)
    if df.empty:
        print(f"No data for {yf_sym}, skipping")
        return

    feats = compute_simple_features(df)
    X = feats.values

    model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter=100)
    model.fit(X)

    outdir = 'models'
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'hmm_{ticker}.pkl')
    joblib.dump({'model': model, 'features': list(feats.columns)}, path)
    print(f"Saved HMM to {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train HMM models for tickers')
    parser.add_argument('--ticker', '-t', help='Single ticker to train (overrides tickers.txt)')
    parser.add_argument('--n-components', '-n', type=int, default=3,
                        help='Number of HMM components')
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        if not os.path.exists('tickers.txt'):
            print('Please create tickers.txt with one ticker per line')
            exit(1)
        with open('tickers.txt') as f:
            tickers = [l.strip() for l in f if l.strip()]

    for t in tickers:
        try:
            train_hmm_for_ticker(t, n_components=args.n_components)
        except Exception as e:
            print(f"Error training {t}: {e}")
