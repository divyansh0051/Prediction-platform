import os
import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BACKTEST ENGINE - Validates trading signals on historical data
# ============================================================================

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

LOOKBACK = 30
XGB_FEATURES = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ema_12', 'ema_26', 'atr', 'stoch_k', 'stoch_d']
LSTM_FEATURES = ['close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_12', 'ema_26', 'atr', 'stoch_k', 'stoch_d']

# Load models
print("Loading models...")
try:
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    with open('models/lstm_meta.pkl', 'rb') as f:
        lstm_meta = pickle.load(f)
    lstm_scaler = lstm_meta['scaler']
    LSTM_MEDIANS = lstm_meta.get('medians', {})
    print("✓ LSTM loaded")
except Exception as e:
    print(f"⚠ LSTM load error: {e}")
    lstm_model = None
    lstm_scaler = None
    LSTM_MEDIANS = {}

try:
    with open('models/xgb_model_v2.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/lstm_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    xgb_scaler = meta['xgb_scaler']
    XGB_MEDIANS = meta.get('xgb_medians', {})
    print("✓ XGB loaded")
except Exception as e:
    print(f"⚠ XGB load error: {e}")
    xgb_model = None
    xgb_scaler = None
    XGB_MEDIANS = {}

print("✓ Models loaded\n")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_features(df):
    """Compute technical indicators"""
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma + (std * 2)
    df['bb_lower'] = sma - (std * 2)
    
    # EMA
    df['ema_12'] = ema12
    df['ema_26'] = ema26
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = ranges.max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Stochastic
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    return df.fillna(0)

# ============================================================================
# BACKTESTING
# ============================================================================

class BacktestEngine:
    def __init__(self, symbol, start_date='2023-01-01', end_date='2024-12-31', initial_capital=100000):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        
        # State
        self.equity = initial_capital
        self.position = 0  # 0=none, 1=long
        self.entry_price = 0
        self.trades = []
        self.equity_history = [initial_capital]
        self.prices = []
        self.signals = []
        
    def fetch_data(self):
        """Fetch and prepare data"""
        print(f"Fetching {self.symbol} data from {self.start_date} to {self.end_date}...")
        try:
            ticker = yf.Ticker(f"{self.symbol}.NS")
            df = ticker.history(start=self.start_date, end=self.end_date)
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df = df.dropna()
            print(f"✓ Fetched {len(df)} candles\n")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def generate_signal(self, feat_df, last_row):
        """Generate buy/sell signal using ensemble"""
        try:
            prob_xgb = 0.5
            prob_lstm = 0.5
            
            # XGB signal
            if xgb_model is not None and xgb_scaler is not None:
                try:
                    x_vals = last_row.reindex(XGB_FEATURES)
                    x_vals = x_vals.fillna(pd.Series(XGB_MEDIANS)).astype(float).values.reshape(1, -1)
                    x_scaled = xgb_scaler.transform(x_vals)
                    prob_xgb = float(xgb_model.predict_proba(x_scaled)[0, 1])
                except Exception:
                    pass
            
            # LSTM signal
            if lstm_model is not None and lstm_scaler is not None:
                try:
                    seq_df = feat_df.reindex(columns=LSTM_FEATURES).fillna(pd.Series(LSTM_MEDIANS))
                    if len(seq_df) < LOOKBACK:
                        seq_vals = seq_df.values
                        pad = np.tile(seq_vals[0], (LOOKBACK - len(seq_df), 1))
                        seq_vals = np.vstack([pad, seq_vals])
                    else:
                        seq_vals = seq_df.iloc[-LOOKBACK:].values
                    
                    seq_scaled = lstm_scaler.transform(seq_vals)
                    prob_lstm = float(lstm_model.predict(seq_scaled.reshape(1, LOOKBACK, -1), verbose=0)[0, 0])
                except Exception:
                    pass
            
            # Ensemble
            prob = 0.6 * prob_lstm + 0.4 * prob_xgb
            signal = 1 if prob > 0.5 else 0  # 1=BUY, 0=SELL
            
            return signal, prob, prob_xgb, prob_lstm
        except Exception as e:
            return 0, 0.5, 0, 0
    
    def run(self, df):
        """Run backtest"""
        print("Running backtest...\n")
        df = compute_features(df)
        
        for idx in range(LOOKBACK, len(df)):
            date = df.iloc[idx]['date']
            price = df.iloc[idx]['close']
            
            feat_df = df.iloc[:idx+1][LSTM_FEATURES].copy()
            last_row = df.iloc[idx][LSTM_FEATURES]
            
            # Generate signal
            signal, prob, prob_xgb, prob_lstm = self.generate_signal(feat_df, last_row)
            self.signals.append({'date': date, 'price': price, 'signal': signal, 'prob': prob})
            self.prices.append(price)
            
            # Trading logic
            if signal == 1 and self.position == 0:  # BUY
                self.position = 1
                self.entry_price = price
                self.trades.append({
                    'type': 'BUY',
                    'date': date,
                    'price': price,
                    'prob': prob
                })
            
            elif signal == 0 and self.position == 1:  # SELL
                pnl = (price - self.entry_price) * 100 / self.entry_price  # percentage
                self.equity = self.equity * (1 + pnl / 100)
                self.equity_history.append(self.equity)
                
                self.trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': price,
                    'pnl': pnl,
                    'equity': self.equity
                })
                self.position = 0
        
        # Close any open position
        if self.position == 1:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - self.entry_price) * 100 / self.entry_price
            self.equity = self.equity * (1 + pnl / 100)
            self.trades.append({
                'type': 'SELL (CLOSE)',
                'date': df.iloc[-1]['date'],
                'price': final_price,
                'pnl': pnl,
                'equity': self.equity
            })
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        equity_arr = np.array(self.equity_history)
        prices_arr = np.array(self.prices)
        
        # Total return
        total_return = (self.equity - self.initial_capital) / self.initial_capital * 100
        
        # Buy & hold return
        buy_hold_return = (prices_arr[-1] - prices_arr[0]) / prices_arr[0] * 100
        
        # Win rate
        closed_trades = [t for t in self.trades if 'pnl' in t]
        if len(closed_trades) > 0:
            wins = len([t for t in closed_trades if t['pnl'] > 0])
            win_rate = wins / len(closed_trades) * 100
        else:
            win_rate = 0
        
        # Sharpe ratio
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Avg trade PnL
        if len(closed_trades) > 0:
            avg_pnl = np.mean([t['pnl'] for t in closed_trades])
        else:
            avg_pnl = 0
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(closed_trades),
            'avg_pnl': avg_pnl,
            'final_equity': self.equity
        }
    
    def print_report(self):
        """Print backtest report"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print(f"BACKTEST REPORT: {self.symbol}")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Final Equity: ${metrics['final_equity']:,.0f}")
        print()
        print(f"Strategy Return: {metrics['total_return']:+.2f}%")
        print(f"Buy & Hold Return: {metrics['buy_hold_return']:+.2f}%")
        print(f"Outperformance: {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
        print()
        print(f"Total Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Avg Trade PnL: {metrics['avg_pnl']:+.2f}%")
        print()
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print("="*60 + "\n")
        
        return metrics

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run backtest for multiple stocks
    symbols = ['RELIANCE', 'INFY', 'TCS', 'HDFCBANK']
    
    results = {}
    for symbol in symbols:
        try:
            bt = BacktestEngine(symbol, start_date='2023-06-01', end_date='2024-12-31')
            df = bt.fetch_data()
            if df is not None:
                bt.run(df)
                metrics = bt.calculate_metrics()
                bt.print_report()
                results[symbol] = metrics
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}\n")
    
    # Summary
    if results:
        print("\nBACKTEST SUMMARY (All Stocks)")
        print("="*60)
        summary_df = pd.DataFrame(results).T
        summary_df = summary_df[['total_return', 'buy_hold_return', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'num_trades']]
        print(summary_df.to_string())
        print("="*60)
