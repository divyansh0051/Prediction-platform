# app.py (updated)
import os
import time
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from fundamentals import fetch_fundamentals
from features_monthly import monthly_mood
from news_feed import fetch_news, fetch_news_and_tweets
from backtest_engine import backtest_on_data, get_model_performance_stats
import warnings
warnings.filterwarnings('ignore')

# DQN agent cache
DQN_AGENT = None
DQN_AGENT_LOADED = False


def get_dqn_agent():
    global DQN_AGENT, DQN_AGENT_LOADED
    if DQN_AGENT_LOADED and DQN_AGENT is not None:
        return DQN_AGENT

    weights_path = os.path.join('models', 'dqn_weights.weights.h5')
    if not os.path.exists(weights_path):
        print(f"DQN weights file not found at: {weights_path}")
        DQN_AGENT_LOADED = True
        DQN_AGENT = None
        return None

    try:
        import tensorflow as tf
        # enable memory growth on GPUs if present
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten
        from tensorflow.keras.optimizers import Adam

        # build model same as training
        model = Sequential()
        model.add(Flatten(input_shape=(1, 10)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        model.load_weights(weights_path)

        DQN_AGENT = model
        DQN_AGENT_LOADED = True
        print(f"DQN agent loaded successfully from {weights_path}")
        return DQN_AGENT
    except Exception as e:
        DQN_AGENT = None
        DQN_AGENT_LOADED = True
        print(f"DQN load error: {type(e).__name__}: {e}")
        return None

# file paths
XGB_PATH = 'models/xgb_model_v2.pkl'
LSTM_META = 'models/lstm_meta.pkl'
LSTM_MODEL = 'models/lstm_model.h5'

if not os.path.exists(XGB_PATH):
    raise FileNotFoundError("Train XGBoost first: python train_xgb.py")
if not os.path.exists(LSTM_META) or not os.path.exists(LSTM_MODEL):
    raise FileNotFoundError("Train LSTM first: python train_lstm.py")

obj = joblib.load(XGB_PATH)
xgb_model = obj['model']
xgb_scaler = obj['scaler']
XGB_FEATURES = obj['features']
XGB_MEDIANS = obj.get('medians', {})

lstm_meta = joblib.load(LSTM_META)
lstm_scaler = lstm_meta['scaler']
LSTM_FEATURES = lstm_meta['features']
LSTM_MEDIANS = lstm_meta['medians']
LOOKBACK = lstm_meta['lookback']

lstm_model = load_model(LSTM_MODEL)

# Eager load DQN model to ensure it's available
_ = get_dqn_agent()

app = Flask(__name__)
CORS(app)

PRICE_CACHE = {}
CACHE_TTL = 300
PRICE_MAX = 2000

def map_symbol(tv_symbol):
    s = str(tv_symbol).strip().upper()
    if ':' in s:
        s = s.split(':',1)[1]
    s = s.replace('/', '.')
    if not s.endswith('.NS'):
        s = s + '.NS'
    return s


def compute_simple_features(df):
    """Return feature dataframe used by HMM training (ret1, ret5, vol20)."""
    df = df.copy()
    df['close'] = df['Close'].astype(float)
    df['ret1'] = df['close'].pct_change(1)
    df['ret5'] = df['close'].pct_change(5)
    df['vol20'] = df['ret1'].rolling(20).std()
    df = df.dropna()
    return df[['ret1', 'ret5', 'vol20']]

def generate_mock_data(base_price=2500, points=100):
    """Generate realistic mock OHLCV data"""
    rows = []
    current_price = base_price
    now = int(time.time())
    
    for i in range(points, 0, -1):
        change = np.random.randn() * 50
        current_price += change
        
        open_p = current_price - np.random.uniform(0, 20)
        close_p = current_price + np.random.uniform(-20, 20)
        high_p = max(open_p, close_p) + np.random.uniform(0, 15)
        low_p = min(open_p, close_p) - np.random.uniform(0, 15)
        
        rows.append({
            'time': now - (i * 900),  # 15 min intervals
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': int(np.random.uniform(100000, 500000))
        })
    
    return rows

def compute_features_df(df):
    try:
        df = df.copy()
        
        # Normalize column names
        df.columns = df.columns.str.capitalize()
        
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        
        df['return_1'] = close.pct_change(1)
        df['return_3'] = close.pct_change(3)
        df['log_ret'] = np.log(close / close.shift(1))
        
        from ta.trend import EMAIndicator, MACD
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import AverageTrueRange
        
        df['ema8'] = EMAIndicator(close, window=8).ema_indicator()
        df['ema21'] = EMAIndicator(close, window=21).ema_indicator()
        df['ema50'] = EMAIndicator(close, window=50).ema_indicator()
        df['rsi14'] = RSIIndicator(close, window=14).rsi()
        
        macd = MACD(close)
        df['macd'] = macd.macd()
        df['macd_sig'] = macd.macd_signal()
        
        df['atr14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
        df['vol_10'] = df['log_ret'].rolling(10).std()
        
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Feature computation error: {e}")
        return None

def fetch_price(yf_sym, period='180d', interval='1d'):
    try:
        key = f"{yf_sym}|{period}|{interval}"
        now = time.time()
        cached = PRICE_CACHE.get(key)
        if cached and (now - cached['ts'] < CACHE_TTL):
            print(f"Using cached data for {yf_sym}")
            return cached['data']
        
        print(f"Fetching {yf_sym} {interval}...")
        df = yf.download(yf_sym, period=period, interval=interval, progress=False)
        
        if df.empty or len(df) < 10:
            print(f"No data from yfinance, using mock data for {yf_sym}")
            rows = generate_mock_data()
        else:
            df = df.dropna().tail(PRICE_MAX)
            rows = []
            for idx, row in df.iterrows():
                rows.append({
                    'time': int(pd.Timestamp(idx).timestamp()),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
        
        PRICE_CACHE[key] = {'ts': now, 'data': rows}
        return rows
    except Exception as e:
        print(f"Price fetch error: {e}")
        return generate_mock_data()

@app.route('/price')
def price():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Missing symbol'}), 400
        
        interval = request.args.get('interval', '1d')
        
        # Set period based on interval
        if interval == '15m':
            period = '7d'  # Get last 7 days of 15min data
        elif interval == '1h':
            period = '30d'  # Get last 30 days of hourly data
        else:  # 1d
            period = '180d'  # Get last 180 days of daily data
        
        yf_sym = map_symbol(symbol)
        print(f"Mapped symbol: {symbol} -> {yf_sym}, interval: {interval}, period: {period}")
        
        rows = fetch_price(yf_sym, period=period, interval=interval)
        
        if not rows:
            rows = generate_mock_data()
        
        return jsonify(rows), 200
    except Exception as e:
        print(f"Price endpoint error: {e}")
        return jsonify(generate_mock_data()), 200

@app.route('/predict_fast')
def predict_fast():
    try:
        symbol = request.args.get('symbol')
        interval = request.args.get('interval', '1d')
        
        if not symbol:
            return jsonify({'error': 'Missing symbol'}), 400
        
        yf_sym = map_symbol(symbol)
        rows = fetch_price(yf_sym, period='180d', interval='1d')
        
        if len(rows) < 60:
            rows = generate_mock_data(points=100)
        
        if len(rows) == 0:
            return jsonify({'error': 'No data'}), 404
        
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df['time'], unit='s')
        
        feat_df = compute_features_df(df)
        # Attach monthly mood features if model expects them
        try:
            monthly_cols = [c for c in LSTM_FEATURES if c.startswith('monthly_') or c in ('momentum','mood_score')]
            if monthly_cols:
                try:
                    s = df['close'] if 'close' in df.columns else df['Close']
                    mood = monthly_mood(pd.DataFrame({'Close': s}, index=df.index))
                    mood = mood.reindex(df.index, method='ffill')
                    for c in monthly_cols:
                        if c in mood.columns:
                            feat_df[c] = mood[c]
                        else:
                            feat_df[c] = np.nan
                except Exception:
                    for c in monthly_cols:
                        feat_df[c] = np.nan
        except Exception:
            pass
        
        if feat_df is None or feat_df.empty:
            return jsonify({'error': 'Feature computation failed'}), 400
        
        last = feat_df.iloc[-1]
        x_vals = last.reindex(XGB_FEATURES)
        x_vals = x_vals.fillna(pd.Series(XGB_MEDIANS)).astype(float).values.reshape(1, -1)
        
        x_scaled = xgb_scaler.transform(x_vals)
        prob_xgb = float(xgb_model.predict_proba(x_scaled)[0, 1])
        
        seq_df = feat_df.reindex(columns=LSTM_FEATURES).fillna(pd.Series(LSTM_MEDIANS))
        if len(seq_df) < LOOKBACK:
            seq_vals = seq_df.values
            pad = np.tile(seq_vals[0], (LOOKBACK - len(seq_df), 1))
            seq_vals = np.vstack([pad, seq_vals])
        else:
            seq_vals = seq_df.iloc[-LOOKBACK:].values
        
        seq_scaled = lstm_scaler.transform(seq_vals)
        prob_lstm = float(lstm_model.predict(seq_scaled.reshape(1, LOOKBACK, -1), verbose=0)[0, 0])
        
        combined_prob = 0.6 * prob_lstm + 0.4 * prob_xgb
        signal = 'BUY' if combined_prob > 0.5 else 'SELL'

        # Fetch fundamentals (non-blocking best-effort)
        try:
            fundamentals = fetch_fundamentals(yf_sym)
        except Exception as e:
            fundamentals = {'error': str(e)}

        # If LSTM expects fundamental features, attach them to last and seq
        try:
            fund_cols = [c for c in LSTM_FEATURES if c in ('trailingPE','marketCap','profitMargins','totalDebt')]
            if fund_cols and isinstance(fundamentals, dict):
                for k in fund_cols:
                    val = fundamentals.get(k)
                    # attach to seq_df and last
                    try:
                        seq_df[k] = val
                    except Exception:
                        pass
                    try:
                        # also ensure last has value
                        pass
                    except Exception:
                        pass
        except Exception:
            pass

        return jsonify({
            'symbol': symbol,
            'prob_xgb': prob_xgb,
            'prob_lstm': prob_lstm,
            'prob_combined': combined_prob,
            'signal': signal,
            'fundamentals': fundamentals
        }), 200
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({
            'symbol': symbol if 'symbol' in locals() else 'UNKNOWN',
            'prob_xgb': 0.52,
            'prob_lstm': 0.48,
            'prob_combined': 0.50,
            'signal': 'BUY'
        }), 200

@app.route('/predict')
def predict():
    return predict_fast()

@app.route('/')
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'frontend'), 'tv_clone.html')


@app.route('/fundamentals')
def fundamentals_route():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Missing symbol'}), 400

        yf_sym = map_symbol(symbol)
        data = fetch_fundamentals(yf_sym)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/hmm_regime')
def hmm_regime():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Missing symbol'}), 400

        # try several model filename variants
        yf_sym = map_symbol(symbol)  # convert to .NS format
        base = str(yf_sym).upper()
        candidates = [f'hmm_{base}.pkl', f'hmm_{symbol.upper()}.pkl', f'hmm_{symbol.upper()}.NS.pkl']
        model_path = None
        for c in candidates:
            p = os.path.join('models', c)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            # missing model; try training one on the fly (may take a few seconds)
            print(f"hmm_regime: no model found for symbol={symbol}, trying to train on the fly")
            try:
                import yfinance as yf
                from hmmlearn.hmm import GaussianHMM

                df = yf.download(yf_sym, period='720d', interval='1d', progress=False)
                if df.empty:
                    raise ValueError('no price data')
                feats = compute_simple_features(df)
                if feats.empty:
                    raise ValueError('no computed features')

                # fit a 3‑state model
                model = GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
                model.fit(feats.values)
                os.makedirs('models', exist_ok=True)
                model_path = os.path.join('models', f'hmm_{yf_sym}.pkl')
                joblib.dump({'model': model, 'features': list(feats.columns)}, model_path)
                print(f"hmm_regime: trained and saved new model at {model_path}")
            except Exception as e:
                print(f"hmm_regime: on-the-fly training failed for {symbol}: {e}")
                return jsonify({'last_state': 0, 'states': [0]}), 200

        try:
            obj = joblib.load(model_path)
            model = obj['model']
        except ModuleNotFoundError:
            # If hmmlearn not installed, return default state
            return jsonify({
                'last_state': 0,
                'states': [0]
            }), 200

        yf_sym = map_symbol(symbol)
        rows = fetch_price(yf_sym, period='720d', interval='1d')
        if not rows:
            return jsonify({'last_state': 0, 'states': [0]}), 200

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df['time'], unit='s')

        # compute simple features similar to training
        df['ret1'] = df['close'].pct_change(1)
        df['ret5'] = df['close'].pct_change(5)
        df['vol20'] = df['ret1'].rolling(20).std()
        feat_df = df[['ret1', 'ret5', 'vol20']].dropna()
        if feat_df.empty:
            return jsonify({'last_state': 0, 'states': [0]}), 200

        X = feat_df.values
        predictions = model.predict(X)
        # Ensure predictions are valid integers
        try:
            states = [int(p) if np.isfinite(p) else 0 for p in predictions]
        except (ValueError, TypeError):
            states = [int(p) for p in predictions]

        return jsonify({
            'last_state': int(states[-1]) if states else 0,
            'states': [int(s) for s in states[-200:]]  # return up to last 200 states
        }), 200
    except Exception as e:
        # On any error, return valid state 0
        return jsonify({'last_state': 0, 'states': [0]}), 200


@app.route('/dqn_status')
def dqn_status():
    """Debug endpoint to check DQN loading status"""
    try:
        dqn = get_dqn_agent()
        weights_path = os.path.join('models', 'dqn_weights.weights.h5')
        return jsonify({
            'dqn_available': dqn is not None,
            'dqn_agent_loaded': DQN_AGENT_LOADED,
            'weights_file_exists': os.path.exists(weights_path),
            'weights_path': weights_path
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dqn_action')
def dqn_action():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Missing symbol'}), 400
        # Try to load DQN agent
        dqn = get_dqn_agent()
        dqn_available = dqn is not None

        yf_sym = map_symbol(symbol)
        rows = fetch_price(yf_sym, period='365d', interval='1d')
        if len(rows) < 20:
            rows = generate_mock_data(points=200)

        # fallback ensemble signal - compute directly
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df['time'], unit='s')
        feat_df = compute_features_df(df)
        if feat_df is not None and not feat_df.empty:
            last = feat_df.iloc[-1]
            x_vals = last.reindex(XGB_FEATURES)
            x_vals = x_vals.fillna(pd.Series(XGB_MEDIANS)).astype(float).values.reshape(1, -1)
            x_scaled = xgb_scaler.transform(x_vals)
            prob_xgb = float(xgb_model.predict_proba(x_scaled)[0, 1])
            seq_df = feat_df.reindex(columns=LSTM_FEATURES).fillna(pd.Series(LSTM_MEDIANS))
            if len(seq_df) < LOOKBACK:
                seq_vals = seq_df.values
                pad = np.tile(seq_vals[0], (LOOKBACK - len(seq_df), 1))
                seq_vals = np.vstack([pad, seq_vals])
            else:
                seq_vals = seq_df.iloc[-LOOKBACK:].values
            seq_scaled = lstm_scaler.transform(seq_vals)
            prob_lstm = float(lstm_model.predict(seq_scaled.reshape(1, LOOKBACK, -1), verbose=0)[0, 0])
            prob = 0.6 * prob_lstm + 0.4 * prob_xgb
        else:
            prob = 0.5
        ensemble_signal = 'BUY' if prob > 0.5 else 'SELL'

        # ensure prob is valid for JSON
        if not np.isfinite(prob):
            prob = 0.5

        result = {
            'symbol': symbol,
            'ensemble_signal': ensemble_signal,
            'prob_combined': float(prob),
            'dqn_available': dqn_available,
        }

        if dqn_available:
            # build environment and observation and get DQN action
            try:
                from envs.trading_env import TradingEnv
                prices = [r['close'] for r in rows]
                env = TradingEnv(prices, window=10)
                # set index to last window
                env.idx = len(prices) - 1
                obs = env._obs()
                x = np.array(obs).reshape(1, 1, -1)

                # get Q-values from the model
                qvals = dqn.predict(x, verbose=0)
                # Ensure Q-values are valid
                cleaned_qvals = np.array(qvals[0])
                cleaned_qvals = np.nan_to_num(cleaned_qvals, nan=0.0, posinf=1.0, neginf=-1.0)
                action_id = int(np.argmax(cleaned_qvals))
                action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

                # convert to safe float values for JSON serialization
                qvals_list = [float(v) for v in cleaned_qvals]

                result.update({
                    'dqn_action_id': int(action_id),
                    'dqn_action': action_map.get(action_id, str(action_id)),
                    'dqn_qvals': qvals_list
                })
            except Exception as e:
                result['dqn_error'] = str(e)
        else:
            result['note'] = 'DQN not trained or failed to load; returning ensemble signal.'

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/news')
def news():
    """Return recent news headlines and VADER sentiment for a symbol (free).
    Uses Google News RSS; no API key required.
    """
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        max_articles = int(request.args.get('max', 12))
        include_tweets = request.args.get('include_tweets', '0') in ('1','true','True')
        use_finbert = request.args.get('use_finbert', '0') in ('1','true','True')
        data = fetch_news_and_tweets(symbol, max_articles=max_articles, include_tweets=include_tweets, use_finbert=use_finbert)
        return jsonify({'symbol': symbol, **data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/news_summary')
def news_summary():
    """Return aggregated summary numbers for UI: newsAvg(0-100), tweetsAvg(0-100), trailingPE"""
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        include_tweets = request.args.get('include_tweets', '0') in ('1','true','True')
        use_finbert = request.args.get('use_finbert', '0') in ('1','true','True')

        data = fetch_news_and_tweets(symbol, max_articles=12, include_tweets=include_tweets, use_finbert=use_finbert)
        articles = data.get('articles', []) or []
        tweets = data.get('tweets', []) or []

        def safe_num(x):
            try:
                v = float(x)
                if np.isfinite(v):
                    return v
            except Exception:
                pass
            return 0.0

        def avg_arr(arr):
            if not arr:
                return 0.0
            s = 0.0
            cnt = 0
            for it in arr:
                # prefer finbert then sentiment
                v = it.get('sentiment_finbert') if it.get('sentiment_finbert') is not None else it.get('sentiment')
                v = safe_num(v)
                s += v
                cnt += 1
            return (s / cnt) * 100.0 if cnt > 0 else 0.0

        news_avg = avg_arr(articles)
        tweets_avg = avg_arr(tweets)

        # fundamentals
        try:
            yf_sym = map_symbol(symbol)
            fund = fetch_fundamentals(yf_sym)
            trailing_pe = None
            if isinstance(fund, dict):
                trailing_pe = fund.get('trailingPE')
        except Exception:
            trailing_pe = None

        return jsonify({'symbol': symbol, 'newsAvg': news_avg, 'tweetsAvg': tweets_avg, 'trailingPE': trailing_pe}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/article_preview')
def article_preview():
    """Fetch a URL and return a small HTML preview (first paragraphs) to embed in modal.
    Returns 200 with {title, preview_html} or 204 if nothing extracted.
    """
    try:
        url = request.args.get('url')
        if not url:
            return ('', 204)
        # basic safety: allow only http/https
        if not (url.startswith('http://') or url.startswith('https://')):
            return ('', 400)

        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; NSE-ML/1.0)'}
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return ('', 404)
        html = resp.text[:200000]

        # try BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            title = (soup.title.string if soup.title and soup.title.string else '')
            # collect first 3 non-empty <p> texts
            paras = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 30:
                    paras.append(text)
                if len(paras) >= 3:
                    break
            if not paras:
                return ('', 204)
            preview_html = '<p>' + '</p><p>'.join(paras) + '</p>'
            return jsonify({'title': title, 'preview_html': preview_html}), 200
        except Exception:
            # fallback: try to extract paragraphs via simple regex
            import re
            matches = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I|re.S)
            cleaned = []
            for m in matches:
                t = re.sub(r'<[^>]+>', '', m).strip()
                if t and len(t) > 30:
                    cleaned.append(t)
                if len(cleaned) >= 3:
                    break
            if not cleaned:
                return ('', 204)
            preview_html = '<p>' + '</p><p>'.join(cleaned) + '</p>'
            return jsonify({'title': '', 'preview_html': preview_html}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest')
def backtest():
    """Run backtest on a symbol and return performance metrics"""
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        start_date = request.args.get('start_date', '2023-06-01')
        end_date = request.args.get('end_date', '2024-12-31')
        
        # Import backtest module
        from backtest import BacktestEngine
        
        print(f"Running backtest for {symbol} ({start_date} to {end_date})...")
        
        # Create and run backtest
        bt = BacktestEngine(symbol, start_date=start_date, end_date=end_date, initial_capital=100000)
        df = bt.fetch_data()
        
        if df is None or len(df) == 0:
            return jsonify({'error': 'No data available for backtest'}), 404
        
        bt.run(df)
        metrics = bt.calculate_metrics()

        
        # Format trades for response
        trades_list = []
        for trade in bt.trades[-20:]:  # Last 20 trades
            trade_dict = dict(trade)
            if 'date' in trade_dict:
                trade_dict['date'] = str(trade_dict['date'])
            trades_list.append(trade_dict)
        
        return jsonify({
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'metrics': metrics,
            'recent_trades': trades_list,
            'final_equity': metrics['final_equity']
        }), 200
        
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/backtest_model')
def backtest_model():
    """Backtest all 3 models on historical data for a symbol."""
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        period = request.args.get('period', '1y')  # 1y, 3y, 5y
        
        yf_sym = map_symbol(symbol)
        rows = fetch_price(yf_sym, period=period, interval='1d')
        
        if len(rows) < 50:
            return jsonify({'error': 'Insufficient historical data'}), 400
        
        df = pd.DataFrame(rows)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time')
        
        results = backtest_on_data(df)
        
        if results is None:
            return jsonify({'error': 'Backtest calculation failed'}), 500
        
        return jsonify({
            'symbol': symbol,
            'period': period,
            'data_points': len(df),
            'results': results,
            'best_model': max(results, key=lambda x: results[x]['win_rate'])
        }), 200
    except Exception as e:
        print(f"Backtest endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_comparison')
def model_comparison():
    """Compare performance of XGB, LSTM, and DQN models."""
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        
        yf_sym = map_symbol(symbol)
        rows = fetch_price(yf_sym, period='1y', interval='1d')
        
        if len(rows) < 50:
            return jsonify({'error': 'Insufficient data'}), 400
        
        df = pd.DataFrame(rows)
        df = df.sort_values('time')
        
        # Get model predictions
        from backtest_engine import predict_xgb, predict_lstm
        
        xgb_probs = predict_xgb(df)
        lstm_probs = predict_lstm(df)
        
        if xgb_probs is None or lstm_probs is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        ensemble_probs = 0.6 * lstm_probs + 0.4 * xgb_probs
        
        # Compare current prediction
        xgb_signal = 'BUY' if xgb_probs[-1] > 0.5 else 'SELL'
        lstm_signal = 'BUY' if lstm_probs[-1] > 0.5 else 'SELL'
        ensemble_signal = 'BUY' if ensemble_probs[-1] > 0.5 else 'SELL'
        
        return jsonify({
            'symbol': symbol,
            'models': {
                'xgb': {
                    'confidence': float(xgb_probs[-1] * 100),
                    'signal': xgb_signal,
                    'avg_confidence': float(np.mean(xgb_probs) * 100)
                },
                'lstm': {
                    'confidence': float(lstm_probs[-1] * 100),
                    'signal': lstm_signal,
                    'avg_confidence': float(np.mean(lstm_probs) * 100)
                },
                'ensemble': {
                    'confidence': float(ensemble_probs[-1] * 100),
                    'signal': ensemble_signal,
                    'avg_confidence': float(np.mean(ensemble_probs) * 100)
                }
            },
            'disagreement': xgb_signal != lstm_signal
        }), 200
    except Exception as e:
        print(f"Model comparison error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_disagreement')
def model_disagreement():
    """Detect when models disagree and alert user."""
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        
        yf_sym = map_symbol(symbol)
        rows = fetch_price(yf_sym, period='30d', interval='1d')
        
        df = pd.DataFrame(rows)
        df = df.sort_values('time')
        
        from backtest_engine import predict_xgb, predict_lstm
        
        xgb_probs = predict_xgb(df)
        lstm_probs = predict_lstm(df)
        
        if xgb_probs is None or lstm_probs is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        xgb_signal = 'BUY' if xgb_probs[-1] > 0.5 else 'SELL'
        lstm_signal = 'BUY' if lstm_probs[-1] > 0.5 else 'SELL'
        
        disagreement = xgb_signal != lstm_signal
        
        return jsonify({
            'symbol': symbol,
            'xgb_signal': xgb_signal,
            'lstm_signal': lstm_signal,
            'xgb_confidence': float(xgb_probs[-1]),
            'lstm_confidence': float(lstm_probs[-1]),
            'disagreement': disagreement,
            'disagreement_strength': float(abs(xgb_probs[-1] - lstm_probs[-1])),
            'alert': 'HIGH' if disagreement and abs(xgb_probs[-1] - lstm_probs[-1]) > 0.3 else ('MEDIUM' if disagreement else 'NONE')
        }), 200
    except Exception as e:
        print(f"Disagreement detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/critical_news')
def critical_news():
    """Detect critical/extreme sentiment news and tweets."""
    try:
        symbol = request.args.get('symbol', 'RELIANCE')
        threshold = float(request.args.get('threshold', '0.7'))  # Sentiment extremes
        
        data = fetch_news_and_tweets(symbol, max_articles=25, include_tweets=True, use_finbert=False)
        articles = data.get('articles', []) or []
        tweets = data.get('tweets', []) or []
        
        critical_items = []
        
        # Find extreme sentiment items
        for item in articles + tweets:
            sentiment = item.get('sentiment_finbert') or item.get('sentiment', 0)
            if abs(sentiment) > threshold:
                critical_items.append({
                    'type': 'article' if 'source' in item else 'tweet',
                    'title': item.get('title') or item.get('text', '')[:100],
                    'sentiment': float(sentiment),
                    'source': item.get('source') or item.get('user', ''),
                    'url': item.get('link') or item.get('url', ''),
                    'published': item.get('published') or item.get('date', ''),
                    'severity': 'CRITICAL' if abs(sentiment) > 0.85 else 'HIGH'
                })
        
        # Sort by sentiment extremity
        critical_items.sort(key=lambda x: abs(x['sentiment']), reverse=True)
        
        return jsonify({
            'symbol': symbol,
            'critical_count': len(critical_items),
            'items': critical_items[:10],  # Top 10 most extreme
            'overall_alert': 'CRITICAL' if len(critical_items) > 3 else ('WARNING' if len(critical_items) > 0 else 'NORMAL')
        }), 200
    except Exception as e:
        print(f"Critical news detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_performance_stats')
def model_performance_stats():
    """Get overall model performance statistics."""
    try:
        stats = get_model_performance_stats()
        if stats is None:
            return jsonify({'error': 'Stats retrieval failed'}), 500
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'frontend'), filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
