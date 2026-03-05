import yfinance as yf
import pandas as pd


def safe_get(d, key):
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return v


def fetch_fundamentals(yf_symbol):
    """Fetch basic fundamental metrics for a symbol using yfinance.

    Returns a dict of common metrics (PE, market cap, revenue, debt ratios, margins).
    """
    try:
        tk = yf.Ticker(yf_symbol)
        info = tk.info or {}

        # Basic metrics from info
        metrics = {
            'symbol': yf_symbol,
            'longName': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'marketCap': safe_get(info, 'marketCap'),
            'trailingPE': safe_get(info, 'trailingPE'),
            'forwardPE': safe_get(info, 'forwardPE'),
            'priceToBook': safe_get(info, 'priceToBook'),
            'debtToEquity': safe_get(info, 'debtToEquity'),
            'profitMargins': safe_get(info, 'profitMargins'),
            'operatingMargins': safe_get(info, 'operatingMargins'),
            'returnOnEquity': safe_get(info, 'returnOnEquity'),
            'currentRatio': safe_get(info, 'currentRatio'),
            'quickRatio': safe_get(info, 'quickRatio'),
            'EBITDA': safe_get(info, 'ebitda'),
        }

        # Try to extract recent financials (Total Revenue, Net Income) from financial statements
        try:
            fin = tk.financials
            if isinstance(fin, pd.DataFrame) and not fin.empty:
                # financials columns are periods; take latest column
                latest = fin.columns[0]
                rev = fin.loc[fin.index.str.contains('Total Revenue|Revenue'), latest] if any(fin.index.str.contains('Total Revenue|Revenue')) else None
                net = fin.loc[fin.index.str.contains('Net Income|NetIncome'), latest] if any(fin.index.str.contains('Net Income|NetIncome')) else None
                if rev is not None and hasattr(rev, 'item'):
                    metrics['totalRevenue'] = safe_get({ 'v': rev.item() }, 'v')
                if net is not None and hasattr(net, 'item'):
                    metrics['netIncome'] = safe_get({ 'v': net.item() }, 'v')
        except Exception:
            pass

        # Balance sheet for debt information
        try:
            bal = tk.balance_sheet
            if isinstance(bal, pd.DataFrame) and not bal.empty:
                latest = bal.columns[0]
                debt = None
                if any(bal.index.str.contains('Total Debt|Long Term Debt')):
                    debt = bal.loc[bal.index.str.contains('Total Debt|Long Term Debt'), latest]
                elif 'Total Liab' in bal.index:
                    debt = bal.loc['Total Liab', latest]
                if debt is not None and hasattr(debt, 'item'):
                    metrics['totalDebt'] = safe_get({ 'v': debt.item() }, 'v')
        except Exception:
            pass

        return metrics
    except Exception as e:
        return {'error': str(e), 'symbol': yf_symbol}
