import pandas as pd
import numpy as np


def monthly_mood(df):
    """Compute simple monthly mood/momentum features from an OHLCV dataframe.

    Expects a DataFrame with a datetime index and a 'close' or 'Close' column.
    Returns a DataFrame indexed by month-end with features like monthly_return, vol, zscore.
    """
    df = df.copy()
    if 'close' in df.columns:
        col = 'close'
    elif 'Close' in df.columns:
        col = 'Close'
    else:
        raise ValueError('DataFrame must contain close or Close column')

    s = df[col].astype(float)
    monthly = s.resample('M').last()
    monthly_ret = monthly.pct_change()
    monthly_vol = s.resample('M').apply(lambda x: x.pct_change().std())

    mood = pd.DataFrame({
        'monthly_return': monthly_ret,
        'monthly_vol': monthly_vol
    })
    mood['momentum'] = mood['monthly_return'].rolling(3).mean()
    mood['mood_score'] = (mood['momentum'] - mood['momentum'].mean()) / (mood['momentum'].std() + 1e-9)
    mood = mood.dropna()
    return mood
