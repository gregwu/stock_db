
import pandas as pd
import numpy as np
from scipy.stats import linregress

def add_technical_indicators(df):
    """
    Add technical indicators to a DataFrame
    Expects columns: CLOSE, VOL, etc. (uppercase)
    """
    # Ensure we have the required columns
    required_cols = ['CLOSE', 'VOL']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Simple Moving Averages
    df['SMA_20'] = df['CLOSE'].rolling(20).mean()
    df['SMA_144'] = df['CLOSE'].rolling(144).mean()

    def calc_slope(series, window=10):
        y = series[-window:]
        x = np.arange(len(y))
        if y.isnull().any() or len(y) < window:
            return np.nan
        try:
            slope, _, _, _, _ = linregress(x, y)
            return slope
        except:
            return np.nan

    # SMA 144 Distance and Slope
    df['SMA_144_Dist'] = (df['CLOSE'] - df['SMA_144']) / df['SMA_144'] * 100
    df['SMA_144_Slope'] = df['SMA_144'].rolling(window=20).apply(lambda x: calc_slope(pd.Series(x)), raw=False)

    # RSI (Relative Strength Index)
    up = df['CLOSE'].diff().clip(lower=0)
    down = -df['CLOSE'].diff().clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['CLOSE'].ewm(span=12).mean()
    ema_26 = df['CLOSE'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Up'] = df['MACD_Hist'].diff() > 0

    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20']
    df['BB_Std'] = df['CLOSE'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['Touches_Lower_BB'] = df['CLOSE'] <= df['BB_Lower']
    df['Touches_Upper_BB'] = df['CLOSE'] >= df['BB_Upper']

    # Volume indicators
    df['Vol_SMA_10'] = df['VOL'].rolling(10).mean()
    df['Volume_Spike'] = df['VOL'] > 1.5 * df['Vol_SMA_10']

    return df
