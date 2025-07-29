import os
import pandas as pd
import logging
import numpy as np
from scipy.stats import linregress
from config import DB_CONFIG
from sqlalchemy import create_engine, text
import warnings


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('savedb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




def db_query(query, params=None):
    """
    Execute a SQL query and return the result as a pandas DataFrame.
    
    Args:
        query (str): SQL query string
        params (tuple, optional): Parameters for parameterized queries
    
    Returns:
        pd.DataFrame: Query results
    """
    try:
        # Create SQLAlchemy engine for PostgreSQL
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        engine = create_engine(connection_string)
        
        # Use connection directly and text() wrapper to avoid immutabledict issues
        with engine.connect() as connection:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
                # Use text() wrapper for the query to avoid parameter issues
                if params is not None and len(params) > 0:
                    df = pd.read_sql(text(query), con=connection, params=params)
                else:
                    df = pd.read_sql(text(query), con=connection)
        
        engine.dispose()
        return df
    except Exception as e:
        print(f"Database query error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error
    

def db_execute(query):
    """
    Execute a SQL command that does not return a result set.
    """
    try:
        # Create SQLAlchemy engine for PostgreSQL
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text(query))
            conn.commit()
        
        engine.dispose()
        return True
    except Exception as e:
        print(f"Database execution error: {e}")
        return False




def clean_volume(v):
    """Clean volume data by converting to integer."""
    try:
        return int(float(v))
    except:
        return 0
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

def manual_rolling_std(series, window=20):
    arr = series.values if isinstance(series, pd.Series) else np.array(series)
    result = [np.nan] * (window - 1)
    for i in range(window - 1, len(arr)):
        result.append(np.std(arr[i-window+1:i+1], ddof=1))
    return result


def calculate_all_technical_indicators(df):
    """
    Calculate comprehensive technical indicators for stock data.
    This function combines basic and enhanced technical features into one unified calculation.
    Expects columns: CLOSE, VOL, HIGH, LOW, OPEN (uppercase)
    """
    logger.info("🔧 Calculating comprehensive technical indicators...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Ensure we have the required base columns
    required_cols = ['CLOSE', 'VOL', 'HIGH', 'LOW', 'OPEN']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"❌ Missing required column '{col}'. Available columns: {list(df.columns)}")
            return df
    
    # =================
    # BASIC PRICE FEATURES
    # =================
    df['PRICE_CHANGE'] = df['CLOSE'].pct_change()
    df['PRICE_CHANGE_ABS'] = df['PRICE_CHANGE'].abs()
    df['HIGH_LOW_RATIO'] = df['HIGH'] / df['LOW']
    df['OPEN_CLOSE_RATIO'] = df['OPEN'] / df['CLOSE']
    df['PRICE_VOLATILITY'] = (df['HIGH'] - df['LOW']) / df['CLOSE']

    # =================
    # MOVING AVERAGES
    # =================
    # Simple Moving Averages (basic + enhanced)
    for window in [5, 20, 50, 144]:
        col_name = f'SMA_{window}'
        df[col_name] = df['CLOSE'].rolling(window=window).mean()
        
        df[f'PRICE_TO_SMA_{window}'] = df['CLOSE'] / df[col_name]
        df[f'SMA_{window}_SLOPE'] = df[col_name].rolling(window=20).apply(lambda x: calc_slope(pd.Series(x)), raw=False)
    
    # SMA_144 specific features
    df['SMA_144_Dist'] = (df['CLOSE'] - df['SMA_144']) / df['SMA_144'] * 100
    

    
    #df['SMA_144_Slope'] = df['SMA_144'].rolling(window=20).apply(lambda x: calc_slope(pd.Series(x)), raw=False)
    
    # Exponential Moving Averages
    df['EMA_12'] = df['CLOSE'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['CLOSE'].ewm(span=26, adjust=False).mean()
    
    # =================
    # MACD FAMILY
    # =================
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Up'] = df['MACD_Hist'].diff() > 0
    
    # Enhanced MACD features
    df['MACD_HISTOGRAM'] = df['MACD_Hist']  # Alias for consistency
    df['MACD_MOMENTUM'] = df['MACD'].pct_change(3)
    
    # =================
    # RSI
    # =================
    # Standard RSI calculation matching TA library
    delta = df['CLOSE'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Calculate the first average gain and loss using simple moving average
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    
    # Apply Wilder's smoothing for subsequent values
    alpha = 1.0 / 14
    for i in range(14, len(df)):
        if pd.notna(avg_gain.iloc[i-1]) and pd.notna(avg_loss.iloc[i-1]):
            avg_gain.iloc[i] = alpha * gain.iloc[i] + (1 - alpha) * avg_gain.iloc[i-1]
            avg_loss.iloc[i] = alpha * loss.iloc[i] + (1 - alpha) * avg_loss.iloc[i-1]
    
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # =================
    # BOLLINGER BANDS
    # =================
    df['BB_Middle'] = df['SMA_20']
    df['BB_Std'] = df['CLOSE'].rolling(20).std(ddof=0)  # Use population std for BB
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['Touches_Lower_BB'] = df['CLOSE'] <= df['BB_Lower']
    df['Touches_Upper_BB'] = df['CLOSE'] >= df['BB_Upper']
    
    # Enhanced Bollinger Bands features
    df['BB_WIDTH'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_POSITION'] = (df['CLOSE'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # =================
    # STOCHASTIC OSCILLATOR
    # =================
    low_14 = df['LOW'].rolling(14).min()
    high_14 = df['HIGH'].rolling(14).max()
    df['STOCH_K'] = ((df['CLOSE'] - low_14) / (high_14 - low_14)) * 100
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
    
    # =================
    # VOLUME INDICATORS
    # =================
    # Basic volume indicators
    df['Vol_SMA_10'] = df['VOL'].rolling(10).mean()
    df['Volume_Spike'] = df['VOL'] > 1.5 * df['Vol_SMA_10']
    
    # Enhanced volume features
    df['VOLUME_SMA'] = df['VOL'].rolling(20).mean()
    df['VOLUME_RATIO'] = df['VOL'] / df['VOLUME_SMA']
    df['VOLUME_MOMENTUM'] = df['VOL'].pct_change(3)
    df['PRICE_VOLUME'] = df['CLOSE'] * df['VOL']
    
    # On-Balance Volume (OBV)
    price_change = df['CLOSE'].diff()
    obv_values = []
    obv = 0
    
    for i in range(len(df)):
        if i == 0 or pd.isna(price_change.iloc[i]):
            obv_values.append(obv)
        elif price_change.iloc[i] > 0:
            obv += df['VOL'].iloc[i]
            obv_values.append(obv)
        elif price_change.iloc[i] < 0:
            obv -= df['VOL'].iloc[i]
            obv_values.append(obv)
        else:  # price_change == 0
            obv_values.append(obv)
    
    df['OBV'] = obv_values
    df['OBV_MOMENTUM'] = df['OBV'].pct_change(5)
    
    # =================
    # VOLATILITY MEASURES
    # =================
    df['VOLATILITY_20'] = df['PRICE_CHANGE'].rolling(20).std() * np.sqrt(252)
    df['VOLATILITY_MOMENTUM'] = df['VOLATILITY_20'].pct_change(5)
    
    # =================
    # SUPPORT/RESISTANCE LEVELS
    # =================
    df['RESISTANCE_20'] = df['HIGH'].rolling(20).max()
    df['SUPPORT_20'] = df['LOW'].rolling(20).min()
    df['RESISTANCE_DISTANCE'] = (df['RESISTANCE_20'] - df['CLOSE']) / df['CLOSE']
    df['SUPPORT_DISTANCE'] = (df['CLOSE'] - df['SUPPORT_20']) / df['CLOSE']
    df['CHANGE_PCT'] = df['CLOSE'].pct_change(periods=1) * 100
    df['CHANGE_LOW'] = (df['LOW'] - df['CLOSE'].shift(1)) / df['CLOSE'].shift(1) * 100
    df['CHANGE_PCT_1D'] = (df['CLOSE'].shift(-1) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_2D'] = (df['CLOSE'].shift(-2) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_3D'] = (df['CLOSE'].shift(-3) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_4D'] = (df['CLOSE'].shift(-4) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_5D'] = (df['CLOSE'].shift(-5) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_6D'] = (df['CLOSE'].shift(-6) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_7D'] = (df['CLOSE'].shift(-7) - df['CLOSE']) / df['CLOSE'] * 100
    df['CHANGE_PCT_14D'] = (df['CLOSE'].shift(-14) - df['CLOSE']) / df['CLOSE'] * 100


    # =================
    # LAGGED FEATURES (KEY PREDICTORS)
    # =================
    for lag in [1, 2, 3, 4, 5]:
        df[f'PRICE_CHANGE_LAG_{lag}'] = df['PRICE_CHANGE'].shift(lag)
        df[f'VOLUME_RATIO_LAG_{lag}'] = df['VOLUME_RATIO'].shift(lag)
    
    logger.info(f"✅ Comprehensive technical indicators calculated. DataFrame now has {len(df.columns)} columns")
    
    return df