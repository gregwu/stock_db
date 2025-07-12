#!/usr/bin/env python3
"""
savedb.py - Integrated Stock Data Processing and Database Loader

This script provides a complete, self-contained solution for:
1. Loading stock data from CSV files
2. Calculating basic and enhanced technical indicators
3. Saving processed data to PostgreSQL database

Features:
- Integrated data loading (no external data_loader dependency)
- Basic technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
- Enhanced technical features (60+ advanced indicators)
- Comprehensive database schema with all features
- Memory-optimized processing
- Error handling and logging

Dependencies: pandas, numpy, psycopg2, sqlalchemy, scipy, python-dotenv, tqdm
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path
from tqdm import tqdm
from sqlalchemy import create_engine, text
import logging
from dotenv import load_dotenv
from scipy.stats import linregress
import ta

pd.set_option('display.max_rows', None)
# Database configuration
# Load environment variables from .env file
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

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

def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not specific database)
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database='postgres'  # Connect to default postgres database
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['database'],))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            logger.info(f"✅ Created database: {DB_CONFIG['database']}")
        else:
            logger.info(f"📊 Database {DB_CONFIG['database']} already exists")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error creating database: {e}")
        raise

def create_stock_data_table(engine):
    """Create the stock_data table with proper schema"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS stock_data (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        open_price REAL NOT NULL,
        high_price REAL NOT NULL,
        low_price REAL NOT NULL,
        close_price REAL NOT NULL,
        volume BIGINT NOT NULL,
        open_interest REAL,
        
        -- Basic Technical Indicators
        sma_20 REAL,
        sma_144 REAL,
        sma_144_dist REAL,
        sma_144_slope REAL,
        rsi_14 REAL,
        macd REAL,
        macd_signal REAL,
        macd_hist REAL,
        macd_up BOOLEAN,
        bb_middle REAL,
        bb_std REAL,
        bb_upper REAL,
        bb_lower REAL,
        touches_lower_bb BOOLEAN,
        touches_upper_bb BOOLEAN,
        vol_sma_10 REAL,
        volume_spike BOOLEAN,
        
        -- Enhanced Price Features
        price_change REAL,
        price_change_abs REAL,
        high_low_ratio REAL,
        open_close_ratio REAL,
        price_volatility REAL,
        
        -- Additional Moving Averages
        sma_5 REAL,
        sma_50 REAL,
        price_to_sma_5 REAL,
        price_to_sma_20 REAL,
        price_to_sma_50 REAL,
        price_to_sma_144 REAL,
        sma_5_slope REAL,
        sma_20_slope REAL,
        sma_50_slope REAL,
        
        -- Exponential Moving Averages
        ema_12 REAL,
        ema_26 REAL,
        
        -- Enhanced MACD Features
        macd_histogram REAL,
        macd_momentum REAL,
        
        -- Enhanced Bollinger Bands
        bb_width REAL,
        bb_position REAL,
        
        -- Stochastic Oscillator
        stoch_k REAL,
        stoch_d REAL,
        
        -- Enhanced Volume Features
        volume_sma REAL,
        volume_ratio REAL,
        volume_momentum REAL,
        price_volume REAL,
        obv REAL,
        obv_momentum REAL,
        
        -- Volatility Measures
        volatility_20 REAL,
        volatility_momentum REAL,
        
        -- Support/Resistance Levels
        resistance_20 REAL,
        support_20 REAL,
        resistance_distance REAL,
        support_distance REAL,
        
        -- Lagged Features (Key Predictors)
        price_change_lag_1 REAL,
        price_change_lag_2 REAL,
        price_change_lag_3 REAL,
        price_change_lag_4 REAL,
        price_change_lag_5 REAL,
        volume_ratio_lag_1 REAL,
        volume_ratio_lag_2 REAL,
        volume_ratio_lag_3 REAL,
        volume_ratio_lag_4 REAL,
        volume_ratio_lag_5 REAL,
        change_pct REAL,
        change_low REAL,
        change_pct_1d REAL,
        change_pct_2d REAL,
        change_pct_3d REAL,
        change_pct_4d REAL,
        change_pct_5d REAL,
        change_pct_6d REAL,
        change_pct_7d REAL,
        change_pct_14d REAL,

        -- Indexes
        UNIQUE(ticker, date)
    );
    
    -- Create indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_stock_data_ticker ON stock_data(ticker);
    CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date);
    CREATE INDEX IF NOT EXISTS idx_stock_data_ticker_date ON stock_data(ticker, date);
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        logger.info("✅ Stock data table created/verified")
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")
        raise

def prepare_dataframe_for_db(df, ticker):
    """Prepare DataFrame for database insertion"""
    # Make a copy to avoid modifying original
    db_df = df.copy()
    
    # Add ticker if not present
    if 'TICKER' not in db_df.columns:
        db_df['TICKER'] = ticker
    
    # Technical indicators should already be calculated by load_csv_file
    # No need for additional calculation here
    logger.info(f"📊 Preparing {len(db_df)} records for {ticker} (already has {len(db_df.columns)} columns)")
    
    # Define columns that should be kept in the database
    allowed_columns = {
        'TICKER': 'ticker',
        'DATE': 'date',
        'OPEN': 'open_price',
        'HIGH': 'high_price',
        'LOW': 'low_price',
        'CLOSE': 'close_price',
        'VOL': 'volume',
        'OPENINT': 'open_interest',
        
        # Basic Technical Indicators (support both old and new formats)
        'SMA_20': 'sma_20',
        'SMA_144': 'sma_144',
        'SMA_144_Dist': 'sma_144_dist',
        'SMA_144_SLOPE': 'sma_144_slope',
        'RSI_14': 'rsi_14',
        'MACD': 'macd',
        'MACD_Signal': 'macd_signal',
        'MACD_Hist': 'macd_hist',
        'MACD_Up': 'macd_up',
        'BB_Middle': 'bb_middle',
        'BB_Std': 'bb_std',
        'BB_Upper': 'bb_upper',
        'BB_Lower': 'bb_lower',
        'Touches_Lower_BB': 'touches_lower_bb',
        'Touches_Upper_BB': 'touches_upper_bb',
        'Vol_SMA_10': 'vol_sma_10',
        'Volume_Spike': 'volume_spike',
        
        # Enhanced Price Features (uppercase)
        'PRICE_CHANGE': 'price_change',
        'PRICE_CHANGE_ABS': 'price_change_abs',
        'HIGH_LOW_RATIO': 'high_low_ratio',
        'OPEN_CLOSE_RATIO': 'open_close_ratio',
        'PRICE_VOLATILITY': 'price_volatility',
        
        # Additional Moving Averages (uppercase)
        'SMA_5': 'sma_5',
        'SMA_50': 'sma_50',
        'PRICE_TO_SMA_5': 'price_to_sma_5',
        'PRICE_TO_SMA_20': 'price_to_sma_20',
        'PRICE_TO_SMA_50': 'price_to_sma_50',
        'PRICE_TO_SMA_144': 'price_to_sma_144',
        'SMA_5_SLOPE': 'sma_5_slope',
        'SMA_20_SLOPE': 'sma_20_slope',
        'SMA_50_SLOPE': 'sma_50_slope',
        
        # Exponential Moving Averages (uppercase)
        'EMA_12': 'ema_12',
        'EMA_26': 'ema_26',
        
        # Enhanced MACD Features (uppercase)
        'MACD_HISTOGRAM': 'macd_histogram',
        'MACD_MOMENTUM': 'macd_momentum',
        
        # Enhanced Bollinger Bands (uppercase)
        'BB_WIDTH': 'bb_width',
        'BB_POSITION': 'bb_position',
        
        # Stochastic Oscillator (uppercase)
        'STOCH_K': 'stoch_k',
        'STOCH_D': 'stoch_d',
        
        # Enhanced Volume Features (uppercase)
        'VOLUME_SMA': 'volume_sma',
        'VOLUME_RATIO': 'volume_ratio',
        'VOLUME_MOMENTUM': 'volume_momentum',
        'PRICE_VOLUME': 'price_volume',
        'OBV': 'obv',
        'OBV_MOMENTUM': 'obv_momentum',
        
        # Volatility Measures (uppercase)
        'VOLATILITY_20': 'volatility_20',
        'VOLATILITY_MOMENTUM': 'volatility_momentum',
        
        # Support/Resistance Levels (uppercase)
        'RESISTANCE_20': 'resistance_20',
        'SUPPORT_20': 'support_20',
        'RESISTANCE_DISTANCE': 'resistance_distance',
        'SUPPORT_DISTANCE': 'support_distance',
        'CHANGE_PCT': 'change_pct',
        'CHANGE_LOW': 'change_low',
        'CHANGE_PCT_1D': 'change_pct_1d',
        'CHANGE_PCT_2D': 'change_pct_2d',
        'CHANGE_PCT_3D': 'change_pct_3d',
        'CHANGE_PCT_4D': 'change_pct_4d',
        'CHANGE_PCT_5D': 'change_pct_5d',
        'CHANGE_PCT_6D': 'change_pct_6d',
        'CHANGE_PCT_7D': 'change_pct_7d',
        'CHANGE_PCT_14D': 'change_pct_14d',

        # Lagged Features (Key Predictors) (uppercase)
        'PRICE_CHANGE_LAG_1': 'price_change_lag_1',
        'PRICE_CHANGE_LAG_2': 'price_change_lag_2',
        'PRICE_CHANGE_LAG_3': 'price_change_lag_3',
        'PRICE_CHANGE_LAG_4': 'price_change_lag_4',
        'PRICE_CHANGE_LAG_5': 'price_change_lag_5',
        'VOLUME_RATIO_LAG_1': 'volume_ratio_lag_1',
        'VOLUME_RATIO_LAG_2': 'volume_ratio_lag_2',
        'VOLUME_RATIO_LAG_3': 'volume_ratio_lag_3',
        'VOLUME_RATIO_LAG_4': 'volume_ratio_lag_4',
        'VOLUME_RATIO_LAG_5': 'volume_ratio_lag_5'
    }
    
    # Log all columns in the DataFrame for debugging
    original_columns = list(db_df.columns)
    #logger.info(f"📋 DataFrame columns for {ticker}: {original_columns}")
    
    # Filter to only keep allowed columns and handle duplicates
    available_columns = []
    used_db_columns = set()
    
    for col in db_df.columns:
        if col in allowed_columns:
            db_col = allowed_columns[col]
            # Only add if we haven't seen this database column yet
            if db_col not in used_db_columns:
                available_columns.append(col)
                used_db_columns.add(db_col)
            else:
                logger.info(f"⚠️ Skipping duplicate mapping for {col} -> {db_col}")
    
    db_df = db_df[available_columns]
    
    # Log any columns that were dropped
    dropped_columns = [col for col in original_columns if col not in available_columns]
    if dropped_columns:
        logger.info(f"🗑️ Dropped columns for {ticker}: {dropped_columns}")
    
    # Rename columns to match database schema
    db_df = db_df.rename(columns={k: v for k, v in allowed_columns.items() if k in db_df.columns})
    
    # Handle data type conversions
    try:
        # Ensure date column is properly formatted
        if 'date' in db_df.columns:
            db_df['date'] = pd.to_datetime(db_df['date']).dt.date
            
        # Convert boolean columns
        boolean_cols = ['macd_up', 'touches_lower_bb', 'touches_upper_bb', 'volume_spike']
        for col in boolean_cols:
            if col in db_df.columns:
                db_df[col] = db_df[col].astype(bool)
                
        # Ensure numeric columns are proper types
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        for col in numeric_cols:
            if col in db_df.columns:
                db_df[col] = pd.to_numeric(db_df[col], errors='coerce')
                
        # Handle volume as integer
        if 'volume' in db_df.columns:
            db_df['volume'] = db_df['volume'].astype('Int64')  # Nullable integer
            
    except Exception as e:
        logger.warning(f"⚠️ Data type conversion warning for {ticker}: {e}")
    
    # Ensure required columns exist
    required_cols = ['ticker', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in required_cols:
        if col not in db_df.columns:
            logger.warning(f"⚠️ Missing required column: {col}")

    #print(f"📊 Prepared DataFrame for {ticker} with {len(db_df)} records and {len(db_df.columns)} columns")
    #print(db_df[['date', 'close_price', 'bb_std']])
    return db_df

def save_dataframe_to_db(df, engine, ticker):
    """Save DataFrame to PostgreSQL database"""
    try:
        # Prepare DataFrame
        db_df = prepare_dataframe_for_db(df, ticker)
        
        # Save to database with conflict resolution
        db_df.to_sql( 'stock_data', engine, if_exists='append', index=False, method='multi')
        
        logger.info(f"✅ Saved {len(db_df)} records for {ticker}")
        return len(db_df)
        
    except Exception as e:
        logger.error(f"❌ Error saving {ticker}: {e}")
        return 0

def process_stock_files(data_directory, engine):
    """Process all stock files in the directory"""
    data_path = Path(data_directory)
    if not data_path.exists():
        logger.error(f"❌ Directory not found: {data_directory}")
        return
    
    # Find all .txt files
    files = list(data_path.rglob("*.txt"))
    logger.info(f"📁 Found {len(files)} files to process")
    
    total_records = 0
    processed_files = 0
    failed_files = 0
    
    # Process files with progress bar
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Extract ticker from filename
            ticker = file_path.stem.split('.')[0].upper()
            #if(ticker != 'TOPS'):
                #logger.info(f"Skipping file {file_path.name} for ticker {ticker}")
            #    continue
            print(f"📄 Processing file: {file_path.name} for ticker {ticker}")
            # Load and process data
            df = load_csv_file(file_path, dropna=False)
            
            if df is not None and len(df) > 0:
                # Save to database

                records_saved = save_dataframe_to_db(df, engine, ticker)
                total_records += records_saved
                processed_files += 1
            else:
                logger.warning(f"⚠️ No data loaded for {ticker}")
                failed_files += 1
                
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path.name}: {e}")
            failed_files += 1
    
    logger.info(f"📊 Processing complete:")
    logger.info(f"   ✅ Files processed: {processed_files}")
    logger.info(f"   ❌ Files failed: {failed_files}")
    logger.info(f"   📈 Total records saved: {total_records}")

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

def load_csv_file(filepath, dropna=True):
    """Load and process CSV stock data file with technical indicators."""
    # Load and clean
    if filepath.stat().st_size == 0:
        raise ValueError("File is empty")

    df = pd.read_csv(filepath, skiprows=1, header=None)
    if len(df) == 0:
        raise ValueError("File contains no data rows")

    df.columns = ["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT"]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    df = df.sort_values("DATE")

    df["VOL"] = df["VOL"].apply(clean_volume)
    df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE", "VOL"])

    # Memory optimization
    df = df.astype({'OPEN': 'float64', 'HIGH': 'float64', 'LOW': 'float64', 'CLOSE': 'float64', 'VOL': 'float64'})

    # Add comprehensive technical indicators
    df = calculate_all_technical_indicators(df)

    # Drop NaNs caused by rolling windows
    if dropna:
        df = df.dropna().reset_index(drop=True)
    
    #print(df[['DATE', 'CLOSE', 'BB_Std']].tail(30))
    #print(df['BB_Std'].isna().sum())
    #print(df['BB_Std'].value_counts())

    return df

def main():
    """Main function"""
    print("🚀 Starting stock data database loader...")
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python savedb.py <data_directory>")
        print("Example: python savedb.py stock_csv")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    
    try:
        # Create database if needed
        create_database_if_not_exists()
        
        # Create database engine
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            logger.info(f"🔗 Connected to PostgreSQL: {result.fetchone()[0]}")
        
        # Create table schema
        create_stock_data_table(engine)
        
        # Process stock files
        process_stock_files(data_directory, engine)
        
        print("🎉 Database loading completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
