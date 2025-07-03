#!/usr/bin/env python3
"""
populate_indicators.py - Populate technical indicators for existing stock data

This script reads stock data from the PostgreSQL database, calculates technical
indicators using the unified calculation function from savedb.py, and updates 
the database with the calculated values.

Updated to use the new unified technical indicator pipeline with uppercase columns.
"""

import os
import sys
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Import the unified technical indicator calculation function
from savedb import calculate_all_technical_indicators

# Load environment variables
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
        logging.FileHandler('populate_indicators.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_unique_tickers(engine):
    """Get all unique tickers from the database"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker"))
        tickers = [row[0] for row in result.fetchall()]
    return tickers

def get_stock_data(engine, ticker):
    """Get stock data for a specific ticker, but only process if there are missing indicators"""
    # First check if there are missing indicators (checking key ones that should exist)
    check_query = text("""
    SELECT COUNT(*) as missing_count
    FROM stock_data 
    WHERE ticker = :ticker 
      AND (sma_20 IS NULL OR sma_144 IS NULL OR rsi_14 IS NULL OR macd IS NULL OR bb_upper IS NULL)
    """)
    
    with engine.connect() as conn:
        result = conn.execute(check_query, {'ticker': ticker})
        missing_count = result.fetchone()[0]
    
    if missing_count == 0:
        logger.info(f"✅ All indicators already calculated for {ticker}")
        return None, None
    
    logger.info(f"📊 Found {missing_count} records missing indicators for {ticker}")
    
    # Get ALL data for proper indicator calculation
    full_query = text("""
    SELECT ticker, date, open_price, high_price, low_price, close_price, 
           volume, open_interest
    FROM stock_data 
    WHERE ticker = :ticker 
    ORDER BY date
    """)
    
    # Get records that need updating
    missing_query = text("""
    SELECT date
    FROM stock_data 
    WHERE ticker = :ticker 
      AND (sma_20 IS NULL OR sma_144 IS NULL OR rsi_14 IS NULL OR macd IS NULL OR bb_upper IS NULL)
    ORDER BY date
    """)
    
    with engine.connect() as conn:
        df_full = pd.read_sql(full_query, conn, params={'ticker': ticker})
        missing_dates_df = pd.read_sql(missing_query, conn, params={'ticker': ticker})
    
    if df_full.empty:
        return None, None
    
    missing_dates = set(missing_dates_df['date'].tolist())
    
    # Rename columns to match the new unified function expectations (uppercase)
    df_full = df_full.rename(columns={
        'open_price': 'OPEN',
        'high_price': 'HIGH', 
        'low_price': 'LOW',
        'close_price': 'CLOSE',
        'volume': 'VOL',
        'open_interest': 'OPENINT',
        'ticker': 'TICKER',
        'date': 'DATE',
    })
    
    return df_full, missing_dates

def update_indicators_in_db(engine, ticker, df_full, missing_dates):
    """Update the technical indicators in the database for specific dates only"""
    if df_full is None or df_full.empty or not missing_dates:
        return 0
    
    # Filter to only update records that were missing indicators
    df_to_update = df_full[df_full['DATE'].isin(missing_dates)].copy()
    
    if df_to_update.empty:
        return 0
    
    # Define all indicator columns that need to be updated
    # This maps DataFrame column names (uppercase) to database column names (lowercase)
    indicator_columns = {
        # Basic Technical Indicators
        'SMA_20': 'sma_20',
        'SMA_144': 'sma_144', 
        'SMA_144_Dist': 'sma_144_dist',
        'SMA_144_Slope': 'sma_144_slope',
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
        
        # Enhanced Price Features
        'PRICE_CHANGE': 'price_change',
        'PRICE_CHANGE_ABS': 'price_change_abs',
        'HIGH_LOW_RATIO': 'high_low_ratio',
        'OPEN_CLOSE_RATIO': 'open_close_ratio',
        'PRICE_VOLATILITY': 'price_volatility',
        
        # Additional Moving Averages
        'SMA_5': 'sma_5',
        'SMA_50': 'sma_50',
        'PRICE_TO_SMA_5': 'price_to_sma_5',
        'PRICE_TO_SMA_20': 'price_to_sma_20',
        'PRICE_TO_SMA_50': 'price_to_sma_50',
        'SMA_5_SLOPE': 'sma_5_slope',
        'SMA_20_SLOPE': 'sma_20_slope',
        'SMA_50_SLOPE': 'sma_50_slope',
        
        # Exponential Moving Averages
        'EMA_12': 'ema_12',
        'EMA_26': 'ema_26',
        
        # Enhanced MACD Features
        'MACD_HISTOGRAM': 'macd_histogram',
        'MACD_MOMENTUM': 'macd_momentum',
        
        # Enhanced Bollinger Bands
        'BB_WIDTH': 'bb_width',
        'BB_POSITION': 'bb_position',
        
        # Stochastic Oscillator
        'STOCH_K': 'stoch_k',
        'STOCH_D': 'stoch_d',
        
        # Enhanced Volume Features
        'VOLUME_SMA': 'volume_sma',
        'VOLUME_RATIO': 'volume_ratio',
        'VOLUME_MOMENTUM': 'volume_momentum',
        'PRICE_VOLUME': 'price_volume',
        'OBV': 'obv',
        'OBV_MOMENTUM': 'obv_momentum',
        
        # Volatility Measures
        'VOLATILITY_20': 'volatility_20',
        'VOLATILITY_MOMENTUM': 'volatility_momentum',
        
        # Support/Resistance Levels
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
        # Lagged Features
        'PRICE_CHANGE_LAG_1': 'price_change_lag_1',
        'PRICE_CHANGE_LAG_2': 'price_change_lag_2',
        'PRICE_CHANGE_LAG_3': 'price_change_lag_3',
        'PRICE_CHANGE_LAG_4': 'price_change_lag_4',
        'PRICE_CHANGE_LAG_5': 'price_change_lag_5',
        'VOLUME_RATIO_LAG_1': 'volume_ratio_lag_1',
        'VOLUME_RATIO_LAG_2': 'volume_ratio_lag_2',
        'VOLUME_RATIO_LAG_3': 'volume_ratio_lag_3',
        'VOLUME_RATIO_LAG_4': 'volume_ratio_lag_4',
        'VOLUME_RATIO_LAG_5': 'volume_ratio_lag_5',
    }
    
    # Filter to only columns that exist in the DataFrame
    available_columns = ['DATE'] + [col for col in indicator_columns.keys() if col in df_to_update.columns]
    update_data = df_to_update[available_columns].copy()
    
    # Convert boolean columns to proper format
    boolean_cols = ['MACD_Up', 'Touches_Lower_BB', 'Touches_Upper_BB', 'Volume_Spike']
    for col in boolean_cols:
        if col in update_data.columns:
            update_data[col] = update_data[col].astype('boolean')
    
    # Replace NaN with None for database
    update_data = update_data.where(pd.notnull(update_data), None)
    
    # Build dynamic update query
    set_clauses = []
    for df_col, db_col in indicator_columns.items():
        if df_col in update_data.columns:
            set_clauses.append(f"{db_col} = :{df_col.lower()}")
    
    if not set_clauses:
        logger.warning(f"No indicator columns found to update for {ticker}")
        return 0
    
    update_query_sql = f"""
    UPDATE stock_data 
    SET {', '.join(set_clauses)}
    WHERE ticker = :ticker AND date = :date
    """
    
    updated_count = 0
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for _, row in update_data.iterrows():
                # Build parameters dictionary
                params = {'ticker': ticker, 'date': row['DATE']}
                
                # Add all indicator values
                for df_col, db_col in indicator_columns.items():
                    if df_col in update_data.columns:
                        params[df_col.lower()] = row[df_col]
                
                result = conn.execute(text(update_query_sql), params)
                updated_count += result.rowcount
            
            trans.commit()
            
        except Exception as e:
            trans.rollback()
            logger.error(f"❌ Error updating {ticker}: {e}")
            raise
    
    return updated_count

def process_ticker(engine, ticker):
    """Process a single ticker: get data, calculate indicators, update database"""
    try:
        # Get stock data and missing dates
        df_full, missing_dates = get_stock_data(engine, ticker)
        if df_full is None or missing_dates is None:
            return 0
        
        if not missing_dates:
            logger.info(f"✅ No missing indicators for {ticker}")
            return 0
        
        # Calculate technical indicators for all data using the unified function
        df_with_indicators = calculate_all_technical_indicators(df_full)
        
        # Update database only for missing records
        updated_count = update_indicators_in_db(engine, ticker, df_with_indicators, missing_dates)
        
        logger.info(f"✅ Updated {updated_count} records for {ticker}")
        return updated_count
        
    except Exception as e:
        logger.error(f"❌ Error processing {ticker}: {e}")
        return 0

def main():
    """Main function - populate technical indicators using unified calculation pipeline"""
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Populate technical indicators for stock data using unified pipeline')
    parser.add_argument('--ticker', type=str, help='Process only this specific ticker')
    args = parser.parse_args()
    
    print("🚀 Starting technical indicators population with unified pipeline...")
    
    try:
        # Create database engine
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            logger.info(f"🔗 Connected to PostgreSQL: {result.fetchone()[0]}")
        
        # Get tickers to process
        if args.ticker:
            tickers = [args.ticker]
            logger.info(f"📊 Processing specific ticker: {args.ticker}")
        else:
            tickers = get_unique_tickers(engine)
            logger.info(f"📊 Found {len(tickers)} tickers to process")
        
        if not tickers:
            logger.warning("⚠️ No tickers found in database")
            return
        
        # Process each ticker
        total_updated = 0
        successful_tickers = 0
        
        for ticker in tqdm(tickers, desc="Processing tickers"):
            updated_count = process_ticker(engine, ticker)
            if updated_count > 0:
                total_updated += updated_count
                successful_tickers += 1
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"   ✅ Tickers processed: {successful_tickers}/{len(tickers)}")
        logger.info(f"   📈 Total records updated: {total_updated}")
        logger.info(f"   🔧 Using unified technical indicator pipeline with {70} features")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
