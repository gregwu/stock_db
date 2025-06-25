#!/usr/bin/env python3
"""
populate_indicators.py - Populate technical indicators for existing stock data

This script reads stock data from the PostgreSQL database, calculates technical
indicators using the feature_engineering module, and updates the database with
the calculated values.
"""

import os
import sys
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from feature_engineering import add_technical_indicators

# Load environment variables
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'gangwu'),
    'user': os.getenv('DB_USER', 'gangwu'),
    'password': os.getenv('DB_PASSWORD', 'gangwu')
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
    # First check if there are missing indicators
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
    SELECT ticker, date, time, open_price, high_price, low_price, close_price, 
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
    
    # Rename columns to match feature_engineering expectations
    df_full = df_full.rename(columns={
        'open_price': 'OPEN',
        'high_price': 'HIGH', 
        'low_price': 'LOW',
        'close_price': 'CLOSE',
        'volume': 'VOL',
        'open_interest': 'OPENINT',
        'ticker': 'TICKER',
        'date': 'DATE',
        'time': 'TIME'
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
    
    # Prepare the data for database update
    update_data = df_to_update[['DATE', 'SMA_20', 'SMA_144', 'SMA_144_Dist', 'SMA_144_Slope',
                               'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Up',
                               'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower', 
                               'Touches_Lower_BB', 'Touches_Upper_BB', 
                               'Vol_SMA_10', 'Volume_Spike']].copy()
    
    # Convert boolean columns to proper format
    boolean_cols = ['MACD_Up', 'Touches_Lower_BB', 'Touches_Upper_BB', 'Volume_Spike']
    for col in boolean_cols:
        if col in update_data.columns:
            update_data[col] = update_data[col].astype('boolean')
    
    # Replace NaN with None for database
    update_data = update_data.where(pd.notnull(update_data), None)
    
    updated_count = 0
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for _, row in update_data.iterrows():
                # Update query for technical indicators
                update_query = text("""
                UPDATE stock_data 
                SET sma_20 = :sma_20,
                    sma_144 = :sma_144,
                    sma_144_dist = :sma_144_dist,
                    sma_144_slope = :sma_144_slope,
                    rsi_14 = :rsi_14,
                    macd = :macd,
                    macd_signal = :macd_signal,
                    macd_hist = :macd_hist,
                    macd_up = :macd_up,
                    bb_middle = :bb_middle,
                    bb_std = :bb_std,
                    bb_upper = :bb_upper,
                    bb_lower = :bb_lower,
                    touches_lower_bb = :touches_lower_bb,
                    touches_upper_bb = :touches_upper_bb,
                    vol_sma_10 = :vol_sma_10,
                    volume_spike = :volume_spike
                WHERE ticker = :ticker AND date = :date
                """)
                
                params = {
                    'ticker': ticker,
                    'date': row['DATE'],
                    'sma_20': row['SMA_20'],
                    'sma_144': row['SMA_144'],
                    'sma_144_dist': row['SMA_144_Dist'],
                    'sma_144_slope': row['SMA_144_Slope'],
                    'rsi_14': row['RSI_14'],
                    'macd': row['MACD'],
                    'macd_signal': row['MACD_Signal'],
                    'macd_hist': row['MACD_Hist'],
                    'macd_up': row['MACD_Up'],
                    'bb_middle': row['BB_Middle'],
                    'bb_std': row['BB_Std'],
                    'bb_upper': row['BB_Upper'],
                    'bb_lower': row['BB_Lower'],
                    'touches_lower_bb': row['Touches_Lower_BB'],
                    'touches_upper_bb': row['Touches_Upper_BB'],
                    'vol_sma_10': row['Vol_SMA_10'],
                    'volume_spike': row['Volume_Spike']
                }
                
                result = conn.execute(update_query, params)
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
        
        # Calculate technical indicators for all data
        df_with_indicators = add_technical_indicators(df_full)
        
        # Update database only for missing records
        updated_count = update_indicators_in_db(engine, ticker, df_with_indicators, missing_dates)
        
        logger.info(f"✅ Updated {updated_count} records for {ticker}")
        return updated_count
        
    except Exception as e:
        logger.error(f"❌ Error processing {ticker}: {e}")
        return 0

def main():
    """Main function"""
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Populate technical indicators for stock data')
    parser.add_argument('--ticker', type=str, help='Process only this specific ticker')
    args = parser.parse_args()
    
    print("🚀 Starting technical indicators population...")
    
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
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
