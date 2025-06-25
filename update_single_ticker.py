#!/usr/bin/env python3
"""
update_single_ticker.py - Update technical indicators for a specific ticker

Usage: python update_single_ticker.py <ticker>
Example: python update_single_ticker.py AAPL.US
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from feature_engineering import add_technical_indicators

# Load environment variables
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def update_ticker_indicators(ticker):
    """Update technical indicators for a specific ticker"""
    # Create database engine
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(connection_string)
    
    print(f"🔄 Processing {ticker}...")
    
    # First, check if there are any records missing indicators
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
        print(f"✅ All indicators already calculated for {ticker}")
        return
    
    print(f"📊 Found {missing_count} records missing indicators for {ticker}")
    
    # Get ALL data for the ticker (needed for proper indicator calculation)
    full_data_query = text("""
    SELECT ticker, date, time, open_price, high_price, low_price, close_price, 
           volume, open_interest
    FROM stock_data 
    WHERE ticker = :ticker 
    ORDER BY date
    """)
    
    with engine.connect() as conn:
        df_full = pd.read_sql(full_data_query, conn, params={'ticker': ticker})
    
    if df_full.empty:
        print(f"❌ No data found for ticker: {ticker}")
        return
    
    # Get records that need updating
    missing_data_query = text("""
    SELECT date
    FROM stock_data 
    WHERE ticker = :ticker 
      AND (sma_20 IS NULL OR sma_144 IS NULL OR rsi_14 IS NULL OR macd IS NULL OR bb_upper IS NULL)
    ORDER BY date
    """)
    
    with engine.connect() as conn:
        missing_dates_df = pd.read_sql(missing_data_query, conn, params={'ticker': ticker})
    
    missing_dates = set(missing_dates_df['date'].tolist())
    print(f"🎯 Will update {len(missing_dates)} specific records")
    
    # Rename columns for feature engineering
    df_full = df_full.rename(columns={
        'close_price': 'CLOSE',
        'volume': 'VOL'
    })
    
    # Calculate indicators for ALL data
    df_full = add_technical_indicators(df_full)
    
    # Filter to only update records that were missing indicators
    df_to_update = df_full[df_full['date'].isin(missing_dates)].copy()
    
    # Update database - only the records that were missing indicators
    updated_count = 0
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for _, row in df_to_update.iterrows():
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
                
                # Handle NaN values
                params = {
                    'ticker': ticker,
                    'date': row['date'],
                    'sma_20': None if pd.isna(row['SMA_20']) else float(row['SMA_20']),
                    'sma_144': None if pd.isna(row['SMA_144']) else float(row['SMA_144']),
                    'sma_144_dist': None if pd.isna(row['SMA_144_Dist']) else float(row['SMA_144_Dist']),
                    'sma_144_slope': None if pd.isna(row['SMA_144_Slope']) else float(row['SMA_144_Slope']),
                    'rsi_14': None if pd.isna(row['RSI_14']) else float(row['RSI_14']),
                    'macd': None if pd.isna(row['MACD']) else float(row['MACD']),
                    'macd_signal': None if pd.isna(row['MACD_Signal']) else float(row['MACD_Signal']),
                    'macd_hist': None if pd.isna(row['MACD_Hist']) else float(row['MACD_Hist']),
                    'macd_up': None if pd.isna(row['MACD_Up']) else bool(row['MACD_Up']),
                    'bb_middle': None if pd.isna(row['BB_Middle']) else float(row['BB_Middle']),
                    'bb_std': None if pd.isna(row['BB_Std']) else float(row['BB_Std']),
                    'bb_upper': None if pd.isna(row['BB_Upper']) else float(row['BB_Upper']),
                    'bb_lower': None if pd.isna(row['BB_Lower']) else float(row['BB_Lower']),
                    'touches_lower_bb': None if pd.isna(row['Touches_Lower_BB']) else bool(row['Touches_Lower_BB']),
                    'touches_upper_bb': None if pd.isna(row['Touches_Upper_BB']) else bool(row['Touches_Upper_BB']),
                    'vol_sma_10': None if pd.isna(row['Vol_SMA_10']) else float(row['Vol_SMA_10']),
                    'volume_spike': None if pd.isna(row['Volume_Spike']) else bool(row['Volume_Spike'])
                }
                
                result = conn.execute(update_query, params)
                updated_count += result.rowcount
            
            trans.commit()
            print(f"✅ Updated {updated_count} records for {ticker}")
            
        except Exception as e:
            trans.rollback()
            print(f"❌ Error updating {ticker}: {e}")
            raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python update_single_ticker.py <ticker>")
        print("Example: python update_single_ticker.py AAPL.US")
        sys.exit(1)
    
    ticker = sys.argv[1]
    update_ticker_indicators(ticker)

if __name__ == "__main__":
    main()
