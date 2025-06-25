#!/usr/bin/env python3
"""
savedb.py - Load stock data from CSV files to PostgreSQL database

This script uses the load_csv_file function from data_loader.py to process
stock data files and save them to a PostgreSQL database.
"""

import os
import sys
import pandas as pd
import psycopg2
from pathlib import Path
from tqdm import tqdm
from sqlalchemy import create_engine, text
import logging
from data_loader import load_csv_file
from dotenv import load_dotenv

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
        time TIME,
        open_price REAL NOT NULL,
        high_price REAL NOT NULL,
        low_price REAL NOT NULL,
        close_price REAL NOT NULL,
        volume BIGINT NOT NULL,
        open_interest REAL,
        
        -- Technical Indicators
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
    
    # Define columns that should be kept in the database
    allowed_columns = {
        'TICKER': 'ticker',
        'DATE': 'date',
        'TIME': 'time',
        'OPEN': 'open_price',
        'HIGH': 'high_price',
        'LOW': 'low_price',
        'CLOSE': 'close_price',
        'VOL': 'volume',
        'OPENINT': 'open_interest',
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
        'Volume_Spike': 'volume_spike'
    }
    
    # Log all columns in the DataFrame for debugging
    logger.info(f"📋 DataFrame columns for {ticker}: {list(df.columns)}")
    
    # Filter to only keep allowed columns
    available_columns = [col for col in db_df.columns if col in allowed_columns]
    db_df = db_df[available_columns]
    
    # Log any columns that were dropped
    dropped_columns = [col for col in df.columns if col not in allowed_columns]
    if dropped_columns:
        logger.info(f"🗑️ Dropped columns for {ticker}: {dropped_columns}")
    
    # Rename columns to match database schema
    db_df = db_df.rename(columns={k: v for k, v in allowed_columns.items() if k in db_df.columns})
    
    # Handle data type conversions
    try:
        # Handle time column - if it's numeric (like 0), convert to None
        if 'time' in db_df.columns:
            # If time column contains only zeros or is numeric, set to None
            if db_df['time'].dtype in ['int64', 'float64'] or (db_df['time'] == 0).all():
                db_df['time'] = None
                logger.info(f"🕐 Set time column to NULL for {ticker} (was numeric)")
            
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
    
    return db_df

def save_dataframe_to_db(df, engine, ticker):
    """Save DataFrame to PostgreSQL database"""
    try:
        # Prepare DataFrame
        db_df = prepare_dataframe_for_db(df, ticker)
        
        # Save to database with conflict resolution
        db_df.to_sql(
            'stock_data',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        
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
