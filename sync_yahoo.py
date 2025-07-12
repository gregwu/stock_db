import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()
# --- Config ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}
tickers = ["AAPL", "MSFT", "GOOGL"]  # Modify or dynamically fetch from DB

def get_last_date_for_ticker(conn, ticker):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(date) FROM stock_data 
            WHERE ticker = %s 
             -- AND date < (CURRENT_DATE - INTERVAL '1 day')
        """, (ticker,))
        result = cur.fetchone()[0]
        return result

def get_unique_tickers(conn):
    with conn.cursor() as cur:
        # Get all tickers that need updates
        cur.execute("""
            SELECT DISTINCT ticker 
            FROM stock_data 
            WHERE ticker IS NOT NULL
              AND ticker LIKE '%.US'
              AND (SELECT MAX(date) FROM stock_data sd2 WHERE sd2.ticker = stock_data.ticker) < CURRENT_DATE
        """)
        results = cur.fetchall()
        
        total_tickers = len(results)
        
        # Filter out tickers with underscores or hyphens in the base symbol
        filtered_tickers = []
        filtered_out_count = 0
        
        for row in results:
            full_ticker = row[0]  # e.g., "AAPL.US"
            if '.' in full_ticker:
                ticker_base = full_ticker.split('.')[0]  # e.g., "AAPL"
                
                # Skip tickers with problematic patterns:
                # 1. Underscores (special classes)
                # 2. Any hyphens (warrants, units, special classes like AACT-WS, AAM-U)
                # 3. Long tickers ending with 'W' (≥5 chars) - these are typically warrants
                #    Short W tickers (≤4 chars like ARW, CDW) are usually legitimate stocks
                should_filter = (
                    '_' in ticker_base or 
                    '-' in ticker_base or
                    (ticker_base.endswith('W') and len(ticker_base) >= 5)
                )
                
                if should_filter:
                    filtered_out_count += 1
                else:
                    filtered_tickers.append(ticker_base)
        
        tickers = sorted(set(filtered_tickers))
        
        if filtered_out_count > 0:
            print(f"🧹 Filtered out {filtered_out_count} problematic tickers (warrants ≥5 chars ending in W, hyphenated instruments) (total: {total_tickers} -> valid: {len(tickers)})")
        
        return tickers


def fetch_yahoo_data(ticker, start_date, include_today=False):
    try:
        # Use a more specific approach to catch yfinance errors
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            # Check for yfinance warnings/errors in the warning messages
            for warning in w:
                warning_msg = str(warning.message)
                if "YFInvalidPeriodError" in warning_msg or "Period 'max' is invalid" in warning_msg:
                    print(f"⚠️ Invalid ticker or delisted: {ticker} (Yahoo Finance period error)")
                    return None
                elif "YFTzMissingError" in warning_msg or "possibly delisted" in warning_msg:
                    print(f"⚠️ Possibly delisted ticker: {ticker}")
                    return None
        
        if df.empty:
            print(f"⚠️ No data found for ticker: {ticker} (empty response)")
            return None
            
    except Exception as e:
        error_msg = str(e)
        if "YFInvalidPeriodError" in error_msg or "Period 'max' is invalid" in error_msg:
            print(f"⚠️ Invalid ticker or delisted: {ticker} (Yahoo Finance doesn't recognize this symbol)")
        elif "YFChartError" in error_msg:
            print(f"⚠️ Chart data unavailable for ticker: {ticker}")
        elif "HTTPError" in error_msg or "404" in error_msg:
            print(f"⚠️ Ticker not found: {ticker} (HTTP 404)")
        elif "YFTzMissingError" in error_msg:
            print(f"⚠️ Possibly delisted ticker: {ticker}")
        else:
            print(f"❌ Error downloading data for {ticker}: {e}")
        return None

    df.reset_index(inplace=True)
    
    # Handle MultiIndex columns (yfinance returns MultiIndex when downloading)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the column names - take the first level (the actual column name)
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    df.rename(columns={
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Volume": "volume",
    }, inplace=True)

    df["ticker"] = ticker
    df["date"] = df["Date"]

    if not include_today:
        today = pd.Timestamp("today").normalize()
        df = df[df["date"] < today]

    return df

def insert_data(conn, df):
    """Insert stock data into database with comprehensive validation"""
    # Ensure the DataFrame is not None or empty
    if df is None or df.empty:
        print("   ⚠️ No data to insert")
        return

    # Handle MultiIndex columns if they still exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    # Ensure we have required columns
    required_columns = ['ticker', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"   ❌ Missing required columns: {missing_columns}")
        return
    
    # Data validation
    initial_count = len(df)
    
    # Remove records with invalid prices (negative or zero)
    price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
    valid_prices = df[price_columns].gt(0).all(axis=1)
    df = df[valid_prices]
    
    # Remove records with invalid high/low relationships
    valid_high_low = df['high_price'] >= df['low_price']
    df = df[valid_high_low]
    
    # Remove records with extreme price relationships (likely data errors)
    # High shouldn't be more than 50% higher than close, low shouldn't be more than 50% lower
    reasonable_high = df['high_price'] <= df['close_price'] * 1.5
    reasonable_low = df['low_price'] >= df['close_price'] * 0.5
    df = df[reasonable_high & reasonable_low]
    
    if len(df) < initial_count:
        removed_count = initial_count - len(df)
        print(f"   🧹 Removed {removed_count} invalid records")
    
    if df.empty:
        print("   ⚠️ No valid data remaining after validation")
        return
    
    # Ensure we have the 'time' column (set to None if not present)
    if 'time' not in df.columns:
        df['time'] = None
    
    # Make sure date column is properly formatted
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Select only the columns we need for the database
    df_insert = df[required_columns].copy()
    
    df_insert = df_insert.where(pd.notnull(df_insert), None)  # Replace NaN with None

    rows = df_insert.to_dict("records")

    with conn.cursor() as cur:
        inserted_count = 0
        for row in rows:
            try:
                cur.execute("""
                    INSERT INTO stock_data (
                        ticker, date, open_price, high_price, low_price, close_price, volume
                    )
                    VALUES (
                        %(ticker)s, %(date)s, %(open_price)s, %(high_price)s, %(low_price)s, %(close_price)s, %(volume)s
                    )
                    ON CONFLICT (ticker, date) DO NOTHING
                """, row)
                if cur.rowcount > 0:
                    inserted_count += 1
            except Exception as e:
                print(f"   ⚠️ Error inserting row for {row.get('date', 'unknown date')}: {e}")
                continue
                
        conn.commit()
        
    ticker_name = df_insert['ticker'].iloc[0] if len(df_insert) > 0 else 'unknown ticker'
    print(f"   ✅ Inserted {inserted_count} new records for {ticker_name}")



def main():
    """Main function to sync Yahoo Finance data for tickers that need updates"""
    import argparse
    
    # Add command line argument parsing for better usability
    parser = argparse.ArgumentParser(description='Sync Yahoo Finance data to PostgreSQL database')
    parser.add_argument('--ticker', type=str, help='Process only this specific ticker (without .US suffix)')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process (for testing)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without making changes')
    args = parser.parse_args()
    
    conn = None
    successful_count = 0
    failed_count = 0
    
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        if args.ticker:
            # Process single ticker
            tickers = [args.ticker]
            print(f"🎯 Processing specific ticker: {args.ticker}")
        else:
            # Get all tickers needing updates
            tickers = get_unique_tickers(conn)
            if args.limit:
                tickers = tickers[:args.limit]
                print(f"🔧 Limited to first {args.limit} tickers for testing")
        
        if not tickers:
            print("✅ No tickers found that need updating. Database is up to date.")
            return True
            
        print(f"📊 Found {len(tickers)} tickers that need updating")
        
        if args.dry_run:
            print("🔍 DRY RUN - Would process these tickers:")
            for ticker in tickers[:10]:  # Show first 10
                print(f"   {ticker}")
            if len(tickers) > 10:
                print(f"   ... and {len(tickers) - 10} more")
            return True
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
            try:
                last_date = get_last_date_for_ticker(conn, ticker + ".US")
                start_date = last_date + pd.Timedelta(days=1) if last_date else "2000-01-01"
                print(f"   📅 Fetching from {start_date}")

                df = fetch_yahoo_data(ticker, start_date, True)
                if df is not None and not df.empty:
                    df["ticker"] = ticker + ".US"  # Store in DB as original format
                    insert_data(conn, df)
                    successful_count += 1
                else:
                    print(f"   ⚠️ No data available")
                    failed_count += 1
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed_count += 1
                continue  # Continue with the next ticker

        print(f"\n🎉 Processing complete!")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📊 Total processed: {successful_count + failed_count}")
        
        return successful_count > 0
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

   

if __name__ == "__main__":
    main()

