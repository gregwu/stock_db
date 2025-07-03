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
        cur.execute("""
            SELECT ticker, MAX(date) AS last_date
            FROM stock_data
            GROUP BY ticker
            HAVING MAX(date) < CURRENT_DATE        
        """)
        results = cur.fetchall()
        tickers = sorted(set(row[0].split('.')[0] for row in results if '.' in row[0]))
        return tickers


def fetch_yahoo_data(ticker, start_date, include_today=False):
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        #print(f"📈 Downloaded data for {ticker} from {start_date} to {datetime.now().date()}")
        if df.empty:
            print(f"⚠️ No data found for ticker: {ticker}")
            return None
    except Exception as e:
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
    print(df.columns.tolist())  # Debug: Show column names before processing
    print(df.head())  # Debug: Show the first few rows of the DataFrame
    # Ensure the DataFrame is not None or empty
    if df is None or df.empty:
        return

    #print("DataFrame columns:", df.columns.tolist())  # Debug: Show column names
    #print(df.head())  # Debug: Show the first few rows of the DataFrame
    
    # Handle MultiIndex columns if they still exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    # Ensure we have the 'time' column (set to None if not present)
    if 'time' not in df.columns:
        df['time'] = None
    
    # Make sure date column is properly formatted
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Select only the columns we need for the database
    required_columns = ['ticker', 'date',  'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    df_insert = df[required_columns].copy()
    
    df_insert = df_insert.where(pd.notnull(df_insert), None)  # Replace NaN with None
    print(df_insert.head())  # Debug: Show the first few rows of the DataFrame to be inserted
    rows = df_insert.to_dict("records")

    with conn.cursor() as cur:
        for row in rows:

            
            cur.execute("""
                INSERT INTO stock_data (
                    ticker, date,  open_price, high_price, low_price, close_price, volume
                )
                VALUES (
                    %(ticker)s, %(date)s,  %(open_price)s, %(high_price)s, %(low_price)s, %(close_price)s, %(volume)s
                )
                ON CONFLICT (ticker, date) DO NOTHING
            """, row)
        conn.commit()
        
    print(f"✅ Inserted {len(rows)} rows for {df_insert['ticker'].iloc[0] if len(df_insert) > 0 else 'unknown ticker'}")



def main():
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],  # Try connecting to default database first
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        tickers = get_unique_tickers(conn)
        if not tickers:
            print("No tickers found in the database. Exiting.")
            return
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
            try:
                last_date = get_last_date_for_ticker(conn, ticker + ".US")  # Ensure match in DB
                start_date = last_date + pd.Timedelta(days=1) if last_date else "2000-01-01"
                print(f"Fetching {ticker} from {start_date}...")

                df = fetch_yahoo_data(ticker, start_date, True)
                if df is not None:
                    df["ticker"] = ticker + ".US"  # Store in DB as original format
                    insert_data(conn, df)
                else:
                    print(f"⚠️ Skipping {ticker} - no data available")
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                continue  # Continue with the next ticker

        print("populate indicators")
        
        print("✅ All tickers processed successfully.")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ psycopg2 connection failed: {e}")
        return False

   

if __name__ == "__main__":
    main()

