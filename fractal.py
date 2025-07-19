import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from io import StringIO
import yfinance as yf
from datetime import datetime, timedelta
import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
import json

# --- Utility Functions ---

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def normalize_and_resample(series, target_length):
    arr = np.array(series)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    resampled = np.interp(
        np.linspace(0, len(arr)-1, target_length), np.arange(len(arr)), arr
    )
    return resampled

def dtw_similarity(a, b):
    distance, path = fastdtw(a, b)
    sim = 1 / (1 + distance)   # Normalize: higher sim is better
    return sim

def advanced_slide_and_compare(series, patterns, window_sizes, threshold=0.95, method='cosine', allow_inversion=False, use_dtw=False):
    matches = []
    for pat_name, pattern in patterns:
        pat_len = len(pattern)
        for win_size in window_sizes:
            for start in range(len(series) - win_size + 1):
                window = series[start:start+win_size]
                window_norm = normalize_and_resample(window, pat_len)
                if use_dtw:
                    sim = dtw_similarity(pattern, window_norm)
                else:
                    sim = 1 - cosine(pattern, window_norm)
                found_inversion = False
                inv_sim = None
                if allow_inversion:
                    if use_dtw:
                        inv_sim = dtw_similarity(-pattern, window_norm)
                    else:
                        inv_sim = 1 - cosine(-pattern, window_norm)
                    if inv_sim > sim:
                        sim = inv_sim
                        found_inversion = True
                if sim >= threshold:
                    matches.append({
                        'pattern': pat_name,
                        'start': start,
                        'size': win_size,
                        'similarity': sim,
                        'window': window_norm,
                        'inverted': found_inversion
                    })
    return matches

def resample_ohlc(df, rule='W'):
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df.resample(rule).agg(ohlc_dict).dropna()

def save_search_results_to_folder(search_results, search_config, data_info, base_folder="fractal_results"):
    """Save search results to organized folder structure"""
    try:
        # Extract ticker name from data_info for folder naming
        ticker_name = "unknown"
        if "for " in data_info:
            # Extract ticker from strings like "123 days for AAPL.US from database"
            ticker_part = data_info.split("for ")[1].split(" ")[0]
            # Remove .US suffix if present
            ticker_name = ticker_part.replace(".US", "")
        elif "from uploaded file" in data_info:
            ticker_name = "uploaded"
        
        # Create timestamped folder with ticker name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = os.path.join(base_folder, f"{ticker_name}_{timestamp}")
        os.makedirs(results_folder, exist_ok=True)
        
        # Extract data from search results
        matches_weekly = search_results['matches_weekly']
        matches_daily = search_results['matches_daily']
        weekly_data = search_results['weekly_data']
        daily_data = search_results['daily_data']
        patterns = search_results['patterns']
        
        # 1. Save search configuration
        config_file = os.path.join(results_folder, "search_config.json")
        config_data = {
            "search_config": search_config,
            "data_info": data_info,
            "timestamp": timestamp,
            "total_weekly_matches": len(matches_weekly),
            "total_daily_matches": len(matches_daily),
            "pattern_names": [p[0] for p in patterns]
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        # 2. Save match data as CSV
        match_data = []
        for m in matches_weekly:
            match_data.append({
                'timeframe': 'weekly',
                'pattern': m['pattern'],
                'start_index': m['start'],
                'window_size': m['size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': weekly_data.index[m['start']],
                'end_date': weekly_data.index[m['start']+m['size']-1]
            })
        
        for m in matches_daily:
            match_data.append({
                'timeframe': 'daily',
                'pattern': m['pattern'],
                'start_index': m['start'],
                'window_size': m['size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': daily_data.index[m['start']],
                'end_date': daily_data.index[m['start']+m['size']-1]
            })
        
        match_df = pd.DataFrame(match_data)
        match_csv_file = os.path.join(results_folder, "all_matches.csv")
        match_df.to_csv(match_csv_file, index=False)
        
        # 3. Save top matches with charts
        charts_folder = os.path.join(results_folder, "charts")
        os.makedirs(charts_folder, exist_ok=True)
        
        def save_top_matches_charts(matches, data_series, dates, timeframe, n=5):
            by_pattern = {}
            for m in matches:
                by_pattern.setdefault(m['pattern'], []).append(m)
            
            for pat_name in by_pattern:
                pattern = [p[1] for p in patterns if p[0]==pat_name][0]
                sorted_matches = sorted(by_pattern[pat_name], key=lambda x: -x['similarity'])
                
                for i, m in enumerate(sorted_matches[:n]):
                    # Normalized pattern comparison chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(range(len(pattern)), pattern, 'k-', lw=3, label='Reference pattern')
                    
                    if m['inverted']:
                        ax.plot(range(len(pattern)), -pattern, 'c--', lw=2, label='Inverted reference')
                        ax.plot(range(len(pattern)), m['window'], 'm-', alpha=0.7, 
                               label=f'Inverted Match (sim={m["similarity"]:.3f})')
                    else:
                        ax.plot(range(len(pattern)), m['window'], 'r-', alpha=0.7, 
                               label=f'Match (sim={m["similarity"]:.3f})')
                    
                    ax.set_title(f'{timeframe.title()} | {pat_name} | Match {i+1}\n'
                               f'{dates[m["start"]].strftime("%Y-%m-%d")} to {dates[m["start"]+m["size"]-1].strftime("%Y-%m-%d")}'
                               f'{" (Inverted)" if m["inverted"] else ""}')
                    ax.set_xlabel('Normalized Time')
                    ax.set_ylabel('Normalized Price')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    chart_filename = f"{timeframe}_{pat_name.replace(' ', '_')}_{i+1}_pattern.png"
                    chart_path = os.path.join(charts_folder, chart_filename)
                    plt.tight_layout()
                    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Actual price chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    idx_range = range(m["start"], m["start"]+m["size"])
                    ax.plot(dates[idx_range], data_series[idx_range], 'b-', lw=2)
                    ax.set_title(f'{timeframe.title()} Actual Price Movement\n'
                               f'{dates[m["start"]].strftime("%Y-%m-%d")} to {dates[m["start"]+m["size"]-1].strftime("%Y-%m-%d")}\n'
                               f'Similarity: {m["similarity"]:.3f}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45)
                    
                    price_filename = f"{timeframe}_{pat_name.replace(' ', '_')}_{i+1}_price.png"
                    price_path = os.path.join(charts_folder, price_filename)
                    plt.tight_layout()
                    plt.savefig(price_path, dpi=150, bbox_inches='tight')
                    plt.close()
        
        # Save charts for both timeframes
        save_top_matches_charts(matches_weekly, weekly_data['close'].values, weekly_data.index, "weekly")
        save_top_matches_charts(matches_daily, daily_data['close'].values, daily_data.index, "daily")
        
        # 4. Save raw data used in analysis
        data_folder = os.path.join(results_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        
        weekly_data.to_csv(os.path.join(data_folder, "weekly_data.csv"))
        daily_data.to_csv(os.path.join(data_folder, "daily_data.csv"))
        
        # 5. Save pattern data
        patterns_folder = os.path.join(results_folder, "patterns")
        os.makedirs(patterns_folder, exist_ok=True)
        
        for pat_name, pattern in patterns:
            pattern_df = pd.DataFrame({'normalized_value': pattern})
            pattern_filename = f"{pat_name.replace(' ', '_')}.csv"
            pattern_df.to_csv(os.path.join(patterns_folder, pattern_filename), index=False)
        
        # 6. Create summary report
        summary_file = os.path.join(results_folder, "README.md")
        with open(summary_file, 'w') as f:
            f.write(f"# Fractal Pattern Search Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Data Source:** {data_info}\n\n")
            f.write(f"## Search Configuration\n")
            f.write(f"- **Similarity Threshold:** {search_config['similarity_threshold']:.2f}\n")
            f.write(f"- **Method:** {search_config['match_method'].upper()}\n")
            f.write(f"- **Allow Inversion:** {search_config['allow_inversion']}\n")
            f.write(f"- **Min Window:** {search_config['min_window_monthly']}\n")
            f.write(f"- **Max Window Multiplier:** {search_config['max_window_multiplier']}\n\n")
            f.write(f"## Results Summary\n")
            f.write(f"- **Weekly Matches:** {len(matches_weekly)}\n")
            f.write(f"- **Daily Matches:** {len(matches_daily)}\n")
            f.write(f"- **Patterns Used:** {', '.join([p[0] for p in patterns])}\n\n")
            f.write(f"## Folder Structure\n")
            f.write(f"- `search_config.json` - Complete search configuration and metadata\n")
            f.write(f"- `all_matches.csv` - All matches with details\n")
            f.write(f"- `charts/` - Pattern comparison and price charts for top matches\n")
            f.write(f"- `data/` - Original weekly and daily data used\n")
            f.write(f"- `patterns/` - Normalized pattern data\n")
        
        return results_folder, len(match_data)
        
    except Exception as e:
        return None, f"Error saving results: {str(e)}"

# --- Streamlit App ---

st.set_page_config(page_title="Advanced Fractal Pattern Explorer", layout="wide")
st.title("Advanced Fractal Pattern Explorer")
st.write("""
**Features:**  
- Upload your daily OHLCV CSV OR fetch data directly from Yahoo Finance using a ticker symbol.
- Search for fractal/self-similar patterns in daily and weekly history that echo the *whole monthly* chart, or match any patterns you upload (pattern library).
- Supports cosine similarity or DTW (dynamic time warping, for stretched patterns).
- Optionally enable inverted pattern search.
- Visualize and export all matches!
""")

# --- Data Input Options ---
st.subheader("Data Input")
data_source = st.radio("Choose data source:", ["Upload Data File", "Yahoo Finance Ticker", "Database"], index=1, key="data_source_radio")

df = None

# Initialize session state for data persistence
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_config' not in st.session_state:
    st.session_state.search_config = None

# Add a flag to track if we're in a rerun to prevent unnecessary clearing
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.initial_load = True
else:
    st.session_state.initial_load = False

if data_source == "Upload Data File":
    uploaded_file = st.file_uploader("Upload daily OHLCV data file", type=['csv', 'txt'])
    
    if uploaded_file:
        try:
            # Auto-detect file format
            # First, try to read a sample to check the format
            sample_df = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)  # Reset file pointer
            
            # Check if it has standard format (with 'date' column)
            if 'date' in sample_df.columns and len(sample_df.columns) >= 5:
                # Standard format with date column
                df = pd.read_csv(uploaded_file, parse_dates=['date']).sort_values('date')
                df = df.set_index('date')
                st.info("📄 Detected standard format (with date column)")
                
            elif len(sample_df.columns) >= 10:
                # Custom format: TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT
                df = pd.read_csv(uploaded_file, header=None, names=['ticker', 'period', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'openint'])
                
                # Check if first row contains placeholders like <DATE>, <TICKER>, etc.
                if df.iloc[0]['date'] == '<DATE>' or str(df.iloc[0]['date']).startswith('<'):
                    # Remove the header row with placeholders
                    df = df.iloc[1:].reset_index(drop=True)
                    st.info("🔧 Removed placeholder header row")
                
                # Get ticker info after processing (after removing placeholder header if present)
                ticker_name = df.iloc[0]['ticker'] if len(df) > 0 else "Unknown"
                
                # Convert date to datetime - handle both YYYYMMDD format and other date formats
                try:
                    # First try standard YYYYMMDD format
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                except:
                    # If that fails, try parsing as generic date format
                    df['date'] = pd.to_datetime(df['date'])
                
                # Ensure numeric columns are properly converted
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Select and rename columns to match expected format
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                df = df.sort_values('date').set_index('date')
                
                st.info(f"📄 Detected custom format (TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT)")
                st.info(f"📊 Loaded data for ticker: {ticker_name}")
                
            else:
                st.error("❌ Unable to detect file format. Please ensure your file has either:")
                st.write("• Standard format: date, open, high, low, close, volume columns")
                st.write("• Custom format: TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT")
                df = None
            
            if df is not None:
                st.success(f"✅ Successfully loaded {len(df)} days of data from data file")
                
                # Store in session state
                st.session_state.loaded_data = df
                st.session_state.data_source_type = "Upload Data File"
                st.session_state.data_info = f"{len(df)} days from uploaded file"
                
                # Clear search results when new data is loaded
                st.session_state.search_results = None
                st.session_state.search_config = None
            
        except Exception as e:
            st.error(f"Error reading data file: {str(e)}")
            st.info("Please ensure your file matches the selected format")
            df = None

elif data_source == "Yahoo Finance Ticker":
    col1, col2 = st.columns(2)
    with col1:
        ticker_symbol = st.text_input("Enter ticker symbol (e.g., AAPL, MSFT, GOOGL)", value="TKOMY", key="yahoo_ticker")
    with col2:
        fetch_button = st.button("Fetch Data", key="yahoo_fetch")
    
    # Auto-load cached data if available
    if ticker_symbol:
        yahoo_data_dir = "yahoo_data"
        csv_filename = os.path.join(yahoo_data_dir, f"{ticker_symbol.upper()}.csv")
        
        if os.path.exists(csv_filename):
            # Load from local file automatically
            df = pd.read_csv(csv_filename, parse_dates=['date'], index_col='date')
            st.info(f"📁 Using cached data: {len(df)} days for {ticker_symbol.upper()}")
            
            # Store in session state
            st.session_state.loaded_data = df
            st.session_state.data_source_type = "Yahoo Finance Ticker"
            st.session_state.data_info = f"{len(df)} days for {ticker_symbol.upper()} (cached)"
            
            # Don't clear search results for cached data - it's the same data
        elif fetch_button:
            # Only fetch new data when button is clicked
            with st.spinner(f"Fetching all available data for {ticker_symbol}..."):
                try:
                    # Create yahoo_data folder if it doesn't exist
                    os.makedirs(yahoo_data_dir, exist_ok=True)
                    
                    # Fetch all available data from Yahoo Finance
                    ticker_data = yf.download(ticker_symbol, period="max", progress=False)
                    
                    if ticker_data.empty or len(ticker_data) == 0:
                        st.error(f"No data found for ticker {ticker_symbol}. Please check the symbol and try again.")
                    elif not isinstance(ticker_data, pd.DataFrame):
                        st.error(f"Invalid data format returned for ticker {ticker_symbol}. The ticker may not exist or may be delisted.")
                    else:
                        # Handle multi-level columns that sometimes occur with yfinance
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            # Flatten multi-level columns
                            ticker_data.columns = [col[0] if isinstance(col, tuple) else col for col in ticker_data.columns]
                        
                        # Validate that we have the required columns
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols = [col for col in required_cols if col not in ticker_data.columns]
                        
                        if missing_cols:
                            st.error(f"Missing required columns for {ticker_symbol}: {missing_cols}")
                        elif len(ticker_data.columns) < 5:
                            st.error(f"Incomplete data returned for ticker {ticker_symbol}. Missing required OHLCV columns.")
                        else:
                            # Convert to the expected format with proper error handling
                            try:
                                df = pd.DataFrame({
                                    'open': ticker_data['Open'],
                                    'high': ticker_data['High'],
                                    'low': ticker_data['Low'],
                                    'close': ticker_data['Close'],
                                    'volume': ticker_data['Volume']
                                })
                                df.index = ticker_data.index
                                df.index.name = 'date'
                                
                                # Additional validation - check for valid numeric data
                                if df['close'].isna().all() or len(df) < 30:
                                    st.error(f"Insufficient or invalid price data for {ticker_symbol}. Need at least 30 days of valid data.")
                                else:
                                    # Save to CSV file
                                    df.to_csv(csv_filename)
                                    st.success(f"✅ Successfully fetched and saved {len(df)} days of data for {ticker_symbol}")
                                    
                                    # Store in session state
                                    st.session_state.loaded_data = df
                                    st.session_state.data_source_type = "Yahoo Finance Ticker"
                                    st.session_state.data_info = f"{len(df)} days for {ticker_symbol.upper()} (fetched)"
                                    
                                    # Clear search results when new data is fetched
                                    st.session_state.search_results = None
                                    st.session_state.search_config = None
                            except Exception as conversion_error:
                                st.error(f"Error converting data format for {ticker_symbol}: {str(conversion_error)}")
                        
                except Exception as e:
                    st.error(f"Error fetching data for {ticker_symbol}: {str(e)}. Please verify the ticker symbol is correct and try again.")
        elif ticker_symbol and not os.path.exists(csv_filename):
            st.warning(f"💾 No cached data found for {ticker_symbol.upper()}. Click 'Fetch Data' to download from Yahoo Finance.")

elif data_source == "Database":
    col1, col2 = st.columns(2)
    with col1:
        db_ticker_symbol = st.text_input("Enter ticker symbol for database lookup (e.g., AAPL, MSFT, GOOGL)", value="AAPL", key="db_ticker")
    with col2:
        load_db_button = st.button("Load from Database", key="db_load")
    
    if load_db_button and db_ticker_symbol:
        with st.spinner(f"Loading {db_ticker_symbol} data from database..."):
            try:
                # Create database connection using .env configuration
                engine = create_engine(
                    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
                )
                
                # Query to get stock data for the ticker
                query = """
                    SELECT date, open_price as open, high_price as high, 
                           low_price as low, close_price as close, volume
                    FROM stock_data 
                    WHERE ticker = %(ticker)s 
                    ORDER BY date ASC
                """
                
                # Execute query and load data - append .US to match database format
                ticker_with_suffix = f"{db_ticker_symbol.upper()}.US"
                df = pd.read_sql_query(query, engine, params={'ticker': ticker_with_suffix}, parse_dates=['date'])
                
                if df.empty:
                    # Clean ticker name for error display (remove .US suffix)
                    display_ticker = ticker_with_suffix.replace(".US", "")
                    st.error(f"No data found for ticker {display_ticker} in the database.")
                    
                    # Show available tickers for reference
                    try:
                        ticker_query = "SELECT DISTINCT ticker FROM stock_data ORDER BY ticker LIMIT 20"
                        available_tickers = pd.read_sql_query(ticker_query, engine)
                        if not available_tickers.empty:
                            # Clean ticker names for display
                            clean_tickers = [t.replace(".US", "") for t in available_tickers['ticker'].head(10).tolist()]
                            st.info(f"💡 Some available tickers in database: {', '.join(clean_tickers)}")
                    except Exception:
                        pass
                else:
                    df = df.set_index('date')
                    # Clean ticker name for display (remove .US suffix)
                    display_ticker = ticker_with_suffix.replace(".US", "")
                    st.success(f"📊 Successfully loaded {len(df)} days of data for {display_ticker} from database")
                    
                    # Store in session state
                    st.session_state.loaded_data = df
                    st.session_state.data_source_type = "Database"
                    st.session_state.data_info = f"{len(df)} days for {display_ticker} from database"
                    
                    # Clear search results when new data is loaded from database
                    st.session_state.search_results = None
                    st.session_state.search_config = None
                    
                engine.dispose()
                
            except Exception as e:
                st.error(f"Database connection error: {str(e)}")
                st.info(f"💡 Check database connection. Using config: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    elif db_ticker_symbol and not load_db_button:
        st.info(f"💾 Click 'Load from Database' to fetch {db_ticker_symbol.upper()} data from PostgreSQL database.")

# Use session state data if available and no new data was loaded
if df is None and st.session_state.loaded_data is not None:
    df = st.session_state.loaded_data
    if st.session_state.data_source_type and st.session_state.data_info:
        st.info(f"📋 Using previously loaded data: {st.session_state.data_info}")

# Data preview section (expandable, no reruns) - placed after data loading
with st.expander("📊 Data Preview (click to expand)", expanded=False):
    if df is not None:
        st.subheader("📊 Raw Data")
        st.dataframe(df.head(10))

        # --- Create monthly and weekly resamples ---
        st.subheader("Auto-Generated Resamples")
        monthly_preview = resample_ohlc(df, 'M')
        weekly_preview = resample_ohlc(df, 'W')
        st.write("Monthly (for reference pattern):")
        st.dataframe(monthly_preview.head())
        st.write("Weekly (for pattern search):")
        st.dataframe(weekly_preview.head())
        
        # Show data statistics
        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days", len(df))
            st.metric("Date Range", f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        with col2:
            st.metric("Weekly Periods", len(weekly_preview))
            st.metric("Monthly Periods", len(monthly_preview))
        with col3:
            try:
                price_min = df['close'].min()
                price_max = df['close'].max()
                latest_price = df['close'].iloc[-1]
                st.metric("Price Range", f"${price_min:.2f} - ${price_max:.2f}")
                st.metric("Latest Price", f"${latest_price:.2f}")
            except (ValueError, TypeError) as e:
                st.metric("Price Range", "Data Error")
                st.metric("Latest Price", "Data Error")
                st.error(f"Price data formatting error: {str(e)}")
        
        # Optional debug info (collapsed by default)
        with st.expander("🔧 Debug Info", expanded=False):
            st.write(f"**Loaded Data:** {'✅ Present' if st.session_state.loaded_data is not None else '❌ None'}")
            st.write(f"**Data Source:** {st.session_state.data_source_type if st.session_state.data_source_type else '❌ None'}")
            st.write(f"**Data Info:** {st.session_state.data_info if st.session_state.data_info else '❌ None'}")
            st.write(f"**Search Results:** {'✅ Present' if st.session_state.search_results is not None else '❌ None'}")
            st.write(f"**Search Config:** {'✅ Present' if st.session_state.search_config is not None else '❌ None'}")
            if st.session_state.search_results is not None:
                results = st.session_state.search_results
                st.write(f"**Weekly Matches:** {len(results['matches_weekly'])}")
                st.write(f"**Daily Matches:** {len(results['matches_daily'])}")
            
            # Add a refresh counter to see if the app is restarting unexpectedly
            if 'page_refresh_count' not in st.session_state:
                st.session_state.page_refresh_count = 0
            st.session_state.page_refresh_count += 1
            st.write(f"**Page Refresh Count:** {st.session_state.page_refresh_count}")
            
            # Show current widget states
            st.write(f"**Current Data Source:** {data_source}")
            st.write(f"**Initial Load:** {'✅ Yes' if st.session_state.get('initial_load', False) else '❌ No (Rerun)'}")
            st.write(f"**App Initialized:** {'✅ Yes' if st.session_state.get('app_initialized', False) else '❌ No'}")
    else:
        st.write("**No data loaded yet**")
        st.info("💡 Load data using one of the options above to see preview and statistics")

if df is not None:
    # --- Pattern Analysis (always visible) ---
    # Create resamples for analysis
    monthly = resample_ohlc(df, 'M')
    weekly = resample_ohlc(df, 'W')

    # --- Pattern Selection: Whole history or upload pattern library ---
    st.sidebar.header("Reference Patterns")
    
    # Preserve pattern library setting if we have stored search config
    default_use_library = False
    if st.session_state.search_config is not None:
        # Check if previous search used custom patterns
        stored_patterns = st.session_state.search_results.get('patterns', [])
        if len(stored_patterns) > 0 and stored_patterns[0][0] != "Monthly Whole History":
            default_use_library = True
    
    use_library = st.sidebar.checkbox("Use pattern library (upload)", value=default_use_library)
    pattern_files = []
    patterns = []
    if use_library:
        pattern_files = st.sidebar.file_uploader(
            "Upload pattern library CSV(s) (each with 'close' column)", type=['csv'], accept_multiple_files=True)
        for pf in pattern_files:
            p_df = pd.read_csv(pf)
            pat = normalize_and_resample(p_df['close'].values, target_length=len(p_df))
            patterns.append((pf.name, pat))
        if not patterns:
            st.warning("No pattern files uploaded. Using whole-history monthly as default.")
            patterns = [("Monthly Whole History", normalize_and_resample(monthly['close'].values, target_length=len(monthly)))]
    else:
        patterns = [("Monthly Whole History", normalize_and_resample(monthly['close'].values, target_length=len(monthly)))]

    # --- Pattern Mining Params ---
    st.sidebar.header("Pattern Mining Parameters")
    
    # Initialize search parameters from session state if available
    if st.session_state.search_config is not None:
        default_threshold = st.session_state.search_config['similarity_threshold']
        default_min_window = st.session_state.search_config['min_window_monthly']
        default_max_multiplier = st.session_state.search_config['max_window_multiplier']
        default_method = st.session_state.search_config['match_method']
        default_inversion = st.session_state.search_config['allow_inversion']
    else:
        default_threshold = 0.93
        default_min_window = 8
        default_max_multiplier = 3
        default_method = 'cosine'
        default_inversion = False
    
    similarity_threshold = st.sidebar.slider("Similarity threshold", 0.80, 0.99, default_threshold, 0.01)
    min_window_monthly = st.sidebar.slider("Min window size (weeks/days)", 4, 30, default_min_window)
    max_window_multiplier = st.sidebar.slider("Max window size multiplier (x monthly length)", 1, 5, default_max_multiplier)
    match_method = st.sidebar.selectbox("Similarity method", ['cosine', 'dtw'], index=0 if default_method == 'cosine' else 1)
    use_dtw = (match_method == 'dtw')
    allow_inversion = st.sidebar.checkbox("Enable inverted pattern search", value=default_inversion)
    
    # Search button
    search_button = st.sidebar.button("🔍 Start Pattern Search", type="primary")
    
    # Clear results button (only show if we have results)
    if st.session_state.search_results is not None:
        clear_button = st.sidebar.button("🗑️ Clear Results")
        if clear_button:
            st.session_state.search_results = None
            st.session_state.search_config = None
            # No need to rerun - Streamlit will handle the update automatically
    
    # --- Define windows ---
    ref_len = len(patterns[0][1])
    window_lengths_weekly = range(
        int(min_window_monthly),
        int(ref_len*max_window_multiplier)+1
    )
    window_lengths_daily = range(
        int(min_window_monthly*5),
        int(ref_len*max_window_multiplier*5)+1, max(1, ref_len//2)
    )
    
    # Show search parameters
    st.subheader("Search Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reference Pattern Length", ref_len)
        st.metric("Similarity Method", match_method.upper())
    with col2:
        st.metric("Weekly Search Window", f"{min(window_lengths_weekly)} - {max(window_lengths_weekly)}")
        st.metric("Similarity Threshold", f"{similarity_threshold:.2f}")
    with col3:
        st.metric("Daily Search Window", f"{min(window_lengths_daily)} - {max(window_lengths_daily)}")
        st.metric("Inverted Search", "Enabled" if allow_inversion else "Disabled")
    
    # --- Run Search only when button is clicked ---
    if search_button:
        with st.spinner("Finding fractal pattern matches..."):
            matches_weekly = advanced_slide_and_compare(
                weekly['close'].values, patterns, window_lengths_weekly, 
                threshold=similarity_threshold, 
                method=match_method, allow_inversion=allow_inversion, use_dtw=use_dtw
            )
            matches_daily = advanced_slide_and_compare(
                df['close'].values, patterns, window_lengths_daily, 
                threshold=similarity_threshold, 
                method=match_method, allow_inversion=allow_inversion, use_dtw=use_dtw
            )
            
            # Store search results and configuration in session state
            st.session_state.search_results = {
                'matches_weekly': matches_weekly,
                'matches_daily': matches_daily,
                'weekly_data': weekly,
                'daily_data': df,
                'patterns': patterns
            }
            st.session_state.search_config = {
                'similarity_threshold': similarity_threshold,
                'match_method': match_method,
                'allow_inversion': allow_inversion,
                'min_window_monthly': min_window_monthly,
                'max_window_multiplier': max_window_multiplier
            }
        
        st.success(f"Found {len(matches_weekly)} weekly matches and {len(matches_daily)} daily matches.")

    # Check if we have stored search results to display
    if st.session_state.search_results is not None:
        matches_weekly = st.session_state.search_results['matches_weekly']
        matches_daily = st.session_state.search_results['matches_daily']
        patterns = st.session_state.search_results['patterns']
        weekly = st.session_state.search_results['weekly_data']
        daily_data = st.session_state.search_results['daily_data']
        
        # Show success message if we just ran a search
        if search_button:
            pass  # Already shown above
        else:
            st.info(f"📊 Displaying previous search results: {len(matches_weekly)} weekly matches and {len(matches_daily)} daily matches.")
            
            # Show configuration used for these results
            config = st.session_state.search_config
            st.write(f"**Previous search config:** Threshold={config['similarity_threshold']:.2f}, "
                    f"Method={config['match_method'].upper()}, "
                    f"Window={config['min_window_monthly']}-{config['max_window_multiplier']}x, "
                    f"Inversion={'ON' if config['allow_inversion'] else 'OFF'}")
    
    # Only show results section if we have search results
    if st.session_state.search_results is not None:

        # --- Show Results ---
        def plot_matches(series, matches, patterns, dates, n=3, title=''):
            shown = 0
            by_pattern = {}
            for m in matches:
                by_pattern.setdefault(m['pattern'], []).append(m)
            for pat_name in by_pattern:
                st.markdown(f"#### {title} - Pattern: **{pat_name}**")
                sorted_matches = sorted(by_pattern[pat_name], key=lambda x: -x['similarity'])
                for i, m in enumerate(sorted_matches[:n]):
                    shown += 1
                    fig, ax = plt.subplots(figsize=(12,4))
                    pattern = [p[1] for p in patterns if p[0]==pat_name][0]
                    ax.plot(range(len(pattern)), pattern, 'k-', lw=3, label='Reference pattern')
                    if m['inverted']:
                        ax.plot(range(len(pattern)), -pattern, 'c--', lw=2, label='Inverted reference')
                        ax.plot(range(len(pattern)), m['window'], 'm-', alpha=0.7, label=f'Inverted Match (sim={m["similarity"]:.2f})')
                    else:
                        ax.plot(range(len(pattern)), m['window'], 'r-', alpha=0.7, label=f'Match (sim={m["similarity"]:.2f})')
                    ax.set_title(f'{title} | {pat_name} | Match {i+1}: {dates[m["start"]]} to {dates[m["start"]+m["size"]-1]}, sim={m["similarity"]:.2f}{" (Inverted)" if m["inverted"] else ""}')
                    ax.legend()
                    st.pyplot(fig)
                    if st.checkbox(f"Show actual chart for {title} {pat_name} Match {i+1}", value=False, key=f"{title}_{pat_name}_{i}"):
                        actual_fig, ax2 = plt.subplots(figsize=(12,3))
                        idx_range = range(m["start"], m["start"]+m["size"])
                        ax2.plot(dates[idx_range], series[idx_range], 'b-')
                        ax2.set_title(f"{title} actual price, {dates[m['start']]} to {dates[m['start']+m['size']-1]}")
                        st.pyplot(actual_fig)
                    if shown >= n * len(by_pattern):
                        break

        st.subheader("Top Weekly Matches")
        plot_matches(weekly['close'].values, matches_weekly, patterns, weekly.index, n=5, title="Weekly")
        st.subheader("Top Daily Matches")
        plot_matches(daily_data['close'].values, matches_daily, patterns, daily_data.index, n=5, title="Daily")

        # Export matches
        st.subheader("Download Match Data")
        match_data = pd.DataFrame([
            {
                'tf': 'weekly',
                'pattern': m['pattern'],
                'start': m['start'],
                'size': m['size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': weekly.index[m['start']],
                'end_date': weekly.index[m['start']+m['size']-1]
            }
            for m in matches_weekly
        ] + [
            {
                'tf': 'daily',
                'pattern': m['pattern'],
                'start': m['start'],
                'size': m['size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': daily_data.index[m['start']],
                'end_date': daily_data.index[m['start']+m['size']-1]
            }
            for m in matches_daily
        ])
        st.dataframe(match_data.head(10))
        csv = match_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download matches as CSV", csv, "fractal_matches.csv", "text/csv")
        
        # Save to folder button
        st.subheader("Save Complete Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Results to Folder", type="secondary"):
                with st.spinner("Saving complete results to folder..."):
                    folder_path, result = save_search_results_to_folder(
                        st.session_state.search_results, 
                        st.session_state.search_config,
                        st.session_state.data_info
                    )
                    if folder_path:
                        st.success(f"✅ Results saved successfully!")
                        st.info(f"📁 **Folder:** `{folder_path}`")
                        st.info(f"📊 **Files saved:** {result} matches + charts + data + config")
                        
                        # Show folder contents
                        st.write("**Folder contents:**")
                        st.write("- 📋 `all_matches.csv` - Complete match data")
                        st.write("- ⚙️ `search_config.json` - Search configuration")
                        st.write("- 📈 `charts/` - Pattern and price charts")
                        st.write("- 📊 `data/` - Weekly and daily data")
                        st.write("- 🧩 `patterns/` - Pattern definitions")
                        st.write("- 📖 `README.md` - Summary report")
                    else:
                        st.error(f"❌ Failed to save: {result}")
        with col2:
            st.info("💡 **Saves:**\n- All match data\n- Top pattern charts\n- Raw data\n- Configuration\n- Summary report")
    else:
        st.info("💡 Configure your search parameters in the sidebar and click '🔍 Start Pattern Search' to begin analysis.")

st.markdown("""
---
**Tips:**  
- For your pattern library, upload CSVs with a `close` column, one file per pattern.
- Try DTW for time-warped/“stretched” pattern matching, or cosine for fast, direct shape matching.
- Use a lower similarity threshold for more matches; higher for only the closest.
- Results show where in your daily/weekly history the reference pattern (or its inverse) recurs in different sizes and positions—true fractal echoes!
""")
