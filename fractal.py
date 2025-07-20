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

def filter_overlapping_matches(matches, series_dates, min_separation_days=30):
    """
    Filter out overlapping or close-proximity matches, keeping only the best similarity from each cluster.
    
    Args:
        matches: List of match dictionaries
        series_dates: DatetimeIndex of the series
        min_separation_days: Minimum days between matches to consider them separate
    
    Returns:
        Filtered list of matches with overlaps removed
    """
    if not matches:
        return matches
    
    # Sort matches by similarity (highest first)
    sorted_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    filtered_matches = []
    
    for match in sorted_matches:
        match_start_date = series_dates[match['start']]
        match_end_date = series_dates[match['start'] + match['size'] - 1]
        
        # Check if this match overlaps or is too close to any already accepted match
        is_too_close = False
        for accepted_match in filtered_matches:
            accepted_start_date = series_dates[accepted_match['start']]
            accepted_end_date = series_dates[accepted_match['start'] + accepted_match['size'] - 1]
            
            # Calculate the gap between matches
            if match_end_date < accepted_start_date:
                # Current match is before accepted match
                gap_days = (accepted_start_date - match_end_date).days
            elif accepted_end_date < match_start_date:
                # Current match is after accepted match
                gap_days = (match_start_date - accepted_end_date).days
            else:
                # Matches overlap
                gap_days = 0
            
            # If gap is less than minimum separation, consider them too close
            if gap_days < min_separation_days:
                is_too_close = True
                break
        
        # Only add this match if it's not too close to existing matches
        if not is_too_close:
            filtered_matches.append(match)
    
    return filtered_matches

def advanced_slide_and_compare(series, patterns, window_sizes, threshold=0.95, method='cosine', allow_inversion=False, use_dtw=False, series_dates=None, exclude_overlap=True, filter_close_matches=True, min_separation_days=30):
    matches = []
    for pat_name, pattern in patterns:
        pat_len = len(pattern)
        for win_size in window_sizes:
            for start in range(len(series) - win_size + 1):
                window = series[start:start+win_size]
                window_norm = normalize_and_resample(window, pat_len)
                
                # Skip self-matches when using "Monthly Whole History" pattern
                if exclude_overlap and pat_name == "Monthly Whole History" and series_dates is not None:
                    # Get the date range of the current window
                    window_start_date = series_dates[start]
                    window_end_date = series_dates[start + win_size - 1]
                    
                    # For monthly pattern, exclude matches that significantly overlap with monthly boundaries
                    # This prevents the system from matching a monthly period to itself
                    window_start_month = window_start_date.replace(day=1)
                    window_end_month = (window_end_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                    
                    # Skip if this window represents a significant portion of any single month
                    # (to avoid matching monthly patterns to themselves)
                    if win_size >= 20:  # For weekly data, 4+ weeks might overlap significantly with monthly
                        # Calculate overlap with month boundaries
                        total_days = (window_end_date - window_start_date).days + 1
                        if total_days >= 25:  # If window covers most of a month, likely a self-match
                            continue
                
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
    
    # Filter out close-proximity matches if requested and dates are available
    if filter_close_matches and series_dates is not None and matches:
        matches = filter_overlapping_matches(matches, series_dates, min_separation_days)
    
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

def save_search_results_cache(search_results, search_config, data_info, cache_folder="fractal_cache"):
    """Save search results to cache for quick reload"""
    try:
        os.makedirs(cache_folder, exist_ok=True)
        
        # Create cache filename based on data source and config
        cache_key = f"{data_info}_{search_config['similarity_threshold']}_{search_config['match_method']}_{search_config['allow_inversion']}"
        # Clean filename
        cache_key = "".join(c for c in cache_key if c.isalnum() or c in (' ', '_')).rstrip()
        cache_key = cache_key.replace(' ', '_')
        cache_filename = os.path.join(cache_folder, f"{cache_key}.json")
        
        # Prepare data for JSON serialization
        cache_data = {
            'search_results': {
                'matches_weekly': [
                    {
                        'pattern': m['pattern'],
                        'start': m['start'],
                        'size': m['size'],
                        'similarity': m['similarity'],
                        'window': m['window'].tolist() if isinstance(m['window'], np.ndarray) else m['window'],
                        'inverted': m['inverted']
                    } for m in search_results['matches_weekly']
                ],
                'matches_daily': [
                    {
                        'pattern': m['pattern'],
                        'start': m['start'],
                        'size': m['size'],
                        'similarity': m['similarity'],
                        'window': m['window'].tolist() if isinstance(m['window'], np.ndarray) else m['window'],
                        'inverted': m['inverted']
                    } for m in search_results['matches_daily']
                ],
                'patterns': [(name, pattern.tolist()) for name, pattern in search_results['patterns']]
            },
            'search_config': search_config,
            'data_info': data_info,
            'timestamp': datetime.now().isoformat(),
            'weekly_data_shape': search_results['weekly_data'].shape,
            'daily_data_shape': search_results['daily_data'].shape
        }
        
        with open(cache_filename, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
            
        return cache_filename
    except Exception as e:
        return None

def load_search_results_cache(data_info, search_config, cache_folder="fractal_cache"):
    """Load search results from cache if available and matching"""
    try:
        if not os.path.exists(cache_folder):
            return None
            
        # Create cache key to look for
        cache_key = f"{data_info}_{search_config['similarity_threshold']}_{search_config['match_method']}_{search_config['allow_inversion']}"
        cache_key = "".join(c for c in cache_key if c.isalnum() or c in (' ', '_')).rstrip()
        cache_key = cache_key.replace(' ', '_')
        cache_filename = os.path.join(cache_folder, f"{cache_key}.json")
        
        if not os.path.exists(cache_filename):
            return None
            
        with open(cache_filename, 'r') as f:
            cache_data = json.load(f)
            
        # Verify the cache matches current config
        if (cache_data['data_info'] == data_info and 
            cache_data['search_config'] == search_config):
            
            # Reconstruct search results
            cached_results = cache_data['search_results']
            cached_results['patterns'] = [(name, np.array(pattern)) for name, pattern in cached_results['patterns']]
            
            # Convert window arrays back to numpy arrays for both weekly and daily matches
            for match in cached_results['matches_weekly']:
                if 'window' in match and isinstance(match['window'], list):
                    match['window'] = np.array(match['window'])
            
            for match in cached_results['matches_daily']:
                if 'window' in match and isinstance(match['window'], list):
                    match['window'] = np.array(match['window'])
            
            return {
                'search_results': cached_results,
                'search_config': cache_data['search_config'],
                'timestamp': cache_data['timestamp']
            }
        else:
            return None
            
    except Exception as e:
        return None

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
        st.markdown("<br>", unsafe_allow_html=True)  # Natural spacing to align with text input
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
        default_exclude_self = st.session_state.search_config.get('exclude_self_matches', True)
        default_filter_close = st.session_state.search_config.get('filter_close_matches', True)
        default_min_separation = st.session_state.search_config.get('min_separation_days', 30)
    else:
        default_threshold = 0.93
        default_min_window = 8
        default_max_multiplier = 3
        default_method = 'cosine'
        default_inversion = False
        default_exclude_self = True
        default_filter_close = True
        default_min_separation = 30
    
    similarity_threshold = st.sidebar.slider("Similarity threshold", 0.80, 0.99, default_threshold, 0.01)
    min_window_monthly = st.sidebar.slider("Min window size (weeks/days)", 4, 30, default_min_window)
    max_window_multiplier = st.sidebar.slider("Max window size multiplier (x monthly length)", 1, 5, default_max_multiplier)
    match_method = st.sidebar.selectbox("Similarity method", ['cosine', 'dtw'], index=0 if default_method == 'cosine' else 1)
    use_dtw = (match_method == 'dtw')
    allow_inversion = st.sidebar.checkbox("Enable inverted pattern search", value=default_inversion)
    exclude_self_matches = st.sidebar.checkbox("Exclude self-matches", value=default_exclude_self, 
                                             help="Prevents matching patterns to themselves (recommended for Monthly Whole History)")
    filter_close_matches = st.sidebar.checkbox("Filter close-proximity matches", value=default_filter_close,
                                              help="Keep only the best match from each time cluster (recommended for cleaner results)")
    
    if filter_close_matches:
        min_separation_days = st.sidebar.slider("Minimum separation (days)", 7, 90, default_min_separation, 7,
                                               help="Minimum days between matches to consider them separate")
    else:
        min_separation_days = 30  # Default value when filtering is disabled
    
    # Search button
    search_button = st.sidebar.button("🔍 Start Pattern Search", type="primary")
    
    # Check for cached results if no current results exist
    if st.session_state.search_results is None and st.session_state.data_info is not None:
        # Build current search config to check cache
        current_config = {
            'similarity_threshold': similarity_threshold,
            'match_method': match_method,
            'allow_inversion': allow_inversion,
            'min_window_monthly': min_window_monthly,
            'max_window_multiplier': max_window_multiplier,
            'exclude_self_matches': exclude_self_matches,
            'filter_close_matches': filter_close_matches,
            'min_separation_days': min_separation_days
        }
        
        cached_data = load_search_results_cache(st.session_state.data_info, current_config)
        if cached_data is not None:
            # Load cached results
            st.session_state.search_results = {
                'matches_weekly': cached_data['search_results']['matches_weekly'],
                'matches_daily': cached_data['search_results']['matches_daily'],
                'weekly_data': weekly,
                'daily_data': df,
                'patterns': cached_data['search_results']['patterns']
            }
            st.session_state.search_config = cached_data['search_config']
            
            # Store cache timestamp for display
            cache_time = datetime.fromisoformat(cached_data['timestamp'])
            st.session_state.cache_loaded_timestamp = cache_time.strftime('%Y-%m-%d %H:%M')
            
            # Show cache load message
            st.sidebar.success(f"📁 Loaded cached results from {cache_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Show quick preview in sidebar
            matches_weekly = cached_data['search_results']['matches_weekly']
            matches_daily = cached_data['search_results']['matches_daily']
            st.sidebar.info(f"📊 {len(matches_weekly)} weekly + {len(matches_daily)} daily matches found")
    
    # Clear results button (only show if we have results)
    if st.session_state.search_results is not None:
        clear_button = st.sidebar.button("🗑️ Clear Results")
        if clear_button:
            st.session_state.search_results = None
            st.session_state.search_config = None
            # No need to rerun - Streamlit will handle the update automatically
    
    # Cache management section
    with st.sidebar.expander("📁 Cache Management", expanded=False):
        cache_folder = "fractal_cache"
        if os.path.exists(cache_folder):
            cache_files = [f for f in os.listdir(cache_folder) if f.endswith('.json')]
            if cache_files:
                st.write(f"**Cached searches:** {len(cache_files)}")
                
                # Show cache files with timestamps
                for cache_file in cache_files[:5]:  # Show max 5 recent
                    try:
                        with open(os.path.join(cache_folder, cache_file), 'r') as f:
                            cache_data = json.load(f)
                        cache_time = datetime.fromisoformat(cache_data['timestamp'])
                        data_source = cache_data['data_info'].split(' ')[-1] if cache_data['data_info'] else "unknown"
                        st.text(f"• {data_source} - {cache_time.strftime('%m/%d %H:%M')}")
                    except:
                        pass
                
                # Clear cache button
                if st.button("🗑️ Clear All Cache"):
                    try:
                        for cache_file in cache_files:
                            os.remove(os.path.join(cache_folder, cache_file))
                        st.success("Cache cleared!")
                    except Exception as e:
                        st.error(f"Error clearing cache: {str(e)}")
            else:
                st.write("No cached searches found")
        else:
            st.write("No cache folder found")
    
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
        # Clear cache timestamp since this is a new search
        if hasattr(st.session_state, 'cache_loaded_timestamp'):
            delattr(st.session_state, 'cache_loaded_timestamp')
            
        with st.spinner("Finding fractal pattern matches..."):
            matches_weekly = advanced_slide_and_compare(
                weekly['close'].values, patterns, window_lengths_weekly, 
                threshold=similarity_threshold, 
                method=match_method, allow_inversion=allow_inversion, use_dtw=use_dtw,
                series_dates=weekly.index, exclude_overlap=exclude_self_matches,
                filter_close_matches=filter_close_matches, min_separation_days=min_separation_days
            )
            matches_daily = advanced_slide_and_compare(
                df['close'].values, patterns, window_lengths_daily, 
                threshold=similarity_threshold, 
                method=match_method, allow_inversion=allow_inversion, use_dtw=use_dtw,
                series_dates=df.index, exclude_overlap=exclude_self_matches,
                filter_close_matches=filter_close_matches, min_separation_days=min_separation_days
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
                'max_window_multiplier': max_window_multiplier,
                'exclude_self_matches': exclude_self_matches,
                'filter_close_matches': filter_close_matches,
                'min_separation_days': min_separation_days
            }
            
            # Save results to cache for quick reload
            cache_file = save_search_results_cache(
                st.session_state.search_results, 
                st.session_state.search_config, 
                st.session_state.data_info
            )
            if cache_file:
                st.sidebar.success("💾 Results cached for quick reload")
            
            # Automatically save complete results to folder
            folder_path, result = save_search_results_to_folder(
                st.session_state.search_results, 
                st.session_state.search_config,
                st.session_state.data_info
            )
            if folder_path:
                st.success(f"✅ Found {len(matches_weekly)} weekly matches and {len(matches_daily)} daily matches.")
                st.info(f"📁 **Results automatically saved to:** `{folder_path}`")
                st.info(f"📊 **Files saved:** {result} matches + charts + data + config")
            else:
                st.success(f"Found {len(matches_weekly)} weekly matches and {len(matches_daily)} daily matches.")
                st.warning(f"⚠️ Could not save results to folder: {result}")

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
            # Check if these results came from cache
            cache_info = ""
            if hasattr(st.session_state, 'cache_loaded_timestamp'):
                cache_time = st.session_state.cache_loaded_timestamp
                cache_info = f" (loaded from cache at {cache_time})"
            
            st.info(f"📊 Displaying search results: {len(matches_weekly)} weekly matches and {len(matches_daily)} daily matches{cache_info}.")
            
            # Show configuration used for these results
            config = st.session_state.search_config
            proximity_info = f" ({config.get('min_separation_days', 30)}d)" if config.get('filter_close_matches', True) else ""
            st.write(f"**Search config:** Threshold={config['similarity_threshold']:.2f}, "
                    f"Method={config['match_method'].upper()}, "
                    f"Window={config['min_window_monthly']}-{config['max_window_multiplier']}x, "
                    f"Inversion={'ON' if config['allow_inversion'] else 'OFF'}, "
                    f"Self-match exclusion={'ON' if config.get('exclude_self_matches', True) else 'OFF'}, "
                    f"Proximity filter={'ON' if config.get('filter_close_matches', True) else 'OFF'}{proximity_info}")
    
    # Only show results section if we have search results
    if st.session_state.search_results is not None:

        # --- Show Results ---
        # Create a simplified summary view instead of detailed charts
        st.subheader("Match Summary")
        st.write("Click on any row in the table below to view detailed charts for that match.")

        # Export matches
        st.subheader("Interactive Match Results")
        
        # Prepare match data with additional info for display
        all_matches = []
        for i, m in enumerate(matches_weekly):
            all_matches.append({
                'index': i,
                'timeframe': 'weekly',
                'pattern': m['pattern'],
                'start_index': m['start'],
                'window_size': m['size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': weekly.index[m['start']],
                'end_date': weekly.index[m['start']+m['size']-1],
                'match_data': m,
                'series': weekly['close'].values,
                'dates': weekly.index
            })
        
        for i, m in enumerate(matches_daily):
            all_matches.append({
                'index': i,
                'timeframe': 'daily',
                'pattern': m['pattern'],
                'start_index': m['start'],
                'window_size': m['size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': daily_data.index[m['start']],
                'end_date': daily_data.index[m['start']+m['size']-1],
                'match_data': m,
                'series': daily_data['close'].values,
                'dates': daily_data.index
            })
        
        # Sort by similarity (highest first)
        all_matches = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)
        
        # Create display dataframe
        match_data = pd.DataFrame([
            {
                'timeframe': m['timeframe'],
                'pattern': m['pattern'],
                'start_index': m['start_index'],
                'window_size': m['window_size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': m['start_date'],
                'end_date': m['end_date']
            }
            for m in all_matches
        ])
        
        # Show summary info
        weekly_count = len(matches_weekly)
        daily_count = len(matches_daily)
        total_count = len(match_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weekly Matches", weekly_count)
        with col2:
            st.metric("Daily Matches", daily_count)
        with col3:
            st.metric("Total Matches", total_count)
        
        # Date range filtering
        st.subheader("Filter by Date Range")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Get the earliest and latest dates from all matches (excluding recent matches)
            one_year_ago = datetime.now().date() - timedelta(days=365)
            historical_matches = [m for m in all_matches if m['end_date'].date() <= one_year_ago]
            
            if historical_matches:
                all_start_dates = [m['start_date'] for m in historical_matches]
                all_end_dates = [m['end_date'] for m in historical_matches]
                min_date = min(all_start_dates).date()
                max_date = max(all_end_dates).date()
            else:
                # Fallback if no historical matches
                min_date = datetime.now().date() - timedelta(days=3650)  # 10 years ago
                max_date = one_year_ago
            
            # Initialize session state variables with proper bounds checking
            if 'date_filter_start' not in st.session_state:
                st.session_state.date_filter_start = min_date
            if 'date_filter_end' not in st.session_state:
                st.session_state.date_filter_end = max_date
            if 'date_filter_active' not in st.session_state:
                st.session_state.date_filter_active = False
            
            # Ensure session state values are within bounds
            if st.session_state.date_filter_start < min_date:
                st.session_state.date_filter_start = min_date
            if st.session_state.date_filter_start > max_date:
                st.session_state.date_filter_start = max_date
            if st.session_state.date_filter_end < min_date:
                st.session_state.date_filter_end = min_date
            if st.session_state.date_filter_end > max_date:
                st.session_state.date_filter_end = max_date
                
            start_date_filter = st.date_input(
                "From date", 
                value=st.session_state.date_filter_start,
                min_value=min_date,
                max_value=max_date,
                key="start_date_filter"
            )
        
        with col2:
            # Ensure end date filter is also within bounds
            if st.session_state.date_filter_end < min_date:
                st.session_state.date_filter_end = min_date
            if st.session_state.date_filter_end > max_date:
                st.session_state.date_filter_end = max_date
                
            end_date_filter = st.date_input(
                "To date", 
                value=st.session_state.date_filter_end,
                min_value=min_date,
                max_value=max_date,
                key="end_date_filter"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Natural spacing to align with date inputs
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                clear_filter = st.button("Clear Filter", key="clear_date_filter")
            with btn_col2:
                apply_filter = st.button("Apply Filter", key="apply_date_filter", type="primary")
            
        # Handle button clicks
        if apply_filter:
            # Update session state with current filter values and activate filter
            st.session_state.date_filter_start = start_date_filter
            st.session_state.date_filter_end = end_date_filter
            st.session_state.date_filter_active = True
            st.rerun()
            
        if clear_filter:
            # Reset to full historical range and deactivate filter
            st.session_state.date_filter_start = min_date
            st.session_state.date_filter_end = max_date
            st.session_state.date_filter_active = False
            st.rerun()
        
        # Debug information - show current filter values and status
        filter_status = "ACTIVE" if st.session_state.date_filter_active else "INACTIVE"
        st.write(f"**Debug:** Filter {filter_status} | Range: {st.session_state.date_filter_start} to {st.session_state.date_filter_end}")
        
        # Apply date filtering
        one_year_ago = datetime.now().date() - timedelta(days=365)
        
        # Filter matches based on session state filter settings
        filtered_matches = []
        debug_excluded_recent = 0
        debug_excluded_date_range = 0
        debug_included = 0
        
        for m in all_matches:
            match_start = m['start_date'].date()
            match_end = m['end_date'].date()
            
            # First exclude matches that end within one year from today
            if match_end > one_year_ago:
                debug_excluded_recent += 1
                continue
                
            # Apply date range filtering if filter is active
            if st.session_state.date_filter_active:
                # Check if match period is within the filter range
                if match_start >= st.session_state.date_filter_start and match_end <= st.session_state.date_filter_end:
                    filtered_matches.append(m)
                    debug_included += 1
                else:
                    debug_excluded_date_range += 1
            else:
                # Show all historical matches when filter is inactive
                filtered_matches.append(m)
                debug_included += 1
        
        # Show debug info
        st.write(f"**Debug:** Total matches: {len(all_matches)}, Excluded (recent): {debug_excluded_recent}, "
                f"Excluded (date range): {debug_excluded_date_range}, Included: {debug_included}")
        
        # Additional debug - show date range of filtered matches
        if filtered_matches:
            filtered_start_dates = [m['start_date'].date() for m in filtered_matches]
            filtered_end_dates = [m['end_date'].date() for m in filtered_matches]
            actual_min_date = min(filtered_start_dates)
            actual_max_date = max(filtered_end_dates)
            st.write(f"**Debug:** Filtered matches date range: {actual_min_date} to {actual_max_date}")
        else:
            st.write(f"**Debug:** No matches after filtering")
        
        # Create filtered display dataframe
        filtered_match_data = pd.DataFrame([
            {
                'timeframe': m['timeframe'],
                'pattern': m['pattern'],
                'start_index': m['start_index'],
                'window_size': m['window_size'],
                'similarity': m['similarity'],
                'inverted': m['inverted'],
                'start_date': m['start_date'],
                'end_date': m['end_date']
            }
            for m in filtered_matches
        ])
        
        # Show filtered results summary
        one_year_ago = datetime.now().date() - timedelta(days=365)
        total_before_date_filter = len([m for m in all_matches if m['end_date'].date() <= one_year_ago])
        
        if st.session_state.date_filter_active:
            st.info(f"📅 Showing {len(filtered_matches)} matches (of {total_before_date_filter} historical matches) "
                   f"filtered from {st.session_state.date_filter_start} to {st.session_state.date_filter_end}")
        else:
            st.info(f"📅 Showing {len(filtered_matches)} historical matches (excluding matches ending after {one_year_ago})")
        
        # Display interactive table with row selection
        st.write("**Matches sorted by similarity (click on a row to view charts):**")
        
        # Use streamlit's dataframe with on_select event
        event = st.dataframe(
            filtered_match_data.head(20),
            column_config={
                "timeframe": st.column_config.TextColumn("Timeframe"),
                "pattern": st.column_config.TextColumn("Pattern"),
                "start_index": st.column_config.NumberColumn("Start Index"),
                "window_size": st.column_config.NumberColumn("Window Size"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                "inverted": st.column_config.CheckboxColumn("Inverted"),
                "start_date": st.column_config.DatetimeColumn("Start Date"),
                "end_date": st.column_config.DatetimeColumn("End Date")
            },
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="match_table"
        )
        
        # Check if a row is selected
        selected_match = None
        if event.selection.rows:
            selected_row_idx = event.selection.rows[0]
            if selected_row_idx < len(filtered_matches):
                selected_match = filtered_matches[selected_row_idx]
        # Display charts for selected match
        if selected_match:
            st.subheader(f"Charts for Selected Match")
            st.write(f"**{selected_match['timeframe'].title()}** | **{selected_match['pattern']}** | "
                    f"Similarity: **{selected_match['similarity']:.3f}** | "
                    f"Date: **{selected_match['start_date'].strftime('%Y-%m-%d')} to {selected_match['end_date'].strftime('%Y-%m-%d')}**")
            
            # Single combined chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            m = selected_match['match_data']
            pattern = [p[1] for p in patterns if p[0] == selected_match['pattern']][0]
            
            # Get price data with context
            series = selected_match['series']
            dates = selected_match['dates']
            window_size = m["size"]
            past_extension = max(1, int(window_size * 0.1))  # 10% of window size
            future_extension = max(1, int(window_size * 0.2))  # 20% of window size
            
            start_extended = max(0, m["start"] - past_extension)
            end_extended = min(len(series), m["start"] + m["size"] + future_extension)
            idx_range_extended = range(start_extended, end_extended)
            idx_range_match = range(m["start"], m["start"] + m["size"])
            
            # Plot the actual price data
            ax.plot(dates[idx_range_extended], series[idx_range_extended], 'lightblue', alpha=0.6, linewidth=1, label='Price Context')
            ax.plot(dates[idx_range_match], series[idx_range_match], 'blue', linewidth=1.5, label='Match Period')
            
            # Create a twin axis for the normalized pattern overlay
            ax2 = ax.twinx()
            
            # Scale and position the normalized patterns to overlay on the match period
            match_prices = series[idx_range_match]
            price_min, price_max = match_prices.min(), match_prices.max()
            price_range = price_max - price_min
            
            # Get the pattern dates for the match period
            pattern_dates = dates[idx_range_match]
            match_length = len(pattern_dates)
            
            # Resample patterns to match the length of the match period
            pattern_resampled = np.interp(
                np.linspace(0, len(pattern)-1, match_length), 
                np.arange(len(pattern)), 
                pattern
            )
            
            # Also resample the match window to ensure correct length
            match_window_resampled = np.interp(
                np.linspace(0, len(m['window'])-1, match_length), 
                np.arange(len(m['window'])), 
                m['window']
            )
            
            # Scale patterns to match price range
            pattern_scaled = pattern_resampled * price_range * 0.3 + price_min + price_range * 0.1
            match_scaled = match_window_resampled * price_range * 0.3 + price_min + price_range * 0.1
            
            # Plot normalized patterns on the twin axis
            ax2.plot(pattern_dates, pattern_scaled, 'k-', linewidth=1, alpha=0.8, label='Reference Pattern (scaled)')
            
            if m['inverted']:
                inverted_pattern_resampled = np.interp(
                    np.linspace(0, len(pattern)-1, match_length), 
                    np.arange(len(pattern)), 
                    -pattern
                )
                inverted_pattern_scaled = inverted_pattern_resampled * price_range * 0.3 + price_min + price_range * 0.1
                ax2.plot(pattern_dates, inverted_pattern_scaled, 'orange', linestyle='--', linewidth=1, alpha=0.8, label='Inverted Reference (scaled)')
                # Match Pattern line removed for clarity
            
            # Styling for main axis
            ax.set_title(f'Price Movement with Pattern Overlay: {selected_match["pattern"]}{" (Inverted)" if m["inverted"] else ""}\n'
                        f'{dates[start_extended].strftime("%Y-%m-%d")} to {dates[end_extended-1].strftime("%Y-%m-%d")} | '
                        f'Similarity: {m["similarity"]:.3f}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, alpha=0.3)
            
            # Styling for twin axis
            ax2.set_ylabel('Normalized Pattern (scaled)', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add data point annotations for easier reading
            # Show a few key data points as text annotations
            extended_prices = series[idx_range_extended]
            extended_dates = dates[idx_range_extended]
            
            # Add start and end markers for the match period
            match_start_date = dates[m["start"]]
            match_end_date = dates[m["start"] + m["size"] - 1]
            match_start_price = series[m["start"]]
            match_end_price = series[m["start"] + m["size"] - 1]
            
            # Add subtle markers for match boundaries
            ax.axvline(x=match_start_date, color='green', alpha=0.5, linestyle=':', linewidth=1)
            ax.axvline(x=match_end_date, color='red', alpha=0.5, linestyle=':', linewidth=1)
            
            # Add text annotations for match boundaries
            ax.annotate(f'Match Start\n{match_start_date.strftime("%Y-%m-%d")}\n${match_start_price:.2f}', 
                       xy=(match_start_date, match_start_price), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                       fontsize=8, ha='left')
            
            ax.annotate(f'Match End\n{match_end_date.strftime("%Y-%m-%d")}\n${match_end_price:.2f}', 
                       xy=(match_end_date, match_end_price), 
                       xytext=(-10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                       fontsize=8, ha='right')
            
            # Add price markers at regular intervals for easier reading
            extended_length = len(idx_range_extended)
            if extended_length > 3:
                # Show 3-5 evenly spaced price points
                step = max(1, extended_length // 4)
                for i in range(0, extended_length, step):
                    if i < len(extended_dates) and i < len(extended_prices):
                        date_point = extended_dates[i]
                        price_point = extended_prices[i]
                        # Add small dot markers
                        ax.plot(date_point, price_point, 'o', color='darkblue', markersize=3, alpha=0.7)
                        # Add small text labels
                        if i % (step * 2) == 0:  # Show text every other marker to avoid crowding
                            ax.annotate(f'${price_point:.2f}', 
                                       xy=(date_point, price_point), 
                                       xytext=(0, 15), textcoords='offset points',
                                       fontsize=7, ha='center', alpha=0.8,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add detailed data table for precise reading
            with st.expander("📊 Detailed Price Data for This Match", expanded=False):
                # Create a detailed dataframe with all the data points in the chart
                chart_data = []
                for i in idx_range_extended:
                    is_match_period = i >= m["start"] and i < (m["start"] + m["size"])
                    chart_data.append({
                        'Date': dates[i].strftime('%Y-%m-%d'),
                        'Price': f"${series[i]:.2f}",
                        'Period': 'Match Period' if is_match_period else 'Context',
                        'Day_of_Week': dates[i].strftime('%A'),
                        'Price_Change': f"{((series[i] / series[i-1] - 1) * 100):.1f}%" if i > 0 and i-1 in idx_range_extended else "N/A"
                    })
                
                chart_df = pd.DataFrame(chart_data)
                
                # Display summary statistics
                match_prices = [series[i] for i in idx_range_match]
                st.write(f"**Match Period Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Start Price", f"${match_prices[0]:.2f}")
                with col2:
                    st.metric("End Price", f"${match_prices[-1]:.2f}")
                with col3:
                    st.metric("Min Price", f"${min(match_prices):.2f}")
                with col4:
                    st.metric("Max Price", f"${max(match_prices):.2f}")
                
                # Show the detailed data table
                st.dataframe(
                    chart_df,
                    column_config={
                        "Date": st.column_config.TextColumn("Date"),
                        "Price": st.column_config.TextColumn("Price"),
                        "Period": st.column_config.TextColumn("Period"),
                        "Day_of_Week": st.column_config.TextColumn("Day"),
                        "Price_Change": st.column_config.TextColumn("% Change")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("👆 Click on any row in the table above to view detailed charts for that match.")
    else:
        st.info("💡 Configure your search parameters in the sidebar and click '🔍 Start Pattern Search' to begin analysis.")


st.markdown("""
---
**Tips:**  
- For your pattern library, upload CSVs with a `close` column, one file per pattern.
- Try DTW for time-warped/"stretched" pattern matching, or cosine for fast, direct shape matching.
- Use a lower similarity threshold for more matches; higher for only the closest.
- Search results are automatically cached - identical searches will load instantly from cache.
- Manage cached searches in the sidebar "Cache Management" section.
- Results show where in your daily/weekly history the reference pattern (or its inverse) recurs in different sizes and positions—true fractal echoes!
""")
