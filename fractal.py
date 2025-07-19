import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from io import StringIO
import yfinance as yf
from datetime import datetime, timedelta

# --- Utility Functions ---

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
data_source = st.radio("Choose data source:", ["Upload CSV File", "Yahoo Finance Ticker"], index=1)

df = None

if data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload daily OHLCV CSV (with 'date' and 'close' columns)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['date']).sort_values('date')
        df = df.set_index('date')

elif data_source == "Yahoo Finance Ticker":
    col1, col2 = st.columns(2)
    with col1:
        ticker_symbol = st.text_input("Enter ticker symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL")
    with col2:
        fetch_button = st.button("Fetch Data")
    
    if fetch_button and ticker_symbol:
        with st.spinner(f"Fetching all available data for {ticker_symbol}..."):
            try:
                # Fetch all available data from Yahoo Finance
                ticker_data = yf.download(ticker_symbol, period="max", progress=False)
                
                # Debug: Check the type and structure of returned data
                st.write(f"Debug - Data type: {type(ticker_data)}")
                st.write(f"Debug - Data shape: {ticker_data.shape if hasattr(ticker_data, 'shape') else 'No shape'}")
                st.write(f"Debug - Data columns: {list(ticker_data.columns) if hasattr(ticker_data, 'columns') else 'No columns'}")
                
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
                        st.write("Available columns:", list(ticker_data.columns))
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
                                st.success(f"Successfully fetched {len(df)} days of data for {ticker_symbol}")
                        except Exception as conversion_error:
                            st.error(f"Error converting data format for {ticker_symbol}: {str(conversion_error)}")
                            st.write("Raw data preview:")
                            st.dataframe(ticker_data.head())
                    
            except Exception as e:
                st.error(f"Error fetching data for {ticker_symbol}: {str(e)}. Please verify the ticker symbol is correct and try again.")

if df is not None:
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

    # --- Create monthly and weekly resamples ---
    st.subheader("Auto-Generated Resamples")
    monthly = resample_ohlc(df, 'M')
    weekly = resample_ohlc(df, 'W')
    st.write("Monthly (for reference pattern):")
    st.dataframe(monthly.head())
    st.write("Weekly (for pattern search):")
    st.dataframe(weekly.head())

    # --- Pattern Selection: Whole history or upload pattern library ---
    st.sidebar.header("Reference Patterns")
    use_library = st.sidebar.checkbox("Use pattern library (upload)", value=False)
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
    similarity_threshold = st.sidebar.slider("Similarity threshold", 0.80, 0.99, 0.93, 0.01)
    min_window_monthly = st.sidebar.slider("Min window size (weeks/days)", 4, 30, 8)
    max_window_multiplier = st.sidebar.slider("Max window size multiplier (x monthly length)", 1, 5, 3)
    match_method = st.sidebar.selectbox("Similarity method", ['cosine', 'dtw'])
    use_dtw = (match_method == 'dtw')
    allow_inversion = st.sidebar.checkbox("Enable inverted pattern search", value=False)
    
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
    st.write(f"Reference pattern length: {ref_len}")
    st.write(f"Weekly search window: {min(window_lengths_weekly)} to {max(window_lengths_weekly)}")
    st.write(f"Daily search window: {min(window_lengths_daily)} to {max(window_lengths_daily)}")
    
    # --- Run Search ---
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
    st.success(f"Found {len(matches_weekly)} weekly matches and {len(matches_daily)} daily matches.")

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
    plot_matches(df['close'].values, matches_daily, patterns, df.index, n=5, title="Daily")

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
            'start_date': df.index[m['start']],
            'end_date': df.index[m['start']+m['size']-1]
        }
        for m in matches_daily
    ])
    st.dataframe(match_data.head(10))
    csv = match_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download matches as CSV", csv, "fractal_matches.csv", "text/csv")

st.markdown("""
---
**Tips:**  
- For your pattern library, upload CSVs with a `close` column, one file per pattern.
- Try DTW for time-warped/“stretched” pattern matching, or cosine for fast, direct shape matching.
- Use a lower similarity threshold for more matches; higher for only the closest.
- Results show where in your daily/weekly history the reference pattern (or its inverse) recurs in different sizes and positions—true fractal echoes!
""")
