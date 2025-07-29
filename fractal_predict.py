#!/usr/bin/env python3
"""
Merged Fractal Pattern Analysis and Stock Prediction Interface
Combines fractal pattern matching with AI-powered stock predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import multiprocessing as mp
import joblib
import os
import warnings
import json
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import subprocess
from sklearn.model_selection import train_test_split
from io import StringIO
import yfinance as yf
from datetime import timedelta
import datetime as dt
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv


# Try to import lightgbm
try:
    import lightgbm as lgb
except ImportError:
    st.error("⚠️ LightGBM not found. Install with: pip install lightgbm")
    lgb = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.append('..')
from util import calculate_all_technical_indicators
from config import features

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Set page config
st.set_page_config(
    page_title="Fractal Pattern Analysis & Stock Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">

<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main .block-container {
        font-family: 'Inter', sans-serif;
        padding-top: 2rem;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .section-header {
        color: #495057;
        font-weight: 700;
        font-size: 1.5rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .tech-indicator {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    /* Professional Criteria Cards */
    .criteria-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .criteria-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .criteria-header {
        font-weight: 600;
        font-size: 1.2rem;
        color: #495057;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Risk Level Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
    }
    
    /* Criteria Items */
    .criteria-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: rgba(248, 249, 250, 0.7);
        border-radius: 8px;
        border-left: 4px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .criteria-item:hover {
        background: rgba(248, 249, 250, 1);
        transform: translateX(5px);
    }
    
    .criteria-item.success {
        border-left-color: #28a745;
        background: rgba(40, 167, 69, 0.1);
    }
    
    .criteria-item.danger {
        border-left-color: #dc3545;
        background: rgba(220, 53, 69, 0.1);
    }
    
    /* Score Badges */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Fractal Pattern Analysis Functions ---

def normalize_and_resample(series, target_length):
    arr = np.array(series)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    resampled = np.interp(
        np.linspace(0, len(arr)-1, target_length), np.arange(len(arr)), arr
    )
    return resampled

def cosine_similarity(a, b):
    """Calculate cosine similarity between two patterns."""
    try:
        return 1 - cosine(a, b)
    except:
        return 0.0

def filter_overlapping_matches(matches, series_dates, min_separation_days=30):
    """Filter out overlapping or close-proximity matches."""
    if not matches:
        return matches
    
    sorted_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    filtered_matches = []
    
    for match in sorted_matches:
        try:
            match_start_date = series_dates[match['start']]
            match_end_date = series_dates[match['start'] + match['size'] - 1]
            
            # Ensure datetime objects
            if not isinstance(match_start_date, pd.Timestamp):
                match_start_date = pd.to_datetime(match_start_date)
            if not isinstance(match_end_date, pd.Timestamp):
                match_end_date = pd.to_datetime(match_end_date)
            
            is_too_close = False
            for accepted_match in filtered_matches:
                accepted_start_date = series_dates[accepted_match['start']]
                accepted_end_date = series_dates[accepted_match['start'] + accepted_match['size'] - 1]
                
                # Ensure datetime objects
                if not isinstance(accepted_start_date, pd.Timestamp):
                    accepted_start_date = pd.to_datetime(accepted_start_date)
                if not isinstance(accepted_end_date, pd.Timestamp):
                    accepted_end_date = pd.to_datetime(accepted_end_date)
                
                if match_end_date < accepted_start_date:
                    gap_days = (accepted_start_date - match_end_date).days
                elif accepted_end_date < match_start_date:
                    gap_days = (match_start_date - accepted_end_date).days
                else:
                    gap_days = 0
                
                if gap_days < min_separation_days:
                    is_too_close = True
                    break
            
            if not is_too_close:
                filtered_matches.append(match)
        except Exception:
            # If date processing fails, skip this match
            continue
    
    return filtered_matches

def advanced_slide_and_compare(series, patterns, window_sizes, threshold=0.85, 
                              series_dates=None, exclude_overlap=True, 
                              filter_close_matches=True, min_separation_days=30, 
                              pattern_date_range=None):
    """Advanced pattern matching using cosine similarity."""
    matches = []
    for pat_name, pattern in patterns:
        pat_len = len(pattern)
        for win_size in window_sizes:
            for start in range(len(series) - win_size + 1):
                window = series[start:start+win_size]
                window_norm = normalize_and_resample(window, pat_len)
                
                # Skip self-matches using 80% overlap detection
                if exclude_overlap and pattern_date_range is not None and series_dates is not None:
                    try:
                        window_start_date = series_dates[start]
                        window_end_date = series_dates[start + win_size - 1]
                        pattern_start, pattern_end = pattern_date_range
                        
                        # Ensure dates are datetime objects
                        if not isinstance(window_start_date, pd.Timestamp):
                            window_start_date = pd.to_datetime(window_start_date)
                        if not isinstance(window_end_date, pd.Timestamp):
                            window_end_date = pd.to_datetime(window_end_date)
                        if not isinstance(pattern_start, pd.Timestamp):
                            pattern_start = pd.to_datetime(pattern_start)
                        if not isinstance(pattern_end, pd.Timestamp):
                            pattern_end = pd.to_datetime(pattern_end)
                        
                        overlap_start = max(window_start_date, pattern_start)
                        overlap_end = min(window_end_date, pattern_end)
                        
                        if overlap_start <= overlap_end:
                            overlap_days = (overlap_end - overlap_start).days + 1
                            window_days = (window_end_date - window_start_date).days + 1
                            pattern_days = (pattern_end - pattern_start).days + 1
                            
                            overlap_pct_window = overlap_days / window_days
                            overlap_pct_pattern = overlap_days / pattern_days
                            
                            if overlap_pct_window >= 0.8 or overlap_pct_pattern >= 0.8:
                                continue
                    except Exception:
                        # If date processing fails, skip overlap detection for this window
                        pass
                
                # Use cosine similarity only
                sim = cosine_similarity(pattern, window_norm)
                
                if sim >= threshold:
                    matches.append({
                        'pattern': pat_name,
                        'start': start,
                        'size': win_size,
                        'similarity': sim,
                        'window': window_norm.tolist() if len(matches) < 1000 else None
                    })
    
    # Filter close matches if requested
    if filter_close_matches and series_dates is not None and matches:
        matches = filter_overlapping_matches(matches, series_dates, min_separation_days)
    
    return matches

def resample_ohlc(df, rule='W'):
    """Resample OHLC data to different timeframes."""
    # Check column names and adjust accordingly
    columns = df.columns.str.lower()
    has_volume = 'volume' in columns
    
    ohlc_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'open':
            ohlc_dict[col] = 'first'
        elif col_lower == 'high':
            ohlc_dict[col] = 'max'
        elif col_lower == 'low':
            ohlc_dict[col] = 'min'
        elif col_lower == 'close':
            ohlc_dict[col] = 'last'
        elif col_lower == 'volume' and has_volume:
            ohlc_dict[col] = 'sum'
    
    return df.resample(rule).agg(ohlc_dict).dropna()

# --- Stock Prediction Functions ---

def calculate_entropy_confidence(probabilities):
    """Calculate entropy-based confidence score."""
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities))
    max_entropy = np.log(len(probabilities))
    confidence = 1 - (entropy / max_entropy)
    return confidence

def create_derived_features(df):
    """Create all derived features matching the training script."""
    
    # Add missing core features that might be named differently
    if 'price_to_sma_5' not in df.columns and 'sma_5' in df.columns:
        df['price_to_sma_5'] = df['close'] / df['sma_5']
    
    if 'price_to_sma_20' not in df.columns and 'sma_20' in df.columns:
        df['price_to_sma_20'] = df['close'] / df['sma_20']
        
    if 'sma_5_slope' not in df.columns and 'sma_5' in df.columns:
        df['sma_5_slope'] = df['sma_5'].pct_change()
        
    if 'sma_20_slope' not in df.columns and 'sma_20' in df.columns:
        df['sma_20_slope'] = df['sma_20'].pct_change()
        
    if 'macd_signal' not in df.columns and 'macd_signal_line' in df.columns:
        df['macd_signal'] = df['macd_signal_line']
        
    if 'macd_hist' not in df.columns and 'macd_histogram' in df.columns:
        df['macd_hist'] = df['macd_histogram']
        
    if 'volume_ratio' not in df.columns and 'vol' in df.columns and 'volume_sma' in df.columns:
        df['volume_ratio'] = np.where(df['volume_sma'] != 0, df['vol'] / df['volume_sma'], 0)
        
    if 'resistance_distance' not in df.columns and 'resistance_20' in df.columns:
        df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close']
    
    # === Create all 14 derived features ===
    
    # Relative Volatility Ratios
    if 'bb_to_volatility' not in df.columns:
        df['bb_to_volatility'] = np.where(df['price_volatility'] != 0, 
                                          df['bb_std'] / df['price_volatility'], 0)
    
    if 'bb_width_to_volatility20' not in df.columns:
        df['bb_width_to_volatility20'] = np.where(df['volatility_20'] != 0, 
                                                  df['bb_width'] / df['volatility_20'], 0)
    
    if 'price_vol_to_vol20' not in df.columns:
        df['price_vol_to_vol20'] = np.where(df['volatility_20'] != 0, 
                                            df['price_volatility'] / df['volatility_20'], 0)
    
    # MACD Divergence Strength
    if 'macd_diff' not in df.columns:
        df['macd_diff'] = df['macd'] - df['macd_signal']
    if 'macd_trend_strength' not in df.columns:
        df['macd_trend_strength'] = df['macd_hist'] * df['macd_momentum']
    
    # SMA Cross Features
    if 'sma_diff_5_20' not in df.columns:
        df['sma_diff_5_20'] = df['price_to_sma_20'] - df['price_to_sma_5']
    if 'sma_slope_diff' not in df.columns:
        df['sma_slope_diff'] = df['sma_5_slope'] - df['sma_20_slope']
    
    # Volume + Price Fusion
    if 'vol_price_momentum' not in df.columns:
        df['vol_price_momentum'] = df['volume_ratio'] * df['price_change_abs']
    if 'obv_macd_interact' not in df.columns:
        df['obv_macd_interact'] = df['obv_momentum'] * df['macd_momentum']
    
    # Stochastic Spread
    if 'stoch_diff' not in df.columns:
        df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    
    # Support/Resistance Distance Ratios
    if 'sr_ratio' not in df.columns:
        df['sr_ratio'] = np.where(df['resistance_distance'] != 0, 
                                  df['support_distance'] / df['resistance_distance'], 0)
    
    if 'move_to_support_ratio' not in df.columns:
        df['move_to_support_ratio'] = np.where(df['support_distance'] != 0, 
                                               df['price_change_abs'] / df['support_distance'], 0)
    
    # Trend Persistence
    if 'trend_consistency_2d' not in df.columns:
        df['trend_consistency_2d'] = df['price_change_lag_1'] * df['price_change_lag_2']
    if 'trend_consistency_3d' not in df.columns:
        df['trend_consistency_3d'] = df['price_change_lag_1'] * df['price_change_lag_3']
    
    return df

def load_stock_data(ticker, keep_uppercase=False):
    """Load and process stock data."""
    try:
        data = yf.download(ticker, period='max')
        if data.empty:
            return None, "No data found for ticker"
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.reset_index(inplace=True)
        data.columns = [col.upper() for col in data.columns]
        data = data.rename(columns={'VOLUME': 'VOL'})
        
        df = calculate_all_technical_indicators(data)
        
        # Only convert to lowercase if not keeping uppercase (for training compatibility)
        if not keep_uppercase:
            df.columns = [col.lower() for col in df.columns]
        
        # Ensure date column is properly set as datetime index
        date_col = 'date' if not keep_uppercase else 'DATE'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        elif df.index.name != date_col:
            df.index = pd.to_datetime(df.index)
        
        return df, None
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('''
    <div class="main-header">
        <h1><i class="fas fa-chart-line"></i> Fractal Pattern Analysis & Stock Prediction</h1>
        <p style="margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;">
            Cosine similarity-based pattern matching with AI-powered stock predictions
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar for unified settings
    with st.sidebar:
        st.header("📊 Analysis Settings")
        
        ticker = st.text_input(
            "Enter Stock Ticker", 
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
        ).upper()
        
        st.markdown("---")
        st.subheader("🔮 Prediction Settings")
        st.info("Uses existing trained models if available")
        
        # Advanced option for training (collapsed by default)
        with st.expander("🔧 Advanced: Model Training"):
            st.warning("⚠️ Model training may fail due to data format compatibility issues.")
            force_retrain = st.checkbox(
                "🔄 Attempt to train new models",
                value=False,
                help="Attempt to train new models (experimental - may not work)"
            )
        
        st.markdown("---")
        st.subheader("🔍 Pattern Analysis Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.7,
            max_value=0.99,
            value=0.85,
            step=0.01,
            help="Minimum cosine similarity score for pattern matches"
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            options=["Daily", "Weekly"],
            index=0,
            help="Timeframe for pattern analysis"
        )
        
        if timeframe == "Daily":
            pattern_length = st.selectbox(
                "Pattern Length (days)",
                options=[10, 15, 20, 30, 45, 60],
                index=2,
                help="Length of the reference pattern in days"
            )
        else:  # Weekly
            pattern_length = st.selectbox(
                "Pattern Length (weeks)",
                options=[4, 6, 8, 12, 20, 26],
                index=2,
                help="Length of the reference pattern in weeks"
            )
        
        st.markdown("---")
        # Single unified button
        run_analysis = st.button("🚀 Run ", type="primary", use_container_width=True)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Stock Prediction Results", "🔍 Fractal Pattern Results"])
    
    if run_analysis:
        if not ticker:
            st.error("Please enter a ticker symbol")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ========== SHARED DATA LOADING ==========
        status_text.text("📊 Fetching stock data...")
        progress_bar.progress(10)
        
        base_ticker = ticker.replace('.US', '')
        df, error = load_stock_data(ticker)
        
        if error:
            st.error(f"Error loading data: {error}")
            return
        
        if len(df) < max(pattern_length * 2, 100):
            st.error(f"Insufficient data. Need at least {max(pattern_length * 2, 100)} days of data.")
            return
        
        # Get current market data for both analyses
        current_price = float(df['close'].iloc[-1])
        previous_price = float(df['close'].iloc[-2])
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100
        
        progress_bar.progress(20)
        
        # ========== STOCK PREDICTION ANALYSIS ==========
        status_text.text("🔧 Processing technical indicators...")
        progress_bar.progress(30)
        
        # Get additional market data for prediction
        current_volume = int(df['vol'].iloc[-1])
        high_today = float(df['high'].iloc[-1])
        low_today = float(df['low'].iloc[-1])
        open_today = float(df['open'].iloc[-1])
        
        df_clean = df.dropna()
        if df_clean.empty:
            st.error("Insufficient data for technical indicators calculation")
            return
        
        df_latest = df_clean.iloc[[-1]]
        
        # Load model and features
        status_text.text("📥 Loading model and features...")
        progress_bar.progress(40)
        
        # Load features used during training - check multiple locations
        features_path = f"models/{base_ticker}_yahoo/features_used_yahoo.json"
        predictor_features_path = f"predictor/models/{base_ticker}_yahoo/features_used_yahoo.json"
        fallback_features_path = f"predictor/models/{base_ticker}/features_used.json"
        
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                training_features = json.load(f)
        elif os.path.exists(predictor_features_path):
            with open(predictor_features_path, 'r') as f:
                training_features = json.load(f)
        elif os.path.exists(fallback_features_path):
            with open(fallback_features_path, 'r') as f:
                training_features = json.load(f)
        else:
            training_features = features
        
        # Create missing derived features
        missing_features = [f for f in training_features if f not in df_latest.columns]
        if missing_features:
            df_latest = create_derived_features(df_latest)
            missing_features = [f for f in training_features if f not in df_latest.columns]
            if missing_features:
                available_features = [f for f in training_features if f in df_latest.columns]
                training_features = available_features
        
        # Prepare model input
        X_live = df_latest[training_features].values.reshape(1, -1)
        
        # Load or train model - check both current directory and predictor directory
        model_path = f"models/{base_ticker}_yahoo/lightgbm_model_yahoo.pkl"
        fallback_model_path = f"models/{base_ticker}/vstm_lightgbm_model.pkl"
        predictor_model_path = f"predictor/models/{base_ticker}_yahoo/lightgbm_model_yahoo.pkl"
        predictor_fallback_path = f"predictor/models/{base_ticker}/vstm_lightgbm_model.pkl"
        
        model = None
        proba = None
        
        # Check if we need to train a new model
        # Set default if force_retrain not defined (in case of UI changes)
        try:
            force_retrain
        except NameError:
            force_retrain = False
            
        if not os.path.exists(model_path) or force_retrain:
            status_text.text("🚀 Training model (this may take a moment)...")
            progress_bar.progress(50)
            
            try:
                result = subprocess.run([sys.executable, "train_wrapper.py", base_ticker], 
                                      capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    st.success("✅ Model training completed successfully!")
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                    else:
                        st.warning("Model training completed but model file not found. Checking fallback location...")
                        if os.path.exists(fallback_model_path):
                            model = joblib.load(fallback_model_path)
                else:
                    st.error(f"❌ Model training failed with return code {result.returncode}")
                    
                    # Show both stdout and stderr if available
                    if result.stdout:
                        st.subheader("Training Output:")
                        st.code(result.stdout, language="text")
                    
                    if result.stderr:
                        st.subheader("Error Output:")
                        st.code(result.stderr, language="text")
                    
                    # Continue with pattern analysis only
                    st.info("📊 Continuing with pattern analysis only...")
                    st.markdown("""
                    **Note:** Model training failed likely due to:
                    - Data format incompatibility between the app and training script
                    - Missing required data columns
                    - Insufficient historical data
                    
                    **Recommendations:**
                    - Use the Pattern Analysis tab which works independently
                    - Contact administrator to pre-train models for this ticker
                    - Use a different ticker that may have existing models
                    """)
            except subprocess.TimeoutExpired:
                st.error("❌ Model training timed out (>10 minutes)")
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
        
        # Try to load existing model if not already loaded
        if model is None:
            if os.path.exists(model_path):
                status_text.text("📥 Loading existing model...")
                progress_bar.progress(50)
                model = joblib.load(model_path)
            elif os.path.exists(fallback_model_path):
                status_text.text("📥 Loading fallback model...")
                progress_bar.progress(50)
                model = joblib.load(fallback_model_path)
            elif os.path.exists(predictor_model_path):
                status_text.text("📥 Loading predictor model...")
                progress_bar.progress(50)
                model = joblib.load(predictor_model_path)
            elif os.path.exists(predictor_fallback_path):
                status_text.text("📥 Loading predictor fallback model...")
                progress_bar.progress(50)
                model = joblib.load(predictor_fallback_path)
            else:
                status_text.text("⚠️ No trained model found...")
                progress_bar.progress(50)
                st.warning(f"No trained model found for {ticker}. Only pattern analysis will be available.")
        
        # Make prediction if model is available
        if model is not None:
            status_text.text("🎯 Making prediction...")
            progress_bar.progress(60)
            try:
                proba = model.predict_proba(X_live)[0]
            except Exception as e:
                st.warning(f"Prediction failed: {str(e)}. Only pattern analysis will be available.")
                proba = None
        
        # ========== FRACTAL PATTERN ANALYSIS ==========
        status_text.text("🔍 Analyzing fractal patterns...")
        progress_bar.progress(70)
        
        # Prepare data based on timeframe
        if timeframe == "Weekly":
            status_text.text("🔍 Resampling to weekly data...")
            df_resampled = resample_ohlc(df, rule='W')
            analysis_df = df_resampled
            timeframe_label = "weeks"
        else:
            analysis_df = df
            timeframe_label = "days"
        
        # Create reference pattern from most recent data
        reference_pattern = analysis_df['close'].tail(pattern_length).values
        reference_pattern_norm = normalize_and_resample(reference_pattern, pattern_length)
        
        # Define pattern date range
        try:
            pattern_start_date = pd.to_datetime(analysis_df.index[-pattern_length])
            pattern_end_date = pd.to_datetime(analysis_df.index[-1])
            pattern_date_range = (pattern_start_date, pattern_end_date)
        except Exception:
            # If date conversion fails, disable overlap detection
            pattern_date_range = None
        
        # Prepare series for comparison
        price_series = analysis_df['close'].values
        series_dates = analysis_df.index
        
        # Define window sizes for pattern matching
        window_sizes = [pattern_length - 5, pattern_length, pattern_length + 5]
        window_sizes = [w for w in window_sizes if w > 0]
        
        # Perform pattern matching
        patterns = [("Reference", reference_pattern_norm)]
        
        matches = advanced_slide_and_compare(
            price_series, 
            patterns, 
            window_sizes,
            threshold=similarity_threshold,
            series_dates=series_dates,
            exclude_overlap=True,
            filter_close_matches=True,
            min_separation_days=30,
            pattern_date_range=pattern_date_range
        )
        
        progress_bar.progress(90)
        status_text.text("✅ Analysis completed!")
        
        # Define class labels for prediction
        class_labels = {
            0: "📉 Drop >10%",
            1: "📉 Drop 5-10%", 
            2: "📊 Drop 0-5%",
            3: "📈 Gain 0-5%",
            4: "📈 Gain 5-10%",
            5: "🚀 Gain >10%"
        }
        
        # Clear progress indicators
        progress_bar.progress(100)
        progress_bar.empty()
        status_text.empty()
        
        # ========== DISPLAY RESULTS IN TABS ==========
        with tab1:
            st.markdown('<div class="section-header"><i class="fas fa-crystal-ball"></i> AI Stock Prediction Results</div>', unsafe_allow_html=True)
            
            # Display basic results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### 📈 {ticker} Current Market Data")
                
                price_color = "green" if price_change >= 0 else "red"
                change_symbol = "+" if price_change >= 0 else ""
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💰 Current Price</h4>
                    <h2 style="color: {price_color};">${current_price:.2f}</h2>
                    <p style="color: {price_color};">{change_symbol}${price_change:.2f} ({price_change_pct:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Market metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📅 Open", f"${open_today:.2f}")
                    st.metric("📊 Volume", f"{current_volume:,.0f}")
                with col_b:
                    st.metric("🔺 High", f"${high_today:.2f}")
                    st.metric("🔻 Low", f"${low_today:.2f}")
            
            with col2:
                st.markdown("### 🔮 Prediction Results")
                
                # Placeholder - will be updated with 7-day model results
                prediction_placeholder = st.empty()
                probability_placeholder = st.empty()
                
                if proba is not None:
                    best_class_idx = np.argmax(proba)
                    best_prob = proba[best_class_idx]
                    prediction_text = class_labels.get(best_class_idx, f"Class {best_class_idx}")
                    
                    with prediction_placeholder.container():
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🎯 Predicted Outcome</h3>
                            <h2>{prediction_text}</h2>
                            <p>Confidence: {best_prob:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with probability_placeholder.container():
                        st.metric("📊 Model Probability", f"{best_prob:.1%}")
                else:
                    with prediction_placeholder.container():
                        st.markdown("""
                        <div class="metric-card">
                            <h4>⚠️ No Model Available</h4>
                            <p>No trained model found for this ticker.<br>
                            Pattern analysis is still available in the second tab.</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add 7-Day Probability Matrix if model is available
            if model is not None:
                st.markdown("### 📊 7-Day Probability Matrix")
                
                # Build multi-day models for 7-day prediction
                if lgb is None:
                    st.error("Cannot build multi-day models without LightGBM")
                else:
                    current_date = pd.Timestamp.now()
                    bubble_data = []
                    day_labels = []
                    
                    st.write("🔄 Building multi-day prediction models...")
                    multiday_progress = st.progress(0)
                    
                    # Generate day labels for next 7 trading days
                    trading_day_count = 0
                    for day in range(1, 15):  # Check more days to get 7 trading days
                        pred_date = current_date + timedelta(days=day)
                        if pred_date.weekday() < 5:  # Monday=0, Friday=4
                            trading_day_count += 1
                            if trading_day_count > 7:
                                break
                            day_label = pred_date.strftime('%a %m/%d')
                            day_labels.append(day_label)
                    
                    # Build models for each day horizon
                    models_dir = f"models/{base_ticker}_multiday"
                    day_models = {}
                    
                    for day_ahead in range(1, 8):
                        multiday_progress.progress(day_ahead / 8)
                        
                        model_file = os.path.join(models_dir, f"{base_ticker}_day_{day_ahead}_model.pkl")
                        
                        # Try to load existing model first
                        if os.path.exists(model_file):
                            try:
                                day_model = joblib.load(model_file)
                                day_models[day_ahead] = day_model
                                continue
                            except:
                                pass
                        
                        # Train new model if not found
                        target_data = []
                        feature_data = []
                        
                        for i in range(len(df_clean) - day_ahead):
                            current_features = df_clean.iloc[i]
                            future_row = df_clean.iloc[i + day_ahead]
                            
                            current_price_val = current_features['close']
                            future_price_val = future_row['close']
                            price_change_pct_val = (future_price_val - current_price_val) / current_price_val
                            
                            # Classify price change
                            if price_change_pct_val >= 0.05:
                                target_class = 4  # Strong Up
                            elif price_change_pct_val >= 0.02:
                                target_class = 3  # Up
                            elif price_change_pct_val >= -0.02:
                                target_class = 2  # Sideways
                            elif price_change_pct_val >= -0.05:
                                target_class = 1  # Down
                            else:
                                target_class = 0  # Strong Down
                            
                            # Extract features
                            features_row = []
                            for feature in training_features:
                                if feature in current_features.index:
                                    features_row.append(current_features[feature])
                                else:
                                    features_row.append(0.0)
                            
                            if not any(np.isnan(features_row)) and not np.isnan(target_class):
                                feature_data.append(features_row)
                                target_data.append(target_class)
                        
                        # Train model if enough data
                        if len(feature_data) > 100:
                            X = np.array(feature_data)
                            y = np.array(target_data)
                            
                            from sklearn.model_selection import train_test_split
                            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train LightGBM
                            train_data = lgb.Dataset(X_train, label=y_train)
                            params = {
                                'objective': 'multiclass',
                                'num_class': 5,
                                'metric': 'multi_logloss',
                                'boosting_type': 'gbdt',
                                'num_leaves': 31,
                                'learning_rate': 0.05,
                                'feature_fraction': 0.8,
                                'bagging_fraction': 0.8,
                                'verbose': -1
                            }
                            
                            day_model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[train_data])
                            day_models[day_ahead] = day_model
                            
                            # Save model
                            os.makedirs(models_dir, exist_ok=True)
                            joblib.dump(day_model, model_file)
                    
                    # Generate predictions for bubble chart
                    for day_ahead in range(1, 8):
                        if day_ahead in day_models:
                            day_proba = day_models[day_ahead].predict(X_live)[0]
                            
                            # Update main prediction with day-1 model
                            if day_ahead == 1:
                                best_class_idx = np.argmax(day_proba)
                                best_prob = day_proba[best_class_idx]
                                class_names = ['Strong Down', 'Down', 'Sideways', 'Up', 'Strong Up']
                                best_class_name = class_names[best_class_idx]
                                
                                with prediction_placeholder.container():
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <h3>🎯 Predicted Outcome (Day-1 Model)</h3>
                                        <h2>{best_class_name}</h2>
                                        <p>Model Confidence: {best_prob:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with probability_placeholder.container():
                                    st.metric("📊 Model Probability", f"{best_prob:.1%}")
                            
                            # Add to bubble chart data
                            day_label = day_labels[day_ahead - 1]
                            class_names = ['Strong Down', 'Down', 'Sideways', 'Up', 'Strong Up']
                            
                            for i, prob in enumerate(day_proba):
                                bubble_data.append({
                                    'Day': day_label,
                                    'Class': class_names[i],
                                    'Probability': prob,
                                    'Size': prob * 100,
                                    'Color': prob
                                })
                    
                    multiday_progress.progress(1.0)
                    multiday_progress.empty()
                    
                    # Create bubble chart
                    if bubble_data:
                        bubble_df = pd.DataFrame(bubble_data)
                        
                        fig_bubble = px.scatter(
                            bubble_df,
                            x='Day',
                            y='Class',
                            size='Size',
                            size_max=50,
                            title="7-Day Probability Matrix - Bubble Size = Probability",
                            hover_data={'Probability': ':.1%', 'Size': False, 'Color': False}
                        )
                        
                        fig_bubble.update_traces(marker=dict(color='grey'))
                        fig_bubble.update_layout(
                            height=500,
                            xaxis_title="Trading Days",
                            yaxis_title="Prediction Classes",
                            xaxis=dict(tickangle=45),
                            showlegend=False
                        )
                        
                        fig_bubble.update_traces(
                            hovertemplate="<b>%{y}</b><br>" +
                                         "Day: %{x}<br>" +
                                         "Probability: %{customdata[0]:.1%}<br>" +
                                         "<extra></extra>",
                            customdata=bubble_df[['Probability']].values
                        )
                        
                        st.plotly_chart(fig_bubble, use_container_width=True)
            
            # Add Professional Analysis Criteria
            if model is not None:
                st.markdown('<div class="section-header"><i class="fas fa-chart-line"></i> Professional Analysis Criteria</div>', unsafe_allow_html=True)
                
                # Calculate professional metrics from current data
                current_data = df.iloc[-1]
                current_price_val = current_data['close']
                
                # Calculate metrics
                sma_144 = current_data.get('sma_144', 0)
                sma_50 = current_data.get('sma_50', 0)
                macd = current_data.get('macd', 0)
                macd_hist = current_data.get('macd_histogram', 0)
                rsi = current_data.get('rsi_14', 0)
                volume_sma = current_data.get('volume_sma', 0)
                volatility_20 = current_data.get('volatility_20', 0)
                
                # Calculate ATR/Price ratio (using volatility as proxy)
                atr_to_price = (volatility_20 / current_price_val) if current_price_val > 0 else 0
                
                # Calculate scores
                trend_score = 0
                safety_score = 0
                relative_strength_score = 0
                
                # Trend score calculation
                if sma_50 > 0 and sma_144 > 0:
                    if current_price_val > sma_144 and sma_50 > sma_144:
                        trend_score += 2
                    if macd > 0 and macd_hist > 0:
                        trend_score += 2
                
                # Safety score calculation
                if 40 <= rsi <= 75:
                    safety_score += 2
                elif 30 <= rsi <= 80:
                    safety_score += 1
                
                if atr_to_price < 0.02:
                    safety_score += 3
                elif atr_to_price < 0.03:
                    safety_score += 2
                elif atr_to_price < 0.05:
                    safety_score += 1
                
                if volume_sma > 2000000:
                    safety_score += 2
                elif volume_sma > 1000000:
                    safety_score += 1
                
                col_prof1, col_prof2, col_prof3 = st.columns(3)
                
                with col_prof1:
                    # Price > SMA144
                    price_vs_sma144_ok = current_price_val > sma_144 if sma_144 > 0 else False
                    class1 = "success" if price_vs_sma144_ok else "danger"
                    color1 = "green" if price_vs_sma144_ok else "red"
                    
                    # SMA50 > SMA144
                    sma50_vs_sma144_ok = sma_50 > sma_144 if sma_50 > 0 and sma_144 > 0 else False
                    class2 = "success" if sma50_vs_sma144_ok else "danger"
                    color2 = "green" if sma50_vs_sma144_ok else "red"
                    
                    # MACD bullish
                    macd_ok = macd > 0 and macd_hist > 0
                    class3 = "success" if macd_ok else "danger"
                    color3 = "green" if macd_ok else "red"
                    
                    st.markdown(f'''
                    <div class="criteria-card">
                        <div class="criteria-header">
                            <i class="fas fa-arrow-trend-up"></i> Strong Trend Indicators
                        </div>
                        <div class="criteria-item {class1}">
                            <strong><i class="fas fa-arrow-up"></i> Price > SMA144:</strong>
                            <span style='color: {color1}; font-weight: 600;'>${current_price_val:.2f} vs ${sma_144:.2f}</span>
                        </div>
                        <div class="criteria-item {class2}">
                            <strong><i class="fas fa-chart-line"></i> SMA50 > SMA144:</strong>
                            <span style='color: {color2}; font-weight: 600;'>${sma_50:.2f} vs ${sma_144:.2f}</span>
                        </div>
                        <div class="criteria-item {class3}">
                            <strong><i class="fas fa-wave-square"></i> MACD Bullish:</strong>
                            <span style='color: {color3}; font-weight: 600;'>MACD: {macd:.4f}, Hist: {macd_hist:.4f}</span>
                        </div>
                        <div class="score-badge" style='background: linear-gradient(135deg, {'#28a745' if trend_score >= 3 else '#ffc107' if trend_score >= 2 else '#dc3545'}, {'#20c997' if trend_score >= 3 else '#fd7e14' if trend_score >= 2 else '#e83e8c'}); color: white;'>
                            <i class="fas fa-chart-line"></i> Trend Score: {trend_score}/4
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_prof2:
                    # ATR/Price ratio
                    atr_ok = atr_to_price < 0.05
                    class4 = "success" if atr_ok else "danger"
                    color4 = "green" if atr_ok else "red"
                    
                    # RSI range
                    rsi_ok = 25 <= rsi <= 85
                    class5 = "success" if rsi_ok else "danger"
                    color5 = "green" if rsi_ok else "red"
                    
                    # Volume
                    volume_ok = volume_sma > 500000
                    class6 = "success" if volume_ok else "danger"
                    color6 = "green" if volume_ok else "red"
                    
                    st.markdown(f'''
                    <div class="criteria-card">
                        <div class="criteria-header">
                            <i class="fas fa-shield-alt"></i> Safety Requirements
                        </div>
                        <div class="criteria-item {class4}">
                            <strong><i class="fas fa-chart-area"></i> ATR/Price < 5%:</strong>
                            <span style='color: {color4}; font-weight: 600;'>{atr_to_price:.3%}</span>
                        </div>
                        <div class="criteria-item {class5}">
                            <strong><i class="fas fa-tachometer-alt"></i> RSI 25-85:</strong>
                            <span style='color: {color5}; font-weight: 600;'>{rsi:.1f}</span>
                        </div>
                        <div class="criteria-item {class6}">
                            <strong><i class="fas fa-water"></i> Volume > 500k:</strong>
                            <span style='color: {color6}; font-weight: 600;'>{volume_sma:,.0f}</span>
                        </div>
                        <div class="score-badge" style='background: linear-gradient(135deg, {'#28a745' if safety_score >= 4 else '#ffc107' if safety_score >= 2 else '#dc3545'}, {'#20c997' if safety_score >= 4 else '#fd7e14' if safety_score >= 2 else '#e83e8c'}); color: white;'>
                            <i class="fas fa-shield-alt"></i> Safety Score: {safety_score}/6
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
                with col_prof3:
                    # Calculate relative strength metrics
                    momentum_items = []
                    
                    # 3M Momentum
                    if len(df) >= 60:
                        price_60d_ago = df['close'].iloc[-60]
                        momentum_3m = (current_price_val - price_60d_ago) / price_60d_ago if price_60d_ago > 0 else 0
                        momentum_ok = momentum_3m > 0.02
                        class7 = "success" if momentum_ok else "danger"
                        color7 = "green" if momentum_ok else "red"
                        if momentum_ok:
                            relative_strength_score += 2
                        momentum_items.append(f"""
<div class="criteria-item {class7}">
    <strong><i class="fas fa-rocket"></i> 3M Momentum > 2%:</strong>
    <span style="color: {color7}; font-weight: 600;">{momentum_3m:.1%}</span>
</div>""")
                    else:
                        momentum_items.append("""
<div class="criteria-item">
    <strong><i class="fas fa-rocket"></i> 3M Momentum:</strong>
    <span style="color: gray; font-weight: 600;">Insufficient data</span>
</div>""")
                    
                    # 52W High proximity
                    if len(df) >= 252:
                        high_52w = df['high'].tail(252).max()
                        distance_from_high = (current_price_val - high_52w) / high_52w
                        near_high_ok = distance_from_high > -0.50
                        class8 = "success" if near_high_ok else "danger"
                        color8 = "green" if near_high_ok else "red"
                        if near_high_ok:
                            relative_strength_score += 2
                        momentum_items.append(f"""
<div class="criteria-item {class8}">
    <strong><i class="fas fa-mountain"></i> Near 52W High:</strong>
    <span style="color: {color8}; font-weight: 600;">{distance_from_high:.1%}</span>
</div>""")
                    else:
                        high_available = df['high'].max()
                        distance_from_high = (current_price_val - high_available) / high_available
                        class8 = "success" if distance_from_high > -0.20 else "danger"
                        color8 = "green" if distance_from_high > -0.20 else "red"
                        if distance_from_high > -0.20:
                            relative_strength_score += 1
                        momentum_items.append(f"""
<div class="criteria-item {class8}">
    <strong><i class="fas fa-mountain"></i> Near Recent High:</strong>
    <span style="color: {color8}; font-weight: 600;">{distance_from_high:.1%}</span>
</div>""")
                    
                    # Recent price stability
                    if len(df) >= 5:
                        price_5d_ago = df['close'].iloc[-5]
                        recent_change = (current_price_val - price_5d_ago) / price_5d_ago
                        stable_ok = -0.10 < recent_change < 0.20
                        class9 = "success" if stable_ok else "danger"
                        color9 = "green" if stable_ok else "red"
                        if stable_ok:
                            relative_strength_score += 1
                        momentum_items.append(f"""
<div class="criteria-item {class9}">
    <strong><i class="fas fa-calendar-week"></i> 5D Change:</strong>
    <span style="color: {color9}; font-weight: 600;">{recent_change:+.1%}</span>
</div>""")
                    
                    # Technical confirmation
                    macd_signal = current_data.get('macd_signal', 0)
                    tech_confirm_ok = macd > macd_signal
                    class10 = "success" if tech_confirm_ok else "danger"
                    color10 = "green" if tech_confirm_ok else "red"
                    if tech_confirm_ok:
                        relative_strength_score += 1
                    momentum_items.append(f"""
<div class="criteria-item {class10}">
    <strong><i class="fas fa-check-circle"></i> MACD > Signal:</strong>
    <span style="color: {color10}; font-weight: 600;">{macd:.4f} vs {macd_signal:.4f}</span>
</div>""")
                    
                    st.markdown(f"""
                    <div class="criteria-card">
                        <div class="criteria-header">
                            <i class="fas fa-rocket"></i> Relative Strength
                        </div>
                        {''.join(momentum_items)}
                        <div class="score-badge" style="background: linear-gradient(135deg, {'#28a745' if relative_strength_score >= 4 else '#ffc107' if relative_strength_score >= 2 else '#dc3545'}, {'#20c997' if relative_strength_score >= 4 else '#fd7e14' if relative_strength_score >= 2 else '#e83e8c'}); color: white;">
                            <i class="fas fa-rocket"></i> Strength Score: {relative_strength_score}/6
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Calculate final total score and display risk level
                final_total_score = trend_score + safety_score + relative_strength_score
                if final_total_score >= 12:
                    final_risk_level = "LOW RISK"
                    final_risk_class = "risk-low"
                elif final_total_score >= 8:
                    final_risk_level = "MEDIUM RISK"
                    final_risk_class = "risk-medium"
                else:
                    final_risk_level = "HIGH RISK"
                    final_risk_class = "risk-high"
                
                st.markdown(f'''
                <div style="text-align: center; margin: 2rem 0;">
                    <div class="risk-badge {final_risk_class}">
                        <i class="fas fa-shield-alt"></i> {final_risk_level}
                    </div>
                    <div class="risk-badge {final_risk_class}">
                        <i class="fas fa-star"></i> Total Score: {final_total_score}/16
                    </div>
                </div>
                <div style="text-align: center; margin: 1rem 0; font-size: 0.9rem; color: #6c757d;">
                    Trend: {trend_score}/4 | Safety: {safety_score}/6 | Strength: {relative_strength_score}/6
                </div>
                ''', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="section-header"><i class="fas fa-search"></i> Fractal Pattern Analysis Results</div>', unsafe_allow_html=True)
            
            # ========== SEARCH CONFIGURATION DISPLAY ==========
            st.subheader("🔧 Search Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Reference Pattern", f"{pattern_length} {timeframe_label}")
                st.metric("Similarity Method", "Cosine Similarity")
                st.metric("Timeframe", timeframe)
            
            with col2:
                st.metric("Pattern Length", f"{pattern_length} periods")
                st.metric("Similarity Threshold", f"{similarity_threshold:.2f}")
                st.metric("Window Sizes", f"{pattern_length-5} to {pattern_length+5}")
            
            with col3:
                st.metric("Analysis Period", f"{len(analysis_df)} {timeframe_label}")
                st.metric("Total Matches Found", len(matches))
                st.metric("Top Matches Shown", min(20, len(matches)))
            
            if matches:
                st.success(f"Found {len(matches)} similar patterns using {timeframe.lower()} data!")
                
                # Sort by similarity
                matches_sorted = sorted(matches, key=lambda x: x['similarity'], reverse=True)
                
                # Add price context table
                st.markdown("### 💰 Actual Price Ranges")
                price_context_data = []
                
                # Current pattern price range
                current_start_price = reference_pattern[0]
                current_end_price = reference_pattern[-1]
                current_min_price = reference_pattern.min()
                current_max_price = reference_pattern.max()
                current_change_pct = ((current_end_price - current_start_price) / current_start_price) * 100
                
                price_context_data.append({
                    "Pattern": "Current (Reference)",
                    "Date Range": f"{pattern_start_date.strftime('%Y-%m-%d')} to {pattern_end_date.strftime('%Y-%m-%d')}",
                    "Start Price": f"${current_start_price:.2f}",
                    "End Price": f"${current_end_price:.2f}",
                    "Min/Max": f"${current_min_price:.2f} / ${current_max_price:.2f}",
                    "Change": f"{current_change_pct:+.1f}%",
                    "Similarity": "1.000"
                })
                
                # Similar patterns price ranges
                for i, match in enumerate(matches_sorted[:20]):
                    match_start_idx = match['start']
                    match_end_idx = match['start'] + match['size']
                    match_prices = price_series[match_start_idx:match_end_idx]
                    match_start_date = series_dates[match_start_idx]
                    match_end_date = series_dates[match_end_idx - 1]
                    
                    start_price = match_prices[0]
                    end_price = match_prices[-1]
                    min_price = match_prices.min()
                    max_price = match_prices.max()
                    change_pct = ((end_price - start_price) / start_price) * 100
                    
                    price_context_data.append({
                        "Pattern": f"Similar #{i+1}",
                        "Date Range": f"{match_start_date.strftime('%Y-%m-%d')} to {match_end_date.strftime('%Y-%m-%d')}",
                        "Start Price": f"${start_price:.2f}",
                        "End Price": f"${end_price:.2f}",
                        "Min/Max": f"${min_price:.2f} / ${max_price:.2f}",
                        "Change": f"{change_pct:+.1f}%",
                        "Similarity": f"{match['similarity']:.3f}"
                    })
                
                price_context_df = pd.DataFrame(price_context_data)
                
                # Display table
                st.dataframe(price_context_df, use_container_width=True)
                
                # Always show top match chart by default
                st.markdown("### 📈 Pattern Match Detail Chart")
                st.write("**Showing Top Match** | "
                        f"**Similarity:** {matches_sorted[0]['similarity']:.3f} | "
                        f"**Date Range:** {series_dates[matches_sorted[0]['start']].strftime('%Y-%m-%d')} to "
                        f"{series_dates[matches_sorted[0]['start'] + matches_sorted[0]['size'] - 1].strftime('%Y-%m-%d')}")
                
                # Use top match for chart
                selected_match = matches_sorted[0]
                
                # Create the similarity overlay chart (similar to fractal.py)
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                # Get match details
                m_start = selected_match['start']
                m_size = selected_match['size']
                m_similarity = selected_match['similarity']
                
                # Calculate extended range for context
                past_extension = max(1, int(m_size * 0.1))  # 10% of window size
                future_extension = max(1, int(m_size * 0.4))  # 40% of window size
                
                start_extended = max(0, m_start - past_extension)
                end_extended = min(len(price_series), m_start + m_size + future_extension)
                idx_range_extended = range(start_extended, end_extended)
                idx_range_match = range(m_start, m_start + m_size)
                
                # Plot the actual price data
                ax.plot(series_dates[idx_range_extended], price_series[idx_range_extended], 'lightblue', alpha=0.6, linewidth=1, label='Price Context')
                ax.plot(series_dates[idx_range_match], price_series[idx_range_match], 'blue', linewidth=2, label='Match Period')
                
                # Create a twin axis for the normalized pattern overlay
                ax2 = ax.twinx()
                
                # Scale and position the normalized patterns to overlay on the match period
                match_prices = price_series[idx_range_match]
                price_min, price_max = match_prices.min(), match_prices.max()
                price_range = price_max - price_min
                
                # Get the pattern dates for the match period
                pattern_dates = series_dates[idx_range_match]
                match_length = len(pattern_dates)
                
                # Resample reference pattern to match the length of the match period
                if len(reference_pattern) != match_length:
                    pattern_resampled = np.interp(
                        np.linspace(0, len(reference_pattern)-1, match_length), 
                        np.arange(len(reference_pattern)), 
                        reference_pattern_norm
                    )
                else:
                    pattern_resampled = reference_pattern_norm
                
                # Scale patterns to match price range
                pattern_scaled = pattern_resampled * price_range * 0.3 + price_min + price_range * 0.1
                
                # Plot normalized patterns on the twin axis
                ax2.plot(pattern_dates, pattern_scaled, 'darkred', linewidth=2, alpha=0.8, label='Reference Pattern (scaled)')
                
                # Styling for main axis
                match_start_date = series_dates[m_start]
                match_end_date = series_dates[m_start + m_size - 1]
                ax.set_title(f'Price Movement with Pattern Overlay - {ticker} ({timeframe})\n'
                            f'{series_dates[start_extended].strftime("%Y-%m-%d")} to {series_dates[end_extended-1].strftime("%Y-%m-%d")} | '
                            f'Similarity: {m_similarity:.3f}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)', color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(idx_range_extended)//10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Styling for twin axis
                ax2.set_ylabel('Normalized Pattern (scaled)', color='darkred')
                ax2.tick_params(axis='y', labelcolor='darkred')
                
                # Add vertical markers for match boundaries
                ax.axvline(x=match_start_date, color='green', alpha=0.5, linestyle=':', linewidth=2, label='Match Start')
                ax.axvline(x=match_end_date, color='red', alpha=0.5, linestyle=':', linewidth=2, label='Match End')
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add match details
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Match Start", match_start_date.strftime('%Y-%m-%d'))
                with col2:
                    st.metric("Match End", match_end_date.strftime('%Y-%m-%d'))
                with col3:
                    st.metric("Duration", f"{m_size} {timeframe_label}")
                with col4:
                    st.metric("Similarity Score", f"{m_similarity:.3f}")
                
            else:
                st.warning(f"No similar patterns found with similarity threshold {similarity_threshold:.2f}")
                st.info("Try lowering the similarity threshold or using a different pattern length.")
    
    else:
        # Show placeholders when no analysis has been run
        with tab1:
            st.markdown('<div class="section-header"><i class="fas fa-crystal-ball"></i> AI Stock Prediction</div>', unsafe_allow_html=True)
            st.info("👆 Click 'Run Complete Analysis' in the sidebar to see stock prediction results.")
        
        with tab2:
            st.markdown('<div class="section-header"><i class="fas fa-search"></i> Fractal Pattern Analysis</div>', unsafe_allow_html=True)
            st.info("👆 Click 'Run Complete Analysis' in the sidebar to see fractal pattern analysis results.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        ⚠️ <strong>Disclaimer</strong>: This tool is for educational purposes only. 
        Pattern analysis and predictions should not be used as sole investment advice. 
        Always consult with a financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()