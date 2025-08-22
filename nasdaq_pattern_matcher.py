#!/usr/bin/env python3
"""
NASDAQ Historical Pattern Matcher - Streamlit App

This app finds similar patterns within a single NASDAQ stock's own historical data 
that match the current 5-day pattern. It searches through ALL available historical 
data using the best available timeframe (1min, 3min, 5min, or 15min).

Features:
- Current 5-day pattern analysis for any NASDAQ stock
- Comprehensive historical pattern search within same stock
- Maximum historical data utilization (all available Yahoo data)
- Automatic timeframe selection based on data availability
- Advanced pattern similarity algorithms with multiple methods
- Visual pattern comparison showing historical matches + outcomes
- Pattern outcome prediction based on historical performance
- Success rate analysis and statistical insights

Usage:
    streamlit run nasdaq_pattern_matcher.py
"""

import streamlit as st
import yfinance as yf  # Needed for current pattern from Yahoo
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import concurrent.futures
import time
import math

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="QQQ Historical Pattern Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pattern-match {
        border: 2px solid #4CAF50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8fff8;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# NASDAQ 100 tickers (subset for performance)
NASDAQ_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA', 'ADBE', 'NFLX',
    'COST', 'AVGO', 'ASML', 'PEP', 'TMUS', 'CSCO', 'CMCSA', 'TXN', 'HON', 'QCOM',
    'AMGN', 'INTU', 'AMD', 'AMAT', 'ISRG', 'SBUX', 'BKNG', 'GILD', 'ADP', 'MDLZ',
    'VRTX', 'FISV', 'CSX', 'REGN', 'ATVI', 'CHTR', 'PYPL', 'NXPI', 'KLAC', 'MRNA',
    'LRCX', 'FTNT', 'ORLY', 'DXCM', 'CTAS', 'FAST', 'PAYX', 'ODFL', 'ROST', 'VRSK'
]

class PatternMatcher:
    """Advanced pattern matching algorithms for stock price data"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def normalize_price_series(self, prices: pd.Series) -> np.ndarray:
        """Normalize price series to 0-1 range"""
        return self.scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()
    
    def calculate_returns(self, prices: pd.Series) -> np.ndarray:
        """Calculate percentage returns"""
        return prices.pct_change().fillna(0).values
    
    def dtw_distance(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Dynamic Time Warping distance (simplified version)"""
        n, m = len(series1), len(series2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(series1[i-1] - series2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return dtw_matrix[n, m]
    
    def cosine_similarity_score(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Calculate cosine similarity between two series"""
        return cosine_similarity([series1], [series2])[0][0]
    
    def correlation_score(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient"""
        corr, _ = pearsonr(series1, series2)
        return corr if not np.isnan(corr) else 0
    
    def find_chart_patterns(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Identify common chart patterns using technical analysis"""
        patterns = {
            'double_top': [],
            'double_bottom': [],
            'head_shoulders': [],
            'triangles': [],
            'support_resistance': []
        }
        
        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            
            # Find peaks and valleys
            peaks, _ = find_peaks(high, distance=10, prominence=np.std(high) * 0.5)
            valleys, _ = find_peaks(-low, distance=10, prominence=np.std(low) * 0.5)
            
            patterns['peaks'] = peaks.tolist()
            patterns['valleys'] = valleys.tolist()
            
            # Simple double top detection
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    if abs(high[peaks[i]] - high[peaks[i+1]]) < np.std(high) * 0.2:
                        patterns['double_top'].extend([peaks[i], peaks[i+1]])
            
            # Simple double bottom detection
            if len(valleys) >= 2:
                for i in range(len(valleys) - 1):
                    if abs(low[valleys[i]] - low[valleys[i+1]]) < np.std(low) * 0.2:
                        patterns['double_bottom'].extend([valleys[i], valleys[i+1]])
            
        except Exception as e:
            st.warning(f"Pattern detection error: {e}")
        
        return patterns
    
    def calculate_pattern_similarity(self, target_data: pd.DataFrame, 
                                   candidate_data: pd.DataFrame,
                                   method: str = 'correlation') -> float:
        """Calculate similarity score between two datasets"""
        try:
            # Use closing prices for comparison
            target_prices = self.normalize_price_series(target_data['Close'])
            candidate_prices = self.normalize_price_series(candidate_data['Close'])
            
            # Ensure same length by taking minimum length
            min_len = min(len(target_prices), len(candidate_prices))
            target_prices = target_prices[-min_len:]
            candidate_prices = candidate_prices[-min_len:]
            
            if method == 'correlation':
                return self.correlation_score(target_prices, candidate_prices)
            elif method == 'cosine':
                return self.cosine_similarity_score(target_prices, candidate_prices)
            elif method == 'dtw':
                return 1 / (1 + self.dtw_distance(target_prices, candidate_prices))
            else:
                return 0
                
        except Exception as e:
            return 0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_maximum_historical_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load 5-minute historical data from qqq.us.txt file (trading days only)"""
    try:
        # Load data from local file
        data_file = "qqq.us.txt"  # Use relative path
        
        # Read CSV file
        data = pd.read_csv(data_file)
        
        if data.empty:
            st.error(f"❌ Could not load data from {data_file}")
            return None
        
        # Parse datetime from DATE and TIME columns  
        data['datetime_str'] = data['<DATE>'].astype(str) + ' ' + data['<TIME>'].astype(str).str.zfill(6)
        data['Datetime'] = pd.to_datetime(data['datetime_str'], format='%Y%m%d %H%M%S')
        
        # Rename columns to match expected format
        data = data.rename(columns={
            '<OPEN>': 'Open',
            '<HIGH>': 'High', 
            '<LOW>': 'Low',
            '<CLOSE>': 'Close',
            '<VOL>': 'Volume'
        })
        
        # Keep only needed columns
        data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Filter out weekends (Saturday=5, Sunday=6)
        data['weekday'] = data['Datetime'].dt.weekday
        trading_data = data[data['weekday'] < 5].copy()  # Monday=0 to Friday=4
        
        if len(trading_data) < 1000:  # Need substantial 5-min bars for pattern matching
            st.error(f"❌ Insufficient 5-minute trading data: {len(trading_data)}")
            return None
            
        trading_data['timeframe'] = '5m'
        
        # Remove helper columns
        trading_data = trading_data.drop(columns=['weekday'])
        
        # Sort by datetime
        trading_data = trading_data.sort_values('Datetime').reset_index(drop=True)
        
        # Calculate data span
        start_date = trading_data['Datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = trading_data['Datetime'].iloc[-1].strftime('%Y-%m-%d')
        
        # Show messages with auto-fade CSS
        st.success(f"✅ Using local 5-minute QQQ data from file")
        st.info(f"📅 Data span: {start_date} to {end_date} ({len(trading_data):,} bars)")
        
        # Add CSS to fade out messages after 3 seconds
        st.markdown("""
        <style>
        div[data-testid="stAlert"] {
            animation: fadeOut 1s ease-in-out 3s forwards;
        }
        
        @keyframes fadeOut {
            0% { opacity: 1; }
            100% { opacity: 0; display: none; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        return trading_data
        
    except Exception as e:
        st.error(f"❌ Error loading historical data from file: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute  
def fetch_current_pattern_from_file(ticker: str, days: int = 5) -> Optional[pd.DataFrame]:
    """Get current 5 trading days pattern from local file (most recent data)"""
    try:
        # Get all historical data from file
        all_data = fetch_maximum_historical_data(ticker)
        
        if all_data is None or len(all_data) < 100:
            return None
        
        # Get unique trading days
        all_data['date_only'] = all_data['Datetime'].dt.date
        unique_days = sorted(all_data['date_only'].unique())
        
        if len(unique_days) < days:
            return None
            
        # Take the last N trading days
        last_days = unique_days[-days:]
        pattern_data = all_data[all_data['date_only'].isin(last_days)].copy()
        
        # Add timeframe info
        pattern_data['timeframe'] = '5m'
        
        # Remove helper columns
        pattern_data = pattern_data.drop(columns=['date_only'])
        
        # Sort by datetime
        pattern_data = pattern_data.sort_values('Datetime').reset_index(drop=True)
        
        # Show data info
        start_date = pattern_data['Datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = pattern_data['Datetime'].iloc[-1].strftime('%Y-%m-%d')
        st.success(f"✅ Current pattern (from file): {start_date} to {end_date} ({len(pattern_data)} bars)")
        
        return pattern_data
        
    except Exception as e:
        st.error(f"Error fetching current pattern from file: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute  
def fetch_current_pattern_from_yahoo(ticker: str, days: int = 5) -> Optional[pd.DataFrame]:
    """Get current 5 trading days pattern from Yahoo Finance (real-time/recent data)"""
    try:
        # Use Yahoo Finance to get the most recent 5-minute data
        # Yahoo allows up to 60 days of 5-minute data
        st.info(f"📡 Fetching current {days}-day pattern from Yahoo Finance...")
        
        # Download recent data from Yahoo Finance
        yf_data = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)
        
        if yf_data.empty:
            st.error(f"❌ No current data available from Yahoo Finance for {ticker}")
            return None
        
        # Reset index to get datetime as column
        yf_data.reset_index(inplace=True)
        
        # Handle multi-level column names from yfinance
        if yf_data.columns.nlevels > 1:
            yf_data.columns = [col[0] if col[1] == ticker else col[0] for col in yf_data.columns]
        
        # Rename columns to match expected format
        yf_data = yf_data.rename(columns={
            'Datetime': 'Datetime',
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Convert UTC timestamps to Eastern Time before filtering
        yf_data['Datetime'] = yf_data['Datetime'].dt.tz_convert('America/New_York')
        
        # Filter to market hours only (9:30 AM - 4:00 PM EST)
        yf_data = yf_data[
            ((yf_data['Datetime'].dt.hour == 9) & (yf_data['Datetime'].dt.minute >= 30)) |
            ((yf_data['Datetime'].dt.hour >= 10) & (yf_data['Datetime'].dt.hour < 16))
        ]
        
        # Filter out weekends
        yf_data = yf_data[yf_data['Datetime'].dt.weekday < 5]
        
        if len(yf_data) < 100:
            st.error(f"❌ Insufficient current data from Yahoo: {len(yf_data)} bars")
            return None
        
        # Get unique trading days
        yf_data['date_only'] = yf_data['Datetime'].dt.date
        unique_days = sorted(yf_data['date_only'].unique())
        
        if len(unique_days) < days:
            st.error(f"❌ Insufficient trading days from Yahoo: {len(unique_days)} days")
            return None
            
        # Take the last N trading days
        last_days = unique_days[-days:]
        pattern_data = yf_data[yf_data['date_only'].isin(last_days)].copy()
        
        # Add timeframe info
        pattern_data['timeframe'] = '5m'
        
        # Remove helper columns
        pattern_data = pattern_data.drop(columns=['date_only'])
        
        # Sort by datetime
        pattern_data = pattern_data.sort_values('Datetime').reset_index(drop=True)
        
        # Show data info
        start_date = pattern_data['Datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = pattern_data['Datetime'].iloc[-1].strftime('%Y-%m-%d')
        st.success(f"✅ Current pattern (from Yahoo): {start_date} to {end_date} ({len(pattern_data)} bars)")
        
        return pattern_data
        
    except Exception as e:
        st.error(f"Error fetching current pattern from Yahoo Finance: {e}")
        return None

def fetch_current_pattern(ticker: str, days: int = 5, use_yahoo: bool = True) -> Optional[pd.DataFrame]:
    """
    Get current pattern data from either Yahoo Finance or local file
    
    Args:
        ticker: Stock ticker symbol
        days: Number of trading days for pattern
        use_yahoo: If True, use Yahoo Finance; if False, use local file
    
    Returns:
        DataFrame with current pattern data
    """
    if use_yahoo:
        return fetch_current_pattern_from_yahoo(ticker, days)
    else:
        return fetch_current_pattern_from_file(ticker, days)

def search_historical_patterns(ticker: str, current_pattern: pd.DataFrame, 
                             similarity_threshold: float = 0.7, 
                             max_patterns: int = 20,
                             outcome_days: int = 5) -> List[Dict]:
    """Search for historical 5-day patterns in 5-minute data similar to current pattern"""
    
    matches = []
    
    try:
        # Fetch maximum available historical data (5-minute trading data)
        historical_data = fetch_maximum_historical_data(ticker)
        
        if historical_data is None or len(historical_data) < 1000:  # Need substantial 5-min data
            st.warning(f"Insufficient historical 5-minute data for {ticker}")
            return matches
        
        # Get the pattern length (number of 5-min bars in current pattern)
        pattern_length = len(current_pattern)
        
        # Create pattern matcher
        matcher = PatternMatcher()
        
        # Group historical data by date to find 5-day patterns
        historical_data['date_only'] = historical_data['Datetime'].dt.date
        unique_dates = sorted(historical_data['date_only'].unique())
        
        # Search through historical data looking for 5-day periods
        for start_date_idx in range(len(unique_dates) - outcome_days - 5):
            # Get 5 consecutive trading days for pattern
            pattern_dates = unique_dates[start_date_idx:start_date_idx + 5]
            historical_window = historical_data[historical_data['date_only'].isin(pattern_dates)].copy()
            
            # Skip if not enough data for this 5-day period
            if len(historical_window) < pattern_length * 0.8:  # Allow some flexibility
                continue
            
            # Skip if this window overlaps with current pattern (recent data)
            # Skip last 10 trading days to avoid overlap
            if start_date_idx >= len(unique_dates) - 15:
                continue
            
            # Truncate to match current pattern length if needed
            if len(historical_window) > pattern_length:
                historical_window = historical_window.iloc[:pattern_length].copy()
            elif len(historical_window) < pattern_length:
                continue
            
            # Calculate similarity
            similarity = matcher.calculate_pattern_similarity(
                current_pattern, historical_window, method='correlation'
            )
            
            if similarity >= similarity_threshold:
                # Get outcome data (next 2 trading days for cleaner visualization)
                outcome_days_to_show = 2
                outcome_start_idx = start_date_idx + 5
                outcome_end_idx = min(outcome_start_idx + outcome_days_to_show, len(unique_dates))
                
                if outcome_end_idx - outcome_start_idx >= outcome_days_to_show:
                    outcome_dates = unique_dates[outcome_start_idx:outcome_end_idx]
                    future_data = historical_data[historical_data['date_only'].isin(outcome_dates)].copy()
                    
                    if len(future_data) > 0:
                        # Calculate pattern outcome
                        pattern_start_price = historical_window['Close'].iloc[0]
                        pattern_end_price = historical_window['Close'].iloc[-1]
                        future_end_price = future_data['Close'].iloc[-1]
                        
                        pattern_change = ((pattern_end_price - pattern_start_price) / pattern_start_price) * 100
                        future_change = ((future_end_price - pattern_end_price) / pattern_end_price) * 100
                        
                        matches.append({
                            'start_index': start_date_idx,
                            'end_index': start_date_idx + 5,
                            'similarity': similarity,
                            'historical_pattern': historical_window.drop(columns=['date_only']),
                            'future_outcome': future_data.drop(columns=['date_only']),
                            'pattern_start_date': historical_window['Datetime'].iloc[0],
                            'pattern_end_date': historical_window['Datetime'].iloc[-1],
                            'pattern_change_pct': pattern_change,
                            'future_change_pct': future_change,
                            'future_high': future_data['High'].max(),
                            'future_low': future_data['Low'].min(),
                            'pattern_start_price': pattern_start_price,
                            'pattern_end_price': pattern_end_price,
                            'future_end_price': future_end_price
                        })
        
        # Sort by similarity and limit results
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:max_patterns]
        
    except Exception as e:
        st.error(f"Error searching historical patterns for {ticker}: {e}")
        return matches


def create_price_chart(data: pd.DataFrame, title: str) -> go.Figure:
    """Create a simple price line chart (no volume)"""
    fig = go.Figure()
    
    # Simple line chart using Close prices
    fig.add_trace(
        go.Scatter(
            x=data['Datetime'],
            y=data['Close'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Current Pattern'
        )
    )
    
    fig.update_layout(
        title=title,
        yaxis_title='Price ($)',
        xaxis_title='Time',
        height=520,  # Increased by 30% (400 * 1.3)
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=False))
    )
    
    return fig

def create_historical_matches_chart(current_pattern: pd.DataFrame, historical_matches: list, title: str, selected_pattern_idx: int = None) -> go.Figure:
    """Create a chart showing current pattern in blue and historical matches in grey"""
    fig = go.Figure()
    
    # Filter matches with similarity > 80%
    high_similarity_matches = [match for match in historical_matches if match['similarity'] > 0.8]
    
    if not high_similarity_matches:
        # Show empty chart with message
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No historical matches found with similarity > 80%",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title=title,
            height=400,
            template='plotly_white'
        )
        return fig
    
    # Normalize all matches to start from the same price point
    base_price = 100  # Use $100 as base for normalization
    
    # Show top 4 matches, but always include selected pattern if specified
    if selected_pattern_idx is not None and selected_pattern_idx < len(high_similarity_matches):
        # Ensure selected pattern is included
        selected_match = high_similarity_matches[selected_pattern_idx]
        top_matches = high_similarity_matches[:4].copy()
        
        # If selected pattern is not in top 4, replace the 4th one with the selected
        if selected_pattern_idx >= 4:
            top_matches[3] = selected_match
            # Update the selected_pattern_idx to point to position 3 in the display
            selected_pattern_idx = 3
    else:
        top_matches = high_similarity_matches[:4]
    
    # Add historical matches first (so current pattern appears on top)
    for idx, match in enumerate(top_matches):
        historical_data = match['historical_pattern']
        future_data = match['future_outcome']
        
        # Limit future outcome to reasonable length (max 20% of pattern length)
        max_outcome_length = max(20, int(len(historical_data) * 0.2))  # At least 20 bars, max 20% of pattern
        if len(future_data) > max_outcome_length:
            future_data = future_data.iloc[:max_outcome_length].copy()
        
        # Combine historical pattern + future outcome for extended view
        combined_data = pd.concat([historical_data, future_data], ignore_index=True)
        
        # Normalize the data to start from base price
        historical_start_price = historical_data['Close'].iloc[0]
        normalized_combined = (combined_data['Close'] / historical_start_price) * base_price
        
        # Create x-axis positions
        pattern_length = len(historical_data)
        x_values = list(range(len(combined_data)))
        
        # Split into pattern part (5 days) and outcome part (extra days)
        pattern_x = x_values[:pattern_length]
        pattern_y = normalized_combined[:pattern_length]
        outcome_x = x_values[pattern_length-1:]  # Include last point of pattern for continuity
        outcome_y = normalized_combined[pattern_length-1:]
        
        # Check if this pattern is selected for highlighting
        is_selected = (selected_pattern_idx is not None and idx == selected_pattern_idx)
        
        if is_selected:
            # Highlight selected pattern with bright color
            base_color = 'rgb(255, 140, 0)'  # Orange
            outcome_color = 'rgb(255, 165, 0)'  # Light orange
            width = 3.0
            opacity = 1.0
        else:
            # Use 4 distinct grey color scales for non-selected patterns
            grey_levels = [
                60,   # Pattern 1: Darkest grey (highest similarity)
                100,  # Pattern 2: Dark grey
                140,  # Pattern 3: Medium grey  
                180   # Pattern 4: Light grey (lowest similarity)
            ]
            
            grey_intensity = grey_levels[idx]
            base_color = f'rgb({grey_intensity},{grey_intensity},{grey_intensity})'
            outcome_color = f'rgb({min(255, grey_intensity + 30)},{min(255, grey_intensity + 30)},{min(255, grey_intensity + 30)})'
            
            # Line width and opacity based on pattern rank
            line_styles = [
                {'width': 2.0, 'opacity': 0.9},   # Pattern 1: Thickest, most opaque
                {'width': 1.8, 'opacity': 0.85},  # Pattern 2
                {'width': 1.5, 'opacity': 0.8},   # Pattern 3 
                {'width': 1.2, 'opacity': 0.7}    # Pattern 4: Thinnest, most transparent
            ]
            
            style = line_styles[idx]
            width = style['width']
            opacity = style['opacity']
        
        # Create a unique legend group for this match
        legend_group = f'group_{idx}'
        legend_name = f'{match["pattern_start_date"].strftime("%Y-%m-%d")} (S:{match["similarity"]:.3f})'
        
        # Add the 5-day pattern part
        fig.add_trace(
            go.Scatter(
                x=pattern_x,
                y=pattern_y,
                mode='lines',
                line=dict(color=base_color, width=width),
                name=legend_name,
                legendgroup=legend_group,
                showlegend=True,  # Show legend for all 4
                opacity=opacity,
                hovertemplate=f'<b>Date:</b> {match["pattern_start_date"].strftime("%Y-%m-%d")}<br>' +
                             f'<b>Similarity:</b> {match["similarity"]:.3f}<br>' +
                             f'<b>Price:</b> $%{{y:.2f}}<br>' +
                             f'<b>Period:</b> %{{x}} (Pattern)<extra></extra>'
            )
        )
        
        # Add the outcome part (what happened after)
        if len(outcome_x) > 1:  # Only add if there's actual outcome data
            fig.add_trace(
                go.Scatter(
                    x=outcome_x,
                    y=outcome_y,
                    mode='lines',
                    line=dict(color=outcome_color, width=width, dash='dot'),
                    name=legend_name,  # Same name to group with pattern
                    legendgroup=legend_group,  # Same group as pattern
                    showlegend=False,  # Don't show separate legend entry
                    opacity=opacity * 0.7,  # Slightly more transparent
                    hovertemplate=f'<b>Date:</b> {match["pattern_start_date"].strftime("%Y-%m-%d")}<br>' +
                                 f'<b>Similarity:</b> {match["similarity"]:.3f}<br>' +
                                 f'<b>Price:</b> $%{{y:.2f}}<br>' +
                                 f'<b>Period:</b> %{{x}} (Outcome)<extra></extra>'
                )
            )
    
    # Add current pattern with alternating blue/light blue for each day (properly aligned)
    current_start_price = current_pattern['Close'].iloc[0]
    
    # Group current pattern by trading days
    current_pattern_copy = current_pattern.copy().reset_index(drop=True)
    current_pattern_copy['date_only'] = current_pattern_copy['Datetime'].dt.date
    unique_current_dates = sorted(current_pattern_copy['date_only'].unique())
    
    # Add each day separately with alternating colors but continuous x-positioning
    x_position = 0
    for day_idx, date in enumerate(unique_current_dates):
        day_data = current_pattern_copy[current_pattern_copy['date_only'] == date]
        
        if len(day_data) == 0:
            continue
            
        # Create continuous x-axis positions for this day
        day_x_values = list(range(x_position, x_position + len(day_data)))
        x_position += len(day_data)
        
        # Normalize prices for this day
        day_normalized = (day_data['Close'] / current_start_price) * base_price
        
        # Alternate between blue and light blue
        color = 'blue' if day_idx % 2 == 0 else 'lightblue'
        
        fig.add_trace(
            go.Scatter(
                x=day_x_values,
                y=day_normalized,
                mode='lines',
                line=dict(color=color, width=3),
                name='Current Pattern' if day_idx == 0 else '',
                legendgroup='current_pattern',  # Group all current pattern days
                showlegend=day_idx == 0,  # Only show legend for first day
                opacity=1.0,
                hovertemplate=f'<b>Current Pattern - Day {day_idx + 1}</b><br>' +
                             f'<b>Date:</b> {date}<br>' +
                             '<b>Price:</b> $%{y:.2f}<br>' +
                             '<b>Period:</b> %{x}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title=title,
        yaxis_title='Normalized Price ($)',
        xaxis_title='5-Minute Intervals (5 Trading Days)',
        height=780,  # Increased by 30% (600 * 1.3)
        showlegend=True,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 QQQ Historical Pattern Matcher</h1>', unsafe_allow_html=True)
    st.markdown("**Find similar patterns in QQQ's own historical data**")
    st.markdown("*Real-time Yahoo data • Historical patterns from comprehensive QQQ dataset*")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎛️ Configuration")
        
        # Fixed to QQQ (NASDAQ ETF)
        target_ticker = "QQQ"
        st.info("📈 **Target Stock:** QQQ (NASDAQ-100 ETF)")
        
        # Data source info (Yahoo Finance is now the default)
        st.subheader("📡 Data Sources")
        st.success("📡 **Current**: Yahoo Finance (real-time) • **Historical**: qqq.us.txt")
        st.caption("⚡ Fresh data from Yahoo API (up to 15-30 min delay)")
        
        # Pattern matching settings
        st.subheader("🔍 Pattern Settings")
        
        pattern_days = st.slider(
            "Current Pattern Length (days)",
            min_value=3,
            max_value=15,
            value=5,
            step=1,
            help="Number of recent trading days to use as current pattern"
        )
        
        st.info("**Similarity Algorithm:** Correlation (Pearson correlation coefficient)")
        st.caption("Uses linear relationship analysis for pattern matching")
        
        min_similarity = st.slider(
            "Minimum Similarity Score",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Minimum similarity score to consider a historical match (default: 80%)"
        )
        
        # Historical search settings
        st.subheader("📊 Historical Search")
        
        max_historical_patterns = st.number_input(
            "Max Historical Patterns",
            min_value=10,
            max_value=100,
            value=30,
            help="Maximum number of historical patterns to find and analyze"
        )
        
        
        # Advanced settings
        st.subheader("⚙️ Advanced Settings")
        
        outcome_days = st.slider(
            "Outcome Analysis Period (days)",
            min_value=1,
            max_value=30,
            value=5,
            help="How many days after pattern to analyze for outcomes"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh (5 min)", value=False)
        
        if st.button("🔍 Find QQQ Historical Patterns", type="primary"):
            st.session_state.run_analysis = True
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Main content area
    if 'run_analysis' not in st.session_state:
        st.info("👈 Configure your settings in the sidebar and click 'Find QQQ Historical Patterns' to start analysis!")
        st.markdown("### 🔍 Ready to Analyze QQQ Patterns")
        st.markdown("""
        This tool will:
        - 📡 **Current Pattern**: Load 5 trading days from Yahoo Finance (real-time data)
        - 📁 **Historical Data**: Search through ALL available historical QQQ data from qqq.us.txt  
        - 🎯 Show top 4 matches with >80% similarity in grey (dark grey for >90%, light grey for 80-90%)
        - 📊 Display current pattern in blue with 2-day outcome predictions (dotted lines)
        - 🔄 **Optimal Performance**: Real-time current data + comprehensive historical patterns
        """)
        return
    
    # Pattern matching analysis
    with st.spinner("🔍 Fetching current pattern and analyzing..."):
        
        # Fetch current pattern for target stock (using Yahoo Finance)
        current_pattern = fetch_current_pattern(target_ticker, pattern_days, use_yahoo=True)
        
        if current_pattern is None or current_pattern.empty:
            st.error(f"❌ Could not fetch current {pattern_days}-day pattern for QQQ")
            return
    
    # Display pattern info (no chart)
    st.subheader(f"🎯 Current {pattern_days}-Day QQQ Pattern Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_pattern['Close'].iloc[-1]:.2f}")
    with col2:
        price_change = current_pattern['Close'].iloc[-1] - current_pattern['Close'].iloc[0]
        st.metric("Pattern Change", f"${price_change:.2f}", f"{(price_change/current_pattern['Close'].iloc[0]*100):.2f}%")
    with col3:
        timeframe = current_pattern['timeframe'].iloc[0] if 'timeframe' in current_pattern.columns else "Unknown"
        st.metric("Timeframe", timeframe)
    with col4:
        st.metric("Data Points", f"{len(current_pattern)}")
    
    # Historical pattern analysis
    st.subheader("🔍 Searching Historical Patterns...")
    
    with st.spinner("Searching through maximum available historical data..."):
        historical_matches = search_historical_patterns(
            target_ticker, 
            current_pattern, 
            min_similarity, 
            max_historical_patterns,
            outcome_days
        )
        
    if not historical_matches:
        st.warning(f"⚠️ No historical patterns found with similarity >= {min_similarity:.2f}")
        st.info("💡 Try lowering the minimum similarity score or increasing pattern length")
    else:
        st.success(f"✅ Found {len(historical_matches)} historical patterns!")
        
        # Display historical matches
        st.subheader("📊 Historical Pattern Matches")
        
        # Summary statistics (for all matches, not just >80%)
        avg_similarity = np.mean([m['similarity'] for m in historical_matches])
        avg_future_change = np.mean([m['future_change_pct'] for m in historical_matches])
        positive_outcomes = len([m for m in historical_matches if m['future_change_pct'] > 0])
        
        # High similarity statistics
        high_similarity_matches = [match for match in historical_matches if match['similarity'] > 0.8]
        if high_similarity_matches:
            high_sim_avg_change = np.mean([m['future_change_pct'] for m in high_similarity_matches])
            high_sim_positive = len([m for m in high_similarity_matches if m['future_change_pct'] > 0])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(historical_matches))
        with col2:
            st.metric("High Similarity (>80%)", len(high_similarity_matches) if high_similarity_matches else 0)
        with col3:
            if high_similarity_matches:
                st.metric(f"High Sim Avg {outcome_days}d Change", f"{high_sim_avg_change:+.2f}%")
            else:
                st.metric(f"High Sim Avg {outcome_days}d Change", "N/A")
        with col4:
            if high_similarity_matches:
                st.metric("High Sim Success Rate", f"{(high_sim_positive/len(high_similarity_matches)*100):.1f}%")
            else:
                st.metric("High Sim Success Rate", "N/A")
        
        # Create summary table first (show only high similarity matches)
        if high_similarity_matches:
            st.subheader("📋 High Similarity Matches Summary (>80%)")
            summary_data = []
            for i, match in enumerate(high_similarity_matches):
                summary_data.append({
                    '#': i + 1,
                    'Date': match['pattern_start_date'].strftime('%Y-%m-%d'),
                    'Similarity': f"{match['similarity']:.3f}",
                    'Pattern Change': f"{match['pattern_change_pct']:+.2f}%",
                    f'{outcome_days}d Future Change': f"{match['future_change_pct']:+.2f}%",
                    'Future High': f"+{((match['future_high'] - match['pattern_end_price'])/match['pattern_end_price']*100):.2f}%",
                    'Future Low': f"{((match['future_low'] - match['pattern_end_price'])/match['pattern_end_price']*100):.2f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Add selection functionality
            st.markdown("**Click on a row to highlight that pattern on the chart:**")
            selected_rows = st.dataframe(
                summary_df, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Store selected pattern in session state
            if selected_rows.selection.rows:
                selected_idx = selected_rows.selection.rows[0]
                st.session_state.selected_pattern_idx = selected_idx
                st.success(f"✅ Selected pattern #{selected_idx + 1}: {summary_data[selected_idx]['Date']} (Similarity: {summary_data[selected_idx]['Similarity']})")
            elif 'selected_pattern_idx' not in st.session_state:
                st.session_state.selected_pattern_idx = None
        else:
            st.warning("⚠️ No patterns found with similarity > 80%. Try lowering the similarity threshold.")
        
        # Show historical matches chart (similarity > 80% only) - after table selection
        st.subheader("📈 Historical Pattern Matches (Similarity > 80%)")
        
        # Create chart showing current pattern in blue and historical matches in grey
        selected_idx = getattr(st.session_state, 'selected_pattern_idx', None)
        historical_fig = create_historical_matches_chart(current_pattern, historical_matches, 
                                                       f"QQQ Pattern Comparison - Current vs Historical Matches + Outcomes",
                                                       selected_pattern_idx=selected_idx)
        st.plotly_chart(historical_fig, use_container_width=True)
        
        st.info(f"💡 **Showing top 4 patterns (+ selected pattern if beyond top 4) of {len(high_similarity_matches)} patterns with similarity > 80%** | " +
                f"🔵 **Blue/Light Blue**: Current Pattern (alternating by day) | ⚫ **4 Grey Scales**: Darkest = Rank 1 (highest similarity), Lightest = Rank 4 + 2-day outcomes (dotted) | 🟠 **Orange**: Selected pattern from table | Similarity range: {min([m['similarity'] for m in high_similarity_matches]):.3f} to {max([m['similarity'] for m in high_similarity_matches]):.3f}")
        
        # Pattern prediction insights
        st.subheader("🔮 Pattern Prediction Summary")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info(f"""
            **📊 Historical Analysis Results:**
            - Total Patterns Found: {len(historical_matches)}
            - Average Similarity: {avg_similarity:.3f}
            - Success Rate: {(positive_outcomes/len(historical_matches)*100):.1f}%
            - Expected {outcome_days}d Change: {avg_future_change:+.2f}%
            - Timeframe Used: {current_pattern['timeframe'].iloc[0] if 'timeframe' in current_pattern.columns else 'Unknown'}
            """)
        
        with insights_col2:
            # Create outcome distribution chart
            outcomes = [m['future_change_pct'] for m in historical_matches]
            fig_dist = px.histogram(
                x=outcomes,
                title=f"{outcome_days}-Day Future Returns Distribution",
                labels={'x': 'Future Return %', 'y': 'Count'},
                nbins=15
            )
            fig_dist.add_vline(
                x=avg_future_change,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_future_change:+.2f}%"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🔍 QQQ Historical Pattern Matcher** - Current data: Yahoo Finance API • Historical data: qqq.us.txt • Built with Streamlit")
    
    # Debug information
    if st.checkbox("🐛 Show Debug Info"):
        st.subheader("Debug Information")
        st.write("**Current Pattern Shape:**", current_pattern.shape)
        st.write("**Pattern Days:**", pattern_days)
        st.write("**Target Stock:**", "QQQ")
        st.write("**Analysis Time:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with st.expander("Current Pattern Data Sample"):
            st.dataframe(current_pattern.head())
            
        if 'historical_matches' in locals():
            st.write("**Historical Matches Found:**", len(historical_matches))
            if historical_matches:
                st.write("**Data Timeframe:**", historical_matches[0]['historical_pattern']['timeframe'].iloc[0] if 'timeframe' in historical_matches[0]['historical_pattern'].columns else 'Unknown')
                st.write("**Outcome Analysis Days:**", outcome_days)

if __name__ == "__main__":
    main()