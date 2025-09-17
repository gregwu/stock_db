#!/usr/bin/env python3
"""
Merged Fractal Pattern Analysis and Stock Prediction Interface
Combines fractal pattern matching with AI-powered stock predictions
"""

import streamlit as st

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Fractal Pattern Analysis & Stock Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import multiprocessing as mp
import joblib
import os
import warnings
import json
import tempfile
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import subprocess
from sklearn.model_selection import train_test_split
from io import StringIO
import yfinance as yf
from datetime import timedelta
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
#import psycopg2
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
from technical_indicators import TechnicalIndicators
from config import features

# Load environment variables
#load_dotenv()

# Database configuration
#DB_CONFIG = {
#    'host': os.getenv('DB_HOST', 'localhost'),
#    'port': int(os.getenv('DB_PORT', 5433)),
#    'database': os.getenv('DB_NAME', 'database'),
#    'user': os.getenv('DB_USER', 'user'),
#    'password': os.getenv('DB_PASSWORD', 'password')
#}

# Technical Analysis Indicators Class

def create_technical_analysis_chart(df, symbol, seasonal_years=2, chart_type="line", show_macd=True, performance_mode=False):
    """Create comprehensive technical analysis chart"""
    if df is None or len(df) == 0:
        st.error("No data available for technical analysis")
        return None
    
    # Use different amounts of data based on performance mode and seasonal_years
    if performance_mode:
        # Performance mode: use reduced data but respect seasonal_years
        chart_days = max(100, seasonal_years * 100)  # At least 100 days per year requested
    else:
        # Normal mode: use seasonal_years to determine chart period
        chart_days = seasonal_years * 252  # ~252 trading days per year
    
    if len(df) > chart_days:
        df_chart = df.tail(chart_days)
    else:
        df_chart = df
    
    # Check if we have enough data for technical indicators
    if len(df_chart) < 20:
        st.warning(f"⚠️ Limited data ({len(df_chart)} days) for technical analysis. Some indicators may not be available.")
        # Create a basic price chart without technical indicators
        return create_basic_price_chart(df_chart, symbol, chart_type)
    
    # Calculate indicators using the chart data
    close = df_chart['close']
    
    # In performance mode, downsample data for chart rendering
    if performance_mode and len(df_chart) > 50:
        # Sample every other data point to reduce rendering load
        df_chart = df_chart.iloc[::2]
        close = df_chart['close']
    
    # Bollinger Bands (20-period, 2 std dev)
    bb_sma, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(close, 20, 2)
    
    # SMA 144 - calculated on full dataset
    sma_144 = TechnicalIndicators.sma(close, 144)
    
    # Price Distance to MA (Fast: 20-period)
    pma_fast, pma_fast_signal, pma_fast_cycle = TechnicalIndicators.price_distance_to_ma(
        close, ma_length=20, signal_length=9, exponential=False
    )
    
    # PMA Threshold Bands
    pma_bands = TechnicalIndicators.pma_threshold_bands(pma_fast)
    
    # MACD (12, 26, 9)
    macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(close, 12, 26, 9)
    
    # RSI (14-period)
    rsi = TechnicalIndicators.rsi(close, 14)
    
    # Create indicators dataframe
    indicators = pd.DataFrame({
        'bb_sma': bb_sma,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'sma_144': sma_144,
        'pma_fast': pma_fast,
        'pma_fast_signal': pma_fast_signal,
        'pma_fast_cycle': pma_fast_cycle,
        'pma_upper_low': pma_bands['upper_low'],
        'pma_lower_low': pma_bands['lower_low'],
        'pma_upper_high': pma_bands['upper_high'],
        'pma_lower_high': pma_bands['lower_high'],
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,
        'rsi': rsi
    }, index=df_chart.index)
    
    # Create the chart
    fig = go.Figure()
    
    # 1. Main price chart with candlesticks
    daily_changes_pct = df_chart['close'].pct_change() * 100
    daily_changes_dollar = df_chart['close'].diff()
    
    # Determine volume column name
    volume_col = None
    if 'volume' in df_chart.columns:
        volume_col = 'volume'
    elif 'vol' in df_chart.columns:
        volume_col = 'vol'
    
    # Create custom hover text
    hover_text = []
    for i, (date, row) in enumerate(df_chart.iterrows()):
        if i == 0:
            change_text = "N/A"
        else:
            pct_change = daily_changes_pct.iloc[i]
            dollar_change = daily_changes_dollar.iloc[i]
            change_text = f"{pct_change:+.2f}% (${dollar_change:+.2f})"
        
        volume_text = f"Volume: {row[volume_col]:,.0f}<br>" if volume_col and volume_col in row else ""
        hover_text.append(
            f"Date: {date.strftime('%Y-%m-%d')}<br>"
            f"Open: ${row['open']:.2f}<br>"
            f"High: ${row['high']:.2f}<br>"
            f"Low: ${row['low']:.2f}<br>"
            f"Close: ${row['close']:.2f}<br>"
            f"{volume_text}"
            f"Daily Change: {change_text}"
        )
    
    # Price chart - either line or candlestick based on chart_type
    if chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart['open'],
                high=df_chart['high'],
                low=df_chart['low'],
                close=df_chart['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red',
                hovertext=hover_text,
                hoverinfo='text'
            )
        )
    else:
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart['close'],
                mode='lines',
                line=dict(color='black', width=2),
                name='Price',
                hovertext=hover_text,
                hoverinfo='text'
            )
        )
    
    # 2. Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['bb_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='blue', width=1),
            hovertemplate='BB Upper: $%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['bb_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='blue', width=1),
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.1)',
            hovertemplate='BB Lower: $%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['bb_sma'],
            mode='lines',
            name='BB Middle (SMA 20)',
            line=dict(color='orange', width=1),
            hovertemplate='BB Middle: $%{y:.2f}<extra></extra>'
        )
    )
    
    # 3. SMA 144 trend line
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['sma_144'],
            mode='lines',
            name='SMA 144 (Trend)',
            line=dict(color='purple', width=3),
            hovertemplate='SMA 144: $%{y:.2f}<extra></extra>'
        )
    )
    
    # 4. Volume chart (using secondary y-axis) - only if volume data exists
    if volume_col:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df_chart['close'], df_chart['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart[volume_col],
                name='Volume',
                marker_color=colors,
                opacity=0.3,
                yaxis='y2',
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            )
        )
    
    # 5. Price Distance to MA indicator
    # PMA threshold bands (background fills)
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['pma_upper_high'],
            mode='lines',
            name='Overbought High',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            showlegend=False,
            yaxis='y3'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['pma_upper_low'],
            mode='lines',
            name='Overbought Low',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.15)',
            showlegend=False,
            yaxis='y3'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['pma_lower_low'],
            mode='lines',
            name='Oversold High',
            line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
            showlegend=False,
            yaxis='y3'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['pma_lower_high'],
            mode='lines',
            name='Oversold Low',
            line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.15)',
            showlegend=False,
            yaxis='y3'
        )
    )
    
    # PMA Fast line
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['pma_fast'],
            mode='lines',
            name='Price/MA %',
            line=dict(color='blue', width=1),
            hovertemplate='Price/MA: %{y:.2f}%<extra></extra>',
            yaxis='y3'
        )
    )
    
    # PMA Signal line
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['pma_fast_signal'],
            mode='lines',
            name='Signal Line',
            line=dict(color='red', width=1),
            hovertemplate='Signal: %{y:.2f}%<extra></extra>',
            yaxis='y3'
        )
    )
    
    # Cycle histogram (as bar chart)
    cycle_colors = ['green' if x > 0 else 'red' for x in indicators['pma_fast_cycle']]
    fig.add_trace(
        go.Bar(
            x=df_chart.index,
            y=indicators['pma_fast_cycle'],
            name='Cycle Histogram',
            marker_color=cycle_colors,
            opacity=0.6,
            hovertemplate='Cycle: %{y:.2f}%<extra></extra>',
            yaxis='y3'
        )
    )
    
    # 6. MACD indicator (conditional)
    if show_macd:
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=indicators['macd_line'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1),
                hovertemplate='MACD: %{y:.4f}<extra></extra>',
                yaxis='y4'
            )
        )
        
        # MACD Signal line
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=indicators['macd_signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=1),
                hovertemplate='MACD Signal: %{y:.4f}<extra></extra>',
                yaxis='y4'
            )
        )
        
        # MACD Histogram - separate positive and negative bars
        positive_hist = indicators['macd_hist'].where(indicators['macd_hist'] > 0, 0)
        negative_hist = indicators['macd_hist'].where(indicators['macd_hist'] <= 0, 0)
        
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=positive_hist,
                name='MACD Hist +',
                marker_color='green',
                opacity=0.5,
                hovertemplate='MACD Hist: %{y:.4f}<extra></extra>',
                yaxis='y4',
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=negative_hist,
                name='MACD Hist -',
                marker_color='red',
                opacity=0.5,
                hovertemplate='MACD Hist: %{y:.4f}<extra></extra>',
                yaxis='y4',
                showlegend=False
            )
        )
    
    # 7. RSI indicator
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=indicators['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2),
            hovertemplate='RSI: %{y:.1f}<extra></extra>',
            yaxis='y5'
        )
    )
    
    # Add reference lines using traces instead of add_hline
    # MACD zero line (conditional)
    if show_macd:
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=[0] * len(df_chart.index),
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name='MACD Zero',
                yaxis='y4',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # RSI reference lines
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=[70] * len(df_chart.index),
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            name='RSI 70',
            yaxis='y5',
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=[50] * len(df_chart.index),
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name='RSI 50',
            yaxis='y5',
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=[30] * len(df_chart.index),
            mode='lines',
            line=dict(color='green', width=1, dash='dot'),
            name='RSI 30',
            yaxis='y5',
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    # 8. Seasonal Component (based on RSI data)
    try:
        # Get seasonal decomposition for the displayed data period
        close_series = df_chart['close']
        # In performance mode, use reduced years for seasonal analysis
        seasonal_years_adjusted = min(seasonal_years, 1) if performance_mode else seasonal_years
        stl_result, _ = get_daily_seasonal(close_series, seasonal_years_adjusted)
        seasonal_series = stl_result.seasonal
        
        # Scale to percentage for better visualization
        seasonal_scaled = seasonal_series * 100
        
        # Add seasonal component as area chart
        fig.add_trace(
            go.Scatter(
                x=seasonal_series.index,
                y=seasonal_scaled,
                mode='lines',
                line=dict(color='cyan', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,255,0.2)',
                name='RSI Seasonal',
                hovertemplate='RSI Seasonal: %{y:.2f}%<extra></extra>',
                yaxis='y6',
                showlegend=False
            )
        )
        
        # Add zero line for seasonal reference
        fig.add_trace(
            go.Scatter(
                x=seasonal_series.index,
                y=[0] * len(seasonal_series.index),
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name='Seasonal Zero',
                yaxis='y6',
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
    except Exception as e:
        # If seasonal analysis fails, add a placeholder message
        if performance_mode:
            st.info("RSI-based seasonal analysis disabled in Performance Mode")
        else:
            st.warning(f"RSI-based seasonal analysis unavailable: {str(e)}")
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Comprehensive Technical Analysis',
        template='plotly_dark',
        height=900 if performance_mode else 1200,
        showlegend=False,
        hovermode='x unified',
        hoverdistance=100,
        spikedistance=1000,
        xaxis=dict(
            showspikes=True,
            spikecolor="rgba(0,150,255,0.8)",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=2,
            spikedash="dash",
            domain=[0.0, 1.0],
            anchor='y',
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends (Saturday and Sunday)
                dict(values=["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-23", "2023-12-25"]),  # US holidays 2023
                dict(values=["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-28", "2024-12-25"]),  # US holidays 2024
                dict(values=["2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-27", "2025-12-25"]),  # US holidays 2025
            ]
        ),
        yaxis=dict(
            title='Price ($)',
            side='left',
            domain=[0.7, 1.0] if volume_col else ([0.6, 1.0] if show_macd else [0.65, 1.0])
        ),
        yaxis2=dict(
            title='Volume',
            side='left',
            showgrid=False,
            domain=[0.58, 0.68] if show_macd else [0.53, 0.63]
        ) if volume_col else None,
        yaxis3=dict(
            title='Price/MA (%)',
            side='left',
            showgrid=True,
            domain=[0.45, 0.55] if show_macd else ([0.38, 0.50] if volume_col else [0.38, 0.50]),
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis4=dict(
            title='MACD',
            side='left',
            showgrid=True,
            domain=[0.30, 0.42],
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ) if show_macd else None,
        yaxis5=dict(
            title='RSI',
            side='left',
            showgrid=True,
            domain=[0.15, 0.27] if show_macd else ([0.20, 0.35] if volume_col else [0.20, 0.35]),
            range=[0, 100],
            tickvals=[0, 30, 50, 70, 100],
            zeroline=False
        ),
        yaxis6=dict(
            title='Seasonal (%)',
            side='left',
            showgrid=True,
            domain=[0.0, 0.12] if show_macd else ([0.0, 0.17] if volume_col else [0.0, 0.17]),
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        )
    )
    
    # Apply enhanced crosshair settings
    fig.update_xaxes(
        showspikes=True,
        spikecolor="rgba(0,150,255,0.8)",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=2,
        spikedash="dash"
    )
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

def create_basic_price_chart(df, symbol, chart_type="line"):
    """Create a basic price chart when limited data is available"""
    if df is None or len(df) == 0:
        return None
    
    fig = go.Figure()
    
    # Create basic candlestick chart
    daily_changes_pct = df['close'].pct_change() * 100
    daily_changes_dollar = df['close'].diff()
    
    # Create custom hover text
    hover_text = []
    for i, (date, row) in enumerate(df.iterrows()):
        if i == 0:
            change_text = "N/A"
        else:
            pct_change = daily_changes_pct.iloc[i]
            dollar_change = daily_changes_dollar.iloc[i]
            change_text = f"{pct_change:+.2f}% (${dollar_change:+.2f})"
        
        volume_text = f"Volume: {row['volume']:,.0f}<br>" if 'volume' in row else ""
        hover_text.append(
            f"Date: {date.strftime('%Y-%m-%d')}<br>"
            f"Open: ${row['open']:.2f}<br>"
            f"High: ${row['high']:.2f}<br>"
            f"Low: ${row['low']:.2f}<br>"
            f"Close: ${row['close']:.2f}<br>"
            f"{volume_text}"
            f"Daily Change: {change_text}"
        )
    
    # Price chart - either line or candlestick based on chart_type
    if chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red',
                hovertext=hover_text,
                hoverinfo='text'
            )
        )
    else:
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                line=dict(color='black', width=2),
                name='Price',
                hovertext=hover_text,
                hoverinfo='text'
            )
        )
    
    # Add volume if available
    if 'volume' in df.columns:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.3,
                yaxis='y2',
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Basic Price Chart ({len(df)} days)',
        template='plotly_dark',
        height=600,
        showlegend=False,
        hovermode='x unified',
        xaxis=dict(
            showspikes=True,
            spikecolor="rgba(0,150,255,0.8)",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=2,
            spikedash="dash",
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends (Saturday and Sunday)
                dict(values=["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-23", "2023-12-25"]),  # US holidays 2023
                dict(values=["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-28", "2024-12-25"]),  # US holidays 2024
                dict(values=["2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-27", "2025-12-25"]),  # US holidays 2025
            ]
        ),
        yaxis=dict(
            title='Price ($)',
            side='left',
            domain=[0.4, 1.0] if 'volume' in df.columns else [0.0, 1.0]
        ),
        yaxis2=dict(
            title='Volume',
            side='left',
            showgrid=False,
            domain=[0.0, 0.35]
        ) if 'volume' in df.columns else None
    )
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

def cleanup_memory():
    """Force garbage collection to free up memory"""
    gc.collect()

def prepare_technical_analysis_data(df, ticker, seasonal_years, chart_type, show_macd, performance_mode):
    """Thread function to prepare technical analysis chart data"""
    try:
        if df is None or len(df) == 0:
            return None
        
        chart_data = create_technical_analysis_chart(df, ticker, seasonal_years, chart_type, show_macd, performance_mode)
        cleanup_memory()
        return chart_data
    except Exception as e:
        return f"Error in technical analysis: {str(e)}"

def prepare_prediction_data(results, tech_only):
    """Thread function to prepare prediction analysis data"""
    try:
        if tech_only:
            return {"tech_only": True, "message": "Technical Analysis Only Mode"}
        
        # Extract prediction data from results
        prediction_data = {
            "model": results.get('model'),
            "proba": results.get('proba'),
            "training_features": results.get('training_features', []),
            "analysis_df": results.get('analysis_df'),
            "current_price": results.get('current_price', 0),
            "ticker": results.get('ticker', ''),
            "tech_only": False
        }
        cleanup_memory()
        return prediction_data
    except Exception as e:
        return f"Error in prediction analysis: {str(e)}"

def prepare_pattern_data(results, tech_only):
    """Thread function to prepare pattern matching data"""
    try:
        if tech_only:
            return {"tech_only": True, "message": "Technical Analysis Only Mode"}
        
        # Extract pattern data from results
        pattern_data = {
            "matches": results.get('matches', []),
            "matches_sorted": results.get('matches_sorted', []),
            "price_series": results.get('price_series'),
            "series_dates": results.get('series_dates'),
            "reference_pattern": results.get('reference_pattern'),
            "similarity_threshold": results.get('similarity_threshold', 0.7),
            "pattern_length": results.get('pattern_length', 60),
            "timeframe": results.get('timeframe', 'Daily'),
            "timeframe_label": results.get('timeframe_label', 'days'),
            "ticker": results.get('ticker', ''),
            "tech_only": False
        }
        cleanup_memory()
        return pattern_data
    except Exception as e:
        return f"Error in pattern analysis: {str(e)}"

def get_seasonal_component(data: pd.Series, years: int = 1) -> pd.Series:
    """Get seasonal component for any data series."""
    # Get last N years of data
    years_ago = data.index[-1] - pd.DateOffset(years=years)
    recent_data = data[data.index >= years_ago].copy().dropna()
    
    min_days = years * 100  # Minimum days per year
    if len(recent_data) < min_days:
        # Fallback: use all available data if insufficient for requested years
        if len(data) >= 200:  # Minimum 200 days for any seasonal analysis
            recent_data = data.copy().dropna()
        else:
            raise ValueError(f"Not enough data for seasonal analysis (need at least 200 days, got {len(data)})")
    
    # Use business day frequency to handle weekends/holidays
    daily_business = recent_data.asfreq("B").interpolate().dropna()
    
    # Adjust period based on available data
    period = min(252, len(daily_business) // 4)
    if period < 10:
        raise ValueError("Not enough data for seasonal decomposition")
    
    # Perform STL decomposition
    result = seasonal_decompose(daily_business, model="additive", period=period, extrapolate_trend="freq")
    return result.seasonal

def get_daily_seasonal(close: pd.Series, years: int = 1) -> tuple:
    """Get daily seasonal decomposition for RSI data."""
    # Get last N years of data
    years_ago = close.index[-1] - pd.DateOffset(years=years)
    recent_close = close[close.index >= years_ago].copy()
    
    min_days = years * 100  # Minimum days per year
    if len(recent_close) < min_days:
        # Fallback: use all available data if insufficient for requested years
        if len(close) >= 200:  # Minimum 200 days for any seasonal analysis
            recent_close = close.copy()
            years = max(1, len(recent_close) // 252)  # Approximate years from available data
        else:
            raise ValueError(f"Not enough data for seasonal analysis (need at least 200 days, got {len(close)})")
    
    # Calculate RSI from the close price data
    rsi_data = TechnicalIndicators.rsi(recent_close, 14)
    
    # Use business day frequency to handle weekends/holidays
    daily_business = rsi_data.asfreq("B").interpolate().dropna()
    
    # Adjust period based on available data
    period = min(252, len(daily_business) // 4)
    if period < 10:
        raise ValueError("Not enough data for seasonal decomposition")
    
    # Perform STL decomposition on RSI data (no log transformation needed for RSI)
    result = seasonal_decompose(daily_business, model="additive", period=period, extrapolate_trend="freq")
    return result, recent_close

def create_seasonal_chart(df, symbol, years):
    """Create a seasonal analysis chart with multiple subplots based on RSI data."""
    if df is None or len(df) == 0:
        st.error("No data available for seasonal analysis")
        return None
    
    # Get the data we need
    close = df['close']
    volume = df.get('volume', df.get('vol', pd.Series()))
    
    try:
        # Get seasonal decomposition (now based on RSI data)
        stl_result, recent_close = get_daily_seasonal(close, years)
        
        # Get corresponding volume data for the same period
        recent_volume = volume[volume.index >= recent_close.index[0]] if not volume.empty else pd.Series()
        
        # Calculate technical indicators
        macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(recent_close, 12, 26, 9)
        rsi = TechnicalIndicators.rsi(recent_close, 14)
        bb_upper, bb_sma, bb_lower = TechnicalIndicators.bollinger_bands(recent_close, 20, 2)
        
        # Create the seasonal chart
        fig = create_seasonal_plotly_chart(recent_close, recent_volume, stl_result, symbol, years, 
                                         macd_line, macd_signal, macd_hist, rsi, bb_upper, bb_sma, bb_lower)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating RSI-based seasonal chart: {str(e)}")
        return None

def create_seasonal_plotly_chart(close, volume, stl_result, symbol, years, 
                                macd_line, macd_signal, macd_hist, rsi, bb_upper, bb_sma, bb_lower):
    """Create the seasonal chart using Plotly with separated sections (seasonal component based on RSI)."""
    
    # Create subplots with 5 rows (Price, Volume, Seasonal, MACD, RSI)
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Price & Bollinger Bands', 'Volume', 'Seasonal Component', 'MACD', 'RSI'],
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # 1. Price chart with Bollinger Bands
    fig.add_trace(
        go.Scatter(x=close.index, y=bb_upper, 
                  mode='lines', line=dict(color='lightblue', width=1),
                  name='BB Upper', showlegend=False), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=close.index, y=bb_lower,
                  mode='lines', line=dict(color='lightblue', width=1),
                  fill='tonexty', fillcolor='rgba(173,216,230,0.1)',
                  name='BB Lower', showlegend=False), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=close.index, y=bb_sma,
                  mode='lines', line=dict(color='orange', width=2),
                  name='SMA 20', showlegend=False), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=close.index, y=close,
                  mode='lines', line=dict(color='black', width=2),
                  name='Price', showlegend=False), row=1, col=1
    )
    
    # 2. Volume chart
    if not volume.empty:
        colors = ['green' if i > 0 and close.iloc[i] > close.iloc[i-1] else 'red' 
                 for i in range(len(volume))]
        fig.add_trace(
            go.Bar(x=volume.index, y=volume, 
                  marker_color=colors, opacity=0.7,
                  name='Volume', showlegend=False), row=2, col=1
        )
    
    # 3. Seasonal component (based on RSI data)
    seasonal_series = stl_result.seasonal
    
    # Ensure we have valid seasonal data
    if len(seasonal_series) > 0:
        # Scale seasonal values to make them more visible (multiply by 100 for percentage display)
        seasonal_scaled = seasonal_series * 100
        
        # Add zero line for reference
        fig.add_hline(y=0, line=dict(color='gray', dash='dot'), row=3, col=1)
        
        # Add seasonal component as area chart
        fig.add_trace(
            go.Scatter(x=seasonal_series.index, 
                      y=seasonal_scaled,
                      mode='lines+markers',
                      line=dict(color='cyan', width=3),
                      marker=dict(size=2, color='cyan'),
                      fill='tozeroy',
                      fillcolor='rgba(0,255,255,0.3)',
                      name='RSI Seasonal (%)', 
                      showlegend=False,
                      hovertemplate='Date: %{x}<br>RSI Seasonal: %{y:.2f}%<extra></extra>'), 
                      row=3, col=1
        )
    
    # 4. MACD
    fig.add_trace(
        go.Scatter(x=macd_line.index, y=macd_line,
                  mode='lines', line=dict(color='blue', width=2),
                  name='MACD', showlegend=False), row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=macd_signal.index, y=macd_signal,
                  mode='lines', line=dict(color='red', width=2),
                  name='Signal', showlegend=False), row=4, col=1
    )
    
    # MACD Histogram
    colors = ['green' if x > 0 else 'red' for x in macd_hist]
    fig.add_trace(
        go.Bar(x=macd_hist.index, y=macd_hist,
              marker_color=colors, opacity=0.6,
              name='MACD Hist', showlegend=False), row=4, col=1
    )
    
    # 5. RSI
    fig.add_trace(
        go.Scatter(x=rsi.index, y=rsi,
                  mode='lines', line=dict(color='purple', width=2),
                  name='RSI', showlegend=False), row=5, col=1
    )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=5, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=5, col=1)
    fig.add_hline(y=50, line=dict(color='gray', dash='dot'), row=5, col=1)
    
    # Update layout
    year_text = "Year" if years == 1 else "Years"
    fig.update_layout(
        title=f'{symbol} - Seasonal Analysis ({years} {year_text})',
        template='plotly_dark',
        height=1200,
        showlegend=False,
        hovermode='x unified',
        xaxis5=dict(
            showspikes=True,
            spikecolor="rgba(0,150,255,0.8)",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=2,
            spikedash="dash"
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal (%)", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="RSI", row=5, col=1)
    
    # Set RSI range
    fig.update_yaxes(range=[0, 100], row=5, col=1)
    
    return fig

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
        
        # URL Parameters info
        with st.expander("🔗 URL Parameters", expanded=False):
            st.info("""
            **Quick Access via URL:**
            
            `?ticker=AAPL` - Set ticker symbol
            
            `?timeframe=Daily` - Set timeframe (Daily/Weekly)
            
            `?pattern_length=60` - Set pattern length
            
            `?tech_only=false` - Enable Full Analysis mode (default is tech-only)
            
            **Examples:**
            
            `?ticker=TSLA` - Auto-load quick technical analysis (default)
            
            `?ticker=MSFT&tech_only=false&timeframe=Weekly&pattern_length=12` - Auto-run full analysis
            
            *Note: Analysis runs automatically when URL parameters are provided.*
            """)
        
        st.markdown("---")
        
        # Get ticker from URL parameter or use default
        url_ticker = st.query_params.get("ticker", "").upper()
        default_ticker = url_ticker if url_ticker else "AAPL"
        
        ticker = st.text_input(
            "Enter Stock Ticker", 
            value=default_ticker,
            help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT). You can also use URL: ?ticker=SYMBOL"
        ).upper()
        
        # Update URL parameter when ticker changes
        if ticker != st.query_params.get("ticker", "").upper():
            st.query_params["ticker"] = ticker
        
        # Technical Analysis Only option with URL parameter support (default: True)
        url_tech_only = st.query_params.get("tech_only", "true").lower() == "true"
        tech_only = st.checkbox(
            "📊 Technical Analysis Only",
            value=url_tech_only,
            help="Load only technical analysis charts and indicators (faster, no AI predictions or pattern matching)"
        )
        
        # Update URL parameter when tech_only changes
        if str(tech_only).lower() != st.query_params.get("tech_only", "true").lower():
            st.query_params["tech_only"] = str(tech_only).lower()
        
        # Different button text based on mode
        if tech_only:
            run_analysis = st.button("📊 Load Technical Analysis", type="primary", use_container_width=True)
        else:
            run_analysis = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
        
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
            value=0.7,
            step=0.01,
            help="Minimum cosine similarity score for pattern matches"
        )
        
        # Get timeframe from URL parameter or use default
        url_timeframe = st.query_params.get("timeframe", "Daily")
        if url_timeframe not in ["Daily", "Weekly"]:
            url_timeframe = "Daily"
        
        timeframe_index = 0 if url_timeframe == "Daily" else 1
        
        timeframe = st.selectbox(
            "Timeframe",
            options=["Daily", "Weekly"],
            index=timeframe_index,
            help="Timeframe for pattern analysis. You can also use URL: ?timeframe=Daily or ?timeframe=Weekly"
        )
        
        # Update URL parameter when timeframe changes
        if timeframe != st.query_params.get("timeframe", "Daily"):
            st.query_params["timeframe"] = timeframe
        
        if timeframe == "Daily":
            # Get pattern_length from URL parameter or use default
            url_pattern_length = st.query_params.get("pattern_length", "60")
            try:
                default_pattern_length = max(5, min(100, int(url_pattern_length)))
            except (ValueError, TypeError):
                default_pattern_length = 60
            
            pattern_length = st.slider(
                "Pattern Length (days)",
                min_value=5,
                max_value=100,
                value=default_pattern_length,
                step=1,
                help="Length of the reference pattern in days. You can also use URL: ?pattern_length=60"
            )
        else:  # Weekly
            # Get pattern_length from URL parameter or use default
            url_pattern_length = st.query_params.get("pattern_length", "8")
            try:
                default_pattern_length = max(2, min(52, int(url_pattern_length)))
            except (ValueError, TypeError):
                default_pattern_length = 8
            
            pattern_length = st.slider(
                "Pattern Length (weeks)",
                min_value=2,
                max_value=52,
                value=default_pattern_length,
                step=2,
                help="Length of the reference pattern in weeks. You can also use URL: ?pattern_length=8"
            )
        
        # Update URL parameter when pattern_length changes
        if str(pattern_length) != st.query_params.get("pattern_length", ""):
            st.query_params["pattern_length"] = str(pattern_length)
        
        st.markdown("---")
        st.subheader("🚀 Performance Settings")
        performance_mode = st.checkbox(
            "Enable Performance Mode",
            value=False,
            help="Reduces computation and memory usage for limited resources (fewer data points, simplified calculations)"
        )
        
        st.markdown("---")
        st.subheader("📊 Chart Display Settings")
        chart_type = st.selectbox(
            "Price Chart Type",
            ["line", "candlestick"],
            index=1,
            help="Choose between line chart or candlestick chart for price display"
        )
        
        show_macd = st.checkbox(
            "Show MACD",
            value=False,
            help="Toggle MACD indicator visibility on the chart"
        )
        
        st.markdown("---")
        st.subheader("📊 Technical Analysis Settings")
        seasonal_years = st.slider(
            "Years",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Number of years of historical data to use for RSI-based seasonal pattern analysis"
        )
        
        st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Technical Analysis", "📈 Stock Prediction Results", "🔍 Fractal Pattern Results"])
    
    # Initialize session state for persisting results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_config' not in st.session_state:
        st.session_state.analysis_config = None
    
    # Auto-run analysis if URL parameters are provided and no cached results exist
    has_url_params = any([
        st.query_params.get("ticker"),
        st.query_params.get("timeframe"),
        st.query_params.get("pattern_length"),
        st.query_params.get("tech_only")
    ])
    auto_run = has_url_params and st.session_state.analysis_results is None
    
    # Check if we should run analysis (button clicked, auto-run, or results exist)
    should_run_analysis = run_analysis or auto_run or st.session_state.analysis_results is not None
    
    if should_run_analysis:
        if not ticker:
            st.error("Please enter a ticker symbol")
            return
        
        # Show auto-run indicator
        if auto_run:
            if tech_only:
                st.info("🔗 Auto-loading technical analysis from URL parameters...")
            else:
                st.info("🔗 Auto-running full analysis from URL parameters...")
        
        # Check if we need to run new analysis or use cached results
        current_config = {
            'ticker': ticker,
            'similarity_threshold': similarity_threshold,
            'timeframe': timeframe,
            'pattern_length': pattern_length,
            'tech_only': tech_only,
            'seasonal_years': seasonal_years
        }
        
        # Run new analysis if button clicked or config changed
        if run_analysis or st.session_state.analysis_config != current_config:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ========== SHARED DATA LOADING ==========
            status_text.text("📊 Fetching stock data...")
            progress_bar.progress(10)
            
            # Initialize training error variable
            training_error = None
            
            base_ticker = ticker.replace('.US', '')
            df, error = load_stock_data(ticker)
            
            # Clean up memory after data loading
            cleanup_memory()
            
            if error:
                st.error(f"Error loading data: {error}")
                return
            
            # Check data requirements with more flexible options
            min_required = max(pattern_length * 2, 60)  # Reduced from 100 to 60
            #st.info(f"🔍 Debug: Data length: {len(df)}, Pattern length: {pattern_length}, Min required: {min_required}")
            if len(df) < min_required:
                st.warning(f"⚠️ Limited data available ({len(df)} days). Recommended: {min_required} days.")
                st.info("📊 Some features may be limited with less data, but analysis will continue.")
                
                # Ask user if they want to continue with limited data
                continue_with_limited = st.sidebar.checkbox("Continue with limited data?", value=True)
                if not continue_with_limited:
                    return
            
            # Auto-adjust pattern length if data is very limited
            if len(df) < pattern_length:
                adjusted_pattern_length = max(10, len(df) // 3)  # Use 1/3 of available data
                st.info(f"📏 Auto-adjusting pattern length from {pattern_length} to {adjusted_pattern_length} days due to limited data.")
                pattern_length = adjusted_pattern_length
            
            # Get current market data for both analyses
            current_price = float(df['close'].iloc[-1])
            previous_price = float(df['close'].iloc[-2])
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            progress_bar.progress(20)
            
            # ========== TECH-ONLY MODE BRANCH ==========
            if tech_only:
                status_text.text("📊 Processing technical indicators (fast mode)...")
                progress_bar.progress(50)
                
                # Get basic market data for technical analysis
                current_volume = int(df['vol'].iloc[-1])
                high_today = float(df['high'].iloc[-1])
                low_today = float(df['low'].iloc[-1])
                open_today = float(df['open'].iloc[-1])
                
                # Initialize variables needed for tech-only mode
                training_error = None
                training_features = []
                model = None
                proba = None
                matches = []
                matches_sorted = []
                analysis_df = df
                timeframe_label = "days"
                
                # Cache minimal results for tech-only mode
                st.session_state.analysis_results = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'current_volume': current_volume,
                    'high_today': high_today,
                    'low_today': low_today,
                    'open_today': open_today,
                    'df': df,
                    'model': None,
                    'proba': None,
                    'matches': [],
                    'matches_sorted': [],
                    'price_series': None,
                    'series_dates': None,
                    'reference_pattern': None,
                    'reference_pattern_norm': None,
                    'timeframe': 'Daily',
                    'timeframe_label': 'days',
                    'similarity_threshold': 0.85,
                    'pattern_length': pattern_length,
                    'analysis_df': df,
                    'X_live': None,
                    'training_features': [],
                    'df_clean': None,
                    'pattern_start_date': None,
                    'pattern_end_date': None,
                    'tech_only': True
                }
                st.session_state.analysis_config = current_config
                
                status_text.text("✅ Technical analysis ready!")
                progress_bar.progress(100)
                progress_bar.empty()
                status_text.empty()
                
                # Jump to display section
                goto_display = True
            else:
                # Continue with full analysis
                goto_display = False
        else:
            # Use cached results
            results = st.session_state.analysis_results
            ticker = results['ticker']
            base_ticker = ticker.replace('.US', '')
            current_price = results['current_price']
            previous_price = results['previous_price']
            price_change = results['price_change']
            price_change_pct = results['price_change_pct']
            current_volume = results['current_volume']
            high_today = results['high_today']
            low_today = results['low_today']
            open_today = results['open_today']
            model = results['model']
            proba = results['proba']
            matches = results['matches']
            matches_sorted = results['matches_sorted']
            price_series = results['price_series']
            series_dates = results['series_dates']
            reference_pattern = results['reference_pattern']
            reference_pattern_norm = results['reference_pattern_norm']
            timeframe = results['timeframe']
            timeframe_label = results['timeframe_label']
            similarity_threshold = results['similarity_threshold']
            pattern_length = results['pattern_length']
            analysis_df = results['analysis_df']
            
            # Load additional variables from cached results
            X_live = results.get('X_live', None)
            training_features = results.get('training_features', [])
            df = results.get('df', None)
            df_clean = results.get('df_clean', None)  # Added df_clean to loading
            pattern_start_date = results.get('pattern_start_date', None)
            pattern_end_date = results.get('pattern_end_date', None)
            training_error = results.get('training_error', None)
            
            # Check if cached results are from tech-only mode
            cached_tech_only = results.get('tech_only', False)
            
            # Skip to display results
            goto_display = True
        
        # ========== STOCK PREDICTION ANALYSIS ==========
        # Skip analysis if using cached results
        if 'goto_display' in locals() and goto_display:
            # Skip to display results
            pass
        else:
            status_text.text("🔧 Processing technical indicators...")
            progress_bar.progress(30)
            
            # Get additional market data for prediction
            current_volume = int(df['vol'].iloc[-1])
            high_today = float(df['high'].iloc[-1])
            low_today = float(df['low'].iloc[-1])
            open_today = float(df['open'].iloc[-1])
            
            df_clean = df.dropna()
            if df_clean.empty:
                st.warning("⚠️ Limited data for technical indicators. Some features may be unavailable.")
                # Continue with basic analysis
                df_clean = df  # Use original data even with NaN values
            
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
            
            # Debug: Show feature count and ensure exact match
            # st.info(f"📊 Loaded {len(training_features)} features for prediction")
            
            # If we don't have enough features, try to use the full 42-feature set from config
            if len(training_features) < 20:  # If we have less than 20 features, try to get all 42
                st.info("🔄 Attempting to load full 42-feature set...")
                try:
                    from config import features as full_features
                    # Check which features are available in the DataFrame
                    available_full_features = [f for f in full_features if f in df_latest.columns]
                    if len(available_full_features) > len(training_features):
                        training_features = available_full_features
                        st.info(f"✅ Successfully loaded {len(training_features)} features from full feature set")
                    else:
                        st.warning(f"⚠️ Full feature set only provides {len(available_full_features)} features")
                except ImportError:
                    st.warning("⚠️ Could not import full feature set from config")
            
            # Final fallback to basic features only if we still have very few features
            if len(training_features) < 10:
                st.warning("⚠️ Very few features available. Using basic features only.")
                # Fallback to basic features
                basic_features = ['close', 'open', 'high', 'low', 'vol', 'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio']
                training_features = [f for f in basic_features if f in df_latest.columns]
                st.info(f"📊 Using {len(training_features)} basic features: {training_features}")
            
            # Store the selected features for consistency
            selected_features = training_features.copy()
            
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
            if 'force_retrain' not in locals():
                force_retrain = False
                
            if not os.path.exists(model_path) or force_retrain:
                status_text.text("🚀 Training model (this may take a moment)...")
                progress_bar.progress(50)
                
                try:
                    # Create a temporary file with the selected features
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(selected_features, f)
                        features_file = f.name
                    
                    # Pass the features file to the training script
                    result = subprocess.run([sys.executable, "train_wrapper.py", base_ticker, "--features", features_file], 
                                          capture_output=True, text=True, timeout=600)
                    
                    # Clean up the temporary file
                    try:
                        os.unlink(features_file)
                    except:
                        pass
                    if result.returncode == 0:
                        st.success("✅ Model training completed successfully!")
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                        else:
                            st.warning("Model training completed but model file not found. Checking fallback location...")
                            if os.path.exists(fallback_model_path):
                                model = joblib.load(fallback_model_path)
                    else:
                        # Store training error for display in Stock Prediction tab
                        training_error = {
                            'type': 'training_failed',
                            'return_code': result.returncode,
                            'stdout': result.stdout,
                            'stderr': result.stderr
                        }
                except subprocess.TimeoutExpired:
                    # Store timeout error for display in Stock Prediction tab
                    training_error = {
                        'type': 'timeout',
                        'message': "Model training timed out (>10 minutes)"
                    }
                except Exception as e:
                    # Store general error for display in Stock Prediction tab
                    training_error = {
                        'type': 'general_error',
                        'message': str(e)
                    }
            
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
                    # Debug: Show feature count mismatch if it occurs
                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                        actual_features = X_live.shape[1]
                        if expected_features != actual_features:
                            st.warning(f"⚠️ Feature count mismatch: Model expects {expected_features} features, but got {actual_features} features.")
                            st.info(f"📊 Model was trained with {expected_features} features, but current data has {actual_features} features.")
                            st.info("🔄 This usually happens when the model was trained with limited data and basic features.")
                    
                    proba = model.predict_proba(X_live)[0]
                except Exception as e:
                    st.warning(f"Prediction failed: {str(e)}. Only pattern analysis will be available.")
                    st.info("💡 This usually happens when the model was trained with different features than what's currently available.")
                    proba = None
        
        # ========== FRACTAL PATTERN ANALYSIS ==========
        # Skip analysis if using cached results
        if 'goto_display' in locals() and goto_display:
            # Skip to display results
            pass
        else:
            status_text.text("🔍 Analyzing fractal patterns...")
            progress_bar.progress(70)
            
            # Check if we have enough data for fractal analysis
            if len(df) < pattern_length * 2:
                st.warning(f"⚠️ Limited data for fractal analysis. Pattern length: {pattern_length}, Available data: {len(df)} days")
                st.info("📊 Fractal analysis may be limited, but basic pattern matching will continue.")
            
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
            
            # Calculate smart window ranges based on reference pattern length
            # Search from 74% to 124% of reference pattern length for focused matching
            # This gives 40-67 range for 54-day pattern
            min_multiplier = 0.74
            max_multiplier = 1.24
            
            smart_min_days = max(5, int(pattern_length * min_multiplier))
            smart_max_days = max(smart_min_days + 1, int(pattern_length * max_multiplier))
            
            # Define window sizes for pattern matching using smart range (same as fractal.py defaults)
            window_sizes = list(range(smart_min_days, smart_max_days + 1))
            
            # Debug: Show the smart range being used
            #st.info(f"🔍 Smart Range: {smart_min_days}-{smart_max_days} days (74%-124% of {pattern_length}-day pattern)")
            
            # Perform pattern matching
            patterns = [("Reference", reference_pattern_norm)]
            
            try:
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
            except Exception as e:
                st.warning(f"⚠️ Pattern matching failed: {str(e)}. Limited data may be the cause.")
                matches = []  # Empty matches list
            
            progress_bar.progress(90)
            status_text.text("✅ Analysis completed!")
            
            # Clear progress indicators
            progress_bar.progress(100)
            progress_bar.empty()
            status_text.empty()
            
            # Store results in session state for persistence
            st.session_state.analysis_results = {
                'ticker': ticker,
                'current_price': current_price,
                'previous_price': previous_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'current_volume': current_volume,
                'high_today': high_today,
                'low_today': low_today,
                'open_today': open_today,
                'model': model,
                'proba': proba,
                'matches': matches,
                'matches_sorted': sorted(matches, key=lambda x: x['similarity'], reverse=True) if matches else [],
                'price_series': price_series,
                'series_dates': series_dates,
                'reference_pattern': reference_pattern,
                'reference_pattern_norm': reference_pattern_norm,
                'timeframe': timeframe,
                'timeframe_label': timeframe_label,
                'similarity_threshold': similarity_threshold,
                'pattern_length': pattern_length,
                'analysis_df': analysis_df,
                'X_live': X_live,
                'training_features': training_features,
                'df': df,
                'df_clean': df_clean,  # Added df_clean to session state
                'pattern_start_date': pattern_start_date,
                'pattern_end_date': pattern_end_date
            }
            st.session_state.analysis_config = current_config
        
        # Define class labels for prediction
        class_labels = {
            0: "📉 Drop >10%",
            1: "📉 Drop 5-10%", 
            2: "📊 Drop 0-5%",
            3: "📈 Gain 0-5%",
            4: "📈 Gain 5-10%",
            5: "🚀 Gain >10%"
        }
        
        # Clean up memory before displaying results
        cleanup_memory()
        
        # ========== PARALLEL COMPUTATION FOR ALL TABS ==========
        # Initialize thread-safe results storage
        if 'threaded_results' not in st.session_state:
            st.session_state.threaded_results = {}
        
        # Check if we need to recompute (only if analysis config changed)
        current_thread_config = {
            'ticker': ticker if 'ticker' in locals() else '',
            'seasonal_years': seasonal_years,
            'chart_type': chart_type,
            'show_macd': show_macd,
            'performance_mode': performance_mode,
            'tech_only': tech_only if 'tech_only' in locals() else True
        }
        
        # Force refresh if seasonal_years changed
        if (st.session_state.threaded_results.get('config', {}).get('seasonal_years') != seasonal_years):
            st.session_state.threaded_results = {}
        
        # Get results for threading (use existing results or cached data)
        if 'results' in locals():
            thread_results = results
        else:
            thread_results = st.session_state.analysis_results
        
        # Run parallel computation for all tabs
        if (st.session_state.threaded_results.get('config') != current_thread_config or 
            'tab_data' not in st.session_state.threaded_results):
            
            with st.spinner("🚀 Processing all analyses in parallel..."):
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit all three tab computations simultaneously
                    future_technical = executor.submit(
                        prepare_technical_analysis_data,
                        df if 'df' in locals() else None,
                        ticker if 'ticker' in locals() else thread_results.get('ticker', ''),
                        seasonal_years,
                        chart_type,
                        show_macd,
                        performance_mode
                    )
                    
                    future_prediction = executor.submit(
                        prepare_prediction_data,
                        thread_results,
                        tech_only if 'tech_only' in locals() else True
                    )
                    
                    future_pattern = executor.submit(
                        prepare_pattern_data,
                        thread_results,
                        tech_only if 'tech_only' in locals() else True
                    )
                    
                    # Collect results as they complete
                    st.session_state.threaded_results = {
                        'config': current_thread_config,
                        'tab_data': {
                            'technical': future_technical.result(),
                            'prediction': future_prediction.result(),
                            'pattern': future_pattern.result()
                        }
                    }
        
        # ========== DISPLAY RESULTS IN TABS ==========
        # ========== TECHNICAL ANALYSIS TAB ==========
        with tab1:
            st.markdown('<div class="section-header"><i class="fas fa-chart-line"></i> Technical Analysis Chart</div>', unsafe_allow_html=True)
            
            # Use threaded results for technical analysis
            tech_data = st.session_state.threaded_results.get('tab_data', {}).get('technical')
            
            if isinstance(tech_data, str) and tech_data.startswith("Error"):
                st.error(tech_data)
            elif tech_data is not None and df is not None and len(df) > 0:
                st.subheader(f"📊 {ticker if 'ticker' in locals() else 'Stock'} Technical Analysis")
                
                # Show performance mode notice if enabled
                if performance_mode:
                    st.info("🚀 **Performance Mode Enabled** - Using reduced data points and simplified calculations for better performance on limited resources.")
                
                # Display the pre-computed chart
                st.plotly_chart(tech_data, use_container_width=True)
                
                # Clean up memory after chart rendering
                cleanup_memory()
                
                # Add Professional Analysis Criteria
                if df is not None and len(df) > 0:
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
                    
                    # Add technical analysis insights
                    st.markdown('<div class="section-header"><i class="fas fa-chart-bar"></i> Technical Analysis Insights</div>', unsafe_allow_html=True)
                    
                    # Calculate current technical indicators
                    close = df['close']
                    current_price = close.iloc[-1]
                    
                    # Bollinger Bands analysis
                    bb_sma, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(close, 20, 2)
                    current_bb_upper = bb_upper.iloc[-1]
                    current_bb_lower = bb_lower.iloc[-1]
                    current_bb_sma = bb_sma.iloc[-1]
                    
                    # SMA analysis
                    sma_20 = TechnicalIndicators.sma(close, 20)
                    sma_50 = TechnicalIndicators.sma(close, 50)
                    sma_144 = TechnicalIndicators.sma(close, 144)
                    
                    current_sma_20 = sma_20.iloc[-1]
                    current_sma_50 = sma_50.iloc[-1]
                    current_sma_144 = sma_144.iloc[-1]
                    
                    # Price Distance to MA analysis
                    pma_fast, pma_fast_signal, pma_fast_cycle = TechnicalIndicators.price_distance_to_ma(
                        close, ma_length=20, signal_length=9, exponential=False
                    )
                    current_pma = pma_fast.iloc[-1]
                    current_signal = pma_fast_signal.iloc[-1]
                    current_cycle = pma_fast_cycle.iloc[-1]
                    
                    # Display technical insights
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # SMA 20 analysis
                        sma_20_above = current_price > current_sma_20
                        sma_20_class = "success" if sma_20_above else "danger"
                        sma_20_color = "green" if sma_20_above else "red"
                        
                        # SMA 50 analysis
                        sma_50_above = current_price > current_sma_50
                        sma_50_class = "success" if sma_50_above else "danger"
                        sma_50_color = "green" if sma_50_above else "red"
                        
                        # SMA 144 analysis
                        sma_144_above = current_price > current_sma_144
                        sma_144_class = "success" if sma_144_above else "danger"
                        sma_144_color = "green" if sma_144_above else "red"
                        
                        # Trend analysis
                        if current_sma_20 > current_sma_50 > current_sma_144:
                            trend_class = "success"
                            trend_color = "green"
                            trend_status = "Bullish"
                        elif current_sma_20 < current_sma_50 < current_sma_144:
                            trend_class = "danger"
                            trend_color = "red"
                            trend_status = "Bearish"
                        else:
                            trend_class = "warning"
                            trend_color = "orange"
                            trend_status = "Mixed"
                        
                        st.markdown(f'''
                        <div class="criteria-card">
                            <div class="criteria-header">
                                <i class="fas fa-chart-line"></i> Moving Averages
                            </div>
                            <div class="criteria-item {sma_20_class}">
                                <strong><i class="fas fa-arrow-up"></i> SMA 20:</strong>
                                <span style='color: {sma_20_color}; font-weight: 600;'>${current_sma_20:.2f} ({current_price - current_sma_20:+.2f})</span>
                            </div>
                            <div class="criteria-item {sma_50_class}">
                                <strong><i class="fas fa-chart-line"></i> SMA 50:</strong>
                                <span style='color: {sma_50_color}; font-weight: 600;'>${current_sma_50:.2f} ({current_price - current_sma_50:+.2f})</span>
                            </div>
                            <div class="criteria-item {sma_144_class}">
                                <strong><i class="fas fa-arrows-alt-h"></i> SMA 144:</strong>
                                <span style='color: {sma_144_color}; font-weight: 600;'>${current_sma_144:.2f} ({current_price - current_sma_144:+.2f})</span>
                            </div>
                            <div class="criteria-item {trend_class}">
                                <strong><i class="fas fa-trending-up"></i> Trend:</strong>
                                <span style='color: {trend_color}; font-weight: 600;'>{trend_status}</span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        # Bollinger Bands analysis
                        bb_position = ((current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)) * 100
                        
                        if current_price > current_bb_upper:
                            bb_class = "danger"
                            bb_color = "red"
                            bb_status = "Overbought"
                        elif current_price < current_bb_lower:
                            bb_class = "success"
                            bb_color = "green"
                            bb_status = "Oversold"
                        else:
                            bb_class = "warning"
                            bb_color = "orange"
                            bb_status = "Neutral"
                        
                        st.markdown(f'''
                        <div class="criteria-card">
                            <div class="criteria-header">
                                <i class="fas fa-chart-area"></i> Bollinger Bands
                            </div>
                            <div class="criteria-item {bb_class}">
                                <strong><i class="fas fa-percentage"></i> BB Position:</strong>
                                <span style='color: {bb_color}; font-weight: 600;'>{bb_position:.1f}% ({bb_status})</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-arrow-up"></i> BB Upper:</strong>
                                <span style='color: #666; font-weight: 600;'>${current_bb_upper:.2f}</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-arrow-down"></i> BB Lower:</strong>
                                <span style='color: #666; font-weight: 600;'>${current_bb_lower:.2f}</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-arrows-alt-h"></i> BB Width:</strong>
                                <span style='color: #666; font-weight: 600;'>${current_bb_upper - current_bb_lower:.2f}</span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        # PMA analysis
                        pma_above = current_pma > 0
                        pma_class = "success" if pma_above else "danger"
                        pma_color = "green" if pma_above else "red"
                        
                        # Cycle analysis
                        cycle_positive = current_cycle > 0
                        cycle_class = "success" if cycle_positive else "danger"
                        cycle_color = "green" if cycle_positive else "red"
                        
                        st.markdown(f'''
                        <div class="criteria-card">
                            <div class="criteria-header">
                                <i class="fas fa-wave-square"></i> Price/MA Analysis
                            </div>
                            <div class="criteria-item {pma_class}">
                                <strong><i class="fas fa-percentage"></i> Price/MA %:</strong>
                                <span style='color: {pma_color}; font-weight: 600;'>{current_pma:.2f}%</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-signal"></i> Signal Line:</strong>
                                <span style='color: #666; font-weight: 600;'>{current_signal:.2f}%</span>
                            </div>
                            <div class="criteria-item {cycle_class}">
                                <strong><i class="fas fa-sync"></i> Cycle:</strong>
                                <span style='color: {cycle_color}; font-weight: 600;'>{current_cycle:.2f}%</span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Add volume analysis (only if volume data exists)
                    if 'volume' in df.columns:
                        st.subheader("📊 Volume Analysis")
                        col_vol1, col_vol2, col_vol3 = st.columns(3)
                        
                        with col_vol1:
                            current_volume = df['volume'].iloc[-1]
                            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                            volume_ratio = current_volume / avg_volume
                            
                            if volume_ratio > 1.5:
                                volume_status = "🔴 High Volume"
                            elif volume_ratio < 0.5:
                                volume_status = "🟢 Low Volume"
                            else:
                                volume_status = "🟡 Normal Volume"
                            
                            st.metric("Current Volume", f"{current_volume:,.0f}")
                            st.metric("Avg Volume (20d)", f"{avg_volume:,.0f}")
                            st.metric("Volume Ratio", f"{volume_ratio:.2f}x", volume_status)
                        
                        with col_vol2:
                            # Price change analysis
                            price_change_1d = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                            price_change_5d = ((current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
                            price_change_20d = ((current_price - df['close'].iloc[-21]) / df['close'].iloc[-21]) * 100
                            
                            st.metric("1-Day Change", f"{price_change_1d:+.2f}%")
                            st.metric("5-Day Change", f"{price_change_5d:+.2f}%")
                            st.metric("20-Day Change", f"{price_change_20d:+.2f}%")
                        
                        with col_vol3:
                            # Volatility analysis
                            returns = df['close'].pct_change().dropna()
                            volatility_20d = returns.rolling(20).std().iloc[-1] * 100
                            volatility_60d = returns.rolling(60).std().iloc[-1] * 100
                            
                            st.metric("20-Day Volatility", f"{volatility_20d:.2f}%")
                            st.metric("60-Day Volatility", f"{volatility_60d:.2f}%")
                            
                            if volatility_20d > volatility_60d:
                                vol_status = "🔴 Increasing"
                            else:
                                vol_status = "🟢 Decreasing"
                            
                            st.metric("Volatility Trend", vol_status)
                    else:
                        # Show price change and volatility analysis without volume
                        st.markdown('<div class="section-header"><i class="fas fa-chart-area"></i> Price & Volatility Analysis</div>', unsafe_allow_html=True)
                        col_price, col_vol = st.columns(2)
                        
                        with col_price:
                            # Price change analysis
                            price_change_1d = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                            price_change_5d = ((current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
                            price_change_20d = ((current_price - df['close'].iloc[-21]) / df['close'].iloc[-21]) * 100
                            
                            # Determine classes and colors for price changes
                            change_1d_class = "success" if price_change_1d >= 0 else "danger"
                            change_1d_color = "green" if price_change_1d >= 0 else "red"
                            
                            change_5d_class = "success" if price_change_5d >= 0 else "danger"
                            change_5d_color = "green" if price_change_5d >= 0 else "red"
                            
                            change_20d_class = "success" if price_change_20d >= 0 else "danger"
                            change_20d_color = "green" if price_change_20d >= 0 else "red"
                            
                            st.markdown(f'''
                            <div class="criteria-card">
                                <div class="criteria-header">
                                    <i class="fas fa-chart-line"></i> Price Performance
                                </div>
                                <div class="criteria-item {change_1d_class}">
                                    <strong><i class="fas fa-calendar-day"></i> 1-Day Change:</strong>
                                    <span style='color: {change_1d_color}; font-weight: 600;'>{price_change_1d:+.2f}%</span>
                                </div>
                                <div class="criteria-item {change_5d_class}">
                                    <strong><i class="fas fa-calendar-week"></i> 5-Day Change:</strong>
                                    <span style='color: {change_5d_color}; font-weight: 600;'>{price_change_5d:+.2f}%</span>
                                </div>
                                <div class="criteria-item {change_20d_class}">
                                    <strong><i class="fas fa-calendar-alt"></i> 20-Day Change:</strong>
                                    <span style='color: {change_20d_color}; font-weight: 600;'>{price_change_20d:+.2f}%</span>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col_vol:
                            # Volatility analysis
                            returns = df['close'].pct_change().dropna()
                            volatility_20d = returns.rolling(20).std().iloc[-1] * 100
                            volatility_60d = returns.rolling(60).std().iloc[-1] * 100
                            
                            # Determine volatility trend and classes
                            vol_increasing = volatility_20d > volatility_60d
                            vol_trend_class = "danger" if vol_increasing else "success"
                            vol_trend_color = "red" if vol_increasing else "green"
                            vol_status = "Increasing" if vol_increasing else "Decreasing"
                            
                            # Volatility level assessment
                            if volatility_20d > 30:
                                vol_level_class = "danger"
                                vol_level_color = "red"
                                vol_level = "High"
                            elif volatility_20d > 15:
                                vol_level_class = "warning"
                                vol_level_color = "orange"
                                vol_level = "Moderate"
                            else:
                                vol_level_class = "success"
                                vol_level_color = "green"
                                vol_level = "Low"
                            
                            st.markdown(f'''
                            <div class="criteria-card">
                                <div class="criteria-header">
                                    <i class="fas fa-wave-square"></i> Volatility Analysis
                                </div>
                                <div class="criteria-item {vol_level_class}">
                                    <strong><i class="fas fa-chart-area"></i> 20-Day Volatility:</strong>
                                    <span style='color: {vol_level_color}; font-weight: 600;'>{volatility_20d:.2f}% ({vol_level})</span>
                                </div>
                                <div class="criteria-item">
                                    <strong><i class="fas fa-history"></i> 60-Day Volatility:</strong>
                                    <span style='color: #666; font-weight: 600;'>{volatility_60d:.2f}%</span>
                                </div>
                                <div class="criteria-item {vol_trend_class}">
                                    <strong><i class="fas fa-trending-up"></i> Volatility Trend:</strong>
                                    <span style='color: {vol_trend_color}; font-weight: 600;'>{vol_status}</span>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # Add support and resistance levels
                    st.markdown('<div class="section-header"><i class="fas fa-crosshairs"></i> Support & Resistance Levels</div>', unsafe_allow_html=True)
                    
                    # Calculate recent highs and lows
                    recent_high = df['high'].tail(20).max()
                    recent_low = df['low'].tail(20).min()
                    current_high = df['high'].iloc[-1]
                    current_low = df['low'].iloc[-1]
                    
                    col_sr1, col_sr2, col_sr3 = st.columns(3)
                    
                    with col_sr1:
                        resistance_distance = ((recent_high - current_price) / current_price) * 100
                        
                        # Determine resistance level status
                        if resistance_distance > 5:
                            resistance_class = "success"
                            resistance_color = "green"
                            resistance_status = "Far"
                        elif resistance_distance > 2:
                            resistance_class = "warning"
                            resistance_color = "orange"
                            resistance_status = "Near"
                        else:
                            resistance_class = "danger"
                            resistance_color = "red"
                            resistance_status = "Very Close"
                        
                        st.markdown(f'''
                        <div class="criteria-card">
                            <div class="criteria-header">
                                <i class="fas fa-arrow-up"></i> Resistance Levels
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-mountain"></i> Recent High (20d):</strong>
                                <span style='color: #666; font-weight: 600;'>${recent_high:.2f}</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-chart-line"></i> Current High:</strong>
                                <span style='color: #666; font-weight: 600;'>${current_high:.2f}</span>
                            </div>
                            <div class="criteria-item {resistance_class}">
                                <strong><i class="fas fa-ruler-vertical"></i> Distance to Resistance:</strong>
                                <span style='color: {resistance_color}; font-weight: 600;'>{resistance_distance:+.2f}% ({resistance_status})</span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_sr2:
                        support_distance = ((current_price - recent_low) / current_price) * 100
                        
                        # Determine support level status
                        if support_distance > 5:
                            support_class = "success"
                            support_color = "green"
                            support_status = "Strong"
                        elif support_distance > 2:
                            support_class = "warning"
                            support_color = "orange"
                            support_status = "Moderate"
                        else:
                            support_class = "danger"
                            support_color = "red"
                            support_status = "Weak"
                        
                        st.markdown(f'''
                        <div class="criteria-card">
                            <div class="criteria-header">
                                <i class="fas fa-arrow-down"></i> Support Levels
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-valley"></i> Recent Low (20d):</strong>
                                <span style='color: #666; font-weight: 600;'>${recent_low:.2f}</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-chart-line"></i> Current Low:</strong>
                                <span style='color: #666; font-weight: 600;'>${current_low:.2f}</span>
                            </div>
                            <div class="criteria-item {support_class}">
                                <strong><i class="fas fa-ruler-vertical"></i> Distance to Support:</strong>
                                <span style='color: {support_color}; font-weight: 600;'>{support_distance:+.2f}% ({support_status})</span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_sr3:
                        # Risk/Reward ratio using technical indicators
                        rr_data = TechnicalIndicators.risk_reward_ratio(current_price, recent_high, recent_low)
                        risk = rr_data['risk']
                        reward = rr_data['reward']
                        risk_reward_ratio = rr_data['ratio']
                        rr_status = rr_data['status']
                        
                        # Determine display colors based on status
                        if rr_status == "Favorable":
                            rr_class = "success"
                            rr_color = "green"
                        elif rr_status == "Balanced":
                            rr_class = "warning"
                            rr_color = "orange"
                        else:
                            rr_class = "danger"
                            rr_color = "red"
                        
                        st.markdown(f'''
                        <div class="criteria-card">
                            <div class="criteria-header">
                                <i class="fas fa-balance-scale"></i> Risk/Reward Analysis
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-exclamation-triangle"></i> Risk:</strong>
                                <span style='color: #666; font-weight: 600;'>${risk:.2f}</span>
                            </div>
                            <div class="criteria-item">
                                <strong><i class="fas fa-gift"></i> Reward:</strong>
                                <span style='color: #666; font-weight: 600;'>${reward:.2f}</span>
                            </div>
                            <div class="criteria-item {rr_class}">
                                <strong><i class="fas fa-calculator"></i> Risk/Reward Ratio:</strong>
                                <span style='color: {rr_color}; font-weight: 600;'>{risk_reward_ratio:.2f} ({rr_status})</span>
                            </div>
                            <div class="score-badge" style='background: linear-gradient(135deg, {rr_color}, {'#20c997' if rr_class == 'success' else '#fd7e14' if rr_class == 'warning' else '#e83e8c'}); color: white;'>
                                <i class="fas fa-chart-pie"></i> R/R Score: {risk_reward_ratio:.2f}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                else:
                    st.error("Failed to create technical analysis chart")
            else:
                st.info("👆 Run analysis first to see technical analysis chart")
        
        with tab2:
            st.markdown('<div class="section-header"><i class="fas fa-crystal-ball"></i> AI Stock Prediction Results</div>', unsafe_allow_html=True)
            
            # Clean up memory before intensive prediction tasks
            cleanup_memory()
            
            # Check if in tech-only mode
            is_tech_only = tech_only if 'tech_only' in locals() else st.session_state.analysis_results.get('tech_only', False)
            
            if is_tech_only:
                st.info("""
                📊 **Technical Analysis Only Mode**
                
                This tab shows AI stock predictions and requires full analysis mode.
                
                To see prediction results:
                1. Uncheck "📊 Technical Analysis Only" in the sidebar
                2. Click "🚀 Run Full Analysis"
                """)
            else:
                # Display training errors if any
                if training_error:
                    if training_error['type'] == 'training_failed':
                        st.error(f"❌ Model training failed with return code {training_error['return_code']}")
                        
                        # Show both stdout and stderr if available
                        if training_error.get('stdout'):
                            st.subheader("Training Output:")
                            st.code(training_error['stdout'], language="text")
                        
                        if training_error.get('stderr'):
                            st.subheader("Error Output:")
                            st.code(training_error['stderr'], language="text")
                        
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
                    elif training_error['type'] == 'timeout':
                        st.error(f"❌ {training_error['message']}")
                    elif training_error['type'] == 'general_error':
                        st.error(f"❌ Training error: {training_error['message']}")
            
            # Display basic results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                price_color = "green" if price_change >= 0 else "red"
                change_symbol = "+" if price_change >= 0 else ""
                
                # Determine price change class
                price_class = "success" if price_change >= 0 else "danger"
                
                st.markdown(f'''
                <div class="criteria-card">
                    <div class="criteria-header">
                        <i class="fas fa-chart-line"></i> {ticker} Current Market Data
                    </div>
                    <div class="criteria-item {price_class}">
                        <strong><i class="fas fa-dollar-sign"></i> Current Price:</strong>
                        <span style='color: {price_color}; font-weight: 600; font-size: 1.2em;'>${current_price:.2f} ({change_symbol}{price_change_pct:+.2f}%)</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-play"></i> Open:</strong>
                        <span style='color: #666; font-weight: 600;'>${open_today:.2f}</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-arrow-up"></i> High:</strong>
                        <span style='color: #666; font-weight: 600;'>${high_today:.2f}</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-arrow-down"></i> Low:</strong>
                        <span style='color: #666; font-weight: 600;'>${low_today:.2f}</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-water"></i> Volume:</strong>
                        <span style='color: #666; font-weight: 600;'>{current_volume:,.0f}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                if proba is not None:
                    best_class_idx = np.argmax(proba)
                    best_prob = proba[best_class_idx]
                    prediction_text = class_labels.get(best_class_idx, f"Class {best_class_idx}")
                    
                    # Determine prediction class based on probability
                    if best_prob >= 0.7:
                        pred_class = "success"
                        pred_color = "green"
                        confidence_level = "High"
                    elif best_prob >= 0.5:
                        pred_class = "warning" 
                        pred_color = "orange"
                        confidence_level = "Medium"
                    else:
                        pred_class = "danger"
                        pred_color = "red"
                        confidence_level = "Low"
                    
                    st.markdown(f'''
                    <div class="criteria-card">
                        <div class="criteria-header">
                            <i class="fas fa-crystal-ball"></i> AI Prediction Results
                        </div>
                        <div class="criteria-item {pred_class}">
                            <strong><i class="fas fa-target"></i> Predicted Outcome:</strong>
                            <span style='color: {pred_color}; font-weight: 600; font-size: 1.1em;'>{prediction_text}</span>
                        </div>
                        <div class="criteria-item {pred_class}">
                            <strong><i class="fas fa-percentage"></i> Confidence:</strong>
                            <span style='color: {pred_color}; font-weight: 600;'>{best_prob:.1%} ({confidence_level})</span>
                        </div>
                        <div class="score-badge" style='background: linear-gradient(135deg, {pred_color}, {'#20c997' if pred_class == 'success' else '#fd7e14' if pred_class == 'warning' else '#e83e8c'}); color: white;'>
                            <i class="fas fa-brain"></i> AI Confidence: {best_prob:.1%}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                    <div class="criteria-card">
                        <div class="criteria-header">
                            <i class="fas fa-exclamation-triangle"></i> Prediction Status
                        </div>
                        <div class="criteria-item danger">
                            <strong><i class="fas fa-times"></i> Model Status:</strong>
                            <span style='color: red; font-weight: 600;'>No Model Available</span>
                        </div>
                        <div class="criteria-item">
                            <strong><i class="fas fa-info-circle"></i> Note:</strong>
                            <span style='color: #666; font-weight: 600;'>Pattern analysis available in Fractal tab</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
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
                                # Check if the loaded model has the same number of features as current training_features
                                if hasattr(day_model, 'n_features_in_'):
                                    if day_model.n_features_in_ == len(training_features):
                                        # Test if the model can actually make predictions
                                        try:
                                            # Create a dummy prediction to test the model
                                            dummy_X = np.zeros((1, len(training_features)))
                                            day_model.predict(dummy_X)
                                            day_models[day_ahead] = day_model
                                            continue
                                        except Exception as pred_error:
                                            st.info(f"🔄 Multi-day model {day_ahead} loaded but prediction test failed: {str(pred_error)}. Retraining...")
                                            # Remove the invalid model from day_models if it was added
                                            if day_ahead in day_models:
                                                del day_models[day_ahead]
                                    else:
                                        st.info(f"🔄 Multi-day model {day_ahead} has {day_model.n_features_in_} features, but current data has {len(training_features)} features. Retraining...")
                                        # Remove the invalid model from day_models if it was added
                                        if day_ahead in day_models:
                                            del day_models[day_ahead]
                                else:
                                    # For models without n_features_in_, test prediction capability
                                    try:
                                        dummy_X = np.zeros((1, len(training_features)))
                                        day_model.predict(dummy_X)
                                        day_models[day_ahead] = day_model
                                        continue
                                    except Exception as pred_error:
                                        st.info(f"🔄 Multi-day model {day_ahead} loaded but prediction test failed: {str(pred_error)}. Retraining...")
                                        # Remove the invalid model from day_models if it was added
                                        if day_ahead in day_models:
                                            del day_models[day_ahead]
                            except Exception as e:
                                st.info(f"🔄 Failed to load multi-day model {day_ahead}: {str(e)}. Will retrain.")
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
                        
                        # Train model if enough data OR if we need to retrain due to validation failure
                        should_train = len(feature_data) > 50 or day_ahead not in day_models
                        
                        if should_train:
                            # if len(feature_data) > 50:
                            #     st.info(f"📊 Training multi-day model {day_ahead} with {len(feature_data)} samples and {len(training_features)} features")
                            # else:
                            #     st.warning(f"⚠️ Retraining multi-day model {day_ahead} with limited data: {len(feature_data)} samples (need at least 50)")
                            
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
                            # st.success(f"✅ Multi-day model {day_ahead} trained and saved successfully!")
                        else:
                            st.warning(f"⚠️ Insufficient data for multi-day model {day_ahead}: {len(feature_data)} samples (need at least 50)")
                            continue
                    
                    # Generate predictions for bubble chart
                    successful_predictions = 0
                    for day_ahead in range(1, 8):
                        day_proba = None
                        
                        if day_ahead in day_models:
                            try:
                                # Check feature count compatibility
                                if hasattr(day_models[day_ahead], 'n_features_in_'):
                                    expected_features = day_models[day_ahead].n_features_in_
                                    actual_features = X_live.shape[1]
                                    if expected_features != actual_features:
                                        st.warning(f"⚠️ Multi-day model {day_ahead} expects {expected_features} features, but got {actual_features} features.")
                                        st.info("🔄 Skipping multi-day prediction due to feature mismatch.")
                                        continue
                                
                                day_proba = day_models[day_ahead].predict(X_live)[0]
                                successful_predictions += 1
                            except Exception as e:
                                st.warning(f"⚠️ Multi-day model {day_ahead} prediction failed: {str(e)}")
                                st.info("🔄 Skipping this multi-day model.")
                                continue
                        else:
                            st.info(f"🔄 Multi-day model {day_ahead} was not available (likely due to validation failure during loading).")
                            continue
                        
                        # Process prediction results (only if we have a valid prediction)
                        if day_proba is not None:
                            # Update main prediction with day-1 model
                            if day_ahead == 1:
                                best_class_idx = np.argmax(day_proba)
                                best_prob = day_proba[best_class_idx]
                                class_names = ['Strong Down', 'Down', 'Sideways', 'Up', 'Strong Up']
                                best_class_name = class_names[best_class_idx]
                                
                                st.success(f"✅ Day-1 Model Prediction: {best_class_name} (Confidence: {best_prob:.1%})")
                            
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
                        st.success(f"✅ Generated predictions for {successful_predictions} out of 7 days")
                    else:
                        st.warning("⚠️ No valid multi-day predictions available. This may be due to feature count mismatches or insufficient data.")
                    
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
                        
                        _ = st.plotly_chart(fig_bubble, use_container_width=True)
            # Show feature count information at the bottom
            st.info(f"📊 Loaded {len(training_features)} features for prediction")
        
        with tab3:
            st.markdown('<div class="section-header"><i class="fas fa-search"></i> Fractal Pattern Analysis Results</div>', unsafe_allow_html=True)
            
            # Clean up memory before intensive pattern matching tasks
            cleanup_memory()
            
            # Check if in tech-only mode
            is_tech_only = tech_only if 'tech_only' in locals() else st.session_state.analysis_results.get('tech_only', False)
            
            if is_tech_only:
                st.info("""
                📊 **Technical Analysis Only Mode**
                
                This tab shows fractal pattern analysis and requires full analysis mode.
                
                To see pattern matching results:
                1. Uncheck "📊 Technical Analysis Only" in the sidebar
                2. Click "🚀 Run Full Analysis"
                """)
            else:
                # ========== SEARCH CONFIGURATION DISPLAY ==========
                st.markdown('<div class="section-header"><i class="fas fa-cog"></i> Search Configuration</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'''
                <div class="criteria-card">
                    <div class="criteria-header">
                        <i class="fas fa-search"></i> Pattern Settings
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-chart-line"></i> Reference Pattern:</strong>
                        <span style='color: #666; font-weight: 600;'>{pattern_length} {timeframe_label}</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-calculator"></i> Similarity Method:</strong>
                        <span style='color: #666; font-weight: 600;'>Cosine Similarity</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-clock"></i> Timeframe:</strong>
                        <span style='color: #666; font-weight: 600;'>{timeframe}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="criteria-card">
                    <div class="criteria-header">
                        <i class="fas fa-sliders-h"></i> Analysis Parameters
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-ruler"></i> Pattern Length:</strong>
                        <span style='color: #666; font-weight: 600;'>{pattern_length} periods</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-percentage"></i> Similarity Threshold:</strong>
                        <span style='color: #666; font-weight: 600;'>{similarity_threshold:.2f}</span>
                    </div>
                    <div class="criteria-item">
                        <strong><i class="fas fa-window-maximize"></i> Window Sizes:</strong>
                        <span style='color: #666; font-weight: 600;'>{max(5, int(pattern_length * 0.74))} to {max(max(5, int(pattern_length * 0.74)) + 1, int(pattern_length * 1.24))}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
                with col3:
                    # Only show Analysis Results section when we have actual matches (not in tech-only mode)
                    if matches and len(matches) > 0:
                        # Determine match quality based on number of matches found
                        if len(matches) > 10:
                            match_class = "success"
                            match_color = "green"
                            match_quality = "Excellent"
                        elif len(matches) > 5:
                            match_class = "warning"
                            match_color = "orange" 
                            match_quality = "Good"
                        elif len(matches) > 0:
                            match_class = "danger"
                            match_color = "red"
                            match_quality = "Limited"
                        else:
                            match_class = "danger"
                            match_color = "red"
                            match_quality = "None"
                        
                        st.markdown(f'''
                    <div class="criteria-card">
                        <div class="criteria-header">
                            <i class="fas fa-chart-bar"></i> Analysis Results
                        </div>
                        <div class="criteria-item">
                            <strong><i class="fas fa-calendar"></i> Analysis Period:</strong>
                            <span style='color: #666; font-weight: 600;'>{len(analysis_df)} {timeframe_label}</span>
                        </div>
                        <div class="criteria-item {match_class}">
                            <strong><i class="fas fa-search"></i> Total Matches:</strong>
                            <span style='color: {match_color}; font-weight: 600;'>{len(matches)} ({match_quality})</span>
                        </div>
                        <div class="criteria-item">
                            <strong><i class="fas fa-list"></i> Top Matches Shown:</strong>
                            <span style='color: #666; font-weight: 600;'>{min(20, len(matches))}</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
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
                
                # Display interactive table with row selection
                st.write("**Click on a row to view detailed charts:**")
                
                # Use streamlit's dataframe with on_select event
                event = st.dataframe(
                    price_context_df,
                    column_config={
                        "Pattern": st.column_config.TextColumn("Pattern"),
                        "Date Range": st.column_config.TextColumn("Date Range"),
                        "Start Price": st.column_config.TextColumn("Start Price"),
                        "End Price": st.column_config.TextColumn("End Price"),
                        "Min/Max": st.column_config.TextColumn("Min/Max"),
                        "Change": st.column_config.TextColumn("Change"),
                        "Similarity": st.column_config.TextColumn("Similarity")
                    },
                    hide_index=True,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="pattern_table"
                )
                
                # Check if a row is selected
                selected_match = None
                selected_row_idx = None
                
                if event.selection.rows:
                    selected_row_idx = event.selection.rows[0]
                    
                    # Adjust for reference pattern row (row 0 is reference, actual matches start from row 1)
                    if selected_row_idx == 0:
                        # Reference pattern selected - use first match for display
                        if matches_sorted:
                            selected_match = matches_sorted[0]
                    elif 1 <= selected_row_idx <= len(matches_sorted):
                        # Similar pattern selected - adjust index
                        match_idx = selected_row_idx - 1
                        selected_match = matches_sorted[match_idx]
                    else:
                        # Fallback to first match
                        if matches_sorted:
                            selected_match = matches_sorted[0]
                else:
                    # Default to top match if no selection
                    if matches_sorted:
                        selected_match = matches_sorted[0]
                
                # Display chart for selected match
                if selected_match is not None:
                    st.markdown("### 📈 Pattern Match Detail Chart")
                    if event.selection.rows and selected_row_idx is not None:
                        if selected_row_idx == 0:
                            st.write(f"**Reference Pattern (using first match)** | "
                                    f"**Similarity:** {selected_match['similarity']:.3f} | "
                                    f"**Date Range:** {series_dates[selected_match['start']].strftime('%Y-%m-%d')} to "
                                    f"{series_dates[selected_match['start'] + selected_match['size'] - 1].strftime('%Y-%m-%d')}")
                        else:
                            st.write(f"**Selected Match #{selected_row_idx}** | "
                                    f"**Similarity:** {selected_match['similarity']:.3f} | "
                                    f"**Date Range:** {series_dates[selected_match['start']].strftime('%Y-%m-%d')} to "
                                    f"{series_dates[selected_match['start'] + selected_match['size'] - 1].strftime('%Y-%m-%d')}")
                    else:
                        st.write("**Showing Top Match** | "
                                f"**Similarity:** {selected_match['similarity']:.3f} | "
                                f"**Date Range:** {series_dates[selected_match['start']].strftime('%Y-%m-%d')} to "
                                f"{series_dates[selected_match['start'] + selected_match['size'] - 1].strftime('%Y-%m-%d')}")
                else:
                    st.warning("No matches available to display.")
                    return
                
                # Create the similarity overlay chart (similar to fractal.py)
                if selected_match is None:
                    st.warning("No matches available to display.")
                    return
                
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
                
                # Add data point annotations for easier reading
                extended_prices = price_series[idx_range_extended]
                extended_dates = series_dates[idx_range_extended]
                
                # Calculate chart bounds for positioning annotations
                price_min_extended = extended_prices.min()
                price_max_extended = extended_prices.max()
                price_range_extended = price_max_extended - price_min_extended
                
                # Position annotations left and right, middle vertically
                left_date = extended_dates[int(len(extended_dates)*0.02)]  # 2% from left
                right_date = extended_dates[int(len(extended_dates)*0.95)]  # 5% from right
                middle_y = price_min_extended + (price_range_extended * 0.5)  # Middle vertically
                
                # Start annotation on left middle
                match_start_price = price_series[m_start]
                ax.annotate(f'Match Start\n{match_start_date.strftime("%Y-%m-%d")}\n${match_start_price:.2f}', 
                           xy=(match_start_date, match_start_price), 
                           xytext=(left_date, middle_y),
                           arrowprops=dict(arrowstyle='->', color='green', alpha=0.6),
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                           fontsize=8, ha='center', va='center')
                
                # End annotation on right middle
                match_end_price = price_series[m_start + m_size - 1]
                ax.annotate(f'Match End\n{match_end_date.strftime("%Y-%m-%d")}\n${match_end_price:.2f}', 
                           xy=(match_end_date, match_end_price), 
                           xytext=(right_date, middle_y),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                           fontsize=8, ha='center', va='center')
                
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
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.tight_layout()
                _ = st.pyplot(fig)
                
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
                
                # Add detailed data table for precise reading
                with st.expander("📊 Detailed Price Data for This Match", expanded=False):
                    # Create a detailed dataframe with all the data points in the chart
                    chart_data = []
                    for i in idx_range_extended:
                        is_match_period = i >= m_start and i < (m_start + m_size)
                        chart_data.append({
                            'Date': series_dates[i].strftime('%Y-%m-%d'),
                            'Price': f"${price_series[i]:.2f}",
                            'Period': 'Match Period' if is_match_period else 'Context',
                            'Day_of_Week': series_dates[i].strftime('%A'),
                            'Price_Change': f"{((price_series[i] / price_series[i-1] - 1) * 100):.1f}%" if i > 0 and i-1 in idx_range_extended else "N/A"
                        })
                    
                    chart_df = pd.DataFrame(chart_data)
                    
                    # Display summary statistics
                    match_prices = [price_series[i] for i in idx_range_match]
                    st.write(f"**Match Period Statistics:**")
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Start Price", f"${match_prices[0]:.2f}")
                        st.metric("End Price", f"${match_prices[-1]:.2f}")
                    with col_stat2:
                        st.metric("Min Price", f"${min(match_prices):.2f}")
                        st.metric("Max Price", f"${max(match_prices):.2f}")
                    with col_stat3:
                        total_change = ((match_prices[-1] - match_prices[0]) / match_prices[0]) * 100
                        st.metric("Total Change", f"{total_change:+.1f}%")
                        st.metric("Avg Daily Change", f"{total_change / len(match_prices):.1f}%")
                    with col_stat4:
                        volatility = np.std(match_prices) / np.mean(match_prices) * 100
                        st.metric("Volatility", f"{volatility:.1f}%")
                        st.metric("Price Range", f"${max(match_prices) - min(match_prices):.2f}")
                    
                    # Display the detailed data table
                    _ = st.dataframe(chart_df, use_container_width=True)
                
            else:
                st.warning(f"No similar patterns found with similarity threshold {similarity_threshold:.2f}")
                st.info("Try lowering the similarity threshold or using a different pattern length.")
    
    else:
        # Show placeholders when no analysis has been run
        with tab1:
            st.markdown('<div class="section-header"><i class="fas fa-chart-line"></i> Technical Analysis</div>', unsafe_allow_html=True)
            st.info("👆 Click 'Run Prediction' in the sidebar to see technical analysis chart.")
        
        with tab2:
            st.markdown('<div class="section-header"><i class="fas fa-crystal-ball"></i> AI Stock Prediction</div>', unsafe_allow_html=True)
            st.info("👆 Click 'Run Prediction' in the sidebar to see stock prediction results.")
        
        with tab3:
            st.markdown('<div class="section-header"><i class="fas fa-search"></i> Fractal Pattern Analysis</div>', unsafe_allow_html=True)
            st.info("👆 Click 'Run Prediction' in the sidebar to see fractal pattern analysis results.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        ⚠️ <strong>Disclaimer</strong>: This tool is for educational purposes only. 
        Pattern analysis and predictions should not be used as sole investment advice. 
        Always consult with a financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Explicitly return None to prevent any value from being displayed
    return None

if __name__ == "__main__":
    _ = main()
