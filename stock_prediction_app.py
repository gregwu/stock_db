import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import yfinance as yf

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

# Initialize session state for predictions and current view date
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'current_view_date' not in st.session_state:
    st.session_state.current_view_date = None
if 'last_base_date' not in st.session_state:
    st.session_state.last_base_date = None

def get_database_connection():
    """Create database connection"""
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def get_available_tickers():
    """Get list of available tickers from database"""
    engine = get_database_connection()
    query = text("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker")
    
    with engine.connect() as conn:
        result = conn.execute(query)
        tickers = [row[0] for row in result.fetchall()]
    
    return tickers

def clean_ticker_display(ticker):
    """Remove .US suffix from ticker for display purposes"""
    return ticker.replace('.US', '') if ticker.endswith('.US') else ticker

def get_stock_data_from_date(ticker, end_date, days_before=50, include_indicators=True):
    """Get stock data for specified ticker starting from days_before the end_date"""
    engine = get_database_connection()
    
    # If we need indicators (especially SMA 144), fetch more data to ensure proper calculation
    buffer_days = 200 if include_indicators else 20
    start_date = end_date - timedelta(days=days_before + buffer_days)  # Larger buffer for indicators
    
    query = text("""
    SELECT date, open_price, high_price, low_price, close_price, volume,
           sma_20, sma_144, rsi_14, macd, macd_signal, bb_upper, bb_lower,
           touches_lower_bb, touches_upper_bb, volume_spike
    FROM stock_data 
    WHERE ticker = :ticker 
      AND date >= :start_date 
      AND date <= :end_date
    ORDER BY date
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })
    
    # Return only the last 'days_before' trading days up to end_date
    if not df.empty and len(df) > days_before:
        df = df.tail(days_before)
    
    return df

def get_next_day_data(ticker, from_date):
    """Get the next trading day's data after from_date"""
    engine = get_database_connection()
    
    query = text("""
    SELECT date, open_price, high_price, low_price, close_price, volume
    FROM stock_data 
    WHERE ticker = :ticker 
      AND date > :from_date
    ORDER BY date
    LIMIT 1
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            'ticker': ticker,
            'from_date': from_date
        })
    
    return df

def get_future_data(ticker, start_date, days=30):
    """Get future data for scoring predictions"""
    engine = get_database_connection()
    end_date = start_date + timedelta(days=days)
    
    query = text("""
    SELECT date, close_price
    FROM stock_data 
    WHERE ticker = :ticker 
      AND date > :start_date 
      AND date <= :end_date
    ORDER BY date
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })
    
    return df

def create_stock_chart(df, ticker, predictions=None, future_data=None):
    """Create interactive stock chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.5, 0.15, 0.175, 0.175]
    )
    
    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open_price'],
            high=df['high_price'],
            low=df['low_price'],
            close=df['close_price'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_20' in df.columns and df['sma_20'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'sma_144' in df.columns and df['sma_144'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_144'], name='SMA 144', line=dict(color='purple')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns and df['bb_upper'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['close_price'], df['open_price'])]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'rsi_14' in df.columns and df['rsi_14'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['rsi_14'], name='RSI', line=dict(color='blue')),
            row=3, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in df.columns and df['macd'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['macd'], name='MACD', line=dict(color='blue')),
            row=4, col=1
        )
    if 'macd_signal' in df.columns and df['macd_signal'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['macd_signal'], name='Signal', line=dict(color='red')),
            row=4, col=1
        )
    
    # Add future data if available
    if future_data is not None and not future_data.empty:
        fig.add_trace(
            go.Scatter(
                x=future_data['date'], 
                y=future_data['close_price'], 
                name='Future Data',
                line=dict(color='lightblue', dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add prediction markers if available
    if predictions:
        last_date = df['date'].iloc[-1]
        last_price = df['close_price'].iloc[-1]
        
        for pred in predictions:
            if pred['ticker'] == ticker:
                # Add prediction markers for all 5 days
                pred_dates = [last_date + timedelta(days=i) for i in [1, 2, 3, 4, 5]]
                colors = ['green' if pred[f'day_{i}_prediction'] == 'UP' else 'red' for i in [1, 2, 3, 4, 5]]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=[last_price] * 5,
                        mode='markers',
                        marker=dict(size=12, color=colors, symbol='triangle-up'),
                        name='Predictions',
                        text=[f"Day {i}: {pred[f'day_{i}_prediction']}" for i in [1, 2, 3, 4, 5]],
                        textposition="top center"
                    ),
                    row=1, col=1
                )
    
    fig.update_layout(
        title=f'{clean_ticker_display(ticker)} Stock Analysis',
        xaxis_title='Date',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def calculate_prediction_score(predictions, ticker):
    """Calculate score for predictions"""
    scores = []
    
    for pred in predictions:
        if pred['ticker'] != ticker:
            continue
            
        # Get actual future data
        future_data = get_future_data(ticker, pred['prediction_date'], days=10)
        
        if future_data.empty:
            continue
        
        base_price = pred['base_price']
        score_data = {'prediction_date': pred['prediction_date'], 'scores': {}}
        
        for day in [1, 2, 3, 4, 5]:
            if len(future_data) >= day:
                actual_price = future_data.iloc[day-1]['close_price']
                
                # For day 1, compare to base price (last day of historical data)
                # For day 2+, compare to previous day's price
                if day == 1:
                    prev_price = base_price
                else:
                    prev_price = future_data.iloc[day-2]['close_price']
                
                actual_direction = 'UP' if actual_price > prev_price else 'DOWN'
                predicted_direction = pred[f'day_{day}_prediction']
                
                correct = actual_direction == predicted_direction
                score_data['scores'][f'day_{day}'] = {
                    'predicted': predicted_direction,
                    'actual': actual_direction,
                    'correct': correct,
                    'base_price': base_price,
                    'prev_price': prev_price,
                    'actual_price': actual_price,
                    'change_pct': ((actual_price - prev_price) / prev_price) * 100
                }
        
        scores.append(score_data)
    
    return scores

def fetch_additional_yahoo_data(ticker, end_date, days_needed=200):
    """Fetch additional historical data from Yahoo Finance for technical indicators"""
    try:
        # Remove .US suffix for Yahoo Finance API
        yahoo_ticker = ticker.replace('.US', '') if ticker.endswith('.US') else ticker
        
        # Calculate start date to get enough data for SMA 144
        start_date = end_date - timedelta(days=days_needed)
        
        print(f"Fetching additional data for {yahoo_ticker} from {start_date} to {end_date}")
        
        df = yf.download(yahoo_ticker, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=True)
        
        if df.empty:
            st.toast(f"⚠️ No additional data found for {yahoo_ticker}", icon='⚠️')
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
        
        df.rename(columns={
            "Date": "date",
            "Open": "open_price", 
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Volume": "volume"
        }, inplace=True)
        
        # Convert date to proper format
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        return df
        
    except Exception as e:
        st.toast(f"❌ Error fetching data from Yahoo Finance: {e}", icon='❌')
        return pd.DataFrame()

def ensure_sufficient_data_for_indicators(ticker, end_date, min_days_needed=200):
    """Ensure we have enough historical data for technical indicators, fetch from Yahoo if needed"""
    
    # First, check how much data we have in the database
    engine = get_database_connection()
    start_check_date = end_date - timedelta(days=min_days_needed + 50)
    
    query = text("""
    SELECT COUNT(*) as count
    FROM stock_data 
    WHERE ticker = :ticker 
      AND date >= :start_date 
      AND date <= :end_date
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'ticker': ticker, 'start_date': start_check_date, 'end_date': end_date})
        existing_count = result.fetchone()[0]
    
    print(f"Found {existing_count} days of data for {ticker}")
    
    # If we don't have enough data, fetch from Yahoo Finance
    if existing_count < min_days_needed:
        with st.spinner(f"Fetching additional historical data for {ticker} to calculate technical indicators..."):
            # Fetch additional data from Yahoo Finance
            yahoo_data = fetch_additional_yahoo_data(ticker, end_date, days_needed=min_days_needed + 50)
            
            if not yahoo_data.empty:
                # Store the additional data in the database
                store_yahoo_data_in_db(yahoo_data, ticker)
                st.toast(f"✅ Successfully fetched and stored {len(yahoo_data)} additional data points", icon='🎉')
            else:
                st.toast("⚠️ Could not fetch additional data from Yahoo Finance", icon='⚠️')

def store_yahoo_data_in_db(df, ticker):
    """Store Yahoo Finance data in the database"""
    try:
        engine = get_database_connection()
        df['ticker'] = ticker
        df['time'] = None  # Set time to None for daily data
        
        # Select only the columns we need
        columns_to_insert = ['ticker', 'date', 'time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        df_insert = df[columns_to_insert].copy()
        
        # Insert data using raw SQL to handle conflicts
        insert_query = text("""
            INSERT INTO stock_data (ticker, date, time, open_price, high_price, low_price, close_price, volume)
            VALUES (:ticker, :date, :time, :open_price, :high_price, :low_price, :close_price, :volume)
            ON CONFLICT (ticker, date) DO NOTHING
        """)
        
        with engine.connect() as conn:
            for _, row in df_insert.iterrows():
                conn.execute(insert_query, {
                    'ticker': row['ticker'],
                    'date': row['date'],
                    'time': row['time'],
                    'open_price': row['open_price'],
                    'high_price': row['high_price'],
                    'low_price': row['low_price'],
                    'close_price': row['close_price'],
                    'volume': row['volume']
                })
            conn.commit()
            
        # After successfully storing data, populate indicators
        with st.spinner(f"Calculating technical indicators for {ticker}..."):
            populate_indicators_for_ticker(ticker)
            st.toast(f"✅ Successfully calculated indicators for {ticker}", icon='📊')
            
    except Exception as e:
        st.toast(f"❌ Error storing data in database: {e}", icon='❌')

def populate_indicators_for_ticker(ticker):
    """Populate technical indicators for a specific ticker after fetching new data"""
    try:
        # Import here to avoid circular imports
        import subprocess
        import sys
        
        # Run the populate_indicators script for this specific ticker
        result = subprocess.run([
            sys.executable, 
            "populate_indicators.py", 
            "--ticker", ticker
        ], capture_output=True, text=True, cwd="/Users/gangwu/git/stock_db")
        
        if result.returncode == 0:
            print(f"Successfully populated indicators for {ticker}")
        else:
            print(f"Error populating indicators for {ticker}: {result.stderr}")
            
    except Exception as e:
        print(f"Error running populate_indicators for {ticker}: {e}")

def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")
    
    st.title("📈 Stock Trend Prediction App")
    st.markdown("Analyze historical stock data and make trend predictions!")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Toggle for showing prediction markers on chart
    show_prediction_markers = st.sidebar.checkbox("Show Prediction Markers on Chart", value=False)
    
    # Get available tickers
    try:
        tickers = get_available_tickers()
        if not tickers:
            st.error("No tickers found in database!")
            return
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return
    
    # Ticker selection
    ticker_options = {clean_ticker_display(ticker): ticker for ticker in tickers}
    selected_display_ticker = st.sidebar.selectbox("Select Ticker", list(ticker_options.keys()))
    selected_ticker = ticker_options[selected_display_ticker]
    
    # Single date selection (base date for prediction)
    base_date = st.sidebar.date_input("Base Date", value=datetime.now() - timedelta(days=30))
    
    # Check if base date changed and reset current view date accordingly
    if st.session_state.last_base_date != base_date:
        st.session_state.current_view_date = base_date
        st.session_state.last_base_date = base_date
    
    # Load data
    try:
        # First ensure we have sufficient historical data for indicators
        ensure_sufficient_data_for_indicators(selected_ticker, st.session_state.current_view_date, min_days_needed=200)
        
        df = get_stock_data_from_date(selected_ticker, st.session_state.current_view_date, days_before=50)
        if df.empty:
            st.warning(f"No data found for {clean_ticker_display(selected_ticker)} around the selected date.")
            return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show future data and prediction markers based on toggle settings
        future_data = None
        predictions_to_show = None
        
        if st.session_state.predictions:
            ticker_predictions = [p for p in st.session_state.predictions if p['ticker'] == selected_ticker]
            if ticker_predictions:
                future_data = get_future_data(selected_ticker, base_date, days=30)
                # Only show prediction markers if toggle is enabled
                if show_prediction_markers:
                    predictions_to_show = st.session_state.predictions
        
        # Create and display chart
        fig = create_stock_chart(df, selected_ticker, predictions_to_show, future_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add "Show Next Day" button
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            if st.button("📅 Show Next Day"):
                next_day_data = get_next_day_data(selected_ticker, st.session_state.current_view_date)
                if not next_day_data.empty:
                    next_date = next_day_data.iloc[0]['date']
                    # Handle both datetime and date objects
                    if hasattr(next_date, 'date'):
                        st.session_state.current_view_date = next_date.date()
                    else:
                        st.session_state.current_view_date = next_date
                    st.rerun()
                else:
                    st.toast("⚠️ No more data available for future dates.", icon='⚠️')
        
        with col_btn2:
            if st.session_state.current_view_date > base_date:
                st.info(f"Currently viewing up to: {st.session_state.current_view_date}")
            else:
                st.info(f"Base date: {base_date}")
    
    with col2:
        st.subheader("📊 Current Stats")
        if not df.empty:
            latest = df.iloc[-1]
            
            # Show the symbol and date of the latest data point
            st.metric("Symbol", clean_ticker_display(selected_ticker))
            st.metric("Date", f"{latest['date']}")
            st.metric("Latest Close", f"${latest['close_price']:.2f}")
            
            if len(df) > 1:
                prev_close = df.iloc[-2]['close_price']
                change = latest['close_price'] - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
            
            if 'volume' in df.columns:
                st.metric("Volume", f"{latest['volume']:,}")
            
            if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']):
                st.metric("RSI (14)", f"{latest['rsi_14']:.2f}")
    
    # Prediction section
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.subheader("🔮 Make Your Predictions")
    
    with st.form("prediction_form"):
        st.markdown(f"**Predicting for: {clean_ticker_display(selected_ticker)}**")
        st.markdown(f"**Base Date: {st.session_state.current_view_date}**")
        
        if not df.empty:
            base_price = df.iloc[-1]['close_price']
            st.markdown(f"**Base Price: ${base_price:.2f}**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.write("**Day 1**")
            day1_pred = st.radio("", ["UP", "DOWN"], key="day1", label_visibility="collapsed")
        with col2:
            st.write("**Day 2**")
            day2_pred = st.radio("", ["UP", "DOWN"], key="day2", label_visibility="collapsed")
        with col3:
            st.write("**Day 3**")
            day3_pred = st.radio("", ["UP", "DOWN"], key="day3", label_visibility="collapsed")
        with col4:
            st.write("**Day 4**")
            day4_pred = st.radio("", ["UP", "DOWN"], key="day4", label_visibility="collapsed")
        with col5:
            st.write("**Day 5**")
            day5_pred = st.radio("", ["UP", "DOWN"], key="day5", label_visibility="collapsed")
        
        submitted = st.form_submit_button("Submit Prediction")
        
        if submitted and not df.empty:
            prediction = {
                'ticker': selected_ticker,
                'prediction_date': st.session_state.current_view_date,
                'base_price': base_price,
                'day_1_prediction': day1_pred,
                'day_2_prediction': day2_pred,
                'day_3_prediction': day3_pred,
                'day_4_prediction': day4_pred,
                'day_5_prediction': day5_pred,
                'timestamp': datetime.now()
            }
            
            st.session_state.predictions.append(prediction)
            st.toast("🎉 Prediction submitted successfully!", icon='✅')
            st.rerun()
    
    # Scoring section - show if there are predictions
    if st.session_state.predictions:
        st.subheader("🏆 Prediction Scores")
        
        # Filter predictions for current ticker
        ticker_predictions = [p for p in st.session_state.predictions if p['ticker'] == selected_ticker]
        
        if ticker_predictions:
            scores = calculate_prediction_score(st.session_state.predictions, selected_ticker)
            
            if scores:
                for score_data in scores:
                    st.write(f"**Prediction from {score_data['prediction_date']}**")
                    score_df_data = []
                    total_correct = 0
                    total_predictions = 0
                    
                    for day, result in score_data['scores'].items():
                        score_df_data.append({
                            'Day': day.replace('day_', '').replace('_', ' ').title(),
                            'Predicted': result['predicted'],
                            'Actual': result['actual'],
                            'Correct': '✅' if result['correct'] else '❌',
                            'Prev Price': f"${result['prev_price']:.2f}",
                            'Actual Price': f"${result['actual_price']:.2f}",
                            'Change %': f"{result['change_pct']:.2f}%"
                        })
                        
                        if result['correct']:
                            total_correct += 1
                        total_predictions += 1
                    
                    if score_df_data:
                        score_df = pd.DataFrame(score_df_data)
                        st.dataframe(score_df, use_container_width=True)
                        
                        accuracy = (total_correct / total_predictions) * 100
                        st.metric("Accuracy", f"{accuracy:.1f}%", f"{total_correct}/{total_predictions}")
                    
                    st.markdown("---")  # Add separator between multiple predictions
            else:
                st.info("Predictions are too recent to score. Check back later!")
        else:
            st.info("No predictions made for this ticker yet.")
    
    # Clear predictions button - show if there are predictions
    if st.session_state.predictions and st.sidebar.button("Clear All Predictions"):
        st.session_state.predictions = []
        st.rerun()

if __name__ == "__main__":
    main()
