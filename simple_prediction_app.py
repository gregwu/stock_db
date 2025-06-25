"""
Simple Stock Prediction App - Lightweight version with basic plotting
Run this if you don't want to install Plotly/Streamlit
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'gangwu'),
    'user': os.getenv('DB_USER', 'gangwu'),
    'password': os.getenv('DB_PASSWORD', 'gangwu')
}

def get_database_connection():
    """Create database connection"""
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def get_available_tickers():
    """Get list of available tickers from database"""
    engine = get_database_connection()
    query = text("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker LIMIT 20")
    
    with engine.connect() as conn:
        result = conn.execute(query)
        tickers = [row[0] for row in result.fetchall()]
    
    return tickers

def get_stock_data(ticker, start_date, end_date):
    """Get stock data for specified ticker and date range"""
    engine = get_database_connection()
    
    query = text("""
    SELECT date, open_price, high_price, low_price, close_price, volume,
           sma_20, sma_144, rsi_14, macd, macd_signal, bb_upper, bb_lower
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
    
    return df

def plot_stock_chart(df, ticker):
    """Create stock chart with matplotlib"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{ticker} Stock Analysis', fontsize=16)
    
    # Price chart
    ax1.plot(df['date'], df['close_price'], label='Close Price', linewidth=2)
    if 'sma_20' in df.columns:
        ax1.plot(df['date'], df['sma_20'], label='SMA 20', alpha=0.7)
    if 'sma_144' in df.columns:
        ax1.plot(df['date'], df['sma_144'], label='SMA 144', alpha=0.7)
    
    ax1.set_title('Price & Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    ax2.bar(df['date'], df['volume'], alpha=0.7, color='lightblue')
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # RSI
    if 'rsi_14' in df.columns and df['rsi_14'].notna().any():
        ax3.plot(df['date'], df['rsi_14'], color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7)
        ax3.set_title('RSI (14)')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    # MACD
    if 'macd' in df.columns and df['macd'].notna().any():
        ax4.plot(df['date'], df['macd'], label='MACD', color='blue')
        if 'macd_signal' in df.columns:
            ax4.plot(df['date'], df['macd_signal'], label='Signal', color='red')
        ax4.set_title('MACD')
        ax4.set_ylabel('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Format x-axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def make_prediction():
    """Interactive prediction function"""
    print("\n📈 Stock Trend Prediction Tool")
    print("=" * 50)
    
    # Get available tickers
    tickers = get_available_tickers()
    
    print(f"\n📊 Available tickers (showing first 20):")
    for i, ticker in enumerate(tickers[:20], 1):
        print(f"{i:2d}. {ticker}")
    
    # Ticker selection
    try:
        choice = int(input(f"\nSelect ticker (1-{len(tickers[:20])}): ")) - 1
        selected_ticker = tickers[choice]
    except (ValueError, IndexError):
        print("Invalid selection!")
        return
    
    # Date range
    print(f"\n📅 Date Range Selection for {selected_ticker}")
    try:
        days_back = int(input("How many days of history to show? (default 90): ") or "90")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
    except ValueError:
        print("Invalid number of days!")
        return
    
    # Get data
    print(f"\n📡 Fetching data for {selected_ticker}...")
    df = get_stock_data(selected_ticker, start_date, end_date)
    
    if df.empty:
        print(f"❌ No data found for {selected_ticker}")
        return
    
    print(f"✅ Found {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    # Show current stats
    latest = df.iloc[-1]
    print(f"\n📊 Current Stats for {selected_ticker}:")
    print(f"   Latest Close: ${latest['close_price']:.2f}")
    print(f"   Latest Date: {latest['date']}")
    
    if len(df) > 1:
        prev_close = df.iloc[-2]['close_price']
        change = latest['close_price'] - prev_close
        change_pct = (change / prev_close) * 100
        print(f"   Daily Change: ${change:.2f} ({change_pct:+.2f}%)")
    
    # Create and show chart
    print(f"\n📈 Generating chart for {selected_ticker}...")
    fig = plot_stock_chart(df, selected_ticker)
    plt.show()
    
    # Make predictions
    print(f"\n🔮 Make Your Predictions for {selected_ticker}")
    print(f"Base Price: ${latest['close_price']:.2f}")
    print(f"Prediction Date: {end_date}")
    
    predictions = {}
    for day in [1, 3, 5]:
        while True:
            pred = input(f"Day {day} prediction (UP/DOWN): ").upper().strip()
            if pred in ['UP', 'DOWN']:
                predictions[f'day_{day}'] = pred
                break
            print("Please enter 'UP' or 'DOWN'")
    
    print(f"\n✅ Predictions submitted for {selected_ticker}:")
    for day, pred in predictions.items():
        print(f"   {day.replace('_', ' ').title()}: {pred}")
    
    # Save predictions to file
    prediction_data = {
        'ticker': selected_ticker,
        'prediction_date': end_date.isoformat(),
        'base_price': float(latest['close_price']),
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON file
    try:
        import json
        predictions_file = 'predictions.json'
        
        # Load existing predictions
        try:
            with open(predictions_file, 'r') as f:
                all_predictions = json.load(f)
        except FileNotFoundError:
            all_predictions = []
        
        # Add new prediction
        all_predictions.append(prediction_data)
        
        # Save back to file
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"💾 Predictions saved to {predictions_file}")
        
    except Exception as e:
        print(f"⚠️ Could not save predictions: {e}")
    
    print("\n🎉 Prediction session complete!")

def main():
    """Main function"""
    print("🚀 Stock Analysis Tool")
    print("Choose an option:")
    print("1. Make predictions")
    print("2. View chart only")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        make_prediction()
    elif choice == "2":
        # Simple chart viewing
        tickers = get_available_tickers()
        print(f"\nAvailable tickers: {tickers[:10]}...")
        ticker = input("Enter ticker symbol: ").strip().upper()
        
        if not ticker.endswith('.US'):
            ticker += '.US'
        
        days = int(input("Days of history (default 90): ") or "90")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        df = get_stock_data(ticker, start_date, end_date)
        if not df.empty:
            fig = plot_stock_chart(df, ticker)
            plt.show()
        else:
            print(f"No data found for {ticker}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
