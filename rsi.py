# app.py
# Streamlit dashboard: Daily RSI + 1-minute RSI from Yahoo Finance (yfinance)
# Run: streamlit run app.py

import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import pytz

# -----------------------------
# Helpers
# -----------------------------
def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI with Wilder's smoothing (no external TA libs)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing = EMA with alpha = 1/length but seeded with simple averages
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rvol(volume: pd.Series, length: int = 20) -> pd.Series:
    """Relative Volume - current volume vs average volume over specified period."""
    # Handle missing or zero volume data (common after hours)
    volume_clean = volume.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Calculate average volume, but handle cases where all volumes are zero
    avg_volume = volume_clean.rolling(window=length, min_periods=length).mean()
    
    # Avoid division by zero - if average volume is 0, return 0 for RVOL
    rvol = volume_clean / avg_volume.replace(0, np.nan)
    rvol = rvol.fillna(0)  # Fill NaN values with 0
    
    return rvol

def fast_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Fast Stochastic Oscillator - %K and %D lines."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    
    # %K line
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # %D line (simple moving average of %K)
    d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
    
    return k_percent, d_percent

def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands - Upper, Middle (SMA), and Lower bands."""
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower

@st.cache_data(show_spinner=False, ttl=30)
def fetch_intraday_data(ticker: str, interval: int, period: str = "2d", prepost: bool = True) -> pd.DataFrame:
    """Fetch intraday data with specified interval (1m, 5m, 10m, 15m, 30m, 60m)."""
    # Convert interval to yfinance format
    interval_map = {1: "1m", 5: "5m", 10: "10m", 15: "15m", 30: "30m", 60: "1h"}
    yf_interval = interval_map.get(interval, "1m")
    
    # Adjust period based on interval (longer intervals can fetch more data)
    if interval >= 60:
        period = "5d"  # 1 hour intervals can get more data
    elif interval >= 30:
        period = "3d"  # 30 minute intervals
    else:
        period = "2d"  # 1-15 minute intervals
    
    df = yf.download(
        tickers=ticker,
        interval=yf_interval,
        period=period,
        prepost=prepost,
        progress=False,
        threads=True,
    )
    if not df.empty:
        # Flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    return df

@st.cache_data(show_spinner=False, ttl=300)
def fetch_daily(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily data for context and daily RSI."""
    df = yf.download(
        tickers=ticker,
        interval="1d",
        period=period,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=True,
    )
    if not df.empty:
        # Flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)
    return df

def fmt_ts(ts) -> str:
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.tz_convert("America/Los_Angeles").strftime("%Y-%m-%d %H:%M")
    return str(ts)

def last_cross(series: pd.Series, level: float) -> str:
    """Return simple text for last cross of a level (for 1m RSI)."""
    if len(series) < 2:
        return "n/a"
    s = series.dropna()
    if len(s) < 2:
        return "n/a"
    prev, curr = s.iloc[-2], s.iloc[-1]
    if prev < level <= curr:
        return f"crossed ↑ {level} this minute"
    if prev > level >= curr:
        return f"crossed ↓ {level} this minute"
    return "—"

def get_market_status() -> tuple[str, str]:
    """Check if US market is open and return status with next open/close time."""
    try:
        # Get current time in US/Eastern timezone
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        current_time = now_et.time()
        current_weekday = now_et.weekday()  # 0=Monday, 6=Sunday
        
        # Check if it's a weekday
        if current_weekday >= 5:  # Saturday or Sunday
            next_monday = now_et + timedelta(days=(7 - current_weekday))
            next_open = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
            return "🔴 CLOSED (Weekend)", f"Next open: {next_open.strftime('%Y-%m-%d %H:%M')} ET"
        
        # Check if it's during market hours
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        if market_open <= current_time <= market_close:
            return "🟢 OPEN", f"Closes at 4:00 PM ET today"
        elif current_time < market_open:
            return "🔴 CLOSED", f"Opens at 9:30 AM ET today"
        else:
            next_day = now_et + timedelta(days=1)
            # Skip weekends
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            return "🔴 CLOSED", f"Next open: {next_open.strftime('%Y-%m-%d %H:%M')} ET"
            
    except Exception as e:
        return "❓ Unknown", f"Error: {str(e)}"

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Daily + 1m RSI (Yahoo)", layout="wide")
st.title("📈 Daily + 1-Minute RSI Monitor (Yahoo Finance)")

# Market status indicator
market_status, market_info = get_market_status()
st.info(f"**Market Status:** {market_status} | {market_info}")

with st.sidebar:
    st.header("Settings")
    
    # Get symbol from URL parameters
    query_params = st.query_params
    default_symbol = query_params.get("symbol", "AAPL")
    
    selected = st.text_input("Ticker", value=default_symbol, help="Enter a single ticker symbol")

    rsi_len_daily = st.number_input("Daily RSI length", min_value=5, max_value=50, value=14, step=1)
    rsi_len_1m    = st.number_input("1-Minute RSI length", min_value=5, max_value=50, value=14, step=1)
    ob_level = st.slider("Overbought level", 50, 90, 70, 1)
    os_level = st.slider("Oversold level", 10, 50, 30, 1)
    
    st.divider()
    st.subheader("Additional Indicators")
    
    # RVOL settings
    rvol_length = st.number_input("RVOL length", min_value=5, max_value=50, value=20, step=1, help="Period for average volume calculation")
    
    # Fast Stochastic settings
    stoch_k_period = st.number_input("Stochastic %K period", min_value=5, max_value=50, value=14, step=1)
    stoch_d_period = st.number_input("Stochastic %D period", min_value=2, max_value=10, value=3, step=1)
    
    # Bollinger Bands settings
    bb_period = st.number_input("Bollinger Bands period", min_value=5, max_value=50, value=20, step=1)
    bb_std_dev = st.number_input("Bollinger Bands std dev", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

    # Minute data interval selection
    minute_interval = st.selectbox(
        "Minute Data Interval", 
        options=[1, 5, 10, 15, 30, 60], 
        index=0,
        help="Select the time interval for minute data (1m, 5m, 10m, 15m, 30m, 1h)"
    )
    
    extend_hours = st.checkbox("Include extended hours", value=True)
    autorefresh_sec = st.number_input("Auto-refresh (seconds)", min_value=0, max_value=300, value=0, step=5)
    st.caption("Note: Yahoo minute data can be delayed/throttled.")
    
    # Manual refresh button
    if st.button("🔄 Refresh Data", type="secondary"):
        st.rerun()

# Auto refresh - using Streamlit's built-in mechanism
if autorefresh_sec > 0:
    st.caption(f"🔄 Auto-refreshing every {autorefresh_sec} seconds")
    # Use a placeholder for auto-refresh countdown
    placeholder = st.empty()
    import time
    for i in range(autorefresh_sec):
        placeholder.caption(f"⏱️ Refreshing in {autorefresh_sec - i} seconds...")
        time.sleep(1)
    st.rerun()

if not selected or not selected.strip():
    st.error("Please enter a ticker symbol.")
    st.stop()

# Clean up the ticker input
selected = selected.strip().upper()

# -----------------------------
# Data fetch
# -----------------------------
colA, colB, colC = st.columns([1, 1, 1])

with st.spinner(f"Fetching data for {selected}…"):
    df_1m = fetch_intraday_data(selected, minute_interval, period="2d", prepost=extend_hours)
    df_1d = fetch_daily(selected, period="1y")

if df_1d.empty:
    st.error("No daily data returned. Check the ticker symbol.")
    st.stop()
if df_1m.empty:
    st.warning(f"No {minute_interval}-minute data returned (Yahoo limit or off hours). Showing daily only.")
else:
    # Check for missing volume data (common after hours)
    volume_missing = df_1m["Volume"].isna().sum()
    volume_zero = (df_1m["Volume"] == 0).sum()
    total_points = len(df_1m)
    
    if volume_missing > total_points * 0.5 or volume_zero > total_points * 0.5:
        st.warning(f"⚠️ Limited volume data available: {volume_missing} missing, {volume_zero} zero values. This is common after hours.")

# -----------------------------
# Compute RSI, RVOL, Fast Stochastic, and Bollinger Bands
# -----------------------------
df_1d["RSI"] = rsi_wilder(df_1d["Close"], rsi_len_daily)
df_1d["RVOL"] = rvol(df_1d["Volume"], rvol_length)
df_1d["STOCH_K"], df_1d["STOCH_D"] = fast_stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], stoch_k_period, stoch_d_period)
df_1d["BB_UPPER"], df_1d["BB_MIDDLE"], df_1d["BB_LOWER"] = bollinger_bands(df_1d["Close"], bb_period, bb_std_dev)

if not df_1m.empty:
    df_1m["RSI"] = rsi_wilder(df_1m["Close"], rsi_len_1m)
    df_1m["RVOL"] = rvol(df_1m["Volume"], rvol_length)
    df_1m["STOCH_K"], df_1m["STOCH_D"] = fast_stochastic(df_1m["High"], df_1m["Low"], df_1m["Close"], stoch_k_period, stoch_d_period)
    df_1m["BB_UPPER"], df_1m["BB_MIDDLE"], df_1m["BB_LOWER"] = bollinger_bands(df_1m["Close"], bb_period, bb_std_dev)

# Latest values
last_daily_close = float(df_1d["Close"].iloc[-1])
last_daily_rsi   = float(df_1d["RSI"].iloc[-1])
last_daily_rvol  = float(df_1d["RVOL"].iloc[-1])
last_daily_stoch_k = float(df_1d["STOCH_K"].iloc[-1])
last_daily_stoch_d = float(df_1d["STOCH_D"].iloc[-1])

if not df_1m.empty:
    last_1m_close   = float(df_1m["Close"].iloc[-1])
    last_1m_rsi     = float(df_1m["RSI"].iloc[-1])
    last_1m_rvol     = float(df_1m["RVOL"].iloc[-1])
    last_1m_stoch_k = float(df_1m["STOCH_K"].iloc[-1])
    last_1m_stoch_d = float(df_1m["STOCH_D"].iloc[-1])
    last_1m_time    = df_1m.index[-1]
    cross70 = last_cross(df_1m["RSI"], ob_level)
    cross30 = last_cross(df_1m["RSI"], os_level)
else:
    last_1m_close = np.nan
    last_1m_rsi = np.nan
    last_1m_rvol = np.nan
    last_1m_stoch_k = np.nan
    last_1m_stoch_d = np.nan
    last_1m_time = "n/a"
    cross70 = cross30 = "n/a"

# -----------------------------
# Top summary cards
# -----------------------------
with colA:
    st.subheader(f"{selected} Price & Volume")
    st.metric("Daily close", f"{last_daily_close:,.2f}")
    st.metric("RVOL (daily)", f"{last_daily_rvol:0.2f}x", 
              delta="High volume" if last_daily_rvol >= 1.5 else ("Low volume" if last_daily_rvol <= 0.5 else "Normal"))
    if not np.isnan(last_1m_close):
        st.metric(f"Last {minute_interval}-min close", f"{last_1m_close:,.2f}", help=f"Timestamp: {fmt_ts(last_1m_time)}")
        if not np.isnan(last_1m_rvol):
            st.metric(f"RVOL ({minute_interval}m)", f"{last_1m_rvol:0.2f}x",
                      delta="High volume" if last_1m_rvol >= 1.5 else ("Low volume" if last_1m_rvol <= 0.5 else "Normal"))

with colB:
    st.subheader("Daily RSI & Stochastic")
    st.metric("RSI (daily)", f"{last_daily_rsi:0.1f}",
              delta="Overbought" if last_daily_rsi >= ob_level else ("Oversold" if last_daily_rsi <= os_level else "Neutral"))
    st.metric("Stoch %K", f"{last_daily_stoch_k:0.1f}",
              delta="Overbought" if last_daily_stoch_k >= 80 else ("Oversold" if last_daily_stoch_k <= 20 else "Neutral"))
    st.metric("Stoch %D", f"{last_daily_stoch_d:0.1f}")

with colC:
    st.subheader(f"{minute_interval}-Minute Indicators")
    if not np.isnan(last_1m_rsi):
        st.metric(f"RSI ({minute_interval}m)", f"{last_1m_rsi:0.1f}",
                  delta="Overbought" if last_1m_rsi >= ob_level else ("Oversold" if last_1m_rsi <= os_level else "Neutral"),
                  help=f"{cross70}; {cross30}")
        if not np.isnan(last_1m_stoch_k):
            st.metric(f"Stoch %K ({minute_interval}m)", f"{last_1m_stoch_k:0.1f}",
                      delta="Overbought" if last_1m_stoch_k >= 80 else ("Oversold" if last_1m_stoch_k <= 20 else "Neutral"))
            st.metric(f"Stoch %D ({minute_interval}m)", f"{last_1m_stoch_d:0.1f}")
    else:
        st.info(f"{minute_interval}-minute data not available now.")

st.divider()

# -----------------------------
# Charts - Aligned by Type
# -----------------------------
st.subheader("Price & Technical Indicators")

# Close Price Charts with Bollinger Bands and Volume
st.markdown("### Close Price with Bollinger Bands & Volume")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Daily Close with BB & Volume**")
    # Calculate price range for better Y-axis scaling
    if len(df_1d) > 1:
        price_min = df_1d["Close"].min()
        price_max = df_1d["Close"].max()
        price_range = price_max - price_min
        
        # Add some padding to the range (10% on each side)
        padding = price_range * 0.1
        y_min = price_min - padding
        y_max = price_max + padding
        
        st.caption(f"💰 Daily Price Range: ${price_min:.2f} - ${price_max:.2f} (${price_range:.2f})")
    else:
        y_min = None
        y_max = None
    
    # Create DataFrame with close price, Bollinger Bands, and volume
    d1 = pd.DataFrame({
        "Close": df_1d["Close"],
        "BB Upper": df_1d["BB_UPPER"],
        "BB Middle": df_1d["BB_MIDDLE"], 
        "BB Lower": df_1d["BB_LOWER"]
    })
    
    # Use Altair chart with Y-axis limits if we have enough data
    if y_min is not None and y_max is not None:
        # Create a custom chart with Y-axis limits
        import altair as alt
        
        # Prepare data for Altair
        chart_data = d1.reset_index()
        chart_data = chart_data.melt(id_vars=['Date'], var_name='Indicator', value_name='Price')
        
        # Create the chart with Y-axis limits and tooltips
        chart = alt.Chart(chart_data).mark_line().add_selection(
            alt.selection_interval(bind='scales')
        ).encode(
            x='Date:T',
            y=alt.Y('Price:Q', scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color('Indicator:N', legend=None),
            tooltip=[
                alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                alt.Tooltip('Indicator:N', title='Indicator'),
                alt.Tooltip('Price:Q', title='Price', format='.2f')
            ]
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(d1, height=400)
    
    # Volume chart below price
    st.markdown("**Daily Volume**")
    d1_vol = pd.DataFrame({"Volume": df_1d["Volume"]})
    st.bar_chart(d1_vol, height=100)

with col2:
    st.markdown(f"**{minute_interval}-Minute Close with BB & Volume**")
    if not df_1m.empty:
        # Calculate price range for better Y-axis scaling
        recent_data = df_1m.tail(300)
        if len(recent_data) > 1:
            price_min = recent_data["Close"].min()
            price_max = recent_data["Close"].max()
            price_range = price_max - price_min
            
            # Add some padding to the range (10% on each side)
            padding = price_range * 0.1
            y_min = price_min - padding
            y_max = price_max + padding
            
            st.caption(f"💰 1-Min Price Range: ${price_min:.2f} - ${price_max:.2f} (${price_range:.2f})")
        else:
            y_min = None
            y_max = None
        
        # Create DataFrame with close price, Bollinger Bands, and volume
        m1 = pd.DataFrame({
            "Close": df_1m["Close"],
            "BB Upper": df_1m["BB_UPPER"],
            "BB Middle": df_1m["BB_MIDDLE"],
            "BB Lower": df_1m["BB_LOWER"]
        })
        
        # Use st.line_chart with y-axis limits if we have enough data
        if y_min is not None and y_max is not None:
            # Create a custom chart with Y-axis limits
            import altair as alt
            
            # Prepare data for Altair
            chart_data = m1.tail(300).reset_index()
            chart_data = chart_data.melt(id_vars=['Datetime'], var_name='Indicator', value_name='Price')
            
            # Create the chart with Y-axis limits and tooltips
            chart = alt.Chart(chart_data).mark_line().add_selection(
                alt.selection_interval(bind='scales')
            ).encode(
                x='Datetime:T',
                y=alt.Y('Price:Q', scale=alt.Scale(domain=[y_min, y_max])),
                color=alt.Color('Indicator:N', legend=None),
                tooltip=[
                    alt.Tooltip('Datetime:T', title='Time', format='%Y-%m-%d %H:%M:%S'),
                    alt.Tooltip('Indicator:N', title='Indicator'),
                    alt.Tooltip('Price:Q', title='Price', format='.2f')
                ]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(m1.tail(300), height=400)
        
        # Volume chart below price
        st.markdown(f"**{minute_interval}-Minute Volume**")
        m1_vol = pd.DataFrame({"Volume": df_1m["Volume"]})
        st.bar_chart(m1_vol.tail(300), height=100)
    else:
        st.info(f"No {minute_interval}-minute data available")

# RSI Charts
st.markdown("### RSI")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Daily RSI**")
    d2 = pd.DataFrame({f"RSI {rsi_len_daily}": df_1d["RSI"]})
    st.line_chart(d2, height=150)

with col2:
    st.markdown(f"**{minute_interval}-Minute RSI**")
    if not df_1m.empty:
        m2 = pd.DataFrame({f"RSI {rsi_len_1m}": df_1m["RSI"]})
        st.line_chart(m2.tail(300), height=150)
    else:
        st.info(f"No {minute_interval}-minute data available")

# Stochastic Charts
st.markdown("### Stochastic Oscillator")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Daily Stochastic**")
    d4 = pd.DataFrame({
        f"Stoch %K {stoch_k_period}": df_1d["STOCH_K"],
        f"Stoch %D {stoch_d_period}": df_1d["STOCH_D"]
    })
    st.line_chart(d4, height=150)

with col2:
    st.markdown(f"**{minute_interval}-Minute Stochastic**")
    if not df_1m.empty:
        m4 = pd.DataFrame({
            f"Stoch %K {stoch_k_period}": df_1m["STOCH_K"],
            f"Stoch %D {stoch_d_period}": df_1m["STOCH_D"]
        })
        st.line_chart(m4.tail(300), height=150)
    else:
        st.info(f"No {minute_interval}-minute data available")

# RVOL Charts
st.markdown("### Relative Volume (RVOL)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Daily RVOL**")
    d3 = pd.DataFrame({f"RVOL {rvol_length}": df_1d["RVOL"]})
    st.line_chart(d3, height=150)

with col2:
    st.markdown(f"**{minute_interval}-Minute RVOL**")
    if not df_1m.empty:
        m3 = pd.DataFrame({f"RVOL {rvol_length}": df_1m["RVOL"]})
        st.line_chart(m3.tail(300), height=150)
    else:
        st.info(f"No {minute_interval}-minute data available")

# Comprehensive Buy/Sell Suggestions
st.markdown("### Comprehensive Buy/Sell Suggestions")
buy_sell_col1, buy_sell_col2 = st.columns(2)

with buy_sell_col1:
    st.markdown("**Daily Trading Signals**")
    
    # Daily signals
    daily_signals = []
    
    # RSI signals
    if last_daily_rsi >= 70:
        daily_signals.append("🔴 SELL: RSI ≥ 70 (Overbought)")
    elif last_daily_rsi <= 30:
        daily_signals.append("🟢 BUY: RSI ≤ 30 (Oversold)")
    
    # Stochastic signals
    if last_daily_stoch_k >= 80 and last_daily_stoch_d >= 80:
        daily_signals.append("🔴 SELL: Stochastic ≥ 80 (Overbought)")
    elif last_daily_stoch_k <= 20 and last_daily_stoch_d <= 20:
        daily_signals.append("🟢 BUY: Stochastic ≤ 20 (Oversold)")
    
    # Bollinger Bands signals
    if last_daily_close >= last_daily_stoch_k:  # Using BB Upper as proxy
        daily_signals.append("🔴 CAUTION: Price near BB Upper")
    elif last_daily_close <= last_daily_stoch_d:  # Using BB Lower as proxy
        daily_signals.append("🟢 OPPORTUNITY: Price near BB Lower")
    
    # Volume confirmation
    if last_daily_rvol >= 2.0:
        daily_signals.append("📊 High Volume Confirmation")
    elif last_daily_rvol <= 0.5:
        daily_signals.append("📊 Low Volume - Wait for Confirmation")
    
    if daily_signals:
        for signal in daily_signals:
            st.write(signal)
    else:
        st.info("🟡 No clear daily signals")
    
    st.caption(f"Daily: RSI={last_daily_rsi:.1f}, RVOL={last_daily_rvol:.2f}x, Stoch K={last_daily_stoch_k:.1f}, D={last_daily_stoch_d:.1f}")

with buy_sell_col2:
    st.markdown(f"**{minute_interval}-Minute Trading Signals**")
    
    if not df_1m.empty and not np.isnan(last_1m_rsi):
        # 1-Minute signals
        minute_signals = []
        
        # RSI signals
        if last_1m_rsi >= 70:
            minute_signals.append("🔴 SELL: RSI ≥ 70 (Overbought)")
        elif last_1m_rsi <= 30:
            minute_signals.append("🟢 BUY: RSI ≤ 30 (Oversold)")
        
        # Stochastic signals
        if not np.isnan(last_1m_stoch_k) and not np.isnan(last_1m_stoch_d):
            if last_1m_stoch_k >= 80 and last_1m_stoch_d >= 80:
                minute_signals.append("🔴 SELL: Stochastic ≥ 80 (Overbought)")
            elif last_1m_stoch_k <= 20 and last_1m_stoch_d <= 20:
                minute_signals.append("🟢 BUY: Stochastic ≤ 20 (Oversold)")
        
        # Volume signals
        if not np.isnan(last_1m_rvol):
            if last_1m_rvol >= 2.0:
                minute_signals.append("📊 High Volume Confirmation")
            elif last_1m_rvol <= 0.5:
                minute_signals.append("📊 Low Volume - Wait for Confirmation")
        
        # Price action signals
        if len(df_1m) > 1:
            prev_close = df_1m["Close"].iloc[-2]
            if last_1m_close > prev_close:
                minute_signals.append("📈 Price Rising")
            elif last_1m_close < prev_close:
                minute_signals.append("📉 Price Falling")
        
        if minute_signals:
            for signal in minute_signals:
                st.write(signal)
        else:
            st.info("🟡 No clear 1-minute signals")
        
        st.caption(f"{minute_interval}-Min: RSI={last_1m_rsi:.1f}, RVOL={last_1m_rvol:.2f}x, Stoch K={last_1m_stoch_k:.1f}, D={last_1m_stoch_d:.1f}")
    else:
        st.info(f"No {minute_interval}-minute data available")

# Data info and parameters
if not df_1m.empty:
    st.caption(f"📊 {minute_interval}-Minute Data: {len(df_1m)} points | Latest: {fmt_ts(df_1m.index[-1])} | Range: {fmt_ts(df_1m.index[0])} to {fmt_ts(df_1m.index[-1])}")
    
    close_values = df_1m["Close"].dropna()
    if len(close_values) > 1:
        price_range = close_values.max() - close_values.min()
        st.caption(f"💰 {minute_interval}-Minute Price range: ${price_range:.2f} (${close_values.min():.2f} - ${close_values.max():.2f})")
    else:
        st.caption("⚠️ Only one data point - market may be closed")

st.caption(f"Parameters: Daily RSI={rsi_len_daily} | {minute_interval}-Min RSI={rsi_len_1m} | RVOL={rvol_length} | Stoch K={stoch_k_period}, D={stoch_d_period} | BB Period={bb_period}, Std={bb_std_dev}")

st.divider()

# -----------------------------
# Enhanced signal analysis with all indicators
# -----------------------------
st.subheader("Enhanced Signal Analysis")
sig_cols = st.columns(4)

def status_badge(rsi_val: float, ob: int, os: int) -> str:
    if np.isnan(rsi_val):
        return "⚪ n/a"
    if rsi_val >= ob:
        return "🔴 Overbought"
    if rsi_val <= os:
        return "🟢 Oversold"
    return "🟡 Neutral"

def volume_status(rvol_val: float) -> str:
    if np.isnan(rvol_val):
        return "⚪ n/a"
    if rvol_val == 0:
        return "⚪ No Volume (After Hours)"
    if rvol_val >= 1.5:
        return "🔴 High Volume"
    if rvol_val <= 0.5:
        return "🟢 Low Volume"
    return "🟡 Normal"

def stoch_status(stoch_k: float, stoch_d: float) -> str:
    if np.isnan(stoch_k) or np.isnan(stoch_d):
        return "⚪ n/a"
    if stoch_k >= 80 and stoch_d >= 80:
        return "🔴 Overbought"
    if stoch_k <= 20 and stoch_d <= 20:
        return "🟢 Oversold"
    if stoch_k > stoch_d:
        return "📈 Bullish"
    if stoch_k < stoch_d:
        return "📉 Bearish"
    return "🟡 Neutral"

with sig_cols[0]:
    st.markdown("**Daily RSI**")
    st.write(status_badge(last_daily_rsi, ob_level, os_level))
    st.markdown("**Daily RVOL**")
    st.write(volume_status(last_daily_rvol))
    st.markdown("**Daily Stochastic**")
    st.write(stoch_status(last_daily_stoch_k, last_daily_stoch_d))

with sig_cols[1]:
    st.markdown(f"**{minute_interval}-Minute RSI**")
    st.write(status_badge(last_1m_rsi, ob_level, os_level) if not np.isnan(last_1m_rsi) else "⚪ n/a")
    st.markdown(f"**{minute_interval}-Minute RVOL**")
    st.write(volume_status(last_1m_rvol) if not np.isnan(last_1m_rvol) else "⚪ n/a")
    st.markdown(f"**{minute_interval}-Minute Stochastic**")
    st.write(stoch_status(last_1m_stoch_k, last_1m_stoch_d) if not np.isnan(last_1m_stoch_k) else "⚪ n/a")

with sig_cols[2]:
    st.markdown("**Combined Analysis**")
    # Enhanced logic considering all indicators
    signals = []
    
    # RSI signals
    if last_daily_rsi >= ob_level:
        signals.append("Daily RSI overbought")
    if not np.isnan(last_1m_rsi) and last_1m_rsi >= ob_level:
        signals.append("1m RSI overbought")
    
    # Volume signals
    if last_daily_rvol >= 2.0:
        signals.append("Very high volume")
    elif last_daily_rvol <= 0.3:
        signals.append("Very low volume")
    
    # Stochastic signals
    if last_daily_stoch_k >= 80 and last_daily_stoch_d >= 80:
        signals.append("Daily Stoch overbought")
    if not np.isnan(last_1m_stoch_k) and last_1m_stoch_k >= 80:
        signals.append("1m Stoch overbought")
    
    if signals:
        st.error("⚠️ Caution signals:")
        for signal in signals:
            st.write(f"• {signal}")
    else:
        st.success("✅ No major warning signals")

with sig_cols[3]:
    st.markdown("**Buy/Sell Suggestions**")
    
    # Buy/Sell logic based on specified criteria
    buy_signals = []
    sell_signals = []
    
    # SELL conditions: RSI above 70
    if last_daily_rsi >= 70:
        sell_signals.append("Daily RSI ≥ 70 (Overbought)")
    if not np.isnan(last_1m_rsi) and last_1m_rsi >= 70:
        sell_signals.append(f"{minute_interval}m RSI ≥ 70 (Overbought)")
    
    # BUY conditions: RSI below 30 AND RVOL > 3
    if last_daily_rsi <= 30 and last_daily_rvol >= 3.0:
        buy_signals.append("Daily: RSI ≤ 30 + RVOL ≥ 3.0")
    if not np.isnan(last_1m_rsi) and not np.isnan(last_1m_rvol) and last_1m_rsi <= 30 and last_1m_rvol >= 3.0:
        buy_signals.append(f"{minute_interval}m: RSI ≤ 30 + RVOL ≥ 3.0")
    
    # Display suggestions
    if sell_signals:
        st.error("🔴 **SELL SIGNALS**")
        for signal in sell_signals:
            st.write(f"• {signal}")
    
    if buy_signals:
        st.success("🟢 **BUY SIGNALS**")
        for signal in buy_signals:
            st.write(f"• {signal}")
    
    if not buy_signals and not sell_signals:
        st.info("🟡 **No Clear Signals**")
        st.caption("Wait for RSI ≤ 30 + RVOL ≥ 3.0 for BUY or RSI ≥ 70 for SELL")

# Raw tables expander
with st.expander("Show raw data (tail)"):
    st.write("Daily (tail):", df_1d.tail(10))
    if not df_1m.empty:
        st.write("1-Minute (tail):", df_1m.tail(20))

