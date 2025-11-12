
# app_no_tops_bottoms.py
# Streamlit app to help avoid buying tops and selling bottoms
# Indicators: Bollinger Bands, RSI(14), MACD(12,26,9), VWAP (intraday), EMA(200)
# Rules:
#   Buy when: RSI crosses up through rsi_buy (default 50) AND close >= middle BB
#             AND price is not stretched (below upper band) AND no "3 green bars" chase
#   Sell when: RSI crosses down through rsi_sell (default 50) OR close < middle BB
#             AND avoid panic (don't sell when below lower band or after 3 red bars)
#
# Requires: streamlit, yfinance, pandas, numpy, matplotlib

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from math import isnan
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="No Tops / No Bottoms", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # yfinance supports: period ∈ {"1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}
    # interval ∈ {"1m","2m", "5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, threads=True, progress=False, prepost=True)
    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns if present (happens with single ticker sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)  # Ensure ['Open','High','Low','Close','Adj Close','Volume']
    df.dropna(inplace=True)

    # Convert timezone from UTC to US Eastern Time (New York)
    # This assumes yfinance returns data in UTC
    if hasattr(df.index, 'tz'):
        if df.index.tz is not None:
            # Convert to US Eastern Time
            df.index = df.index.tz_convert('America/New_York')
        else:
            # If no timezone info, assume UTC and localize then convert
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def bollinger_bands(close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    return mid, upper, lower

def ema(series: pd.Series, length: int = 200):
    return series.ewm(span=length, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    # Valid for intraday intervals (sum over the session keyed by date)
    if "Volume" not in df or df.empty:
        return pd.Series(index=df.index, dtype=float)
    date_index = df.index.tz_localize(None).date if hasattr(df.index, "tz_localize") else df.index.date
    session = pd.Series(date_index, index=df.index)
    pv = (df["Close"] * df["Volume"]).groupby(session).cumsum()
    vv = df["Volume"].groupby(session).cumsum().replace(0, np.nan)
    return pv / vv

def last_cross(series: pd.Series, level: float, direction: str = "above"):
    """Returns True if the last bar crossed the level in the given direction."""
    if len(series) < 2:
        return False
    prev, curr = series.iloc[-2], series.iloc[-1]
    if np.isnan(prev) or np.isnan(curr):
        return False
    if direction == "above":
        return prev < level <= curr
    else:
        return prev > level >= curr

def consecutive_bars(close: pd.Series, up: bool = True, n: int = 3) -> bool:
    if len(close) < n + 1:
        return False
    diffs = close.diff().iloc[-n:]
    return bool((diffs > 0).all()) if up else bool((diffs < 0).all())

def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if df.empty:
        return df

    # Indicators
    df["RSI"] = rsi(df["Close"], params["rsi_len"])
    macd_line, macd_sig, macd_hist = macd(df["Close"], params["macd_fast"], params["macd_slow"], params["macd_sig"])
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd_line, macd_sig, macd_hist
    df["BB_mid"], df["BB_up"], df["BB_lo"] = bollinger_bands(df["Close"], params["bb_len"], params["bb_mult"])

    # Trend guides
    if params["use_vwap"] and params["interval"].endswith("m"):
        df["VWAP"] = vwap(df)
    else:
        df["VWAP"] = np.nan
    df["EMA200"] = ema(df["Close"], 200)

    # Rules per bar
    df["BuyRaw"] = False
    df["SellRaw"] = False

    rsi_buy = params["rsi_buy"]
    rsi_sell = params["rsi_sell"]

    # Buy when RSI crosses above rsi_buy and price >= BB mid
    cross_up = (df["RSI"].shift(1) < rsi_buy) & (df["RSI"] >= rsi_buy)
    bb_ok = pd.Series(df["Close"].values >= df["BB_mid"].values, index=df.index)
    stretched = pd.Series(df["Close"].values >= df["BB_up"].values, index=df.index)
    no_chase = ~(
        (stretched) |
        (df["Close"].diff().rolling(3).apply(lambda x: (x > 0).all(), raw=True) > 0)
    )
    trend_ok = pd.Series(True, index=df.index)
    if params["use_vwap"] and df["VWAP"].notna().any():
        trend_ok = pd.Series(df["Close"].values >= df["VWAP"].values, index=df.index)
    elif params["use_ema200"]:
        trend_ok = pd.Series(df["Close"].values >= df["EMA200"].values, index=df.index)
    df["BuyRaw"] = cross_up & bb_ok & no_chase & trend_ok

    # Sell when RSI crosses below rsi_sell or price < BB mid
    cross_down = (df["RSI"].shift(1) > rsi_sell) & (df["RSI"] <= rsi_sell)
    bb_break = pd.Series(df["Close"].values < df["BB_mid"].values, index=df.index)
    panic_zone = pd.Series(df["Close"].values <= df["BB_lo"].values, index=df.index) | (df["Close"].diff().rolling(3).apply(lambda x: (x < 0).all(), raw=True) > 0)
    no_panic = ~panic_zone
    df["SellRaw"] = (cross_down | bb_break) & no_panic

    # Compact labels
    df["Signal"] = np.where(df["BuyRaw"], "BUY",
                      np.where(df["SellRaw"], "SELL", ""))

    return df

def plot_combined_chart(df: pd.DataFrame, ticker: str, rsi_buy: float, rsi_sell: float, show_signals: bool = False, chart_type: str = "Line", chart_height: int = 800):
    """Combined chart with Price, RSI, and MACD using single x-axis and multiple y-axes for proper crosshair"""

    # Create figure with single x-axis and multiple y-axes (like fractal_predict.py)
    fig = go.Figure()

    # Price chart - either line or candlestick (yaxis)
    if chart_type == "Candlestick":
        # Create hover text for candlestick
        hover_text = [
            f"Time: {idx.strftime('%Y-%m-%d %H:%M')}<br>"
            f"Open: {row['Open']:.2f}<br>"
            f"High: {row['High']:.2f}<br>"
            f"Low: {row['Low']:.2f}<br>"
            f"Close: {row['Close']:.2f}<br>"
            f"Volume: {row.get('Volume', 0):,.0f}"
            for idx, row in df.iterrows()
        ]
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color='green',
            decreasing_line_color='red',
            hovertext=hover_text,
            hoverinfo='text',
            yaxis='y'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Close",
            line=dict(width=2),
            yaxis='y'
        ))

    if "BB_up" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB Upper", opacity=0.6, yaxis='y'))
    if "BB_mid" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", opacity=0.6, yaxis='y'))
    if "BB_lo" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB Lower", opacity=0.6, yaxis='y'))
    if "VWAP" in df and df["VWAP"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", opacity=0.6, yaxis='y'))
    if "EMA200" in df and df["EMA200"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", opacity=0.6, yaxis='y'))

    # Add BUY and SELL signal markers if enabled
    if show_signals:
        buy_signals = df[df["Signal"] == "BUY"]
        sell_signals = df[df["Signal"] == "SELL"]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals["Close"],
                mode='markers', name='BUY Signal',
                marker=dict(symbol='circle', size=8, color='green'),
                showlegend=True,
                yaxis='y'
            ))
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index, y=sell_signals["Close"],
                mode='markers', name='SELL Signal',
                marker=dict(symbol='circle', size=8, color='red'),
                showlegend=True,
                yaxis='y'
            ))

    # Volume chart (yaxis2) - colored bars based on price change
    if "Volume" in df.columns:
        volume_colors = ['green' if i > 0 and df["Close"].iloc[i] >= df["Close"].iloc[i-1] else 'red'
                        for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                            marker_color=volume_colors, opacity=0.5, yaxis='y2'))

    # RSI chart (yaxis3)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", line=dict(width=2, color='blue'), yaxis='y3'))

    # RSI reference lines
    fig.add_hline(y=rsi_buy, line_dash="dash", line_color="gray", opacity=0.7, yref='y3')
    fig.add_hline(y=rsi_sell, line_dash="dash", line_color="gray", opacity=0.7, yref='y3')
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref='y3')
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref='y3')

    # MACD chart (yaxis4)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=2), yaxis='y4'))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal", line=dict(width=2), yaxis='y4'))

    # MACD histogram with colors
    colors = ['green' if val >= 0 else 'red' for val in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist", marker_color=colors, opacity=0.5, yaxis='y4'))

    # Add shaded regions for extended hours
    # Data timestamps are now in US Eastern Time (New York)
    # Regular market hours: 9:30 AM to 4:00 PM ET
    # Extended hours: before 9:30 AM or after 4:00 PM ET
    # Only applicable for intraday data
    if hasattr(df.index, 'hour'):
        shapes = []
        for i in range(len(df)):
            timestamp = df.index[i]
            if hasattr(timestamp, 'hour'):
                hour = timestamp.hour
                minute = timestamp.minute
                # Extended hours: before 9:30 AM or after 4:00 PM ET
                if (hour < 9) or (hour == 9 and minute < 30) or (hour >= 16):
                    # Add vertical rectangle for this bar
                    if i < len(df) - 1:
                        x0 = df.index[i]
                        x1 = df.index[i + 1]
                    else:
                        # Last bar, estimate width
                        if len(df) > 1:
                            delta = df.index[i] - df.index[i-1]
                            x0 = df.index[i]
                            x1 = df.index[i] + delta
                        else:
                            continue

                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=x0,
                        x1=x1,
                        y0=0,
                        y1=1,
                        fillcolor="lightgray",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    ))

    else:
        shapes = []

    # Update layout with single x-axis and multiple y-axes using domains
    fig.update_layout(
        height=chart_height,
        showlegend=True,
        hovermode='x unified',
        hoverdistance=100,
        spikedistance=1000,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=10)
        ),
        # Single x-axis with spike configuration
        # Data is displayed in US Eastern Time (New York)
        xaxis=dict(
            showspikes=True,
            spikecolor='gray',
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1,
            spikedash='solid',
            rangeslider=dict(visible=False),
            # Display time in Eastern Time
            tickformat='%H:%M',
            # Hide gaps for weekends and overnight (but keep extended hours)
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[20, 4], pattern="hour"),  # Hide 8:00 PM to 4:00 AM ET (overnight gap only)
            ]
        ),
        # yaxis for Price chart (top ~50%)
        yaxis=dict(
            title=f'{ticker} Price',
            domain=[0.50, 1.0]
        ),
        # yaxis2 for Volume chart (below price, ~15%)
        yaxis2=dict(
            title='Volume',
            domain=[0.35, 0.48],
            showgrid=False
        ),
        # yaxis3 for RSI chart (middle ~15%)
        yaxis3=dict(
            title='RSI',
            domain=[0.19, 0.33],
            range=[0, 100]
        ),
        # yaxis4 for MACD chart (bottom ~15%)
        yaxis4=dict(
            title='MACD',
            domain=[0.0, 0.17]
        ),
        # Add shapes for extended hours shading
        shapes=shapes
    )

    # Apply crosshair to all y-axes
    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikedash='solid',
        spikethickness=1,
        spikecolor='gray'
    )

    # Use Plotly config
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'toImage']
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

def latest_signal_card(df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
        return
    last = df.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Close", f"{last['Close']:.2f}")
    cols[1].metric("RSI(14)", f"{last['RSI']:.2f}")
    cols[2].metric("BB %B", f"{(last['Close']-last['BB_lo'])/max(1e-9,(last['BB_up']-last['BB_lo'])):.2f}")
    cols[3].metric("Signal", last["Signal"] if last["Signal"] else "—")

def rules_summary(params: dict):
    st.markdown("### Strategy Rules")
    st.markdown(
        f"""
- **Buy** when RSI crosses **up** through **{params['rsi_buy']}** and Close ≥ BB Mid;
  avoid entries if price ≥ BB Upper or after **3 consecutive green bars**;
  must also be ≥ {'VWAP' if params['use_vwap'] else 'EMA200' if params['use_ema200'] else 'none'} (if enabled).
- **Sell** when RSI crosses **down** through **{params['rsi_sell']}** or Close < BB Mid;
  avoid panic exits if price ≤ BB Lower or after **3 consecutive red bars**.
        """
    )

def signal_reasons(df: pd.DataFrame, params: dict):
    """Display the reasons for the latest signal"""
    if df.empty:
        return

    # Find the last signal
    signals = df[df["Signal"].isin(["BUY", "SELL"])]
    if signals.empty:
        st.info("No BUY or SELL signals found in the current data.")
        return

    last_signal_row = signals.iloc[-1]
    signal_type = last_signal_row["Signal"]
    signal_time = last_signal_row.name.strftime('%Y-%m-%d %H:%M') if hasattr(last_signal_row.name, 'strftime') else str(last_signal_row.name)

    st.markdown("### Latest Signal Reasons")
    st.markdown(f"**Signal Type:** {signal_type} at {signal_time}")

    # Get the values at signal time
    close = last_signal_row["Close"]
    rsi = last_signal_row["RSI"]
    bb_mid = last_signal_row["BB_mid"]
    bb_up = last_signal_row["BB_up"]
    bb_lo = last_signal_row["BB_lo"]

    reasons = []

    if signal_type == "BUY":
        rsi_buy = params['rsi_buy']
        # Check RSI cross
        idx = signals.index[-1]
        idx_pos = df.index.get_loc(idx)
        if idx_pos > 0:
            prev_rsi = df.iloc[idx_pos - 1]["RSI"]
            if prev_rsi < rsi_buy <= rsi:
                reasons.append(f"✓ RSI crossed above {rsi_buy} (from {prev_rsi:.2f} to {rsi:.2f})")

        # Check BB position
        if close >= bb_mid:
            reasons.append(f"✓ Close ({close:.2f}) ≥ BB Mid ({bb_mid:.2f})")

        # Check not stretched
        if close < bb_up:
            reasons.append(f"✓ Price not stretched (Close {close:.2f} < BB Upper {bb_up:.2f})")

        # Check VWAP/EMA200 if enabled
        if params['use_vwap'] and 'VWAP' in last_signal_row and not pd.isna(last_signal_row['VWAP']):
            vwap = last_signal_row['VWAP']
            if close >= vwap:
                reasons.append(f"✓ Close ({close:.2f}) ≥ VWAP ({vwap:.2f})")
        elif params['use_ema200'] and 'EMA200' in last_signal_row and not pd.isna(last_signal_row['EMA200']):
            ema200 = last_signal_row['EMA200']
            if close >= ema200:
                reasons.append(f"✓ Close ({close:.2f}) ≥ EMA200 ({ema200:.2f})")

    elif signal_type == "SELL":
        rsi_sell = params['rsi_sell']
        # Check RSI cross
        idx = signals.index[-1]
        idx_pos = df.index.get_loc(idx)
        if idx_pos > 0:
            prev_rsi = df.iloc[idx_pos - 1]["RSI"]
            if prev_rsi > rsi_sell >= rsi:
                reasons.append(f"✓ RSI crossed below {rsi_sell} (from {prev_rsi:.2f} to {rsi:.2f})")

        # Check BB break
        if close < bb_mid:
            reasons.append(f"✓ Close ({close:.2f}) < BB Mid ({bb_mid:.2f})")

        # Check not in panic zone
        if close > bb_lo:
            reasons.append(f"✓ Not in panic zone (Close {close:.2f} > BB Lower {bb_lo:.2f})")

    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.markdown("- Signal triggered (check detailed conditions)")

def main():
    st.title("🧭 Entry/Exit Helper")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker", value="CIFR")
        period = st.selectbox("Period", ["1d","5d","2wk","1mo","3mo","6mo","1y","2y","5y","ytd"], index=0)
        interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","1h","1d"], index=0)

        st.header("Indicator Params")
        bb_len = st.number_input("BB Length", 5, 200, 20)
        bb_mult = st.number_input("BB Mult", 1.0, 4.0, 2.0, step=0.1)
        rsi_len = st.number_input("RSI Length", 5, 50, 14)
        rsi_buy = st.number_input("RSI Buy Cross Level", 20, 80, 50)
        rsi_sell = st.number_input("RSI Sell Cross Level", 20, 80, 50)

        macd_fast = st.number_input("MACD Fast", 2, 50, 12)
        macd_slow = st.number_input("MACD Slow", 5, 100, 26)
        macd_sig = st.number_input("MACD Signal", 2, 30, 9)

        st.header("Trend Filters")
        use_vwap = st.checkbox("Require VWAP for entries (intraday only)", value=False)
        use_ema200 = st.checkbox("Require EMA200 if VWAP unavailable", value=False)

        st.header("Chart Display")
        chart_type = st.radio("Price Chart Type", ["Line", "Candlestick"], index=0)
        chart_height = st.slider("Chart Height (pixels)", 400, 2000, 1200, step=50)
        show_signals = st.checkbox("Show BUY/SELL signal dots on chart", value=False)
        show_table = st.checkbox("Show data table", value=False)

        params = dict(
            bb_len=bb_len,
            bb_mult=float(bb_mult),
            rsi_len=int(rsi_len),
            rsi_buy=float(rsi_buy),
            rsi_sell=float(rsi_sell),
            macd_fast=int(macd_fast),
            macd_slow=int(macd_slow),
            macd_sig=int(macd_sig),
            use_vwap=bool(use_vwap),
            use_ema200=bool(use_ema200),
            interval=interval
        )

        st.divider()
        st.caption("Tip: For real-time-ish intraday, use period ≤ 5d with 1m/2m/5m intervals.")

    df = load_data(ticker, period, interval)
    if df.empty:
        st.warning("No data downloaded. Try a different period/interval.")
        return

    df = generate_signals(df.copy(), params)

    # Display last signal
    latest_signal_card(df)

    # Chart - full width
    plot_combined_chart(df, ticker, params["rsi_buy"], params["rsi_sell"], show_signals, chart_type, chart_height)

    # Rules and data table
    rules_summary(params)

    # Show signal reasons
    signal_reasons(df, params)

    if show_table:
        st.dataframe(df.tail(200)[["Close","RSI","BB_mid","BB_up","BB_lo","VWAP","EMA200","MACD","MACD_sig","MACD_hist","Signal"]], use_container_width=True)

    # Export
    st.download_button(
        label="Download latest signals (CSV)",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_{period}_{interval}_signals.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
