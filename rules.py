import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go

# ---------- Indicator helpers ----------

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def bollinger_bands(close, length=20, std=2):
    ma = close.rolling(length).mean()
    dev = close.rolling(length).std()
    upper = ma + std * dev
    lower = ma - std * dev
    return ma, upper, lower

def extract_full_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    For 1-minute intraday data, ensure we keep the most complete
    trading day (including extended hours) so the chart shows a full session.
    """
    if df.empty:
        return df
    try:
        # Convert index to timezone-aware (NY) for date comparisons
        tz = 'America/New_York'
        if df.index.tz is None:
            idx = df.index.tz_localize('UTC').tz_convert(tz)
        else:
            idx = df.index.tz_convert(tz)
        df = df.copy()
        df.index = idx

        date_counts = pd.Series(df.index.date).value_counts()
        if date_counts.empty:
            return df

        # Prefer today's date (Eastern). If missing or too sparse, fall back.
        tzinfo = df.index.tz
        today_et = dt.datetime.now(tzinfo).date()
        today_mask = (df.index.date == today_et)
        today_df = df.loc[today_mask].copy()
        if not today_df.empty and len(today_df) >= 100:
            return today_df

        # Otherwise choose the most recent date present
        sorted_dates = sorted(date_counts.index)
        latest_date = sorted_dates[-1]
        latest_df = df.loc[df.index.date == latest_date].copy()
        if not latest_df.empty:
            return latest_df

        # Fall back to the date with most rows (most complete session)
        target_date = date_counts.idxmax()
        fallback_df = df.loc[df.index.date == target_date].copy()
        return fallback_df if not fallback_df.empty else df
    except Exception:
        # In case of any unexpected issue, just return original dataframe
        return df

# ---------- Core backtest for ONE symbol (no BTC filter inside) ----------

def backtest_symbol(df1,
                    stop_loss=0.02,
                    tp_pct=0.04,
                    avoid_after="15:00"):
    """
    Apply Greg's rules to a single symbol.
    Returns:
        df1 (1m data with RSI),
        df5 (5m resampled with EMA/BB),
        trades_df,
        logs_df
    """
    logs = []

    # 1-min RSI
    df1 = df1.copy()
    df1["rsi_1m"] = rsi(df1["Close"], 14)

    # Resample to 5-min
    df5 = df1.resample("5T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # 5-min EMA9, EMA21 and BB(20,2)
    df5["ema9"] = ema(df5["Close"], 9)
    df5["ema21"] = ema(df5["Close"], 21)
    bb_mid, bb_up, bb_low = bollinger_bands(df5["Close"], 20, 2)
    df5["bb_mid"] = bb_mid
    df5["bb_up"] = bb_up
    df5["bb_low"] = bb_low

    # Map last 1-min RSI into each 5-min candle
    df5["rsi_1m_last"] = df1["rsi_1m"].resample("5T").last()

    trades = []
    position = None
    setup_active = False

    for t, row in df5.iterrows():
        time_str = t.strftime("%H:%M")

        # Optional: avoid last hour of session
        if avoid_after is not None and time_str >= avoid_after:
            continue

        close = row["Close"]
        ema9 = row["ema9"]
        bb_low_v = row["bb_low"]
        rsi_last = row["rsi_1m_last"]
        vol = row["Volume"]
        prev_vol = df5["Volume"].shift(1).loc[t]

        # ---- Manage open position ----
        if position is not None:
            bar_high = row["High"]
            bar_low = row["Low"]

            # Stop loss
            if bar_low <= position["entry_price"] * (1 - stop_loss):
                exit_price = position["entry_price"] * (1 - stop_loss)
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "SL"))
                logs.append({
                    "time": t,
                    "event": "exit_SL",
                    "price": exit_price,
                    "note": "Stop loss hit"
                })
                position = None
                continue

            # Take profit
            if bar_high >= position["entry_price"] * (1 + tp_pct):
                exit_price = position["entry_price"] * (1 + tp_pct)
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "TP"))
                logs.append({
                    "time": t,
                    "event": "exit_TP",
                    "price": exit_price,
                    "note": "Take profit hit"
                })
                position = None
                continue

        # ---- If no position, look for setup / entry ----
        if position is None:
            # 1) Oversold alert: 1m RSI < 30
            if rsi_last < 30:
                if not setup_active:
                    logs.append({
                        "time": t,
                        "event": "oversold_alert",
                        "price": close,
                        "note": f"1m RSI < 30 (RSI={rsi_last:.1f})"
                    })
                setup_active = True

            # 2) Entry when price recovers on 5-min
            if setup_active:
                base_ok = (
                    (bb_low_v is not np.nan and close > bb_low_v) and
                    (ema9 is not np.nan and close > ema9) and
                    (prev_vol is not np.nan and vol >= prev_vol)
                )

                if base_ok:
                    position = {"entry_time": t, "entry_price": close}
                    setup_active = False
                    trades.append((t, close, None, None, "ENTRY"))
                    logs.append({
                        "time": t,
                        "event": "entry",
                        "price": close,
                        "note": "Entry after oversold + price > BB low and EMA9, volume rising"
                    })

    # Close any open position at end of data
    if position is not None:
        last_time = df5.index[-1]
        last_close = df5["Close"].iloc[-1]
        trades.append((position["entry_time"], position["entry_price"],
                       last_time, last_close, "EOD"))
        logs.append({
            "time": last_time,
            "event": "exit_EOD",
            "price": last_close,
            "note": "Exit at end of data"
        })

    trades_df = pd.DataFrame(trades,
                             columns=["entry_time", "entry_price",
                                      "exit_time", "exit_price", "reason"])
    if not trades_df.empty:
        trades_df["exit_time"] = trades_df["exit_time"].fillna(trades_df["exit_time"].ffill())
        trades_df["exit_price"] = trades_df["exit_price"].fillna(trades_df["entry_price"])
        trades_df["return_pct"] = (trades_df["exit_price"] - trades_df["entry_price"]) \
                                  / trades_df["entry_price"] * 100

    logs_df = pd.DataFrame(logs).sort_values("time") if logs else \
        pd.DataFrame(columns=["time", "event", "price", "note"])

    return df1, df5, trades_df, logs_df

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Greg's Oversold → Recovery Scalping", layout="wide")

st.title("Greg's Oversold → Recovery Scalping Helper")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Main ticker", "IREN")
    period = st.selectbox("Data period", ["1d", "5d"], index=0)

    check_btc_first = st.checkbox(
        "Check BTC-USD with the same rules before running main ticker",
        value=False
    )

    show_signals = st.checkbox("Show buy/sell signals on chart", value=True)

    run_backtest_btn = st.button("Run backtest", use_container_width=True)

if run_backtest_btn:
    # ---- Optional BTC pre-check ----
    if check_btc_first:
        st.subheader("BTC-USD environment check")
        with st.spinner("Downloading BTC-USD data..."):
            btc_period = "5d" if period == "1d" else period
            btc_raw = yf.download("BTC-USD", period=btc_period,
                                  interval="1m", progress=False, prepost=True)

        if btc_raw.empty:
            st.warning("No BTC-USD data returned.")
        else:
            # Handle MultiIndex columns if present
            if isinstance(btc_raw.columns, pd.MultiIndex):
                btc_raw.columns = btc_raw.columns.get_level_values(0)

            # Convert timezone from UTC to US Eastern Time (New York)
            if hasattr(btc_raw.index, 'tz'):
                if btc_raw.index.tz is not None:
                    btc_raw.index = btc_raw.index.tz_convert('America/New_York')
                else:
                    btc_raw.index = btc_raw.index.tz_localize('UTC').tz_convert('America/New_York')

            if period == "1d":
                btc_raw = extract_full_day(btc_raw)

            btc_df1, btc_df5, btc_trades, btc_logs = backtest_symbol(btc_raw)

            if btc_trades.empty:
                st.write("No BTC trades with current rules (either no RSI<30 alert or no clean recovery).")
            else:
                st.dataframe(btc_trades)
                total = len(btc_trades)
                winrate = (btc_trades["return_pct"] > 0).mean()
                avg_ret = btc_trades["return_pct"].mean()
                cum_ret = (1 + btc_trades["return_pct"]/100).prod() - 1

                st.markdown(f"- **BTC total trades:** {total}")
                st.markdown(f"- **BTC win rate:** {winrate:.1%}")
                st.markdown(f"- **BTC avg return:** {avg_ret:.2f}%")
                st.markdown(f"- **BTC cumulative return:** {cum_ret*100:.2f}%")

            st.markdown("**BTC rule/event log:**")
            st.dataframe(btc_logs)

    # ---- Main ticker (IREN or whatever you choose) ----
    with st.spinner(f"Downloading {ticker} data..."):
        # Download 2 days for "1d" period to show more data
        main_period = "2d" if period == "1d" else period
        raw = yf.download(ticker, period=main_period,
                          interval="1m", progress=False, prepost=True)

    if raw.empty:
        st.error("No data returned for main ticker.")
    else:
        # Handle MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Convert timezone from UTC to US Eastern Time (New York)
        if hasattr(raw.index, 'tz'):
            if raw.index.tz is not None:
                raw.index = raw.index.tz_convert('America/New_York')
            else:
                raw.index = raw.index.tz_localize('UTC').tz_convert('America/New_York')

        df1, df5, trades_df, logs_df = backtest_symbol(raw)

        # Chart with entries/exits (5-min) - single chart with multiple y-axes
        st.subheader("5-min chart with signals")

        # Create figure with single x-axis and multiple y-axes (like check.py)
        fig = go.Figure()

        # Price chart - Candlestick (yaxis)
        fig.add_trace(go.Candlestick(
            x=df5.index,
            open=df5["Open"],
            high=df5["High"],
            low=df5["Low"],
            close=df5["Close"],
            name=ticker,
            yaxis='y'
        ))

        # Add Bollinger Bands in grey
        fig.add_trace(go.Scatter(x=df5.index, y=df5["bb_up"],
                                 name="BB Upper", line=dict(width=1, color='grey'),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5.index, y=df5["bb_mid"],
                                 name="BB Mid", line=dict(width=1, color='grey'),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5.index, y=df5["bb_low"],
                                 name="BB Lower", line=dict(width=1, color='grey'),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5.index, y=df5["ema9"],
                                 name="EMA9", line=dict(width=1),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5.index, y=df5["ema21"],
                                 name="EMA21", line=dict(width=1, color='purple'),
                                 yaxis='y'))

        if show_signals and not trades_df.empty:
            # Filter out "ENTRY" rows to avoid duplicate signals
            completed_trades = trades_df[trades_df["reason"] != "ENTRY"].copy()

            if not completed_trades.empty:
                # Calculate offset to position signals away from candlesticks
                price_range = df5["High"].max() - df5["Low"].min()
                offset = price_range * 0.03  # 3% of price range

                # Position entry signals below the entry price
                entry_y = completed_trades["entry_price"] - offset
                # Position exit signals above the exit price
                exit_y = completed_trades["exit_price"] + offset

                fig.add_trace(go.Scatter(
                    x=completed_trades["entry_time"],
                    y=entry_y,
                    mode="markers",
                    marker=dict(size=12, symbol="triangle-up", color='blue'),
                    name="Entries",
                    yaxis='y'
                ))
                fig.add_trace(go.Scatter(
                    x=completed_trades["exit_time"],
                    y=exit_y,
                    mode="markers",
                    marker=dict(size=12, symbol="x", color='orange'),
                    name="Exits",
                    yaxis='y'
                ))

        # Volume chart (yaxis2)
        volume_colors = ['green' if i > 0 and df5["Close"].iloc[i] >= df5["Close"].iloc[i-1] else 'red'
                        for i in range(len(df5))]
        fig.add_trace(go.Bar(
            x=df5.index,
            y=df5["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.5,
            yaxis='y2'
        ))

        # RSI chart (yaxis3)
        fig.add_trace(go.Scatter(x=df1.index, y=df1["rsi_1m"],
                                 name="RSI(14)", line=dict(width=2, color='blue'),
                                 yaxis='y3'))

        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref='y3')
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref='y3')

        # Add shaded regions for extended hours
        # Data timestamps are now in US Eastern Time (New York)
        # Regular market hours: 9:30 AM to 4:00 PM ET
        # Extended hours: before 9:30 AM or after 4:00 PM ET
        shapes = []
        if hasattr(df5.index, 'hour'):
            for i in range(len(df5)):
                timestamp = df5.index[i]
                if hasattr(timestamp, 'hour'):
                    hour = timestamp.hour
                    minute = timestamp.minute
                    # Extended hours: before 9:30 AM or after 4:00 PM ET
                    if (hour < 9) or (hour == 9 and minute < 30) or (hour >= 16):
                        # Add vertical rectangle for this bar
                        if i < len(df5) - 1:
                            x0 = df5.index[i]
                            x1 = df5.index[i + 1]
                        else:
                            # Last bar, estimate width
                            if len(df5) > 1:
                                bar_width = df5.index[-1] - df5.index[-2]
                                x0 = df5.index[i]
                                x1 = x0 + bar_width
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
                            fillcolor="grey",
                            opacity=0.1,
                            layer="below",
                            line_width=0,
                        ))
        else:
            shapes = []

        # Update layout with single x-axis and multiple y-axes using domains
        # Remove gaps by hiding weekends and overnight hours (8pm to 4am ET)
        rangebreaks = [
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[20, 4], pattern="hour")  # Hide overnight hours (8pm to 4am)
        ]

        fig.update_layout(
            showlegend=False,
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000,
            height=1067,
            template='plotly_white',
            # Single x-axis with spike configuration
            xaxis=dict(
                showspikes=True,
                spikecolor='gray',
                spikesnap='cursor',
                spikemode='across',
                spikethickness=1,
                spikedash='solid',
                rangeslider=dict(visible=False),
                tickformat='%H:%M',
                hoverformat='%Y-%m-%d %H:%M',
                rangebreaks=rangebreaks
            ),
            # yaxis for Price chart (top ~60%)
            yaxis=dict(
                title=f'{ticker} Price',
                domain=[0.40, 1.0]
            ),
            # yaxis2 for Volume chart (middle ~20%)
            yaxis2=dict(
                title='Volume',
                domain=[0.20, 0.38],
                showgrid=False
            ),
            # yaxis3 for RSI chart (bottom ~18%)
            yaxis3=dict(
                title='RSI',
                domain=[0.0, 0.18],
                range=[0, 100]
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

        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.subheader(f"Backtest for {ticker}")
        if trades_df.empty:
            st.write("No trades with current rules.")
        else:
            # Filter out rows where reason is "ENTRY" for display
            display_trades = trades_df[trades_df["reason"] != "ENTRY"].copy()

            st.dataframe(display_trades)
            total = len(display_trades)
            winrate = (display_trades["return_pct"] > 0).mean()
            avg_ret = display_trades["return_pct"].mean()
            cum_ret = (1 + display_trades["return_pct"]/100).prod() - 1

            st.markdown(f"- **Total trades:** {total}")
            st.markdown(f"- **Win rate:** {winrate:.1%}")
            st.markdown(f"- **Average return per trade:** {avg_ret:.2f}%")
            st.markdown(f"- **Cumulative return:** {cum_ret*100:.2f}%")

        # Logs + download
        st.subheader("Rule / event log")
        st.dataframe(logs_df)
        if not logs_df.empty:
            csv = logs_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download main ticker log as CSV",
                data=csv,
                file_name=f"{ticker}_rule_log.csv",
                mime="text/csv"
            )

