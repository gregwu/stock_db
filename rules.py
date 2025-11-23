import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go
import json
import os

# ---------- Settings persistence ----------

SETTINGS_FILE = ".streamlit/rules_settings.json"

def load_settings():
    """Load settings from file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_settings(settings):
    """Save settings to file"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

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

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

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
                    avoid_after="15:00",
                    use_rsi=True,
                    rsi_threshold=30,
                    use_bb_cross_up=False,
                    use_ema=True,
                    use_volume=True,
                    use_stop_loss=True,
                    use_take_profit=True,
                    use_rsi_overbought=False,
                    rsi_overbought_threshold=70,
                    use_ema_cross_up=False,
                    use_ema_cross_down=False,
                    use_price_below_ema9=False,
                    use_bb_cross_down=False,
                    use_macd_cross_up=False,
                    use_macd_cross_down=False,
                    use_price_above_ema21=False,
                    use_price_below_ema21=False,
                    use_macd_below_threshold=False,
                    macd_below_threshold=0.0,
                    use_macd_above_threshold=False,
                    macd_above_threshold=0.0):
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

    # Calculate MACD
    macd_line, signal_line, histogram = macd(df5["Close"])
    df5["macd"] = macd_line
    df5["macd_signal"] = signal_line
    df5["macd_hist"] = histogram

    # Map last 1-min RSI into each 5-min candle
    df5["rsi_1m_last"] = df1["rsi_1m"].resample("5T").last()

    # Calculate EMA crossovers (for entry and exit detection)
    df5["ema9_prev"] = df5["ema9"].shift(1)
    df5["ema21_prev"] = df5["ema21"].shift(1)

    # Calculate previous MACD values for crossover detection
    df5["macd_prev"] = df5["macd"].shift(1)
    df5["macd_signal_prev"] = df5["macd_signal"].shift(1)

    # Calculate previous close for BB crossover detection
    df5["close_prev"] = df5["Close"].shift(1)

    trades = []
    position = None
    setup_active = False

    for t, row in df5.iterrows():
        time_str = t.strftime("%H:%M")

        # Optional: avoid last hour of session
        if avoid_after is not None and time_str >= avoid_after:
            continue

        close = row["Close"]
        close_prev = row["close_prev"]
        ema9 = row["ema9"]
        ema21 = row["ema21"]
        ema9_prev = row["ema9_prev"]
        ema21_prev = row["ema21_prev"]
        bb_low_v = row["bb_low"]
        bb_up_v = row["bb_up"]
        rsi_last = row["rsi_1m_last"]
        vol = row["Volume"]
        prev_vol = df5["Volume"].shift(1).loc[t]
        macd_val = row["macd"]
        macd_signal_val = row["macd_signal"]
        macd_prev = row["macd_prev"]
        macd_signal_prev = row["macd_signal_prev"]

        # Detect EMA crossovers
        ema_cross_up = (ema9_prev is not np.nan and ema21_prev is not np.nan and
                        ema9 is not np.nan and ema21 is not np.nan and
                        ema9_prev <= ema21_prev and ema9 > ema21)
        ema_cross_down = (ema9_prev is not np.nan and ema21_prev is not np.nan and
                          ema9 is not np.nan and ema21 is not np.nan and
                          ema9_prev >= ema21_prev and ema9 < ema21)

        # Detect BB crossovers
        bb_cross_up = (close_prev is not np.nan and bb_up_v is not np.nan and
                       close is not np.nan and
                       close_prev <= bb_up_v and close > bb_up_v)
        bb_cross_down = (close_prev is not np.nan and bb_low_v is not np.nan and
                         close is not np.nan and
                         close_prev >= bb_low_v and close < bb_low_v)

        # Detect MACD crossovers
        macd_cross_up = (macd_prev is not np.nan and macd_signal_prev is not np.nan and
                        macd_val is not np.nan and macd_signal_val is not np.nan and
                        macd_prev <= macd_signal_prev and macd_val > macd_signal_val)
        macd_cross_down = (macd_prev is not np.nan and macd_signal_prev is not np.nan and
                          macd_val is not np.nan and macd_signal_val is not np.nan and
                          macd_prev >= macd_signal_prev and macd_val < macd_signal_val)

        # ---- Manage open position ----
        if position is not None:
            bar_high = row["High"]
            bar_low = row["Low"]

            # Stop loss (if enabled)
            if use_stop_loss and bar_low <= position["entry_price"] * (1 - stop_loss):
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

            # Take profit (if enabled)
            if use_take_profit and bar_high >= position["entry_price"] * (1 + tp_pct):
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

            # RSI Overbought exit
            if use_rsi_overbought and rsi_last > rsi_overbought_threshold:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "RSI_OB"))
                logs.append({
                    "time": t,
                    "event": "exit_RSI_OB",
                    "price": exit_price,
                    "note": f"RSI overbought exit (RSI={rsi_last:.1f} > {rsi_overbought_threshold})"
                })
                position = None
                continue

            # EMA cross down exit (bearish signal)
            if use_ema_cross_down and ema_cross_down:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "EMA_X_DOWN"))
                logs.append({
                    "time": t,
                    "event": "exit_EMA_X_DOWN",
                    "price": exit_price,
                    "note": f"EMA9 crossed below EMA21 (bearish)"
                })
                position = None
                continue

            # Price below EMA9 exit
            if use_price_below_ema9 and ema9 is not np.nan and close < ema9:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "PRICE_BELOW_EMA9"))
                logs.append({
                    "time": t,
                    "event": "exit_PRICE_BELOW_EMA9",
                    "price": exit_price,
                    "note": f"Price fell below EMA9 (Close={close:.2f}, EMA9={ema9:.2f})"
                })
                position = None
                continue

            # BB cross down exit (price crosses below lower band)
            if use_bb_cross_down and bb_cross_down:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "BB_X_DOWN"))
                logs.append({
                    "time": t,
                    "event": "exit_BB_X_DOWN",
                    "price": exit_price,
                    "note": f"Price crossed below BB lower (Close={close:.2f}, BB Low={bb_low_v:.2f})"
                })
                position = None
                continue

            # MACD cross down exit (MACD line crosses below signal line - bearish)
            if use_macd_cross_down and macd_cross_down:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "MACD_X_DOWN"))
                logs.append({
                    "time": t,
                    "event": "exit_MACD_X_DOWN",
                    "price": exit_price,
                    "note": f"MACD crossed below signal line (MACD={macd_val:.4f}, Signal={macd_signal_val:.4f})"
                })
                position = None
                continue

            # Price below EMA21 exit
            if use_price_below_ema21 and ema21 is not np.nan and close < ema21:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "PRICE_BELOW_EMA21"))
                logs.append({
                    "time": t,
                    "event": "exit_PRICE_BELOW_EMA21",
                    "price": exit_price,
                    "note": f"Price fell below EMA21 (Close={close:.2f}, EMA21={ema21:.2f})"
                })
                position = None
                continue

            # MACD above threshold exit
            if use_macd_above_threshold and macd_val is not np.nan and macd_val > macd_above_threshold:
                exit_price = close
                trades.append((position["entry_time"], position["entry_price"],
                               t, exit_price, "MACD_ABOVE_THRESHOLD"))
                logs.append({
                    "time": t,
                    "event": "exit_MACD_ABOVE_THRESHOLD",
                    "price": exit_price,
                    "note": f"MACD exceeded threshold (MACD={macd_val:.4f}, Threshold={macd_above_threshold:.4f})"
                })
                position = None
                continue

        # ---- If no position, look for setup / entry ----
        if position is None:
            # 1) Oversold alert: 1m RSI < threshold (if enabled)
            if use_rsi and rsi_last < rsi_threshold:
                if not setup_active:
                    logs.append({
                        "time": t,
                        "event": "oversold_alert",
                        "price": close,
                        "note": f"1m RSI < {rsi_threshold} (RSI={rsi_last:.1f})"
                    })
                setup_active = True
            elif not use_rsi:
                # If RSI check is disabled, always consider setup active
                setup_active = True

            # 2) Entry when price recovers on 5-min
            if setup_active:
                # Build entry conditions based on enabled rules
                conditions = []

                if use_ema_cross_up:
                    conditions.append(ema_cross_up)
                if use_bb_cross_up:
                    conditions.append(bb_cross_up)
                if use_macd_cross_up:
                    conditions.append(macd_cross_up)
                if use_ema:
                    conditions.append(ema9 is not np.nan and close > ema9)
                if use_price_above_ema21:
                    conditions.append(ema21 is not np.nan and close > ema21)
                if use_macd_below_threshold:
                    conditions.append(macd_val is not np.nan and macd_val < macd_below_threshold)
                if use_volume:
                    conditions.append(prev_vol is not np.nan and vol >= prev_vol)

                # If no conditions are enabled, don't allow entry
                if not conditions:
                    base_ok = False
                else:
                    base_ok = all(conditions)

                if base_ok:
                    position = {"entry_time": t, "entry_price": close}
                    setup_active = False
                    trades.append((t, close, None, None, "ENTRY"))

                    # Build note based on which conditions were checked
                    note_parts = []
                    if use_rsi:
                        note_parts.append(f"RSI < {rsi_threshold}")
                    if use_ema_cross_up:
                        note_parts.append("EMA9 crossed above EMA21")
                    if use_bb_cross_up:
                        note_parts.append("price crossed above BB upper")
                    if use_macd_cross_up:
                        note_parts.append(f"MACD crossed above signal (MACD={macd_val:.4f}, Signal={macd_signal_val:.4f})")
                    if use_ema:
                        note_parts.append("price > EMA9")
                    if use_price_above_ema21:
                        note_parts.append(f"price > EMA21 (Close={close:.2f}, EMA21={ema21:.2f})")
                    if use_macd_below_threshold:
                        note_parts.append(f"MACD < {macd_below_threshold} (MACD={macd_val:.4f})")
                    if use_volume:
                        note_parts.append("volume rising")

                    logs.append({
                        "time": t,
                        "event": "entry",
                        "price": close,
                        "note": f"Entry: {', '.join(note_parts)}"
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

# Initialize session state for settings persistence
if 'settings' not in st.session_state:
    # Try to load saved settings from file
    saved_settings = load_settings()
    if saved_settings:
        st.session_state.settings = saved_settings
    else:
        # Default settings
        st.session_state.settings = {
            'ticker': 'TQQQ',
            'period': '1d',
            'interval': '1m',
            'chart_height': 1150,
            'use_rsi': True,
            'rsi_threshold': 30,
            'use_ema_cross_up': False,
            'use_bb_cross_up': False,
            'use_macd_cross_up': False,
            'use_ema': True,
            'use_price_above_ema21': False,
            'use_volume': True,
            'use_stop_loss': True,
            'stop_loss_pct': 2.0,
            'use_take_profit': True,
            'take_profit_pct': 4.0,
            'use_rsi_overbought': True,
            'rsi_overbought_threshold': 70,
            'use_ema_cross_down': False,
            'use_price_below_ema9': False,
            'use_price_below_ema21': False,
            'use_bb_cross_down': False,
            'use_macd_cross_down': False,
            'use_macd_below_threshold': False,
            'macd_below_threshold': 0.0,
            'use_macd_above_threshold': False,
            'macd_above_threshold': 0.0,
            'use_time_filter': False,
            'avoid_after_time': '15:00',
            'show_signals': True,
            'show_reports': True
        }

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Main ticker", st.session_state.settings['ticker'])
    st.session_state.settings['ticker'] = ticker

    period_options = ["1d", "5d", "2wk", "1mo", "2mo", "3mo"]
    try:
        period_index = period_options.index(st.session_state.settings['period'])
    except ValueError:
        period_index = 0
    period = st.selectbox("Data period", period_options, index=period_index)
    st.session_state.settings['period'] = period

    # Add interval selector like check.py
    if 'interval' not in st.session_state.settings:
        st.session_state.settings['interval'] = "1m"

    interval_options = ["1m", "2m", "5m", "15m", "30m", "1h", "1d"]
    try:
        interval_index = interval_options.index(st.session_state.settings['interval'])
    except ValueError:
        interval_index = 0
    interval = st.selectbox("Data interval", interval_options, index=interval_index)
    st.session_state.settings['interval'] = interval

    run_backtest_btn = st.button("Run backtest", use_container_width=True)

    st.divider()

    st.subheader("Trading Rules")

    # Entry Conditions
    st.markdown("**Entry Conditions (all must be met):**")
    use_rsi = st.checkbox("RSI Oversold", value=st.session_state.settings['use_rsi'],
                         help="1m RSI must be below threshold before entry")
    st.session_state.settings['use_rsi'] = use_rsi

    rsi_threshold = st.number_input("RSI oversold threshold", min_value=10, max_value=50,
                                    value=st.session_state.settings['rsi_threshold'],
                                    disabled=not use_rsi,
                                    help="Alert when 1m RSI falls below this level")
    st.session_state.settings['rsi_threshold'] = rsi_threshold

    use_ema_cross_up = st.checkbox("EMA9 crosses above EMA21",
                                    value=st.session_state.settings['use_ema_cross_up'],
                                    help="Entry when EMA9 crosses above EMA21 (bullish)")
    st.session_state.settings['use_ema_cross_up'] = use_ema_cross_up

    use_bb_cross_up = st.checkbox("Price crosses above BB Upper",
                                   value=st.session_state.settings['use_bb_cross_up'],
                                   help="Entry when price crosses above Bollinger Band upper line")
    st.session_state.settings['use_bb_cross_up'] = use_bb_cross_up

    # Handle backwards compatibility for use_macd_cross_up
    if 'use_macd_cross_up' not in st.session_state.settings:
        st.session_state.settings['use_macd_cross_up'] = False

    use_macd_cross_up = st.checkbox("MACD crosses above Signal line",
                                     value=st.session_state.settings['use_macd_cross_up'],
                                     help="Entry when MACD line crosses above signal line (bullish)")
    st.session_state.settings['use_macd_cross_up'] = use_macd_cross_up

    use_ema = st.checkbox("Price > EMA9", value=st.session_state.settings['use_ema'])
    st.session_state.settings['use_ema'] = use_ema

    # Handle backwards compatibility for use_price_above_ema21
    if 'use_price_above_ema21' not in st.session_state.settings:
        st.session_state.settings['use_price_above_ema21'] = False

    use_price_above_ema21 = st.checkbox("Price > EMA21",
                                         value=st.session_state.settings['use_price_above_ema21'],
                                         help="Entry when price is above EMA21")
    st.session_state.settings['use_price_above_ema21'] = use_price_above_ema21

    # Handle backwards compatibility for use_macd_below_threshold
    if 'use_macd_below_threshold' not in st.session_state.settings:
        st.session_state.settings['use_macd_below_threshold'] = False
    if 'macd_below_threshold' not in st.session_state.settings:
        st.session_state.settings['macd_below_threshold'] = 0.0

    use_macd_below_threshold = st.checkbox("MACD < Threshold",
                                            value=st.session_state.settings['use_macd_below_threshold'],
                                            help="Entry when MACD is below threshold")
    st.session_state.settings['use_macd_below_threshold'] = use_macd_below_threshold

    macd_below_threshold = st.number_input("MACD below threshold value",
                                            value=st.session_state.settings['macd_below_threshold'],
                                            step=0.1,
                                            disabled=not use_macd_below_threshold,
                                            help="Enter when MACD is below this value")
    st.session_state.settings['macd_below_threshold'] = macd_below_threshold

    use_volume = st.checkbox("Volume Rising", value=st.session_state.settings['use_volume'],
                            help="Current 5m volume >= previous 5m volume")
    st.session_state.settings['use_volume'] = use_volume

    # Exit Rules
    st.markdown("**Exit Rules:**")
    use_stop_loss = st.checkbox("Exit on Stop Loss", value=st.session_state.settings['use_stop_loss'],
                                help="Exit when price drops by specified %")
    st.session_state.settings['use_stop_loss'] = use_stop_loss

    stop_loss_pct = st.number_input("Stop Loss %", min_value=0.5, max_value=10.0,
                                     value=st.session_state.settings['stop_loss_pct'], step=0.5,
                                     disabled=not use_stop_loss,
                                     help="Exit if price drops by this %")
    st.session_state.settings['stop_loss_pct'] = stop_loss_pct
    stop_loss_pct = stop_loss_pct / 100

    use_take_profit = st.checkbox("Exit on Take Profit", value=st.session_state.settings['use_take_profit'],
                                   help="Exit when price rises by specified %")
    st.session_state.settings['use_take_profit'] = use_take_profit

    take_profit_pct = st.number_input("Take Profit %", min_value=0.5, max_value=20.0,
                                       value=st.session_state.settings['take_profit_pct'], step=0.5,
                                       disabled=not use_take_profit,
                                       help="Exit if price rises by this %")
    st.session_state.settings['take_profit_pct'] = take_profit_pct
    take_profit_pct = take_profit_pct / 100

    use_rsi_overbought = st.checkbox("Exit on RSI Overbought",
                                      value=st.session_state.settings['use_rsi_overbought'],
                                      help="Exit when 1m RSI exceeds this level")
    st.session_state.settings['use_rsi_overbought'] = use_rsi_overbought

    rsi_overbought_threshold = st.number_input("RSI overbought threshold",
                                                min_value=50, max_value=90,
                                                value=st.session_state.settings['rsi_overbought_threshold'],
                                                disabled=not use_rsi_overbought,
                                                help="Exit when 1m RSI rises above this level")
    st.session_state.settings['rsi_overbought_threshold'] = rsi_overbought_threshold

    use_ema_cross_down = st.checkbox("Exit on EMA9 crosses below EMA21",
                                      value=st.session_state.settings['use_ema_cross_down'],
                                      help="Exit when EMA9 crosses below EMA21 (bearish)")
    st.session_state.settings['use_ema_cross_down'] = use_ema_cross_down

    use_price_below_ema9 = st.checkbox("Exit on Price < EMA9",
                                        value=st.session_state.settings['use_price_below_ema9'],
                                        help="Exit when price falls below EMA9")
    st.session_state.settings['use_price_below_ema9'] = use_price_below_ema9

    # Handle backwards compatibility for use_price_below_ema21
    if 'use_price_below_ema21' not in st.session_state.settings:
        st.session_state.settings['use_price_below_ema21'] = False

    use_price_below_ema21 = st.checkbox("Exit on Price < EMA21",
                                         value=st.session_state.settings['use_price_below_ema21'],
                                         help="Exit when price falls below EMA21")
    st.session_state.settings['use_price_below_ema21'] = use_price_below_ema21

    use_bb_cross_down = st.checkbox("Exit on Price crosses below BB Lower",
                                     value=st.session_state.settings['use_bb_cross_down'],
                                     help="Exit when price crosses below Bollinger Band lower line")
    st.session_state.settings['use_bb_cross_down'] = use_bb_cross_down

    # Handle backwards compatibility for use_macd_cross_down
    if 'use_macd_cross_down' not in st.session_state.settings:
        st.session_state.settings['use_macd_cross_down'] = False

    use_macd_cross_down = st.checkbox("Exit on MACD crosses below Signal line",
                                       value=st.session_state.settings['use_macd_cross_down'],
                                       help="Exit when MACD line crosses below signal line (bearish)")
    st.session_state.settings['use_macd_cross_down'] = use_macd_cross_down

    # Handle backwards compatibility for use_macd_above_threshold
    if 'use_macd_above_threshold' not in st.session_state.settings:
        st.session_state.settings['use_macd_above_threshold'] = False
    if 'macd_above_threshold' not in st.session_state.settings:
        st.session_state.settings['macd_above_threshold'] = 0.0

    use_macd_above_threshold = st.checkbox("Exit on MACD > Threshold",
                                            value=st.session_state.settings['use_macd_above_threshold'],
                                            help="Exit when MACD exceeds threshold")
    st.session_state.settings['use_macd_above_threshold'] = use_macd_above_threshold

    macd_above_threshold = st.number_input("MACD above threshold value",
                                            value=st.session_state.settings['macd_above_threshold'],
                                            step=0.1,
                                            disabled=not use_macd_above_threshold,
                                            help="Exit when MACD exceeds this value")
    st.session_state.settings['macd_above_threshold'] = macd_above_threshold

    # Time Filter
    use_time_filter = st.checkbox("Avoid trading after specific time",
                                   value=st.session_state.settings['use_time_filter'])
    st.session_state.settings['use_time_filter'] = use_time_filter

    avoid_after_time = st.text_input("Avoid entries after (HH:MM)",
                                     value=st.session_state.settings['avoid_after_time'],
                                     disabled=not use_time_filter,
                                     help="No new entries after this time (24h format)")
    st.session_state.settings['avoid_after_time'] = avoid_after_time

    st.divider()

    show_signals = st.checkbox("Show buy/sell signals on chart",
                               value=st.session_state.settings['show_signals'])
    st.session_state.settings['show_signals'] = show_signals

    # Handle backwards compatibility for show_reports
    if 'show_reports' not in st.session_state.settings:
        st.session_state.settings['show_reports'] = True

    show_reports = st.checkbox("Show backtest reports",
                               value=st.session_state.settings['show_reports'],
                               help="Show backtest summary table and event logs")
    st.session_state.settings['show_reports'] = show_reports

    # Handle backwards compatibility for chart_height
    if 'chart_height' not in st.session_state.settings:
        st.session_state.settings['chart_height'] = 1150

    chart_height = st.slider("Chart height (pixels)", min_value=600, max_value=2000,
                             value=st.session_state.settings['chart_height'], step=50,
                             help="Adjust the height of the chart")
    st.session_state.settings['chart_height'] = chart_height

    # Save settings to file
    save_settings(st.session_state.settings)

if run_backtest_btn:
    # ---- Main ticker ----
    with st.spinner(f"Downloading {ticker} data..."):
        # Map period to yfinance compatible values and adjust for better data display
        # yfinance supports: {"1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}
        period_map = {
            "1d": "5d",    # Download 5 days for 1d period to get enough data for indicators
            "5d": "5d",
            "2wk": "1mo",  # yfinance doesn't support 2wk, use 1mo instead
            "1mo": "1mo",
            "2mo": "3mo",  # yfinance doesn't support 2mo, use 3mo instead
            "3mo": "3mo"
        }
        main_period = period_map.get(period, period)
        # Use extended hours for intraday intervals only (like check.py does)
        use_extended_hours = interval not in ["1d", "5d", "1wk", "1mo", "3mo"]
        raw = yf.download(ticker, period=main_period,
                          interval=interval, progress=False, prepost=use_extended_hours)

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

        df1, df5, trades_df, logs_df = backtest_symbol(
            raw,
            stop_loss=stop_loss_pct,
            tp_pct=take_profit_pct,
            avoid_after=avoid_after_time if use_time_filter else None,
            use_rsi=use_rsi,
            rsi_threshold=rsi_threshold,
            use_bb_cross_up=use_bb_cross_up,
            use_ema=use_ema,
            use_volume=use_volume,
            use_stop_loss=use_stop_loss,
            use_take_profit=use_take_profit,
            use_rsi_overbought=use_rsi_overbought,
            rsi_overbought_threshold=rsi_overbought_threshold,
            use_ema_cross_up=use_ema_cross_up,
            use_ema_cross_down=use_ema_cross_down,
            use_price_below_ema9=use_price_below_ema9,
            use_bb_cross_down=use_bb_cross_down,
            use_macd_cross_up=use_macd_cross_up,
            use_macd_cross_down=use_macd_cross_down,
            use_price_above_ema21=use_price_above_ema21,
            use_price_below_ema21=use_price_below_ema21,
            use_macd_below_threshold=use_macd_below_threshold,
            macd_below_threshold=macd_below_threshold,
            use_macd_above_threshold=use_macd_above_threshold,
            macd_above_threshold=macd_above_threshold
        )

        # Filter chart data based on selected period
        if not df1.empty:
            last_time = df1.index[-1]

            if period == "1d":
                # Show only last 26 hours for 1d period
                cutoff_time = last_time - pd.Timedelta(hours=26)
                df1_display = df1[df1.index >= cutoff_time].copy()
                df5_display = df5[df5.index >= cutoff_time].copy()
            elif period == "2wk":
                # Show only last 2 weeks (14 days) for 2wk period
                cutoff_time = last_time - pd.Timedelta(days=14)
                df1_display = df1[df1.index >= cutoff_time].copy()
                df5_display = df5[df5.index >= cutoff_time].copy()
            elif period == "2mo":
                # Show only last 2 months (60 days) for 2mo period
                cutoff_time = last_time - pd.Timedelta(days=60)
                df1_display = df1[df1.index >= cutoff_time].copy()
                df5_display = df5[df5.index >= cutoff_time].copy()
            else:
                df1_display = df1
                df5_display = df5
        else:
            df1_display = df1
            df5_display = df5

        # Chart with entries/exits (5-min) - single chart with multiple y-axes
        st.subheader("5-min chart with signals")

        # Create figure with single x-axis and multiple y-axes (like check.py)
        fig = go.Figure()

        # Price chart - Candlestick (yaxis)
        fig.add_trace(go.Candlestick(
            x=df5_display.index,
            open=df5_display["Open"],
            high=df5_display["High"],
            low=df5_display["Low"],
            close=df5_display["Close"],
            name=ticker,
            yaxis='y'
        ))

        # Add Bollinger Bands in grey
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["bb_up"],
                                 name="BB Upper", line=dict(width=1, color='grey'),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["bb_mid"],
                                 name="BB Mid", line=dict(width=1, color='grey'),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["bb_low"],
                                 name="BB Lower", line=dict(width=1, color='grey'),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["ema9"],
                                 name="EMA9", line=dict(width=1),
                                 yaxis='y'))
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["ema21"],
                                 name="EMA21", line=dict(width=1, color='purple'),
                                 yaxis='y'))

        if show_signals and not trades_df.empty:
            # Filter out "ENTRY" rows to avoid duplicate signals
            completed_trades = trades_df[trades_df["reason"] != "ENTRY"].copy()

            if not completed_trades.empty:
                # Position signals on RSI chart at fixed positions
                # Entry signals at RSI level 20 (bottom)
                # Exit signals at RSI level 80 (top)

                # Get entry notes from logs_df
                entry_notes = []
                for entry_time in completed_trades["entry_time"]:
                    matching_log = logs_df[(logs_df["time"] == entry_time) & (logs_df["event"] == "entry")]
                    if not matching_log.empty:
                        entry_notes.append(matching_log.iloc[0]["note"])
                    else:
                        entry_notes.append("Entry")

                # Get exit notes from logs_df
                exit_info = []
                for idx, row in completed_trades.iterrows():
                    exit_time = row["exit_time"]
                    reason = row["reason"]
                    matching_log = logs_df[(logs_df["time"] == exit_time) & (logs_df["event"].str.contains("exit"))]
                    if not matching_log.empty:
                        exit_info.append(f"{reason}: {matching_log.iloc[0]['note']}")
                    else:
                        exit_info.append(f"{reason}")

                fig.add_trace(go.Scatter(
                    x=completed_trades["entry_time"],
                    y=[20] * len(completed_trades),
                    mode="markers",
                    marker=dict(size=12, symbol="triangle-up", color='green'),
                    name="Entries",
                    text=entry_notes,
                    hoverinfo='text+x',
                    yaxis='y3'
                ))
                fig.add_trace(go.Scatter(
                    x=completed_trades["exit_time"],
                    y=[80] * len(completed_trades),
                    mode="markers",
                    marker=dict(size=12, symbol="triangle-down", color='red'),
                    name="Exits",
                    text=exit_info,
                    hoverinfo='text+x',
                    yaxis='y3'
                ))

        # Volume chart (yaxis2)
        volume_colors = ['green' if i > 0 and df5_display["Close"].iloc[i] >= df5_display["Close"].iloc[i-1] else 'red'
                        for i in range(len(df5_display))]
        fig.add_trace(go.Bar(
            x=df5_display.index,
            y=df5_display["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.5,
            yaxis='y2'
        ))

        # RSI chart (yaxis3)
        fig.add_trace(go.Scatter(x=df1_display.index, y=df1_display["rsi_1m"],
                                 name="RSI(14)", line=dict(width=2, color='navy'),
                                 yaxis='y3'))

        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref='y3')
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref='y3')

        # MACD chart (yaxis4)
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["macd"],
                                 name="MACD", line=dict(width=2, color='navy'),
                                 yaxis='y4'))
        fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["macd_signal"],
                                 name="Signal", line=dict(width=2, color='orange'),
                                 yaxis='y4'))

        # MACD histogram
        hist_colors = ['green' if val >= 0 else 'red' for val in df5_display["macd_hist"]]
        fig.add_trace(go.Bar(
            x=df5_display.index,
            y=df5_display["macd_hist"],
            name="MACD Histogram",
            marker_color=hist_colors,
            opacity=0.5,
            yaxis='y4'
        ))

        # MACD zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, yref='y4')

        # Add shaded regions for extended hours
        # Data timestamps are now in US Eastern Time (New York)
        # Regular market hours: 9:30 AM to 4:00 PM ET
        # Extended hours: before 9:30 AM or after 4:00 PM ET
        shapes = []
        if hasattr(df5_display.index, 'hour'):
            for i in range(len(df5_display)):
                timestamp = df5_display.index[i]
                if hasattr(timestamp, 'hour'):
                    hour = timestamp.hour
                    minute = timestamp.minute
                    # Extended hours: before 9:30 AM or after 4:00 PM ET
                    if (hour < 9) or (hour == 9 and minute < 30) or (hour >= 16):
                        # Add vertical rectangle for this bar
                        if i < len(df5_display) - 1:
                            x0 = df5_display.index[i]
                            x1 = df5_display.index[i + 1]
                        else:
                            # Last bar, estimate width
                            if len(df5_display) > 1:
                                bar_width = df5_display.index[-1] - df5_display.index[-2]
                                x0 = df5_display.index[i]
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

        # Set x-axis range to match filtered period
        xaxis_range = None
        if period in ["1d", "2wk", "2mo"] and not df5_display.empty:
            # Set range to show from cutoff_time to last_time
            xaxis_range = [cutoff_time, last_time]

        fig.update_layout(
            showlegend=False,
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000,
            height=chart_height,
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
                rangebreaks=rangebreaks,
                range=xaxis_range
            ),
            # yaxis for Price chart (top 47% - increased from 45%)
            yaxis=dict(
                title=f'{ticker} Price',
                domain=[0.53, 1.0]  # 0.47 height
            ),
            # yaxis2 for Volume chart (15%)
            yaxis2=dict(
                title='Volume',
                domain=[0.36, 0.51],  # 0.15 height
                showgrid=False
            ),
            # yaxis3 for RSI chart (15%)
            yaxis3=dict(
                title='RSI',
                domain=[0.19, 0.34],  # 0.15 height
                range=[0, 100]
            ),
            # yaxis4 for MACD chart (bottom 16%)
            yaxis4=dict(
                title='MACD',
                domain=[0.0, 0.16],  # 0.16 height
                showgrid=True
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

        # Reports section (only show if enabled)
        if show_reports:
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

