import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go
import json
import os
from scipy.signal import argrelextrema

# ---------- Settings persistence ----------

SETTINGS_FILE = ".streamlit/rules_settings.json"
ALPACA_CONFIG_FILE = "alpaca.json"

def load_alpaca_config():
    """Load alpaca.json configuration"""
    if os.path.exists(ALPACA_CONFIG_FILE):
        try:
            with open(ALPACA_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {ALPACA_CONFIG_FILE}: {e}")
            return None
    return None

def get_available_tickers():
    """Get list of tickers from alpaca.json"""
    config = load_alpaca_config()
    if config and 'signal_actions' in config and 'tickers' in config['signal_actions']:
        return list(config['signal_actions']['tickers'].keys())
    return ['TQQQ']  # Fallback default

def load_strategy_from_alpaca(ticker=None):
    """Load strategy settings from alpaca.json, with optional ticker-specific overrides"""
    config = load_alpaca_config()
    if not config or 'strategy' not in config:
        return None

    strategy = config['strategy']

    # If ticker specified, check for ticker-specific overrides
    if ticker and 'signal_actions' in config:
        ticker_configs = config['signal_actions'].get('tickers', {})
        ticker_config = ticker_configs.get(ticker, {})
        ticker_strategy = ticker_config.get('strategy', {})

        if ticker_strategy:
            # Merge ticker-specific overrides
            import copy
            merged_strategy = copy.deepcopy(strategy)

            if 'entry_conditions' in ticker_strategy:
                if 'entry_conditions' not in merged_strategy:
                    merged_strategy['entry_conditions'] = {}
                merged_strategy['entry_conditions'].update(ticker_strategy['entry_conditions'])

            if 'exit_conditions' in ticker_strategy:
                if 'exit_conditions' not in merged_strategy:
                    merged_strategy['exit_conditions'] = {}
                merged_strategy['exit_conditions'].update(ticker_strategy['exit_conditions'])

            if 'risk_management' in ticker_strategy:
                if 'risk_management' not in merged_strategy:
                    merged_strategy['risk_management'] = {}
                merged_strategy['risk_management'].update(ticker_strategy['risk_management'])

            return merged_strategy

    return strategy

def load_settings(ticker=None):
    """Load settings from file, optionally for a specific ticker"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                all_settings = json.load(f)

            # If ticker specified, return ticker-specific settings
            if ticker and isinstance(all_settings, dict) and 'tickers' in all_settings:
                return all_settings.get('tickers', {}).get(ticker)

            # Return global settings (backward compatibility)
            return all_settings
        except Exception:
            pass
    return None

def save_settings(settings, ticker=None):
    """Save settings to file, optionally for a specific ticker"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        # Load existing settings
        all_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    all_settings = json.load(f)
            except Exception:
                pass

        # If ticker specified, save under ticker key
        if ticker:
            if 'tickers' not in all_settings:
                all_settings['tickers'] = {}
            all_settings['tickers'][ticker] = settings
        else:
            # Save as global settings (backward compatibility)
            all_settings = settings

        # Write back to file
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(all_settings, f, indent=2)
    except Exception:
        pass

def save_settings_to_alpaca(settings, ticker):
    """Save settings to alpaca.json for a specific ticker"""
    try:
        config = load_alpaca_config()
        if not config:
            return False

        # Convert UI settings to alpaca.json format
        entry_conditions = {
            'use_rsi': settings.get('use_rsi', False),
            'rsi_threshold': settings.get('rsi_threshold', 30),
            'use_ema_cross_up': settings.get('use_ema_cross_up', False),
            'use_bb_cross_up': settings.get('use_bb_cross_up', False),
            'use_macd_cross_up': settings.get('use_macd_cross_up', False),
            'use_price_vs_ema9': settings.get('use_price_vs_ema9', False),
            'use_price_vs_ema21': settings.get('use_price_vs_ema21', False),
            'use_macd_threshold': settings.get('use_macd_threshold', False),
            'macd_threshold': settings.get('macd_threshold', 0.1),
            'use_macd_valley': settings.get('use_macd_valley', False),
            'use_ema': settings.get('use_ema', False),
            'use_volume': settings.get('use_volume', False)
        }

        exit_conditions = {
            'use_rsi_exit': settings.get('use_rsi_exit', False),
            'rsi_exit_threshold': settings.get('rsi_exit_threshold', 70),
            'use_ema_cross_down': settings.get('use_ema_cross_down', False),
            'use_bb_cross_down': settings.get('use_bb_cross_down', False),
            'use_macd_cross_down': settings.get('use_macd_cross_down', False),
            'use_price_vs_ema9_exit': settings.get('use_price_vs_ema9_exit', False),
            'use_price_vs_ema21_exit': settings.get('use_price_vs_ema21_exit', False),
            'use_macd_peak': settings.get('use_macd_peak', False)
        }

        risk_management = {
            'stop_loss': settings.get('stop_loss', 0.02),
            'take_profit': settings.get('take_profit', 0.03),
            'use_stop_loss': settings.get('use_stop_loss', True),
            'use_take_profit': settings.get('use_take_profit', False)
        }

        # Prepare ticker strategy
        ticker_strategy = {
            'entry_conditions': entry_conditions,
            'exit_conditions': exit_conditions,
            'risk_management': risk_management
        }

        # Update config
        if 'signal_actions' not in config:
            config['signal_actions'] = {'tickers': {}}
        if 'tickers' not in config['signal_actions']:
            config['signal_actions']['tickers'] = {}
        if ticker not in config['signal_actions']['tickers']:
            return False

        # Save strategy to ticker config
        config['signal_actions']['tickers'][ticker]['strategy'] = ticker_strategy

        # Write back to alpaca.json
        with open(ALPACA_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error saving to alpaca.json: {e}")
        return False

def update_ticker_enabled_status(ticker, enabled):
    """Update ticker enabled status in alpaca.json"""
    try:
        config = load_alpaca_config()
        if not config:
            return False

        if 'signal_actions' not in config or 'tickers' not in config['signal_actions']:
            return False

        if ticker not in config['signal_actions']['tickers']:
            return False

        # Update enabled status
        config['signal_actions']['tickers'][ticker]['enabled'] = enabled

        # Save back to file
        with open(ALPACA_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error updating ticker status: {e}")
        return False

def get_ticker_enabled_status(ticker):
    """Get ticker enabled status from alpaca.json"""
    config = load_alpaca_config()
    if config and 'signal_actions' in config and 'tickers' in config['signal_actions']:
        ticker_config = config['signal_actions']['tickers'].get(ticker, {})
        return ticker_config.get('enabled', True)
    return True

def get_ticker_default_quantity(ticker):
    """Get ticker default_quantity from alpaca.json"""
    config = load_alpaca_config()
    if config and 'signal_actions' in config and 'tickers' in config['signal_actions']:
        ticker_config = config['signal_actions']['tickers'].get(ticker, {})
        return ticker_config.get('default_quantity', 100)
    return 100

def update_ticker_default_quantity(ticker, quantity):
    """Update ticker default_quantity in alpaca.json"""
    try:
        config = load_alpaca_config()
        if not config:
            return False

        if 'signal_actions' not in config or 'tickers' not in config['signal_actions']:
            return False

        if ticker not in config['signal_actions']['tickers']:
            return False

        # Update default_quantity
        config['signal_actions']['tickers'][ticker]['default_quantity'] = quantity

        # Save back to file
        with open(ALPACA_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error updating default_quantity: {e}")
        return False

def delete_ticker(ticker):
    """Delete ticker from alpaca.json"""
    try:
        config = load_alpaca_config()
        if not config:
            return False

        if 'signal_actions' not in config or 'tickers' not in config['signal_actions']:
            return False

        if ticker not in config['signal_actions']['tickers']:
            return False

        # Delete ticker
        del config['signal_actions']['tickers'][ticker]

        # Save back to file
        with open(ALPACA_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error deleting ticker: {e}")
        return False

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
                    macd_above_threshold=0.0,
                    use_macd_peak=False,
                    use_macd_valley=False):
    """
    Apply Greg's rules to a single symbol.
    Returns:
        df1 (original interval data with RSI),
        df5 (same as df1, kept for backward compatibility),
        trades_df,
        logs_df
    """
    logs = []

    # Calculate RSI on the selected interval data
    df1 = df1.copy()
    df1["rsi"] = rsi(df1["Close"], 14)

    # Use the input data directly (no resampling)
    # df5 is kept for backward compatibility but points to same data
    df5 = df1.copy()

    # Calculate EMA9, EMA21 and BB(20,2) on selected interval
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

    # Detect MACD peaks and valleys using argrelextrema (order=3 means look 3 candles left and right)
    # Peak: local maximum (detected regardless of sign)
    # Valley: local minimum (detected regardless of sign)
    df5["macd_peak"] = False
    df5["macd_valley"] = False
    if len(df5) > 6:  # Need at least 7 candles for order=3
        macd_values = df5["macd"].values
        peak_indices = argrelextrema(macd_values, np.greater, order=3)[0]
        valley_indices = argrelextrema(macd_values, np.less, order=3)[0]

        # Mark all peaks (no sign restriction)
        for idx in peak_indices:
            df5.iloc[idx, df5.columns.get_loc("macd_peak")] = True

        # Mark all valleys (no sign restriction)
        for idx in valley_indices:
            df5.iloc[idx, df5.columns.get_loc("macd_valley")] = True

    # RSI is now calculated on the selected interval
    df5["rsi_last"] = df5["rsi"]

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
        rsi_last = row["rsi_last"]
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

        # Detect MACD peaks and valleys
        is_macd_peak = row.get("macd_peak", False)
        is_macd_valley = row.get("macd_valley", False)

        # ---- Manage open position ----
        if position is not None:
            bar_high = row["High"]
            bar_low = row["Low"]

            # Stop loss (if enabled) - always takes priority
            if use_stop_loss and bar_low <= position["entry_price"] * (1 - stop_loss):
                exit_price = position["entry_price"] * (1 - stop_loss)
                entry_reason = position.get("entry_reason", "N/A")
                trades.append((position["entry_time"], position["entry_price"], entry_reason,
                               t, exit_price, "SL"))
                logs.append({
                    "time": t,
                    "event": "exit_SL",
                    "price": exit_price,
                    "note": "Stop loss hit"
                })
                position = None
                continue

            # Take profit (if enabled) - always takes priority
            if use_take_profit and bar_high >= position["entry_price"] * (1 + tp_pct):
                exit_price = position["entry_price"] * (1 + tp_pct)
                entry_reason = position.get("entry_reason", "N/A")
                trades.append((position["entry_time"], position["entry_price"], entry_reason,
                               t, exit_price, "TP"))
                logs.append({
                    "time": t,
                    "event": "exit_TP",
                    "price": exit_price,
                    "note": "Take profit hit"
                })
                position = None
                continue

            # Build exit conditions list - ALL must be satisfied
            exit_conditions = []

            if use_rsi_overbought:
                exit_conditions.append(rsi_last > rsi_overbought_threshold)
            if use_ema_cross_down:
                exit_conditions.append(ema_cross_down)
            if use_price_below_ema9:
                exit_conditions.append(ema9 is not np.nan and close < ema9)
            if use_bb_cross_down:
                exit_conditions.append(bb_cross_down)
            if use_macd_cross_down:
                exit_conditions.append(macd_cross_down)
            if use_price_below_ema21:
                exit_conditions.append(ema21 is not np.nan and close < ema21)
            if use_macd_above_threshold:
                exit_conditions.append(macd_val is not np.nan and macd_val > macd_above_threshold)
            if use_macd_peak:
                exit_conditions.append(is_macd_peak)

            # Check if all enabled exit conditions are met
            if exit_conditions and all(exit_conditions):
                exit_price = close

                # Build exit note based on which conditions were checked
                exit_note_parts = []
                if use_rsi_overbought:
                    exit_note_parts.append(f"RSI > {rsi_overbought_threshold} (RSI={rsi_last:.1f})")
                if use_ema_cross_down:
                    exit_note_parts.append("EMA9 crossed below EMA21")
                if use_price_below_ema9:
                    exit_note_parts.append(f"Price < EMA9 (Close={close:.2f}, EMA9={ema9:.2f})")
                if use_bb_cross_down:
                    exit_note_parts.append(f"Price crossed below BB lower (Close={close:.2f}, BB Low={bb_low_v:.2f})")
                if use_macd_cross_down:
                    exit_note_parts.append(f"MACD crossed below signal (MACD={macd_val:.4f}, Signal={macd_signal_val:.4f})")
                if use_price_below_ema21:
                    exit_note_parts.append(f"Price < EMA21 (Close={close:.2f}, EMA21={ema21:.2f})")
                if use_macd_above_threshold:
                    exit_note_parts.append(f"MACD > {macd_above_threshold} (MACD={macd_val:.4f})")
                if use_macd_peak:
                    exit_note_parts.append(f"MACD peak (MACD={macd_val:.4f})")

                exit_reason = ', '.join(exit_note_parts)
                entry_reason = position.get("entry_reason", "N/A")
                trades.append((position["entry_time"], position["entry_price"], entry_reason,
                               t, exit_price, f"Exit: {exit_reason}"))
                logs.append({
                    "time": t,
                    "event": "exit_conditions_met",
                    "price": exit_price,
                    "note": f"Exit: {', '.join(exit_note_parts)}"
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

            # 2) Entry when price recovers on selected interval
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
                if use_macd_valley:
                    conditions.append(is_macd_valley)
                if use_volume:
                    conditions.append(prev_vol is not np.nan and vol >= prev_vol)

                # If no conditions are enabled, don't allow entry
                if not conditions:
                    base_ok = False
                else:
                    base_ok = all(conditions)

                if base_ok:
                    # Build note based on which conditions were checked
                    note_parts = []
                    if use_rsi:
                        note_parts.append(f"RSI < {rsi_threshold} (RSI={rsi_last:.1f})")
                    if use_ema_cross_up:
                        note_parts.append(f"EMA9 crossed above EMA21 (EMA9={ema9:.2f}, EMA21={ema21:.2f})")
                    if use_bb_cross_up:
                        note_parts.append(f"Price crossed above BB upper (Close={close:.2f}, BB Upper={bb_up_v:.2f})")
                    if use_macd_cross_up:
                        note_parts.append(f"MACD crossed above signal (MACD={macd_val:.4f}, Signal={macd_signal_val:.4f})")
                    if use_ema:
                        note_parts.append(f"Price > EMA9 (Close={close:.2f}, EMA9={ema9:.2f})")
                    if use_price_above_ema21:
                        note_parts.append(f"Price > EMA21 (Close={close:.2f}, EMA21={ema21:.2f})")
                    if use_macd_below_threshold:
                        note_parts.append(f"MACD < {macd_below_threshold} (MACD={macd_val:.4f})")
                    if use_macd_valley:
                        note_parts.append(f"MACD valley detected (MACD={macd_val:.4f})")
                    if use_volume:
                        note_parts.append(f"Volume rising (Vol={vol:.0f}, PrevVol={prev_vol:.0f})")

                    entry_reason = ', '.join(note_parts)
                    position = {"entry_time": t, "entry_price": close, "entry_reason": entry_reason}
                    setup_active = False
                    trades.append((t, close, f"Entry: {entry_reason}", None, None, None))

                    logs.append({
                        "time": t,
                        "event": "entry",
                        "price": close,
                        "note": f"Entry: {entry_reason}"
                    })

    # Close any open position at end of data
    if position is not None:
        last_time = df5.index[-1]
        last_close = df5["Close"].iloc[-1]
        entry_reason = position.get("entry_reason", "N/A")
        trades.append((position["entry_time"], position["entry_price"], entry_reason,
                       last_time, last_close, "EOD"))
        logs.append({
            "time": last_time,
            "event": "exit_EOD",
            "price": last_close,
            "note": "Exit at end of data"
        })

    trades_df = pd.DataFrame(trades,
                             columns=["entry_time", "entry_price", "entry_reason",
                                      "exit_time", "exit_price", "exit_reason"])
    if not trades_df.empty:
        trades_df["exit_time"] = trades_df["exit_time"].fillna(trades_df["exit_time"].ffill())
        trades_df["exit_price"] = trades_df["exit_price"].fillna(trades_df["entry_price"])
        trades_df["return_pct"] = (trades_df["exit_price"] - trades_df["entry_price"]) \
                                  / trades_df["entry_price"] * 100

    logs_df = pd.DataFrame(logs).sort_values("time") if logs else \
        pd.DataFrame(columns=["time", "event", "price", "note"])

    return df1, df5, trades_df, logs_df

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Strategy Scalping", layout="wide")

st.title("Strategy Scalping ")

# Initialize session state for settings persistence
if 'settings' not in st.session_state:
    # Initialize with default ticker
    default_ticker = get_available_tickers()[0]

    # Try to load saved settings for default ticker
    saved_settings = load_settings(default_ticker)
    if saved_settings:
        st.session_state.settings = saved_settings
        st.session_state.settings['ticker'] = default_ticker  # Ensure ticker is set
    else:
        # Default settings
        st.session_state.settings = {
            'ticker': default_ticker,
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
            'use_macd_valley': False,
            'use_macd_peak': False,
            'use_time_filter': False,
            'avoid_after_time': '15:00',
            'show_signals': True,
            'show_reports': True
        }

with st.sidebar:
    st.header("Settings")

    # Get available tickers from alpaca.json
    available_tickers = get_available_tickers()

    # Add option to create new ticker
    ticker_options = available_tickers + ["➕ Create New Ticker..."]

    # Get current ticker from settings, or use first available
    current_ticker = st.session_state.settings.get('ticker', available_tickers[0])

    # Find index of current ticker in available list
    try:
        ticker_index = ticker_options.index(current_ticker)
    except (ValueError, AttributeError):
        ticker_index = 0

    selected_option = st.selectbox(
        "Main ticker",
        ticker_options,
        index=ticker_index,
        help="Select ticker from alpaca.json or create a new one"
    )

    # Handle new ticker creation
    if selected_option == "➕ Create New Ticker...":
        new_ticker = st.text_input(
            "Enter new ticker symbol",
            placeholder="e.g., AAPL, MSFT, etc.",
            help="Enter a valid ticker symbol to create"
        ).upper()

        if new_ticker:
            if new_ticker in available_tickers:
                st.warning(f"⚠️ {new_ticker} already exists in configuration")
                ticker = new_ticker
            else:
                if st.button("✅ Create Ticker", use_container_width=True):
                    # Create new ticker in alpaca.json with default configuration
                    config = load_alpaca_config()
                    if config and 'signal_actions' in config and 'tickers' in config['signal_actions']:
                        config['signal_actions']['tickers'][new_ticker] = {
                            "enabled": False,
                            "default_quantity": 100,
                            "entry": {
                                "actions": [
                                    {"type": "BUY", "ticker": new_ticker, "quantity": 100}
                                ],
                                "description": f"{new_ticker} entry signal - Go long"
                            },
                            "exit_conditions_met": {
                                "actions": [
                                    {"type": "SELL_ALL", "ticker": new_ticker}
                                ],
                                "description": f"{new_ticker} exit conditions met - Close position"
                            },
                            "exit_SL": {
                                "actions": [
                                    {"type": "SELL_ALL", "ticker": new_ticker}
                                ],
                                "description": f"{new_ticker} stop loss hit - Close position"
                            }
                        }

                        try:
                            with open(ALPACA_CONFIG_FILE, 'w') as f:
                                json.dump(config, f, indent=2)
                            st.success(f"✅ Created new ticker {new_ticker} in alpaca.json (disabled)")
                            st.session_state.settings['ticker'] = new_ticker
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create ticker: {e}")
                    else:
                        st.error("Failed to load alpaca.json")
                ticker = current_ticker  # Keep current ticker while creating
        else:
            ticker = current_ticker  # Keep current ticker if no input
    else:
        ticker = selected_option

    st.session_state.settings['ticker'] = ticker

    # Auto-load strategy from alpaca.json when ticker changes
    if 'previous_ticker' not in st.session_state:
        st.session_state.previous_ticker = ticker

    if ticker != st.session_state.previous_ticker:
        # Ticker changed, auto-load strategy from alpaca.json
        alpaca_strategy = load_strategy_from_alpaca(ticker)
        if alpaca_strategy:
            # Update settings from alpaca.json
            entry = alpaca_strategy.get('entry_conditions', {})
            exit_cond = alpaca_strategy.get('exit_conditions', {})
            risk = alpaca_strategy.get('risk_management', {})

            # Entry conditions
            st.session_state.settings['use_rsi'] = entry.get('use_rsi', False)
            st.session_state.settings['rsi_threshold'] = entry.get('rsi_threshold', 30)
            st.session_state.settings['use_ema_cross_up'] = entry.get('use_ema_cross_up', False)
            st.session_state.settings['use_bb_cross_up'] = entry.get('use_bb_cross_up', False)
            st.session_state.settings['use_macd_cross_up'] = entry.get('use_macd_cross_up', False)
            st.session_state.settings['use_price_vs_ema9'] = entry.get('use_price_vs_ema9', False)
            st.session_state.settings['use_price_vs_ema21'] = entry.get('use_price_vs_ema21', False)
            st.session_state.settings['use_macd_threshold'] = entry.get('use_macd_threshold', False)
            st.session_state.settings['macd_threshold'] = entry.get('macd_threshold', 0.1)
            st.session_state.settings['use_macd_valley'] = entry.get('use_macd_valley', False)

            # Exit conditions
            st.session_state.settings['use_rsi_exit'] = exit_cond.get('use_rsi_exit', False)
            st.session_state.settings['rsi_exit_threshold'] = exit_cond.get('rsi_exit_threshold', 70)
            st.session_state.settings['use_ema_cross_down'] = exit_cond.get('use_ema_cross_down', False)
            st.session_state.settings['use_bb_cross_down'] = exit_cond.get('use_bb_cross_down', False)
            st.session_state.settings['use_macd_cross_down'] = exit_cond.get('use_macd_cross_down', False)
            st.session_state.settings['use_price_vs_ema9_exit'] = exit_cond.get('use_price_vs_ema9_exit', False)
            st.session_state.settings['use_price_vs_ema21_exit'] = exit_cond.get('use_price_vs_ema21_exit', False)
            st.session_state.settings['use_macd_peak'] = exit_cond.get('use_macd_peak', False)

            # Risk management
            st.session_state.settings['stop_loss'] = risk.get('stop_loss', 0.02)
            st.session_state.settings['take_profit'] = risk.get('take_profit', 0.03)

            # Interval and period
            st.session_state.settings['interval'] = alpaca_strategy.get('interval', '5m')
            st.session_state.settings['period'] = alpaca_strategy.get('period', '1d')

            # Update previous ticker
            st.session_state.previous_ticker = ticker
            st.rerun()

    # Add ticker enable/disable control
    st.divider()

    current_status = get_ticker_enabled_status(ticker)
    status_emoji = "✅" if current_status else "❌"
    status_text = "Enabled" if current_status else "Disabled"

    st.markdown(f"**Ticker Status:** {status_emoji} {status_text}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Enable", use_container_width=True, disabled=current_status):
            if update_ticker_enabled_status(ticker, True):
                st.success(f"✅ {ticker} enabled in alpaca.json")
                st.rerun()
            else:
                st.error(f"Failed to enable {ticker}")

    with col2:
        if st.button("❌ Disable", use_container_width=True, disabled=not current_status):
            if update_ticker_enabled_status(ticker, False):
                st.warning(f"❌ {ticker} disabled in alpaca.json")
                st.rerun()
            else:
                st.error(f"Failed to disable {ticker}")

    # Add delete ticker button with confirmation
    if 'confirm_delete_ticker' not in st.session_state:
        st.session_state.confirm_delete_ticker = None

    if st.session_state.confirm_delete_ticker == ticker:
        # Show confirmation state
        st.warning(f"⚠️ Are you sure you want to delete {ticker}?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Delete", use_container_width=True, key=f"confirm_del_{ticker}"):
                if delete_ticker(ticker):
                    st.success(f"✅ Deleted {ticker} from alpaca.json")
                    st.session_state.confirm_delete_ticker = None
                    # Switch to first available ticker
                    available_tickers = get_available_tickers()
                    if available_tickers:
                        st.session_state.settings['ticker'] = available_tickers[0]
                        st.session_state.previous_ticker = available_tickers[0]
                    st.rerun()
                else:
                    st.error(f"Failed to delete {ticker}")
                    st.session_state.confirm_delete_ticker = None
        with col2:
            if st.button("❌ Cancel", use_container_width=True, key=f"cancel_del_{ticker}"):
                st.session_state.confirm_delete_ticker = None
                st.rerun()
    else:
        # Show delete button
        if st.button("🗑️ Delete Ticker", use_container_width=True, type="secondary", key=f"delete_{ticker}"):
            st.session_state.confirm_delete_ticker = ticker
            st.rerun()

    st.divider()

    # Add default_quantity control
    current_quantity = get_ticker_default_quantity(ticker)
    new_quantity = st.number_input(
        "Default Quantity",
        min_value=1,
        max_value=10000,
        value=current_quantity,
        step=1,
        help="Number of shares to trade for this ticker"
    )

    if new_quantity != current_quantity:
        if st.button("💾 Update Quantity", use_container_width=True):
            if update_ticker_default_quantity(ticker, new_quantity):
                st.success(f"✅ Updated {ticker} default_quantity to {new_quantity}")
                st.rerun()
            else:
                st.error(f"Failed to update default_quantity for {ticker}")

    st.divider()

    period_options = ["1d", "5d", "2wk", "1mo", "2mo", "3mo", "6mo", "1y"]
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
                         help="RSI must be below threshold before entry")
    st.session_state.settings['use_rsi'] = use_rsi

    rsi_threshold = st.number_input("RSI oversold threshold", min_value=10, max_value=50,
                                    value=st.session_state.settings['rsi_threshold'],
                                    disabled=not use_rsi,
                                    help="Alert when RSI falls below this level")
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

    # Handle backwards compatibility for use_macd_valley
    if 'use_macd_valley' not in st.session_state.settings:
        st.session_state.settings['use_macd_valley'] = False

    use_macd_valley = st.checkbox("MACD Valley (turning up)",
                                   value=st.session_state.settings['use_macd_valley'],
                                   help="Entry when MACD valley is detected (MACD turning up)")
    st.session_state.settings['use_macd_valley'] = use_macd_valley

    use_volume = st.checkbox("Volume Rising", value=st.session_state.settings['use_volume'],
                            help="Current candle volume >= previous candle volume")
    st.session_state.settings['use_volume'] = use_volume

    # Exit Rules
    st.markdown("**Exit Rules (all must be met):**")
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
                                      help="Exit when RSI exceeds this level")
    st.session_state.settings['use_rsi_overbought'] = use_rsi_overbought

    rsi_overbought_threshold = st.number_input("RSI overbought threshold",
                                                min_value=50, max_value=90,
                                                value=st.session_state.settings['rsi_overbought_threshold'],
                                                disabled=not use_rsi_overbought,
                                                help="Exit when RSI rises above this level")
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

    # Handle backwards compatibility for use_macd_peak
    if 'use_macd_peak' not in st.session_state.settings:
        st.session_state.settings['use_macd_peak'] = False

    use_macd_peak = st.checkbox("Exit on MACD Peak (turning down)",
                                 value=st.session_state.settings['use_macd_peak'],
                                 help="Exit when MACD peak is detected (MACD turning down)")
    st.session_state.settings['use_macd_peak'] = use_macd_peak

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
    # Save current settings to alpaca.json before running backtest
    with st.spinner(f"Saving strategy for {ticker} to alpaca.json..."):
        if save_settings_to_alpaca(st.session_state.settings, ticker):
            st.success(f"💾 Saved strategy for {ticker} to alpaca.json", icon="✅")
        else:
            st.warning(f"⚠️ Could not save strategy for {ticker}")

    # ---- Main ticker ----
    with st.spinner(f"Downloading {ticker} data..."):
        # Map period to yfinance compatible values and adjust for better data display
        # yfinance supports: {"1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}

        # When interval is 1d (daily), adjust period to get more historical data
        if interval == "1d":
            # For daily interval, map period differently to get sufficient data
            period_map = {
                "1d": "5d",     # Get 5 days for 1d period
                "5d": "5d",     # Get 5 days for 5d period
                "2wk": "1mo",   # Get 1 month for 2wk period
                "1mo": "3mo",   # Get 3 months for 1mo period (need more data for daily indicators)
                "2mo": "6mo",   # Get 6 months for 2mo period
                "3mo": "1y",    # Get 1 year for 3mo period
                "6mo": "2y",    # Get 2 years for 6mo period
                "1y": "5y"      # Get 5 years for 1y period
            }
        else:
            # For intraday intervals, use existing mapping
            period_map = {
                "1d": "5d",    # Download 5 days for 1d period to get enough data for indicators
                "5d": "5d",
                "2wk": "1mo",  # yfinance doesn't support 2wk, use 1mo instead
                "1mo": "1mo",
                "2mo": "3mo",  # yfinance doesn't support 2mo, use 3mo instead
                "3mo": "3mo",
                "6mo": "6mo",
                "1y": "1y"
            }

        main_period = period_map.get(period, period)

        # Validate interval/period combination based on yfinance limits
        # 1m, 2m, 5m, 15m, 30m: max 60 days
        # 1h, 90m: max 730 days
        intraday_short = ["1m", "2m", "5m", "15m", "30m"]
        intraday_hourly = ["1h", "90m"]

        if interval in intraday_short:
            # For short intervals, limit to 60 days
            if main_period in ["3mo", "6mo", "1y"]:
                st.warning(f"⚠️ {interval} interval only supports up to 60 days of data. Adjusting period from {main_period} to 60d.")
                main_period = "60d"
        elif interval in intraday_hourly:
            # For hourly intervals, limit to 730 days
            if main_period == "1y" and period == "1y":
                # 1y is ok, it's ~365 days
                pass

        # Use extended hours for intraday intervals only
        use_extended_hours = interval not in ["1d", "5d", "1wk", "1mo", "3mo"]
        raw = yf.download(ticker, period=main_period,
                          interval=interval, progress=False, prepost=use_extended_hours)

    if raw.empty:
        st.error(f"No data returned for main ticker. This may happen if the period ({period}) exceeds yfinance limits for the {interval} interval. Try a shorter period.")
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
            macd_above_threshold=macd_above_threshold,
            use_macd_peak=use_macd_peak,
            use_macd_valley=use_macd_valley
        )

        # Filter chart data based on selected period
        if not df1.empty:
            last_time = df1.index[-1]

            if period == "1d":
                # Show only last 26 hours for 1d period (for intraday) or 1 day (for daily)
                if interval == "1d":
                    cutoff_time = last_time - pd.Timedelta(days=1)
                else:
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
            elif period == "6mo":
                # Show only last 6 months (180 days) for 6mo period
                cutoff_time = last_time - pd.Timedelta(days=180)
                df1_display = df1[df1.index >= cutoff_time].copy()
                df5_display = df5[df5.index >= cutoff_time].copy()
            elif period == "1y":
                # Show only last 1 year (365 days) for 1y period
                cutoff_time = last_time - pd.Timedelta(days=365)
                df1_display = df1[df1.index >= cutoff_time].copy()
                df5_display = df5[df5.index >= cutoff_time].copy()
            else:
                df1_display = df1
                df5_display = df5
        else:
            df1_display = df1
            df5_display = df5

        # Chart with entries/exits - single chart with multiple y-axes
        st.subheader(f"{interval} chart with signals")

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
            # Filter for rows that have exit_reason (completed trades)
            completed_trades = trades_df[trades_df["exit_reason"].notna()].copy()

            if not completed_trades.empty:
                # Position signals on RSI chart at fixed positions
                # Entry signals at RSI level 20 (bottom)
                # Exit signals at RSI level 80 (top)

                # Get entry notes - use entry_reason column
                entry_notes = completed_trades["entry_reason"].tolist()

                # Get exit info - use exit_reason column
                exit_info = completed_trades["exit_reason"].tolist()

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
        fig.add_trace(go.Scatter(x=df1_display.index, y=df1_display["rsi"],
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

        # Detect MACD peaks and valleys using argrelextrema (order=3 means look 3 candles left and right)
        # Peak: local maximum (any sign), Valley: local minimum (any sign)
        if len(df5_display) > 6:  # Need at least 7 candles for order=3
            macd_values = df5_display["macd"].values
            peak_indices = argrelextrema(macd_values, np.greater, order=3)[0]
            valley_indices = argrelextrema(macd_values, np.less, order=3)[0]

            # Add peak markers (red triangles pointing down) - all peaks
            if len(peak_indices) > 0:
                peak_times = df5_display.index[peak_indices]
                peak_values = macd_values[peak_indices]
                fig.add_trace(go.Scatter(
                    x=peak_times,
                    y=peak_values,
                    mode='markers',
                    name='MACD Peak',
                    marker=dict(size=10, symbol='triangle-down', color='red'),
                    yaxis='y4',
                    hoverinfo='x+y'
                ))

            # Add valley markers (green triangles pointing up) - all valleys
            if len(valley_indices) > 0:
                valley_times = df5_display.index[valley_indices]
                valley_values = macd_values[valley_indices]
                fig.add_trace(go.Scatter(
                    x=valley_times,
                    y=valley_values,
                    mode='markers',
                    name='MACD Valley',
                    marker=dict(size=10, symbol='triangle-up', color='green'),
                    yaxis='y4',
                    hoverinfo='x+y'
                ))

        # MACD zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, yref='y4')

        # Add shaded regions for extended hours (only for intraday intervals)
        # Data timestamps are now in US Eastern Time (New York)
        # Regular market hours: 9:30 AM to 4:00 PM ET
        # Extended hours: before 9:30 AM or after 4:00 PM ET
        shapes = []
        # Only show extended hours shading for intraday intervals (not for daily data)
        if interval != "1d" and hasattr(df5_display.index, 'hour'):
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

        # Update layout with single x-axis and multiple y-axes using domains
        # Remove gaps by hiding weekends and overnight hours (only for intraday intervals)
        if interval == "1d":
            # For daily data, don't hide any gaps - show continuous chart
            rangebreaks = []
        else:
            # For intraday data, hide both weekends and overnight hours
            rangebreaks = [
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[20, 4], pattern="hour")  # Hide overnight hours (8pm to 4am)
            ]

        # Set x-axis range to match filtered period
        # For better display, use actual data range with small padding
        xaxis_range = None
        if not df5_display.empty and len(df5_display) > 0:
            # Calculate padding based on data range (5% on each side)
            time_range = df5_display.index[-1] - df5_display.index[0]
            if time_range.total_seconds() > 0:
                padding = time_range * 0.05
                xaxis_range = [df5_display.index[0] - padding, df5_display.index[-1] + padding]

        # Configure x-axis tick format based on interval
        if interval == "1d":
            tick_format = '%Y-%m-%d'
            hover_format = '%Y-%m-%d'
        else:
            tick_format = '%H:%M'
            hover_format = '%Y-%m-%d %H:%M'

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
                tickformat=tick_format,
                hoverformat=hover_format,
                rangebreaks=rangebreaks,
                range=xaxis_range
            ),
            # yaxis for Price chart (top 50%)
            yaxis=dict(
                title=f'{ticker} Price',
                domain=[0.50, 1.0]  # 0.50 height
            ),
            # yaxis2 for Volume chart (15%)
            yaxis2=dict(
                title='Volume',
                domain=[0.33, 0.48],  # 0.15 height
                showgrid=False
            ),
            # yaxis3 for RSI chart (15%)
            yaxis3=dict(
                title='RSI',
                domain=[0.16, 0.31],  # 0.15 height
                range=[0, 100]
            ),
            # yaxis4 for MACD chart (16%)
            yaxis4=dict(
                title='MACD',
                domain=[0.0, 0.15],  # 0.16 height (bottom panel)
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
                # Filter for rows that have exit_reason (completed trades) for display
                display_trades = trades_df[trades_df["exit_reason"].notna()].copy()

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

