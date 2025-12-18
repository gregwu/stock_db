import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go
import json
import os
import subprocess
import signal
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from scipy.signal import argrelextrema
from trading_config import is_market_hours

# Load environment variables
load_dotenv()

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

def get_limit_order_slippage_pct():
    """Get limit order slippage percentage from alpaca.json"""
    config = load_alpaca_config()
    if config:
        trading = config.get('strategy', {}).get('trading', {})
        return trading.get('limit_order_slippage_pct', 2.0)
    return 2.0  # Default fallback

def get_account_status():
    """Get trading account status from alpaca.json configuration"""
    config = load_alpaca_config()
    if config:
        use_paper = config.get('strategy', {}).get('trading', {}).get('use_paper', True)
        return "PAPER TRADING" if use_paper else "⚠️  LIVE TRADING ⚠️"
    return "⚠️  CONFIG ERROR"

def show_account_status():
    """Display account status in Streamlit"""
    account_type = get_account_status()
    
    if "PAPER" in account_type:
        st.success(f"🟢 {account_type}")
    else:
        st.error(f"🔴 {account_type} - REAL MONEY!")

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

            # Override period and interval if present in ticker strategy
            if 'period' in ticker_strategy:
                merged_strategy['period'] = ticker_strategy['period']

            if 'interval' in ticker_strategy:
                merged_strategy['interval'] = ticker_strategy['interval']

            if 'interval_2' in ticker_strategy:
                merged_strategy['interval_2'] = ticker_strategy['interval_2']

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

@st.cache_data(ttl=600)
def fetch_similarity_data(ticker):
    """Fetch 60d of 5m data for cosine similarity analysis."""
    try:
        df = yf.download(
            ticker,
            period="60d",
            interval="5m",
            progress=False,
            prepost=True
        )
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if hasattr(df.index, 'tz'):
            if df.index.tz is not None:
                df.index = df.index.tz_convert('America/New_York')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        return df
    except Exception:
        return pd.DataFrame()

def normalize_pattern(values):
    """Zero-mean and unit-normalize a price sequence to compare shapes."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    arr = arr - np.nanmean(arr)
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return None
    return arr / norm

def cosine_similarity_score(a_values, b_values):
    """Compute cosine similarity between two equally-sized vectors."""
    if len(a_values) != len(b_values) or len(a_values) == 0:
        return np.nan
    a_norm = normalize_pattern(a_values)
    b_norm = normalize_pattern(b_values)
    if a_norm is None or b_norm is None:
        return np.nan
    return float(np.dot(a_norm, b_norm))

def compute_pattern_matches(df, reference_bars=316, min_similarity=0.85, top_n=5):
    """
    Given a 5m dataframe, compare the most recent reference_bars of data
    against prior history using cosine similarity.
    """
    result = {
        "reference": None,
        "matches": pd.DataFrame(),
        "best_match_range": None
    }

    if df.empty or "Close" not in df.columns:
        return result

    working = df.copy().dropna(subset=["Close"])
    if working.empty:
        return result

    if len(working) <= reference_bars:
        return result

    reference_df = working.iloc[-reference_bars:].copy()
    history_df = working.iloc[:-reference_bars].copy()

    if reference_df.empty or len(history_df) <= reference_bars:
        return result

    window_len = reference_bars
    reference_values = reference_df["Close"].values
    ref_norm = normalize_pattern(reference_values)
    if ref_norm is None:
        return result

    scores = []
    history_close = history_df["Close"]
    ref_start_time = reference_df.index[0]

    for start_idx in range(0, len(history_close) - window_len + 1):
        window_vals = history_close.iloc[start_idx:start_idx + window_len].values
        sim = cosine_similarity_score(reference_values, window_vals)
        if np.isnan(sim):
            continue
        start_time = history_close.index[start_idx]
        end_time = history_close.index[start_idx + window_len - 1]
        if sim >= min_similarity:
            scores.append({
                "match_start": start_time,
                "match_end": end_time,
                "similarity": sim,
                "bars": window_len,
                "days_ago": (ref_start_time - start_time).days,
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            })

    if not scores:
        return result

    matches_df = (
        pd.DataFrame(scores)
        .sort_values("similarity", ascending=False)
    )
    if matches_df.empty:
        return result

    ref_start_ts = reference_df.index[0]
    matches_df = matches_df[matches_df["match_start"] != ref_start_ts]

    if len(reference_df.index) >= 2:
        bar_duration = reference_df.index[1] - reference_df.index[0]
    else:
        bar_duration = pd.Timedelta(minutes=5)
    min_spacing = bar_duration * window_len

    filtered_rows = []
    seen_starts = []
    for _, row in matches_df.iterrows():
        start_ts = row["match_start"]
        if all(abs(start_ts - seen) >= min_spacing for seen in seen_starts):
            filtered_rows.append(row)
            seen_starts.append(start_ts)
            if len(filtered_rows) >= top_n:
                break

    matches_df = pd.DataFrame(filtered_rows)
    if matches_df.empty:
        return result
    result["reference"] = reference_df
    result["matches"] = matches_df
    if not matches_df.empty:
        result["best_match_range"] = (
            matches_df.iloc[0]["match_start"],
            matches_df.iloc[0]["match_end"]
        )
    return result

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

def settings_have_changed(current_settings, loaded_settings):
    """Check if current settings differ from loaded settings"""
    # List of keys to compare (include period and interval)
    # Use UI field names (not alpaca.json field names)
    strategy_keys = [
        'period', 'interval', 'interval_2',  # Data settings
        'use_rsi', 'rsi_threshold', 'use_ema_cross_up', 'use_bb_cross_up',
        'use_bb_width', 'bb_width_threshold',
        'use_macd_cross_up', 'use_primary_macd_cross_up',
        'use_ema', 'use_price_above_ema21',
        'use_macd_threshold', 'macd_threshold',
        'use_macd_below_threshold', 'macd_below_threshold',
        'use_primary_macd_below_threshold', 'primary_macd_below_threshold',
        'use_macd_valley', 'use_primary_macd_valley',
        'use_ema21_slope_entry', 'ema21_slope_entry_threshold',
        'use_rsi_overbought', 'rsi_overbought_threshold', 'use_ema_cross_down',
        'use_bb_cross_down', 'use_bb_width_exit', 'bb_width_exit_threshold',
        'use_macd_cross_down', 'use_primary_macd_cross_down',
        'use_price_below_ema9',
        'use_price_below_ema21', 'use_macd_peak', 'use_primary_macd_peak',
        'stop_loss_pct', 'take_profit_pct', 'use_stop_loss', 'use_take_profit',
        'use_volume',
        'use_macd_above_threshold', 'macd_above_threshold',
        'use_primary_macd_above_threshold', 'primary_macd_above_threshold',
        'use_ema21_slope_exit', 'ema21_slope_exit_threshold'
    ]

    for key in strategy_keys:
        if current_settings.get(key) != loaded_settings.get(key):
            return True
    return False

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
            'use_bb_width': settings.get('use_bb_width', False),
            'bb_width_threshold': settings.get('bb_width_threshold', 5.0),
            'use_macd_cross_up': settings.get('use_macd_cross_up', False),
            'use_primary_macd_cross_up': settings.get('use_primary_macd_cross_up', False),
            'use_price_vs_ema9': settings.get('use_ema', False),  # Map UI field to alpaca.json field
            'use_price_vs_ema21': settings.get('use_price_above_ema21', False),  # Map UI field to alpaca.json field
            # Map UI field use_macd_below_threshold to alpaca.json field use_macd_threshold
            'use_macd_threshold': settings.get('use_macd_below_threshold', False),
            'macd_threshold': settings.get('macd_below_threshold', 0.1),
            'use_primary_macd_below_threshold': settings.get('use_primary_macd_below_threshold', False),
            'primary_macd_below_threshold': settings.get('primary_macd_below_threshold', 0.1),
            'use_macd_valley': settings.get('use_macd_valley', False),
            'use_primary_macd_valley': settings.get('use_primary_macd_valley', False),
            'use_ema': settings.get('use_ema', False),  # Also keep this for backward compatibility
            'use_volume': settings.get('use_volume', False),
            'use_ema21_slope_entry': settings.get('use_ema21_slope_entry', False),
            'ema21_slope_entry_threshold': settings.get('ema21_slope_entry_threshold', 0.0)
        }

        exit_conditions = {
            'use_rsi_exit': settings.get('use_rsi_overbought', False),  # Map UI field to alpaca.json field
            'rsi_exit_threshold': settings.get('rsi_overbought_threshold', 70),  # Map UI field to alpaca.json field
            'use_ema_cross_down': settings.get('use_ema_cross_down', False),
            'use_bb_cross_down': settings.get('use_bb_cross_down', False),
            'use_bb_width_exit': settings.get('use_bb_width_exit', False),
            'bb_width_exit_threshold': settings.get('bb_width_exit_threshold', 10.0),
            'use_macd_cross_down': settings.get('use_macd_cross_down', False),
            'use_primary_macd_cross_down': settings.get('use_primary_macd_cross_down', False),
            'use_price_vs_ema9_exit': settings.get('use_price_below_ema9', False),  # Map UI field to alpaca.json field
            'use_price_vs_ema21_exit': settings.get('use_price_below_ema21', False),  # Map UI field to alpaca.json field
            'use_macd_peak': settings.get('use_macd_peak', False),
            'use_primary_macd_peak': settings.get('use_primary_macd_peak', False),
            'use_macd_above_threshold': settings.get('use_macd_above_threshold', False),
            'macd_above_threshold': settings.get('macd_above_threshold', 0.0),
            'use_primary_macd_above_threshold': settings.get('use_primary_macd_above_threshold', False),
            'primary_macd_above_threshold': settings.get('primary_macd_above_threshold', 0.0),
            'use_ema21_slope_exit': settings.get('use_ema21_slope_exit', False),
            'ema21_slope_exit_threshold': settings.get('ema21_slope_exit_threshold', 0.0)
        }

        risk_management = {
            'stop_loss': settings.get('stop_loss_pct', 2.0) / 100,  # Convert percentage to decimal
            'take_profit': settings.get('take_profit_pct', 3.0) / 100,  # Convert percentage to decimal
            'use_stop_loss': settings.get('use_stop_loss', True),
            'use_take_profit': settings.get('use_take_profit', False)
        }

        # Prepare ticker strategy
        ticker_strategy = {
            'interval': settings.get('interval', '5m'),
            'interval_2': settings.get('interval_2', '1h'),
            'period': settings.get('period', '1d'),
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

def update_trading_settings(max_buy_slippage, max_sell_slippage, limit_order_slippage, avoid_extended_hours):
    """Update trading settings in alpaca.json"""
    try:
        config = load_alpaca_config()
        if not config:
            return False

        if 'strategy' not in config:
            config['strategy'] = {}
        if 'trading' not in config['strategy']:
            config['strategy']['trading'] = {}

        # Update trading settings
        config['strategy']['trading']['max_buy_slippage_pct'] = max_buy_slippage
        config['strategy']['trading']['max_sell_slippage_pct'] = max_sell_slippage
        config['strategy']['trading']['limit_order_slippage_pct'] = limit_order_slippage
        config['strategy']['trading']['avoid_extended_hours'] = avoid_extended_hours

        # Save back to file
        with open(ALPACA_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error updating trading settings: {e}")
        return False

def get_trading_settings():
    """Get trading settings from alpaca.json"""
    config = load_alpaca_config()
    if not config or 'strategy' not in config or 'trading' not in config['strategy']:
        return {
            'max_buy_slippage_pct': 0.9,
            'max_sell_slippage_pct': 0.9,
            'limit_order_slippage_pct': 2.0,
            'avoid_extended_hours': False
        }
    return config['strategy']['trading']

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

# ---------- Manual trading functions ----------

def get_alpaca_api():
    """Initialize and return Alpaca API instance"""
    try:
        from alpaca_wrapper import AlpacaAPI
        from dotenv import load_dotenv

        load_dotenv()

        # Get trading mode from config
        config = load_alpaca_config()
        use_paper = True  # Default to paper
        if config and 'strategy' in config and 'trading' in config['strategy']:
            use_paper = config['strategy']['trading'].get('use_paper', True)

        api = AlpacaAPI(paper=use_paper)
        if api.login():
            return api
        return None
    except Exception as e:
        st.error(f"Failed to initialize Alpaca API: {e}")
        return None

def get_current_price(ticker):
    """Get current market price for a ticker"""
    try:
        api = get_alpaca_api()
        if not api:
            return None

        quote = api.quote(ticker)
        if quote and 'last' in quote:
            return quote['last']
        return None
    except Exception as e:
        st.error(f"Failed to get price for {ticker}: {e}")
        return None

def get_recent_filled_trades(ticker, limit=200):
    """Fetch recent filled trades for a ticker from Alpaca"""
    try:
        api = get_alpaca_api()
        if not api:
            return []

        orders = api.get_recent_filled_orders(limit=limit)
        fills = []
        for order in orders:
            if order.get('symbol') != ticker:
                continue

            filled_at = order.get('filled_at')
            if not filled_at:
                continue

            filled_ts = pd.Timestamp(filled_at)
            if filled_ts.tzinfo is None:
                filled_ts = filled_ts.tz_localize('UTC')
            filled_ts = filled_ts.tz_convert('America/New_York')

            price_val = order.get('filled_avg_price') or order.get('limit_price')
            try:
                price_val = float(price_val) if price_val is not None else None
            except Exception:
                price_val = None

            qty_val = order.get('filled_qty') or order.get('qty')
            try:
                qty_val = float(qty_val) if qty_val is not None else 0.0
            except Exception:
                qty_val = 0.0

            fills.append({
                'time': filled_ts,
                'price': price_val,
                'qty': qty_val,
                'side': str(order.get('side', '')).upper()
            })

        fills.sort(key=lambda x: x['time'])
        return fills
    except Exception as e:
        st.warning(f"Failed to load filled trades from Alpaca: {e}")
        return []

def calculate_actual_return_pct(fills):
    """Calculate realized return percentage from filled trades"""
    realized_pl = 0.0
    invested_capital = 0.0
    position_qty = 0.0
    avg_price = 0.0

    for trade in fills:
        price = trade.get('price')
        qty = trade.get('qty', 0.0) or 0.0
        side = str(trade.get('side', '')).upper()

        if price is None or qty <= 0:
            continue

        if side == 'BUY':
            total_cost = avg_price * position_qty + price * qty
            position_qty += qty
            if position_qty > 0:
                avg_price = total_cost / position_qty
        elif side == 'SELL':
            if position_qty <= 0:
                continue
            sell_qty = min(qty, position_qty)
            realized_pl += (price - avg_price) * sell_qty
            invested_capital += avg_price * sell_qty
            position_qty -= sell_qty
            if position_qty == 0:
                avg_price = 0.0

    if invested_capital > 0:
        return realized_pl / invested_capital * 100
    return None

def send_manual_order_email(subject, message):
    """
    Send email notification for manual orders
    """
    try:
        gmail_address = os.getenv('GMAIL_ADDRESS')
        gmail_app_password = os.getenv('GMAIL_APP_PASSWORD')

        if not gmail_address or not gmail_app_password:
            return False

        msg = MIMEMultipart()
        msg['From'] = gmail_address
        msg['To'] = gmail_address
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_address, gmail_app_password)
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def place_manual_buy(ticker, quantity, order_type="AUTO", limit_price=None):
    """
    Place a manual buy order

    Args:
        ticker: Stock symbol
        quantity: Number of shares
        order_type: "MKT" for market, "LMT" for limit, "AUTO" for automatic selection
        limit_price: Limit price (required if order_type is "LMT")
    """
    try:
        api = get_alpaca_api()
        if not api:
            return False, "Failed to connect to Alpaca API"

        # Get current price
        price = get_current_price(ticker)
        if not price:
            return False, f"Failed to get current price for {ticker}"

        # Determine order type and whether we're in extended hours
        in_market_hours = is_market_hours()
        limit_slippage_pct = get_limit_order_slippage_pct()

        if order_type == "AUTO":
            # Auto-select based on market hours
            if in_market_hours:
                order_type = "MKT"
            else:
                order_type = "LMT"
                limit_price = round(price * (1 + limit_slippage_pct / 100), 2)

        # Place order
        if order_type == "MKT":
            order = api.place_order(
                ticker=ticker,
                qty=quantity,
                action="BUY",
                order_type="MKT",
                extended_hours=not in_market_hours  # Only enable extended_hours if NOT in regular hours
            )
            order_type_desc = "MARKET"
        else:  # LMT
            if limit_price is None:
                # Default limit price with slippage
                limit_price = round(price * (1 + limit_slippage_pct / 100), 2)

            order = api.place_order(
                ticker=ticker,
                qty=quantity,
                action="BUY",
                order_type="LMT",
                price=limit_price,
                extended_hours=not in_market_hours  # Only enable extended_hours if NOT in regular hours
            )
            order_type_desc = f"LIMIT @ ${limit_price:.2f}"

        if order and 'order_id' in order:
            # Send email notification
            subject = f"📊 MANUAL BUY ORDER - {ticker}"
            email_message = f"""Manual Buy Order Placed

Ticker: {ticker}
Quantity: {quantity} shares
Current Market Price: ${price:.2f}
Order Type: {order_type_desc}
Order ID: {order['order_id']}
Status: {order.get('status', 'Submitted')}

Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This order was placed manually through the Streamlit UI.
"""
            send_manual_order_email(subject, email_message)

            return True, f"✅ BUY order placed: {quantity} shares of {ticker} ({order_type_desc})\nOrder ID: {order['order_id']}"
        elif order and 'error' in order:
            # API returned an error
            return False, f"❌ API Error: {order['error_type']}: {order['error']}\nAttempted: {order_type} order for {quantity} shares of {ticker}"
        else:
            # Unexpected response
            return False, f"Order placement failed - unexpected response. Check logs for details.\nAttempted: {order_type} order for {quantity} shares of {ticker}"

    except Exception as e:
        return False, f"Failed to place buy order: {e}"

def place_manual_sell(ticker, quantity, order_type="AUTO", limit_price=None):
    """
    Place a manual sell order

    Args:
        ticker: Stock symbol
        quantity: Number of shares
        order_type: "MKT" for market, "LMT" for limit, "AUTO" for automatic selection
        limit_price: Limit price (required if order_type is "LMT")
    """
    try:
        api = get_alpaca_api()
        if not api:
            return False, "Failed to connect to Alpaca API"

        # Check if we have this position
        positions = api.get_positions()
        position = next((p for p in positions if p['symbol'] == ticker), None)

        if not position:
            return False, f"No position in {ticker} to sell"

        available_qty = int(position['qty'])
        if quantity > available_qty:
            return False, f"Insufficient shares. You have {available_qty} shares of {ticker}"

        # Get current price
        price = get_current_price(ticker)
        if not price:
            return False, f"Failed to get current price for {ticker}"

        # Determine order type and whether we're in extended hours
        in_market_hours = is_market_hours()
        limit_slippage_pct = get_limit_order_slippage_pct()

        if order_type == "AUTO":
            # Auto-select based on market hours
            if in_market_hours:
                order_type = "MKT"
            else:
                order_type = "LMT"
                limit_price = round(price * (1 - limit_slippage_pct / 100), 2)

        # Place order
        if order_type == "MKT":
            order = api.place_order(
                ticker=ticker,
                qty=quantity,
                action="SELL",
                order_type="MKT",
                extended_hours=not in_market_hours  # Only enable extended_hours if NOT in regular hours
            )
            order_type_desc = "MARKET"
        else:  # LMT
            if limit_price is None:
                # Default limit price with slippage
                limit_price = round(price * (1 - limit_slippage_pct / 100), 2)

            order = api.place_order(
                ticker=ticker,
                qty=quantity,
                action="SELL",
                order_type="LMT",
                price=limit_price,
                extended_hours=not in_market_hours  # Only enable extended_hours if NOT in regular hours
            )
            order_type_desc = f"LIMIT @ ${limit_price:.2f}"

        if order and 'order_id' in order:
            # Calculate P&L if position exists
            pnl_info = ""
            if position:
                avg_entry = float(position['avg_entry_price'])
                pnl = (price - avg_entry) * quantity
                pnl_pct = ((price - avg_entry) / avg_entry) * 100
                pnl_info = f"\nEntry Price: ${avg_entry:.2f}\nP&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"

            # Send email notification
            subject = f"📊 MANUAL SELL ORDER - {ticker}"
            email_message = f"""Manual Sell Order Placed

Ticker: {ticker}
Quantity: {quantity} shares
Current Market Price: ${price:.2f}{pnl_info}
Order Type: {order_type_desc}
Order ID: {order['order_id']}
Status: {order.get('status', 'Submitted')}

Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This order was placed manually through the Streamlit UI.
"""
            send_manual_order_email(subject, email_message)

            return True, f"✅ SELL order placed: {quantity} shares of {ticker} ({order_type_desc})\nOrder ID: {order['order_id']}"
        elif order and 'error' in order:
            # API returned an error
            return False, f"❌ API Error: {order['error_type']}: {order['error']}\nAttempted: {order_type} order for {quantity} shares of {ticker}"
        else:
            # Unexpected response
            return False, f"Order placement failed - unexpected response. Check logs for details.\nAttempted: {order_type} order for {quantity} shares of {ticker}"

    except Exception as e:
        return False, f"Failed to place sell order: {e}"

def get_account_info():
    """Get Alpaca account information"""
    try:
        api = get_alpaca_api()
        if not api:
            return None

        account = api.get_account()
        if account:
            return {
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity)
            }
        return None
    except Exception as e:
        st.error(f"Failed to get account info: {e}")
        return None

def get_positions():
    """Get current positions"""
    try:
        api = get_alpaca_api()
        if not api:
            return []

        return api.get_positions()
    except Exception as e:
        st.error(f"Failed to get positions: {e}")
        return []

# ---------- Bot control functions ----------

def get_bot_status():
    """Check if alpaca_trader.py is running"""
    pid_file = ".alpaca_trader.pid"

    if not os.path.exists(pid_file):
        return False, None

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Check if process is running
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
            return True, pid
        except OSError:
            # Process doesn't exist, remove stale PID file
            os.remove(pid_file)
            return False, None
    except Exception:
        return False, None

def start_bot():
    """Start alpaca_trader.py in background"""
    try:
        # Check if already running
        is_running, pid = get_bot_status()
        if is_running:
            return False, f"Bot already running (PID: {pid})"

        # Start bot in background
        process = subprocess.Popen(
            ['python3', 'alpaca_trader.py'],
            stdout=open('alpaca_trader.out', 'w'),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # Create new process group
        )

        # Save PID
        with open('.alpaca_trader.pid', 'w') as f:
            f.write(str(process.pid))

        return True, f"Bot started successfully (PID: {process.pid})"
    except Exception as e:
        return False, f"Failed to start bot: {e}"

def stop_bot():
    """Stop alpaca_trader.py"""
    try:
        is_running, pid = get_bot_status()

        if not is_running:
            return False, "Bot is not running"

        # Try graceful shutdown first
        try:
            os.kill(pid, signal.SIGTERM)

            # Wait a bit for graceful shutdown
            import time
            time.sleep(2)

            # Check if still running
            try:
                os.kill(pid, 0)
                # Still running, force kill
                os.kill(pid, signal.SIGKILL)
            except OSError:
                # Process stopped
                pass
        except Exception as e:
            return False, f"Failed to stop bot: {e}"

        # Remove PID file
        if os.path.exists('.alpaca_trader.pid'):
            os.remove('.alpaca_trader.pid')

        return True, "Bot stopped successfully"
    except Exception as e:
        return False, f"Failed to stop bot: {e}"

def restart_bot():
    """Restart alpaca_trader.py"""
    # Stop first
    success, msg = stop_bot()
    if not success and "not running" not in msg:
        return False, msg

    # Wait a bit
    import time
    time.sleep(1)

    # Start
    return start_bot()

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

def compute_macd_features_df(df):
    """Return MACD, signal, histogram, and peak/valley flags for the provided price DataFrame."""
    result = pd.DataFrame(index=df.index)
    if df.empty or "Close" not in df.columns:
        return result

    macd_line, signal_line, histogram = macd(df["Close"])
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_hist"] = histogram
    result["macd_peak"] = False
    result["macd_valley"] = False

    if len(result) > 6:
        macd_values = result["macd"].values
        peak_indices = argrelextrema(macd_values, np.greater, order=3)[0]
        valley_indices = argrelextrema(macd_values, np.less, order=3)[0]

        for idx in peak_indices:
            result.iloc[idx, result.columns.get_loc("macd_peak")] = True
        for idx in valley_indices:
            result.iloc[idx, result.columns.get_loc("macd_valley")] = True

    return result

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
                    use_rsi=True,
                    rsi_threshold=30,
                    use_bb_cross_up=False,
                    use_bb_width=False,
                    bb_width_threshold=5.0,
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
                    use_bb_width_exit=False,
                    bb_width_exit_threshold=10.0,
                    use_macd_cross_up=False,
                    use_primary_macd_cross_up=False,
                    use_macd_cross_down=False,
                    use_primary_macd_cross_down=False,
                    use_price_above_ema21=False,
                    use_price_below_ema21=False,
                    use_macd_below_threshold=False,
                    macd_below_threshold=0.0,
                    use_primary_macd_below_threshold=False,
                    primary_macd_below_threshold=0.0,
                    use_macd_above_threshold=False,
                    macd_above_threshold=0.0,
                    use_primary_macd_above_threshold=False,
                    primary_macd_above_threshold=0.0,
                    use_macd_peak=False,
                    use_primary_macd_peak=False,
                    use_macd_valley=False,
                    use_primary_macd_valley=False,
                    use_ema21_slope_entry=False,
                    ema21_slope_entry_threshold=0.0,
                    use_ema21_slope_exit=False,
                    ema21_slope_exit_threshold=0.0,
                    avoid_extended_hours=False):
    """
    Apply Greg's rules to a single symbol.
    Returns:
        df1 (original interval data with RSI),
        df5 (same as df1, kept for backward compatibility),
        trades_df,
        logs_df
    """
    logs = []
    extended_hours_skipped_count = 0  # Track entries skipped due to extended hours

    # Calculate RSI on the selected interval data
    df1 = df1.copy()
    df1["rsi"] = rsi(df1["Close"], 14)

    # Use the input data directly (no resampling)
    # df5 is kept for backward compatibility but points to same data
    df5 = df1.copy()

    # Calculate EMA9, EMA21 and BB(20,2) on selected interval
    df5["ema9"] = ema(df5["Close"], 9)
    df5["ema21"] = ema(df5["Close"], 21)

    # Calculate EMA21 slope (percentage change from previous bar)
    df5["ema21_slope"] = ((df5["ema21"] - df5["ema21"].shift(1)) / df5["ema21"].shift(1) * 100).fillna(0)

    bb_mid, bb_up, bb_low = bollinger_bands(df5["Close"], 20, 2)
    df5["bb_mid"] = bb_mid
    df5["bb_up"] = bb_up
    df5["bb_low"] = bb_low
    # Calculate BB width (percentage of price)
    df5["bb_width"] = ((bb_up - bb_low) / bb_mid * 100).fillna(0)

    macd_features = compute_macd_features_df(df5)
    for col in macd_features.columns:
        df5[col] = macd_features[col]
    for col in ["macd", "macd_signal", "macd_hist"]:
        if col not in df5.columns:
            df5[col] = np.nan
    for col in ["macd_peak", "macd_valley"]:
        if col not in df5.columns:
            df5[col] = False

    # RSI is now calculated on the selected interval
    df5["rsi_last"] = df5["rsi"]

    # Calculate EMA crossovers (for entry and exit detection)
    df5["ema9_prev"] = df5["ema9"].shift(1)
    df5["ema21_prev"] = df5["ema21"].shift(1)

    # Calculate previous MACD values for crossover detection
    df5["macd_prev"] = df5["macd"].shift(1)
    df5["macd_signal_prev"] = df5["macd_signal"].shift(1)
    if "primary_macd" in df5.columns and "primary_macd_prev" not in df5.columns:
        df5["primary_macd_prev"] = df5["primary_macd"].shift(1)
    if "primary_macd_signal" in df5.columns and "primary_macd_signal_prev" not in df5.columns:
        df5["primary_macd_signal_prev"] = df5["primary_macd_signal"].shift(1)
    if "primary_macd_peak" not in df5.columns:
        df5["primary_macd_peak"] = False
    if "primary_macd_valley" not in df5.columns:
        df5["primary_macd_valley"] = False

    # Calculate previous close for BB crossover detection
    df5["close_prev"] = df5["Close"].shift(1)

    trades = []
    position = None
    setup_active = False
    last_exit_price = None  # Track last exit price for price drop check
    last_exit_time = None  # Track last exit time for timeout reset

    for t, row in df5.iterrows():
        # Check if we should avoid extended hours trading
        if avoid_extended_hours:
            # Regular market hours: 9:30 AM - 4:00 PM ET
            hour = t.hour
            minute = t.minute
            time_minutes = hour * 60 + minute
            market_open_minutes = 9 * 60 + 30  # 9:30 AM
            market_close_minutes = 16 * 60  # 4:00 PM

            # Skip if outside regular hours (only for entries, exits can still happen)
            is_regular_hours = market_open_minutes <= time_minutes < market_close_minutes
        else:
            is_regular_hours = True  # Allow all hours if not avoiding extended

        close = row["Close"]
        close_prev = row["close_prev"]
        ema9 = row["ema9"]
        ema21 = row["ema21"]
        ema9_prev = row["ema9_prev"]
        ema21_prev = row["ema21_prev"]
        ema21_slope = row["ema21_slope"]
        bb_low_v = row["bb_low"]
        bb_up_v = row["bb_up"]
        bb_width_val = row["bb_width"]
        rsi_last = row["rsi_last"]
        vol = row["Volume"]
        prev_vol = df5["Volume"].shift(1).loc[t]
        macd_val = row["macd"]
        macd_signal_val = row["macd_signal"]
        macd_prev = row["macd_prev"]
        macd_signal_prev = row["macd_signal_prev"]
        primary_macd_val = row.get("primary_macd", np.nan)
        primary_macd_signal_val = row.get("primary_macd_signal", np.nan)
        primary_macd_prev = row.get("primary_macd_prev", np.nan)
        primary_macd_signal_prev = row.get("primary_macd_signal_prev", np.nan)

        # Detect EMA crossovers
        ema_cross_up = (ema9_prev is not np.nan and ema21_prev is not np.nan and
                        ema9 is not np.nan and ema21 is not np.nan and
                        ema9_prev <= ema21_prev and ema9 > ema21)
        ema_cross_down = (ema9_prev is not np.nan and ema21_prev is not np.nan and
                          ema9 is not np.nan and ema21 is not np.nan and
                          ema9_prev >= ema21_prev and ema9 < ema21)

        # Detect BB crossovers
        # BB cross up: price crosses BELOW lower band (entry - oversold)
        bb_cross_up = (close_prev is not np.nan and bb_low_v is not np.nan and
                       close is not np.nan and
                       close_prev >= bb_low_v and close < bb_low_v)
        # BB cross down: price crosses ABOVE upper band (exit - overbought)
        bb_cross_down = (close_prev is not np.nan and bb_up_v is not np.nan and
                         close is not np.nan and
                         close_prev <= bb_up_v and close > bb_up_v)

        # Detect MACD crossovers
        macd_cross_up = (macd_prev is not np.nan and macd_signal_prev is not np.nan and
                        macd_val is not np.nan and macd_signal_val is not np.nan and
                        macd_prev <= macd_signal_prev and macd_val > macd_signal_val)
        macd_cross_down = (macd_prev is not np.nan and macd_signal_prev is not np.nan and
                          macd_val is not np.nan and macd_signal_val is not np.nan and
                          macd_prev >= macd_signal_prev and macd_val < macd_signal_val)
        primary_macd_cross_up = (primary_macd_prev is not np.nan and primary_macd_signal_prev is not np.nan and
                                 primary_macd_val is not np.nan and primary_macd_signal_val is not np.nan and
                                 primary_macd_prev <= primary_macd_signal_prev and primary_macd_val > primary_macd_signal_val)
        primary_macd_cross_down = (primary_macd_prev is not np.nan and primary_macd_signal_prev is not np.nan and
                                   primary_macd_val is not np.nan and primary_macd_signal_val is not np.nan and
                                   primary_macd_prev >= primary_macd_signal_prev and primary_macd_val < primary_macd_signal_val)

        # Detect MACD peaks and valleys
        is_macd_peak = row.get("macd_peak", False)
        is_macd_valley = row.get("macd_valley", False)
        is_primary_macd_peak = row.get("primary_macd_peak", False)
        is_primary_macd_valley = row.get("primary_macd_valley", False)

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
                last_exit_price = exit_price  # Save exit price for price drop check
                last_exit_time = t  # Save exit time for timeout reset
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
                last_exit_price = exit_price  # Save exit price for price drop check
                last_exit_time = t  # Save exit time for timeout reset
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
            if use_bb_width_exit:
                exit_conditions.append(bb_width_val is not np.nan and bb_width_val > bb_width_exit_threshold)
            if use_macd_cross_down:
                exit_conditions.append(macd_cross_down)
            if use_primary_macd_cross_down:
                exit_conditions.append(primary_macd_cross_down)
            if use_price_below_ema21:
                exit_conditions.append(ema21 is not np.nan and close < ema21)
            if use_macd_above_threshold:
                exit_conditions.append(macd_val is not np.nan and macd_val > macd_above_threshold)
            if use_macd_peak:
                exit_conditions.append(is_macd_peak)
            if use_primary_macd_above_threshold:
                exit_conditions.append(primary_macd_val is not np.nan and primary_macd_val > primary_macd_above_threshold)
            if use_primary_macd_peak:
                exit_conditions.append(is_primary_macd_peak)
            if use_ema21_slope_exit:
                exit_conditions.append(ema21_slope is not np.nan and ema21_slope < ema21_slope_exit_threshold)

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
                    exit_note_parts.append(f"Price crossed above BB upper (Close={close:.2f}, BB Upper={bb_up_v:.2f})")
                if use_bb_width_exit:
                    exit_note_parts.append(f"BB width > {bb_width_exit_threshold}% (Width={bb_width_val:.2f}%)")
                if use_macd_cross_down:
                    exit_note_parts.append(f"MACD crossed below signal (MACD={macd_val:.4f}, Signal={macd_signal_val:.4f})")
                if use_primary_macd_cross_down:
                    exit_note_parts.append(f"Primary MACD crossed below signal (MACD={primary_macd_val:.4f}, Signal={primary_macd_signal_val:.4f})")
                if use_price_below_ema21:
                    exit_note_parts.append(f"Price < EMA21 (Close={close:.2f}, EMA21={ema21:.2f})")
                if use_macd_above_threshold:
                    exit_note_parts.append(f"MACD > {macd_above_threshold} (MACD={macd_val:.4f})")
                if use_macd_peak:
                    exit_note_parts.append(f"MACD peak (MACD={macd_val:.4f})")
                if use_primary_macd_above_threshold:
                    exit_note_parts.append(f"Primary MACD > {primary_macd_above_threshold} (MACD={primary_macd_val:.4f})")
                if use_primary_macd_peak:
                    exit_note_parts.append(f"Primary MACD peak (MACD={primary_macd_val:.4f})")
                if use_ema21_slope_exit:
                    exit_note_parts.append(f"EMA21 slope < {ema21_slope_exit_threshold}% (Slope={ema21_slope:.4f}%)")

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
                last_exit_price = exit_price  # Save exit price for price drop check
                last_exit_time = t  # Save exit time for timeout reset
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

                if use_rsi:
                    conditions.append(rsi_last < rsi_threshold)
                if use_ema_cross_up:
                    conditions.append(ema_cross_up)
                if use_bb_cross_up:
                    conditions.append(bb_cross_up)
                if use_bb_width:
                    conditions.append(bb_width_val is not np.nan and bb_width_val > bb_width_threshold)
                if use_macd_cross_up:
                    conditions.append(macd_cross_up)
                if use_primary_macd_cross_up:
                    conditions.append(primary_macd_cross_up)
                if use_ema:
                    conditions.append(ema9 is not np.nan and close > ema9)
                if use_price_above_ema21:
                    conditions.append(ema21 is not np.nan and close > ema21)
                if use_macd_below_threshold:
                    conditions.append(macd_val is not np.nan and macd_val < macd_below_threshold)
                if use_primary_macd_below_threshold:
                    conditions.append(primary_macd_val is not np.nan and primary_macd_val < primary_macd_below_threshold)
                if use_macd_valley:
                    conditions.append(is_macd_valley)
                if use_primary_macd_valley:
                    conditions.append(is_primary_macd_valley)
                if use_volume:
                    conditions.append(prev_vol is not np.nan and vol >= prev_vol)
                if use_ema21_slope_entry:
                    conditions.append(ema21_slope is not np.nan and ema21_slope > ema21_slope_entry_threshold)

                # If no conditions are enabled, don't allow entry
                if not conditions:
                    base_ok = False
                else:
                    base_ok = all(conditions)

                # Check if entry would have happened
                if base_ok:
                    # Check extended hours FIRST before doing anything else
                    if avoid_extended_hours and not is_regular_hours:
                        # Entry conditions met but blocked by extended hours filter
                        extended_hours_skipped_count += 1
                        logs.append({
                            "time": t,
                            "event": "entry_skipped_extended_hours",
                            "price": close,
                            "note": f"Entry blocked: Extended hours ({t.strftime('%H:%M')} ET)"
                        })
                        # CRITICAL: Do not create position, skip to next iteration
                        continue

                    # Build note based on which conditions were checked
                    note_parts = []
                    if use_rsi:
                        note_parts.append(f"RSI < {rsi_threshold} (RSI={rsi_last:.1f})")
                    if use_ema_cross_up:
                        note_parts.append(f"EMA9 crossed above EMA21 (EMA9={ema9:.2f}, EMA21={ema21:.2f})")
                    if use_bb_cross_up:
                        note_parts.append(f"Price crossed below BB lower (Close={close:.2f}, BB Lower={bb_low_v:.2f})")
                    if use_bb_width:
                        note_parts.append(f"BB width > {bb_width_threshold}% (Width={bb_width_val:.2f}%)")
                    if use_macd_cross_up:
                        note_parts.append(f"MACD crossed above signal (MACD={macd_val:.4f}, Signal={macd_signal_val:.4f})")
                    if use_primary_macd_cross_up:
                        note_parts.append(f"Primary MACD crossed above signal (MACD={primary_macd_val:.4f}, Signal={primary_macd_signal_val:.4f})")
                    if use_ema:
                        note_parts.append(f"Price > EMA9 (Close={close:.2f}, EMA9={ema9:.2f})")
                    if use_price_above_ema21:
                        note_parts.append(f"Price > EMA21 (Close={close:.2f}, EMA21={ema21:.2f})")
                    if use_macd_below_threshold:
                        note_parts.append(f"MACD < {macd_below_threshold} (MACD={macd_val:.4f})")
                    if use_primary_macd_below_threshold:
                        note_parts.append(f"Primary MACD < {primary_macd_below_threshold} (MACD={primary_macd_val:.4f})")
                    if use_macd_valley:
                        note_parts.append(f"MACD valley detected (MACD={macd_val:.4f})")
                    if use_primary_macd_valley:
                        note_parts.append(f"Primary MACD valley detected (MACD={primary_macd_val:.4f})")
                    if use_volume:
                        note_parts.append(f"Volume rising (Vol={vol:.0f}, PrevVol={prev_vol:.0f})")
                    if use_ema21_slope_entry:
                        note_parts.append(f"EMA21 slope > {ema21_slope_entry_threshold}% (Slope={ema21_slope:.4f}%)")

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

# Get ticker early for page config (before any other st commands)
def get_ticker_for_page_config():
    """Get ticker name for page config before session state is fully initialized"""
    # First, check URL query parameters
    query_params = st.query_params
    url_ticker = query_params.get('ticker', None)

    # If URL parameter exists and is valid, use it
    if url_ticker:
        url_ticker = url_ticker.upper()
        available = get_available_tickers()
        if url_ticker in available:
            return url_ticker

    # Try to get from session state
    if 'settings' in st.session_state and 'ticker' in st.session_state.settings:
        return st.session_state.settings['ticker']

    # Otherwise get default ticker
    available = get_available_tickers()
    return available[0] if available else 'TQQQ'

page_ticker = get_ticker_for_page_config()
st.set_page_config(page_title=f"{page_ticker} - Strategy Scalping", layout="wide")

# Create placeholder for title that will be updated with price info
title_placeholder = st.empty()
title_placeholder.title(f"Strategy Scalping - {page_ticker}")

# ---------- Cached Settings Management ----------

@st.cache_data(ttl=1)  # Cache for 1 second to prevent stale reads
def get_settings_from_file(ticker):
    """Get fresh settings from alpaca.json with caching to prevent timing issues"""
    alpaca_strategy = load_strategy_from_alpaca(ticker)
    if not alpaca_strategy:
        return None
        
    entry = alpaca_strategy.get('entry_conditions', {})
    exit_cond = alpaca_strategy.get('exit_conditions', {})
    risk = alpaca_strategy.get('risk_management', {})
    
    return {
        'ticker': ticker,
        'period': alpaca_strategy.get('period', '1d'),
        'interval': alpaca_strategy.get('interval', '5m'),
        'interval_2': alpaca_strategy.get('interval_2', '1h'),
        'chart_height': 1150,
        # Entry conditions from alpaca.json
        'use_rsi': entry.get('use_rsi', False),
        'rsi_threshold': entry.get('rsi_threshold', 30),
        'use_ema_cross_up': entry.get('use_ema_cross_up', False),
        'use_bb_cross_up': entry.get('use_bb_cross_up', False),
        'use_bb_width': entry.get('use_bb_width', False),
        'bb_width_threshold': entry.get('bb_width_threshold', 5.0),
        'use_macd_cross_up': entry.get('use_macd_cross_up', False),
        'use_primary_macd_cross_up': entry.get('use_primary_macd_cross_up', False),
        'use_ema': entry.get('use_ema', False),
        'use_price_above_ema21': entry.get('use_price_above_ema21', False),
        'use_price_vs_ema9': entry.get('use_price_vs_ema9', False),
        'use_price_vs_ema21': entry.get('use_price_vs_ema21', False),
        'use_volume': entry.get('use_volume', False),
        'use_macd_threshold': entry.get('use_macd_threshold', False),
        'macd_threshold': entry.get('macd_threshold', 0.1),
        'use_macd_below_threshold': entry.get('use_macd_threshold', False),
        'macd_below_threshold': entry.get('macd_threshold', 0.1),
        'use_primary_macd_below_threshold': entry.get('use_primary_macd_below_threshold', False),
        'primary_macd_below_threshold': entry.get('primary_macd_below_threshold', 0.1),
        'use_macd_valley': entry.get('use_macd_valley', False),
        'use_primary_macd_valley': entry.get('use_primary_macd_valley', False),
        'use_ema21_slope_entry': entry.get('use_ema21_slope_entry', False),
        'ema21_slope_entry_threshold': entry.get('ema21_slope_entry_threshold', 0.0),
        # Exit conditions from alpaca.json
        'use_rsi_exit': exit_cond.get('use_rsi_exit', False),
        'rsi_exit_threshold': exit_cond.get('rsi_exit_threshold', 70),
        'use_rsi_overbought': exit_cond.get('use_rsi_exit', False),
        'rsi_overbought_threshold': exit_cond.get('rsi_exit_threshold', 70),
        'use_ema_cross_down': exit_cond.get('use_ema_cross_down', False),
        'use_bb_cross_down': exit_cond.get('use_bb_cross_down', False),
        'use_bb_width_exit': exit_cond.get('use_bb_width_exit', False),
        'bb_width_exit_threshold': exit_cond.get('bb_width_exit_threshold', 10.0),
        'use_macd_cross_down': exit_cond.get('use_macd_cross_down', False),
        'use_primary_macd_cross_down': exit_cond.get('use_primary_macd_cross_down', False),
        'use_price_vs_ema9_exit': exit_cond.get('use_price_vs_ema9_exit', False),
        'use_price_vs_ema21_exit': exit_cond.get('use_price_vs_ema21_exit', False),
        'use_price_below_ema9': exit_cond.get('use_price_vs_ema9_exit', False),
        'use_price_below_ema21': exit_cond.get('use_price_vs_ema21_exit', False),
        'use_macd_peak': exit_cond.get('use_macd_peak', False),
        'use_primary_macd_peak': exit_cond.get('use_primary_macd_peak', False),
        'use_macd_above_threshold': exit_cond.get('use_macd_above_threshold', False),
        'macd_above_threshold': exit_cond.get('macd_above_threshold', 0.0),
        'use_primary_macd_above_threshold': exit_cond.get('use_primary_macd_above_threshold', False),
        'primary_macd_above_threshold': exit_cond.get('primary_macd_above_threshold', 0.0),
        'use_ema21_slope_exit': exit_cond.get('use_ema21_slope_exit', False),
        'ema21_slope_exit_threshold': exit_cond.get('ema21_slope_exit_threshold', 0.0),
        # Risk management from alpaca.json
        'stop_loss_pct': risk.get('stop_loss', 0.02) * 100,
        'take_profit_pct': risk.get('take_profit', 0.03) * 100,
        'use_stop_loss': risk.get('use_stop_loss', True),
        'use_take_profit': risk.get('use_take_profit', False),
        # UI settings (defaults)
        'show_signals': True,
        'show_reports': True
    }

def init_session_state_once(ticker):
    """Initialize session state only once per ticker to prevent overwrites"""
    session_key = f"initialized_{ticker}"
    
    if session_key not in st.session_state:
        # First time for this ticker - load from file
        fresh_settings = get_settings_from_file(ticker)
        if fresh_settings:
            st.session_state.settings = fresh_settings
            st.session_state[session_key] = True

def auto_save_on_change(setting_key):
    """Universal auto-save callback for any setting"""
    if 'settings' in st.session_state:
        ticker = st.session_state.settings.get('ticker')
        if ticker and save_settings_to_alpaca(st.session_state.settings, ticker):
            # Clear cache to force reload of fresh data
            get_settings_from_file.clear()

def create_setting_callback(setting_key, ticker, widget_key=None):
    """Create a callback function for a specific setting"""
    if widget_key is None:
        widget_key = f"{setting_key}_widget_{ticker}"

    def callback():
        st.session_state.settings[setting_key] = st.session_state[widget_key]
        auto_save_on_change(setting_key)

    return callback


def handle_ticker_selection(selected_option, available_tickers, current_ticker):
    """Render ticker management UI and return the active ticker symbol."""
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
                if st.button("✅ Create Ticker", use_container_width=True, key=f"create_ticker_{new_ticker}"):
                    config = load_alpaca_config()
                    if config and 'signal_actions' in config and 'tickers' in config['signal_actions']:
                        config['signal_actions']['tickers'][new_ticker] = {
                            "enabled": False,
                            "default_quantity": 100,
                            "entry": {
                                "enabled": True,
                                "actions": [
                                    {"type": "BUY", "ticker": new_ticker, "quantity": 100}
                                ],
                                "description": f"{new_ticker} entry signal - Go long"
                            },
                            "exit_conditions_met": {
                                "enabled": True,
                                "actions": [
                                    {"type": "SELL_ALL", "ticker": new_ticker}
                                ],
                                "description": f"{new_ticker} exit conditions met - Close position"
                            },
                            "exit_SL": {
                                "enabled": True,
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
                ticker = current_ticker
        else:
            ticker = current_ticker
    else:
        ticker = selected_option

    st.session_state.settings['ticker'] = ticker
    st.query_params['ticker'] = ticker
    return ticker

# Initialize session state for settings persistence
if 'settings' not in st.session_state:
    # Check URL parameter first, otherwise use default
    query_params = st.query_params
    url_ticker = query_params.get('ticker', None)

    if url_ticker:
        url_ticker = url_ticker.upper()
        available = get_available_tickers()
        default_ticker = url_ticker if url_ticker in available else available[0]
    else:
        default_ticker = get_available_tickers()[0]

    init_session_state_once(default_ticker)

# Get current ticker and ensure proper initialization
current_ticker = st.session_state.get('settings', {}).get('ticker')
if current_ticker:
    init_session_state_once(current_ticker)

with st.sidebar:
    # Show account status at top of sidebar
    show_account_status()
    st.divider()
    
    st.header("Settings")

    # Add custom CSS for extremely small button text
    st.markdown("""
        <style>
        div.stButton > button {
            font-size: 9px !important;
            padding: 2px 4px !important;
            min-height: 25px !important;
        }
        div.stButton > button p {
            font-size: 9px !important;
        }
        </style>
    """, unsafe_allow_html=True)

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

    sidebar_backtest_tab, sidebar_pattern_tab = st.tabs(["Backtest", "Pattern Matching"])

    with sidebar_backtest_tab:
        ticker = handle_ticker_selection(selected_option, available_tickers, current_ticker)
        # Bot Control Section
        st.subheader("🤖 Trading Bot Control")

        # Check bot status
        is_running, pid = get_bot_status()
        if is_running:
            st.success(f"✅ Bot Running (PID: {pid})")
        else:
            st.warning("❌ Bot Stopped")

        # Control buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Start", use_container_width=True, disabled=is_running, key=f"bot_start_{is_running}"):
                success, msg = start_bot()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not is_running, key=f"bot_stop_{is_running}"):
                success, msg = stop_bot()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with col3:
            if st.button("🔄 Restart", use_container_width=True, key=f"bot_restart_{pid}"):
                with st.spinner("Restarting bot..."):
                    success, msg = restart_bot()
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.divider()

        # Trading Configuration Section
        st.subheader("⚙️ Alpaca Trading Settings")

        # Load current trading settings
        trading_settings = get_trading_settings()

        # Slippage configuration with expander
        with st.expander("📊 Slippage Configuration", expanded=False):
            st.caption("Configure slippage limits for buy/sell orders")

            max_buy_slippage = st.number_input(
                "Max buy slippage (%)",
                min_value=0.1,
                max_value=10.0,
                value=trading_settings.get('max_buy_slippage_pct', 0.9),
                step=0.1,
                help="Skip buy order if current price is higher than signal price by this percentage"
            )

            max_sell_slippage = st.number_input(
                "Max sell slippage (%)",
                min_value=0.1,
                max_value=10.0,
                value=trading_settings.get('max_sell_slippage_pct', 0.9),
                step=0.1,
                help="Skip sell order if current price is lower than signal price by this percentage"
            )

            limit_order_slippage = st.number_input(
                "Limit order slippage (%)",
                min_value=0.1,
                max_value=10.0,
                value=trading_settings.get('limit_order_slippage_pct', 2.0),
                step=0.1,
                help="Slippage percentage for limit orders during extended hours"
            )

            st.caption("ℹ️ Market orders are used during regular hours (9:30 AM - 4:00 PM ET), limit orders with slippage during extended hours")

        # Extended hours configuration
        avoid_extended_hours = st.checkbox(
            "🕐 Avoid trading in extended hours",
            value=trading_settings.get('avoid_extended_hours', False),
            help="Only trade during regular market hours (9:30 AM - 4:00 PM ET). Applies to both backtesting and live trading. When disabled, uses limit orders with larger slippage for extended hours in live trading."
        )

        # Auto-save trading settings
        update_trading_settings(max_buy_slippage, max_sell_slippage, limit_order_slippage, avoid_extended_hours)

        st.divider()

        # Auto-load strategy from alpaca.json when ticker changes
        if 'previous_ticker' not in st.session_state:
            st.session_state.previous_ticker = ticker

        # Only reload strategy from alpaca.json when ticker changes
        # This prevents overwriting user's UI changes
        should_reload = ticker != st.session_state.previous_ticker

        if should_reload:
            alpaca_strategy = load_strategy_from_alpaca(ticker)
        else:
            alpaca_strategy = None

        # Only update from alpaca.json on first load or explicit ticker change
        should_load_from_alpaca = (
            alpaca_strategy and
            (
                'settings_initialized' not in st.session_state or
                ticker != st.session_state.get('last_loaded_ticker', '')
            )
        )

        if should_load_from_alpaca:
            # Update settings from alpaca.json
            entry = alpaca_strategy.get('entry_conditions', {})
            exit_cond = alpaca_strategy.get('exit_conditions', {})
            risk = alpaca_strategy.get('risk_management', {})

            # Mark this ticker as loaded and settings as initialized
            st.session_state.last_loaded_ticker = ticker
            st.session_state.settings_initialized = True

            # Entry conditions from alpaca.json
            st.session_state.settings['use_rsi'] = entry.get('use_rsi', False)
            st.session_state.settings['rsi_threshold'] = entry.get('rsi_threshold', 30)
            st.session_state.settings['use_ema_cross_up'] = entry.get('use_ema_cross_up', False)
            st.session_state.settings['use_bb_cross_up'] = entry.get('use_bb_cross_up', False)
            st.session_state.settings['use_bb_width'] = entry.get('use_bb_width', False)
            st.session_state.settings['bb_width_threshold'] = entry.get('bb_width_threshold', 5.0)
            st.session_state.settings['use_macd_cross_up'] = entry.get('use_macd_cross_up', False)
            st.session_state.settings['use_primary_macd_cross_up'] = entry.get('use_primary_macd_cross_up', False)
            st.session_state.settings['use_ema'] = entry.get('use_ema', False)
            st.session_state.settings['use_price_above_ema21'] = entry.get('use_price_above_ema21', False)
            st.session_state.settings['use_price_vs_ema9'] = entry.get('use_price_vs_ema9', False)
            st.session_state.settings['use_price_vs_ema21'] = entry.get('use_price_vs_ema21', False)
            st.session_state.settings['use_volume'] = entry.get('use_volume', False)
            st.session_state.settings['use_macd_threshold'] = entry.get('use_macd_threshold', False)
            st.session_state.settings['macd_threshold'] = entry.get('macd_threshold', 0.1)
            # Map use_macd_threshold to UI field use_macd_below_threshold
            st.session_state.settings['use_macd_below_threshold'] = entry.get('use_macd_threshold', False)
            st.session_state.settings['macd_below_threshold'] = entry.get('macd_threshold', 0.1)
            st.session_state.settings['use_primary_macd_below_threshold'] = entry.get('use_primary_macd_below_threshold', False)
            st.session_state.settings['primary_macd_below_threshold'] = entry.get('primary_macd_below_threshold', 0.1)
            st.session_state.settings['use_macd_valley'] = entry.get('use_macd_valley', False)
            st.session_state.settings['use_primary_macd_valley'] = entry.get('use_primary_macd_valley', False)
            st.session_state.settings['use_ema21_slope_entry'] = entry.get('use_ema21_slope_entry', False)
            st.session_state.settings['ema21_slope_entry_threshold'] = entry.get('ema21_slope_entry_threshold', 0.0)

            # Exit conditions from alpaca.json
            st.session_state.settings['use_rsi_exit'] = exit_cond.get('use_rsi_exit', False)
            st.session_state.settings['rsi_exit_threshold'] = exit_cond.get('rsi_exit_threshold', 70)
            st.session_state.settings['use_rsi_overbought'] = exit_cond.get('use_rsi_exit', False)  # Alias
            st.session_state.settings['rsi_overbought_threshold'] = exit_cond.get('rsi_exit_threshold', 70)  # Alias
            st.session_state.settings['use_ema_cross_down'] = exit_cond.get('use_ema_cross_down', False)
            st.session_state.settings['use_price_below_ema9'] = exit_cond.get('use_price_vs_ema9_exit', False)
            st.session_state.settings['use_price_below_ema21'] = exit_cond.get('use_price_vs_ema21_exit', False)
            st.session_state.settings['use_bb_cross_down'] = exit_cond.get('use_bb_cross_down', False)
            st.session_state.settings['use_bb_width_exit'] = exit_cond.get('use_bb_width_exit', False)
            st.session_state.settings['bb_width_exit_threshold'] = exit_cond.get('bb_width_exit_threshold', 10.0)
            st.session_state.settings['use_macd_cross_down'] = exit_cond.get('use_macd_cross_down', False)
            st.session_state.settings['use_primary_macd_cross_down'] = exit_cond.get('use_primary_macd_cross_down', False)
            st.session_state.settings['use_price_vs_ema9_exit'] = exit_cond.get('use_price_vs_ema9_exit', False)
            st.session_state.settings['use_price_vs_ema21_exit'] = exit_cond.get('use_price_vs_ema21_exit', False)
            st.session_state.settings['use_macd_peak'] = exit_cond.get('use_macd_peak', False)
            st.session_state.settings['use_primary_macd_peak'] = exit_cond.get('use_primary_macd_peak', False)
            st.session_state.settings['use_macd_above_threshold'] = exit_cond.get('use_macd_above_threshold', False)
            st.session_state.settings['macd_above_threshold'] = exit_cond.get('macd_above_threshold', 0.0)
            st.session_state.settings['use_primary_macd_above_threshold'] = exit_cond.get('use_primary_macd_above_threshold', False)
            st.session_state.settings['primary_macd_above_threshold'] = exit_cond.get('primary_macd_above_threshold', 0.0)
            st.session_state.settings['use_ema21_slope_exit'] = exit_cond.get('use_ema21_slope_exit', False)
            st.session_state.settings['ema21_slope_exit_threshold'] = exit_cond.get('ema21_slope_exit_threshold', 0.0)

            # Risk management from alpaca.json
            st.session_state.settings['use_stop_loss'] = risk.get('use_stop_loss', True)
            st.session_state.settings['stop_loss_pct'] = risk.get('stop_loss', 0.02) * 100  # Convert to percentage
            st.session_state.settings['use_take_profit'] = risk.get('use_take_profit', False)
            st.session_state.settings['take_profit_pct'] = risk.get('take_profit', 0.03) * 100  # Convert to percentage

            # Interval and period
            st.session_state.settings['interval'] = alpaca_strategy.get('interval', '5m')
            st.session_state.settings['interval_2'] = alpaca_strategy.get('interval_2', '10m')
            st.session_state.settings['period'] = alpaca_strategy.get('period', '1d')



            # Save a snapshot of loaded settings to track changes
            st.session_state.loaded_settings = st.session_state.settings.copy()

            # Only trigger rerun if ticker actually changed to avoid infinite loops
            if ticker != st.session_state.previous_ticker:
                st.session_state.previous_ticker = ticker
                st.rerun()

        # Initialize loaded_settings if not exists (for first load)
        if 'loaded_settings' not in st.session_state:
            st.session_state.loaded_settings = st.session_state.settings.copy()

        # Add ticker enable/disable/delete control in one line
        st.divider()

        current_status = get_ticker_enabled_status(ticker)
        status_emoji = "✅" if current_status else "❌"
        status_text = "Enabled" if current_status else "Disabled"

        st.markdown(f"**Ticker Status:** {status_emoji} {status_text}")

        # Check if in delete confirmation mode
        if 'confirm_delete_ticker' not in st.session_state:
            st.session_state.confirm_delete_ticker = None

        if st.session_state.confirm_delete_ticker == ticker:
            # Show confirmation state with confirm/cancel buttons
            st.warning(f"⚠️ Delete {ticker}?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm", use_container_width=True, key=f"confirm_del_{ticker}"):
                    if delete_ticker(ticker):
                        st.success(f"✅ Deleted {ticker}")
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
            # Show all three buttons in one line
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Enable", use_container_width=True, disabled=current_status, key=f"enable_{ticker}"):
                    if update_ticker_enabled_status(ticker, True):
                        st.success(f"✅ {ticker} enabled")
                        st.rerun()
                    else:
                        st.error(f"Failed to enable {ticker}")

            with col2:
                if st.button("❌ Disable", use_container_width=True, disabled=not current_status, key=f"disable_{ticker}"):
                    if update_ticker_enabled_status(ticker, False):
                        st.warning(f"❌ {ticker} disabled")
                        st.rerun()
                    else:
                        st.error(f"Failed to disable {ticker}")

            with col3:
                if st.button("🗑️ Delete", use_container_width=True, type="secondary", key=f"delete_{ticker}"):
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
            help="Number of shares to trade for this ticker",
            key=f'default_quantity_{ticker}'  # Add unique key per ticker to force reload
        )

        # Auto-save quantity when changed
        if new_quantity != current_quantity:
            update_ticker_default_quantity(ticker, new_quantity)

        st.divider()

        # Manual Trading Section (collapsible)
        with st.expander("📈 Manual Trading", expanded=False):
            # Get account info and current price
            account_info = get_account_info()

            if account_info:
                # Display account summary in compact format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Portfolio", f"${account_info['portfolio_value']:,.0f}")
                    st.metric("Cash", f"${account_info['cash']:,.0f}")
                with col2:
                    st.metric("Equity", f"${account_info['equity']:,.0f}")
                    st.metric("Buying Power", f"${account_info['buying_power']:,.0f}")

                # Get positions
                positions = get_positions()
                if positions:
                    st.caption(f"**Current Positions ({len(positions)}):**")
                    for pos in positions:
                        pnl_pct = float(pos['unrealized_plpc']) * 100
                        pnl_color = "🟢" if pnl_pct >= 0 else "🔴"
                        st.caption(f"{pnl_color} {pos['symbol']}: {pos['qty']} shares @ ${float(pos['current_price']):.2f} ({pnl_pct:+.2f}%)")
                else:
                    st.caption("No open positions")
            else:
                st.warning("⚠️ Unable to connect to Alpaca API")

            st.divider()

            # Manual Buy/Sell Controls
            st.caption("**Quick Trade:**")

            # Get current price
            current_price = get_current_price(ticker)
            if current_price:
                st.caption(f"💲 Current Price: **${current_price:.2f}**")

            # Trade quantity input
            trade_quantity = st.number_input(
                "Quantity",
                min_value=1,
                max_value=10000,
                value=current_quantity,
                step=1,
                key=f"trade_qty_{ticker}",
                help="Number of shares to buy or sell"
            )

            # Order type selection
            order_type_option = st.radio(
                "Order Type",
                options=["Auto (Market/Limit based on hours)", "Market Order", "Limit Order"],
                index=0,
                key=f"order_type_{ticker}",
                help="Auto: Market during regular hours (9:30 AM-4 PM ET), Limit during extended hours"
            )

            # Map selection to order type
            if order_type_option == "Market Order":
                selected_order_type = "MKT"
                limit_price_value = None
            elif order_type_option == "Limit Order":
                selected_order_type = "LMT"
                # Show limit price input for limit orders
                # Get slippage from config to avoid circular import
                trading_settings_for_limit = get_trading_settings()
                limit_slippage_pct = trading_settings_for_limit.get('limit_order_slippage_pct', 2.0)
                default_buy_limit = round(current_price * (1 + limit_slippage_pct / 100), 2) if current_price else 0
                default_sell_limit = round(current_price * (1 - limit_slippage_pct / 100), 2) if current_price else 0

                col_b, col_s = st.columns(2)
                with col_b:
                    buy_limit_price = st.number_input(
                        "Buy Limit Price",
                        min_value=0.01,
                        value=default_buy_limit,
                        step=0.01,
                        key=f"buy_limit_{ticker}",
                        help="Price for buy limit order"
                    )
                with col_s:
                    sell_limit_price = st.number_input(
                        "Sell Limit Price",
                        min_value=0.01,
                        value=default_sell_limit,
                        step=0.01,
                        key=f"sell_limit_{ticker}",
                        help="Price for sell limit order"
                    )
            else:  # Auto
                selected_order_type = "AUTO"
                limit_price_value = None
                buy_limit_price = None
                sell_limit_price = None

            # Buy and Sell buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🟢 BUY", use_container_width=True, type="primary", key=f"buy_{ticker}"):
                    with st.spinner(f"Placing BUY order for {trade_quantity} shares of {ticker}..."):
                        # Use appropriate limit price if limit order
                        limit_price = buy_limit_price if selected_order_type == "LMT" else None
                        success, message = place_manual_buy(ticker, trade_quantity, order_type=selected_order_type, limit_price=limit_price)
                        if success:
                            st.success(message)
                            # Refresh account info after order
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)

            with col2:
                if st.button("🔴 SELL", use_container_width=True, type="secondary", key=f"sell_{ticker}"):
                    with st.spinner(f"Placing SELL order for {trade_quantity} shares of {ticker}..."):
                        # Use appropriate limit price if limit order
                        limit_price = sell_limit_price if selected_order_type == "LMT" else None
                        success, message = place_manual_sell(ticker, trade_quantity, order_type=selected_order_type, limit_price=limit_price)
                        if success:
                            st.success(message)
                            # Refresh account info after order
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)

        st.divider()

        period_options = ["1d", "5d", "2wk", "1mo", "2mo", "3mo", "6mo", "1y"]
        try:
            period_index = period_options.index(st.session_state.settings['period'])
        except ValueError:
            period_index = 0
        def update_period():
            st.session_state.settings['period'] = st.session_state[f'period_widget_{ticker}']
            auto_save_on_change('period')

        period = st.selectbox("Data period", period_options, index=period_index,
                             key=f'period_widget_{ticker}', on_change=update_period)
        st.session_state.settings['period'] = period

        # Add interval selector like check.py
        if 'interval' not in st.session_state.settings:
            st.session_state.settings['interval'] = "5m"

        interval_options = ["1m", "2m", "3m", "5m", "10m", "15m", "30m", "1h", "1d"]
        try:
            interval_index = interval_options.index(st.session_state.settings['interval'])
        except ValueError:
            interval_index = 0

        def update_interval():
            st.session_state.settings['interval'] = st.session_state[f'interval_widget_{ticker}']
            auto_save_on_change('interval')

        interval = st.selectbox("Data interval", interval_options, index=interval_index,
                               key=f'interval_widget_{ticker}', on_change=update_interval)
        st.session_state.settings['interval'] = interval


        if 'interval_2' not in st.session_state.settings:
            st.session_state.settings['interval_2'] = "10m"

        try:
            interval_2_index = interval_options.index(st.session_state.settings['interval_2'])
        except (ValueError, KeyError):
            interval_2_index = 7  # Default to "1h"

        def update_interval_2():
            st.session_state.settings['interval_2'] = st.session_state[f'interval_2_widget_{ticker}']
            auto_save_on_change('interval_2')

        interval_2 = st.selectbox("Data interval 2 ", interval_options, index=interval_2_index,
                                  key=f'interval_2_widget_{ticker}', on_change=update_interval_2,
                                  help="Select a different timeframe for comparison MACD")
        st.session_state.settings['interval_2'] = interval_2
        primary_macd_display_name = f"MACD ({interval})"
        macd_display_name = f"MACD ({interval_2})"

        st.divider()

        st.subheader("Trading Rules")

        # Entry Conditions
        st.markdown("**Entry Conditions (all must be met):**")
        use_rsi = st.checkbox("RSI Oversold", value=st.session_state.settings['use_rsi'],
                             help="RSI must be below threshold before entry",
                             key=f'use_rsi_widget_{ticker}', on_change=create_setting_callback('use_rsi', ticker))
        st.session_state.settings['use_rsi'] = use_rsi

        rsi_threshold = st.number_input("RSI oversold threshold", min_value=0, max_value=100,
                                        value=st.session_state.settings['rsi_threshold'],
                                        disabled=not use_rsi,
                                        help="Alert when RSI falls below this level",
                                        key=f'rsi_threshold_widget_{ticker}', on_change=create_setting_callback('rsi_threshold', ticker))
        st.session_state.settings['rsi_threshold'] = rsi_threshold

        use_ema_cross_up = st.checkbox("EMA9 crosses above EMA21",
                                        value=st.session_state.settings['use_ema_cross_up'],
                                        help="Entry when EMA9 crosses above EMA21 (bullish)",
                                        key=f'use_ema_cross_up_widget_{ticker}', on_change=create_setting_callback('use_ema_cross_up', ticker))
        st.session_state.settings['use_ema_cross_up'] = use_ema_cross_up

        use_bb_cross_up = st.checkbox("Price crosses below BB Lower",
                                       value=st.session_state.settings['use_bb_cross_up'],
                                       help="Entry when price crosses below Bollinger Band lower line (oversold)",
                                       key=f'use_bb_cross_up_widget_{ticker}', on_change=create_setting_callback('use_bb_cross_up', ticker))
        st.session_state.settings['use_bb_cross_up'] = use_bb_cross_up

        # Handle backwards compatibility for use_bb_width
        if 'use_bb_width' not in st.session_state.settings:
            st.session_state.settings['use_bb_width'] = False
        if 'bb_width_threshold' not in st.session_state.settings:
            st.session_state.settings['bb_width_threshold'] = 5.0

        use_bb_width = st.checkbox("BB Width > threshold (high volatility)",
                                   value=st.session_state.settings['use_bb_width'],
                                   help="Entry when Bollinger Bands width is above threshold (high volatility - trending market)",
                                   key=f'use_bb_width_widget_{ticker}', on_change=create_setting_callback('use_bb_width', ticker))
        st.session_state.settings['use_bb_width'] = use_bb_width

        bb_width_threshold = st.number_input("BB Width threshold (%)",
                                             min_value=0.1, max_value=20.0,
                                             value=st.session_state.settings['bb_width_threshold'],
                                             step=0.001,
                                             format="%.3f",
                                             disabled=not use_bb_width,
                                             help="Entry when BB width is above this percentage (typical: 5-10%)",
                                             key=f'bb_width_threshold_widget_{ticker}', on_change=create_setting_callback('bb_width_threshold', ticker))
        st.session_state.settings['bb_width_threshold'] = bb_width_threshold

        # Primary MACD entry options
        if 'use_primary_macd_cross_up' not in st.session_state.settings:
            st.session_state.settings['use_primary_macd_cross_up'] = False

        use_primary_macd_cross_up = st.checkbox(f"{primary_macd_display_name} crosses above Signal line",
                                                value=st.session_state.settings['use_primary_macd_cross_up'],
                                                help=f"Entry when {primary_macd_display_name} line crosses above signal line (bullish)",
                                                key=f'use_primary_macd_cross_up_widget_{ticker}', on_change=create_setting_callback('use_primary_macd_cross_up', ticker))
        st.session_state.settings['use_primary_macd_cross_up'] = use_primary_macd_cross_up

        if 'use_primary_macd_below_threshold' not in st.session_state.settings:
            st.session_state.settings['use_primary_macd_below_threshold'] = False
        if 'primary_macd_below_threshold' not in st.session_state.settings:
            st.session_state.settings['primary_macd_below_threshold'] = 0.0

        use_primary_macd_below_threshold = st.checkbox(f"{primary_macd_display_name} < Threshold",
                                                       value=st.session_state.settings['use_primary_macd_below_threshold'],
                                                       help=f"Entry when {primary_macd_display_name} is below threshold",
                                                       key=f'use_primary_macd_below_threshold_widget_{ticker}', on_change=create_setting_callback('use_primary_macd_below_threshold', ticker))
        st.session_state.settings['use_primary_macd_below_threshold'] = use_primary_macd_below_threshold

        primary_macd_below_threshold = st.number_input(f"{primary_macd_display_name} below threshold value",
                                                       value=st.session_state.settings['primary_macd_below_threshold'],
                                                       step=0.001,
                                                       format="%.3f",
                                                       disabled=not use_primary_macd_below_threshold,
                                                       help=f"Enter when {primary_macd_display_name} is below this value",
                                                       key=f'primary_macd_below_threshold_widget_{ticker}', on_change=create_setting_callback('primary_macd_below_threshold', ticker))
        st.session_state.settings['primary_macd_below_threshold'] = primary_macd_below_threshold

        if 'use_primary_macd_valley' not in st.session_state.settings:
            st.session_state.settings['use_primary_macd_valley'] = False

        use_primary_macd_valley = st.checkbox(f"{primary_macd_display_name} Valley (turning up)",
                                              value=st.session_state.settings['use_primary_macd_valley'],
                                              help=f"Entry when {primary_macd_display_name} valley is detected (turning up)",
                                              key=f'use_primary_macd_valley_widget_{ticker}', on_change=create_setting_callback('use_primary_macd_valley', ticker))
        st.session_state.settings['use_primary_macd_valley'] = use_primary_macd_valley

        # Secondary MACD entry options
        # Handle backwards compatibility for use_macd_cross_up
        if 'use_macd_cross_up' not in st.session_state.settings:
            st.session_state.settings['use_macd_cross_up'] = False

        use_macd_cross_up = st.checkbox(f"{macd_display_name} crosses above Signal line",
                                         value=st.session_state.settings['use_macd_cross_up'],
                                         help=f"Entry when {macd_display_name} line crosses above signal line (bullish)",
                                         key=f'use_macd_cross_up_widget_{ticker}', on_change=create_setting_callback('use_macd_cross_up', ticker))
        st.session_state.settings['use_macd_cross_up'] = use_macd_cross_up

        use_ema = st.checkbox("Price > EMA9", value=st.session_state.settings['use_ema'],
                             key=f'use_ema_widget_{ticker}', on_change=create_setting_callback('use_ema', ticker))
        st.session_state.settings['use_ema'] = use_ema

        # Handle backwards compatibility for use_price_above_ema21
        if 'use_price_above_ema21' not in st.session_state.settings:
            st.session_state.settings['use_price_above_ema21'] = False

        use_price_above_ema21 = st.checkbox("Price > EMA21",
                                             value=st.session_state.settings['use_price_above_ema21'],
                                             help="Entry when price is above EMA21",
                                             key='use_price_above_ema21_widget', on_change=create_setting_callback('use_price_above_ema21', ticker))
        st.session_state.settings['use_price_above_ema21'] = use_price_above_ema21

        # Handle backwards compatibility for use_macd_below_threshold
        if 'use_macd_below_threshold' not in st.session_state.settings:
            st.session_state.settings['use_macd_below_threshold'] = False
        if 'macd_below_threshold' not in st.session_state.settings:
            st.session_state.settings['macd_below_threshold'] = 0.0

        use_macd_below_threshold = st.checkbox(f"{macd_display_name} < Threshold",
                                                value=st.session_state.settings['use_macd_below_threshold'],
                                                help=f"Entry when {macd_display_name} is below threshold",
                                                key=f'use_macd_below_threshold_widget_{ticker}', on_change=create_setting_callback('use_macd_below_threshold', ticker))
        st.session_state.settings['use_macd_below_threshold'] = use_macd_below_threshold

        macd_below_threshold = st.number_input(f"{macd_display_name} below threshold value",
                                                value=st.session_state.settings['macd_below_threshold'],
                                                step=0.001,
                                                format="%.3f",
                                                disabled=not use_macd_below_threshold,
                                                help=f"Enter when {macd_display_name} is below this value",
                                                key=f'macd_below_threshold_widget_{ticker}', on_change=create_setting_callback('macd_below_threshold', ticker))
        st.session_state.settings['macd_below_threshold'] = macd_below_threshold

        # Handle backwards compatibility for use_macd_valley
        if 'use_macd_valley' not in st.session_state.settings:
            st.session_state.settings['use_macd_valley'] = False

        use_macd_valley = st.checkbox(f"{macd_display_name} Valley (turning up)",
                                       value=st.session_state.settings['use_macd_valley'],
                                       help=f"Entry when {macd_display_name} valley is detected (turning up)",
                                       key=f'use_macd_valley_widget_{ticker}', on_change=create_setting_callback('use_macd_valley', ticker))
        st.session_state.settings['use_macd_valley'] = use_macd_valley

        use_volume = st.checkbox("Volume Rising", value=st.session_state.settings['use_volume'],
                                help="Current candle volume >= previous candle volume",
                                key=f'use_volume_widget_{ticker}', on_change=create_setting_callback('use_volume', ticker))
        st.session_state.settings['use_volume'] = use_volume

        # Handle backwards compatibility for use_ema21_slope_entry
        if 'use_ema21_slope_entry' not in st.session_state.settings:
            st.session_state.settings['use_ema21_slope_entry'] = False
        if 'ema21_slope_entry_threshold' not in st.session_state.settings:
            st.session_state.settings['ema21_slope_entry_threshold'] = 0.0

        use_ema21_slope_entry = st.checkbox("EMA21 Slope > Threshold (rising)",
                                            value=st.session_state.settings['use_ema21_slope_entry'],
                                            help="Entry when EMA21 slope is above threshold (trending up)",
                                            key='use_ema21_slope_entry_widget', on_change=create_setting_callback('use_ema21_slope_entry', ticker))
        st.session_state.settings['use_ema21_slope_entry'] = use_ema21_slope_entry

        ema21_slope_entry_threshold = st.number_input("EMA21 slope entry threshold (%)",
                                                      min_value=-1.0, max_value=1.0,
                                                      value=st.session_state.settings['ema21_slope_entry_threshold'],
                                                      step=0.001,
                                                      format="%.3f",
                                                      disabled=not use_ema21_slope_entry,
                                                      help="Entry when EMA21 slope is above this % (typical: 0.01-0.1%)",
                                                      key='ema21_slope_entry_threshold_widget', on_change=create_setting_callback('ema21_slope_entry_threshold', ticker))
        st.session_state.settings['ema21_slope_entry_threshold'] = ema21_slope_entry_threshold

        # Price drop from exit requirement
        # Exit Rules
        st.markdown("**Exit Rules (all must be met):**")
        use_stop_loss = st.checkbox("Exit on Stop Loss", value=st.session_state.settings['use_stop_loss'],
                                    help="Exit when price drops by specified %",
                                    key=f'use_stop_loss_widget_{ticker}', on_change=create_setting_callback('use_stop_loss', ticker))
        st.session_state.settings['use_stop_loss'] = use_stop_loss

        stop_loss_pct = st.number_input("Stop Loss %", min_value=0.5, max_value=10.0,
                                         value=st.session_state.settings['stop_loss_pct'], step=0.5,
                                         disabled=not use_stop_loss,
                                         help="Exit if price drops by this %",
                                         key=f'stop_loss_pct_widget_{ticker}', on_change=create_setting_callback('stop_loss_pct', ticker))
        st.session_state.settings['stop_loss_pct'] = stop_loss_pct
        stop_loss_pct = stop_loss_pct / 100

        use_take_profit = st.checkbox("Exit on Take Profit", value=st.session_state.settings['use_take_profit'],
                                       help="Exit when price rises by specified %",
                                       key=f'use_take_profit_widget_{ticker}', on_change=create_setting_callback('use_take_profit', ticker))
        st.session_state.settings['use_take_profit'] = use_take_profit

        take_profit_pct = st.number_input("Take Profit %", min_value=0.5, max_value=20.0,
                                           value=st.session_state.settings['take_profit_pct'], step=0.5,
                                           disabled=not use_take_profit,
                                           help="Exit if price rises by this %",
                                           key=f'take_profit_pct_widget_{ticker}', on_change=create_setting_callback('take_profit_pct', ticker))
        st.session_state.settings['take_profit_pct'] = take_profit_pct
        take_profit_pct = take_profit_pct / 100

        use_rsi_overbought = st.checkbox("Exit on RSI Overbought",
                                          value=st.session_state.settings['use_rsi_overbought'],
                                          help="Exit when RSI exceeds this level",
                                          key=f'use_rsi_overbought_widget_{ticker}', on_change=create_setting_callback('use_rsi_overbought', ticker))
        st.session_state.settings['use_rsi_overbought'] = use_rsi_overbought

        rsi_overbought_threshold = st.number_input("RSI overbought threshold",
                                                    min_value=0, max_value=100,
                                                    value=st.session_state.settings['rsi_overbought_threshold'],
                                                    disabled=not use_rsi_overbought,
                                                    help="Exit when RSI rises above this level",
                                                    key=f'rsi_overbought_threshold_widget_{ticker}', on_change=create_setting_callback('rsi_overbought_threshold', ticker))
        st.session_state.settings['rsi_overbought_threshold'] = rsi_overbought_threshold

        use_ema_cross_down = st.checkbox("Exit on EMA9 crosses below EMA21",
                                          value=st.session_state.settings['use_ema_cross_down'],
                                          help="Exit when EMA9 crosses below EMA21 (bearish)",
                                          key=f'use_ema_cross_down_widget_{ticker}', on_change=create_setting_callback('use_ema_cross_down', ticker))
        st.session_state.settings['use_ema_cross_down'] = use_ema_cross_down

        use_price_below_ema9 = st.checkbox("Exit on Price < EMA9",
                                            value=st.session_state.settings['use_price_below_ema9'],
                                            help="Exit when price falls below EMA9",
                                            key='use_price_below_ema9_widget', on_change=create_setting_callback('use_price_below_ema9', ticker))
        st.session_state.settings['use_price_below_ema9'] = use_price_below_ema9

        # Handle backwards compatibility for use_price_below_ema21
        if 'use_price_below_ema21' not in st.session_state.settings:
            st.session_state.settings['use_price_below_ema21'] = False

        use_price_below_ema21 = st.checkbox("Exit on Price < EMA21",
                                             value=st.session_state.settings['use_price_below_ema21'],
                                             help="Exit when price falls below EMA21",
                                             key='use_price_below_ema21_widget', on_change=create_setting_callback('use_price_below_ema21', ticker))
        st.session_state.settings['use_price_below_ema21'] = use_price_below_ema21

        use_bb_cross_down = st.checkbox("Exit on Price crosses above BB Upper",
                                         value=st.session_state.settings['use_bb_cross_down'],
                                         help="Exit when price crosses above Bollinger Band upper line (overbought)",
                                         key=f'use_bb_cross_down_widget_{ticker}', on_change=create_setting_callback('use_bb_cross_down', ticker))
        st.session_state.settings['use_bb_cross_down'] = use_bb_cross_down

        # Handle backwards compatibility for use_bb_width_exit
        if 'use_bb_width_exit' not in st.session_state.settings:
            st.session_state.settings['use_bb_width_exit'] = False
        if 'bb_width_exit_threshold' not in st.session_state.settings:
            st.session_state.settings['bb_width_exit_threshold'] = 10.0

        use_bb_width_exit = st.checkbox("Exit on BB Width > threshold (high volatility)",
                                        value=st.session_state.settings['use_bb_width_exit'],
                                        help="Exit when Bollinger Bands width exceeds threshold (volatility expanding - potential reversal)",
                                        key=f'use_bb_width_exit_widget_{ticker}', on_change=create_setting_callback('use_bb_width_exit', ticker))
        st.session_state.settings['use_bb_width_exit'] = use_bb_width_exit

        bb_width_exit_threshold = st.number_input("BB Width exit threshold (%)",
                                                  min_value=0.1, max_value=20.0,
                                                  value=st.session_state.settings['bb_width_exit_threshold'],
                                                  step=0.001,
                                                  format="%.3f",
                                                  disabled=not use_bb_width_exit,
                                                  help="Exit when BB width exceeds this percentage (typical: 8-15%)",
                                                  key=f'bb_width_exit_threshold_widget_{ticker}', on_change=create_setting_callback('bb_width_exit_threshold', ticker))
        st.session_state.settings['bb_width_exit_threshold'] = bb_width_exit_threshold

        # Primary MACD exit options
        if 'use_primary_macd_cross_down' not in st.session_state.settings:
            st.session_state.settings['use_primary_macd_cross_down'] = False

        use_primary_macd_cross_down = st.checkbox(f"Exit on {primary_macd_display_name} crosses below Signal line",
                                                  value=st.session_state.settings['use_primary_macd_cross_down'],
                                                  help=f"Exit when {primary_macd_display_name} line crosses below signal line (bearish)",
                                                  key=f'use_primary_macd_cross_down_widget_{ticker}', on_change=create_setting_callback('use_primary_macd_cross_down', ticker))
        st.session_state.settings['use_primary_macd_cross_down'] = use_primary_macd_cross_down

        if 'use_primary_macd_above_threshold' not in st.session_state.settings:
            st.session_state.settings['use_primary_macd_above_threshold'] = False
        if 'primary_macd_above_threshold' not in st.session_state.settings:
            st.session_state.settings['primary_macd_above_threshold'] = 0.0

        use_primary_macd_above_threshold = st.checkbox(f"Exit on {primary_macd_display_name} > Threshold",
                                                       value=st.session_state.settings['use_primary_macd_above_threshold'],
                                                       help=f"Exit when {primary_macd_display_name} exceeds threshold",
                                                       key=f'use_primary_macd_above_threshold_widget_{ticker}', on_change=create_setting_callback('use_primary_macd_above_threshold', ticker))
        st.session_state.settings['use_primary_macd_above_threshold'] = use_primary_macd_above_threshold

        primary_macd_above_threshold = st.number_input(f"{primary_macd_display_name} above threshold value",
                                                       value=st.session_state.settings['primary_macd_above_threshold'],
                                                       step=0.001,
                                                       format="%.3f",
                                                       disabled=not use_primary_macd_above_threshold,
                                                       help=f"Exit when {primary_macd_display_name} exceeds this value",
                                                       key=f'primary_macd_above_threshold_widget_{ticker}', on_change=create_setting_callback('primary_macd_above_threshold', ticker))
        st.session_state.settings['primary_macd_above_threshold'] = primary_macd_above_threshold

        if 'use_primary_macd_peak' not in st.session_state.settings:
            st.session_state.settings['use_primary_macd_peak'] = False

        use_primary_macd_peak = st.checkbox(f"Exit on {primary_macd_display_name} Peak (turning down)",
                                            value=st.session_state.settings['use_primary_macd_peak'],
                                            help=f"Exit when {primary_macd_display_name} peak is detected (turning down)",
                                            key=f'use_primary_macd_peak_widget_{ticker}', on_change=create_setting_callback('use_primary_macd_peak', ticker))
        st.session_state.settings['use_primary_macd_peak'] = use_primary_macd_peak

        # Secondary MACD exit options
        # Handle backwards compatibility for use_macd_cross_down
        if 'use_macd_cross_down' not in st.session_state.settings:
            st.session_state.settings['use_macd_cross_down'] = False

        use_macd_cross_down = st.checkbox(f"Exit on {macd_display_name} crosses below Signal line",
                                           value=st.session_state.settings['use_macd_cross_down'],
                                           help=f"Exit when {macd_display_name} line crosses below signal line (bearish)",
                                           key=f'use_macd_cross_down_widget_{ticker}', on_change=create_setting_callback('use_macd_cross_down', ticker))
        st.session_state.settings['use_macd_cross_down'] = use_macd_cross_down

        # Handle backwards compatibility for use_macd_above_threshold
        if 'use_macd_above_threshold' not in st.session_state.settings:
            st.session_state.settings['use_macd_above_threshold'] = False
        if 'macd_above_threshold' not in st.session_state.settings:
            st.session_state.settings['macd_above_threshold'] = 0.0

        use_macd_above_threshold = st.checkbox(f"Exit on {macd_display_name} > Threshold",
                                                value=st.session_state.settings['use_macd_above_threshold'],
                                                help=f"Exit when {macd_display_name} exceeds threshold",
                                                key=f'use_macd_above_threshold_widget_{ticker}', on_change=create_setting_callback('use_macd_above_threshold', ticker))
        st.session_state.settings['use_macd_above_threshold'] = use_macd_above_threshold

        macd_above_threshold = st.number_input(f"{macd_display_name} above threshold value",
                                                value=st.session_state.settings['macd_above_threshold'],
                                                step=0.001,
                                                format="%.3f",
                                                disabled=not use_macd_above_threshold,
                                                help=f"Exit when {macd_display_name} exceeds this value",
                                                key=f'macd_above_threshold_widget_{ticker}', on_change=create_setting_callback('macd_above_threshold', ticker))
        st.session_state.settings['macd_above_threshold'] = macd_above_threshold

        # Handle backwards compatibility for use_macd_peak
        if 'use_macd_peak' not in st.session_state.settings:
            st.session_state.settings['use_macd_peak'] = False

        use_macd_peak = st.checkbox(f"Exit on {macd_display_name} Peak (turning down)",
                                     value=st.session_state.settings['use_macd_peak'],
                                     help=f"Exit when {macd_display_name} peak is detected (turning down)",
                                     key=f'use_macd_peak_widget_{ticker}', on_change=create_setting_callback('use_macd_peak', ticker))
        st.session_state.settings['use_macd_peak'] = use_macd_peak

        # Handle backwards compatibility for use_ema21_slope_exit
        if 'use_ema21_slope_exit' not in st.session_state.settings:
            st.session_state.settings['use_ema21_slope_exit'] = False
        if 'ema21_slope_exit_threshold' not in st.session_state.settings:
            st.session_state.settings['ema21_slope_exit_threshold'] = 0.0

        use_ema21_slope_exit = st.checkbox("Exit on EMA21 Slope < Threshold (falling)",
                                           value=st.session_state.settings['use_ema21_slope_exit'],
                                           help="Exit when EMA21 slope is below threshold (trending down)",
                                           key='use_ema21_slope_exit_widget', on_change=create_setting_callback('use_ema21_slope_exit', ticker))
        st.session_state.settings['use_ema21_slope_exit'] = use_ema21_slope_exit

        ema21_slope_exit_threshold = st.number_input("EMA21 slope exit threshold (%)",
                                                     min_value=-1.0, max_value=1.0,
                                                     value=st.session_state.settings['ema21_slope_exit_threshold'],
                                                     step=0.001,
                                                     format="%.3f",
                                                     disabled=not use_ema21_slope_exit,
                                                     help="Exit when EMA21 slope is below this % (typical: -0.1-0.0%)",
                                                     key='ema21_slope_exit_threshold_widget', on_change=create_setting_callback('ema21_slope_exit_threshold', ticker))
        st.session_state.settings['ema21_slope_exit_threshold'] = ema21_slope_exit_threshold

        st.divider()

        show_signals = st.checkbox("Show buy/sell signals on chart",
                                   value=st.session_state.settings['show_signals'],
                                   key=f'show_signals_widget_{ticker}', on_change=create_setting_callback('show_signals', ticker))
        st.session_state.settings['show_signals'] = show_signals

        # Handle backwards compatibility for show_reports
        if 'show_reports' not in st.session_state.settings:
            st.session_state.settings['show_reports'] = True

        show_reports = st.checkbox("Show backtest reports",
                                   value=st.session_state.settings['show_reports'],
                                   help="Show backtest summary table and event logs",
                                   key=f'show_reports_widget_{ticker}', on_change=create_setting_callback('show_reports', ticker))
        st.session_state.settings['show_reports'] = show_reports

        # Handle backwards compatibility for chart_height
        if 'chart_height' not in st.session_state.settings:
            st.session_state.settings['chart_height'] = 1150

        chart_height = st.slider("Chart height (pixels)", min_value=600, max_value=2000,
                                 value=st.session_state.settings['chart_height'], step=50,
                                 help="Adjust the height of the chart",
                                 key=f'chart_height_widget_{ticker}', on_change=create_setting_callback('chart_height', ticker))
        st.session_state.settings['chart_height'] = chart_height

        # Save settings to file
        save_settings(st.session_state.settings)

    with sidebar_pattern_tab:
        st.subheader("📐 Pattern Matching Settings")
        st.caption("Control how many 5m bars define the reference window and how strict the cosine filter is.")
        reference_bars_input = st.number_input(
            "Reference bars (5m)",
            min_value=50,
            max_value=2000,
            value=int(st.session_state.get('pattern_reference_bars', 316)),
            step=10,
            help="Length of the latest 5-minute window used as the reference pattern",
            key=f'pattern_reference_bars_widget_{ticker}'
        )
        st.session_state['pattern_reference_bars'] = int(reference_bars_input)

        similarity_input = st.slider(
            "Similarity threshold",
            min_value=0.50,
            max_value=0.99,
            value=float(st.session_state.get('pattern_similarity_threshold', 0.85)),
            step=0.01,
            help="Minimum cosine similarity required for historical matches",
            key=f'pattern_similarity_threshold_widget_{ticker}'
        )
        st.session_state['pattern_similarity_threshold'] = float(similarity_input)
pattern_reference_bars = int(st.session_state.get('pattern_reference_bars', 316))
pattern_similarity_threshold = float(st.session_state.get('pattern_similarity_threshold', 0.85))

# Auto-save to alpaca.json if settings have changed
if settings_have_changed(st.session_state.settings, st.session_state.get('loaded_settings', {})):
    if save_settings_to_alpaca(st.session_state.settings, ticker):
        # Update loaded_settings snapshot after successful save
        st.session_state.loaded_settings = st.session_state.settings.copy()
        st.success(f"✅ Settings saved to alpaca.json for {ticker}", icon="💾")

        # Reload from JSON to ensure UI reflects what was actually saved
        alpaca_strategy = load_strategy_from_alpaca(ticker)
        if alpaca_strategy:
            entry = alpaca_strategy.get('entry_conditions', {})
            exit_cond = alpaca_strategy.get('exit_conditions', {})
            risk = alpaca_strategy.get('risk_management', {})

            # Update session state with the saved values
            st.session_state.settings.update({
                'interval': alpaca_strategy.get('interval', '5m'),
                'interval_2': alpaca_strategy.get('interval_2', '10m'),
                'period': alpaca_strategy.get('period', '1d'),
                'use_rsi': entry.get('use_rsi', False),
                'rsi_threshold': entry.get('rsi_threshold', 30),
                'use_ema_cross_up': entry.get('use_ema_cross_up', False),
                'use_bb_cross_up': entry.get('use_bb_cross_up', False),
                'use_bb_width': entry.get('use_bb_width', False),
                'bb_width_threshold': entry.get('bb_width_threshold', 5.0),
                'use_macd_cross_up': entry.get('use_macd_cross_up', False),
                'use_primary_macd_cross_up': entry.get('use_primary_macd_cross_up', False),
                'use_ema': entry.get('use_ema', False),
                'use_macd_threshold': entry.get('use_macd_threshold', False),
                'macd_threshold': entry.get('macd_threshold', 0.0),
                'use_macd_below_threshold': entry.get('use_macd_threshold', False),
                'macd_below_threshold': entry.get('macd_threshold', 0.0),
                'use_primary_macd_below_threshold': entry.get('use_primary_macd_below_threshold', False),
                'primary_macd_below_threshold': entry.get('primary_macd_below_threshold', 0.0),
                'use_macd_valley': entry.get('use_macd_valley', False),
                'use_primary_macd_valley': entry.get('use_primary_macd_valley', False),
                'use_rsi_overbought': exit_cond.get('use_rsi_exit', False),
                'rsi_overbought_threshold': exit_cond.get('rsi_exit_threshold', 70),
                'use_ema_cross_down': exit_cond.get('use_ema_cross_down', False),
                'use_bb_cross_down': exit_cond.get('use_bb_cross_down', False),
                'use_macd_cross_down': exit_cond.get('use_macd_cross_down', False),
                'use_primary_macd_cross_down': exit_cond.get('use_primary_macd_cross_down', False),
                'use_macd_peak': exit_cond.get('use_macd_peak', False),
                'use_primary_macd_peak': exit_cond.get('use_primary_macd_peak', False),
                'use_stop_loss': risk.get('use_stop_loss', True),
                'stop_loss_pct': risk.get('stop_loss', 0.02) * 100,
                'use_take_profit': risk.get('use_take_profit', False),
                'take_profit_pct': risk.get('take_profit', 0.03) * 100,
                'use_macd_above_threshold': exit_cond.get('use_macd_above_threshold', False),
                'macd_above_threshold': exit_cond.get('macd_above_threshold', 0.0),
                'use_primary_macd_above_threshold': exit_cond.get('use_primary_macd_above_threshold', False),
                'primary_macd_above_threshold': exit_cond.get('primary_macd_above_threshold', 0.0)
            })
            # Update the loaded_settings to match the reloaded values
            st.session_state.loaded_settings = st.session_state.settings.copy()

tab_backtest, tab_pattern = st.tabs(["Backtest", "Pattern Matching"])

with tab_backtest:
    # ---- Auto-run backtest and display chart ----
    with st.spinner(f"Downloading {ticker} data..."):
        # Helper to map requested period based on interval
        def resolve_period(selected_interval, selected_period):
            if selected_interval == "1d":
                mapping = {
                    "1d": "5d",
                    "5d": "5d",
                    "2wk": "1mo",
                    "1mo": "3mo",
                    "2mo": "6mo",
                    "3mo": "1y",
                    "6mo": "2y",
                    "1y": "5y"
                }
            else:
                mapping = {
                    "1d": "5d",
                    "5d": "5d",
                    "2wk": "1mo",
                    "1mo": "1mo",
                    "2mo": "3mo",
                    "3mo": "3mo",
                    "6mo": "6mo",
                    "1y": "1y"
                }
            return mapping.get(selected_period, selected_period)

        main_period = resolve_period(interval, period)
        trading_period = resolve_period(interval_2, period)

        # Validate interval/period combination based on yfinance limits
        # 1m, 2m, 3m, 5m, 10m, 15m, 30m: max 60 days (3m and 10m use 1m data aggregation)
        # 1h, 90m: max 730 days
        intraday_short = ["1m", "2m", "3m", "5m", "10m", "15m", "30m"]
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

        if interval_2 in intraday_short:
            if trading_period in ["3mo", "6mo", "1y"]:
                st.warning(f"⚠️ {interval_2} interval only supports up to 60 days of data. Adjusting secondary period from {trading_period} to 60d.")
                trading_period = "60d"
        elif interval_2 in intraday_hourly:
            if trading_period == "1y" and period == "1y":
                pass

        # Use extended hours flags per interval
        use_extended_hours = interval not in ["1d", "5d", "1wk", "1mo", "3mo"]
        use_extended_hours_trading = interval_2 not in ["1d", "5d", "1wk", "1mo", "3mo"]

        # Yahoo Finance doesn't support 3m or 10m intervals natively
        # Fetch 1-minute data and aggregate for these intervals
        unsupported_intervals = ["3m", "10m"]

        if interval in unsupported_intervals:
            # Fetch 1-minute data first
            raw = yf.download(ticker, period=main_period,
                             interval='1m', progress=False, prepost=use_extended_hours)

            if raw.empty:
                st.error(f"No 1-minute data returned for ticker {ticker}. Cannot aggregate to {interval} interval.")
            else:
                # Handle MultiIndex columns if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # Convert timezone before resampling
                if hasattr(raw.index, 'tz'):
                    if raw.index.tz is not None:
                        raw.index = raw.index.tz_convert('America/New_York')
                    else:
                        raw.index = raw.index.tz_localize('UTC').tz_convert('America/New_York')

                # Aggregate to desired interval
                interval_map = {
                    '3m': '3min',
                    '10m': '10min'
                }
                resample_freq = interval_map[interval]

                raw = raw.resample(resample_freq).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                st.info(f"ℹ️ Aggregated 1-minute data to {interval} interval ({len(raw)} bars)")
        else:
            # Use Yahoo's native interval
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

        # Continue processing if we have data
        if not raw.empty:
            actual_return_pct = None
            chart_data = raw.copy()
            # Update title with current price and change
            try:
                session_end = chart_data.index[-1]
                daily_slice = chart_data[chart_data.index.date == session_end.date()]
                current_price = daily_slice['Close'].iloc[-1]
                previous_price = daily_slice['Open'].iloc[0] if not daily_slice.empty else chart_data['Close'].iloc[0]
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0

                change_color = "🟢" if price_change >= 0 else "🔴"
                change_sign = "+" if price_change >= 0 else ""
                title_html = f"""
                <h1>Strategy Scalping - {ticker}
                <span style='font-size: 0.8em; color: {"#00FF00" if price_change >= 0 else "#FF4444"};'>
                ${current_price:.2f} {change_sign}{price_change:.2f} ({change_sign}{price_change_pct:.2f}%) {change_color}
                </span></h1>
                """
                title_placeholder.markdown(title_html, unsafe_allow_html=True)
            except Exception:
                pass

            # Download data for second interval (trading data + MACD panel)
            df_macd2_full = pd.DataFrame()
            trading_data = pd.DataFrame()
            trading_interval = interval
            raw_2 = pd.DataFrame()
            if interval_2 in unsupported_intervals:
                raw_2 = yf.download(ticker, period=trading_period,
                                   interval='1m', progress=False, prepost=use_extended_hours_trading)
                if not raw_2.empty:
                    if isinstance(raw_2.columns, pd.MultiIndex):
                        raw_2.columns = raw_2.columns.get_level_values(0)
                    if hasattr(raw_2.index, 'tz'):
                        if raw_2.index.tz is not None:
                            raw_2.index = raw_2.index.tz_convert('America/New_York')
                        else:
                            raw_2.index = raw_2.index.tz_localize('UTC').tz_convert('America/New_York')

                    interval_map = {'3m': '3min', '10m': '10min'}
                    resample_freq = interval_map[interval_2]
                    raw_2 = raw_2.resample(resample_freq).agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min',
                        'Close': 'last', 'Volume': 'sum'
                    }).dropna()
            else:
                raw_2 = yf.download(ticker, period=trading_period,
                                   interval=interval_2, progress=False, prepost=use_extended_hours_trading)
                if not raw_2.empty:
                    if isinstance(raw_2.columns, pd.MultiIndex):
                        raw_2.columns = raw_2.columns.get_level_values(0)
                    if hasattr(raw_2.index, 'tz'):
                        if raw_2.index.tz is not None:
                            raw_2.index = raw_2.index.tz_convert('America/New_York')
                        else:
                            raw_2.index = raw_2.index.tz_localize('UTC').tz_convert('America/New_York')

            if not raw_2.empty:
                trading_data = raw_2.copy()
                trading_interval = interval_2
                macd_line_2, signal_line_2, histogram_2 = macd(trading_data["Close"])
                df_macd2_full = pd.DataFrame({
                    'macd': macd_line_2,
                    'macd_signal': signal_line_2,
                    'macd_hist': histogram_2
                }, index=trading_data.index)
            else:
                trading_data = chart_data.copy()

            primary_macd_features = pd.DataFrame()
            if not chart_data.empty:
                primary_macd_features = compute_macd_features_df(chart_data).add_prefix('primary_')
            if not trading_data.empty and not primary_macd_features.empty:
                aligned_primary = primary_macd_features.reindex(trading_data.index, method='pad')
                trading_data = trading_data.copy()
                for col in aligned_primary.columns:
                    trading_data[col] = aligned_primary[col]
                if "primary_macd" in trading_data.columns:
                    trading_data["primary_macd_prev"] = trading_data["primary_macd"].shift(1)
                if "primary_macd_signal" in trading_data.columns:
                    trading_data["primary_macd_signal_prev"] = trading_data["primary_macd_signal"].shift(1)
                if "primary_macd_peak" in trading_data.columns:
                    trading_data["primary_macd_peak"] = trading_data["primary_macd_peak"].fillna(False)
                if "primary_macd_valley" in trading_data.columns:
                    trading_data["primary_macd_valley"] = trading_data["primary_macd_valley"].fillna(False)

            # Determine analysis window based on trading data
            analysis_start = None
            if not trading_data.empty:
                last_time = trading_data.index[-1]

                if period == "1d":
                    if trading_interval == "1d":
                        analysis_start = last_time - pd.Timedelta(days=1)
                    else:
                        last_date = last_time.date()
                        analysis_start = pd.Timestamp(last_date).tz_localize('America/New_York') + pd.Timedelta(hours=4)
                elif period == "5d":
                    analysis_start = last_time - pd.Timedelta(days=5)
                elif period == "2wk":
                    analysis_start = last_time - pd.Timedelta(weeks=2)
                elif period == "1mo":
                    analysis_start = last_time - pd.Timedelta(days=30)
                elif period == "2mo":
                    analysis_start = last_time - pd.Timedelta(days=60)
                elif period == "3mo":
                    analysis_start = last_time - pd.Timedelta(days=90)
                elif period == "6mo":
                    analysis_start = last_time - pd.Timedelta(days=180)
                elif period == "1y":
                    analysis_start = last_time - pd.Timedelta(days=365)

            # Get avoid_extended_hours setting from trading settings
            avoid_extended_hours_setting = get_trading_settings().get('avoid_extended_hours', False)

            if avoid_extended_hours_setting:
                st.info("ℹ️ Extended hours avoidance enabled - only entries during 9:30 AM - 4:00 PM ET will be allowed")

            _, _, trades_df, logs_df = backtest_symbol(
                trading_data,
                stop_loss=stop_loss_pct,
                tp_pct=take_profit_pct,
                use_rsi=use_rsi,
                rsi_threshold=rsi_threshold,
                use_bb_cross_up=use_bb_cross_up,
                use_bb_width=use_bb_width,
                bb_width_threshold=bb_width_threshold,
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
                use_bb_width_exit=use_bb_width_exit,
                bb_width_exit_threshold=bb_width_exit_threshold,
                use_macd_cross_up=use_macd_cross_up,
                use_primary_macd_cross_up=use_primary_macd_cross_up,
                use_macd_cross_down=use_macd_cross_down,
                use_primary_macd_cross_down=use_primary_macd_cross_down,
                use_price_above_ema21=use_price_above_ema21,
                use_price_below_ema21=use_price_below_ema21,
                use_macd_below_threshold=use_macd_below_threshold,
                macd_below_threshold=macd_below_threshold,
                use_primary_macd_below_threshold=use_primary_macd_below_threshold,
                primary_macd_below_threshold=primary_macd_below_threshold,
                use_macd_above_threshold=use_macd_above_threshold,
                macd_above_threshold=macd_above_threshold,
                use_macd_peak=use_macd_peak,
                use_macd_valley=use_macd_valley,
                use_primary_macd_above_threshold=use_primary_macd_above_threshold,
                primary_macd_above_threshold=primary_macd_above_threshold,
                use_primary_macd_peak=use_primary_macd_peak,
                use_primary_macd_valley=use_primary_macd_valley,
                use_ema21_slope_entry=use_ema21_slope_entry,
                ema21_slope_entry_threshold=ema21_slope_entry_threshold,
                use_ema21_slope_exit=use_ema21_slope_exit,
                ema21_slope_exit_threshold=ema21_slope_exit_threshold,
                avoid_extended_hours=avoid_extended_hours_setting
            )

            df1_chart, df5_chart, _, _ = backtest_symbol(
                chart_data,
                stop_loss=stop_loss_pct,
                tp_pct=take_profit_pct,
                use_rsi=use_rsi,
                rsi_threshold=rsi_threshold,
                use_bb_cross_up=use_bb_cross_up,
                use_bb_width=use_bb_width,
                bb_width_threshold=bb_width_threshold,
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
                use_bb_width_exit=use_bb_width_exit,
                bb_width_exit_threshold=bb_width_exit_threshold,
                use_macd_cross_up=use_macd_cross_up,
                use_primary_macd_cross_up=use_primary_macd_cross_up,
                use_macd_cross_down=use_macd_cross_down,
                use_primary_macd_cross_down=use_primary_macd_cross_down,
                use_price_above_ema21=use_price_above_ema21,
                use_price_below_ema21=use_price_below_ema21,
                use_macd_below_threshold=use_macd_below_threshold,
                macd_below_threshold=macd_below_threshold,
                use_primary_macd_below_threshold=use_primary_macd_below_threshold,
                primary_macd_below_threshold=primary_macd_below_threshold,
                use_macd_above_threshold=use_macd_above_threshold,
                macd_above_threshold=macd_above_threshold,
                use_macd_peak=use_macd_peak,
                use_macd_valley=use_macd_valley,
                use_primary_macd_above_threshold=use_primary_macd_above_threshold,
                primary_macd_above_threshold=primary_macd_above_threshold,
                use_primary_macd_peak=use_primary_macd_peak,
                use_primary_macd_valley=use_primary_macd_valley,
                use_ema21_slope_entry=use_ema21_slope_entry,
                ema21_slope_entry_threshold=ema21_slope_entry_threshold,
                use_ema21_slope_exit=use_ema21_slope_exit,
                ema21_slope_exit_threshold=ema21_slope_exit_threshold,
                avoid_extended_hours=avoid_extended_hours_setting
            )

            if analysis_start is not None:
                if not trades_df.empty and 'entry_time' in trades_df.columns:
                    trades_df = trades_df[trades_df['entry_time'] >= analysis_start].copy()
                if not logs_df.empty and 'time' in logs_df.columns:
                    logs_df = logs_df[logs_df['time'] >= analysis_start].copy()

            df1 = df1_chart
            df5 = df5_chart
            df1_display = df1
            df5_display = df5
            if not df1.empty:
                last_time = df1.index[-1]
                display_start = None

                if period == "1d":
                    if interval == "1d":
                        display_start = last_time - pd.Timedelta(days=1)
                    else:
                        last_date = last_time.date()
                        display_start = pd.Timestamp(last_date).tz_localize('America/New_York') + pd.Timedelta(hours=4)
                elif period == "5d":
                    display_start = last_time - pd.Timedelta(days=5)
                elif period == "2wk":
                    display_start = last_time - pd.Timedelta(weeks=2)
                elif period == "1mo":
                    display_start = last_time - pd.Timedelta(days=30)
                elif period == "2mo":
                    display_start = last_time - pd.Timedelta(days=60)
                elif period == "3mo":
                    display_start = last_time - pd.Timedelta(days=90)
                elif period == "6mo":
                    display_start = last_time - pd.Timedelta(days=180)
                elif period == "1y":
                    display_start = last_time - pd.Timedelta(days=365)

                if display_start is not None:
                    df1_display = df1[df1.index >= display_start].copy()
                    df5_display = df5[df5.index >= display_start].copy()

            # Prepare MACD2 data for chart display
            df_macd2 = pd.DataFrame()
            if not df_macd2_full.empty:
                df_macd2 = df_macd2_full.copy()
                if not df5_display.empty:
                    start_time = df5_display.index[0]
                    end_time = df5_display.index[-1]
                    df_macd2 = df_macd2[(df_macd2.index >= start_time) & (df_macd2.index <= end_time)]

                    if interval_2 == interval:
                        aligned_index = df_macd2.index.intersection(df5_display.index)
                        df_macd2 = df_macd2.loc[aligned_index]

            # Chart with entries/exits - single chart with multiple y-axes
            # Create header with reload button on the right
            col_left, col_right = st.columns([4, 1])
            with col_left:
                st.subheader(f"{ticker} - {interval} chart with signals")
            with col_right:
                if st.button("🔄 Reload", help="Refresh chart with latest market data", key="reload_chart_btn"):
                    st.rerun()

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

            # Add BB Mid with BB Width in customdata for tooltip
            fig.add_trace(go.Scatter(
                x=df5_display.index,
                y=df5_display["bb_mid"],
                name="BB Mid",
                line=dict(width=1, color='grey'),
                yaxis='y',
                customdata=df5_display["bb_width"].fillna(0).values,
                hovertemplate='BB Mid: %{y:.2f} (Width: %{customdata:.2f}%)<extra></extra>'
            ))

            fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["bb_low"],
                                     name="BB Lower", line=dict(width=1, color='grey'),
                                     yaxis='y'))
            fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["ema9"],
                                     name="EMA9", line=dict(width=1),
                                     yaxis='y'))
            # Create custom hover text for EMA21 with slope
            # Note: slope is already calculated as percentage (×100), so display as-is
            ema21_hover = [
                f"EMA21: {ema:.2f}<br>Slope: {slope:.4f}%"
                for ema, slope in zip(df5_display["ema21"], df5_display["ema21_slope"])
            ]
            fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["ema21"],
                                     name="EMA21", line=dict(width=1, color='purple'),
                                     hovertemplate='%{text}<extra></extra>',
                                     text=ema21_hover,
                                     yaxis='y'))

            if show_signals and not trades_df.empty:
                # Show all entries (including last entry even if it only has EOD exit)
                all_entries = trades_df[trades_df["entry_time"].notna()].copy()

                # Show only non-EOD exits
                real_exits = trades_df[
                    (trades_df["exit_reason"].notna()) &
                    (trades_df["exit_reason"] != "EOD")
                ].copy()

                if not all_entries.empty:
                    # Position signals on RSI chart at fixed positions
                    # Entry signals at RSI level 20 (bottom)
                    # Exit signals at RSI level 80 (top)

                    # Build enhanced entry tooltips with buy price
                    entry_tooltips = []
                    for idx, row in all_entries.iterrows():
                        # Group entry conditions with line breaks for readability
                        entry_reason = row['entry_reason']
                        # Remove "Entry: " prefix if present
                        if entry_reason.startswith("Entry: "):
                            entry_reason = entry_reason[7:]

                        # Split conditions and format with line breaks
                        conditions = entry_reason.split(', ')
                        formatted_conditions = '<br>  • ' + '<br>  • '.join(conditions)

                        # Format entry time
                        entry_display = row['entry_time']
                        if hasattr(entry_display, 'tz_convert'):
                            entry_display = entry_display.tz_convert('America/New_York')
                        entry_time_str = entry_display.strftime('%Y-%m-%d %H:%M %Z')

                        tooltip = f"<b>ENTRY</b>{formatted_conditions}<br><b>Time:</b> {entry_time_str}<br><b>Buy:</b> ${row['entry_price']:.2f}"
                        entry_tooltips.append(tooltip)

                    # Add entry signals to chart
                    fig.add_trace(go.Scatter(
                        x=all_entries["entry_time"],
                        y=[20] * len(all_entries),
                        mode="markers",
                        marker=dict(size=12, symbol="triangle-up", color='green'),
                        name="Entries",
                        text=entry_tooltips,
                        hoverinfo='text',
                        yaxis='y3'
                    ))

                if not real_exits.empty:
                    # Build enhanced exit tooltips with sell price and return
                    exit_tooltips = []
                    exit_colors = []
                    for _, row in real_exits.iterrows():
                        # Group exit conditions with line breaks for readability
                        exit_reason = row['exit_reason']
                        # Remove "Exit: " prefix if present
                        if exit_reason.startswith("Exit: "):
                            exit_reason = exit_reason[6:]

                        # Format exit reason - could be "SL", "TP", or multiple conditions
                        if exit_reason in ["SL", "TP"]:
                            formatted_exit = f"<b>EXIT: {exit_reason}</b>"
                        else:
                            # Split conditions and format with line breaks
                            conditions = exit_reason.split(', ')
                            formatted_conditions = '<br>  • ' + '<br>  • '.join(conditions)
                            formatted_exit = f"<b>EXIT</b>{formatted_conditions}"

                        # Format exit time
                        exit_display = row['exit_time']
                        if hasattr(exit_display, 'tz_convert'):
                            exit_display = exit_display.tz_convert('America/New_York')
                        exit_time_str = exit_display.strftime('%Y-%m-%d %H:%M %Z')

                        tooltip = f"{formatted_exit}<br><b>Time:</b> {exit_time_str}<br><b>Sell:</b> ${row['exit_price']:.2f}<br><b>Return:</b> {row['return_pct']:.2f}%"
                        exit_tooltips.append(tooltip)
                        # Use red for profit, black for loss
                        exit_colors.append('red' if row['return_pct'] >= 0 else 'black')

                    # Add exit signals to chart
                    fig.add_trace(go.Scatter(
                        x=real_exits["exit_time"],
                        y=[80] * len(real_exits),
                        mode="markers",
                        marker=dict(size=12, symbol="triangle-down", color=exit_colors),
                        name="Exits",
                        text=exit_tooltips,
                        hoverinfo='text',
                        yaxis='y3'
                    ))

            # Overlay filled Alpaca trades on price chart (arrow markers at fill price)
            alpaca_fills = []
            if ticker and not df5_display.empty:
                alpaca_fills = get_recent_filled_trades(ticker)
                actual_return_pct = calculate_actual_return_pct(alpaca_fills)

            if alpaca_fills and not df5_display.empty:
                fill_times = []
                fill_prices = []
                fill_colors = []
                fill_symbols = []
                fill_tooltips = []

                price_series = df5_display["Close"]
                start_time = df5_display.index[0]
                end_time = df5_display.index[-1]

                for trade in alpaca_fills:
                    trade_time = trade['time']
                    if trade_time < start_time or trade_time > end_time:
                        continue

                    price_val = trade.get('price')
                    if price_val is None:
                        idx = price_series.index.get_indexer([trade_time], method='nearest')
                        if len(idx) == 0 or idx[0] == -1:
                            continue
                        price_val = price_series.iloc[idx[0]]

                    fill_times.append(trade_time)
                    fill_prices.append(price_val)
                    side = trade.get('side', '').upper()
                    is_buy = (side == 'BUY')
                    fill_colors.append('royalblue' if is_buy else 'darkorange')
                    fill_symbols.append('arrow-up' if is_buy else 'arrow-down')

                    qty = trade.get('qty', 0.0)
                    qty_str = f"{qty:.2f}".rstrip('0').rstrip('.')
                    fill_tooltips.append(
                        f"<b>{side or 'TRADE'}</b><br>"
                        f"Qty: {qty_str}<br>"
                        f"Price: ${price_val:.2f}<br>"
                        f"Time: {trade_time.strftime('%Y-%m-%d %H:%M %Z')}"
                    )

                if fill_times:
                    fig.add_trace(go.Scatter(
                        x=fill_times,
                        y=fill_prices,
                        mode="markers",
                        marker=dict(
                            size=14,
                            symbol=fill_symbols,
                            color=fill_colors,
                            line=dict(width=1, color='black')
                        ),
                        name="Filled Trades",
                        text=fill_tooltips,
                        hoverinfo='text',
                        yaxis='y'
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
                                     name="MACD", line=dict(width=2, color='orange'),
                                     yaxis='y4'))
            fig.add_trace(go.Scatter(x=df5_display.index, y=df5_display["macd_signal"],
                                     name="Signal", line=dict(width=2, color='purple'),
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

            # Second MACD chart (yaxis5) - different interval for comparison
            if not df_macd2.empty:
                fig.add_trace(go.Scatter(x=df_macd2.index, y=df_macd2["macd"],
                                         name=f"MACD ({interval_2})", line=dict(width=2, color='orange'),
                                         yaxis='y5'))
                fig.add_trace(go.Scatter(x=df_macd2.index, y=df_macd2["macd_signal"],
                                         name=f"Signal ({interval_2})", line=dict(width=2, color='purple'),
                                         yaxis='y5'))

                # MACD2 histogram
                hist_colors_2 = ['green' if val >= 0 else 'red' for val in df_macd2["macd_hist"]]
                fig.add_trace(go.Bar(
                    x=df_macd2.index,
                    y=df_macd2["macd_hist"],
                    name=f"MACD Histogram ({interval_2})",
                    marker_color=hist_colors_2,
                    opacity=0.5,
                    yaxis='y5'
                ))

                # Detect MACD2 peaks and valleys
                if len(df_macd2) > 6:
                    macd_values_2 = df_macd2["macd"].values
                    peak_indices_2 = argrelextrema(macd_values_2, np.greater, order=3)[0]
                    valley_indices_2 = argrelextrema(macd_values_2, np.less, order=3)[0]

                    if len(peak_indices_2) > 0:
                        peak_times_2 = df_macd2.index[peak_indices_2]
                        peak_values_2 = macd_values_2[peak_indices_2]
                        fig.add_trace(go.Scatter(
                            x=peak_times_2,
                            y=peak_values_2,
                            mode='markers',
                            name=f'MACD Peak ({interval_2})',
                            marker=dict(size=10, symbol='triangle-down', color='red'),
                            yaxis='y5',
                            hoverinfo='x+y'
                        ))

                    if len(valley_indices_2) > 0:
                        valley_times_2 = df_macd2.index[valley_indices_2]
                        valley_values_2 = macd_values_2[valley_indices_2]
                        fig.add_trace(go.Scatter(
                            x=valley_times_2,
                            y=valley_values_2,
                            mode='markers',
                            name=f'MACD Valley ({interval_2})',
                            marker=dict(size=10, symbol='triangle-up', color='green'),
                            yaxis='y5',
                            hoverinfo='x+y'
                        ))

                # MACD2 zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, yref='y5')

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
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.4)",
                font_color="black",
                font=dict(size=12),
                namelength=-1
            ),
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
                # yaxis for Price chart (top 45%)
                yaxis=dict(
                    title=f'{ticker} Price',
                    domain=[0.55, 1.0]  # 0.45 height
                ),
                # yaxis2 for Volume chart (12%)
                yaxis2=dict(
                    title='Volume',
                    domain=[0.42, 0.54],  # 0.12 height
                    showgrid=False
                ),
                # yaxis3 for RSI chart (12%)
                yaxis3=dict(
                    title='RSI',
                    domain=[0.29, 0.41],  # 0.12 height
                    range=[0, 100]
                ),
                # yaxis4 for MACD chart 1 (14%)
                yaxis4=dict(
                    title=f'MACD ({interval})',
                    domain=[0.14, 0.28],  # 0.14 height
                    showgrid=True
                ),
                # yaxis5 for MACD chart 2 (14%)
                yaxis5=dict(
                    title=f'MACD ({interval_2})',
                    domain=[0.0, 0.13],  # 0.14 height (bottom panel)
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
                if trades_df.empty:
                    st.subheader(f"Backtest for {ticker}")
                    st.write("No trades with current rules.")
                else:
                    # Filter for rows that have exit_reason (completed trades)
                    # Include all trades for statistics, but mark EOD separately
                    display_trades = trades_df[trades_df["exit_reason"].notna()].copy()

                    # Calculate cumulative return for title
                    cum_ret = (1 + display_trades["return_pct"]/100).prod() - 1
                    cum_ret_pct = cum_ret * 100

                    # Color code the return in title
                    if cum_ret_pct > 0:
                        return_emoji = "📈"
                        return_color = "green"
                    elif cum_ret_pct < 0:
                        return_emoji = "📉"
                        return_color = "red"
                    else:
                        return_emoji = "➖"
                        return_color = "gray"

                    backtest_title = f"Backtest for {ticker} - Return: :{return_color}[{return_emoji} {cum_ret_pct:+.2f}%]"

                    if actual_return_pct is not None:
                        if actual_return_pct > 0:
                            actual_emoji = "✅"
                            actual_color = "green"
                        elif actual_return_pct < 0:
                            actual_emoji = "⚠️"
                            actual_color = "red"
                        else:
                            actual_emoji = "➖"
                            actual_color = "gray"
                        backtest_title += f" | Actual: :{actual_color}[{actual_emoji} {actual_return_pct:+.2f}%]"

                    st.subheader(backtest_title)

                    total = len(display_trades)
                    winrate = (display_trades["return_pct"] > 0).mean()
                    avg_ret = display_trades["return_pct"].mean()

                    st.markdown(f"- **Total trades:** {total}")
                    st.markdown(f"- **Win rate:** {winrate:.1%}")
                    st.markdown(f"- **Average return per trade:** {avg_ret:.2f}%")
                    st.markdown(f"- **Cumulative return:** {cum_ret_pct:.2f}%")

                    st.dataframe(display_trades)

                # Logs + download
                st.subheader("Rule / event log")

                # Show count of skipped extended hours entries if any
                if avoid_extended_hours_setting and not logs_df.empty:
                    skipped_count = len(logs_df[logs_df['event'] == 'entry_skipped_extended_hours'])
                    if skipped_count > 0:
                        st.warning(f"⏰ {skipped_count} entry signal(s) were blocked due to extended hours filter")

                st.dataframe(logs_df)
                if not logs_df.empty:
                    csv = logs_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download main ticker log as CSV",
                        data=csv,
                        file_name=f"{ticker}_rule_log.csv",
                        mime="text/csv"
                    )


with tab_pattern:
    st.subheader(f"{ticker} - Cosine Similarity Matches (5m / 60d)")
    st.caption("Reference pattern uses the most recent two trading days of 5-minute Yahoo Finance data.")

    similarity_data = fetch_similarity_data(ticker)
    if similarity_data.empty:
        st.info("No 5-minute data returned for similarity analysis. Try again later.")
    else:
        pattern_results = compute_pattern_matches(
            similarity_data,
            reference_bars=pattern_reference_bars,
            min_similarity=pattern_similarity_threshold,
            top_n=5
        )
        reference_df = pattern_results["reference"]
        matches_df = pattern_results["matches"]

        if reference_df is None or matches_df.empty:
            st.info("Not enough historical data to compare or no similar windows were found.")
        else:
            reference_start = reference_df.index[0]
            reference_end = reference_df.index[-1]
            st.markdown(
                f"- **Reference window:** {reference_start:%Y-%m-%d %H:%M} → {reference_end:%Y-%m-%d %H:%M} "
                f"({len(reference_df)} bars)"
            )

            display_df = matches_df.copy().reset_index(drop=True)
            match_start_series = display_df["match_start"]
            match_end_series = display_df["match_end"]
            for series_name, series in [("Match Start", match_start_series), ("Match End", match_end_series)]:
                try:
                    if series.dt.tz is not None:
                        series = series.dt.tz_convert('America/New_York')
                except (AttributeError, TypeError, ValueError):
                    pass
                try:
                    series = series.dt.tz_localize(None)
                except (AttributeError, TypeError, ValueError):
                    pass
                display_df[series_name] = series
            display_df["Cosine Similarity"] = display_df["similarity"].round(4)
            display_df["Bars"] = display_df["bars"]
            display_df["Duration (hrs)"] = display_df["duration_hours"].round(2)
            display_df["Days Ago"] = display_df["days_ago"]
            display_df = display_df[["Match Start", "Match End", "Cosine Similarity", "Bars", "Duration (hrs)", "Days Ago"]]

            selection_state_key = f"cosine_match_idx_{ticker}"
            selected_idx = st.session_state.get(selection_state_key, 0)
            if selected_idx >= len(matches_df):
                selected_idx = 0
            st.session_state[selection_state_key] = selected_idx

            selected_row = matches_df.iloc[selected_idx]
            st.markdown(
                f"Currently showing match **#{selected_idx + 1}** "
                f"({selected_row['match_start']:%Y-%m-%d} → {selected_row['match_end']:%Y-%m-%d}) "
                f"with cosine similarity **{selected_row['similarity']:.4f}**."
            )

            match_slice = similarity_data.loc[selected_row["match_start"]:selected_row["match_end"]].copy()
            if match_slice.empty or len(match_slice) != len(reference_df):
                match_slice = None

            full_fig = go.Figure()
            full_fig.add_trace(go.Scatter(
                x=similarity_data.index,
                y=similarity_data["Close"],
                name="Close (60d)",
                line=dict(color='lightgray')
            ))

            # Highlight the reference window on its original timeline
            reference_trace = reference_df["Close"]
            full_fig.add_trace(go.Scatter(
                x=reference_trace.index,
                y=reference_trace.values,
                name="Reference window",
                line=dict(color='royalblue', width=4)
            ))

            # Highlight up to three top matches (plus the selected one if needed)
            highlight_indices = list(range(min(3, len(matches_df))))
            if selected_idx not in highlight_indices and selected_idx < len(matches_df):
                highlight_indices.append(selected_idx)

            seen = set()
            match_colors = ['darkorange', 'orange', 'gold']
            color_iter = iter(match_colors)
            for position, match_idx in enumerate(highlight_indices, start=1):
                if match_idx in seen or match_idx >= len(matches_df):
                    continue
                seen.add(match_idx)
                segment = similarity_data.loc[
                    matches_df.iloc[match_idx]["match_start"]:matches_df.iloc[match_idx]["match_end"]
                ].copy()
                if segment.empty:
                    continue
                color = next(color_iter, 'gray')
                is_selected = (match_idx == selected_idx)
                rank_label = ""
                if match_idx in [0, 1, 2]:
                    rank_label = f"Top {match_idx + 1} "
                label = f"{rank_label}Match #{match_idx + 1} ({matches_df.iloc[match_idx]['similarity']:.4f})"
                if is_selected:
                    label += " [selected]"
                full_fig.add_trace(go.Scatter(
                    x=segment.index,
                    y=segment["Close"],
                    name=label,
                    line=dict(
                        color=color,
                        width=3 if is_selected else 2,
                        dash='dash'
                    )
                ))

            # Shade extended hours with light grey background
            shapes = []
            idx = similarity_data.index
            if len(idx) > 1:
                try:
                    idx_ny = idx.tz_convert('America/New_York')
                except (TypeError, AttributeError, ValueError):
                    idx_ny = idx.tz_localize('America/New_York')

                def is_extended(ts):
                    if ts.weekday() >= 5:
                        return True
                    if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
                        return True
                    if ts.hour > 15 or (ts.hour == 15 and ts.minute >= 60):
                        return True
                    if ts.hour == 16 and ts.minute >= 0:
                        return True
                    if ts.hour > 16:
                        return True
                    return False

                ext_start = None
                for i, (raw_ts, ny_ts) in enumerate(zip(idx, idx_ny)):
                    extended = is_extended(ny_ts)
                    next_ts = idx[i + 1] if i + 1 < len(idx) else raw_ts + (idx[i] - idx[i - 1]) if i > 0 else raw_ts
                    if extended and ext_start is None:
                        ext_start = raw_ts
                    if not extended and ext_start is not None:
                        shapes.append(dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=ext_start,
                            x1=raw_ts,
                            y0=0,
                            y1=1,
                            fillcolor="rgba(200,200,200,0.2)",
                            layer="below",
                            line_width=0,
                        ))
                        ext_start = None
                    if extended and i == len(idx) - 1 and ext_start is not None:
                        shapes.append(dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=ext_start,
                            x1=next_ts,
                            y0=0,
                            y1=1,
                            fillcolor="rgba(200,200,200,0.2)",
                            layer="below",
                            line_width=0,
                        ))

            tick_format = '%H:%M'
            hover_format = '%Y-%m-%d %H:%M'
            rangebreaks = []
            if not similarity_data.empty:
                first_idx = similarity_data.index[0]
                if hasattr(first_idx, 'hour'):
                    rangebreaks = [
                        dict(bounds=["sat", "mon"]),
                        dict(bounds=[20, 4], pattern="hour")
                    ]
                else:
                    tick_format = '%Y-%m-%d'
                    hover_format = '%Y-%m-%d'

            full_fig.update_layout(
                height=640,
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=20),
                xaxis_title="Time",
                yaxis_title="Close Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.4)",
                    font_color="black",
                    font=dict(size=12),
                    namelength=-1
                ),
                shapes=shapes,
                xaxis=dict(
                    rangebreaks=rangebreaks,
                    tickformat=tick_format,
                    hoverformat=hover_format
                )
            )
            st.plotly_chart(full_fig, use_container_width=True)

            if match_slice is not None:
                st.write(
                    f"Match starts {selected_row['match_start']:%Y-%m-%d %H:%M} "
                    f"({len(match_slice)} bars aligned with reference)."
                )

            event = st.dataframe(
                display_df,
                column_config={
                    "Match Start": st.column_config.DatetimeColumn("Match Start", format="YYYY-MM-DD HH:mm", disabled=True),
                    "Match End": st.column_config.DatetimeColumn("Match End", format="YYYY-MM-DD HH:mm", disabled=True),
                    "Cosine Similarity": st.column_config.NumberColumn("Cosine Similarity", format="%.4f", disabled=True),
                    "Bars": st.column_config.NumberColumn("Bars", disabled=True),
                    "Days Ago": st.column_config.NumberColumn("Days Ago", disabled=True),
                    "Duration (hrs)": st.column_config.NumberColumn("Duration (hrs)", format="%.2f", disabled=True)
                },
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f'cosine_matches_table_{ticker}'
            )

            new_idx = selected_idx
            if event.selection.rows:
                new_idx = event.selection.rows[0]
                if new_idx >= len(matches_df):
                    new_idx = 0

            if new_idx != st.session_state.get(selection_state_key, 0):
                st.session_state[selection_state_key] = new_idx
                st.rerun()

components.html(
    """
    <script>
    (function syncTabs() {
        const rootDoc = window.parent?.document;
        if (!rootDoc) {
            return;
        }
        const tabBlocks = Array.from(rootDoc.querySelectorAll('[data-testid="stTabs"]'));
        const mainTabs = tabBlocks.find(block => !block.closest('[data-testid="stSidebar"]'));
        const sidebarTabs = tabBlocks.find(block => block.closest('[data-testid="stSidebar"]'));
        if (!mainTabs || !sidebarTabs) {
            window.setTimeout(syncTabs, 500);
            return;
        }

        const hookButtons = (sourceButtons, targetButtons) => {
            sourceButtons.forEach((btn, idx) => {
                btn.addEventListener('click', () => {
                    window.setTimeout(() => {
                        const target = targetButtons[idx];
                        if (target && target.getAttribute('aria-selected') !== 'true') {
                            target.click();
                        }
                    }, 0);
                });
            });
        };

        const mainButtons = mainTabs.querySelectorAll('button[role="tab"]');
        const sidebarButtons = sidebarTabs.querySelectorAll('button[role="tab"]');

        if (mainButtons.length !== sidebarButtons.length) {
            return;
        }

        hookButtons(mainButtons, sidebarButtons);
        hookButtons(sidebarButtons, mainButtons);
    })();
    </script>
    """,
    height=0,
)
