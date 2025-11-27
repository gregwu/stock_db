#!/usr/bin/env python3
"""
Alpaca Strategy Trader - Automated trading with Alpaca
Monitors strategy signals and executes real trades via Alpaca API
"""
import os
import time
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import logging

# Import Alpaca
from alpaca_wrapper import AlpacaAPI

# Import functions from rules.py
from rules import (
    rsi, ema, bollinger_bands, macd, backtest_symbol
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_trader.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Email Configuration
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')

# Alpaca Configuration from alpaca_config.py
try:
    from alpaca_config import USE_PAPER, POSITION_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
except ImportError:
    # Defaults if config doesn't exist
    USE_PAPER = True  # SAFETY: Default to paper trading
    POSITION_SIZE = 10
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.03

# Tracking files
STATE_FILE = '.alpaca_trader_state.json'
CONFIG_FILE = 'alpaca.json'  # Unified configuration file
PORTFOLIO_STATE_FILE = 'alpaca_portfolio_state.json'

# Alpaca API instance
alpaca_api = None

# Signal actions configuration (loaded at startup)
signal_actions_config = None


def load_config():
    """
    Load unified configuration from alpaca.json
    Returns tuple: (strategy_settings, signal_actions)
    Fails loudly if configuration is missing or invalid
    """
    if not os.path.exists(CONFIG_FILE):
        error_msg = f"❌ CONFIGURATION ERROR: {CONFIG_FILE} not found!"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logging.info(f"✅ Loaded configuration from {CONFIG_FILE}")

        # Extract strategy settings
        strategy = config.get('strategy', {})
        signal_actions = config.get('signal_actions', {})
        workflow = config.get('workflow', {})

        if not strategy:
            error_msg = f"❌ CONFIGURATION ERROR: 'strategy' section missing in {CONFIG_FILE}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if not signal_actions:
            error_msg = f"❌ CONFIGURATION ERROR: 'signal_actions' section missing in {CONFIG_FILE}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Add workflow to signal_actions for compatibility
        signal_actions['workflow'] = workflow

        return strategy, signal_actions
    except json.JSONDecodeError as e:
        error_msg = f"❌ CONFIGURATION ERROR: Invalid JSON in {CONFIG_FILE}: {e}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"❌ CONFIGURATION ERROR: Failed to load {CONFIG_FILE}: {e}"
        logging.error(error_msg)
        raise


def load_alpaca_settings():
    """
    Load strategy settings and return flattened dictionary
    This function maintains compatibility with existing code
    """
    strategy, _ = load_config()

    if not strategy:
        return None

    # Flatten the nested structure into a flat settings dictionary
    settings = {
        'tickers': strategy.get('tickers', None),
        'ticker': strategy.get('ticker', 'TQQQ'),  # Legacy single ticker
        'interval': strategy.get('interval', '5m'),
        'period': strategy.get('period', '1d'),
    }

    # Entry conditions
    entry = strategy.get('entry_conditions', {})
    settings['use_rsi'] = entry.get('use_rsi', False)
    settings['rsi_threshold'] = entry.get('rsi_threshold', 30)
    settings['use_ema_cross_up'] = entry.get('use_ema_cross_up', False)
    settings['use_bb_cross_up'] = entry.get('use_bb_cross_up', False)
    settings['use_macd_cross_up'] = entry.get('use_macd_cross_up', False)
    settings['use_price_vs_ema9'] = entry.get('use_price_vs_ema9', False)
    settings['use_price_vs_ema21'] = entry.get('use_price_vs_ema21', False)
    settings['use_macd_threshold'] = entry.get('use_macd_threshold', False)
    settings['macd_threshold'] = entry.get('macd_threshold', 0)
    settings['use_macd_valley'] = entry.get('use_macd_valley', False)
    settings['use_ema'] = entry.get('use_ema', False)
    settings['use_volume'] = entry.get('use_volume', False)

    # Exit conditions
    exit_cond = strategy.get('exit_conditions', {})
    settings['use_rsi_exit'] = exit_cond.get('use_rsi_exit', False)
    settings['rsi_exit_threshold'] = exit_cond.get('rsi_exit_threshold', 70)
    settings['use_ema_cross_down'] = exit_cond.get('use_ema_cross_down', False)
    settings['use_bb_cross_down'] = exit_cond.get('use_bb_cross_down', False)
    settings['use_macd_cross_down'] = exit_cond.get('use_macd_cross_down', False)
    settings['use_price_vs_ema9_exit'] = exit_cond.get('use_price_vs_ema9_exit', False)
    settings['use_price_vs_ema21_exit'] = exit_cond.get('use_price_vs_ema21_exit', False)
    settings['use_macd_peak'] = exit_cond.get('use_macd_peak', False)

    # Risk management
    risk = strategy.get('risk_management', {})
    settings['stop_loss'] = risk.get('stop_loss', 0.02)
    settings['take_profit'] = risk.get('take_profit', 0.03)
    settings['use_stop_loss'] = risk.get('use_stop_loss', True)
    settings['use_take_profit'] = risk.get('use_take_profit', True)

    # Trading settings
    trading = strategy.get('trading', {})
    settings['position_size'] = trading.get('position_size', 10)
    settings['check_interval_seconds'] = trading.get('check_interval_seconds', 300)

    # Time filter settings
    time_filter = strategy.get('time_filter', {})
    settings['use_time_filter'] = time_filter.get('use_time_filter', False)
    settings['avoid_after_time'] = time_filter.get('avoid_after_time', '15:00')

    return settings


def load_signal_actions():
    """
    Load signal actions configuration
    This function maintains compatibility with existing code
    """
    _, signal_actions = load_config()
    return signal_actions


def load_state():
    """Load the last known state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            # Migrate old state format to new format
            if 'positions' not in state:
                # Convert single position to multi-position format
                state['positions'] = {}
                if state.get('current_position'):
                    state['positions'][state['current_position']] = {
                        'ticker': state['current_position'],
                        'entry_price': state.get('entry_price'),
                        'entry_time': state.get('entry_time'),
                        'entry_conditions': state.get('entry_conditions'),
                        'quantity': state.get('position_size', 0)
                    }
            return state
    return {
        'last_check_time': None,
        'current_position': None,  # Legacy - for backward compatibility
        'entry_price': None,        # Legacy - for backward compatibility
        'entry_time': None,         # Legacy - for backward compatibility
        'entry_conditions': None,   # Legacy - for backward compatibility
        'position_size': 0,         # Legacy - for backward compatibility
        'order_ids': [],
        'positions': {}  # New multi-ticker tracking: {ticker: {entry_price, entry_time, quantity, ...}}
    }


def save_state(state):
    """Save current state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def send_email_alert(subject, message):
    """Send alert via email"""
    if not all([GMAIL_ADDRESS, GMAIL_APP_PASSWORD]):
        logging.warning("Gmail credentials not configured. Alert not sent.")
        return False

    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = GMAIL_ADDRESS

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        logging.info(f"Alert sent: {subject}")
        return True
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")
        return False


def save_portfolio_state():
    """Save current portfolio positions to state file"""
    try:
        positions = alpaca_api.get_positions()
        portfolio_state = {
            'timestamp': datetime.now().isoformat(),
            'positions': []
        }

        for pos in positions:
            portfolio_state['positions'].append({
                'ticker': pos['symbol'],
                'quantity': pos['qty'],
                'cost_basis': pos['cost_basis'],
                'current_price': pos['current_price'],
                'market_value': pos['market_value'],
                'unrealized_pnl': pos['unrealized_pl'],
                'unrealized_pnl_pct': pos['unrealized_plpc']
            })

        # Save to portfolio state file
        with open('.alpaca_portfolio_state.json', 'w') as f:
            json.dump(portfolio_state, f, indent=2)

        logging.info(f"Portfolio saved: {len(portfolio_state['positions'])} positions")
        return portfolio_state

    except Exception as e:
        logging.error(f"Failed to save portfolio: {e}")
        return None


def sell_all_positions():
    """Sell all current positions in portfolio"""
    try:
        # First check for any pending orders
        logging.info("Checking for pending orders...")
        current_orders = alpaca_api.get_current_orders()

        if current_orders:
            logging.warning(f"Found {len(current_orders)} pending orders")
            for order in current_orders:
                order_id = order['order_id']
                ticker = order['symbol']
                action = order['side']
                status = order['status']
                logging.info(f"  Order {order_id}: {action} {ticker} - Status: {status}")

            # Cancel pending orders before selling
            logging.info("Cancelling pending orders...")
            for order in current_orders:
                order_id = order['order_id']
                try:
                    alpaca_api.cancel(order_id)
                    logging.info(f"  Cancelled order {order_id}")
                except Exception as cancel_error:
                    logging.warning(f"  Could not cancel order {order_id}: {cancel_error}")

            # Wait a moment for cancellations to process
            time.sleep(2)

        # Now get current positions
        positions = alpaca_api.get_positions()

        if not positions:
            logging.info("No existing positions to sell")
            return True

        logging.info(f"Found {len(positions)} positions to sell")

        for pos in positions:
            ticker = pos['symbol']
            quantity = int(pos['qty'])

            if quantity > 0:
                logging.info(f"Selling existing position: {quantity} shares of {ticker}")
                current_price = pos['current_price']

                order = place_sell_order(ticker, quantity, current_price, "Clear existing position")

                if order:
                    logging.info(f"Successfully placed sell order for {ticker}")
                else:
                    logging.error(f"Failed to sell {ticker}")
                    return False

        return True

    except Exception as e:
        logging.error(f"Failed to sell positions: {e}")
        return False


def initialize_alpaca():
    """Initialize and connect to Alpaca"""
    global alpaca_api

    try:
        alpaca_api = AlpacaAPI(paper=USE_PAPER)

        if not alpaca_api.login():
            return False

        account_type = "PAPER" if USE_PAPER else "LIVE"
        logging.info(f"Alpaca connected: {account_type} account")

        # Get account info
        account = alpaca_api.get_account()
        portfolio_value = float(account.portfolio_value) if account else 0
        buying_power = float(account.buying_power) if account else 0
        cash = float(account.cash) if account else 0
        equity = float(account.equity) if account else 0

        # Load strategy settings
        settings = load_alpaca_settings()

        # Save initial portfolio state
        portfolio = save_portfolio_state()

        # Build position summary
        position_summary = []
        if portfolio and portfolio['positions']:
            for pos in portfolio['positions']:
                position_summary.append(f"  {pos['ticker']}: {pos['quantity']} shares @ ${pos['current_price']:.2f} (P&L: ${pos['unrealized_pnl']:.2f})")

        # Build settings summary
        settings_summary = ""
        if settings:
            settings_summary = f"""
═══════════════════════════════════════
STRATEGY SETTINGS
═══════════════════════════════════════
Ticker: {settings.get('ticker', 'TQQQ')}
Interval: {settings.get('interval', '5m')}
Period: {settings.get('period', '1d')}

Entry Conditions:"""
            if settings.get('use_rsi'):
                settings_summary += f"\n  - RSI < {settings.get('rsi_threshold', 30)}"
            if settings.get('use_ema_cross_up'):
                settings_summary += f"\n  - EMA9 cross above EMA21"
            if settings.get('use_price_vs_ema21'):
                settings_summary += f"\n  - Price > EMA21"
            if settings.get('use_macd_valley'):
                settings_summary += f"\n  - MACD Valley (turning up)"

            settings_summary += "\n\nExit Conditions:"
            if settings.get('use_rsi_exit'):
                settings_summary += f"\n  - RSI > {settings.get('rsi_exit_threshold', 70)}"
            if settings.get('use_ema_cross_down'):
                settings_summary += f"\n  - EMA9 cross below EMA21"
            if settings.get('use_macd_peak'):
                settings_summary += f"\n  - MACD Peak (turning down)"

            settings_summary += f"\n\nRisk Management:"
            settings_summary += f"\n  - Stop Loss: {settings.get('stop_loss', 0.02)*100:.1f}%"
            settings_summary += f"\n  - Take Profit: {settings.get('take_profit', 0.03)*100:.1f}%"

        message = f"""🤖 ALPACA TRADING BOT STARTED

═══════════════════════════════════════
ACCOUNT INFORMATION
═══════════════════════════════════════
Account Type: {account_type}
Portfolio Value: ${portfolio_value:,.2f}
Equity: ${equity:,.2f}
Cash: ${cash:,.2f}
Buying Power: ${buying_power:,.2f}

═══════════════════════════════════════
TRADING CONFIGURATION
═══════════════════════════════════════
Position Size: {POSITION_SIZE} shares
Stop Loss: {STOP_LOSS_PCT*100:.1f}%
Take Profit: {TAKE_PROFIT_PCT*100:.1f}%
Strategy: TQQQ/SQQQ Pair Trading

Signal Logic:
  Entry Signal → Buy TQQQ
  Exit Signal → Buy SQQQ
{settings_summary}
"""

        if position_summary:
            message += f"""
═══════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════
{chr(10).join(position_summary)}
"""
        else:
            message += f"""
═══════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════
  No open positions
"""

        message += f"""
═══════════════════════════════════════
MONITORING
═══════════════════════════════════════
Check Frequency: Every 5 minutes
Email Alerts: Enabled
Started: {pd.Timestamp.now()}
═══════════════════════════════════════

The bot is now actively monitoring for trading signals.
You will receive email alerts for all trades.
"""

        send_email_alert("🤖 Alpaca Trading Bot Started", message)
        return True

    except Exception as e:
        logging.error(f"Failed to initialize Alpaca: {e}")
        send_email_alert(
            "❌ Trading Bot Error",
            f"Failed to connect to Alpaca: {e}"
        )
        return False


def place_buy_order(ticker, qty, price, reason, entry_conditions=None):
    """Place a buy order"""
    try:
        logging.info(f"Placing BUY order: {qty} {ticker} @ ${price:.2f}")

        # Get account info
        account = alpaca_api.get_account()
        portfolio_value = float(account.portfolio_value) if account else 0
        buying_power = float(account.buying_power) if account else 0
        cash = float(account.cash) if account else 0

        # Get current positions
        positions = alpaca_api.get_positions()
        position_summary = []
        for pos in positions:
            position_summary.append(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f} (P&L: ${pos['unrealized_pl']:.2f})")

        order = alpaca_api.place_order(
            ticker=ticker,
            qty=qty,
            action="BUY",
            order_type="MKT"
        )

        message = f"""✅ BUY ORDER PLACED

═══════════════════════════════════════
ORDER DETAILS
═══════════════════════════════════════
Ticker: {ticker}
Quantity: {qty} shares
Price: ${price:.2f}
Order Value: ${price * qty:.2f}
Order ID: {order['order_id'] if order else 'FAILED'}

═══════════════════════════════════════
CONDITIONS
═══════════════════════════════════════
{reason}
"""
        if entry_conditions:
            message += f"""
═══════════════════════════════════════
PREVIOUS ENTRY CONDITIONS
═══════════════════════════════════════
{entry_conditions}
"""

        message += f"""
═══════════════════════════════════════
ACCOUNT STATUS
═══════════════════════════════════════
Portfolio Value: ${portfolio_value:,.2f}
Cash: ${cash:,.2f}
Buying Power: ${buying_power:,.2f}
"""

        if position_summary:
            message += f"""
═══════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════
{chr(10).join(position_summary)}
"""
        else:
            message += f"""
═══════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════
  No open positions
"""

        message += f"""
═══════════════════════════════════════
Time: {pd.Timestamp.now()}
Account: {'PAPER' if USE_PAPER else 'LIVE'}
═══════════════════════════════════════
"""

        send_email_alert("🟢 BUY ORDER - Alpaca Trading Bot", message)
        return order

    except Exception as e:
        logging.error(f"Failed to place buy order: {e}")
        send_email_alert("❌ ORDER FAILED", f"Failed to place BUY order for {ticker}: {e}")
        return None


def place_sell_order(ticker, qty, price, reason, entry_conditions=None):
    """Place a sell order"""
    try:
        logging.info(f"Placing SELL order: {qty} {ticker} @ ${price:.2f}")

        # Get account info
        account = alpaca_api.get_account()
        portfolio_value = float(account.portfolio_value) if account else 0
        buying_power = float(account.buying_power) if account else 0
        cash = float(account.cash) if account else 0

        # Get current positions
        positions = alpaca_api.get_positions()
        position_summary = []
        for pos in positions:
            position_summary.append(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f} (P&L: ${pos['unrealized_pl']:.2f})")

        order = alpaca_api.place_order(
            ticker=ticker,
            qty=qty,
            action="SELL",
            order_type="MKT"
        )

        message = f"""✅ SELL ORDER PLACED

═══════════════════════════════════════
ORDER DETAILS
═══════════════════════════════════════
Ticker: {ticker}
Quantity: {qty} shares
Price: ${price:.2f}
Order Value: ${price * qty:.2f}
Order ID: {order['order_id'] if order else 'FAILED'}

═══════════════════════════════════════
CONDITIONS
═══════════════════════════════════════
{reason}
"""
        if entry_conditions:
            message += f"""
═══════════════════════════════════════
PREVIOUS ENTRY CONDITIONS
═══════════════════════════════════════
{entry_conditions}
"""

        message += f"""
═══════════════════════════════════════
ACCOUNT STATUS
═══════════════════════════════════════
Portfolio Value: ${portfolio_value:,.2f}
Cash: ${cash:,.2f}
Buying Power: ${buying_power:,.2f}
"""

        if position_summary:
            message += f"""
═══════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════
{chr(10).join(position_summary)}
"""
        else:
            message += f"""
═══════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════
  No open positions
"""

        message += f"""
═══════════════════════════════════════
Time: {pd.Timestamp.now()}
Account: {'PAPER' if USE_PAPER else 'LIVE'}
═══════════════════════════════════════
"""

        send_email_alert("🔴 SELL ORDER - Alpaca Trading Bot", message)
        return order

    except Exception as e:
        logging.error(f"Failed to place sell order: {e}")
        send_email_alert("❌ ORDER FAILED", f"Failed to place SELL order for {ticker}: {e}")
        return None


def check_stop_loss_take_profit(state):
    """Check if stop loss or take profit levels are hit"""
    if not state['current_position'] or not state['entry_price']:
        return False

    try:
        ticker = state['current_position']
        quote = alpaca_api.quote(ticker)

        if not quote:
            return False

        current_price = quote['last']
        entry_price = state['entry_price']

        pnl_pct = (current_price - entry_price) / entry_price

        # Check stop loss
        if pnl_pct <= -STOP_LOSS_PCT:
            logging.warning(f"⚠️  STOP LOSS triggered! P&L: {pnl_pct*100:.2f}%")

            # Sell all existing positions first
            if not sell_all_positions():
                logging.error("Failed to sell positions for stop loss")
                return False

            # Then place opposite order
            new_ticker = 'SQQQ' if ticker == 'TQQQ' else 'TQQQ'
            buy_order = place_buy_order(new_ticker, POSITION_SIZE, current_price,
                                       f"Stop Loss Exit from {ticker}",
                                       state.get('entry_conditions'))

            if buy_order:
                # Update state
                state['current_position'] = new_ticker
                state['entry_price'] = current_price
                state['entry_time'] = str(datetime.now())
                state['entry_conditions'] = f"Stop Loss Exit from {ticker}"
                state['position_size'] = POSITION_SIZE
                save_state(state)
                return True

        # Check take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            logging.info(f"🎯 TAKE PROFIT triggered! P&L: {pnl_pct*100:.2f}%")

            # Sell all existing positions first
            if not sell_all_positions():
                logging.error("Failed to sell positions for take profit")
                return False

            # Then place opposite order
            new_ticker = 'SQQQ' if ticker == 'TQQQ' else 'TQQQ'
            buy_order = place_buy_order(new_ticker, POSITION_SIZE, current_price,
                                       f"Take Profit Exit from {ticker}",
                                       state.get('entry_conditions'))

            if buy_order:
                new_ticker = 'SQQQ' if ticker == 'TQQQ' else 'TQQQ'
                state['current_position'] = new_ticker
                state['entry_price'] = current_price
                state['entry_time'] = str(datetime.now())
                state['entry_conditions'] = f"Take Profit Exit from {ticker}"
                state['position_size'] = POSITION_SIZE
                save_state(state)
                return True

    except Exception as e:
        logging.error(f"Error checking SL/TP: {e}")

    return False


def execute_action(action_config, price, note, timestamp, state):
    """
    Execute a single action based on configuration
    Returns True if successful, False otherwise
    """
    action_type = action_config.get('type', 'BUY')
    ticker = action_config.get('ticker')
    quantity = action_config.get('quantity', POSITION_SIZE)
    description = action_config.get('description', '')

    if action_type == 'BUY':
        if not ticker:
            logging.error(f"      ❌ BUY action missing ticker")
            return False

        logging.info(f"      Executing: BUY {quantity} shares of {ticker}")
        if description:
            logging.info(f"      Reason: {description}")

        order = place_buy_order(ticker, quantity, price, note)
        if order:
            logging.info(f"      ✅ BUY order placed: {quantity} shares of {ticker} @ ${price:.2f}")

            # Update state - both legacy single position and new multi-ticker tracking
            state['current_position'] = ticker
            state['entry_price'] = price
            state['entry_time'] = str(timestamp)
            state['entry_conditions'] = note
            state['position_size'] = quantity
            state['order_ids'].append(order['order_id'])

            # Track multiple positions
            if 'positions' not in state:
                state['positions'] = {}

            # Add or update position for this ticker
            if ticker in state['positions']:
                # Update existing position (average price)
                old_qty = state['positions'][ticker].get('quantity', 0)
                old_price = state['positions'][ticker].get('entry_price', price)
                new_qty = old_qty + quantity
                avg_price = ((old_price * old_qty) + (price * quantity)) / new_qty
                state['positions'][ticker]['quantity'] = new_qty
                state['positions'][ticker]['entry_price'] = avg_price
            else:
                # New position
                state['positions'][ticker] = {
                    'ticker': ticker,
                    'entry_price': price,
                    'entry_time': str(timestamp),
                    'entry_conditions': note,
                    'quantity': quantity
                }

            return True
        else:
            logging.error(f"      ❌ BUY order failed for {ticker}")
            return False

    elif action_type == 'SELL':
        if not ticker:
            logging.error(f"      ❌ SELL action missing ticker")
            return False

        logging.info(f"      Executing: SELL {quantity} shares of {ticker}")
        if description:
            logging.info(f"      Reason: {description}")

        # Check if we have this position
        positions = alpaca_api.get_positions() if alpaca_api else []
        position = next((p for p in positions if p['symbol'] == ticker), None)

        if not position:
            logging.warning(f"      ⚠️  No position in {ticker} to sell")
            return False

        # If quantity is specified, sell that amount; otherwise sell all
        available_qty = int(position['qty'])
        sell_qty = min(quantity, available_qty) if quantity else available_qty

        if sell_qty <= 0:
            logging.warning(f"      ⚠️  No shares to sell for {ticker}")
            return False

        order = place_sell_order(ticker, sell_qty, price, note)
        if order:
            logging.info(f"      ✅ SELL order placed: {sell_qty} shares of {ticker} @ ${price:.2f}")

            # Update multi-ticker tracking
            if 'positions' in state and ticker in state['positions']:
                current_qty = state['positions'][ticker].get('quantity', 0)
                remaining_qty = current_qty - sell_qty

                if remaining_qty <= 0:
                    # Completely sold out of this ticker
                    del state['positions'][ticker]
                    logging.info(f"      📊 Position in {ticker} closed completely")
                else:
                    # Partially sold
                    state['positions'][ticker]['quantity'] = remaining_qty
                    logging.info(f"      📊 Position in {ticker} reduced to {remaining_qty} shares")

            return True
        else:
            logging.error(f"      ❌ SELL order failed for {ticker}")
            return False

    elif action_type == 'SELL_ALL':
        # SELL_ALL can work two ways:
        # 1. With ticker specified: Sell ALL shares of that specific ticker
        # 2. Without ticker: Sell ALL shares of ALL tickers (legacy behavior)

        if ticker:
            # Sell all shares of specific ticker
            logging.info(f"      Executing: SELL ALL shares of {ticker}")

            # Check if we have this position
            positions = alpaca_api.get_positions() if alpaca_api else []
            position = next((p for p in positions if p['symbol'] == ticker), None)

            if not position:
                logging.warning(f"      ⚠️  No position in {ticker} to sell")
                return False

            available_qty = int(position['qty'])
            if available_qty <= 0:
                logging.warning(f"      ⚠️  No shares to sell for {ticker}")
                return False

            logging.info(f"      Found {available_qty} shares of {ticker} to sell")

            order = place_sell_order(ticker, available_qty, price, note)
            if order:
                logging.info(f"      ✅ SELL ALL order placed: {available_qty} shares of {ticker} @ ${price:.2f}")

                # Update multi-ticker tracking - remove this ticker completely
                if 'positions' in state and ticker in state['positions']:
                    del state['positions'][ticker]
                    logging.info(f"      📊 Position in {ticker} closed completely")

                # Update legacy fields if this was the current position
                if state.get('current_position') == ticker:
                    state['current_position'] = None
                    state['entry_price'] = None
                    state['position_size'] = 0

                return True
            else:
                logging.error(f"      ❌ SELL ALL order failed for {ticker}")
                return False

        else:
            # No ticker specified - sell ALL tickers (legacy behavior)
            logging.info(f"      Executing: SELL ALL positions (all tickers)")

            # Check if we have any positions to sell
            positions = alpaca_api.get_positions() if alpaca_api else []

            if not positions:
                logging.warning(f"      ⚠️  No positions to sell - portfolio is empty")
                logging.info(f"      Skipping SELL_ALL (nothing to sell)")

                # Still clear tracking to ensure state is clean
                if 'positions' in state:
                    state['positions'] = {}
                state['current_position'] = None
                state['entry_price'] = None
                state['position_size'] = 0

                return True  # Not an error - just nothing to do

            logging.info(f"      Found {len(positions)} position(s) to sell:")
            for pos in positions:
                logging.info(f"        - {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f}")

            result = sell_all_positions()
            if result:
                logging.info(f"      ✅ All positions sold")

                # Clear multi-ticker tracking
                if 'positions' in state:
                    state['positions'] = {}
                    logging.info(f"      📊 All position tracking cleared")

                # Clear legacy fields
                state['current_position'] = None
                state['entry_price'] = None
                state['position_size'] = 0

                return True
            else:
                logging.error(f"      ❌ Failed to sell all positions")
                return False

    elif action_type == 'NOTIFY':
        message = action_config.get('message', 'Signal detected')
        logging.info(f"      📧 NOTIFY: {message}")
        # Could send email/SMS here if implemented
        return True

    elif action_type == 'LOG':
        message = action_config.get('message', 'Action logged')
        logging.info(f"      📝 LOG: {message}")
        return True

    else:
        logging.error(f"      ❌ Unknown action type: {action_type}")
        return False


def process_signal_with_config(event, price, note, timestamp, state, ticker=None):
    """
    Process a signal based on configuration from alpaca_signal_actions.json
    Supports multiple actions per signal and ticker-specific actions

    Args:
        event: Signal type (entry, exit_conditions_met, etc.)
        price: Current price
        note: Signal description
        timestamp: Signal timestamp
        state: Bot state dictionary
        ticker: Ticker that generated the signal (optional)

    Returns:
        True if signal was processed, False otherwise
    """
    global signal_actions_config

    if not signal_actions_config:
        logging.warning("No signal actions config loaded, using hardcoded defaults")
        return False

    workflow = signal_actions_config.get('workflow', {})

    # NEW: Look up ticker-specific signal actions first
    signal_config = None
    ticker_enabled = True  # Default to enabled

    if ticker:
        # Try to find ticker-specific configuration
        ticker_configs = signal_actions_config.get('tickers', {})
        if ticker in ticker_configs:
            ticker_config = ticker_configs[ticker]

            # Check if ticker is enabled at ticker level
            ticker_enabled = ticker_config.get('enabled', True)
            if not ticker_enabled:
                logging.info(f"      Ticker {ticker} is DISABLED at ticker level")
                logging.info(f"      Action: IGNORE")
                logging.info("=" * 60)
                return False

            signal_config = ticker_config.get('signal_actions', ticker_config)
            logging.info(f"      Using ticker-specific config for {ticker}")
        else:
            logging.info(f"      No config for {ticker}, using default")

    # Fallback to old format or default
    if not signal_config:
        # Try old format first (backward compatibility)
        signal_config = signal_actions_config.get('signal_actions', {})

        # If still not found, use default (new format) or default_signal_actions (legacy)
        if not signal_config:
            signal_config = signal_actions_config.get('default', {})
            if not signal_config:
                signal_config = signal_actions_config.get('default_signal_actions', {})
            logging.info(f"      Using default signal actions")

    # Check if this signal type is configured
    if event not in signal_config:
        logging.warning(f"      Signal type '{event}' not found in configuration")
        return False

    config = signal_config[event]

    # Check if signal is enabled
    if not config.get('enabled', False):
        logging.info(f"      Signal type: {event} (DISABLED)")
        logging.info(f"      Action: IGNORE")
        logging.info("=" * 60)
        return False

    # Get actions list (new format) or single action (old format for backward compatibility)
    actions = config.get('actions', [])

    # Backward compatibility: if old format (single action/ticker), convert to new format
    if not actions and config.get('action'):
        old_action = config.get('action')
        old_ticker = config.get('ticker')
        if old_action != 'IGNORE':
            actions = [{
                'type': old_action,
                'ticker': old_ticker,
                'quantity': POSITION_SIZE,
                'description': config.get('description', '')
            }]

    # If no actions, treat as ignore
    if not actions:
        logging.info(f"      Signal type: {event} (NO ACTIONS CONFIGURED)")
        logging.info(f"      Action: IGNORE")
        logging.info("=" * 60)
        return False

    # Log signal classification
    logging.info(f"      Signal type: {event}")
    logging.info(f"      Actions configured: {len(actions)}")

    # Calculate P&L if exiting a position
    if workflow.get('log_pnl_on_exit', True) and event in ['exit_SL', 'exit_TP', 'exit_conditions_met']:
        if state.get('current_position') and state.get('entry_price'):
            pnl_pct = ((price - state['entry_price']) / state['entry_price']) * 100
            pnl_dollars = (price - state['entry_price']) * state.get('position_size', 0)
            logging.info(f"      Trade P&L: ${pnl_dollars:.2f} ({pnl_pct:+.2f}%)")

    logging.info("")

    # [2] Save Portfolio State
    if workflow.get('save_state_before_actions', True):
        logging.info("[2/4] Saving portfolio state...")
        save_portfolio_state()
        logging.info("      ✅ Portfolio state saved")
        logging.info("")

    # [3] Clear Existing Positions
    if workflow.get('clear_positions_before_actions', True):
        logging.info("[3/4] Clearing existing positions...")
        if not sell_all_positions():
            logging.error("      ❌ Failed to sell existing positions")
            logging.error("      ABORT: Skipping actions")
            logging.info("=" * 60)
            return False
        else:
            logging.info("      ✅ All positions cleared")
            logging.info("")

    # [4] Execute Actions
    logging.info(f"[4/4] Executing {len(actions)} action(s)...")
    logging.info("")

    success_count = 0
    for idx, action_config in enumerate(actions, 1):
        logging.info(f"      Action {idx}/{len(actions)}:")
        if execute_action(action_config, price, note, timestamp, state):
            success_count += 1
        logging.info("")

    # Save state after all actions
    save_state(state)

    logging.info(f"      Summary: {success_count}/{len(actions)} actions completed successfully")
    logging.info("=" * 60)

    return success_count > 0


def run_strategy():
    """Run the trading strategy and check for signals"""
    logging.info("=" * 60)
    logging.info("Running strategy check...")

    state = load_state()

    # Check stop loss / take profit first
    if check_stop_loss_take_profit(state):
        logging.info("SL/TP executed, skipping regular strategy check")
        return

    settings = load_alpaca_settings()

    if not settings:
        logging.error(f"No settings file found. Please create {CONFIG_FILE}")
        return

    # Support both old format (single ticker) and new format (multiple tickers)
    tickers = settings.get('tickers', None)
    if tickers is None:
        # Fallback to old 'ticker' field for backward compatibility
        ticker = settings.get('ticker', 'TQQQ')
        tickers = [ticker]

    if not isinstance(tickers, list):
        tickers = [tickers]

    interval = settings.get('interval', '5m')
    period = settings.get('period', '1d')

    # Filter to only enabled tickers for display
    ticker_configs = signal_actions_config.get('tickers', {})
    enabled_tickers = [t for t in tickers if ticker_configs.get(t, {}).get('enabled', True)]
    disabled_tickers = [t for t in tickers if not ticker_configs.get(t, {}).get('enabled', True)]

    if enabled_tickers:
        logging.info(f"✅ Monitoring {len(enabled_tickers)} enabled ticker(s): {', '.join(enabled_tickers)}")
    if disabled_tickers:
        logging.info(f"⚠️  Skipping {len(disabled_tickers)} disabled ticker(s): {', '.join(disabled_tickers)}")

    # Check each ticker for signals
    for ticker in tickers:
        # Check if ticker is enabled in signal_actions config FIRST
        ticker_configs = signal_actions_config.get('tickers', {})
        if ticker in ticker_configs:
            ticker_enabled = ticker_configs[ticker].get('enabled', True)
            if not ticker_enabled:
                # Skip disabled ticker silently (already shown in summary above)
                continue

        logging.info("-" * 60)
        logging.info(f"Checking {ticker}...")

        # Download recent data
        try:
            logging.info(f"Downloading data for {ticker}...")
            download_period = "5d"  # Always download 5 days for context
            use_extended_hours = interval not in ["1d", "5d", "1wk", "1mo", "3mo"]
            raw = yf.download(ticker, period=download_period, interval=interval,
                             progress=False, prepost=use_extended_hours)

            if raw.empty:
                logging.error(f"No data returned for {ticker}")
                continue  # Skip to next ticker

            # Handle MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw.copy()

            # Run backtest to get latest signal
            logging.info(f"Running backtest for {ticker}...")
            df1, df5, trades_df, logs_df = backtest_symbol(
                df1=df,
                rsi_threshold=settings.get('rsi_threshold', 30),
                use_rsi=settings.get('use_rsi', True),
                use_ema=settings.get('use_ema', True),
                use_volume=settings.get('use_volume', True),
                stop_loss=settings.get('stop_loss', 0.02),
                tp_pct=settings.get('take_profit', 0.03),
                avoid_after=None,  # Time filter disabled
                use_stop_loss=settings.get('use_stop_loss', True),
                use_take_profit=settings.get('use_take_profit', True),
                use_rsi_overbought=settings.get('use_rsi_exit', False),
                rsi_overbought_threshold=settings.get('rsi_exit_threshold', 70),
                use_ema_cross_up=settings.get('use_ema_cross_up', False),
                use_ema_cross_down=settings.get('use_ema_cross_down', False),
                use_price_below_ema9=settings.get('use_price_vs_ema9_exit', False),
                use_bb_cross_up=settings.get('use_bb_cross_up', False),
                use_bb_cross_down=settings.get('use_bb_cross_down', False),
                use_macd_cross_up=settings.get('use_macd_cross_up', False),
                use_macd_cross_down=settings.get('use_macd_cross_down', False),
                use_price_above_ema21=settings.get('use_price_vs_ema21', False),
                use_price_below_ema21=settings.get('use_price_vs_ema21_exit', False),
                use_macd_below_threshold=settings.get('use_macd_threshold', False),
                macd_below_threshold=settings.get('macd_threshold', 0),
                use_macd_above_threshold=settings.get('use_macd_threshold', False),
                macd_above_threshold=settings.get('macd_threshold', 0),
                use_macd_peak=settings.get('use_macd_peak', False),
                use_macd_valley=settings.get('use_macd_valley', False)
            )

            # Get current price from DataFrame
            current_price = df['Close'].iloc[-1]

            # Check for recent signals (last 5 minutes)
            if not logs_df.empty:
                now = pd.Timestamp.now(tz=logs_df['time'].iloc[-1].tz)
                recent_logs = logs_df[logs_df['time'] >= now - pd.Timedelta(minutes=5)]

                if not recent_logs.empty:
                    latest_log = recent_logs.iloc[-1]
                    event = latest_log['event']
                    price = latest_log['price']
                    note = latest_log['note']
                    timestamp = latest_log['time']

                    logging.info(f"Latest signal for {ticker}: {event}")
                    logging.info(f"Time: {timestamp}")
                    logging.info(f"Price: ${price:.2f}")
                    logging.info(f"Details: {note}")

                    # Create unique check key per ticker to track signals independently
                    ticker_check_key = f'last_check_{ticker}'
                    if ticker_check_key not in state:
                        state[ticker_check_key] = None

                    # Check if this is a new signal (not already processed)
                    if state[ticker_check_key] != str(timestamp):
                        logging.info("=" * 60)
                        logging.info(f"🔔 NEW SIGNAL DETECTED FOR {ticker}")
                        logging.info("=" * 60)

                        # [1] Signal Classification
                        logging.info("[1/4] Classifying signal...")

                        # Add ticker metadata to note
                        note_with_ticker = f"[{ticker}] {note}"

                        # Process signal using configuration (pass ticker for ticker-specific actions)
                        if process_signal_with_config(event, price, note_with_ticker, timestamp, state, ticker=ticker):
                            # Update last check time for this ticker only if signal was processed
                            state[ticker_check_key] = str(timestamp)
                            save_state(state)
                    else:
                        logging.info(f"Signal for {ticker} already processed")
                else:
                    # No signals in last 5 minutes - show the most recent one for info
                    if len(logs_df) > 0:
                        last_signal = logs_df.iloc[-1]
                        time_diff = now - last_signal['time']
                        logging.info(f"No recent signals for {ticker} (last 5 minutes)")
                        logging.info(f"Most recent signal: {last_signal['event']} at {last_signal['time']} ({time_diff.total_seconds()/60:.1f} minutes ago)")
                    else:
                        logging.info(f"No signals in backtest logs for {ticker}")
            else:
                logging.info(f"No signals in backtest logs for {ticker}")

        except Exception as e:
            logging.error(f"Error running strategy for {ticker}: {e}", exc_info=True)
            send_email_alert("❌ Strategy Error", f"Error for {ticker}: {e}")


def display_settings(settings):
    """Display strategy settings in a formatted way"""
    if not settings:
        logging.warning("No settings file found. Using defaults.")
        return

    logging.info("=" * 60)
    logging.info("STRATEGY SETTINGS")
    logging.info("=" * 60)

    # Basic settings - support both old and new format
    tickers = settings.get('tickers', None)
    if tickers is None:
        # Fallback to old 'ticker' field
        ticker = settings.get('ticker', 'N/A')
        logging.info(f"Ticker: {ticker}")
    else:
        if isinstance(tickers, list):
            logging.info(f"Tickers: {', '.join(tickers)} ({len(tickers)} total)")
        else:
            logging.info(f"Tickers: {tickers}")

    logging.info(f"Interval: {settings.get('interval', 'N/A')}")
    logging.info(f"Period: {settings.get('period', 'N/A')}")

    # Entry conditions
    logging.info("\nEntry Conditions:")
    if settings.get('use_rsi', False):
        logging.info(f"  - RSI < {settings.get('rsi_threshold', 30)}")
    if settings.get('use_ema_cross_up', False):
        logging.info(f"  - EMA9 cross above EMA21")
    if settings.get('use_bb_cross_up', False):
        logging.info(f"  - Price cross above BB upper")
    if settings.get('use_macd_cross_up', False):
        logging.info(f"  - MACD cross above signal")
    if settings.get('use_price_vs_ema9', False):
        logging.info(f"  - Price > EMA9")
    if settings.get('use_price_vs_ema21', False):
        logging.info(f"  - Price > EMA21")
    if settings.get('use_macd_threshold', False):
        logging.info(f"  - MACD > {settings.get('macd_threshold', 0)}")
    if settings.get('use_macd_valley', False):
        logging.info(f"  - MACD Valley (turning up)")

    # Exit conditions
    logging.info("\nExit Conditions:")
    if settings.get('use_rsi_exit', False):
        logging.info(f"  - RSI > {settings.get('rsi_exit_threshold', 70)}")
    if settings.get('use_ema_cross_down', False):
        logging.info(f"  - EMA9 cross below EMA21")
    if settings.get('use_bb_cross_down', False):
        logging.info(f"  - Price cross below BB lower")
    if settings.get('use_macd_cross_down', False):
        logging.info(f"  - MACD cross below signal")
    if settings.get('use_price_vs_ema9_exit', False):
        logging.info(f"  - Price < EMA9")
    if settings.get('use_price_vs_ema21_exit', False):
        logging.info(f"  - Price < EMA21")
    if settings.get('use_macd_peak', False):
        logging.info(f"  - MACD Peak (turning down)")

    # Risk management
    logging.info("\nRisk Management:")
    logging.info(f"  - Stop Loss: {settings.get('stop_loss', 0)*100:.1f}%")
    logging.info(f"  - Take Profit: {settings.get('take_profit', 0)*100:.1f}%")

    logging.info("=" * 60)


def display_signal_actions(signal_actions):
    """Display signal-to-action mappings in a formatted way"""
    if not signal_actions:
        logging.warning("No signal actions configuration found. Using defaults.")
        return

    logging.info("=" * 60)
    logging.info("SIGNAL → ACTION MAPPINGS")
    logging.info("=" * 60)

    # NEW FORMAT: Display ticker-specific signal actions
    ticker_configs = signal_actions.get('tickers', {})
    if ticker_configs:
        logging.info("\nTicker-Specific Actions:")
        for ticker, ticker_config in ticker_configs.items():
            # Check ticker-level enabled flag
            ticker_enabled = ticker_config.get('enabled', True)
            default_qty = ticker_config.get('default_quantity', 'N/A')
            enabled_icon = "✅" if ticker_enabled else "❌"

            logging.info(f"\n  [{ticker}] {enabled_icon} (Enabled: {ticker_enabled}, Default Qty: {default_qty})")

            # If ticker is disabled, skip showing signal details
            if not ticker_enabled:
                logging.info(f"    ⚠️  All signals DISABLED at ticker level")
                continue

            # Get signal configs - check both old format (signal_actions key) and new format (top level)
            signal_config = ticker_config.get('signal_actions', ticker_config)

            # Iterate through signal types, skipping ticker-level settings
            for signal_type, config in signal_config.items():
                # Skip ticker-level settings
                if signal_type in ['enabled', 'default_quantity']:
                    continue
                enabled_status = "✅" if config.get('enabled', False) else "❌"
                actions = config.get('actions', [])

                if not actions or not config.get('enabled', False):
                    logging.info(f"    {enabled_status} {signal_type:20s} → NO ACTIONS")
                elif len(actions) == 1:
                    act = actions[0]
                    qty_str = f"{act.get('quantity', '')}" if act.get('quantity') else "ALL"
                    action_str = f"{act.get('type', 'BUY')} {qty_str} {act.get('ticker', '?')}"
                    logging.info(f"    {enabled_status} {signal_type:20s} → {action_str}")
                else:
                    logging.info(f"    {enabled_status} {signal_type:20s} → {len(actions)} actions:")
                    for idx, act in enumerate(actions, 1):
                        qty_str = f"{act.get('quantity', '')}" if act.get('quantity') else "ALL"
                        action_str = f"{act.get('type', 'BUY')} {qty_str} {act.get('ticker', '?')}"
                        logging.info(f"        {idx}. {action_str}")

        # Display default actions if present
        default_config = signal_actions.get('default', {})
        if not default_config:
            default_config = signal_actions.get('default_signal_actions', {})  # Legacy
        if default_config:
            logging.info(f"\n  [DEFAULT] (fallback)")
            for signal_type, config in default_config.items():
                enabled_status = "✅" if config.get('enabled', False) else "❌"
                actions = config.get('actions', [])
                if not actions:
                    logging.info(f"    {enabled_status} {signal_type:20s} → NO ACTIONS")

    # OLD FORMAT: Backward compatibility
    else:
        signal_config = signal_actions.get('signal_actions', {})
        for signal_type, config in signal_config.items():
            enabled_status = "✅" if config.get('enabled', False) else "❌"

            # Support both old format (single action) and new format (multiple actions)
            actions = config.get('actions', [])

            # Backward compatibility with old format
            if not actions and config.get('action'):
                old_action = config.get('action')
                old_ticker = config.get('ticker')
                if old_action == 'IGNORE' or not config.get('enabled', False):
                    logging.info(f"  {enabled_status} {signal_type:20s} → IGNORE")
                else:
                    logging.info(f"  {enabled_status} {signal_type:20s} → {old_action} {old_ticker}")
            elif not actions or not config.get('enabled', False):
                logging.info(f"  {enabled_status} {signal_type:20s} → NO ACTIONS")
            elif len(actions) == 1:
                # Single action - show on one line
                act = actions[0]
                action_str = f"{act.get('type', 'BUY')} {act.get('quantity', '?')} {act.get('ticker', '?')}"
                logging.info(f"  {enabled_status} {signal_type:20s} → {action_str}")
            else:
                # Multiple actions - show count and list
                logging.info(f"  {enabled_status} {signal_type:20s} → {len(actions)} actions:")
                for idx, act in enumerate(actions, 1):
                    action_str = f"{act.get('type', 'BUY')} {act.get('quantity', '?')} {act.get('ticker', '?')}"
                    logging.info(f"      {idx}. {action_str}")

    # Display workflow settings
    workflow = signal_actions.get('workflow', {})
    logging.info("\nWorkflow Settings:")
    logging.info(f"  - Clear positions before actions: {workflow.get('clear_positions_before_actions', True)}")
    logging.info(f"  - Save state before actions: {workflow.get('save_state_before_actions', True)}")
    logging.info(f"  - Log P&L on exit: {workflow.get('log_pnl_on_exit', True)}")

    logging.info("=" * 60)


def main():
    """Main trading loop"""
    logging.info("=" * 60)
    logging.info("ALPACA STRATEGY TRADER - AUTOMATED PAIR TRADING")
    logging.info("=" * 60)

    account_type = "PAPER TRADING" if USE_PAPER else "⚠️  LIVE TRADING ⚠️"
    logging.info(f"Mode: {account_type}")
    logging.info(f"Position Size: {POSITION_SIZE} shares")
    logging.info(f"Strategy: TQQQ (long) / SQQQ (short)")
    logging.info(f"Email: {GMAIL_ADDRESS}")
    logging.info(f"Stop Loss: {STOP_LOSS_PCT*100:.1f}%")
    logging.info(f"Take Profit: {TAKE_PROFIT_PCT*100:.1f}%")
    logging.info("Press Ctrl+C to stop")
    logging.info("=" * 60)
    logging.info("")

    # Load and display strategy settings
    settings = load_alpaca_settings()
    display_settings(settings)
    logging.info("")

    # Load signal-to-action mappings
    global signal_actions_config
    signal_actions_config = load_signal_actions()
    display_signal_actions(signal_actions_config)
    logging.info("")

    # Initialize Alpaca
    if not initialize_alpaca():
        logging.error("Cannot proceed without Alpaca connection")
        return

    # Get check interval from config
    check_interval = settings.get('check_interval_seconds', 120) if settings else 120
    logging.info(f"Check interval: {check_interval} seconds ({check_interval/60:.1f} minutes)")
    logging.info("")

    # Run immediately on start
    run_strategy()

    # Then run at configured interval
    try:
        while True:
            time.sleep(check_interval)
            run_strategy()
    except KeyboardInterrupt:
        logging.info("\nAlpaca Strategy Trader Stopped")

        # Build stop email with enabled tickers info
        tickers = settings.get('tickers', []) if settings else []
        ticker_configs = signal_actions_config.get('tickers', {})
        enabled_tickers = [t for t in tickers if ticker_configs.get(t, {}).get('enabled', True)]
        disabled_tickers = [t for t in tickers if not ticker_configs.get(t, {}).get('enabled', True)]

        enabled_list = ', '.join(enabled_tickers) if enabled_tickers else 'None'
        disabled_list = ', '.join(disabled_tickers) if disabled_tickers else 'None'

        stop_message = f"""Alpaca Trading Bot Stopped

Mode: {account_type}
Monitored Tickers: {enabled_list}
Disabled Tickers: {disabled_list}
Stop Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot has been stopped manually."""

        send_email_alert("🛑 Trading Bot Stopped", stop_message)


if __name__ == "__main__":
    main()
