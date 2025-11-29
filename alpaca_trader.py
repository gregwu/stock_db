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
import signal
import sys
import pytz

# Import Alpaca
from alpaca_wrapper import AlpacaAPI

# Import functions from rules.py
from rules import (
    rsi, ema, bollinger_bands, macd, backtest_symbol, load_strategy_from_alpaca
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

# Default configuration values (will be overridden by alpaca.json)
USE_PAPER = True  # SAFETY: Default to paper trading
POSITION_SIZE = 100
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.03
MAX_BUY_SLIPPAGE_PCT = 0.5  # Skip buy if price rose > 0.5%
MAX_SELL_SLIPPAGE_PCT = 1.0  # Skip sell if price dropped > 1.0%
EMAIL_NOTIFICATIONS_ENABLED = True
EMAIL_ON_BOT_START = True
EMAIL_ON_BOT_STOP = True
EMAIL_ON_ENTRY = True
EMAIL_ON_EXIT = True
EMAIL_ON_ERRORS = True

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


def get_ticker_strategy(ticker, default_strategy, signal_actions_config):
    """
    Get strategy settings for a specific ticker, with ticker-level overrides

    Args:
        ticker: Ticker symbol (e.g., 'TQQQ')
        default_strategy: Default strategy settings from strategy section
        signal_actions_config: Signal actions config containing ticker-specific overrides

    Returns:
        Merged strategy settings for the ticker
    """
    # Start with default strategy
    strategy = default_strategy.copy()

    # Check if ticker has strategy overrides
    ticker_configs = signal_actions_config.get('tickers', {})
    ticker_config = ticker_configs.get(ticker, {})
    ticker_strategy = ticker_config.get('strategy', {})

    if ticker_strategy:
        # Merge ticker-specific strategy settings
        # Ticker settings override defaults

        # Override entry_conditions if present
        if 'entry_conditions' in ticker_strategy:
            if 'entry_conditions' not in strategy:
                strategy['entry_conditions'] = {}
            strategy['entry_conditions'].update(ticker_strategy['entry_conditions'])

        # Override exit_conditions if present
        if 'exit_conditions' in ticker_strategy:
            if 'exit_conditions' not in strategy:
                strategy['exit_conditions'] = {}
            strategy['exit_conditions'].update(ticker_strategy['exit_conditions'])

        # Override risk_management if present
        if 'risk_management' in ticker_strategy:
            if 'risk_management' not in strategy:
                strategy['risk_management'] = {}
            strategy['risk_management'].update(ticker_strategy['risk_management'])

        # Override trading if present
        if 'trading' in ticker_strategy:
            if 'trading' not in strategy:
                strategy['trading'] = {}
            strategy['trading'].update(ticker_strategy['trading'])

    return strategy


def flatten_strategy(strategy):
    """
    Flatten nested strategy structure into flat settings dictionary

    Args:
        strategy: Nested strategy structure

    Returns:
        Flattened settings dictionary
    """
    settings = {
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


def load_alpaca_settings():
    """
    Load DEFAULT strategy settings and return flattened dictionary
    This function maintains compatibility with existing code
    For ticker-specific settings, use get_ticker_strategy() + flatten_strategy()
    """
    strategy, _ = load_config()

    if not strategy:
        return None

    return flatten_strategy(strategy)


def load_signal_actions():
    """
    Load signal actions configuration
    This function maintains compatibility with existing code
    """
    _, signal_actions = load_config()
    return signal_actions


def load_trading_config():
    """
    Load trading configuration from alpaca.json and update global variables
    """
    global USE_PAPER, POSITION_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
    global MAX_BUY_SLIPPAGE_PCT, MAX_SELL_SLIPPAGE_PCT
    global EMAIL_NOTIFICATIONS_ENABLED, EMAIL_ON_BOT_START, EMAIL_ON_BOT_STOP
    global EMAIL_ON_ENTRY, EMAIL_ON_EXIT, EMAIL_ON_ERRORS

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        strategy = config.get('strategy', {})
        trading = strategy.get('trading', {})

        # Update trading settings
        USE_PAPER = trading.get('use_paper', True)
        POSITION_SIZE = trading.get('position_size', 100)
        MAX_BUY_SLIPPAGE_PCT = trading.get('max_buy_slippage_pct', 0.5)
        MAX_SELL_SLIPPAGE_PCT = trading.get('max_sell_slippage_pct', 1.0)

        # Risk management (from strategy.risk_management)
        risk_mgmt = strategy.get('risk_management', {})
        STOP_LOSS_PCT = risk_mgmt.get('stop_loss', 0.02)
        TAKE_PROFIT_PCT = risk_mgmt.get('take_profit', 0.03)

        # Email notifications
        email_config = trading.get('email_notifications', {})
        EMAIL_NOTIFICATIONS_ENABLED = email_config.get('enabled', True)
        EMAIL_ON_BOT_START = email_config.get('on_bot_start', True)
        EMAIL_ON_BOT_STOP = email_config.get('on_bot_stop', True)
        EMAIL_ON_ENTRY = email_config.get('on_entry', True)
        EMAIL_ON_EXIT = email_config.get('on_exit', True)
        EMAIL_ON_ERRORS = email_config.get('on_errors', True)

        logging.info("✅ Trading configuration loaded from alpaca.json")
        logging.info(f"   Max buy slippage: {MAX_BUY_SLIPPAGE_PCT}%")
        logging.info(f"   Max sell slippage: {MAX_SELL_SLIPPAGE_PCT}%")

    except Exception as e:
        logging.warning(f"⚠️  Failed to load trading config from alpaca.json: {e}")
        logging.warning("Using default values")


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


def send_email_alert(subject, message, force=False):
    """
    Send alert via email

    Args:
        subject: Email subject
        message: Email body
        force: If True, bypass EMAIL_NOTIFICATIONS_ENABLED check (for critical alerts)
    """
    # Check if email notifications are enabled (unless forced)
    if not force and not EMAIL_NOTIFICATIONS_ENABLED:
        logging.debug(f"Email notifications disabled. Skipping: {subject}")
        return False

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

        logging.info(f"✉️  Email sent: {subject}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}")
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


def initialize_alpaca(settings_config=None, signal_actions=None):
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

        # Load strategy settings if not provided
        settings = settings_config if settings_config else load_alpaca_settings()

        # Save initial portfolio state
        portfolio = save_portfolio_state()

        # Build position summary
        position_summary = []
        if portfolio and portfolio['positions']:
            for pos in portfolio['positions']:
                position_summary.append(f"  {pos['ticker']}: {pos['quantity']} shares @ ${pos['current_price']:.2f} (P&L: ${pos['unrealized_pnl']:.2f})")

        # Build ticker-specific strategy settings summary
        settings_summary = ""
        if settings:
            settings_summary = f"""
═══════════════════════════════════════
GLOBAL SETTINGS
═══════════════════════════════════════
Interval: {settings.get('interval', '5m')}
Period: {settings.get('period', '1d')}"""

        # Build enabled/disabled ticker lists and trading strategies
        enabled_list = "N/A"
        disabled_list = "N/A"
        trading_strategies = ""

        if signal_actions:
            ticker_configs = signal_actions.get('tickers', {})
            # Get all tickers from signal_actions config
            all_tickers = list(ticker_configs.keys())
            enabled_tickers = [t for t in all_tickers if ticker_configs.get(t, {}).get('enabled', True)]
            disabled_tickers = [t for t in all_tickers if not ticker_configs.get(t, {}).get('enabled', True)]
            enabled_list = ', '.join(enabled_tickers) if enabled_tickers else 'None'
            disabled_list = ', '.join(disabled_tickers) if disabled_tickers else 'None'

            # Build trading strategies for each enabled ticker
            for ticker in enabled_tickers:
                ticker_config = ticker_configs.get(ticker, {})

                # Get strategy configuration
                strategy = ticker_config.get('strategy', {})
                entry_cond = strategy.get('entry_conditions', {})
                exit_cond = strategy.get('exit_conditions', {})
                risk_mgmt = strategy.get('risk_management', {})

                # Get entry actions
                entry_actions = ticker_config.get('entry', {}).get('actions', [])
                entry_tickers = [a.get('ticker') for a in entry_actions if a.get('type') == 'BUY']
                sell_tickers = [a.get('ticker') for a in entry_actions if a.get('type') == 'SELL_ALL']

                # Get exit actions
                exit_actions = ticker_config.get('exit_conditions_met', {}).get('actions', [])
                exit_buy_tickers = [a.get('ticker') for a in exit_actions if a.get('type') == 'BUY']
                exit_sell_tickers = [a.get('ticker') for a in exit_actions if a.get('type') == 'SELL_ALL']

                trading_strategies += f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                trading_strategies += f"\n{ticker} STRATEGY"
                trading_strategies += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                # Entry conditions
                trading_strategies += f"\n\nEntry Conditions:"
                entry_conditions_list = []
                if entry_cond.get('use_rsi'):
                    entry_conditions_list.append(f"RSI < {entry_cond.get('rsi_threshold', 30)}")
                if entry_cond.get('use_ema_cross_up'):
                    entry_conditions_list.append("EMA9 cross above EMA21")
                if entry_cond.get('use_bb_cross_up'):
                    entry_conditions_list.append("Price cross above BB Upper")
                if entry_cond.get('use_bb_width'):
                    entry_conditions_list.append(f"BB Width > {entry_cond.get('bb_width_threshold', 0.4)}%")
                if entry_cond.get('use_macd_cross_up'):
                    entry_conditions_list.append("MACD cross above Signal")
                if entry_cond.get('use_price_vs_ema9'):
                    entry_conditions_list.append("Price > EMA9")
                if entry_cond.get('use_price_vs_ema21'):
                    entry_conditions_list.append("Price > EMA21")
                if entry_cond.get('use_macd_threshold'):
                    entry_conditions_list.append(f"MACD < {entry_cond.get('macd_threshold', 0.1)}")
                if entry_cond.get('use_macd_valley'):
                    entry_conditions_list.append("MACD Valley (turning up)")
                if entry_cond.get('use_volume'):
                    entry_conditions_list.append("Volume Rising")
                if entry_cond.get('use_price_drop_from_exit'):
                    drop_pct = entry_cond.get('price_drop_from_exit_pct', 2.0)
                    entry_conditions_list.append(f"Price must drop {drop_pct}% from last exit")

                if entry_conditions_list:
                    for cond in entry_conditions_list:
                        trading_strategies += f"\n  ✓ {cond}"
                else:
                    trading_strategies += "\n  (No conditions - any signal triggers)"

                # Entry actions
                if entry_tickers or sell_tickers:
                    trading_strategies += f"\n\n  Actions:"
                    if entry_tickers:
                        trading_strategies += f"\n    → BUY {', '.join(entry_tickers)}"
                    if sell_tickers:
                        trading_strategies += f"\n    → SELL {', '.join(sell_tickers)}"

                # Exit conditions
                trading_strategies += f"\n\nExit Conditions:"
                exit_conditions_list = []
                if exit_cond.get('use_rsi_exit'):
                    exit_conditions_list.append(f"RSI > {exit_cond.get('rsi_exit_threshold', 70)}")
                if exit_cond.get('use_ema_cross_down'):
                    exit_conditions_list.append("EMA9 cross below EMA21")
                if exit_cond.get('use_bb_cross_down'):
                    exit_conditions_list.append("Price cross below BB Lower")
                if exit_cond.get('use_bb_width_exit'):
                    exit_conditions_list.append(f"BB Width > {exit_cond.get('bb_width_exit_threshold', 0.4)}%")
                if exit_cond.get('use_macd_cross_down'):
                    exit_conditions_list.append("MACD cross below Signal")
                if exit_cond.get('use_price_vs_ema9_exit'):
                    exit_conditions_list.append("Price < EMA9")
                if exit_cond.get('use_price_vs_ema21_exit'):
                    exit_conditions_list.append("Price < EMA21")
                if exit_cond.get('use_macd_peak'):
                    exit_conditions_list.append("MACD Peak (turning down)")

                if exit_conditions_list:
                    for cond in exit_conditions_list:
                        trading_strategies += f"\n  ✓ {cond}"
                else:
                    trading_strategies += "\n  (No conditions - any signal triggers)"

                # Exit actions
                if exit_sell_tickers or exit_buy_tickers:
                    trading_strategies += f"\n\n  Actions:"
                    if exit_sell_tickers:
                        trading_strategies += f"\n    → SELL {', '.join(exit_sell_tickers)}"
                    if exit_buy_tickers:
                        trading_strategies += f"\n    → BUY {', '.join(exit_buy_tickers)}"

                # Risk management
                trading_strategies += f"\n\nRisk Management:"
                if risk_mgmt.get('use_stop_loss'):
                    trading_strategies += f"\n  🛑 Stop Loss: {risk_mgmt.get('stop_loss', 0.02)*100:.1f}%"
                else:
                    trading_strategies += f"\n  🛑 Stop Loss: Disabled"
                if risk_mgmt.get('use_take_profit'):
                    trading_strategies += f"\n  🎯 Take Profit: {risk_mgmt.get('take_profit', 0.03)*100:.1f}%"
                else:
                    trading_strategies += f"\n  🎯 Take Profit: Disabled"

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

Monitored Tickers: {enabled_list}
Disabled Tickers: {disabled_list}

Trading Strategies:{trading_strategies}
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

        # Get check interval from settings
        check_interval_seconds = settings.get('check_interval_seconds', 120) if settings else 120
        check_interval_minutes = check_interval_seconds / 60

        eastern = pytz.timezone('America/New_York')
        current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

        message += f"""
═══════════════════════════════════════
MONITORING
═══════════════════════════════════════
Check Frequency: Every {check_interval_minutes:.1f} minute{'' if check_interval_minutes == 1 else 's'} ({check_interval_seconds} seconds)
Email Alerts: {'Enabled' if EMAIL_NOTIFICATIONS_ENABLED else 'Disabled'}
Started: {current_time}
═══════════════════════════════════════

The bot is now actively monitoring for trading signals.
You will receive email alerts for all trades.
"""

        if EMAIL_ON_BOT_START:
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


def check_order_status(order_id, ticker, action_type):
    """
    Check order status and send email notification when filled

    Args:
        order_id: Alpaca order ID
        ticker: Stock ticker
        action_type: 'BUY' or 'SELL'

    Returns:
        Order status dict or None
    """
    try:
        import time

        # Wait a moment for order to process
        time.sleep(2)

        order = alpaca_api.get_order(order_id)

        if not order:
            logging.warning(f"Could not retrieve order {order_id}")
            return None

        status = order['status']
        filled_qty = order.get('filled_qty', 0)
        filled_price = order.get('filled_avg_price')

        logging.info(f"Order {order_id} status: {status}")

        # Send notification based on status
        eastern = pytz.timezone('America/New_York')
        current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

        if status == 'filled':
            # Order completely filled
            emoji = "✅" if action_type == "BUY" else "💰"
            subject = f"{emoji} ORDER FILLED - {ticker} {action_type}"

            message = f"""Order Filled Successfully

Ticker: {ticker}
Action: {action_type}
Quantity: {filled_qty} shares
Filled Price: ${filled_price:.2f}
Total Value: ${filled_qty * filled_price:.2f}
Status: FILLED
Order ID: {order_id}
Filled Time: {current_time}

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""

            if EMAIL_ON_ENTRY or EMAIL_ON_EXIT:
                send_email_alert(subject, message)

            logging.info(f"✅ Order {order_id} filled: {filled_qty} shares @ ${filled_price:.2f}")

        elif status == 'partially_filled':
            # Order partially filled
            total_qty = order.get('qty', 0)
            remaining_qty = total_qty - filled_qty

            subject = f"⚠️ ORDER PARTIALLY FILLED - {ticker} {action_type}"

            message = f"""Order Partially Filled

Ticker: {ticker}
Action: {action_type}
Ordered: {total_qty} shares
Filled: {filled_qty} shares
Remaining: {remaining_qty} shares
Filled Price: ${filled_price:.2f} (average)
Status: PARTIALLY FILLED
Order ID: {order_id}
Time: {current_time}

The order is still active and may fill completely.

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""

            if EMAIL_ON_ENTRY or EMAIL_ON_EXIT:
                send_email_alert(subject, message)

            logging.warning(f"⚠️  Order {order_id} partially filled: {filled_qty}/{total_qty} shares")

        elif status in ['canceled', 'expired', 'rejected']:
            # Order failed
            subject = f"❌ ORDER {status.upper()} - {ticker} {action_type}"

            message = f"""Order {status.capitalize()}

Ticker: {ticker}
Action: {action_type}
Quantity: {order.get('qty', 0)} shares
Status: {status.upper()}
Order ID: {order_id}
Time: {current_time}

The order was {status} and did not fill.

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""

            if EMAIL_ON_ERRORS:
                send_email_alert(subject, message)

            logging.error(f"❌ Order {order_id} {status}")

        return order

    except Exception as e:
        logging.error(f"Failed to check order status for {order_id}: {e}")
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

            # Sell all existing positions
            if not sell_all_positions():
                logging.error("Failed to sell positions for stop loss")
                return False

            # Clear position state
            state['current_position'] = None
            state['entry_price'] = None
            state['entry_time'] = None
            state['entry_conditions'] = None
            state['position_size'] = 0
            save_state(state)

            logging.info(f"✅ Stop loss executed - all positions closed")
            return True

        # Check take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            logging.info(f"🎯 TAKE PROFIT triggered! P&L: {pnl_pct*100:.2f}%")

            # Sell all existing positions
            if not sell_all_positions():
                logging.error("Failed to sell positions for take profit")
                return False

            # Clear position state
            state['current_position'] = None
            state['entry_price'] = None
            state['entry_time'] = None
            state['entry_conditions'] = None
            state['position_size'] = 0
            save_state(state)

            logging.info(f"✅ Take profit executed - all positions closed")
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

        # Check current market price before placing order
        try:
            quote = alpaca_api.quote(ticker)
            if quote and 'last' in quote:
                current_price = quote['last']
                signal_price = price
                price_diff_pct = ((current_price - signal_price) / signal_price) * 100

                logging.info(f"      Signal price: ${signal_price:.2f}")
                logging.info(f"      Current price: ${current_price:.2f}")
                logging.info(f"      Price difference: {price_diff_pct:+.2f}%")

                # Skip buy if current price is higher than signal price by max slippage threshold
                if price_diff_pct > MAX_BUY_SLIPPAGE_PCT:
                    logging.warning(f"      ⚠️  SKIPPING BUY - Current price ${current_price:.2f} is {price_diff_pct:.2f}% higher than signal price ${signal_price:.2f}")

                    # Still send email notification
                    if EMAIL_ON_ENTRY:
                        eastern = pytz.timezone('America/New_York')
                        current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

                        email_subject = f"⚠️ BUY SKIPPED - {ticker} (Price Too High)"
                        email_body = f"""Buy Order Skipped - Price Moved Too Much

Ticker: {ticker}
Action: BUY {quantity} shares (SKIPPED)
Signal Price: ${signal_price:.2f}
Current Price: ${current_price:.2f}
Price Difference: {price_diff_pct:+.2f}%
Time: {current_time}

Details: {note}

Reason: Current market price is {price_diff_pct:.2f}% higher than signal price.
This indicates the signal is too old or market has moved significantly.

Order was NOT placed to avoid buying at a worse price.

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""
                        send_email_alert(email_subject, email_body)

                    return False
        except Exception as e:
            logging.warning(f"      Could not verify current price: {e}. Proceeding with order.")

        # Check if current price is at least X% below last exit price
        # Load strategy config to get the threshold
        try:
            strategy = load_strategy_from_alpaca(ticker)
            if strategy:
                entry_cond = strategy.get('entry_conditions', {})
                use_price_drop = entry_cond.get('use_price_drop_from_exit', False)
                price_drop_threshold = entry_cond.get('price_drop_from_exit_pct', 2.0)

                if use_price_drop:
                    # Check if we have a last exit price for this ticker
                    if 'last_exit_prices' in state and ticker in state['last_exit_prices']:
                        last_exit_price = state['last_exit_prices'][ticker]
                        current_price = price  # Use signal price as current

                        # Calculate how much price has dropped from last exit
                        price_drop_pct = ((last_exit_price - current_price) / last_exit_price) * 100

                        logging.info(f"      Last exit price: ${last_exit_price:.2f}")
                        logging.info(f"      Current price: ${current_price:.2f}")
                        logging.info(f"      Price drop from exit: {price_drop_pct:.2f}%")

                        # Skip buy if price hasn't dropped enough from last exit
                        if price_drop_pct < price_drop_threshold:
                            logging.warning(f"      ⚠️  SKIPPING BUY - Price only dropped {price_drop_pct:.2f}% from last exit (need {price_drop_threshold}%)")

                            # Send email notification
                            if EMAIL_ON_ENTRY:
                                eastern = pytz.timezone('America/New_York')
                                current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

                                email_subject = f"⚠️ BUY SKIPPED - {ticker} (Price Not Low Enough)"
                                email_body = f"""Buy Order Skipped - Price Has Not Dropped Enough From Last Exit

Ticker: {ticker}
Action: BUY {quantity} shares (SKIPPED)
Last Exit Price: ${last_exit_price:.2f}
Current Price: ${current_price:.2f}
Price Drop: {price_drop_pct:.2f}%
Required Drop: {price_drop_threshold}%
Time: {current_time}

Details: {note}

Reason: Current price is only {price_drop_pct:.2f}% below the last exit price.
Strategy requires at least {price_drop_threshold}% drop before re-entering.

This prevents buying back at a similar or higher price after exiting.

Order was NOT placed.

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""
                                send_email_alert(email_subject, email_body)

                            return False
                        else:
                            logging.info(f"      ✅ Price drop check passed: {price_drop_pct:.2f}% >= {price_drop_threshold}%")
                    else:
                        logging.info(f"      No previous exit price found for {ticker}, skipping price drop check")
        except Exception as e:
            logging.warning(f"      Could not check price drop from exit: {e}. Proceeding with order.")

        order = place_buy_order(ticker, quantity, price, note)
        if order:
            logging.info(f"      ✅ BUY order placed: {quantity} shares of {ticker} @ ${price:.2f}")

            # Check order status after placement
            if 'order_id' in order:
                check_order_status(order['order_id'], ticker, 'BUY')

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

            # Send email notification for entry
            if EMAIL_ON_ENTRY:
                eastern = pytz.timezone('America/New_York')
                current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

                email_subject = f"🟢 ENTRY SIGNAL - {ticker}"
                email_body = f"""Entry Signal Executed

Ticker: {ticker}
Action: BUY {quantity} shares
Price: ${price:.2f}
Signal Time: {timestamp}
Current Time: {current_time}

Details: {note}

Reason: {description if description else 'Entry signal detected'}

Order ID: {order.get('order_id', 'N/A')}
Status: {order.get('status', 'N/A')}

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""
                send_email_alert(email_subject, email_body)

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

            # Check order status after placement
            if 'order_id' in order:
                check_order_status(order['order_id'], ticker, 'SELL')

            # Calculate P&L if we have entry price
            pnl_info = ""
            pnl_dollars = 0
            pnl_pct = 0
            has_pnl = False

            if 'positions' in state and ticker in state['positions']:
                entry_price = state['positions'][ticker].get('entry_price', 0)
                if entry_price > 0:
                    pnl_dollars = (price - entry_price) * sell_qty
                    pnl_pct = ((price - entry_price) / entry_price) * 100
                    pnl_info = f"\n\nP&L:\nEntry Price: ${entry_price:.2f}\nExit Price: ${price:.2f}\nProfit/Loss: ${pnl_dollars:.2f} ({pnl_pct:+.2f}%)"
                    has_pnl = True

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

            # Send email notification for exit
            if EMAIL_ON_EXIT:
                eastern = pytz.timezone('America/New_York')
                current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

                # Choose emoji based on P&L
                if has_pnl:
                    if pnl_dollars >= 0:
                        exit_emoji = "💰"  # Green/Profit
                        result_text = "PROFIT"
                    else:
                        exit_emoji = "⬛"  # Black/Loss
                        result_text = "LOSS"
                    email_subject = f"{exit_emoji} EXIT SIGNAL - {ticker} ({result_text})"
                else:
                    exit_emoji = "🔴"  # Red (unknown P&L)
                    email_subject = f"{exit_emoji} EXIT SIGNAL - {ticker}"

                email_body = f"""Exit Signal Executed

Ticker: {ticker}
Action: SELL {sell_qty} shares
Price: ${price:.2f}
Signal Time: {timestamp}
Current Time: {current_time}

Details: {note}

Reason: {description if description else 'Exit signal detected'}{pnl_info}

Order ID: {order.get('order_id', 'N/A')}
Status: {order.get('status', 'N/A')}

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""
                send_email_alert(email_subject, email_body)

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

            # Check current market price before placing sell order
            try:
                quote = alpaca_api.quote(ticker)
                if quote and 'last' in quote:
                    current_price = quote['last']
                    signal_price = price
                    price_diff_pct = ((signal_price - current_price) / signal_price) * 100

                    logging.info(f"      Signal price: ${signal_price:.2f}")
                    logging.info(f"      Current price: ${current_price:.2f}")
                    logging.info(f"      Price difference: {price_diff_pct:+.2f}%")

                    # Skip sell if signal price is greater than current price by max slippage threshold
                    if price_diff_pct > MAX_SELL_SLIPPAGE_PCT:
                        logging.warning(f"      ⚠️  SKIPPING SELL - Signal price ${signal_price:.2f} is {price_diff_pct:.2f}% higher than current price ${current_price:.2f}")

                        # Still send email notification
                        if EMAIL_ON_EXIT:
                            eastern = pytz.timezone('America/New_York')
                            current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

                            email_subject = f"⚠️ SELL SKIPPED - {ticker} (Price Too Low)"
                            email_body = f"""Sell Order Skipped - Price Dropped Too Much

Ticker: {ticker}
Action: SELL {available_qty} shares (SKIPPED)
Signal Price: ${signal_price:.2f}
Current Price: ${current_price:.2f}
Price Drop: {price_diff_pct:.2f}%
Time: {current_time}

Details: {note}

Reason: Current market price is {price_diff_pct:.2f}% lower than signal price.
This indicates the signal is too old or market has dropped significantly.

Order was NOT placed to avoid selling at a worse price.

---
Alpaca Trading Bot ({USE_PAPER and 'PAPER' or 'LIVE'} Trading)
"""
                            send_email_alert(email_subject, email_body)

                        return False
            except Exception as e:
                logging.warning(f"      Could not verify current price: {e}. Proceeding with order.")

            order = place_sell_order(ticker, available_qty, price, note)
            if order:
                logging.info(f"      ✅ SELL ALL order placed: {available_qty} shares of {ticker} @ ${price:.2f}")

                # Check order status after placement
                if 'order_id' in order:
                    check_order_status(order['order_id'], ticker, 'SELL')

                # Track last exit price before removing position
                if 'last_exit_prices' not in state:
                    state['last_exit_prices'] = {}
                state['last_exit_prices'][ticker] = price
                logging.info(f"      📊 Last exit price for {ticker}: ${price:.2f}")

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

    # Load default strategy settings
    default_settings = load_alpaca_settings()

    if not default_settings:
        logging.error(f"No settings file found. Please create {CONFIG_FILE}")
        return

    # Load default strategy (for merging with ticker-specific overrides)
    default_strategy, _ = load_config()

    # Get tickers from signal_actions config (tickers defined there, not in strategy)
    ticker_configs = signal_actions_config.get('tickers', {})
    tickers = list(ticker_configs.keys())

    if not tickers:
        logging.error("No tickers defined in signal_actions config")
        return

    interval = default_settings.get('interval', '5m')
    period = default_settings.get('period', '1d')

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

            # Convert timezone from UTC to US Eastern Time (New York)
            if hasattr(raw.index, 'tz'):
                if raw.index.tz is not None:
                    raw.index = raw.index.tz_convert('America/New_York')
                else:
                    raw.index = raw.index.tz_localize('UTC').tz_convert('America/New_York')

            df = raw.copy()

            # Get ticker-specific strategy settings (with overrides)
            ticker_strategy = get_ticker_strategy(ticker, default_strategy, signal_actions_config)
            ticker_settings = flatten_strategy(ticker_strategy)

            # Log if using ticker-specific overrides
            ticker_config = ticker_configs.get(ticker, {})
            if 'strategy' in ticker_config:
                logging.info(f"  ℹ️  Using ticker-specific strategy overrides for {ticker}")

            # Run backtest to get latest signal
            logging.info(f"Running backtest for {ticker}...")
            df1, df5, trades_df, logs_df = backtest_symbol(
                df1=df,
                rsi_threshold=ticker_settings.get('rsi_threshold', 30),
                use_rsi=ticker_settings.get('use_rsi', True),
                use_ema=ticker_settings.get('use_ema', True),
                use_volume=ticker_settings.get('use_volume', True),
                stop_loss=ticker_settings.get('stop_loss', 0.02),
                tp_pct=ticker_settings.get('take_profit', 0.03),
                avoid_after=None,  # Time filter disabled
                use_stop_loss=ticker_settings.get('use_stop_loss', True),
                use_take_profit=ticker_settings.get('use_take_profit', True),
                use_rsi_overbought=ticker_settings.get('use_rsi_exit', False),
                rsi_overbought_threshold=ticker_settings.get('rsi_exit_threshold', 70),
                use_ema_cross_up=ticker_settings.get('use_ema_cross_up', False),
                use_ema_cross_down=ticker_settings.get('use_ema_cross_down', False),
                use_price_below_ema9=ticker_settings.get('use_price_vs_ema9_exit', False),
                use_bb_cross_up=ticker_settings.get('use_bb_cross_up', False),
                use_bb_cross_down=ticker_settings.get('use_bb_cross_down', False),
                use_macd_cross_up=ticker_settings.get('use_macd_cross_up', False),
                use_macd_cross_down=ticker_settings.get('use_macd_cross_down', False),
                use_price_above_ema21=ticker_settings.get('use_price_vs_ema21', False),
                use_price_below_ema21=ticker_settings.get('use_price_vs_ema21_exit', False),
                use_macd_below_threshold=ticker_settings.get('use_macd_threshold', False),
                macd_below_threshold=ticker_settings.get('macd_threshold', 0),
                use_macd_above_threshold=ticker_settings.get('use_macd_threshold', False),
                macd_above_threshold=ticker_settings.get('macd_threshold', 0),
                use_macd_peak=ticker_settings.get('use_macd_peak', False),
                use_macd_valley=ticker_settings.get('use_macd_valley', False)
            )

            # Get current price from DataFrame
            current_price = df['Close'].iloc[-1]

            # Check for recent signals in backtest data
            if not logs_df.empty:
                # Get most recent signal from backtest (not based on current time)
                # Filter out exit_EOD first
                actionable_logs = logs_df[logs_df['event'] != 'exit_EOD']

                if not actionable_logs.empty:
                    # Use the most recent actionable signal
                    latest_log = actionable_logs.tail(1).iloc[-1]
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
                    # No actionable signals (only exit_EOD)
                    logging.info(f"No actionable signals for {ticker} (only exit_EOD markers)")
            else:
                logging.info(f"No signals in backtest logs for {ticker}")

        except Exception as e:
            logging.error(f"Error running strategy for {ticker}: {e}", exc_info=True)
            if EMAIL_ON_ERRORS:
                send_email_alert("❌ Strategy Error", f"Error for {ticker}: {e}")


def display_settings(settings):
    """Display strategy settings in a formatted way"""
    if not settings:
        logging.warning("No settings file found. Using defaults.")
        return

    logging.info("=" * 60)
    logging.info("STRATEGY SETTINGS")
    logging.info("=" * 60)

    # Tickers are now defined in signal_actions, not in strategy settings
    # Strategy settings apply to all tickers

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


def send_stop_email(account_type):
    """Send email notification when bot stops"""
    try:
        # Build stop email with enabled tickers info
        ticker_configs = signal_actions_config.get('tickers', {}) if signal_actions_config else {}
        all_tickers = list(ticker_configs.keys())
        enabled_tickers = [t for t in all_tickers if ticker_configs.get(t, {}).get('enabled', True)]
        disabled_tickers = [t for t in all_tickers if not ticker_configs.get(t, {}).get('enabled', True)]

        enabled_list = ', '.join(enabled_tickers) if enabled_tickers else 'None'
        disabled_list = ', '.join(disabled_tickers) if disabled_tickers else 'None'

        eastern = pytz.timezone('America/New_York')
        current_time = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')

        stop_message = f"""Alpaca Trading Bot Stopped

Mode: {account_type}
Monitored Tickers: {enabled_list}
Disabled Tickers: {disabled_list}
Stop Time: {current_time}

Bot has been stopped manually."""

        if EMAIL_ON_BOT_STOP:
            send_email_alert("🛑 Trading Bot Stopped", stop_message)
    except Exception as e:
        logging.error(f"Failed to send stop email: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGTERM, SIGINT)"""
    logging.info("\nReceived shutdown signal. Stopping bot...")

    # Determine account type for email
    account_type = "PAPER TRADING" if USE_PAPER else "⚠️  LIVE TRADING ⚠️"

    # Send stop email
    send_stop_email(account_type)

    # Exit gracefully
    sys.exit(0)


def main():
    """Main trading loop"""
    logging.info("=" * 60)
    logging.info("ALPACA STRATEGY TRADER - AUTOMATED TRADING")
    logging.info("=" * 60)

    # Load trading configuration from alpaca.json first
    load_trading_config()

    account_type = "PAPER TRADING" if USE_PAPER else "⚠️  LIVE TRADING ⚠️"
    logging.info(f"Mode: {account_type}")
    logging.info(f"Position Size: {POSITION_SIZE} shares (default)")
    logging.info(f"Email: {GMAIL_ADDRESS}")
    logging.info(f"Stop Loss: {STOP_LOSS_PCT*100:.1f}%")
    logging.info(f"Take Profit: {TAKE_PROFIT_PCT*100:.1f}%")
    logging.info("Press Ctrl+C to stop")
    logging.info("=" * 60)
    logging.info("")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)  # Handle kill command
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C

    # Load and display strategy settings
    settings = load_alpaca_settings()
    display_settings(settings)
    logging.info("")

    # Load signal-to-action mappings
    global signal_actions_config
    signal_actions_config = load_signal_actions()
    display_signal_actions(signal_actions_config)
    logging.info("")

    # Initialize Alpaca with settings and signal actions for startup email
    if not initialize_alpaca(settings, signal_actions_config):
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
        # Handle Ctrl+C (already handled by signal handler, but kept for fallback)
        logging.info("\nAlpaca Strategy Trader Stopped (KeyboardInterrupt)")
        send_stop_email(account_type)


if __name__ == "__main__":
    main()
