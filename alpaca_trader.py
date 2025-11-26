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

# Tracking file
STATE_FILE = '.alpaca_trader_state.json'
CONFIG_FILE = 'alpaca_strategy_config.json'

# Alpaca API instance
alpaca_api = None


def load_alpaca_settings():
    """Load settings from alpaca_strategy_config.json"""
    if not os.path.exists(CONFIG_FILE):
        logging.error(f"Configuration file {CONFIG_FILE} not found!")
        return None

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        # Flatten the nested structure into a flat settings dictionary
        settings = {
            'ticker': config.get('ticker', 'TQQQ'),
            'interval': config.get('interval', '5m'),
            'period': config.get('period', '1d'),
        }

        # Entry conditions
        entry = config.get('entry_conditions', {})
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
        exit_cond = config.get('exit_conditions', {})
        settings['use_rsi_exit'] = exit_cond.get('use_rsi_exit', False)
        settings['rsi_exit_threshold'] = exit_cond.get('rsi_exit_threshold', 70)
        settings['use_ema_cross_down'] = exit_cond.get('use_ema_cross_down', False)
        settings['use_bb_cross_down'] = exit_cond.get('use_bb_cross_down', False)
        settings['use_macd_cross_down'] = exit_cond.get('use_macd_cross_down', False)
        settings['use_price_vs_ema9_exit'] = exit_cond.get('use_price_vs_ema9_exit', False)
        settings['use_price_vs_ema21_exit'] = exit_cond.get('use_price_vs_ema21_exit', False)
        settings['use_macd_peak'] = exit_cond.get('use_macd_peak', False)

        # Risk management
        risk = config.get('risk_management', {})
        settings['stop_loss'] = risk.get('stop_loss', 0.02)
        settings['take_profit'] = risk.get('take_profit', 0.03)
        settings['use_stop_loss'] = risk.get('use_stop_loss', True)
        settings['use_take_profit'] = risk.get('use_take_profit', True)

        # Trading settings
        trading = config.get('trading', {})
        settings['position_size'] = trading.get('position_size', 10)
        settings['check_interval_seconds'] = trading.get('check_interval_seconds', 300)

        # Time filter settings
        time_filter = config.get('time_filter', {})
        settings['use_time_filter'] = time_filter.get('use_time_filter', False)
        settings['avoid_after_time'] = time_filter.get('avoid_after_time', '15:00')

        logging.info(f"Loaded settings from {CONFIG_FILE}")
        return settings

    except Exception as e:
        logging.error(f"Failed to load settings from {CONFIG_FILE}: {e}")
        return None


def load_state():
    """Load the last known state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'last_check_time': None,
        'current_position': None,  # 'TQQQ' or 'SQQQ' or None
        'entry_price': None,
        'entry_time': None,
        'entry_conditions': None,  # Detailed conditions that triggered entry
        'position_size': 0,
        'order_ids': []
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

    ticker = settings.get('ticker', 'TQQQ')
    interval = settings.get('interval', '5m')
    period = settings.get('period', '1d')

    # Download recent data
    try:
        logging.info("Downloading data...")
        download_period = "5d"  # Always download 5 days for context
        use_extended_hours = interval not in ["1d", "5d", "1wk", "1mo", "3mo"]
        raw = yf.download(ticker, period=download_period, interval=interval,
                         progress=False, prepost=use_extended_hours)

        if raw.empty:
            logging.error(f"No data returned for {ticker}")
            return

        # Handle MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.copy()

        # Run backtest to get latest signal
        logging.info("Running backtest...")
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

                logging.info(f"Latest signal: {event}")
                logging.info(f"Time: {timestamp}")
                logging.info(f"Price: ${price:.2f}")
                logging.info(f"Details: {note}")

                # Check if this is a new signal (not already processed)
                if state['last_check_time'] != str(timestamp):

                    # Save portfolio state before making any trades
                    logging.info("Saving current portfolio state...")
                    save_portfolio_state()

                    # Ignore exit_EOD signals (backtest artifacts)
                    if event == 'exit_EOD':
                        logging.info("Ignoring exit_EOD signal (backtest artifact)")

                    # ENTRY SIGNAL - Buy TQQQ
                    elif event == 'entry':
                        logging.info("📈 BUY SIGNAL: Buying TQQQ")

                        # First, sell all existing positions
                        if not sell_all_positions():
                            logging.error("Failed to sell existing positions, skipping buy order")
                        else:
                            # Then buy TQQQ
                            order = place_buy_order('TQQQ', POSITION_SIZE, price, note)
                            if order:
                                state['current_position'] = 'TQQQ'
                                state['entry_price'] = price
                                state['entry_time'] = str(timestamp)
                                state['entry_conditions'] = note  # Store detailed entry conditions
                                state['position_size'] = POSITION_SIZE
                                state['order_ids'].append(order['order_id'])

                    # EXIT SIGNAL - Buy SQQQ
                    elif event in ['exit_SL', 'exit_TP', 'exit_conditions_met']:
                        exit_reason = 'Stop Loss' if event == 'exit_SL' else 'Take Profit' if event == 'exit_TP' else 'Exit Conditions'
                        logging.info(f"📉 SELL SIGNAL ({exit_reason}): Buying SQQQ")

                        # Calculate P&L if we were holding TQQQ
                        if state['current_position'] == 'TQQQ' and state['entry_price']:
                            pnl_pct = ((price - state['entry_price']) / state['entry_price']) * 100
                            pnl_dollars = (price - state['entry_price']) * state['position_size']
                            logging.info(f"TQQQ Trade P&L: ${pnl_dollars:.2f} ({pnl_pct:+.2f}%)")

                        # First, sell all existing positions
                        if not sell_all_positions():
                            logging.error("Failed to sell existing positions, skipping buy order")
                        else:
                            # Then buy SQQQ - pass previous entry conditions for context
                            previous_entry_conditions = state.get('entry_conditions')
                            order = place_buy_order('SQQQ', POSITION_SIZE, price, note, entry_conditions=previous_entry_conditions)
                            if order:
                                state['current_position'] = 'SQQQ'
                                state['entry_price'] = price
                                state['entry_time'] = str(timestamp)
                                state['entry_conditions'] = note  # Store new entry conditions for SQQQ
                                state['position_size'] = POSITION_SIZE
                                state['order_ids'].append(order['order_id'])

                    # Update last check time
                    state['last_check_time'] = str(timestamp)
                    save_state(state)
                else:
                    logging.info("Signal already processed")
            else:
                # No signals in last 5 minutes - show the most recent one for info
                if len(logs_df) > 0:
                    last_signal = logs_df.iloc[-1]
                    time_diff = now - last_signal['time']
                    logging.info(f"No recent signals (last 5 minutes)")
                    logging.info(f"Most recent signal: {last_signal['event']} at {last_signal['time']} ({time_diff.total_seconds()/60:.1f} minutes ago)")
                else:
                    logging.info("No signals in backtest logs")
        else:
            logging.info("No signals in backtest logs")

    except Exception as e:
        logging.error(f"Error running strategy: {e}", exc_info=True)
        send_email_alert("❌ Strategy Error", f"Error: {e}")


def display_settings(settings):
    """Display strategy settings in a formatted way"""
    if not settings:
        logging.warning("No settings file found. Using defaults.")
        return

    logging.info("=" * 60)
    logging.info("STRATEGY SETTINGS")
    logging.info("=" * 60)

    # Basic settings
    logging.info(f"Ticker: {settings.get('ticker', 'N/A')}")
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
        send_email_alert("🛑 Trading Bot Stopped", "Alpaca strategy trader has been stopped manually")


if __name__ == "__main__":
    main()
