#!/usr/bin/env python3
"""
Strategy Trader - Automated trading with Webull integration
Monitors strategy signals and executes real trades
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

# Import Webull
from webull_wrapper import WebullAPI

# Import functions from rules.py
from rules import (
    rsi, ema, bollinger_bands, macd, backtest_symbol, load_settings
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_trader.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Email Configuration
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')

# Webull Configuration from webull_config.py
try:
    from webull_config import USE_PAPER, POSITION_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
except ImportError:
    # Defaults if config doesn't exist
    USE_PAPER = True  # SAFETY: Default to paper trading
    POSITION_SIZE = 10
    STOP_LOSS_PCT = -0.02
    TAKE_PROFIT_PCT = 0.03

# Tracking file
STATE_FILE = '.strategy_trader_state.json'

# Webull API instance
wb_api = None


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
        positions = wb_api.get_positions()
        portfolio_state = {
            'timestamp': datetime.now().isoformat(),
            'positions': []
        }

        for pos in positions:
            portfolio_state['positions'].append({
                'ticker': pos.get('ticker', {}).get('symbol'),
                'quantity': pos.get('position'),
                'cost_basis': pos.get('costPrice'),
                'current_price': pos.get('marketValue'),
                'unrealized_pnl': pos.get('unrealizedProfitLoss'),
                'unrealized_pnl_pct': pos.get('unrealizedProfitLossRate')
            })

        # Save to portfolio state file
        with open('.portfolio_state.json', 'w') as f:
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
        current_orders = wb_api.get_current_orders()

        if current_orders:
            logging.warning(f"Found {len(current_orders)} pending orders")
            for order in current_orders:
                order_id = order.get('orderId')
                ticker = order.get('ticker', {}).get('symbol', 'Unknown')
                action = order.get('action', 'Unknown')
                status = order.get('status', 'Unknown')
                logging.info(f"  Order {order_id}: {action} {ticker} - Status: {status}")

            # Cancel pending orders before selling
            logging.info("Cancelling pending orders...")
            for order in current_orders:
                order_id = order.get('orderId')
                if order_id:
                    try:
                        wb_api.cancel(order_id)
                        logging.info(f"  Cancelled order {order_id}")
                    except Exception as cancel_error:
                        logging.warning(f"  Could not cancel order {order_id}: {cancel_error}")

            # Wait a moment for cancellations to process
            time.sleep(2)

        # Now get current positions
        positions = wb_api.get_positions()

        if not positions:
            logging.info("No existing positions to sell")
            return True

        logging.info(f"Found {len(positions)} positions to sell")

        for pos in positions:
            ticker = pos.get('ticker', {}).get('symbol')
            quantity = float(pos.get('position', 0))

            if quantity > 0:
                logging.info(f"Selling existing position: {quantity} shares of {ticker}")
                current_price = float(pos.get('lastPrice', 0))

                order = place_sell_order(ticker, int(quantity), current_price, "Clear existing position")

                if order:
                    logging.info(f"Successfully placed sell order for {ticker}")
                else:
                    logging.error(f"Failed to sell {ticker}")
                    return False

        return True

    except Exception as e:
        logging.error(f"Failed to sell positions: {e}")
        return False


def initialize_webull():
    """Initialize and login to Webull"""
    global wb_api

    try:
        wb_api = WebullAPI()
        wb_api.login()

        account_type = "PAPER" if USE_PAPER else "LIVE"
        logging.info(f"Webull connected: {account_type} account")

        # Save initial portfolio state
        portfolio = save_portfolio_state()

        portfolio_msg = ""
        if portfolio and portfolio['positions']:
            portfolio_msg = f"\n\nCurrent Holdings:"
            for pos in portfolio['positions']:
                portfolio_msg += f"\n  {pos['ticker']}: {pos['quantity']} shares (${pos['unrealized_pnl']:.2f} P&L)"

        send_email_alert(
            "🤖 Trading Bot Started",
            f"Strategy Trader initialized\nAccount: {account_type}\nPosition Size: {POSITION_SIZE} shares\nStrategy: TQQQ/SQQQ Pair Trading\n\nBuy Signal → Buy TQQQ, Sell SQQQ\nSell Signal → Sell TQQQ, Buy SQQQ{portfolio_msg}"
        )
        return True

    except Exception as e:
        logging.error(f"Failed to initialize Webull: {e}")
        send_email_alert(
            "❌ Trading Bot Error",
            f"Failed to connect to Webull: {e}"
        )
        return False


def place_buy_order(ticker, qty, price, reason, entry_conditions=None):
    """Place a buy order"""
    try:
        logging.info(f"Placing BUY order: {qty} {ticker} @ ${price:.2f}")

        order = wb_api.place_order(
            ticker=ticker,
            qty=qty,
            action="BUY",
            order_type="MKT"
        )

        message = f"""✅ BUY ORDER PLACED

Ticker: {ticker}
Quantity: {qty} shares
Price: ${price:.2f}

Conditions:
{reason}
"""

        # If this is an exit and we have entry conditions, show them too
        if entry_conditions:
            message += f"""
Entry Conditions (when position was opened):
{entry_conditions}
"""

        message += f"""
Order ID: {order.get('orderId', 'N/A')}
"""

        send_email_alert("🟢 BUY ORDER", message)

        return order

    except Exception as e:
        logging.error(f"Buy order failed: {e}")
        send_email_alert("❌ BUY ORDER FAILED", f"Error: {e}")
        return None


def place_sell_order(ticker, qty, price, reason, entry_conditions=None):
    """Place a sell order"""
    try:
        logging.info(f"Placing SELL order: {qty} {ticker} @ ${price:.2f}")

        order = wb_api.place_order(
            ticker=ticker,
            qty=qty,
            action="SELL",
            order_type="MKT"
        )

        message = f"""🔴 SELL ORDER PLACED

Ticker: {ticker}
Quantity: {qty} shares
Price: ${price:.2f}

Conditions:
{reason}
"""

        # Show entry conditions if available
        if entry_conditions:
            message += f"""
Entry Conditions (when position was opened):
{entry_conditions}
"""

        message += f"""
Order ID: {order.get('orderId', 'N/A')}
"""

        send_email_alert("🔴 SELL ORDER", message)

        return order

    except Exception as e:
        logging.error(f"Sell order failed: {e}")
        send_email_alert("❌ SELL ORDER FAILED", f"Error: {e}")
        return None


def run_strategy():
    """Run the trading strategy and execute trades"""
    logging.info("=" * 60)
    logging.info("Running strategy check...")

    # Save current portfolio state at the start of each check
    logging.info("Saving portfolio snapshot...")
    save_portfolio_state()

    # Load settings
    settings = load_settings()
    if not settings:
        logging.error("No settings found")
        return

    ticker = settings.get('ticker', 'TQQQ')
    interval = settings.get('interval', '1m')

    logging.info(f"Ticker: {ticker}, Interval: {interval}")

    try:
        # Download recent data
        logging.info("Downloading data...")
        period = "5d"
        use_extended_hours = interval not in ["1d", "5d", "1wk", "1mo", "3mo"]
        raw = yf.download(ticker, period=period, interval=interval,
                         progress=False, prepost=use_extended_hours)

        if raw.empty:
            logging.error("No data returned")
            return

        # Handle MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.copy()

        # Run backtest
        logging.info("Running backtest...")
        df1, df5, trades_df, logs_df = backtest_symbol(
            df1=df,
            rsi_threshold=settings.get('rsi_threshold', 30),
            use_rsi=settings.get('use_rsi', True),
            use_ema=settings.get('use_ema', True),
            use_volume=settings.get('use_volume', True),
            stop_loss=settings.get('stop_loss_pct', 2.0) / 100,
            tp_pct=settings.get('take_profit_pct', 4.0) / 100,
            use_stop_loss=settings.get('use_stop_loss', True),
            use_take_profit=settings.get('use_take_profit', True),
            use_rsi_overbought=settings.get('use_rsi_overbought', True),
            rsi_overbought_threshold=settings.get('rsi_overbought_threshold', 70),
            use_ema_cross_up=settings.get('use_ema_cross_up', False),
            use_ema_cross_down=settings.get('use_ema_cross_down', False),
            use_price_below_ema9=settings.get('use_price_below_ema9', False),
            use_bb_cross_up=settings.get('use_bb_cross_up', False),
            use_bb_cross_down=settings.get('use_bb_cross_down', False),
            use_macd_cross_up=settings.get('use_macd_cross_up', False),
            use_macd_cross_down=settings.get('use_macd_cross_down', False),
            use_price_above_ema21=settings.get('use_price_above_ema21', False),
            use_price_below_ema21=settings.get('use_price_below_ema21', False),
            use_macd_below_threshold=settings.get('use_macd_below_threshold', False),
            macd_below_threshold=settings.get('macd_below_threshold', 0.0),
            use_macd_above_threshold=settings.get('use_macd_above_threshold', False),
            macd_above_threshold=settings.get('macd_above_threshold', 0.0),
            use_macd_peak=settings.get('use_macd_peak', False),
            use_macd_valley=settings.get('use_macd_valley', False)
        )

        # Load current state
        state = load_state()

        # Get current price
        current_price = df['Close'].iloc[-1]

        # Check for recent signals (last 10 minutes)
        if not logs_df.empty:
            now = pd.Timestamp.now(tz=logs_df['time'].iloc[-1].tz)
            recent_logs = logs_df[logs_df['time'] >= now - pd.Timedelta(minutes=10)]

            if not recent_logs.empty:
                latest_log = recent_logs.iloc[-1]
                event = latest_log['event']
                price = latest_log['price']
                note = latest_log['note']
                timestamp = latest_log['time']

                # Check if this is a new signal
                if state['last_check_time'] != str(timestamp):

                    # Save portfolio state before making any trades
                    logging.info("Saving current portfolio state...")
                    save_portfolio_state()

                    # Sell all existing positions first
                    logging.info("Clearing all existing positions...")
                    if not sell_all_positions():
                        logging.error("Failed to clear positions, skipping signal")
                        return

                    # ENTRY SIGNAL - Buy TQQQ
                    if event == 'entry':
                        logging.info("BUY SIGNAL: Buying TQQQ")

                        # Buy TQQQ
                        order = place_buy_order('TQQQ', POSITION_SIZE, price, note)
                        if order:
                            state['current_position'] = 'TQQQ'
                            state['entry_price'] = price
                            state['entry_time'] = str(timestamp)
                            state['entry_conditions'] = note  # Store detailed entry conditions
                            state['position_size'] = POSITION_SIZE
                            state['order_ids'].append(order.get('orderId'))

                    # EXIT SIGNAL - Buy SQQQ
                    elif event in ['exit_SL', 'exit_TP', 'exit_conditions_met']:
                        exit_reason = 'Stop Loss' if event == 'exit_SL' else 'Take Profit' if event == 'exit_TP' else 'Exit Conditions'
                        logging.info(f"SELL SIGNAL ({exit_reason}): Buying SQQQ")

                        # Calculate P&L if we were holding TQQQ
                        if state['current_position'] == 'TQQQ' and state['entry_price']:
                            pnl_pct = ((price - state['entry_price']) / state['entry_price']) * 100
                            pnl_dollars = (price - state['entry_price']) * state['position_size']
                            logging.info(f"TQQQ Trade P&L: ${pnl_dollars:.2f} ({pnl_pct:+.2f}%)")

                        # Buy SQQQ - pass previous entry conditions for context
                        previous_entry_conditions = state.get('entry_conditions')
                        order = place_buy_order('SQQQ', POSITION_SIZE, price, note, entry_conditions=previous_entry_conditions)
                        if order:
                            state['current_position'] = 'SQQQ'
                            state['entry_price'] = price
                            state['entry_time'] = str(timestamp)
                            state['entry_conditions'] = note  # Store new entry conditions for SQQQ
                            state['position_size'] = POSITION_SIZE
                            state['order_ids'].append(order.get('orderId'))

                    # Update last check time
                    state['last_check_time'] = str(timestamp)
                    save_state(state)
                else:
                    logging.info("No new signals (already processed)")
            else:
                logging.info("No recent signals in last 10 minutes")
        else:
            logging.info("No signals found")

        # Show current status
        current_rsi = df5['rsi'].iloc[-1] if 'rsi' in df5.columns else None

        status = f"Status: TQQQ Price=${current_price:.2f}"
        if current_rsi:
            status += f", RSI={current_rsi:.1f}"
        if state['current_position']:
            status += f", Holding={state['current_position']} ({state['position_size']} shares)"
            if state['entry_price']:
                pnl = ((current_price - state['entry_price']) / state['entry_price']) * 100
                status += f", P&L={pnl:+.2f}%"
        else:
            status += ", Position=NONE"

        logging.info(status)

    except Exception as e:
        logging.error(f"Error running strategy: {e}", exc_info=True)
        send_email_alert("❌ Strategy Error", f"Error: {e}")


def main():
    """Main trading loop"""
    logging.info("=" * 60)
    logging.info("STRATEGY TRADER - AUTOMATED PAIR TRADING")
    logging.info("=" * 60)

    account_type = "PAPER TRADING" if USE_PAPER else "⚠️  LIVE TRADING ⚠️"
    logging.info(f"Mode: {account_type}")
    logging.info(f"Position Size: {POSITION_SIZE} shares")
    logging.info(f"Strategy: TQQQ (long) / SQQQ (short)")
    logging.info(f"Email: {GMAIL_ADDRESS}")
    logging.info("Will check for signals every 5 minutes")
    logging.info("Press Ctrl+C to stop")
    logging.info("=" * 60)

    # Initialize Webull
    if not initialize_webull():
        logging.error("Cannot proceed without Webull connection")
        return

    # Run immediately on start
    run_strategy()

    # Then run every 5 minutes
    try:
        while True:
            time.sleep(300)  # 5 minutes
            run_strategy()
    except KeyboardInterrupt:
        logging.info("\nStrategy Trader Stopped")
        send_email_alert("🛑 Trading Bot Stopped", "Strategy trader has been stopped manually")


if __name__ == "__main__":
    main()
