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
        'current_position': None,  # 'long' or None
        'entry_price': None,
        'entry_time': None,
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


def initialize_webull():
    """Initialize and login to Webull"""
    global wb_api

    try:
        wb_api = WebullAPI()
        wb_api.login()

        account_type = "PAPER" if USE_PAPER else "LIVE"
        logging.info(f"Webull connected: {account_type} account")

        send_email_alert(
            "🤖 Trading Bot Started",
            f"Strategy Trader initialized\nAccount: {account_type}\nPosition Size: {POSITION_SIZE} shares"
        )
        return True

    except Exception as e:
        logging.error(f"Failed to initialize Webull: {e}")
        send_email_alert(
            "❌ Trading Bot Error",
            f"Failed to connect to Webull: {e}"
        )
        return False


def place_buy_order(ticker, qty, price, reason):
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
Reason: {reason}

Order ID: {order.get('orderId', 'N/A')}
"""

        send_email_alert("🟢 BUY ORDER", message)

        return order

    except Exception as e:
        logging.error(f"Buy order failed: {e}")
        send_email_alert("❌ BUY ORDER FAILED", f"Error: {e}")
        return None


def place_sell_order(ticker, qty, price, reason):
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
Reason: {reason}

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

                    # ENTRY SIGNAL
                    if event == 'entry' and not state['current_position']:
                        order = place_buy_order(ticker, POSITION_SIZE, price, note)

                        if order:
                            state['current_position'] = 'long'
                            state['entry_price'] = price
                            state['entry_time'] = str(timestamp)
                            state['position_size'] = POSITION_SIZE
                            state['order_ids'].append(order.get('orderId'))

                    # EXIT SIGNAL
                    elif event in ['exit_SL', 'exit_TP', 'exit_conditions_met'] and state['current_position']:
                        exit_reason = 'Stop Loss' if event == 'exit_SL' else 'Take Profit' if event == 'exit_TP' else 'Exit Conditions'

                        order = place_sell_order(ticker, state['position_size'], price, f"{exit_reason}: {note}")

                        if order:
                            # Calculate P&L
                            if state['entry_price']:
                                pnl_pct = ((price - state['entry_price']) / state['entry_price']) * 100
                                pnl_dollars = (price - state['entry_price']) * state['position_size']

                                logging.info(f"Trade P&L: ${pnl_dollars:.2f} ({pnl_pct:+.2f}%)")

                            # Clear position
                            state['current_position'] = None
                            state['entry_price'] = None
                            state['entry_time'] = None
                            state['position_size'] = 0

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

        status = f"Status: Price=${current_price:.2f}"
        if current_rsi:
            status += f", RSI={current_rsi:.1f}"
        if state['current_position']:
            status += f", Position=LONG ({state['position_size']} shares)"
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
    logging.info("STRATEGY TRADER - AUTOMATED TRADING")
    logging.info("=" * 60)

    account_type = "PAPER TRADING" if USE_PAPER else "⚠️  LIVE TRADING ⚠️"
    logging.info(f"Mode: {account_type}")
    logging.info(f"Position Size: {POSITION_SIZE} shares")
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
