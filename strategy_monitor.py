#!/usr/bin/env python3
"""
Strategy Monitor - FREE VERSION using Email-to-SMS
No Twilio account needed!
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

# Import functions from rules.py
from rules import (
    rsi, ema, bollinger_bands, macd, backtest_symbol, load_settings
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_monitor.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# SMS Configuration
PHONE_NUMBER = os.getenv('PHONE_NUMBER', '4084689972')
CARRIER_GATEWAY = os.getenv('CARRIER_GATEWAY', 'txt.att.net')  # Default to AT&T

# Gmail Configuration (free)
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')  # App-specific password

# Tracking file to avoid duplicate alerts
STATE_FILE = '.strategy_monitor_state.json'


def load_state():
    """Load the last known state (positions, alerts sent)"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'last_alert_time': None,
        'current_position': None,
        'entry_price': None,
        'entry_time': None
    }


def save_state(state):
    """Save current state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def send_sms_via_email(message):
    """
    Send SMS using email-to-SMS gateway (100% FREE!)

    Common carrier gateways:
    - AT&T: txt.att.net
    - T-Mobile: tmomail.net
    - Verizon: vtext.com
    - Sprint: messaging.sprintpcs.com
    """
    if not all([GMAIL_ADDRESS, GMAIL_APP_PASSWORD]):
        logging.warning("Gmail credentials not configured. SMS not sent.")
        logging.info(f"Would have sent: {message}")
        return False

    try:
        # Create SMS address
        sms_address = f"{PHONE_NUMBER}@{CARRIER_GATEWAY}"

        # Create email message
        msg = MIMEText(message)
        msg['Subject'] = 'Trading Alert'
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = sms_address

        # Send via Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        logging.info(f"SMS sent successfully via email gateway")
        return True
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")
        return False


def run_strategy():
    """Run the trading strategy and check for signals"""
    logging.info("=" * 60)
    logging.info("Running strategy check...")

    # Load settings from rules.py
    settings = load_settings()
    if not settings:
        # Use default settings
        settings = {
            'ticker': 'TQQQ',
            'period': '1d',
            'interval': '1m',
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
            'use_macd_peak': False
        }

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

                # Get current price
                current_price = df['Close'].iloc[-1]

                # Check if this is a new signal (not already alerted)
                if state['last_alert_time'] != str(timestamp):
                    message = None

                    if event == 'entry':
                        # Keep message short for SMS (160 char limit)
                        message = f"BUY {ticker} ${price:.2f} @ {timestamp.strftime('%H:%M')}"

                        # Update state
                        state['current_position'] = 'long'
                        state['entry_price'] = price
                        state['entry_time'] = str(timestamp)

                    elif event in ['exit_SL', 'exit_TP', 'exit_conditions_met']:
                        exit_reason = 'SL' if event == 'exit_SL' else 'TP' if event == 'exit_TP' else 'Exit'

                        message = f"SELL {ticker} ${price:.2f} {exit_reason}"

                        # Calculate P&L if we had a position
                        if state['entry_price']:
                            pnl_pct = ((price - state['entry_price']) / state['entry_price']) * 100
                            message += f" {pnl_pct:+.1f}%"

                        # Clear position
                        state['current_position'] = None
                        state['entry_price'] = None
                        state['entry_time'] = None

                    if message:
                        logging.info(f"New signal detected: {event}")
                        logging.info(message)
                        send_sms_via_email(message)

                        # Update last alert time
                        state['last_alert_time'] = str(timestamp)
                        save_state(state)
                else:
                    logging.info("No new signals (already alerted)")
            else:
                logging.info("No recent signals in last 10 minutes")
        else:
            logging.info("No signals found")

        # Show current status
        current_price = df['Close'].iloc[-1]
        current_rsi = df5['rsi'].iloc[-1] if 'rsi' in df5.columns else None

        status = f"Status: Price=${current_price:.2f}"
        if current_rsi:
            status += f", RSI={current_rsi:.1f}"
        if state['current_position']:
            status += f", Position={state['current_position'].upper()}"
            if state['entry_price']:
                pnl = ((current_price - state['entry_price']) / state['entry_price']) * 100
                status += f", P&L={pnl:+.2f}%"
        else:
            status += ", Position=NONE"

        logging.info(status)

    except Exception as e:
        logging.error(f"Error running strategy: {e}", exc_info=True)


def main():
    """Main monitoring loop"""
    logging.info("Strategy Monitor Started (FREE VERSION)")
    logging.info(f"Phone Number: {PHONE_NUMBER}")
    logging.info(f"Carrier Gateway: {CARRIER_GATEWAY}")
    logging.info("Will check for signals every 5 minutes")
    logging.info("Press Ctrl+C to stop")

    # Check Gmail configuration
    if not all([GMAIL_ADDRESS, GMAIL_APP_PASSWORD]):
        logging.warning("⚠️  Gmail credentials not configured in .env file")
        logging.warning("Please update .env with:")
        logging.warning("  GMAIL_ADDRESS=your_email@gmail.com")
        logging.warning("  GMAIL_APP_PASSWORD=your_app_password")
        logging.warning("  CARRIER_GATEWAY=txt.att.net  (or your carrier)")
        logging.warning("")
        logging.warning("How to get Gmail App Password:")
        logging.warning("1. Go to: https://myaccount.google.com/apppasswords")
        logging.warning("2. Create app password for 'Mail'")
        logging.warning("3. Copy the 16-character password")
        logging.warning("")
        logging.warning("Common carrier gateways:")
        logging.warning("  AT&T:     txt.att.net")
        logging.warning("  T-Mobile: tmomail.net")
        logging.warning("  Verizon:  vtext.com")
        logging.warning("  Sprint:   messaging.sprintpcs.com")
        logging.warning("")
        logging.warning("Monitor will run but SMS alerts will not be sent.")
        logging.warning("")

    # Run immediately on start
    run_strategy()

    # Then run every 5 minutes
    try:
        while True:
            time.sleep(300)  # 5 minutes
            run_strategy()
    except KeyboardInterrupt:
        logging.info("\nStrategy Monitor Stopped")


if __name__ == "__main__":
    main()
