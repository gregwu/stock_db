# ----------------------------------------
# ALPACA CONFIGURATION FILE
# ----------------------------------------
# This file ONLY contains API credentials and references.
# ALL trading settings are configured in alpaca.json
# ----------------------------------------
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca credentials (loaded from .env for security)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Email Configuration (loaded from .env for security)
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')

# ========================================
# IMPORTANT: ALL TRADING SETTINGS ARE IN alpaca.json
# ========================================
#
# DO NOT add settings here. Configure everything in alpaca.json:
#
# - use_paper (PAPER/LIVE trading mode)
# - position_size (default shares per trade)
# - check_interval_seconds (how often to check for signals)
# - email_notifications (on_bot_start, on_bot_stop, on_entry, on_exit, on_errors)
# - stop_loss / take_profit (in strategy.risk_management)
# - All strategy settings (entry/exit conditions, risk management)
# - All signal actions (what to do on each signal)
#
# Location: alpaca.json -> strategy -> trading
#
# Example:
# {
#   "strategy": {
#     "trading": {
#       "position_size": 100,
#       "check_interval_seconds": 120,
#       "use_paper": true,
#       "email_notifications": {
#         "enabled": true,
#         "on_bot_start": true,
#         "on_bot_stop": true,
#         "on_entry": true,
#         "on_exit": true,
#         "on_errors": true
#       }
#     }
#   }
# }
#
# ========================================

# Validate credentials
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("⚠️  WARNING: Alpaca API credentials not found in .env file")
    print("Please add the following to your .env file:")
    print("  ALPACA_API_KEY=your_api_key")
    print("  ALPACA_SECRET_KEY=your_secret_key")
    print("")
    print("Get your API keys from: https://app.alpaca.markets/paper/dashboard/overview")
    print("(Use Paper Trading keys for testing)")

if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
    print("")
    print("⚠️  WARNING: Gmail credentials not found in .env file")
    print("Email notifications will be disabled.")
    print("To enable email notifications, add to your .env file:")
    print("  GMAIL_ADDRESS=your-email@gmail.com")
    print("  GMAIL_APP_PASSWORD=your-16-char-app-password")
    print("")
    print("Get Gmail App Password from: https://myaccount.google.com/apppasswords")
