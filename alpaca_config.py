# ----------------------------------------
# ALPACA CONFIGURATION FILE
# ----------------------------------------
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Switch between LIVE and PAPER account
USE_PAPER = True  # True = PAPER trading | False = LIVE trading ⚠️

# Alpaca credentials (loaded from .env for security)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Trading settings
# NOTE: TICKER and POSITION_SIZE are now configured in alpaca_signal_actions.json
# Each action can specify its own ticker and quantity for maximum flexibility
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.03  # 3% take profit
POSITION_SIZE = 100  # Default position size (fallback if not specified in action)

# Logging
LOG_FILE = "alpaca_trader.log"

# Validate credentials
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("⚠️  WARNING: Alpaca API credentials not found in .env file")
    print("Please add the following to your .env file:")
    print("  ALPACA_API_KEY=your_api_key")
    print("  ALPACA_SECRET_KEY=your_secret_key")
    print("")
    print("Get your API keys from: https://app.alpaca.markets/paper/dashboard/overview")
    print("(Use Paper Trading keys for testing)")
