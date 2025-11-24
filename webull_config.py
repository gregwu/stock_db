# ----------------------------------------
# CONFIGURATION FILE
# ----------------------------------------
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Switch between LIVE and PAPER account
USE_PAPER = False       # False = LIVE trading | True = PAPER trading

# Webull credentials (loaded from .env for security)
EMAIL = os.getenv('WEBULL_EMAIL')
PASSWORD = os.getenv('WEBULL_PASSWORD')
DEVICE_NAME = os.getenv('WEBULL_DEVICE_NAME', 'my_device')  # anything unique

# Optional 2FA code function
def get_2fa_code():
    code = input("Enter Webull 2FA Code: ")
    return code

# Trading settings
TICKER = "TQQQ"     # or SQQQ or QQQ
POSITION_SIZE = 100  # shares per trade
STOP_LOSS_PCT = -0.02
TAKE_PROFIT_PCT = 0.03

# Logging
LOG_FILE = "trade_log.txt"

