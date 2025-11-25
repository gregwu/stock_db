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
PHONE = os.getenv('WEBULL_PHONE')
PASSWORD = os.getenv('WEBULL_PASSWORD')
DEVICE_NAME = os.getenv('WEBULL_DEVICE_NAME', 'my_device')  # anything unique
LOGIN_METHOD = os.getenv('WEBULL_LOGIN_METHOD', 'email').lower()  # 'email' or 'phone'
TRADE_TOKEN = os.getenv('WEBULL_TRADE_TOKEN', '')  # Trade token for placing orders

# Optional 2FA code function
def get_2fa_code():
    """
    Get 2FA code from user

    Webull sends 2FA codes via SMS when using phone number login.
    """
    print("\n" + "=" * 60)
    print("WEBULL 2FA REQUIRED")
    print("=" * 60)

    if LOGIN_METHOD == 'phone' and PHONE:
        print(f"\n📱 SMS sent to: ***-***-{PHONE[-4:]}")
        print("\nCheck your text messages for the 6-digit code from Webull.")
    else:
        print("\nWhere to find your 2FA code:")
        print("1. Check SMS on your phone (registered with Webull)")
        print("2. Open Webull mobile app → Check notifications/messages")
        print("3. Check email (if email 2FA is enabled)")

    print("\nThe code typically arrives within 30 seconds and expires in 2-5 minutes.")
    print("=" * 60)

    code = input("\nEnter the 6-digit 2FA code: ").strip()

    if not code:
        print("⚠️  No code entered. Login will fail.")
    elif not code.isdigit():
        print("⚠️  Warning: Code should be numeric digits only")
    elif len(code) != 6:
        print(f"⚠️  Warning: Code should be 6 digits (you entered {len(code)})")

    return code

# Trading settings
TICKER = "TQQQ"     # or SQQQ or QQQ
POSITION_SIZE = 10  # shares per trade
STOP_LOSS_PCT = -0.02
TAKE_PROFIT_PCT = 0.03

# Logging
LOG_FILE = "trade_log.txt"

