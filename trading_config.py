#!/usr/bin/env python3
"""
Shared trading configuration and utility functions
Used by both rules.py and alpaca_trader.py to avoid circular imports
"""
from datetime import datetime, time
import pytz

# Trading configuration constants
LIMIT_ORDER_SLIPPAGE_PCT = 2.0  # Default slippage percentage for limit orders

def is_market_hours():
    """
    Check if current time is within regular market hours (9:30 AM - 4:00 PM ET)

    Returns:
        bool: True if within market hours, False otherwise
    """
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    current_time = now_et.time()

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)

    # Check if today is a weekday (Monday=0, Sunday=6)
    is_weekday = now_et.weekday() < 5

    return is_weekday and market_open <= current_time <= market_close
