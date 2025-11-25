# Alpaca Trading Setup Guide

This guide will help you set up automated trading with Alpaca.

## Why Alpaca?

- **Commission-free trading** on stocks and ETFs
- **No account minimum** for paper trading
- **Simple REST API** - no 2FA hassles
- **Paper trading** account for risk-free testing
- **Real-time market data** included

## Step 1: Create Alpaca Account

1. Go to [Alpaca Markets](https://alpaca.markets/)
2. Click "Sign Up" and create an account
3. Complete the verification process (for live trading) or skip for paper trading

## Step 2: Get API Keys

### Paper Trading (Recommended for Testing)

1. Go to [Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
2. Click on "API Keys" in the left sidebar
3. Click "Generate New Key"
4. Give it a name like "Trading Bot"
5. Copy both the **API Key** and **Secret Key** immediately
   - ⚠️ You can only see the Secret Key once!

### Live Trading (Real Money)

1. Go to [Live Trading Dashboard](https://app.alpaca.markets/live/dashboard/overview)
2. Follow the same steps as paper trading
3. ⚠️ **Warning**: Use live trading only after thorough testing with paper trading!

## Step 3: Configure .env File

Add your Alpaca API credentials to the `.env` file:

```bash
# Alpaca Trading Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

**Example:**
```bash
ALPACA_API_KEY=PKABCDEF1234567890
ALPACA_SECRET_KEY=xyz123abc456def789ghi012jkl345mno678pqr901stu234
```

## Step 4: Install Dependencies

Install the Alpaca Python library:

```bash
pip install alpaca-py
```

Or if using the requirements file:

```bash
pip install -r requirements_alpaca.txt
```

## Step 5: Test Connection

Test your Alpaca connection:

```bash
python alpaca_wrapper.py
```

This will:
- Connect to your Alpaca account
- Display account information (equity, cash, buying power)
- Get a quote for TQQQ
- List current positions and orders

**Expected output:**
```
Testing Alpaca API...
[+] Alpaca API initialized (PAPER account)
[+] Connected to Alpaca
[+] Account ID: xxx-xxx-xxx
[+] Buying Power: $100,000.00
[+] Portfolio Value: $100,000.00
[+] Cash: $100,000.00

[+] Getting quote for TQQQ...
  Last: $45.23
  Bid: $45.22
  Ask: $45.24

✅ Alpaca API test completed successfully!
```

## Step 6: Configure Trading Settings

Edit `alpaca_config.py` to customize:

```python
# Switch between LIVE and PAPER account
USE_PAPER = True  # True = PAPER trading | False = LIVE trading ⚠️

# Trading settings
TICKER = "TQQQ"     # Main ticker for strategy signals
POSITION_SIZE = 10  # shares per trade
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.03  # 3% take profit
```

## Step 7: Run the Trading Bot

Start the Alpaca trading bot:

```bash
python alpaca_trader.py
```

The bot will:
1. Connect to Alpaca
2. Load your strategy settings from the Streamlit app
3. Monitor for trading signals every 5 minutes
4. Execute trades automatically based on signals
5. Send email alerts for all trades
6. Manage stop loss and take profit levels

## Configuration Files

### alpaca_config.py
- Trading mode (paper vs live)
- Position size
- Stop loss and take profit percentages

### alpaca_wrapper.py
- API wrapper for Alpaca
- Handles orders, quotes, positions, portfolio

### alpaca_trader.py
- Main trading bot
- Strategy execution
- Email alerts
- SL/TP management

## Safety Features

### Paper Trading Default
The config defaults to paper trading for safety:
```python
USE_PAPER = True  # True = PAPER trading | False = LIVE trading
```

### Stop Loss Protection
Automatic stop loss at -2% (configurable):
```python
STOP_LOSS_PCT = 0.02  # 2% stop loss
```

### Take Profit
Automatic profit taking at +3% (configurable):
```python
TAKE_PROFIT_PCT = 0.03  # 3% take profit
```

### Email Alerts
Get notified for every trade:
- Order confirmations
- Stop loss triggers
- Take profit executions
- Errors and warnings

## Monitoring

### Check Bot Status
```bash
# View recent logs
tail -f alpaca_trader.log

# View current state
cat .alpaca_trader_state.json

# View portfolio state
cat .alpaca_portfolio_state.json
```

### State File
The bot maintains state in `.alpaca_trader_state.json`:
```json
{
  "last_check_time": "2025-11-25 10:30:00",
  "current_position": "TQQQ",
  "entry_price": 45.23,
  "entry_time": "2025-11-25 09:45:00",
  "entry_conditions": "Entry: RSI < 30 (RSI=28.5)...",
  "position_size": 10,
  "order_ids": ["xxx-xxx-xxx"]
}
```

## Comparing with Webull

### Alpaca Advantages
✅ No 2FA required for API access
✅ Simpler authentication (just API keys)
✅ Better paper trading support
✅ More reliable API
✅ Commission-free trading
✅ Real-time data included

### Webull Advantages
✅ Extended hours trading
✅ More international markets
✅ Mobile app with better UI

## Troubleshooting

### "API credentials not found"
- Make sure you added `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to `.env`
- Check that the keys are correct (no extra spaces)

### "Failed to connect to Alpaca"
- Verify your API keys are valid
- Check if you're using paper trading keys with `USE_PAPER = True`
- Make sure your internet connection is working

### "Order rejected"
- Check if you have sufficient buying power
- Verify the ticker symbol is valid
- Check if market is open (9:30 AM - 4:00 PM ET)

### "No positions found"
- This is normal if you haven't made any trades yet
- Paper accounts start with $100,000 cash

## Next Steps

1. ✅ Test with paper trading first
2. ✅ Monitor for a few days to verify strategy
3. ✅ Review all trades and alerts
4. ✅ Adjust strategy settings as needed
5. ⚠️ Only switch to live trading after thorough testing

## Support

- [Alpaca Documentation](https://docs.alpaca.markets/)
- [Alpaca Python SDK](https://alpaca.markets/docs/python-sdk/)
- [Alpaca Community Forum](https://forum.alpaca.markets/)
- [API Status](https://status.alpaca.markets/)

## Important Notes

⚠️ **Risk Warning**: Trading involves risk. Use paper trading for testing before risking real money.

⚠️ **Market Hours**: Alpaca supports regular hours (9:30 AM - 4:00 PM ET) and extended hours trading.

⚠️ **Pattern Day Trader Rule**: If your account is under $25,000, you're limited to 3 day trades per 5 trading days.

⚠️ **API Rate Limits**: Alpaca has rate limits on API calls. The bot checks every 5 minutes to stay within limits.
