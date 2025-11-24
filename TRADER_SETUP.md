# Strategy Trader Setup Guide

## Overview

**strategy_trader.py** is an automated trading system that:
- Monitors your trading strategy every 5 minutes
- Places **real orders** via Webull API when signals are detected
- Sends email alerts for all trades
- Tracks positions and calculates P&L
- Supports both Paper Trading (safe) and Live Trading

---

## 🚨 IMPORTANT: Monitor vs Trader

Your project has **TWO different systems**:

| Feature | strategy_monitor.py | strategy_trader.py |
|---------|-------------------|-------------------|
| **Purpose** | Watch-only alerts | Automated trading |
| **Actions** | Sends emails | Places real orders |
| **Risk** | None | Your money |
| **Use when** | Testing strategies | Proven strategy |

⚠️ **Never run both at the same time!** Pick one:
- **Testing phase**: Use monitor (emails only)
- **Trading phase**: Use trader (real orders)

---

## Prerequisites

### 1. Webull Account
- Sign up at [webull.com](https://www.webull.com)
- Enable Paper Trading account (recommended for testing)
- Have your login credentials ready

### 2. Gmail Setup
Follow the same setup as strategy_monitor:
1. Gmail account with "App Password" generated
2. Update `.env` with:
   ```bash
   GMAIL_ADDRESS=your_email@gmail.com
   GMAIL_APP_PASSWORD=your_16_char_app_password
   ```

### 3. Python Dependencies
```bash
pip install python-dotenv yfinance pandas numpy scipy webull
```

---

## Configuration

### Step 1: Update .env with Webull Credentials

Edit `.env` and add your Webull credentials:

```bash
# Webull Trading Configuration
WEBULL_EMAIL=your_webull_email@example.com
WEBULL_PASSWORD=your_webull_password
WEBULL_DEVICE_NAME=my_laptop
```

**⚠️ SECURITY**: Never commit `.env` to version control! Your credentials are stored securely here.

### Step 2: Configure Trading Settings

Edit `webull_config.py` to adjust trading parameters:

```python
# Switch between LIVE and PAPER account
USE_PAPER = True  # ⚠️ KEEP THIS True UNTIL YOU'RE READY!

# Trading settings
TICKER = "TQQQ"
POSITION_SIZE = 10  # shares per trade
STOP_LOSS_PCT = -0.02  # -2%
TAKE_PROFIT_PCT = 0.03  # +3%
```

**⚠️ CRITICAL**: Leave `USE_PAPER = True` for testing!

### Step 3: Test Webull Connection

```bash
python3 -c "from webull_wrapper import WebullAPI; wb = WebullAPI(); wb.login(); print('✅ Connected')"
```

You'll be prompted for your 2FA code. If login succeeds, you're ready.

### Step 3: Test Email Alerts

```bash
python3 test_sms.py
```

You should receive a test email within seconds.

---

## Running the Trader

### Start Trading Bot

```bash
chmod +x start_trader.sh
./start_trader.sh
```

**What happens:**
1. Checks if webull_config.py is configured
2. Warns if LIVE trading is enabled
3. Starts trader in background
4. Sends you a startup email alert

### Check Status

```bash
chmod +x trader_status.sh
./trader_status.sh
```

Shows:
- Running status
- Current position (if any)
- Recent logs
- P&L for open position

### Stop Trading Bot

```bash
chmod +x stop_trader.sh
./stop_trader.sh
```

Gracefully stops the trader. Your final position is saved.

### View Live Logs

```bash
tail -f strategy_trader.log
```

Press `Ctrl+C` to exit log view.

---

## How It Works

### 1. Signal Detection
Every 5 minutes, the trader:
- Downloads recent price data via yfinance
- Runs your backtest strategy from `rules.py`
- Checks for entry/exit signals in the last 10 minutes

### 2. Entry Signal
When a **BUY signal** is detected:
```
✅ Entry conditions met: RSI < 30, Price below EMA9
→ Places BUY order for 10 shares of TQQQ
→ Sends email: "🟢 BUY ORDER PLACED"
→ Tracks entry price and position
```

### 3. Exit Signal
When an **EXIT signal** is detected:
```
🔴 Stop loss triggered: Price dropped 2%
→ Places SELL order for 10 shares of TQQQ
→ Sends email: "🔴 SELL ORDER - Stop Loss"
→ Calculates P&L: -$4.50 (-2.0%)
→ Clears position
```

### 4. Email Alerts
You receive emails for:
- **Startup**: "🤖 Trading Bot Started"
- **Buy orders**: "🟢 BUY ORDER PLACED"
- **Sell orders**: "🔴 SELL ORDER PLACED"
- **Errors**: "❌ ORDER FAILED"
- **Shutdown**: "🛑 Trading Bot Stopped"

---

## Paper Trading vs Live Trading

### Paper Trading (Recommended First)
```python
USE_PAPER = True
```

**Pros:**
- ✅ No real money at risk
- ✅ Test your strategy safely
- ✅ Same order flow as live
- ✅ See how bot performs

**Cons:**
- ⚠️ Fills may be unrealistic (instant fills)
- ⚠️ No slippage simulation

**Use for:** Testing strategy for at least 1-2 weeks

### Live Trading
```python
USE_PAPER = False
```

**Pros:**
- ✅ Real profits possible
- ✅ Realistic order fills
- ✅ True market conditions

**Cons:**
- 🚨 **REAL MONEY AT RISK**
- 🚨 Losses are permanent
- 🚨 No undo button

**Use for:** Only after paper trading proves profitable

---

## Safety Features

1. **Default to Paper**: Script defaults to `USE_PAPER = True`
2. **Confirmation Prompt**: Requires typing "yes" for live trading
3. **Email Alerts**: Every action sends an alert
4. **State Tracking**: Position saved in `.strategy_trader_state.json`
5. **No Duplicate Orders**: Checks last signal timestamp
6. **Stop Loss**: Automatic exit if position drops 2%
7. **Take Profit**: Automatic exit if position gains 3%

---

## Position Management

### Current Position Tracking

The trader saves state in `.strategy_trader_state.json`:

```json
{
  "last_check_time": "2025-01-15 10:35:00",
  "current_position": "long",
  "entry_price": 45.32,
  "entry_time": "2025-01-15 09:30:00",
  "position_size": 10,
  "order_ids": ["WB123456789"]
}
```

### Position Lifecycle

```
1. No Position → Entry Signal Detected
2. Place BUY order
3. Position: LONG (track entry price)
4. Monitor P&L on every check
5. Exit Signal Detected (SL/TP/Conditions)
6. Place SELL order
7. Calculate final P&L
8. Position: NONE
```

### Multiple Positions
Currently supports **one position at a time**. The trader will:
- Ignore entry signals if already in a position
- Only exit when stop loss, take profit, or exit conditions are met

---

## Troubleshooting

### "Login failed"
- Check EMAIL and PASSWORD in webull_config.py
- Verify 2FA code is correct
- Try logging into Webull website manually first

### "No signals found"
- Strategy may not have conditions met
- Check if market is open (for intraday strategies)
- View logs: `tail -f strategy_trader.log`

### "Order failed"
- Check if you have buying power
- Verify ticker symbol is correct
- Check if market is open for trading

### "Email alerts not working"
- Run `python3 test_sms.py` to test
- Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env
- Verify Gmail app password is correct (16 characters, no spaces)

### "Trader stopped unexpectedly"
- Check logs: `cat strategy_trader.log`
- Look for error messages
- Verify internet connection
- Check if Webull session expired (re-login required)

---

## Strategy Configuration

Your trading strategy is configured in **Streamlit UI** (`rules.py`):
- RSI thresholds
- EMA crossovers
- MACD peaks/valleys
- Stop loss / Take profit percentages

The trader automatically uses these settings from `.strategy_settings.json`.

**To update strategy:**
1. Run Streamlit: `streamlit run rules.py`
2. Adjust parameters in sidebar
3. Click "Save Settings"
4. Restart trader to use new settings

---

## Best Practices

### 1. Start Small
```python
POSITION_SIZE = 1  # Start with 1 share
```

### 2. Paper Trade First
- Run paper trading for 1-2 weeks minimum
- Track win rate and P&L
- Make sure you're profitable before going live

### 3. Set Realistic Limits
```python
STOP_LOSS_PCT = -0.02  # Exit at -2% loss
TAKE_PROFIT_PCT = 0.03  # Exit at +3% gain
```

### 4. Monitor Regularly
- Check `./trader_status.sh` daily
- Review email alerts
- Look at logs for any issues

### 5. Test Strategy Changes
- Always test new strategies in paper mode first
- Never change strategy while in an open position
- Document what settings work

### 6. Risk Management
- Never risk more than 1-2% of account per trade
- Don't trade more than 3-5 positions simultaneously
- Keep position size reasonable

---

## Going Live Checklist

Before switching `USE_PAPER = False`:

- [ ] Paper traded for at least 2 weeks
- [ ] Strategy is profitable in paper trading
- [ ] Win rate is acceptable (>50%)
- [ ] Email alerts working reliably
- [ ] Understand all entry/exit rules
- [ ] Have realistic expectations
- [ ] Started with small position size
- [ ] Set appropriate stop loss
- [ ] Have sufficient buying power
- [ ] Mentally prepared for losses

**Remember:** Past paper performance doesn't guarantee live results!

---

## Files Reference

### Main Files
- `strategy_trader.py` - Main trading bot script
- `webull_wrapper.py` - Webull API interface
- `webull_config.py` - Trading configuration
- `rules.py` - Strategy logic (used by both monitor and trader)

### Control Scripts
- `start_trader.sh` - Start the bot
- `stop_trader.sh` - Stop the bot
- `trader_status.sh` - Check status

### Data Files
- `.strategy_trader_state.json` - Current position state
- `strategy_trader.log` - Activity logs
- `.trader.pid` - Process ID file

### Configuration
- `.env` - Email credentials
- `.strategy_settings.json` - Strategy parameters from Streamlit

---

## Support

If you encounter issues:
1. Check logs: `tail -f strategy_trader.log`
2. Verify configuration: `cat webull_config.py`
3. Test components individually:
   - Webull connection: `python3 -c "from webull_wrapper import WebullAPI; wb = WebullAPI(); wb.login()"`
   - Email alerts: `python3 test_sms.py`
   - Strategy: `streamlit run rules.py`

---

## Legal Disclaimer

**USE AT YOUR OWN RISK**

This automated trading system:
- Can result in financial losses
- May have bugs or unexpected behavior
- Does not guarantee profits
- Is provided "as-is" without warranty

The author is not responsible for:
- Trading losses
- Order execution errors
- Software bugs
- Market conditions

**You are responsible for:**
- Understanding the code
- Testing thoroughly
- Managing your risk
- Monitoring your trades
- All trading decisions

Automated trading carries significant risk. Only trade with money you can afford to lose.

---

## Quick Reference

```bash
# Start trading bot
./start_trader.sh

# Check status
./trader_status.sh

# View logs
tail -f strategy_trader.log

# Stop trading bot
./stop_trader.sh

# Test email
python3 test_sms.py

# Change strategy
streamlit run rules.py
```

**Happy Trading! 📈**

*(But seriously, test in paper mode first)* 🧪
