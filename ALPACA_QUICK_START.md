# Alpaca Trading - Quick Start

Get up and running with Alpaca trading in 5 minutes!

## Prerequisites

Your `.env` file already has Alpaca credentials:
```bash
ALPACA_API_KEY=PKSF6YFEW4FRPSPQUGJAUZZYFE
ALPACA_SECRET_KEY=bodvhmPSD6yTFkPbxL8rcuUiPKpTeZ6SbAKDNDcn4zh
```

## Step 1: Install Dependencies

```bash
pip install alpaca-py
```

Or install all requirements:
```bash
pip install -r requirements_alpaca.txt
```

## Step 2: Test Connection

```bash
python alpaca_wrapper.py
```

Expected output:
```
[+] Alpaca API initialized (PAPER account)
[+] Connected to Alpaca
[+] Account ID: xxx-xxx-xxx
[+] Buying Power: $100,000.00
✅ Alpaca API test completed successfully!
```

## Step 3: Configure Strategy Settings

Edit `alpaca_strategy_config.json` to customize your trading strategy:
- Entry/exit conditions (RSI, MACD, EMA, etc.)
- Ticker and timeframe
- Risk management (stop loss, take profit)
- Position size and check interval

Edit `alpaca_config.py` to change:
- Paper vs Live trading (default: Paper)

## Step 4: Start Trading Bot

```bash
./start_alpaca_trader.sh
```

The bot will:
- ✅ Load your strategy from alpaca_strategy_config.json
- ✅ Monitor for signals every 5 minutes
- ✅ Execute trades automatically
- ✅ Send email alerts
- ✅ Manage stop loss and take profit

## Management Commands

### Check Status
```bash
./alpaca_trader_status.sh
```

### View Logs
```bash
tail -f alpaca_trader.log
```

### Stop Bot
```bash
./stop_alpaca_trader.sh
```

## Key Features

### No 2FA Required
Unlike Webull, Alpaca uses API keys - no SMS codes needed!

### Paper Trading Default
Safe testing with virtual $100,000:
```python
USE_PAPER = True  # In alpaca_config.py
```

### Automatic Risk Management
- Stop loss: Exits at -2% loss
- Take profit: Exits at +3% gain

### Email Alerts
Get notified for:
- 🟢 Buy orders
- 🔴 Sell orders
- ⚠️ Stop loss triggers
- 🎯 Take profit executions
- ❌ Errors

## Strategy Settings

The bot uses settings from `alpaca_strategy_config.json`:
1. Edit the JSON file to configure your strategy
2. Set entry/exit conditions (RSI, MACD, EMA, etc.)
3. Configure risk management (stop loss, take profit)
4. Restart the bot to apply changes

This configuration is independent from the Streamlit UI, so your bot settings won't change when you test different strategies in the UI.

## Files Created

| File | Purpose |
|------|---------|
| `alpaca_wrapper.py` | API wrapper for Alpaca |
| `alpaca_config.py` | Trading mode configuration (Paper/Live) |
| `alpaca_strategy_config.json` | Strategy settings (entry/exit conditions, risk management) |
| `alpaca_trader.py` | Main trading bot |
| `start_alpaca_trader.sh` | Start bot script |
| `stop_alpaca_trader.sh` | Stop bot script |
| `alpaca_trader_status.sh` | Check status script |
| `.alpaca_trader_state.json` | Bot state file |
| `alpaca_trader.log` | Log file |

## Monitoring

### Current Position
```bash
cat .alpaca_trader_state.json
```

Shows:
- Current position (TQQQ or SQQQ)
- Entry price and time
- Position size
- Entry conditions

### Portfolio
Check your positions at:
- Paper: https://app.alpaca.markets/paper/dashboard/overview
- Live: https://app.alpaca.markets/live/dashboard/overview

## Troubleshooting

### Bot won't start
- Check if strategy settings exist: `ls -la alpaca_strategy_config.json`
- Create the config file if missing (see alpaca_strategy_config.json template)

### No trades executing
- Verify strategy conditions are met
- Check logs: `tail -f alpaca_trader.log`
- Market must be open (9:30 AM - 4:00 PM ET)

### Email alerts not working
- Check Gmail credentials in `.env`
- Verify app password is correct

## Safety Checklist

Before switching to live trading:

- [ ] Test with paper trading for at least 1 week
- [ ] Review all trades in paper account
- [ ] Verify stop loss and take profit work correctly
- [ ] Check email alerts are working
- [ ] Understand the strategy completely
- [ ] Start with small position sizes
- [ ] Monitor closely for first few days

## Next Steps

1. ✅ Keep bot running in paper trading mode
2. ✅ Monitor performance daily
3. ✅ Review trades and adjust strategy
4. ✅ Test different settings
5. ⚠️ Only go live after thorough testing

## Support

Need help? Check:
- [Alpaca Setup Guide](ALPACA_SETUP.md) - Full detailed guide
- [Alpaca Documentation](https://docs.alpaca.markets/)
- Bot logs: `alpaca_trader.log`

## Comparison: Webull vs Alpaca

| Feature | Webull | Alpaca |
|---------|--------|--------|
| API Access | Requires 2FA | Simple API keys ✅ |
| Paper Trading | Limited | Full featured ✅ |
| Commission | Free ✅ | Free ✅ |
| Extended Hours | Yes ✅ | Yes ✅ |
| Setup Complexity | High | Low ✅ |
| API Reliability | Medium | High ✅ |

**Recommendation**: Use Alpaca for automated trading - it's much simpler!
