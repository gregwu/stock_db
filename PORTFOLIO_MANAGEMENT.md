# Portfolio Management & Position Clearing

## Overview

The trading bot now includes **portfolio management** features:
1. **Save portfolio state** before executing any trades
2. **Sell all existing positions** before entering new positions
3. **Periodic portfolio snapshots** every 5 minutes

This ensures clean position management and full transparency of your holdings.

---

## New Features

### 1. Portfolio State Saving

The bot automatically saves your current portfolio to `.portfolio_state.json`:

```json
{
  "timestamp": "2025-01-15T10:35:00",
  "positions": [
    {
      "ticker": "TQQQ",
      "quantity": 10,
      "cost_basis": 62.30,
      "current_price": 64.50,
      "unrealized_pnl": 22.00,
      "unrealized_pnl_pct": 0.0353
    }
  ]
}
```

**When saved:**
- On bot startup (shows initial holdings)
- Every 5 minutes during monitoring
- Before executing any trade signal

### 2. Position Clearing Before Trades

When a new signal is detected:

```
1. Save current portfolio state
   ↓
2. Sell ALL existing positions (TQQQ, SQQQ, or any other holdings)
   ↓
3. Execute new signal (Buy TQQQ or Buy SQQQ)
   ↓
4. Update state and continue monitoring
```

**Benefits:**
- ✅ Clean slate for each new signal
- ✅ No mixed positions (e.g., holding both TQQQ and SQQQ)
- ✅ Avoid position size drift from partial fills
- ✅ Clear audit trail of portfolio changes

### 3. Startup Portfolio Report

When the bot starts, you receive an email showing your current holdings:

```
🤖 Trading Bot Started

Strategy Trader initialized
Account: PAPER
Position Size: 10 shares
Strategy: TQQQ/SQQQ Pair Trading

Buy Signal → Buy TQQQ, Sell SQQQ
Sell Signal → Sell TQQQ, Buy SQQQ

Current Holdings:
  TQQQ: 10 shares ($22.00 P&L)
  SQQQ: 5 shares (-$8.50 P&L)
```

This lets you see exactly what's in your account before the bot starts trading.

---

## Trading Workflow

### Scenario: Bot Detects BUY Signal

```
Current State: Holding 5 shares SQQQ from previous trade

Step 1: Save Portfolio
  → .portfolio_state.json updated
  → Timestamp: 2025-01-15 10:35:00
  → Position: 5 SQQQ @ $45.20

Step 2: Clear Positions
  → Sell 5 shares SQQQ @ $45.20
  → Order placed: SELL SQQQ
  → Order filled: +$226.00

Step 3: Execute Signal
  → Buy 10 shares TQQQ @ $62.30
  → Order placed: BUY TQQQ
  → Order filled: -$623.00

Step 4: Update State
  → current_position: "TQQQ"
  → position_size: 10
  → entry_price: 62.30
```

### Scenario: Bot Detects SELL Signal

```
Current State: Holding 10 shares TQQQ from previous trade

Step 1: Save Portfolio
  → .portfolio_state.json updated
  → Timestamp: 2025-01-15 11:05:00
  → Position: 10 TQQQ @ $64.50

Step 2: Clear Positions
  → Sell 10 shares TQQQ @ $64.50
  → Order placed: SELL TQQQ
  → Order filled: +$645.00
  → P&L: $22.00 (+3.53%)

Step 3: Execute Signal
  → Buy 10 shares SQQQ @ $43.80
  → Order placed: BUY SQQQ
  → Order filled: -$438.00

Step 4: Update State
  → current_position: "SQQQ"
  → position_size: 10
  → entry_price: 43.80
```

---

## File Structure

### `.portfolio_state.json`
Updated every 5 minutes and before each trade:
- Current timestamp
- All positions in account
- Cost basis, current price, P&L for each position

### `.strategy_trader_state.json`
Tracks the bot's internal state:
- Last signal timestamp
- Current position (TQQQ/SQQQ/None)
- Entry price and time
- Position size
- Order IDs

---

## Benefits

### ✅ Clean Position Management
- No leftover positions from manual trades
- No mixed positions (holding both TQQQ and SQQQ)
- Always exactly POSITION_SIZE shares

### ✅ Full Transparency
- Know exactly what's in your account at all times
- Portfolio snapshots saved every 5 minutes
- Clear audit trail of all changes

### ✅ Reduced Errors
- Avoids "insufficient shares" errors from position drift
- Prevents accumulation of small positions
- Ensures consistent position sizing

### ✅ Easy Debugging
- Review `.portfolio_state.json` to see account state
- Compare before/after portfolio for each trade
- Track P&L accurately

---

## Edge Cases Handled

### Mixed Positions
```
Before: Holding 10 TQQQ + 5 SQQQ (manual trades)
Signal: BUY TQQQ

Actions:
1. Sell 10 TQQQ → Clear first position
2. Sell 5 SQQQ → Clear second position
3. Buy 10 TQQQ → Execute new signal

After: Holding 10 TQQQ (clean)
```

### Partial Fills
```
Before: Holding 7 TQQQ (partial fill from previous signal)
Signal: SELL (exit to SQQQ)

Actions:
1. Sell 7 TQQQ → Clear whatever we have
2. Buy 10 SQQQ → Execute new signal (full POSITION_SIZE)

After: Holding 10 SQQQ (back to standard size)
```

### Multiple Tickers
```
Before: Holding 10 TQQQ + 20 QQQ + 5 AAPL (manual positions)
Signal: SELL (exit to SQQQ)

Actions:
1. Sell 10 TQQQ → Clear position 1
2. Sell 20 QQQ → Clear position 2
3. Sell 5 AAPL → Clear position 3
4. Buy 10 SQQQ → Execute new signal

After: Holding 10 SQQQ (clean slate)
```

---

## Monitoring

### Check Portfolio State
```bash
cat .portfolio_state.json
```

Shows most recent portfolio snapshot.

### View Portfolio History
```bash
# Search logs for portfolio saves
grep "Portfolio saved" strategy_trader.log
```

### Compare Before/After
```bash
# Before trade
2025-01-15 10:35:00 - INFO - Portfolio saved: 1 positions

# After trade
2025-01-15 10:35:15 - INFO - Successfully sold SQQQ
2025-01-15 10:35:20 - INFO - BUY ORDER PLACED: TQQQ
```

---

## Configuration

No additional configuration needed! The portfolio management features work automatically.

### Optional: Disable Position Clearing

If you want to keep existing positions and only add new ones, you can comment out the clearing step in `strategy_trader.py`:

```python
# # Sell all existing positions first
# logging.info("Clearing all existing positions...")
# if not sell_all_positions():
#     logging.error("Failed to clear positions, skipping signal")
#     return
```

**⚠️ Not recommended** - this can lead to mixed positions and sizing issues.

---

## Troubleshooting

### "Failed to get positions"
- Check Webull connection
- Verify account is logged in
- Try restarting the bot

### "Failed to sell positions"
- Check if positions have pending orders
- Verify market is open for trading
- Check buying power for sell orders

### Portfolio file empty
- Bot may not have connected yet
- Check logs for errors: `tail -f strategy_trader.log`
- Manually save: `save_portfolio_state()` in Python

### Position size mismatch
```
Expected: 10 shares TQQQ
Actual: 7 shares TQQQ

Cause: Partial fill from previous order
Fix: Next signal will clear all and buy correct size
```

---

## API Methods

### `save_portfolio_state()`
Saves current portfolio to `.portfolio_state.json`
- Returns: portfolio state dict
- Called: Every 5 min + before trades

### `sell_all_positions()`
Sells all positions in account
- Returns: True if successful, False otherwise
- Called: Before executing new signals

### `get_positions()`
Gets current positions from Webull API
- Returns: List of position dicts
- Called: By save/sell functions

---

## Safety Features

### Fail-Safe on Sell Errors
```python
if not sell_all_positions():
    logging.error("Failed to clear positions, skipping signal")
    return  # Don't execute new signal if clearing failed
```

If the bot can't sell existing positions, it won't place new orders to avoid over-leveraging.

### Portfolio Snapshots
Every action is logged with timestamps:
- When portfolio was saved
- Which positions were sold
- What orders were placed
- Final state after trades

### Email Alerts
Receive alerts for:
- ✅ Successful position clears
- ❌ Failed position clears
- 📊 Portfolio state on startup

---

## Best Practices

### 1. Review Startup Email
Check the initial holdings report to ensure you're starting with the expected positions.

### 2. Monitor Portfolio File
```bash
watch -n 60 cat .portfolio_state.json
```
See real-time updates every minute.

### 3. Archive Portfolio States
```bash
# Save daily snapshots
cp .portfolio_state.json backups/portfolio_$(date +%Y%m%d_%H%M%S).json
```

### 4. Clean Slate Testing
Start with empty account (all cash) for first test run to see the full workflow.

---

## Summary

The bot now provides **complete portfolio management**:

✅ Saves portfolio state every 5 minutes
✅ Shows initial holdings on startup
✅ Clears ALL positions before new signals
✅ Ensures consistent position sizing
✅ Provides full audit trail

This eliminates common issues like mixed positions, partial fills, and position size drift.

**Your trading is now fully transparent and controlled!** 📊
