#!/bin/bash
# ----------------------------------------
# Start the Strategy Trader (Automated Trading)
# ----------------------------------------

echo "=========================================="
echo "STRATEGY TRADER - AUTOMATED TRADING"
echo "=========================================="
echo ""

# Check if already running
if [ -f .trader.pid ]; then
    PID=$(cat .trader.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "❌ Trader is already running (PID: $PID)"
        echo "   Use ./stop_trader.sh to stop it first"
        exit 1
    else
        # Stale PID file, remove it
        rm .trader.pid
    fi
fi

# Check if Python script exists
if [ ! -f strategy_trader.py ]; then
    echo "❌ Error: strategy_trader.py not found"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "   Create .env with:"
    echo "     GMAIL_ADDRESS=your_email@gmail.com"
    echo "     GMAIL_APP_PASSWORD=your_app_password"
    echo "     WEBULL_EMAIL=your_webull_email@example.com"
    echo "     WEBULL_PASSWORD=your_webull_password"
    echo "     WEBULL_DEVICE_NAME=my_laptop"
    exit 1
fi

# Check if Webull credentials are configured in .env
if grep -q "your_webull_email@example.com" .env || grep -q "your_webull_password" .env; then
    echo "⚠️  WARNING: Webull credentials not configured in .env!"
    echo "   Please update WEBULL_EMAIL, WEBULL_PASSWORD, and WEBULL_DEVICE_NAME"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show configuration
echo "Configuration:"
echo "  - Script: strategy_trader.py"
echo "  - Log: strategy_trader.log"
echo "  - State: .strategy_trader_state.json"
echo ""

# Check trading mode
if grep -q "USE_PAPER = False" webull_config.py; then
    echo "⚠️⚠️⚠️  LIVE TRADING MODE  ⚠️⚠️⚠️"
    echo "This will place REAL orders with REAL money!"
    echo ""
    read -p "Are you ABSOLUTELY SURE? (yes/no) " -r
    if [[ ! $REPLY == "yes" ]]; then
        echo "Cancelled. Change USE_PAPER to True for paper trading."
        exit 1
    fi
else
    echo "✅ Paper Trading Mode (Safe)"
fi

echo ""
echo "Starting trader in background..."

# Start the trader with nohup
nohup python3 strategy_trader.py > /dev/null 2>&1 &
PID=$!

# Save PID
echo $PID > .trader.pid

echo "✅ Trader started successfully!"
echo "   PID: $PID"
echo ""
echo "Commands:"
echo "  ./stop_trader.sh       - Stop the trader"
echo "  ./trader_status.sh     - Check status"
echo "  tail -f strategy_trader.log - View live logs"
echo ""
echo "You will receive email alerts for all trades."
echo "=========================================="
