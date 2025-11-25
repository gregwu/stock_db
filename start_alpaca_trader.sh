#!/bin/bash
# Start Alpaca Trading Bot in background

echo "=========================================="
echo "Starting Alpaca Trading Bot..."
echo "=========================================="

# Check if already running
if [ -f .alpaca_trader.pid ]; then
    OLD_PID=$(cat .alpaca_trader.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Alpaca trader is already running (PID: $OLD_PID)"
        echo "Use ./stop_alpaca_trader.sh to stop it first"
        exit 1
    else
        echo "Removing stale PID file..."
        rm .alpaca_trader.pid
    fi
fi

# Check if config exists
if [ ! -f alpaca_config.py ]; then
    echo "❌ Error: alpaca_config.py not found"
    exit 1
fi

# Check mode
python3 << 'EOF'
from alpaca_config import USE_PAPER
if USE_PAPER:
    print("Mode: ✅ PAPER TRADING (Safe)")
else:
    print("Mode: ⚠️  LIVE TRADING (Real Money!)")
    print("")
    response = input("Are you sure you want to trade with real money? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled. Edit alpaca_config.py to change USE_PAPER = True")
        exit(1)
EOF

# Start the bot
nohup python3 alpaca_trader.py > alpaca_trader.out 2>&1 &
PID=$!
echo $PID > .alpaca_trader.pid

echo ""
echo "✅ Alpaca trader started successfully!"
echo "PID: $PID"
echo ""
echo "Monitor logs:"
echo "  tail -f alpaca_trader.log"
echo ""
echo "Check status:"
echo "  ./alpaca_trader_status.sh"
echo ""
echo "Stop trader:"
echo "  ./stop_alpaca_trader.sh"
echo "=========================================="
