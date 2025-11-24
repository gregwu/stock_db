#!/bin/bash
# ----------------------------------------
# Stop the Strategy Trader
# ----------------------------------------

echo "=========================================="
echo "STOPPING STRATEGY TRADER"
echo "=========================================="
echo ""

if [ ! -f .trader.pid ]; then
    echo "❌ Trader is not running (no PID file found)"
    exit 1
fi

PID=$(cat .trader.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Trader process not found (PID: $PID)"
    echo "   Removing stale PID file..."
    rm .trader.pid
    exit 1
fi

echo "Stopping trader (PID: $PID)..."
kill $PID

# Wait for process to stop
sleep 2

if ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Process still running, forcing stop..."
    kill -9 $PID
    sleep 1
fi

rm .trader.pid

echo "✅ Trader stopped successfully"
echo ""
echo "Final position saved in .strategy_trader_state.json"
echo "=========================================="
