#!/bin/bash
# Stop Alpaca Trading Bot

echo "=========================================="
echo "Stopping Alpaca Trading Bot..."
echo "=========================================="

if [ ! -f .alpaca_trader.pid ]; then
    echo "❌ No PID file found. Alpaca trader may not be running."
    exit 1
fi

PID=$(cat .alpaca_trader.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping process $PID..."
    kill $PID

    # Wait for process to stop
    sleep 2

    if ps -p $PID > /dev/null 2>&1; then
        echo "Process still running, force killing..."
        kill -9 $PID
    fi

    rm .alpaca_trader.pid
    echo "✅ Alpaca trader stopped successfully"
else
    echo "❌ Process $PID not found (may have already stopped)"
    rm .alpaca_trader.pid
fi

echo "=========================================="
