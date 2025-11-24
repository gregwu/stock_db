#!/bin/bash
# Script to stop the strategy monitor

echo "Stopping Trading Strategy Monitor..."
echo "===================================="
echo ""

# Check if monitor is running
if ! pgrep -f "strategy_monitor.py" > /dev/null; then
    echo "ℹ️  Monitor is not running"
    exit 0
fi

# Get PID
PID=$(pgrep -f "strategy_monitor.py")
echo "Found monitor process (PID: $PID)"

# Stop the process
pkill -f strategy_monitor.py

sleep 2

# Verify it stopped
if pgrep -f "strategy_monitor.py" > /dev/null; then
    echo "⚠️  Process still running. Forcing kill..."
    pkill -9 -f strategy_monitor.py
    sleep 1
fi

# Final check
if ! pgrep -f "strategy_monitor.py" > /dev/null; then
    echo "✅ Monitor stopped successfully"
else
    echo "❌ Failed to stop monitor. Try manually: kill -9 $PID"
    exit 1
fi
