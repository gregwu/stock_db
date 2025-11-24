#!/bin/bash
# Script to start the strategy monitor

echo "Starting Trading Strategy Monitor..."
echo "=================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env file with your Twilio credentials"
    exit 1
fi

# Check if twilio is installed
if ! python3 -c "import twilio" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip3 install -r requirements_monitor.txt
fi

# Check if monitor is already running
if pgrep -f "strategy_monitor.py" > /dev/null; then
    echo "⚠️  Monitor is already running!"
    echo ""
    echo "To stop it, run: ./stop_monitor.sh"
    echo "Or kill it manually: pkill -f strategy_monitor.py"
    exit 1
fi

# Start the monitor in background
echo "🚀 Starting monitor in background..."
nohup python3 strategy_monitor.py > monitor_output.log 2>&1 &

sleep 2

# Check if started successfully
if pgrep -f "strategy_monitor.py" > /dev/null; then
    PID=$(pgrep -f "strategy_monitor.py")
    echo "✅ Monitor started successfully (PID: $PID)"
    echo ""
    echo "📋 Useful commands:"
    echo "  View logs:        tail -f strategy_monitor.log"
    echo "  View output:      tail -f monitor_output.log"
    echo "  Check status:     ps aux | grep strategy_monitor"
    echo "  Stop monitor:     ./stop_monitor.sh"
    echo ""
else
    echo "❌ Failed to start monitor"
    echo "Check monitor_output.log for errors"
    exit 1
fi
