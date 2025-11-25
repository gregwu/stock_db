#!/bin/bash
# ----------------------------------------
# Check Strategy Trader Status
# ----------------------------------------

echo "=========================================="
echo "STRATEGY TRADER STATUS"
echo "=========================================="
echo ""

# Check if PID file exists
if [ ! -f .trader.pid ]; then
    echo "Status: ❌ NOT RUNNING"
    echo ""
    echo "Start with: ./start_trader.sh"
    exit 0
fi

PID=$(cat .trader.pid)

# Check if process is actually running
if ps -p $PID > /dev/null 2>&1; then
    echo "Status: ✅ RUNNING"
    echo "PID: $PID"
    echo ""

    # Show trading mode
    if grep -q "USE_PAPER = False" webull_config.py; then
        echo "Mode: ⚠️  LIVE TRADING"
    else
        echo "Mode: ✅ PAPER TRADING"
    fi
    echo ""

    # Show current position if state file exists
    if [ -f .strategy_trader_state.json ]; then
        echo "Current State:"
        python3 << 'EOF'
import json
try:
    with open('.strategy_trader_state.json', 'r') as f:
        state = json.load(f)

    if state.get('current_position'):
        print(f"  Position: {state['current_position']}")
        print(f"  Size: {state.get('position_size', 0)} shares")
        print(f"  Entry Price: ${state.get('entry_price', 0):.2f}")
        print(f"  Entry Time: {state.get('entry_time', 'N/A')}")
    else:
        print('  Position: NONE')

    if state.get('last_check_time'):
        print(f"  Last Check: {state.get('last_check_time')}")
except:
    print('  (unable to read state)')
EOF
        echo ""
    fi

    # Show last few log entries
    if [ -f strategy_trader.log ]; then
        echo "Recent Logs (last 5 lines):"
        echo "---"
        tail -5 strategy_trader.log | sed 's/^/  /'
        echo "---"
        echo ""
        echo "View full logs: tail -f strategy_trader.log"
    fi
else
    echo "Status: ❌ NOT RUNNING (stale PID file)"
    rm .trader.pid
    echo ""
    echo "Start with: ./start_trader.sh"
fi

echo "=========================================="
