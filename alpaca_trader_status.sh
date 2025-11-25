#!/bin/bash
# Check Alpaca Trading Bot Status

echo "=========================================="
echo "ALPACA TRADER STATUS"
echo "=========================================="

if [ -f .alpaca_trader.pid ]; then
    PID=$(cat .alpaca_trader.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: ✅ RUNNING (PID: $PID)"
        echo ""

        # Show mode
        python3 << 'EOF'
from alpaca_config import USE_PAPER
if USE_PAPER:
    print("Mode: ✅ PAPER TRADING")
else:
    print("Mode: ⚠️  LIVE TRADING")
EOF
        echo ""

        # Show current position if state file exists
        if [ -f .alpaca_trader_state.json ]; then
            echo "Current State:"
            python3 << 'EOF'
import json
try:
    with open('.alpaca_trader_state.json', 'r') as f:
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
        if [ -f alpaca_trader.log ]; then
            echo "Recent Logs (last 5 lines):"
            echo "---"
            tail -5 alpaca_trader.log | sed 's/^/  /'
            echo "---"
        fi

        echo ""
        echo "Full logs: tail -f alpaca_trader.log"
    fi
else
    echo "Status: ❌ NOT RUNNING (stale PID file)"
    rm .alpaca_trader.pid
    echo ""
    echo "Start with: ./start_alpaca_trader.sh"
fi

echo "=========================================="
