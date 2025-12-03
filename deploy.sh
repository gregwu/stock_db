#!/bin/bash

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Find and kill rules.py process
echo "Looking for rules.py process..."
PID=$(ps ax | grep rules.py | grep -v grep | awk '{print $1}')

if [ -n "$PID" ]; then
    echo "Killing rules.py process (PID: $PID)..."
    kill $PID
    sleep 4

    # Check if process is still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Process still running, trying SIGTERM..."
        kill -15 $PID
        sleep 2

        # Check again
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process still running, forcing kill with SIGKILL..."
            kill -9 $PID
            sleep 1
        fi
    fi

    echo "✅ Process killed successfully"
else
    echo "No rules.py process found."
fi

# Start rules.py
echo "Starting rules.py..."
./run_rules.sh
