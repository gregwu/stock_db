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
    sleep 1
else
    echo "No rules.py process found."
fi

# Start rules.py
echo "Starting rules.py..."
./run_rules.sh
