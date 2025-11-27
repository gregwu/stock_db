#!/bin/bash
# Launch script for Streamlit SeekingAlpha Manager

PYTHON_EXE="/usr/bin/python"
SCRIPT_DIR="/home/greg/stock_db"
STREAMLIT_APP="$SCRIPT_DIR/rules.py"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "🚀 Starting Stragegy ..."
echo "📊 Access the app at: http://localhost:8505"
echo "🛑 Press Ctrl+C to stop the server"
echo
echo "🆕 New Features:"
echo "  • ✏️  Stragegy Rules"
echo

# Launch Streamlit app
nohup "$PYTHON_EXE" -m streamlit run "$STREAMLIT_APP" --server.port 8505 --server.address localhost --server.baseUrlPath qqq &

