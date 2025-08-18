#!/bin/bash
# Launch script for Streamlit SeekingAlpha Manager

PYTHON_EXE="/usr/bin/python"
SCRIPT_DIR="/home/greg/stock_db"
STREAMLIT_APP="$SCRIPT_DIR/watchlist.py"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "🚀 Starting Watchlist ..."
echo "📊 Access the app at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo
echo "🆕 New Features:"
echo "  • ✏️ Inline table editing - click on any field to edit"
echo "  • 🗑️ Delete buttons on each row"
echo "  • 💾 Batch save/cancel changes"
echo "  • 🔧 Bulk operations support"
echo "  • 📊 Real-time change tracking"
echo

# Launch Streamlit app
nohup "$PYTHON_EXE" -m streamlit run "$STREAMLIT_APP" --server.port 8501 --server.address localhost --server.baseUrlPath seekingalpha &

