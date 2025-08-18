
PYTHON_EXE="/usr/bin/python"
SCRIPT_DIR="/home/greg/stock_db"
STREAMLIT_APP="$SCRIPT_DIR/fractal.py"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "🚀 Starting Fractal ..."
echo "📊 Access the app at: http://localhost:8502"
echo "🛑 Press Ctrl+C to stop the server"
echo
echo "🆕 New Features:"
echo "  • 📊 Real-time change tracking"
echo

# Launch Streamlit app
nohup "$PYTHON_EXE" -m streamlit run "$STREAMLIT_APP" --server.port 8502 --server.address localhost --server.baseUrlPath fractal &
