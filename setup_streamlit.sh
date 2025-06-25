#!/bin/bash
"""
Setup and run the Stock Prediction Streamlit App
"""

echo "🚀 Setting up Stock Prediction App..."

# Install required packages
echo "📦 Installing required packages..."
pip install -r requirements_streamlit.txt

echo "✅ Setup complete!"
echo ""
echo "🌟 To run the app, use:"
echo "streamlit run stock_prediction_app.py"
echo ""
echo "📖 Features:"
echo "- Interactive stock charts with technical indicators"
echo "- Historical data visualization"
echo "- Trend prediction (1, 3, 5 days)"
echo "- Prediction scoring and accuracy tracking"
echo "- Future data overlay (when available)"
echo ""
