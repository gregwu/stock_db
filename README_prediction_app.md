# 📈 Stock Prediction App

A comprehensive stock analysis and trend prediction application that connects to your PostgreSQL stock database.

## 🌟 Features

### 📊 Streamlit Web App (`stock_prediction_app.py`)
- **Interactive Charts**: Candlestick charts with technical indicators
- **Real-time Analysis**: SMA, RSI, MACD, Bollinger Bands
- **Trend Prediction**: Predict 1, 3, and 5-day trends (UP/DOWN)
- **Prediction Scoring**: Track your prediction accuracy over time
- **Future Data Overlay**: Shows next 30 days when available

### 📈 Simple Command Line App (`simple_prediction_app.py`)
- **Lightweight**: Uses matplotlib for basic charts
- **No Web Dependencies**: Runs in terminal
- **Same Predictions**: Make and save trend predictions
- **JSON Storage**: Saves predictions to local file

## 🚀 Quick Start

### Option 1: Full Streamlit Web App

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run stock_prediction_app.py
   ```

3. **Open Browser**: The app will open at `http://localhost:8501`

### Option 2: Simple Command Line App

1. **Install Basic Dependencies**:
   ```bash
   pip install matplotlib pandas sqlalchemy psycopg2-binary python-dotenv
   ```

2. **Run the App**:
   ```bash
   python simple_prediction_app.py
   ```

## 📋 Requirements

### Database Requirements
- PostgreSQL database with `stock_data` table
- `.env` file with database configuration:
  ```
  DB_HOST=localhost
  DB_PORT=5433
  DB_NAME=gangwu
  DB_USER=gangwu
  DB_PASSWORD=gangwu
  ```

### Python Dependencies

#### Full App (Streamlit):
```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
```

#### Simple App (Command Line):
```
matplotlib>=3.5.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
```

## 🎯 How to Use

### Making Predictions

1. **Select Ticker**: Choose from available tickers in your database
2. **Set Date Range**: Pick start and end dates for historical analysis
3. **Analyze Chart**: Review price action, volume, and technical indicators
4. **Make Predictions**: Predict if price will be UP or DOWN after 1, 3, and 5 days
5. **Submit**: Your predictions are saved and can be scored later

### Prediction Scoring

The app automatically scores your predictions when future data becomes available:

- **Accuracy Tracking**: Shows percentage of correct predictions
- **Visual Feedback**: ✅ for correct predictions, ❌ for incorrect
- **Price Analysis**: Shows actual vs predicted price movements
- **Historical Performance**: Track your prediction accuracy over time

### Chart Features

#### Price Chart:
- **Candlestick Chart**: OHLC data visualization
- **Moving Averages**: SMA 20 and SMA 144
- **Bollinger Bands**: Upper and lower bands
- **Volume Overlay**: Trading volume with color coding

#### Technical Indicators:
- **RSI (14)**: Relative Strength Index with overbought/oversold levels
- **MACD**: Moving Average Convergence Divergence with signal line
- **Volume Analysis**: Volume spikes and patterns

## 📁 File Structure

```
stock_db/
├── stock_prediction_app.py      # Main Streamlit web app
├── simple_prediction_app.py     # Command line version
├── requirements_streamlit.txt    # Dependencies for web app
├── setup_streamlit.sh           # Setup script
├── .env                         # Database configuration
└── predictions.json             # Saved predictions (auto-created)
```

## 🔧 Troubleshooting

### Database Connection Issues
1. Check your `.env` file configuration
2. Ensure PostgreSQL is running
3. Verify database credentials
4. Test connection with: `python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://user:pass@host:port/db'); print('Connected!' if engine.connect() else 'Failed')"`

### Missing Dependencies
- **Streamlit not found**: `pip install streamlit`
- **Plotly not found**: `pip install plotly`
- **Matplotlib not found**: `pip install matplotlib`

### No Data Found
1. Check if ticker exists in database: `SELECT DISTINCT ticker FROM stock_data LIMIT 10;`
2. Verify date range has data
3. Ensure technical indicators are populated (run `populate_indicators.py`)

## 💡 Tips for Better Predictions

1. **Study the Trends**: Look at moving averages and trend direction
2. **Check RSI**: Values above 70 may indicate overbought, below 30 oversold
3. **Volume Confirmation**: High volume often confirms price movements
4. **MACD Signals**: MACD crossing above signal line can indicate uptrend
5. **Bollinger Bands**: Prices touching bands may indicate reversal points

## 🎯 Example Workflow

1. **Launch App**: `streamlit run stock_prediction_app.py`
2. **Select AAPL.US** from dropdown
3. **Set Range**: Last 90 days
4. **Analyze**: Look at recent trend, RSI around 45, MACD turning positive
5. **Predict**: 1-day UP, 3-day UP, 5-day DOWN (expecting short-term rally then pullback)
6. **Submit**: Click "Submit Prediction"
7. **Track**: Check back in 5 days to see your score!

## 🚀 Future Enhancements

- **Machine Learning**: Auto-predictions based on technical indicators
- **Portfolio Tracking**: Track multiple tickers and overall performance
- **Advanced Charts**: More technical indicators and chart types
- **Export Features**: Export predictions and results to CSV
- **Social Features**: Compare predictions with other users

---

**Happy Trading! 📈🎯**

*Remember: This is for educational purposes. Always do your own research before making investment decisions.*
