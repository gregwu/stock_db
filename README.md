# 📈 Stock Trend Prediction App

A robust Streamlit application for stock analysis and trend prediction with PostgreSQL backend integration and automatic Yahoo Finance data fetching.

## 🌟 Features

### 📊 **Data Management**
- **PostgreSQL Integration**: Stores historical stock data with technical indicators
- **Automatic Yahoo Finance Sync**: Fetches missing historical data automatically
- **Technical Indicators**: SMA 20, SMA 144, RSI, MACD, Bollinger Bands, volume analysis
- **Smart Data Buffering**: Ensures sufficient data (200+ days) for accurate SMA 144 calculations

### 📈 **Interactive Charts**
- **Multi-panel Charts**: Price/Moving Averages, Volume, RSI, MACD
- **Technical Overlays**: Configurable display of all technical indicators
- **Future Data Visualization**: Optional overlay of future price data
- **Prediction Markers**: Visual markers for your trend predictions (toggleable)

### 🔮 **Prediction System**
- **5-Day Trend Predictions**: Predict UP/DOWN movements for 1-5 days ahead
- **Progressive Data Reveal**: "Show Next Day" button to reveal future data incrementally
- **Accurate Scoring**: Compare predictions against actual price movements
- **Prediction Tracking**: Session-based storage of all predictions and scores

### 🎯 **User Experience**
- **Auto-Dismissing Messages**: Toast notifications that disappear after 3 seconds
- **Real-time Updates**: Live chart updates as you reveal future data
- **Clean Interface**: Modern UI with intuitive controls and clear feedback
- **Symbol Display**: Clean ticker symbols (removes .US suffix)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Required packages (see requirements)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd stock_db
```

2. **Set up virtual environment**
```bash
python -m venv stock_db
source stock_db/bin/activate  # On macOS/Linux
# or
stock_db\Scripts\activate     # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements_streamlit.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

5. **Run the application**
```bash
streamlit run stock_prediction_app.py
```

## 📁 Project Structure

```
stock_db/
├── stock_prediction_app.py      # Main Streamlit application
├── stock_prediction_db.py       # Alternative version
├── sync_yahoo.py               # Yahoo Finance data sync script
├── feature_engineering.py      # Technical indicators calculation
├── populate_indicators.py      # Batch indicator population
├── update_single_ticker.py     # Single ticker update utility
├── savedb.py                   # Database setup utility
├── simple_prediction_app.py    # CLI version for non-GUI environments
├── requirements_streamlit.txt   # Python dependencies
├── setup_streamlit.sh          # Setup script
├── .env                        # Environment variables (create this)
└── README.md                   # This file
```

## 🔧 Configuration

### Database Setup
1. Create PostgreSQL database
2. Run database setup script:
```bash
python savedb.py
```

3. Sync initial stock data:
```bash
python sync_yahoo.py
```

4. Populate technical indicators:
```bash
python populate_indicators.py
```

### Adding New Tickers
Add tickers to the `tickers` list in `sync_yahoo.py` and run:
```bash
python sync_yahoo.py
```

## 📚 Usage Guide

### 1. **Select Stock and Date**
- Choose a ticker from the sidebar dropdown
- Set a base date for your analysis
- The app shows 50 days of price history before the base date

### 2. **Analyze Charts**
- View candlestick charts with technical indicators
- Toggle prediction markers on/off in the sidebar
- Examine volume, RSI, and MACD in separate panels

### 3. **Make Predictions**
- Use the prediction form to forecast 1-5 days ahead
- Submit predictions for UP/DOWN trend movements
- Predictions are stored in session memory

### 4. **Progressive Data Reveal**
- Click "Show Next Day" to reveal future price data incrementally
- Watch how your predictions compare to actual price movements
- View updated charts and statistics in real-time

### 5. **Score Your Predictions**
- Automatic scoring compares predicted vs. actual price directions
- View detailed breakdown of each prediction's accuracy
- Track overall prediction accuracy percentage

## 🛠️ Technical Details

### Data Flow
1. **Data Check**: App verifies sufficient historical data exists
2. **Auto-Fetch**: If needed, fetches additional data from Yahoo Finance
3. **Indicator Calculation**: Automatically calculates technical indicators
4. **Display**: Shows interactive charts with all data and indicators

### Technical Indicators
- **SMA 20**: 20-day Simple Moving Average
- **SMA 144**: 144-day Simple Moving Average (requires 200+ days of data)
- **RSI 14**: 14-day Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper and lower bands with touch detection
- **Volume Analysis**: Volume spikes and patterns

### Prediction Scoring Logic
- **Day 1**: Compares to base date price
- **Day 2-5**: Compares to previous day's price
- **Direction**: UP if price increases, DOWN if price decreases
- **Accuracy**: Percentage of correct predictions

## 🔍 Troubleshooting

### Common Issues

**Database Connection Errors**
- Verify PostgreSQL is running
- Check connection parameters in `.env` file
- Ensure database exists and user has proper permissions

**Missing Data**
- App automatically fetches missing data from Yahoo Finance
- Check internet connection for Yahoo Finance API access
- Verify ticker symbols are valid

**Technical Indicator Issues**
- Ensure sufficient historical data (200+ days for SMA 144)
- Run `python populate_indicators.py` to recalculate indicators
- Check for data gaps in the database

**Performance Issues**
- Large date ranges may slow chart rendering
- Consider reducing the number of days displayed
- Check database query performance

## 📦 Dependencies

### Core Requirements
```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
yfinance>=0.2.0
```

### Additional Tools
- **PostgreSQL**: Database backend
- **Yahoo Finance API**: Historical stock data source

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free stock data API
- **Streamlit**: For the excellent web app framework
- **Plotly**: For interactive charting capabilities
- **PostgreSQL**: For robust data storage

## 📞 Support

For issues, questions, or contributions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include error messages, screenshots, and steps to reproduce

---

**Built with ❤️ using Streamlit, PostgreSQL, and Yahoo Finance API**
