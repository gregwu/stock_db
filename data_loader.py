import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from feature_engineering import add_technical_indicators

def clean_volume(v):
    try:
        return int(float(v))
    except:
        return 0
    
def load_stock_yf(symbol, start = 40, end = 1):
    try:
        today = datetime.today()
        start_date = (today - timedelta(days=start)).strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=end)).strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            print(f"No data found on Yahoo Finance for {symbol}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df = df.rename(columns={"Date": "DATE", "Open": "OPEN", "High": "HIGH", "Low": "LOW", "Close": "CLOSE", "Volume": "VOL"})
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df.sort_values("DATE")
            df["VOL"] = df["VOL"].apply(clean_volume)
            df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE", "VOL"])
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]].copy()
            
            # Add technical indicators to match the training data
            df = add_technical_indicators(df)
            df = df.dropna()  # Remove NaN values created by technical indicators
            
        return df
    except Exception as e:
        print(f"Failed to fetch {symbol} from Yahoo Finance: {e}")
        return None

def load_stock_file(filepath):

    # Check if file is empty
    if filepath.stat().st_size == 0:
        raise ValueError(f"File is empty")
    
    df = pd.read_csv(filepath, skiprows=1, header=None)
    
    # Check if we got any data after reading
    if len(df) == 0:
        raise ValueError(f"File contains no data rows")
    
    df.columns = ["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT"]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    df = df.sort_values("DATE")

    df["VOL"] = df["VOL"].apply(clean_volume)
    df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE", "VOL"])
    
    # Return only numeric features for model training
    numeric_df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]].copy()
    
    return numeric_df

def load_csv_file(filepath, dropna=True):
    # Load and clean
    if filepath.stat().st_size == 0:
        raise ValueError("File is empty")

    df = pd.read_csv(filepath, skiprows=1, header=None)
    if len(df) == 0:
        raise ValueError("File contains no data rows")

    df.columns = ["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT"]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    df = df.sort_values("DATE")

    df["VOL"] = df["VOL"].apply(clean_volume)
    df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE", "VOL"])

    # 🧠 Memory optimization
    df = df.astype({'OPEN': 'float32', 'HIGH': 'float32', 'LOW': 'float32',
                    'CLOSE': 'float32', 'VOL': 'float32'})

    # 📈 Add technical indicators
    df = add_technical_indicators(df)

    # 📉 Drop NaNs caused by rolling windows
    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df

