#!/usr/bin/env python3
"""
QQQ 5-Minute Data Downloader from Polygon.io

This script downloads all available QQQ 5-minute data from Polygon.io
and saves it to a CSV file. It handles rate limits, pagination, and data
validation automatically.

Features:
- Downloads 5-minute OHLCV data (optimal balance of detail and efficiency)
- Handles Polygon.io rate limits (5 calls per minute for free tier)
- Saves data in the same format as qqq.us.txt
- Progress tracking and error handling
- Automatic date range detection

Usage:
    python download_qqq_polygon.py
    
Requirements:
    pip install requests python-dotenv pandas
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict
# import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qqq_download.log'),
        logging.StreamHandler()
    ]
)

class PolygonQQQDownloader:
    """Download QQQ 5-minute data from Polygon.io API"""
    
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables")
        
        self.base_url = "https://api.polygon.io"
        self.symbol = "QQQ"
        self.timespan = "minute"
        self.multiplier = 5  # 5-minute bars for optimal balance
        
        # Rate limiting (free tier: 5 calls per minute)
        self.rate_limit_calls = 5
        self.rate_limit_period = 60  # seconds
        self.last_calls = []
        
        logging.info(f"Initialized Polygon downloader for {self.symbol} ({self.multiplier}-minute data)")
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to stay within API limits"""
        now = time.time()
        
        # Remove calls older than rate_limit_period
        self.last_calls = [call_time for call_time in self.last_calls 
                          if now - call_time < self.rate_limit_period]
        
        # If we've made too many calls recently, wait
        if len(self.last_calls) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (now - self.last_calls[0]) + 1
            if sleep_time > 0:
                logging.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        # Record this call
        self.last_calls.append(now)
    
    def get_aggregates(self, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        Get minute aggregates for a date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of minute bars or None if error
        """
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/v2/aggs/ticker/{self.symbol}/range/{self.multiplier}/{self.timespan}/{start_date}/{end_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,  # Max limit
            'apikey': self.api_key
        }
        
        try:
            logging.info(f"Requesting data for {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    results = data['results']
                    logging.info(f"Retrieved {len(results)} minute bars")
                    return results
                else:
                    logging.warning(f"No data for {start_date} to {end_date}: {data.get('message', 'Unknown error')}")
                    return []
            
            elif response.status_code == 429:
                logging.warning("Rate limit exceeded, waiting...")
                time.sleep(60)
                return self.get_aggregates(start_date, end_date)  # Retry
            
            else:
                logging.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return None
    
    def convert_to_qqq_format(self, polygon_data: List[Dict]) -> pd.DataFrame:
        """
        Convert Polygon.io data to the same format as qqq.us.txt
        
        Polygon format: {'v': volume, 'vw': vwap, 'o': open, 'c': close, 'h': high, 'l': low, 't': timestamp}
        QQQ format: <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
        """
        if not polygon_data:
            return pd.DataFrame()
        
        converted_data = []
        
        for bar in polygon_data:
            # Convert timestamp (milliseconds) to datetime
            dt = datetime.fromtimestamp(bar['t'] / 1000)
            
            # Format date and time
            date_str = dt.strftime('%Y%m%d')
            time_str = dt.strftime('%H%M%S')
            
            # Convert to QQQ format
            converted_data.append({
                'TICKER': f'{self.symbol}.US',
                'PER': '5',  # 5 minute bars
                'DATE': date_str,
                'TIME': time_str,
                'OPEN': bar['o'],
                'HIGH': bar['h'],
                'LOW': bar['l'],
                'CLOSE': bar['c'],
                'VOL': bar['v'],
                'OPENINT': '0'
            })
        
        df = pd.DataFrame(converted_data)
        return df
    
    def download_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data for a specific date range"""
        all_data = []
        
        # Split large date ranges into smaller chunks (7 days each)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_dt = start_dt
        
        while current_dt <= end_dt:
            chunk_end_dt = min(current_dt + timedelta(days=6), end_dt)
            
            chunk_start = current_dt.strftime('%Y-%m-%d')
            chunk_end = chunk_end_dt.strftime('%Y-%m-%d')
            
            chunk_data = self.get_aggregates(chunk_start, chunk_end)
            
            if chunk_data is not None:
                all_data.extend(chunk_data)
            else:
                logging.error(f"Failed to download data for {chunk_start} to {chunk_end}")
            
            current_dt = chunk_end_dt + timedelta(days=1)
            
            # Small delay between chunks
            time.sleep(1)
        
        logging.info(f"Total bars collected: {len(all_data)}")
        return self.convert_to_qqq_format(all_data)
    
    def download_all_available(self, years_back: int = 2) -> pd.DataFrame:
        """
        Download all available minute data
        
        Args:
            years_back: How many years back to download (default: 2 years)
            
        Returns:
            DataFrame with all minute data
        """
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')
        
        logging.info(f"Downloading {self.symbol} {self.multiplier}-minute data from {start_date} to {end_date}")
        
        return self.download_date_range(start_date, end_date)
    
    def save_to_file(self, df: pd.DataFrame, filename: str = 'qqq_polygon_data.txt'):
        """Save DataFrame to file in QQQ format"""
        if df.empty:
            logging.warning("No data to save")
            return
        
        # Sort by date and time
        df['datetime_sort'] = pd.to_datetime(df['DATE'] + df['TIME'], format='%Y%m%d%H%M%S')
        df = df.sort_values('datetime_sort').drop(columns=['datetime_sort'])
        
        # Write header and data
        with open(filename, 'w') as f:
            f.write('<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n')
            
            for _, row in df.iterrows():
                f.write(f"{row['TICKER']},{row['PER']},{row['DATE']},{row['TIME']},{row['OPEN']},{row['HIGH']},{row['LOW']},{row['CLOSE']},{row['VOL']},{row['OPENINT']}\n")
        
        logging.info(f"Saved {len(df)} {self.multiplier}-minute bars to {filename}")
        
        # Print summary statistics
        start_date = df['DATE'].min()
        end_date = df['DATE'].max()
        logging.info(f"Data range: {start_date} to {end_date}")
        logging.info(f"Total trading days: {df['DATE'].nunique()}")
        logging.info(f"Average bars per day: {len(df) / df['DATE'].nunique():.1f}")

def main():
    """Main execution function"""
    try:
        downloader = PolygonQQQDownloader()
        
        # Download all available data (2 years back)
        df = downloader.download_all_available(years_back=5)
        
        if not df.empty:
            # Save to file
            filename = f'qqq_polygon_5min_{datetime.now().strftime("%Y%m%d")}.txt'
            downloader.save_to_file(df, filename)
            
            print(f"\n✅ Download completed successfully!")
            print(f"📁 File saved: {filename}")
            print(f"📊 Total bars: {len(df):,}")
            print(f"📅 Date range: {df['DATE'].min()} to {df['DATE'].max()}")
            
        else:
            print("❌ No data downloaded")
            
    except Exception as e:
        logging.error(f"Download failed: {e}")
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()