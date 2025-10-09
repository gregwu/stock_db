#!/usr/bin/env python3
"""
scanner.py - Stock Scanner and Prediction Tool

This script provides stock scanning and prediction capabilities using technical indicators.
It can be run standalone or called from other scripts like watchlist.py.

Features:
- Technical indicator calculation using shared technical_indicators.py
- Stock screening based on technical criteria
- Prediction mode for generating buy/sell signals
- Yahoo Finance integration for real-time stock data
- Database integration for ticker lists (seekingalpha table)
- Configurable ticker lists and scanning limits

Usage:
    python scanner.py [options]
    python scanner.py --max-tickers 10                    # Use database tickers -> scanner_reports/
    python scanner.py --tickers AAPL GOOGL MSFT          # Specific tickers -> scanner_reports/
    python scanner.py --ticker-file my_tickers.txt        # From file -> scanner_reports/
    python scanner.py --output-dir custom_dir --max-tickers 5  # Custom output directory
    
    # Filter examples:
    python scanner.py --filter-tickers AAPL MSFT         # Only show results for specific tickers
    python scanner.py --min-rr-ratio 1.5                 # Only show R/R ratio >= 1.5
    python scanner.py --max-atr-price-ratio 0.03         # Only show ATR/Price <= 3%
    python scanner.py --min-bb-position 0.2 --max-bb-position 0.8  # BB position between 20%-80%
    python scanner.py --min-vri-20 1.2 --max-vri-20 3.0  # VRI_20 between 1.2-3.0
    python scanner.py --min-cmf-20 -0.1 --max-cmf-20 0.3 # CMF_20 between -0.1 to +0.3
    python scanner.py --min-ad-momentum -5 --max-ad-momentum 10  # A/D momentum between -5% to +10%
    python scanner.py --min-rsi 30 --max-rsi 70          # RSI between 30-70 (neutral zone)
    
    # ML model examples:
    python scanner.py --use-ml                           # Use ML predictions
    python scanner.py --rebuild-models                   # Force rebuild all ML models
    python scanner.py --use-ml --rebuild-models          # Use ML with fresh models
"""

import argparse
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import joblib

# Try to import LightGBM for ML predictions (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Import our shared technical indicators and util functions
from technical_indicators import TechnicalIndicators, calculate_comprehensive_indicators
import sys
sys.path.append('..')
# Removed: from util import calculate_all_technical_indicators - using calculate_comprehensive_indicators instead

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Streamlit configuration
STREAMLIT_URL = os.getenv('STREAMLIT_URL', '/ypredict/')

# Fallback ticker list if database is unavailable
FALLBACK_TICKERS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA', 'V',
    'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE',
    'CRM', 'PYPL', 'INTC', 'CMCSA', 'PFE', 'T', 'VZ', 'KO', 'PEP', 'ABT',
    'COST', 'TMO', 'LLY', 'ABBV', 'ACN', 'DHR', 'TXN', 'MCD', 'QCOM', 'HON',
    'IBM', 'CVX', 'XOM', 'CAT', 'GE', 'MMM', 'BA', 'CSCO', 'AMD', 'COP'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_option_data(current_price: float, ticker: str = None) -> Dict[str, any]:
    """
    Calculate option-related data including OTM percentage and DTE
    
    Args:
        current_price: Current stock price
        ticker: Stock ticker (optional, for future option chain integration)
    
    Returns:
        Dictionary with option price and expiration data
    """
    try:
        # For now, we'll generate synthetic option data based on common patterns
        # In the future, this could be enhanced with real option chain data from APIs
        
        # Calculate suggested option strike prices (typically OTM)
        # For calls: strikes above current price
        # For puts: strikes below current price
        
        # Vary OTM percentage based on stock price and volatility
        # Higher priced stocks and higher volatility tend to use higher OTM percentages
        if current_price < 50:
            # Lower priced stocks: 5-15% OTM
            target_otm_percentage = 0.05 + (current_price / 50) * 0.10  # 5-15%
        elif current_price < 100:
            # Mid priced stocks: 8-12% OTM  
            target_otm_percentage = 0.08 + (current_price - 50) / 50 * 0.04  # 8-12%
        elif current_price < 200:
            # Higher priced stocks: 10-20% OTM
            target_otm_percentage = 0.10 + (current_price - 100) / 100 * 0.10  # 10-20%
        else:
            # Very high priced stocks: 15-25% OTM
            target_otm_percentage = 0.15 + min((current_price - 200) / 200 * 0.10, 0.10)  # 15-25%
        
        # Add some randomness based on ticker hash for variety (but deterministic)
        if ticker:
            # Use ticker hash to add ±2% variation
            hash_val = hash(ticker) % 100
            variation = (hash_val / 100 - 0.5) * 0.04  # ±2%
            target_otm_percentage += variation
            # Keep within reasonable bounds
            target_otm_percentage = max(0.05, min(0.30, target_otm_percentage))
        
        # Calculate call strike (target % OTM)
        call_strike = current_price * (1 + target_otm_percentage)
        
        # Calculate actual OTM percentage based on strike vs current price
        # For calls: OTM% = (strike - current_price) / current_price * 100
        actual_otm_percentage = ((call_strike - current_price) / current_price) * 100
        
        # Estimate option price using simplified Black-Scholes approximation
        # This is a rough estimate - in production, use real option data
        days_to_expiration = 30  # Default to monthly options
        volatility = 0.25  # Assume 25% implied volatility
        risk_free_rate = 0.05  # 5% risk-free rate
        
        # Simplified option price estimation (very rough)
        time_value = current_price * volatility * (days_to_expiration / 365) ** 0.5
        intrinsic_value = max(0, current_price - call_strike)  # For ITM calls (should be 0 for OTM)
        estimated_option_price = intrinsic_value + time_value * 0.3  # Rough estimate
        
        # Calculate expiration date (assume next monthly expiration)
        from datetime import datetime, timedelta
        import calendar
        
        today = datetime.now().date()
        # Find third Friday of next month (standard monthly expiration)
        if today.month == 12:
            next_month = 1
            next_year = today.year + 1
        else:
            next_month = today.month + 1
            next_year = today.year
        
        # Find third Friday
        first_day = datetime(next_year, next_month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)  # Add two weeks
        
        # Calculate DTE
        dte = (third_friday.date() - today).days
        
        return {
            'option_price': round(estimated_option_price, 2),
            'otm_percentage': round(actual_otm_percentage, 1),  # Actual calculated OTM %
            'strike_price': round(call_strike, 2),
            'expiration_date': third_friday.date(),
            'dte': dte,
            'option_type': 'CALL',
            'implied_volatility': round(volatility * 100, 1)  # Convert to percentage
        }
        
    except Exception as e:
        logger.warning(f"Error calculating option data: {e}")
        # Return default/fallback values
        from datetime import datetime, timedelta
        fallback_strike = current_price * 1.1 if current_price > 0 else 110.0
        fallback_current = current_price if current_price > 0 else 100.0
        fallback_otm = ((fallback_strike - fallback_current) / fallback_current) * 100
        
        return {
            'option_price': 0.0,
            'otm_percentage': round(fallback_otm, 1),  # Correctly calculated OTM %
            'strike_price': fallback_strike,
            'expiration_date': (datetime.now() + timedelta(days=30)).date(),
            'dte': 30,
            'option_type': 'CALL',
            'implied_volatility': 25.0
        }


def get_existing_option_data(ticker: str) -> Optional[Dict[str, any]]:
    """
    Get existing option data (strike price and expiration) from seekingalpha table
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with existing option data or None if not found
    """
    try:
        # Create database connection
        conn_string = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        engine = create_engine(conn_string)
        
        # Query for most recent record for this ticker
        query = text("""
        SELECT price, exp 
        FROM seekingalpha 
        WHERE ticker = :ticker 
        AND price IS NOT NULL 
        AND exp IS NOT NULL
        ORDER BY date_added DESC 
        LIMIT 1
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {'ticker': ticker}).fetchone()
            
        if result:
            strike_price, expiration_date = result
            
            # Calculate OTM percentage and DTE
            from datetime import datetime, date
            today = datetime.now().date()
            
            if isinstance(expiration_date, str):
                exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            else:
                exp_date = expiration_date
                
            dte = (exp_date - today).days if exp_date > today else 0
            
            return {
                'strike_price': float(strike_price),
                'expiration_date': exp_date,
                'dte': dte,
                'source': 'database'
            }
            
    except Exception as e:
        logger.warning(f"Could not get existing option data for {ticker}: {e}")
        
    return None


def load_tickers_from_database() -> List[str]:
    """Load ticker list from seekingalpha table"""
    try:
        # Create database connection
        conn_string = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        engine = create_engine(conn_string)
        
        # Query to get unique tickers from seekingalpha table
        query = text("""
        SELECT DISTINCT ticker 
        FROM seekingalpha 
        WHERE ticker IS NOT NULL 
        ORDER BY ticker
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            tickers = [row[0] for row in result.fetchall()]
        
        if tickers:
            logger.info(f"Loaded {len(tickers)} tickers from seekingalpha database table")
            return tickers
        else:
            logger.warning("No tickers found in seekingalpha table, using fallback list")
            return FALLBACK_TICKERS
            
    except Exception as e:
        logger.error(f"Error loading tickers from database: {e}")
        logger.info("Using fallback ticker list")
        return FALLBACK_TICKERS


class StockScanner:
    """Stock scanner for technical analysis and predictions"""
    
    def __init__(self, ticker_list: List[str] = None, use_database: bool = True, use_ml_predictions: bool = True, rebuild_models: bool = False):
        """Initialize scanner with ticker list and prediction method"""
        if ticker_list:
            self.ticker_list = ticker_list
            logger.info(f"Initialized scanner with custom ticker list: {len(self.ticker_list)} tickers")
        elif use_database:
            self.ticker_list = load_tickers_from_database()
        else:
            self.ticker_list = FALLBACK_TICKERS
            logger.info(f"Initialized scanner with fallback ticker list: {len(self.ticker_list)} tickers")
        
        # Set prediction method
        self.use_ml_predictions = use_ml_predictions and LIGHTGBM_AVAILABLE
        if use_ml_predictions and not LIGHTGBM_AVAILABLE:
            logger.warning("ML predictions requested but LightGBM not available. Using confidence-based predictions.")
        
        # Set model rebuild option
        self.rebuild_models = rebuild_models
        if rebuild_models:
            logger.info("Model rebuild mode enabled - will retrain models even if existing ones are found")
        
        prediction_type = "ML-based" if self.use_ml_predictions else "confidence-based"
        logger.info(f"Scanner using {prediction_type} predictions")
    
    def get_tickers(self, date_filter: str = "latest", max_tickers: Optional[int] = None) -> List[str]:
        """
        Get list of tickers for scanning
        
        Args:
            date_filter: Filter parameter (maintained for compatibility but not used with Yahoo data)
            max_tickers: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        logger.info(f"Getting tickers with max: {max_tickers}")
        
        # Use the configured ticker list
        tickers = self.ticker_list.copy()
        
        # Apply max_tickers limit if specified
        if max_tickers and max_tickers < len(tickers):
            tickers = tickers[:max_tickers]
        
        logger.info(f"Selected {len(tickers)} tickers for scanning")
        return tickers
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get stock data for a specific ticker from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            period: Period of data to retrieve (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with stock data and technical indicators or None if no data found
        """
        try:
            # Download data from Yahoo Finance
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if data.empty:
                logger.warning(f"No data found for ticker: {ticker}")
                return None
            
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Reset index and standardize column names
            data.reset_index(inplace=True)
            data.columns = [col.upper() for col in data.columns]
            data = data.rename(columns={'VOLUME': 'VOL'})
            
            # Calculate all technical indicators including derived features
            df = calculate_comprehensive_indicators(data)
            
            # Convert columns to lowercase for compatibility
            df.columns = [col.lower() for col in df.columns]
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Ensure we have the basic OHLCV columns in both cases
            # Use pd.concat to avoid DataFrame fragmentation warnings
            ohlcv_columns = pd.DataFrame({
                'OPEN': df['open'],
                'HIGH': df['high'],
                'LOW': df['low'],
                'CLOSE': df['close'],
                'VOL': df['vol']
            })
            df = pd.concat([df, ohlcv_columns], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None
    
    def calculate_technical_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical analysis scores for a stock
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            Dictionary with various technical scores including risk/reward metrics
        """
        if df is None or len(df) < 20:
            return {}
        
        # Calculate additional indicators if needed
        # Use pd.concat to avoid DataFrame fragmentation warnings
        additional_indicators = {}
        
        if 'RSI_14' not in df.columns:
            additional_indicators['RSI_14'] = TechnicalIndicators.rsi(df['CLOSE'], 14)
        
        if 'BB_Middle' not in df.columns:
            bb_middle, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(df['CLOSE'])
            additional_indicators['BB_Middle'] = bb_middle
            additional_indicators['BB_Upper'] = bb_upper
            additional_indicators['BB_Lower'] = bb_lower
        
        # Add all additional indicators at once to avoid fragmentation
        if additional_indicators:
            indicators_df = pd.DataFrame(additional_indicators, index=df.index)
            df = pd.concat([df, indicators_df], axis=1)
        
        # Get the latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Get current price and recent high/low for risk/reward calculation
        current_price = latest['close']
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        scores = {
            'current_price': current_price,
            'recent_high': recent_high,
            'recent_low': recent_low
        }
        
        # Price momentum scores
        if 'close' in latest:
            current_price = latest['close']
            previous_price = prev['close']
            scores['price_change_pct'] = ((current_price - previous_price) / previous_price) * 100
        
        # RSI score (ideal range 30-70)
        rsi_col = 'RSI_14' if 'RSI_14' in latest else 'rsi_14'
        if rsi_col in latest and not pd.isna(latest[rsi_col]):
            rsi = latest[rsi_col]
            if 30 <= rsi <= 70:
                scores['rsi_score'] = 1.0
            elif 25 <= rsi <= 75:
                scores['rsi_score'] = 0.5
            else:
                scores['rsi_score'] = 0.0
            scores['rsi_value'] = rsi
        
        # MACD score
        if all(col in latest for col in ['macd', 'macd_signal', 'macd_hist']):
            macd_line = latest['macd']
            macd_signal = latest['macd_signal']
            macd_hist = latest['macd_hist']
            
            # Bullish MACD conditions
            macd_bullish = macd_line > macd_signal and macd_hist > 0
            scores['macd_bullish'] = 1.0 if macd_bullish else 0.0
            scores['macd_value'] = macd_line
            scores['macd_histogram'] = macd_hist
        
        # Bollinger Bands position
        bb_upper_col = 'BB_Upper' if 'BB_Upper' in latest else 'bb_upper'
        bb_lower_col = 'BB_Lower' if 'BB_Lower' in latest else 'bb_lower'
        bb_middle_col = 'BB_Middle' if 'BB_Middle' in latest else 'bb_middle'
        
        if all(col in latest for col in ['close', bb_upper_col, bb_lower_col]):
            price = latest['close']
            bb_upper = latest[bb_upper_col]
            bb_lower = latest[bb_lower_col]
            
            if not (pd.isna(bb_upper) or pd.isna(bb_lower)):
                bb_position = (price - bb_lower) / (bb_upper - bb_lower)
                scores['bb_position'] = bb_position
                
                # Ideal position is 0.2 to 0.8 (not too extreme)
                if 0.2 <= bb_position <= 0.8:
                    scores['bb_score'] = 1.0
                elif 0.1 <= bb_position <= 0.9:
                    scores['bb_score'] = 0.5
                else:
                    scores['bb_score'] = 0.0
        
        # ATR/Price ratio (using volatility_20 as ATR proxy)
        volatility_col = 'volatility_20' if 'volatility_20' in latest else 'VOLATILITY_20'
        if volatility_col in latest and not pd.isna(latest[volatility_col]):
            volatility_20 = latest[volatility_col]
            atr_to_price_ratio = (volatility_20 / current_price) if current_price > 0 else 0
            scores['atr_to_price_ratio'] = atr_to_price_ratio
            
            # Score based on volatility levels (lower is better for stability)
            if atr_to_price_ratio < 0.02:  # Very low volatility
                scores['atr_score'] = 1.0
            elif atr_to_price_ratio < 0.03:  # Low volatility
                scores['atr_score'] = 0.8
            elif atr_to_price_ratio < 0.05:  # Moderate volatility
                scores['atr_score'] = 0.6
            else:  # High volatility
                scores['atr_score'] = 0.3

        # Volume analysis
        if 'vol' in latest and 'vol' in df.columns:
            current_volume = latest['vol']
            avg_volume = df['vol'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            scores['volume_ratio'] = volume_ratio
            scores['volume_score'] = min(volume_ratio / 2.0, 1.0)  # Cap at 1.0
        
        # Volume Ratio Indicator (VRI) - using the new indicators
        if 'vri_20' in latest and not pd.isna(latest['vri_20']):
            scores['vri_20'] = latest['vri_20']
        if 'vri_10' in latest and not pd.isna(latest['vri_10']):
            scores['vri_10'] = latest['vri_10']
        
        # Chaikin Money Flow (CMF)
        if 'cmf_20' in latest and not pd.isna(latest['cmf_20']):
            scores['cmf_20'] = latest['cmf_20']
        if 'cmf_10' in latest and not pd.isna(latest['cmf_10']):
            scores['cmf_10'] = latest['cmf_10']
        
        # Accumulation/Distribution Line momentum
        if 'ad_momentum' in latest and not pd.isna(latest['ad_momentum']):
            scores['ad_momentum'] = latest['ad_momentum']
        
        # Moving average trend
        if all(col in latest for col in ['close', 'sma_20', 'sma_144']):
            price = latest['close']
            sma_20 = latest['sma_20']
            sma_144 = latest['sma_144']
            
            if not (pd.isna(sma_20) or pd.isna(sma_144)):
                above_sma20 = price > sma_20
                above_sma144 = price > sma_144
                sma20_above_sma144 = sma_20 > sma_144
                
                trend_score = sum([above_sma20, above_sma144, sma20_above_sma144]) / 3.0
                scores['trend_score'] = trend_score
        
        # Calculate risk/reward metrics using technical indicators
        rr_data = TechnicalIndicators.risk_reward_ratio(current_price, recent_high, recent_low)
        scores.update({
            'risk_amount': rr_data['risk'],
            'reward_amount': rr_data['reward'],
            'risk_reward_ratio': rr_data['ratio'],
            'rr_status': rr_data['status']
        })
        
        # Calculate overall composite score
        score_components = [
            scores.get('rsi_score', 0) * 0.2,
            scores.get('macd_bullish', 0) * 0.2,
            scores.get('bb_score', 0) * 0.2,
            scores.get('volume_score', 0) * 0.2,
            scores.get('trend_score', 0) * 0.2
        ]
        scores['composite_score'] = sum(score_components)
        
        return scores
    
    def calculate_professional_analysis_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate professional analysis criteria scores similar to fractal_predict.py
        
        Args:
            df: DataFrame with stock data and technical indicators
            
        Returns:
            Dictionary with trend_score, safety_score, relative_strength_score, and total_score
        """
        if df is None or len(df) == 0:
            return {'trend_score': 0, 'safety_score': 0, 'relative_strength_score': 0, 'total_score': 0}
        
        current_data = df.iloc[-1]
        current_price = current_data['close']
        
        # Get technical indicators
        sma_144 = current_data.get('sma_144', 0)
        sma_50 = current_data.get('sma_50', 0)
        macd = current_data.get('macd', 0)
        macd_hist = current_data.get('macd_histogram', 0)
        rsi = current_data.get('rsi_14', 0)
        volume_sma = current_data.get('volume_sma', 0)
        volatility_20 = current_data.get('volatility_20', 0)
        
        # Calculate ATR/Price ratio (using volatility as proxy)
        atr_to_price = (volatility_20 / current_price) if current_price > 0 else 0
        
        # Initialize scores
        trend_score = 0
        safety_score = 0
        relative_strength_score = 0
        
        # Trend score calculation (0-4 points)
        if sma_50 > 0 and sma_144 > 0:
            if current_price > sma_144 and sma_50 > sma_144:
                trend_score += 2
            if macd > 0 and macd_hist > 0:
                trend_score += 2
        
        # Safety score calculation (0-6 points)
        if 40 <= rsi <= 75:
            safety_score += 2
        elif 30 <= rsi <= 80:
            safety_score += 1
        
        if atr_to_price < 0.02:
            safety_score += 3
        elif atr_to_price < 0.03:
            safety_score += 2
        elif atr_to_price < 0.05:
            safety_score += 1
        
        if volume_sma > 2000000:
            safety_score += 2
        elif volume_sma > 1000000:
            safety_score += 1
        
        # Relative strength score calculation (0-6 points)
        # 3M Momentum
        if len(df) >= 60:
            price_60d_ago = df['close'].iloc[-60]
            momentum_3m = (current_price - price_60d_ago) / price_60d_ago if price_60d_ago > 0 else 0
            if momentum_3m > 0.02:
                relative_strength_score += 2
        
        # 52W High proximity
        if len(df) >= 252:
            high_52w = df['high'].tail(252).max()
            distance_from_high = (current_price - high_52w) / high_52w
            if distance_from_high > -0.50:
                relative_strength_score += 2
        else:
            high_available = df['high'].max()
            distance_from_high = (current_price - high_available) / high_available
            if distance_from_high > -0.20:
                relative_strength_score += 1
        
        # Recent price stability
        if len(df) >= 5:
            price_5d_ago = df['close'].iloc[-5]
            recent_change = (current_price - price_5d_ago) / price_5d_ago
            if -0.10 < recent_change < 0.20:
                relative_strength_score += 1
        
        # Technical confirmation
        macd_signal = current_data.get('macd_signal', 0)
        if macd > macd_signal:
            relative_strength_score += 1
        
        # Calculate total score
        total_score = trend_score + safety_score + relative_strength_score
        
        return {
            'trend_score': trend_score,
            'safety_score': safety_score,
            'relative_strength_score': relative_strength_score,
            'total_score': total_score
        }
    
    def train_lightgbm_model(self, ticker: str, df: pd.DataFrame) -> Optional[object]:
        """
        Train a new LightGBM model for a ticker using the provided data
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with stock data and technical indicators
            
        Returns:
            Trained model or None if training failed
        """
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not available for training")
            return None
        
        try:
            logger.info(f"🔄 Training LightGBM model for {ticker}...")
            
            # Import training dependencies
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_class_weight
            from imblearn.over_sampling import SMOTE
            from sklearn.metrics import accuracy_score, classification_report
            
            # Create prediction labels (6-class classification)
            df_train = df.copy()
            df_train['change_pct_1d'] = (df_train['close'].shift(-1) - df_train['close']) / df_train['close'] * 100
            
            def classify_change(change_pct):
                if pd.isna(change_pct):
                    return None
                elif change_pct <= -10:
                    return 0  # Drop >10%
                elif change_pct > -10 and change_pct <= -5:
                    return 1  # Drop 5-10%
                elif change_pct > -5 and change_pct < 0:
                    return 2  # Drop 0-5%
                elif change_pct >= 0 and change_pct < 5:
                    return 3  # Gain 0-5%
                elif change_pct >= 5 and change_pct < 10:
                    return 4  # Gain 5-10%
                elif change_pct >= 10:
                    return 5  # Gain >10%
                else:
                    return None
            
            df_train['label_class'] = df_train['change_pct_1d'].apply(classify_change)
            df_train = df_train.dropna(subset=['label_class'])
            df_train['label_class'] = df_train['label_class'].astype(int)
            
            if len(df_train) < 100:
                logger.warning(f"Insufficient data for training {ticker}: {len(df_train)} samples (need at least 100)")
                return None
            
            # Get features from config
            from config import features
            available_features = [f.lower() for f in features if f.lower() in df_train.columns]
            
            if len(available_features) < 20:
                logger.warning(f"Insufficient features for training {ticker}: {len(available_features)} available (need at least 20)")
                return None
            
            # Prepare training data
            X = df_train[available_features].fillna(0)
            y = df_train['label_class']
            
            # Check class distribution
            class_counts = y.value_counts()
            logger.info(f"Class distribution for {ticker}: {dict(class_counts)}")
            
            if len(class_counts) < 3:
                logger.warning(f"Insufficient class diversity for {ticker}: only {len(class_counts)} classes")
                return None
            
            # Filter out classes with too few samples for stratified split
            min_samples_per_class = 2
            valid_classes = class_counts[class_counts >= min_samples_per_class].index
            
            if len(valid_classes) < 3:
                logger.warning(f"Insufficient samples for training {ticker}: need at least {min_samples_per_class} samples per class")
                return None
            
            # Filter data to only include valid classes
            mask = y.isin(valid_classes)
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            logger.info(f"Using {len(valid_classes)} classes with sufficient samples: {list(valid_classes)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
            
            # Apply SMOTE for class balancing
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"Applied SMOTE: {len(X_train)} -> {len(X_train_balanced)} samples")
            except Exception as e:
                logger.warning(f"SMOTE failed, using original data: {e}")
                X_train_balanced, y_train_balanced = X_train, y_train
            
            # Calculate class weights
            classes = np.unique(y_train_balanced)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
            class_weight_dict = dict(zip(classes, class_weights))
            
            # Train LightGBM model (adjust num_class based on actual classes)
            actual_num_classes = len(valid_classes)
            model = lgb.LGBMClassifier(
                objective='multiclass' if actual_num_classes > 2 else 'binary',
                num_class=actual_num_classes if actual_num_classes > 2 else None,
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                class_weight=class_weight_dict,
                random_state=42,
                n_estimators=100,
                verbosity=-1
            )
            
            model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ Model trained for {ticker} - Accuracy: {accuracy:.3f}")
            
            # Save model and class mapping
            model_dir = f"models/{ticker}_yahoo"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/lightgbm_model_yahoo.pkl"
            
            # Save model with metadata
            model_data = {
                'model': model,
                'valid_classes': list(valid_classes),
                'feature_names': available_features,
                'accuracy': accuracy,
                'class_mapping': {
                    0: "Drop >10%", 1: "Drop 5-10%", 2: "Drop 0-5%",
                    3: "Gain 0-5%", 4: "Gain 5-10%", 5: "Gain >10%"
                }
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"💾 Model saved to {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model training failed for {ticker}: {e}")
            return None

    def load_or_train_ml_model(self, ticker: str, df: pd.DataFrame) -> Optional[object]:
        """
        Load LightGBM model for ticker if available, otherwise train a new one
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with stock data (needed for training)
            
        Returns:
            Loaded or newly trained model, or None if training failed
        """
        if not LIGHTGBM_AVAILABLE:
            return None
        
        # If rebuild_models is enabled, skip loading and go straight to training
        if self.rebuild_models:
            logger.info(f"🔄 Rebuild mode enabled - training new model for {ticker}")
            return self.train_lightgbm_model(ticker, df)
        
        # Check for model in various locations
        model_paths = [
            f"models/{ticker}_yahoo/lightgbm_model_yahoo.pkl",
            f"models/{ticker}/vstm_lightgbm_model.pkl",
            f"predictor/models/{ticker}_yahoo/lightgbm_model_yahoo.pkl",
            f"predictor/models/{ticker}/vstm_lightgbm_model.pkl"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    loaded_data = joblib.load(model_path)
                    # Handle new format (with metadata) and old format (just model)
                    if isinstance(loaded_data, dict) and 'model' in loaded_data:
                        model = loaded_data['model']
                        # Store metadata for prediction use
                        model._scanner_metadata = loaded_data
                    else:
                        # Old format - just the model
                        model = loaded_data
                        model._scanner_metadata = None
                    
                    logger.info(f"📁 Loaded existing ML model for {ticker} from {model_path}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load model {model_path}: {e}")
        
        # No model found, train a new one
        logger.info(f"🚀 No model found for {ticker}, training new model...")
        return self.train_lightgbm_model(ticker, df)
    
    def predict_with_ml(self, ticker: str, df: pd.DataFrame, scores: Dict[str, float]) -> Optional[Dict[str, any]]:
        """
        Generate ML-based prediction using LightGBM model
        
        Args:
            ticker: Stock ticker symbol
            df: Stock data DataFrame
            scores: Technical analysis scores
            
        Returns:
            ML prediction dictionary or None if not available
        """
        if not self.use_ml_predictions or not LIGHTGBM_AVAILABLE:
            return None
        
        model = self.load_or_train_ml_model(ticker, df)
        if model is None:
            logger.warning(f"Could not load or train ML model for {ticker}")
            return None
        
        try:
            # Prepare features for ML prediction
            # Use feature names from model metadata if available, otherwise fall back to config
            if hasattr(model, '_scanner_metadata') and model._scanner_metadata and 'feature_names' in model._scanner_metadata:
                # Use the exact feature names the model was trained with
                model_feature_names = model._scanner_metadata['feature_names']
                logger.debug(f"Using model feature names: {len(model_feature_names)} features")
            else:
                # Fall back to config features
                from config import features
                model_feature_names = features
                logger.debug(f"Using config feature names: {len(model_feature_names)} features")
            
            # Get latest row for prediction
            latest_data = df.iloc[-1:].copy()
            
            # Select features that exist in the data (case-insensitive matching)
            available_features = []
            for feature in model_feature_names:
                # Try exact match first, then case-insensitive
                if feature in latest_data.columns:
                    available_features.append(feature)
                elif feature.lower() in latest_data.columns:
                    available_features.append(feature.lower())
                elif feature.upper() in latest_data.columns:
                    available_features.append(feature.upper())
            
            if len(available_features) < 10:  # Minimum feature threshold
                logger.warning(f"Insufficient features for ML prediction: {len(available_features)} available")
                return None
            
            # Prepare feature matrix with proper feature names
            X = latest_data[available_features].copy()
            
            # Ensure feature names match what the model expects
            if hasattr(model, 'feature_name_') and model.feature_name_:
                # LightGBM model with feature names - ensure exact match
                expected_features = model.feature_name_[:len(X.columns)]
                X.columns = expected_features
                logger.debug(f"Set feature names to match model: {expected_features[:5]}...")
            
            # Handle any NaN values
            if X.isnull().any().any():
                X = X.fillna(0)
                logger.debug("Handled NaN values in feature matrix")
            
            # Log feature information for debugging
            logger.debug(f"Feature matrix shape: {X.shape}")
            logger.debug(f"Feature names: {list(X.columns)[:5]}...")
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                # Scikit-learn style model
                probabilities = model.predict_proba(X)[0]
            else:
                # LightGBM model
                if hasattr(model, 'best_iteration'):
                    probabilities = model.predict(X, num_iteration=model.best_iteration)[0]
                else:
                    probabilities = model.predict(X)[0]
            
            # Class mapping (from train_lightgbm_yahoo_42.py)
            class_labels = {
                0: "Drop >10%",
                1: "Drop 5-10%", 
                2: "Drop 0-5%",
                3: "Gain 0-5%",
                4: "Gain 5-10%",
                5: "Gain >10%"
            }
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx] * 100
            
            # Map probability index to actual class number
            if hasattr(model, 'classes_'):
                predicted_class = model.classes_[predicted_idx]
            else:
                # Fallback to assuming direct mapping
                predicted_class = predicted_idx
            
            # Convert class to signal and price prediction
            current_price = scores.get('current_price', df.iloc[-1]['close'])
            
            if predicted_class <= 2:  # Drop classes
                signal = "SELL" if predicted_class <= 1 else "HOLD"
                # Predict price decrease
                if predicted_class == 0:  # Drop >10%
                    price_change_pct = -12
                elif predicted_class == 1:  # Drop 5-10%
                    price_change_pct = -7.5
                else:  # Drop 0-5%
                    price_change_pct = -2.5
            else:  # Gain classes
                signal = "BUY" if predicted_class >= 4 else "HOLD"
                # Predict price increase
                if predicted_class == 3:  # Gain 0-5%
                    price_change_pct = 2.5
                elif predicted_class == 4:  # Gain 5-10%
                    price_change_pct = 7.5
                else:  # Gain >10%
                    price_change_pct = 12
            
            predicted_price = current_price * (1 + price_change_pct / 100)
            
            # Try to get model classes mapping from model metadata
            model_classes = None
            if hasattr(model, 'classes_'):
                model_classes = model.classes_.tolist()
            elif hasattr(model, '_scanner_metadata') and model._scanner_metadata:
                valid_classes = model._scanner_metadata.get('valid_classes')
                if valid_classes:
                    model_classes = valid_classes
            
            return {
                'method': 'ML',
                'model_class': predicted_class,
                'class_label': class_labels[predicted_class],
                'probabilities': probabilities.tolist(),
                'model_classes': model_classes,
                'signal': signal,
                'confidence': confidence,
                'price_change_pct': price_change_pct,
                'predicted_price': predicted_price
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"ML prediction failed for {ticker}: {error_msg}")
            
            # Check if it's a feature mismatch error and retrain
            if "features" in error_msg and ("expecting" in error_msg or "mismatch" in error_msg):
                logger.info(f"🔄 Feature mismatch detected for {ticker}. Retraining model...")
                
                # Remove old model and retrain
                import os, shutil
                model_dir = f"models/{ticker}_yahoo"
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                
                # Retrain the model
                new_model = self.train_lightgbm_model(ticker, df)
                if new_model:
                    logger.info(f"✅ Successfully retrained model for {ticker}. Retrying prediction...")
                    # Retry prediction with new model
                    try:
                        return self.predict_with_ml(ticker, df, scores)
                    except Exception as retry_e:
                        logger.error(f"Prediction still failed after retraining for {ticker}: {retry_e}")
                
            return None
    
    def generate_prediction(self, ticker: str, scores: Dict[str, float], df: pd.DataFrame = None) -> Dict[str, any]:
        """
        Generate buy/sell prediction based on technical scores or ML model
        
        Args:
            ticker: Stock ticker symbol
            scores: Technical analysis scores
            df: Stock data DataFrame (required for ML predictions)
            
        Returns:
            Dictionary with prediction details
        """
        # Calculate professional analysis scores
        prof_scores = self.calculate_professional_analysis_score(df) if df is not None else {'trend_score': 0, 'safety_score': 0, 'relative_strength_score': 0, 'total_score': 0}
        
        # Try ML prediction first if enabled
        if self.use_ml_predictions and df is not None:
            ml_prediction = self.predict_with_ml(ticker, df, scores)
            if ml_prediction is not None:
                # Use ML prediction but include technical scores
                current_price = scores.get('current_price', df.iloc[-1]['close'] if df is not None else 100)
                
                prediction = {
                    'ticker': ticker,
                    'signal': ml_prediction['signal'],
                    'confidence': round(ml_prediction['confidence'], 2),
                    'composite_score': round(scores.get('composite_score', 0), 3),
                    'current_price': current_price,
                    'predicted_price': round(ml_prediction['predicted_price'], 2),
                    'price_change': round(ml_prediction['price_change_pct'], 2),
                    'timestamp': datetime.now().isoformat(),
                    'technical_scores': scores,
                    'professional_scores': prof_scores,
                    'ml_prediction': {
                        'method': ml_prediction['method'],
                        'model_class': ml_prediction['model_class'],
                        'class_label': ml_prediction['class_label'],
                        'probabilities': ml_prediction['probabilities'],
                        'model_classes': ml_prediction.get('model_classes')
                    }
                }
                
                return prediction
        
        # Fallback to confidence-based prediction
        composite_score = scores.get('composite_score', 0)
        
        # Determine signal based on composite score
        if composite_score >= 0.7:
            signal = "BUY"
            confidence = min(composite_score * 100, 95)
        elif composite_score >= 0.5:
            signal = "HOLD"
            confidence = composite_score * 80
        elif composite_score >= 0.3:
            signal = "WEAK_HOLD"
            confidence = composite_score * 60
        else:
            signal = "SELL"
            confidence = (1 - composite_score) * 70
        
        # Calculate price prediction based on signal and confidence
        current_price = scores.get('current_price', 100)
        if signal == 'BUY':
            # Predict price increase based on confidence (up to 15% for high confidence)
            price_prediction = current_price * (1 + (confidence / 100) * 0.15)
        elif signal == 'SELL':
            # Predict price decrease based on confidence (up to 15% for high confidence)
            price_prediction = current_price * (1 - (confidence / 100) * 0.15)
        else:
            # HOLD signals predict minimal change (±2% based on confidence)
            price_prediction = current_price * (1 + ((confidence - 50) / 100) * 0.02)
        
        prediction = {
            'ticker': ticker,
            'signal': signal,
            'confidence': round(confidence, 2),
            'composite_score': round(composite_score, 3),
            'current_price': current_price,
            'predicted_price': round(price_prediction, 2),
            'price_change': round(((price_prediction - current_price) / current_price) * 100, 2),
            'timestamp': datetime.now().isoformat(),
            'technical_scores': scores,
            'professional_scores': prof_scores
        }
        
        return prediction
    
    def apply_filters(self, predictions: List[Dict], filters: Dict) -> List[Dict]:
        """
        Apply filters to predictions based on criteria
        
        Args:
            predictions: List of prediction dictionaries
            filters: Dictionary containing filter criteria
            
        Returns:
            Filtered list of predictions
        """
        if not filters:
            return predictions
        
        filtered_predictions = []
        
        for pred in predictions:
            # Apply ticker filter
            if filters.get('ticker') and pred['ticker'] not in filters['ticker']:
                continue
            
            tech_scores = pred.get('technical_scores', {})
            
            # Apply R/R Ratio filter
            rr_ratio = tech_scores.get('risk_reward_ratio', 0)
            if filters.get('min_rr_ratio') and rr_ratio < filters['min_rr_ratio']:
                continue
            if filters.get('max_rr_ratio') and rr_ratio > filters['max_rr_ratio']:
                continue
            
            # Apply ATR/Price filter
            atr_ratio = tech_scores.get('atr_to_price_ratio', 0)
            if filters.get('max_atr_price_ratio') and atr_ratio > filters['max_atr_price_ratio']:
                continue
            
            # Apply BB Position filter
            bb_position = tech_scores.get('bb_position', 0)
            if filters.get('min_bb_position') and bb_position < filters['min_bb_position']:
                continue
            if filters.get('max_bb_position') and bb_position > filters['max_bb_position']:
                continue
            
            # Apply VRI filters
            vri_20 = tech_scores.get('vri_20', 0)
            if filters.get('min_vri_20') is not None and vri_20 < filters['min_vri_20']:
                continue
            if filters.get('max_vri_20') is not None and vri_20 > filters['max_vri_20']:
                continue
            
            vri_10 = tech_scores.get('vri_10', 0)
            if filters.get('min_vri_10') is not None and vri_10 < filters['min_vri_10']:
                continue
            if filters.get('max_vri_10') is not None and vri_10 > filters['max_vri_10']:
                continue
            
            # Apply CMF filters
            cmf_20 = tech_scores.get('cmf_20', 0)
            if filters.get('min_cmf_20') is not None and cmf_20 < filters['min_cmf_20']:
                continue
            if filters.get('max_cmf_20') is not None and cmf_20 > filters['max_cmf_20']:
                continue
            
            cmf_10 = tech_scores.get('cmf_10', 0)
            if filters.get('min_cmf_10') is not None and cmf_10 < filters['min_cmf_10']:
                continue
            if filters.get('max_cmf_10') is not None and cmf_10 > filters['max_cmf_10']:
                continue
            
            # Apply A/D Line momentum filter
            ad_momentum = tech_scores.get('ad_momentum', 0)
            if filters.get('min_ad_momentum') is not None and ad_momentum < filters['min_ad_momentum']:
                continue
            if filters.get('max_ad_momentum') is not None and ad_momentum > filters['max_ad_momentum']:
                continue
            
            # Apply RSI filter
            rsi_value = tech_scores.get('rsi_value', 0)
            if filters.get('min_rsi') is not None and rsi_value < filters['min_rsi']:
                continue
            if filters.get('max_rsi') is not None and rsi_value > filters['max_rsi']:
                continue
            
            filtered_predictions.append(pred)
        
        logger.info(f"Applied filters: {len(predictions)} -> {len(filtered_predictions)} predictions")
        return filtered_predictions

    def scan_stocks(self, date_filter: str = "latest", max_tickers: Optional[int] = None, 
                   predict_only: bool = False, filters: Dict = None) -> List[Dict]:
        """
        Scan stocks and generate predictions
        
        Args:
            date_filter: Date filter for ticker selection
            max_tickers: Maximum number of tickers to scan
            predict_only: Only generate predictions, don't run full scan
            filters: Dictionary containing filter criteria
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Starting stock scan - predict_only: {predict_only}")
        
        tickers = self.get_tickers(date_filter, max_tickers)
        predictions = []
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            
            # Get stock data
            df = self.get_stock_data(ticker)
            if df is None:
                continue
            
            # Calculate technical scores
            scores = self.calculate_technical_scores(df)
            if not scores:
                logger.warning(f"Could not calculate scores for {ticker}")
                continue
            
            # Generate prediction
            prediction = self.generate_prediction(ticker, scores, df)
            predictions.append(prediction)
            
            # Log significant predictions
            if prediction['confidence'] >= 70:
                logger.info(f"{ticker}: {prediction['signal']} (confidence: {prediction['confidence']}%)")
        
        # Apply filters if provided
        if filters:
            predictions = self.apply_filters(predictions, filters)
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Note: Predictions are no longer saved to database automatically
        # Database entries should only be managed manually through the watchlist interface
        
        logger.info(f"Scan complete: {len(predictions)} predictions generated")
        return predictions
    
    def save_predictions(self, predictions: List[Dict], output_file: str = None, output_dir: str = "scanner_reports") -> str:
        """
        Save predictions to CSV file
        
        Args:
            predictions: List of prediction dictionaries
            output_file: Output file path (optional)
            output_dir: Output directory (default: scanner_reports)
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"scanner_predictions_{timestamp}.csv"
        
        # If output_file is just a filename, combine with output_dir
        if not os.path.dirname(output_file):
            output_file = os.path.join(output_dir, output_file)
        
        # Convert to DataFrame for easier saving
        df_predictions = pd.DataFrame(predictions)
        
        # Flatten technical_scores for CSV
        if not df_predictions.empty and 'technical_scores' in df_predictions.columns:
            tech_scores_df = pd.json_normalize(df_predictions['technical_scores'])
            tech_scores_df.columns = [f"tech_{col}" for col in tech_scores_df.columns]
            df_predictions = pd.concat([df_predictions.drop('technical_scores', axis=1), tech_scores_df], axis=1)
        
        df_predictions.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")
        
        return output_file
    
    def save_predictions_to_database(self, predictions: List[Dict]) -> int:
        """
        Save prediction data to seekingalpha table
        
        Args:
            predictions: List of prediction dictionaries containing ticker, price, exp, etc.
            
        Returns:
            Number of records successfully saved
        """
        try:
            # Import required for database operations
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            saved_count = 0
            failed_count = 0
            
            # Connect to database
            conn = psycopg2.connect(**DB_CONFIG)
            
            for prediction in predictions:
                try:
                    ticker = prediction.get('ticker', '').upper()
                    price = prediction.get('price', 0.0)
                    exp_date = prediction.get('exp')
                    premiums = prediction.get('price', 0.0)  # Use option price as premiums
                    
                    if not ticker:
                        logger.warning("Skipping prediction with empty ticker")
                        continue
                    
                    # Convert date to proper format if it's a string
                    if isinstance(exp_date, str):
                        from datetime import datetime
                        exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                    
                    with conn.cursor() as cur:
                        # Insert or update record with option data
                        cur.execute("""
                            INSERT INTO seekingalpha (ticker, price, exp, premiums, date_added)
                            VALUES (%s, %s, %s, %s, CURRENT_DATE)
                            ON CONFLICT (ticker, date_added) 
                            DO UPDATE SET 
                                price = EXCLUDED.price,
                                exp = EXCLUDED.exp,
                                premiums = EXCLUDED.premiums
                        """, (ticker, price, exp_date, premiums))
                        
                        if cur.rowcount > 0:
                            saved_count += 1
                        else:
                            failed_count += 1
                            
                except Exception as e:
                    logger.error(f"Error saving prediction for {prediction.get('ticker', 'unknown')}: {e}")
                    failed_count += 1
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database save complete: {saved_count} saved, {failed_count} failed")
            return saved_count
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            return 0
    
    def generate_html_report(self, predictions: List[Dict], output_dir: str = "scanner_reports") -> str:
        """
        Generate comprehensive HTML report for scanner predictions
        
        Args:
            predictions: List of prediction dictionaries
            output_dir: Directory to save the report (default: scanner_reports)
            
        Returns:
            Path to the generated HTML report
        """
        if not predictions:
            logger.error("No predictions to generate report")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for easier processing
        df_predictions = pd.DataFrame(predictions)
        
        # Fetch premiums from database for each ticker
        try:
            import psycopg2
            conn = psycopg2.connect(**DB_CONFIG)
            with conn.cursor() as cur:
                # Get all tickers in the predictions
                tickers = [pred['ticker'] for pred in predictions]
                if tickers:
                    cur.execute("""
                        SELECT DISTINCT ON (ticker) ticker, premiums 
                        FROM seekingalpha 
                        WHERE ticker = ANY(%s)
                        ORDER BY ticker, date_added DESC
                    """, (tickers,))
                    
                    # Create a mapping of ticker -> premiums
                    rows = cur.fetchall()
                    db_premiums = {row[0]: row[1] for row in rows if row[1] is not None}
                    
                    logger.info(f"Found premiums for {len(db_premiums)} tickers in database: {list(db_premiums.keys())}")
                    
                    # Update predictions with database premiums only
                    for pred in predictions:
                        ticker = pred['ticker']
                        premium = db_premiums.get(ticker, 0.0)
                        pred['db_premiums'] = premium
                        if premium > 0:
                            logger.info(f"Using premium ${premium:.2f} for {ticker}")
            conn.close()
        except Exception as e:
            logger.warning(f"Error fetching premiums from database: {e}")
            # Set default values
            for pred in predictions:
                pred['db_premiums'] = 0.0
        
        # Flatten technical_scores for analysis
        if 'technical_scores' in df_predictions.columns:
            tech_scores_df = pd.json_normalize(df_predictions['technical_scores'])
            tech_scores_df.columns = [f"tech_{col}" for col in tech_scores_df.columns]
            df_analysis = pd.concat([df_predictions.drop('technical_scores', axis=1), tech_scores_df], axis=1)
        else:
            df_analysis = df_predictions
        
        # Calculate statistics
        total_predictions = len(predictions)
        buy_signals = [p for p in predictions if p['signal'] == 'BUY']
        sell_signals = [p for p in predictions if p['signal'] == 'SELL']
        hold_signals = [p for p in predictions if p['signal'] in ['HOLD', 'WEAK_HOLD']]
        
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        hold_count = len(hold_signals)
        
        avg_confidence = df_predictions['confidence'].mean() if not df_predictions.empty else 0
        
        # Generate timestamp
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Helper function to format numbers with K/M suffixes
        def format_number(value):
            if value >= 1000000:
                return f"{value/1000000:.1f}M"
            elif value >= 1000:
                return f"{value/1000:.1f}K"
            else:
                return f"{value:.2f}"
        
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stock Scanner Predictions Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }}
        .stat h3 {{ margin: 0; color: #333; }}
        .stat p {{ margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #667eea; }}
        .controls {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .filter-buttons {{ margin: 10px 0; }}
        .filter-btn {{ 
            padding: 8px 15px; margin: 5px; border: none; border-radius: 20px; cursor: pointer; 
            font-weight: bold; transition: all 0.3s;
        }}
        .filter-btn.active {{ box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}
        .btn-all {{ background: #6c757d; color: white; }}
        .btn-all.active {{ background: #5a6268; }}
        .btn-buy {{ background: #28a745; color: white; }}
        .btn-buy.active {{ background: #218838; }}
        .btn-sell {{ background: #dc3545; color: white; }}
        .btn-sell.active {{ background: #c82333; }}
        .btn-hold {{ background: #ffc107; color: black; }}
        .btn-hold.active {{ background: #e0a800; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; cursor: pointer; position: sticky; top: 0; }}
        th:hover {{ opacity: 0.8; }}
        .ticker {{ font-weight: bold; color: #0066cc; }}
        .signal-BUY {{ background: #d4edda; color: #155724; font-weight: bold; }}
        .signal-SELL {{ background: #f8d7da; color: #721c24; font-weight: bold; }}
        .signal-HOLD {{ background: #fff3cd; color: #856404; font-weight: bold; }}
        .signal-WEAK_HOLD {{ background: #f8f9fa; color: #6c757d; font-weight: bold; }}
        .confidence-high {{ color: #28a745; font-weight: bold; }}
        .confidence-medium {{ color: #ffc107; font-weight: bold; }}
        .confidence-low {{ color: #dc3545; font-weight: bold; }}
        .rsi-oversold {{ color: #28a745; }}
        .rsi-overbought {{ color: #dc3545; }}
        .rsi-neutral {{ color: #6c757d; }}
        .macd-bullish {{ color: #28a745; }}
        .macd-bearish {{ color: #dc3545; }}
        .price-positive {{ color: #28a745; font-weight: bold; }}
        .price-negative {{ color: #dc3545; font-weight: bold; }}
        .price-neutral {{ color: #6c757d; }}
        .risk-amount {{ color: #dc3545; }}
        .reward-amount {{ color: #28a745; }}
        .good-ratio {{ color: #28a745; font-weight: bold; }}
        .decent-ratio {{ color: #ffc107; }}
        .poor-ratio {{ color: #dc3545; }}
        .ml-negative {{ color: #dc3545; font-weight: bold; }}
        .ml-positive {{ color: #28a745; font-weight: bold; }}
        .ml-class-small {{ font-size: 0.8em; color: #6c757d; font-style: italic; }}
        .ml-class-drop-severe {{ color: #dc3545; font-weight: bold; }}  /* Drop >10% - Class 0 */
        .ml-class-drop-moderate {{ color: #fd7e14; font-weight: bold; }} /* Drop 5-10% - Class 1 */
        .ml-class-drop-mild {{ color: #ffc107; font-weight: bold; }}     /* Drop 0-5% - Class 2 */
        .ml-class-gain-mild {{ color: #20c997; font-weight: bold; }}     /* Gain 0-5% - Class 3 */
        .ml-class-gain-moderate {{ color: #28a745; font-weight: bold; }} /* Gain 5-10% - Class 4 */
        .ml-class-gain-severe {{ color: #198754; font-weight: bold; }}   /* Gain >10% - Class 5 */
        .vri-high {{ color: #28a745; font-weight: bold; }}               /* VRI > 1.5 */
        .vri-normal {{ color: #6c757d; }}                                /* VRI 0.5-1.5 */
        .vri-low {{ color: #dc3545; }}                                   /* VRI < 0.5 */
        .cmf-positive {{ color: #28a745; font-weight: bold; }}           /* CMF > 0.1 */
        .cmf-neutral {{ color: #6c757d; }}                               /* CMF -0.1 to 0.1 */
        .cmf-negative {{ color: #dc3545; font-weight: bold; }}           /* CMF < -0.1 */
        .ad-momentum-positive {{ color: #28a745; font-weight: bold; }}   /* A/D Momentum > 0 */
        .ad-momentum-negative {{ color: #dc3545; font-weight: bold; }}   /* A/D Momentum < 0 */
        .ad-momentum-neutral {{ color: #6c757d; }}                       /* A/D Momentum ≈ 0 */
        .sort-desc::after {{ content: ' ↓'; }}
        .sort-asc::after {{ content: ' ↑'; }}
        .footer {{ margin-top: 30px; text-align: center; color: #6c757d; font-size: 0.9em; }}
    </style>
    <script>
        let currentSort = {{ column: 'ml_class', direction: 'desc' }};
        let currentFilter = 'all';
        
        function sortTable(columnIndex, columnName) {{
            const table = document.getElementById('resultsTable');
            const tbody = table.getElementsByTagName('tbody')[0];
            const rows = Array.from(tbody.getElementsByTagName('tr')).filter(row => row.style.display !== 'none');
            
            // Toggle sort direction
            if (currentSort.column === columnName) {{
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSort.column = columnName;
                currentSort.direction = 'desc';
            }}
            
            // Update header indicators
            document.querySelectorAll('th').forEach(th => {{
                th.className = th.className.replace(' sort-asc', '').replace(' sort-desc', '');
            }});
            document.querySelectorAll('th')[columnIndex].className += ' sort-' + currentSort.direction;
            
            // Sort rows
            rows.sort((a, b) => {{
                let aVal = a.getElementsByTagName('td')[columnIndex].textContent.trim();
                let bVal = b.getElementsByTagName('td')[columnIndex].textContent.trim();
                
                // Special handling for different column types
                if (columnName === 'risk_amount' || columnName === 'reward_amount') {{
                    // Extract percentage values from parentheses, e.g., "(9.6%)" -> 9.6
                    const aMatch = aVal.match(/\\(([0-9.-]+)%\\)/);
                    const bMatch = bVal.match(/\\(([0-9.-]+)%\\)/);
                    
                    if (aMatch && bMatch) {{
                        aVal = parseFloat(aMatch[1]);
                        bVal = parseFloat(bMatch[1]);
                    }}
                }} else if (columnName === 'predicted_price') {{
                    // Extract percentage values from parentheses for predicted price, e.g., "(+7.5%)" -> 7.5
                    const aMatch = aVal.match(/\\(([+-]?[0-9.-]+)%\\)/);
                    const bMatch = bVal.match(/\\(([+-]?[0-9.-]+)%\\)/);
                    
                    if (aMatch && bMatch) {{
                        aVal = parseFloat(aMatch[1]);
                        bVal = parseFloat(bMatch[1]);
                    }}
                }} else if (columnName === 'option_price') {{
                    // For strike price column, sort by OTM percentage, e.g., "(-97.8% ITM)" -> -97.8
                    const aMatch = aVal.match(/\\(([+-]?[0-9.-]+)% (?:OTM|ITM)\\)/);
                    const bMatch = bVal.match(/\\(([+-]?[0-9.-]+)% (?:OTM|ITM)\\)/);
                    
                    if (aMatch && bMatch) {{
                        aVal = parseFloat(aMatch[1]);
                        bVal = parseFloat(bMatch[1]);
                    }}
                }} else if (columnName === 'ml_class') {{
                    // For ML class, sort by class number first, then by confidence percentage
                    // Extract class number from the displayed class labels
                    const getClassNumber = (text) => {{
                        if (text.includes('Drop >10%')) return 0;
                        if (text.includes('Drop 5-10%')) return 1;
                        if (text.includes('Drop 0-5%')) return 2;
                        if (text.includes('Gain 0-5%')) return 3;
                        if (text.includes('Gain 5-10%')) return 4;
                        if (text.includes('Gain >10%')) return 5;
                        return 999; // Unknown class
                    }};
                    
                    const aClassNum = getClassNumber(aVal);
                    const bClassNum = getClassNumber(bVal);
                    
                    if (aClassNum !== bClassNum) {{
                        // Sort by class number first
                        aVal = aClassNum;
                        bVal = bClassNum;
                    }} else {{
                        // Same class, sort by confidence percentage
                        const aConfMatch = aVal.match(/([0-9.-]+)%/);
                        const bConfMatch = bVal.match(/([0-9.-]+)%/);
                        
                        if (aConfMatch && bConfMatch) {{
                            aVal = parseFloat(aConfMatch[1]);
                            bVal = parseFloat(bConfMatch[1]);
                        }}
                    }}
                }} else {{
                    // Handle numeric values for other columns
                    if (!isNaN(parseFloat(aVal)) && !isNaN(parseFloat(bVal))) {{
                        aVal = parseFloat(aVal);
                        bVal = parseFloat(bVal);
                    }}
                }}
                
                if (currentSort.direction === 'asc') {{
                    return aVal > bVal ? 1 : -1;
                }} else {{
                    return aVal < bVal ? 1 : -1;
                }}
            }});
            
            // Reorder rows in DOM
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        
        function applyAdvancedFilters() {{
            const tickerFilter = document.getElementById('tickerFilter').value.toUpperCase().trim();
            const minRRRatioValue = document.getElementById('minRRRatio').value.trim();
            const maxATRRatioValue = document.getElementById('maxATRRatio').value.trim();
            const minBBPositionValue = document.getElementById('minBBPosition').value.trim();
            const maxBBPositionValue = document.getElementById('maxBBPosition').value.trim();
            const minVRI20Value = document.getElementById('minVRI20').value.trim();
            const maxVRI20Value = document.getElementById('maxVRI20').value.trim();
            const minVRI10Value = document.getElementById('minVRI10').value.trim();
            const maxVRI10Value = document.getElementById('maxVRI10').value.trim();
            const minCMF20Value = document.getElementById('minCMF20').value.trim();
            const maxCMF20Value = document.getElementById('maxCMF20').value.trim();
            const minCMF10Value = document.getElementById('minCMF10').value.trim();
            const maxCMF10Value = document.getElementById('maxCMF10').value.trim();
            const minADMomentumValue = document.getElementById('minADMomentum').value.trim();
            const maxADMomentumValue = document.getElementById('maxADMomentum').value.trim();
            const minRSIValue = document.getElementById('minRSI').value.trim();
            const maxRSIValue = document.getElementById('maxRSI').value.trim();
            const minTotalScoreValue = document.getElementById('minTotalScore').value.trim();
            const minTrendScoreValue = document.getElementById('minTrendScore').value.trim();
            const maxTrendScoreValue = document.getElementById('maxTrendScore').value.trim();
            const mlClassFilterValue = document.getElementById('mlClassFilter').value.trim();
            
            const minRRRatio = minRRRatioValue ? parseFloat(minRRRatioValue) : 0;
            const maxATRRatio = maxATRRatioValue ? parseFloat(maxATRRatioValue) : 0;
            const minBBPosition = minBBPositionValue ? parseFloat(minBBPositionValue) : 0;
            const maxBBPosition = maxBBPositionValue ? parseFloat(maxBBPositionValue) : 0;
            const minVRI20 = minVRI20Value ? parseFloat(minVRI20Value) : 0;
            const maxVRI20 = maxVRI20Value ? parseFloat(maxVRI20Value) : 0;
            const minVRI10 = minVRI10Value ? parseFloat(minVRI10Value) : 0;
            const maxVRI10 = maxVRI10Value ? parseFloat(maxVRI10Value) : 0;
            const minCMF20 = minCMF20Value ? parseFloat(minCMF20Value) : 0;
            const maxCMF20 = maxCMF20Value ? parseFloat(maxCMF20Value) : 0;
            const minCMF10 = minCMF10Value ? parseFloat(minCMF10Value) : 0;
            const maxCMF10 = maxCMF10Value ? parseFloat(maxCMF10Value) : 0;
            const minADMomentum = minADMomentumValue ? parseFloat(minADMomentumValue) : 0;
            const maxADMomentum = maxADMomentumValue ? parseFloat(maxADMomentumValue) : 0;
            const minTotalScore = minTotalScoreValue ? parseFloat(minTotalScoreValue) : 0;
            const minTrendScore = minTrendScoreValue ? parseFloat(minTrendScoreValue) : 0;
            const maxTrendScore = maxTrendScoreValue ? parseFloat(maxTrendScoreValue) : 0;
            
            const rows = document.querySelectorAll('#resultsTable tbody tr');
            let visibleCount = 0;
            
            rows.forEach(row => {{
                let showRow = true;
                const cells = row.getElementsByTagName('td');
                
                if (cells.length > 0) {{
                    // Apply signal filter (existing)
                    const signalCell = cells[2];
                    const signalText = signalCell ? signalCell.textContent.trim() : '';
                    if (currentFilter !== 'all') {{
                        if (currentFilter === 'hold' && !(signalText === 'HOLD' || signalText === 'WEAK_HOLD')) {{
                            showRow = false;
                        }} else if (currentFilter !== 'hold' && signalText !== currentFilter) {{
                            showRow = false;
                        }}
                    }}
                    
                    if (showRow) {{
                        // Apply ticker filter
                        const tickerCell = cells[1];
                        const ticker = tickerCell ? tickerCell.textContent.trim() : '';
                        if (tickerFilter && !ticker.includes(tickerFilter)) {{
                            showRow = false;
                        }}
                        
                        // Apply R/R Ratio filter
                        const rrCell = cells[10]; // R/R Ratio is column 10
                        if (rrCell && showRow) {{
                            const rrText = rrCell.textContent.trim();
                            const rrMatch = rrText.match(/([0-9.-]+)/); // Match first number (before space and status)
                            const rrRatio = rrMatch ? parseFloat(rrMatch[1]) : 0;
                            if (minRRRatioValue && rrRatio < minRRRatio) {{
                                showRow = false;
                            }}
                        }}
                        
                        // Apply ATR/Price filter
                        const atrCell = cells[12]; // ATR/Price is column 12
                        if (atrCell && showRow) {{
                            const atrText = atrCell.textContent.trim();
                            const atrMatch = atrText.match(/([0-9.-]+)%/);
                            const atrPercent = atrMatch ? parseFloat(atrMatch[1]) : 0; // Already in percentage
                            if (maxATRRatioValue && atrPercent > maxATRRatio) {{
                                showRow = false;
                            }}
                        }}
                        
                        // Apply BB Position filter
                        const bbCell = cells[13]; // BB Position is column 13
                        if (bbCell && showRow) {{
                            const bbText = bbCell.textContent.trim();
                            const bbMatch = bbText.match(/([0-9.-]+)%/);
                            const bbPercent = bbMatch ? parseFloat(bbMatch[1]) : 0; // Already in percentage
                            if ((minBBPositionValue && bbPercent < minBBPosition) || 
                                (maxBBPositionValue && bbPercent > maxBBPosition)) {{
                                showRow = false;
                            }}
                        }}
                        
                        // Apply VRI 20 filter
                        if ((minVRI20Value || maxVRI20Value) && showRow) {{
                            const vri20Cell = cells[14]; // VRI 20 is column 14
                            if (vri20Cell) {{
                                const vri20Text = vri20Cell.textContent.trim();
                                const vri20Match = vri20Text.match(/([0-9.-]+)/);
                                const vri20Value = vri20Match ? parseFloat(vri20Match[1]) : 0;
                                if ((minVRI20Value && vri20Value < minVRI20) || 
                                    (maxVRI20Value && vri20Value > maxVRI20)) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply VRI 10 filter
                        if ((minVRI10Value || maxVRI10Value) && showRow) {{
                            const vri10Cell = cells[15]; // VRI 10 is column 15
                            if (vri10Cell) {{
                                const vri10Text = vri10Cell.textContent.trim();
                                const vri10Match = vri10Text.match(/([0-9.-]+)/);
                                const vri10Value = vri10Match ? parseFloat(vri10Match[1]) : 0;
                                if ((minVRI10Value && vri10Value < minVRI10) || 
                                    (maxVRI10Value && vri10Value > maxVRI10)) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply CMF 20 filter
                        if ((minCMF20Value || maxCMF20Value) && showRow) {{
                            const cmf20Cell = cells[16]; // CMF 20 is column 16
                            if (cmf20Cell) {{
                                const cmf20Text = cmf20Cell.textContent.trim();
                                const cmf20Match = cmf20Text.match(/([0-9.-]+)/);
                                const cmf20Value = cmf20Match ? parseFloat(cmf20Match[1]) : 0;
                                if ((minCMF20Value && cmf20Value < minCMF20) || 
                                    (maxCMF20Value && cmf20Value > maxCMF20)) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply CMF 10 filter
                        if ((minCMF10Value || maxCMF10Value) && showRow) {{
                            const cmf10Cell = cells[17]; // CMF 10 is column 17
                            if (cmf10Cell) {{
                                const cmf10Text = cmf10Cell.textContent.trim();
                                const cmf10Match = cmf10Text.match(/([0-9.-]+)/);
                                const cmf10Value = cmf10Match ? parseFloat(cmf10Match[1]) : 0;
                                if ((minCMF10Value && cmf10Value < minCMF10) || 
                                    (maxCMF10Value && cmf10Value > maxCMF10)) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply A/D Momentum filter
                        if ((minADMomentumValue || maxADMomentumValue) && showRow) {{
                            const adMomentumCell = cells[18]; // A/D Momentum is column 18
                            if (adMomentumCell) {{
                                const adMomentumText = adMomentumCell.textContent.trim();
                                const adMomentumMatch = adMomentumText.match(/([0-9.-]+)%/);
                                const adMomentumValue = adMomentumMatch ? parseFloat(adMomentumMatch[1]) : 0;
                                if ((minADMomentumValue && adMomentumValue < minADMomentum) || 
                                    (maxADMomentumValue && adMomentumValue > maxADMomentum)) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply RSI filter
                        if ((minRSIValue || maxRSIValue) && showRow) {{
                            const rsiCell = cells[11]; // RSI is column 11
                            if (rsiCell) {{
                                const rsiText = rsiCell.textContent.trim();
                                const rsiMatch = rsiText.match(/([0-9.-]+)/);
                                const rsiValue = rsiMatch ? parseFloat(rsiMatch[1]) : 0;
                                if ((minRSIValue && rsiValue < parseFloat(minRSIValue)) || 
                                    (maxRSIValue && rsiValue > parseFloat(maxRSIValue))) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply total score filter
                        if (minTotalScoreValue && showRow) {{
                            const totalScoreCell = cells[23]; // Total score is column 23
                            if (totalScoreCell) {{
                                const totalScoreText = totalScoreCell.textContent.trim();
                                const totalScoreMatch = totalScoreText.match(/([0-9]+)\/16/);
                                const totalScore = totalScoreMatch ? parseFloat(totalScoreMatch[1]) : 0;
                                if (totalScore < minTotalScore) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply trend score filter
                        if ((minTrendScoreValue || maxTrendScoreValue) && showRow) {{
                            const trendCell = cells[19]; // Trend score is column 19
                            if (trendCell) {{
                                const trendText = trendCell.textContent.trim();
                                const trendMatch = trendText.match(/([0-9.-]+)%/);
                                const trendPercent = trendMatch ? parseFloat(trendMatch[1]) : 0;
                                if ((minTrendScoreValue && trendPercent < minTrendScore) || 
                                    (maxTrendScoreValue && trendPercent > maxTrendScore)) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                        
                        // Apply ML Class filter
                        if (mlClassFilterValue && showRow) {{
                            const mlClassCell = cells[22]; // ML Class is column 22
                            if (mlClassCell) {{
                                const mlClassText = mlClassCell.textContent.trim();
                                let matchesFilter = false;
                                
                                if (mlClassFilterValue === 'drops') {{
                                    // Check for drop classes (0-2)
                                    matchesFilter = mlClassText.includes('Drop >10%') || 
                                                  mlClassText.includes('Drop 5-10%') || 
                                                  mlClassText.includes('Drop 0-5%');
                                }} else if (mlClassFilterValue === 'gains') {{
                                    // Check for gain classes (3-5)
                                    matchesFilter = mlClassText.includes('Gain 0-5%') || 
                                                  mlClassText.includes('Gain 5-10%') || 
                                                  mlClassText.includes('Gain >10%');
                                }} else {{
                                    // Check for specific class number
                                    const classLabels = {{
                                        '0': 'Drop >10%',
                                        '1': 'Drop 5-10%', 
                                        '2': 'Drop 0-5%',
                                        '3': 'Gain 0-5%',
                                        '4': 'Gain 5-10%',
                                        '5': 'Gain >10%'
                                    }};
                                    const expectedLabel = classLabels[mlClassFilterValue];
                                    matchesFilter = expectedLabel && mlClassText.includes(expectedLabel);
                                }}
                                
                                if (!matchesFilter) {{
                                    showRow = false;
                                }}
                            }}
                        }}
                    }}
                }}
                
                if (showRow) {{
                    row.style.display = '';
                    visibleCount++;
                }} else {{
                    row.style.display = 'none';
                }}
            }});
            
            // Update filter count display
            updateFilterCounts(visibleCount);
        }}
        
        function clearAdvancedFilters() {{
            document.getElementById('tickerFilter').value = '';
            document.getElementById('minRRRatio').value = '';
            document.getElementById('maxATRRatio').value = '';
            document.getElementById('minBBPosition').value = '';
            document.getElementById('maxBBPosition').value = '';
            document.getElementById('minVRI20').value = '';
            document.getElementById('maxVRI20').value = '';
            document.getElementById('minVRI10').value = '';
            document.getElementById('maxVRI10').value = '';
            document.getElementById('minCMF20').value = '';
            document.getElementById('maxCMF20').value = '';
            document.getElementById('minCMF10').value = '';
            document.getElementById('maxCMF10').value = '';
            document.getElementById('minADMomentum').value = '';
            document.getElementById('maxADMomentum').value = '';
            document.getElementById('minTotalScore').value = '';
            document.getElementById('minTrendScore').value = '';
            document.getElementById('maxTrendScore').value = '';
            document.getElementById('mlClassFilter').value = '';
            
            // Reset to current signal filter only
            filterTable(currentFilter);
        }}
        
        function updateFilterCounts(visibleCount) {{
            // Update the filter status display
            const allButtons = document.querySelectorAll('.filter-btn');
            allButtons.forEach(btn => {{
                const originalText = btn.textContent.split(' (')[0];
                const signal = btn.classList.contains('btn-buy') ? 'BUY' :
                              btn.classList.contains('btn-sell') ? 'SELL' :
                              btn.classList.contains('btn-hold') ? 'HOLD' :
                              'All';
                
                if (btn.classList.contains('active')) {{
                    btn.textContent = `${{originalText}} (${{visibleCount}})`;
                }}
            }});
            console.log('Visible rows after filtering: ' + visibleCount);
        }}
        
        function filterTable(signal) {{
            currentFilter = signal;
            
            // Update button states
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            const targetButton = document.querySelector('.btn-' + signal);
            if (targetButton) {{
                targetButton.classList.add('active');
            }}
            
            // Apply both signal and advanced filters
            applyAdvancedFilters();
        }}
        
        window.onload = function() {{
            // Set initial sort
            sortTable(14, 'ml_class');
        }};
    </script>
</head>
<body>
    <div class="header">
        <h1>📊 Stock Scanner Report</h1>
        <p>Generated on {report_time} | {total_predictions} stocks analyzed</p>
    </div>
    
    <div class="stats">
        <div class="stat">
            <h3>🚀 BUY Signals</h3>
            <p>{buy_count}</p>
        </div>
        <div class="stat">
            <h3>📉 SELL Signals</h3>
            <p>{sell_count}</p>
        </div>
        <div class="stat">
            <h3>⏸️ HOLD Signals</h3>
            <p>{hold_count}</p>
        </div>
        <div class="stat">
            <h3>🎯 Avg Confidence</h3>
            <p>{avg_confidence:.1f}%</p>
        </div>
    </div>
    
    <div class="controls">
        <h3>📋 Prediction Results</h3>
        <div class="filter-buttons">
            <button class="filter-btn btn-all active" onclick="filterTable('all')">All ({total_predictions})</button>
            <button class="filter-btn btn-buy" onclick="filterTable('BUY')">BUY ({buy_count})</button>
            <button class="filter-btn btn-sell" onclick="filterTable('SELL')">SELL ({sell_count})</button>
            <button class="filter-btn btn-hold" onclick="filterTable('hold')">HOLD ({hold_count})</button>
        </div>
        
        <div class="advanced-filters" style="margin-top: 15px; padding: 15px; background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px;">
            <h4 style="margin: 0 0 10px 0; color: #495057;">🔧 Advanced Filters</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                <div>
                    <label for="tickerFilter" style="font-weight: bold; display: block; margin-bottom: 5px;">🎯 Ticker Search:</label>
                    <input type="text" id="tickerFilter" placeholder="e.g., AAPL, MSFT" onkeyup="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minRRRatio" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Min R/R Ratio:</label>
                    <input type="number" id="minRRRatio" placeholder="e.g., 1.5" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxATRRatio" style="font-weight: bold; display: block; margin-bottom: 5px;">📈 Max ATR/Price %:</label>
                    <input type="number" id="maxATRRatio" placeholder="e.g., 5" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minBBPosition" style="font-weight: bold; display: block; margin-bottom: 5px;">🎈 Min BB Position %:</label>
                    <input type="number" id="minBBPosition" placeholder="e.g., 20" min="0" max="100" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxBBPosition" style="font-weight: bold; display: block; margin-bottom: 5px;">🎈 Max BB Position %:</label>
                    <input type="number" id="maxBBPosition" placeholder="e.g., 80" min="0" max="100" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minVRI20" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Min VRI_20:</label>
                    <input type="number" id="minVRI20" placeholder="e.g., 1.2" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxVRI20" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Max VRI_20:</label>
                    <input type="number" id="maxVRI20" placeholder="e.g., 3.0" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minVRI10" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Min VRI_10:</label>
                    <input type="number" id="minVRI10" placeholder="e.g., 1.0" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxVRI10" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Max VRI_10:</label>
                    <input type="number" id="maxVRI10" placeholder="e.g., 2.5" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minCMF20" style="font-weight: bold; display: block; margin-bottom: 5px;">💰 Min CMF_20:</label>
                    <input type="number" id="minCMF20" placeholder="e.g., -0.1" step="0.01" min="-1" max="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxCMF20" style="font-weight: bold; display: block; margin-bottom: 5px;">💰 Max CMF_20:</label>
                    <input type="number" id="maxCMF20" placeholder="e.g., 0.3" step="0.01" min="-1" max="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minCMF10" style="font-weight: bold; display: block; margin-bottom: 5px;">💰 Min CMF_10:</label>
                    <input type="number" id="minCMF10" placeholder="e.g., -0.2" step="0.01" min="-1" max="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxCMF10" style="font-weight: bold; display: block; margin-bottom: 5px;">💰 Max CMF_10:</label>
                    <input type="number" id="maxCMF10" placeholder="e.g., 0.4" step="0.01" min="-1" max="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minADMomentum" style="font-weight: bold; display: block; margin-bottom: 5px;">📈 Min A/D Momentum %:</label>
                    <input type="number" id="minADMomentum" placeholder="e.g., -5" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxADMomentum" style="font-weight: bold; display: block; margin-bottom: 5px;">📈 Max A/D Momentum %:</label>
                    <input type="number" id="maxADMomentum" placeholder="e.g., 10" step="0.1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minRSI" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Min RSI:</label>
                    <input type="number" id="minRSI" placeholder="e.g., 30" min="0" max="100" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxRSI" style="font-weight: bold; display: block; margin-bottom: 5px;">📊 Max RSI:</label>
                    <input type="number" id="maxRSI" placeholder="e.g., 70" min="0" max="100" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minTotalScore" style="font-weight: bold; display: block; margin-bottom: 5px;">⭐ Min Total Score:</label>
                    <input type="number" id="minTotalScore" placeholder="e.g., 8" min="0" max="16" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="minTrendScore" style="font-weight: bold; display: block; margin-bottom: 5px;">📈 Min Trend Score %:</label>
                    <input type="number" id="minTrendScore" placeholder="e.g., 50" min="0" max="100" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="maxTrendScore" style="font-weight: bold; display: block; margin-bottom: 5px;">📈 Max Trend Score %:</label>
                    <input type="number" id="maxTrendScore" placeholder="e.g., 90" min="0" max="100" step="1" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                </div>
                <div>
                    <label for="mlClassFilter" style="font-weight: bold; display: block; margin-bottom: 5px;">🤖 ML Class:</label>
                    <select id="mlClassFilter" onchange="applyAdvancedFilters()" style="width: 100%; padding: 5px; border: 1px solid #ced4da; border-radius: 4px;">
                        <option value="">All ML Classes</option>
                        <option value="0">Drop >10% (Severe)</option>
                        <option value="1">Drop 5-10% (Moderate)</option>
                        <option value="2">Drop 0-5% (Mild)</option>
                        <option value="3">Gain 0-5% (Mild)</option>
                        <option value="4">Gain 5-10% (Moderate)</option>
                        <option value="5">Gain >10% (Severe)</option>
                        <option value="drops">All Drops (0-2)</option>
                        <option value="gains">All Gains (3-5)</option>
                    </select>
                </div>
                <div style="display: flex; align-items: end;">
                    <button onclick="clearAdvancedFilters()" style="padding: 8px 15px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; width: 100%;">🗑️ Clear Filters</button>
                </div>
            </div>
        </div>
    </div>
    
    <table id="resultsTable">
        <thead>
            <tr>
                <th onclick="sortTable(0, 'rank')">#</th>
                <th onclick="sortTable(1, 'ticker')">Ticker</th>
                <th onclick="sortTable(2, 'signal')">Signal</th>
                <th onclick="sortTable(3, 'current_price')">Current Price</th>
                <th onclick="sortTable(4, 'predicted_price')">Predicted Price</th>
                <th onclick="sortTable(5, 'option_price')">Strike Price (OTM%)</th>
                <th onclick="sortTable(6, 'premiums')">Premiums</th>
                <th onclick="sortTable(7, 'exp_date')">Exp Date (DTE)</th>
                <th onclick="sortTable(8, 'risk_amount')">Risk</th>
                <th onclick="sortTable(9, 'reward_amount')">Reward</th>
                <th onclick="sortTable(10, 'rr_ratio')">R/R Ratio</th>
                <th onclick="sortTable(11, 'rsi')">RSI</th>
                <th onclick="sortTable(12, 'atr_ratio')">ATR/Price</th>
                <th onclick="sortTable(13, 'bb_position')">BB Position</th>
                <th onclick="sortTable(14, 'vri_20')">VRI 20</th>
                <th onclick="sortTable(15, 'vri_10')">VRI 10</th>
                <th onclick="sortTable(16, 'cmf_20')">CMF 20</th>
                <th onclick="sortTable(17, 'cmf_10')">CMF 10</th>
                <th onclick="sortTable(18, 'ad_momentum')">A/D Momentum</th>
                <th onclick="sortTable(19, 'trend_score')">Trend</th>
                <th onclick="sortTable(20, 'ml_negative')">ML Negative</th>
                <th onclick="sortTable(21, 'ml_positive')">ML Positive</th>
                <th onclick="sortTable(22, 'ml_class')">ML Class</th>
                <th onclick="sortTable(23, 'total_score')">Total Score</th>
            </tr>
        </thead>
        <tbody>"""
        
        # Sort predictions by confidence (highest first) for display
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        for i, pred in enumerate(sorted_predictions, 1):
            ticker = pred['ticker']
            signal = pred['signal']
            confidence = pred['confidence']
            current_price = pred.get('current_price', 0)
            predicted_price = pred.get('predicted_price', 0)
            price_change = pred.get('price_change', 0)
            
            # Get premiums from database only
            option_premium = pred.get('db_premiums', 0.0)  # Premiums from database only
            
            # Calculate option data for the report
            option_data = calculate_option_data(current_price, ticker)
            otm_percentage = option_data.get('otm_percentage', 0.0)
            exp_date = option_data.get('expiration_date', 'N/A')
            dte = option_data.get('dte', 0)
            strike_price = option_data.get('strike_price', 0.0)
            
            # Get technical scores
            tech_scores = pred.get('technical_scores', {})
            rsi_value = tech_scores.get('rsi_value', 0)
            trend_score = tech_scores.get('trend_score', 0)
            risk_amount = tech_scores.get('risk_amount', 0)
            reward_amount = tech_scores.get('reward_amount', 0)
            rr_ratio = tech_scores.get('risk_reward_ratio', 0)
            rr_status = tech_scores.get('rr_status', 'Unknown')
            atr_ratio = tech_scores.get('atr_to_price_ratio', 0)
            bb_position = tech_scores.get('bb_position', 0)
            
            # Get new volume and money flow indicators
            vri_20 = tech_scores.get('vri_20', 0)
            vri_10 = tech_scores.get('vri_10', 0)
            cmf_20 = tech_scores.get('cmf_20', 0)
            cmf_10 = tech_scores.get('cmf_10', 0)
            ad_momentum = tech_scores.get('ad_momentum', 0)
            
            # Get professional analysis scores
            prof_scores = pred.get('professional_scores', {})
            total_score = prof_scores.get('total_score', 0)
            trend_prof_score = prof_scores.get('trend_score', 0)
            safety_score = prof_scores.get('safety_score', 0)
            strength_score = prof_scores.get('relative_strength_score', 0)
            
            # Calculate risk and reward percentages
            risk_percentage = (risk_amount / current_price * 100) if current_price > 0 else 0
            reward_percentage = (reward_amount / current_price * 100) if current_price > 0 else 0
            
            # Get ML prediction data
            ml_prediction = pred.get('ml_prediction', {})
            ml_probabilities = ml_prediction.get('probabilities', [])
            ml_class = ml_prediction.get('class_label', '')
            ml_class_num = ml_prediction.get('model_class', -1)
            has_ml_prediction = bool(ml_prediction.get('method') == 'ML')
            
            # Calculate ML negative and positive probabilities
            if ml_probabilities and len(ml_probabilities) >= 1:
                # Get model_classes from model metadata if available
                model_classes = ml_prediction.get('model_classes')
                
                # Map probabilities to actual classes
                ml_negative = 0.0
                ml_positive = 0.0
                
                if model_classes:
                    # Use actual model_classes mapping
                    # Class mapping: 0=Drop>10%, 1=Drop5-10%, 2=Drop0-5%, 3=Gain0-5%, 4=Gain5-10%, 5=Gain>10%
                    # Negative classes: 0, 1, 2 (drops)
                    # Positive classes: 3, 4, 5 (gains)
                    
                    for i, class_num in enumerate(model_classes):
                        if i < len(ml_probabilities):
                            if class_num <= 2:  # Drop classes
                                ml_negative += ml_probabilities[i]
                            else:  # Gain classes
                                ml_positive += ml_probabilities[i]
                else:
                    # Fallback logic when model_classes not available
                    if len(ml_probabilities) == 6:
                        # Full 6-class model - assume standard order
                        ml_negative = sum(ml_probabilities[0:3])  # Classes 0,1,2 (drops)
                        ml_positive = sum(ml_probabilities[3:6])  # Classes 3,4,5 (gains)
                    else:
                        # Filtered model - use heuristic based on predicted class
                        if ml_class_num <= 2:
                            # Drop prediction - assume model has more drop classes
                            neg_count = min(3, len(ml_probabilities) - 1)  # Leave at least 1 for positive
                            ml_negative = sum(ml_probabilities[:neg_count])
                            ml_positive = sum(ml_probabilities[neg_count:])
                        else:
                            # Gain prediction - assume model has more gain classes
                            pos_count = min(3, len(ml_probabilities) - 1)  # Leave at least 1 for negative
                            ml_positive = sum(ml_probabilities[-pos_count:])
                            ml_negative = sum(ml_probabilities[:-pos_count])
                
                ml_negative_pct = f"{ml_negative:.1%}"
                ml_positive_pct = f"{ml_positive:.1%}"
            else:
                # When ML predictions are not available, provide technical analysis based estimates
                # Use signal and confidence to estimate probability distribution
                if signal == 'BUY':
                    # Strong buy signal suggests higher positive probability
                    ml_positive = 0.60 + (confidence / 100) * 0.25  # 60-85% positive
                    ml_negative = 1.0 - ml_positive
                elif signal == 'SELL':
                    # Strong sell signal suggests higher negative probability  
                    ml_negative = 0.60 + (confidence / 100) * 0.25  # 60-85% negative
                    ml_positive = 1.0 - ml_negative
                else:
                    # Hold signals suggest more balanced probabilities
                    ml_positive = 0.45 + (confidence / 100) * 0.10  # 45-55% positive
                    ml_negative = 1.0 - ml_positive
                
                ml_negative_pct = f"{ml_negative:.1%}"
                ml_positive_pct = f"{ml_positive:.1%}"
            
            # Apply CSS classes based on values
            signal_class = f"signal-{signal}"
            confidence_class = "confidence-high" if confidence >= 70 else "confidence-medium" if confidence >= 50 else "confidence-low"
            rsi_class = "rsi-oversold" if rsi_value < 30 else "rsi-overbought" if rsi_value > 70 else "rsi-neutral"
            
            # CSS classes for new indicators
            vri_20_class = "vri-high" if vri_20 > 1.5 else "vri-low" if vri_20 < 0.5 else "vri-normal"
            vri_10_class = "vri-high" if vri_10 > 1.5 else "vri-low" if vri_10 < 0.5 else "vri-normal"
            cmf_20_class = "cmf-positive" if cmf_20 > 0.1 else "cmf-negative" if cmf_20 < -0.1 else "cmf-neutral"
            cmf_10_class = "cmf-positive" if cmf_10 > 0.1 else "cmf-negative" if cmf_10 < -0.1 else "cmf-neutral"
            ad_momentum_class = "ad-momentum-positive" if ad_momentum > 0.01 else "ad-momentum-negative" if ad_momentum < -0.01 else "ad-momentum-neutral"
            
            # Price change styling
            price_change_class = "price-positive" if price_change > 0 else "price-negative" if price_change < 0 else "price-neutral"
            
            # Risk/Reward ratio styling
            if rr_ratio >= 2.0:
                rr_class = "good-ratio"
            elif rr_ratio >= 1.0:
                rr_class = "decent-ratio"
            else:
                rr_class = "poor-ratio"
            
            # ML class-based styling
            if has_ml_prediction and ml_class_num >= 0:
                ml_class_styles = {
                    0: "ml-class-drop-severe",     # Drop >10%
                    1: "ml-class-drop-moderate",   # Drop 5-10%
                    2: "ml-class-drop-mild",       # Drop 0-5%
                    3: "ml-class-gain-mild",       # Gain 0-5%
                    4: "ml-class-gain-moderate",   # Gain 5-10%
                    5: "ml-class-gain-severe"      # Gain >10%
                }
                ml_class_css = ml_class_styles.get(ml_class_num, "confidence-medium")
            else:
                # Fallback to confidence-based styling for non-ML predictions
                ml_class_css = confidence_class
            
            html_report += f"""
            <tr>
                <td><strong>{i}</strong></td>
                <td class="ticker"><a href="{STREAMLIT_URL}/?ticker={ticker}&pattern_length=60" target="_blank">{ticker}</a></td>
                <td class="{signal_class}">{signal}</td>
                <td>${current_price:.2f}</td>
                <td class="{price_change_class}">${predicted_price:.2f}<br><small>({price_change:+.1f}%)</small></td>
                <td>${strike_price:.2f}<br><small>({otm_percentage:.1f}% {'ITM' if otm_percentage < 0 else 'OTM'})</small></td>
                <td>${format_number(option_premium)}</td>
                <td>{exp_date}<br><small>({dte} DTE)</small></td>
                <td class="risk-amount">${risk_amount:.2f}<br><small>({risk_percentage:.1f}%)</small></td>
                <td class="reward-amount">${reward_amount:.2f}<br><small>({reward_percentage:.1f}%)</small></td>
                <td class="{rr_class}">{rr_ratio:.2f} ({rr_status})</td>
                <td class="{rsi_class}">{rsi_value:.1f}</td>
                <td>{atr_ratio:.1%}</td>
                <td>{bb_position:.1%}</td>
                <td class="{vri_20_class}">{vri_20:.2f}</td>
                <td class="{vri_10_class}">{vri_10:.2f}</td>
                <td class="{cmf_20_class}">{cmf_20:.3f}</td>
                <td class="{cmf_10_class}">{cmf_10:.3f}</td>
                <td class="{ad_momentum_class}">{ad_momentum:.1%}</td>
                <td>{trend_score:.1%}</td>
                <td><span class="ml-negative">{ml_negative_pct}</span></td>
                <td><span class="ml-positive">{ml_positive_pct}</span></td>
                <td class="{ml_class_css}">
                    {confidence:.1f}%
                    {f'<br><small class="ml-class-small">{ml_class}</small>' if has_ml_prediction and ml_class else ''}
                </td>
                <td class="{'good-ratio' if total_score >= 12 else 'decent-ratio' if total_score >= 8 else 'poor-ratio'}">
                    <strong>{total_score}/16</strong>
                    <br><small style="color: #666;">T:{trend_prof_score}/4 S:{safety_score}/6 R:{strength_score}/6</small>
                </td>
            </tr>"""
        
        html_report += """
        </tbody>
    </table>
    
    <div class="footer">
        <p><strong>Stock Scanner Report</strong> - Technical analysis powered by Yahoo Finance data</p>
        <p>🔍 Scanner uses RSI, MACD, Bollinger Bands, Volume Ratio (VRI), Chaikin Money Flow (CMF), A/D Momentum, and Trend analysis for predictions</p>
        <p>💰 Risk/Reward ratios calculated using 20-day support/resistance levels</p>
        <p>📈 Price predictions based on signal strength and confidence levels</p>
        <p>🤖 ML predictions use trained LightGBM models for 6-class price change forecasting</p>
    </div>
</body>
</html>"""
        
        # Save HTML report
        html_filename = f"scanner_report_{timestamp}.html"
        html_path = os.path.join(output_dir, html_filename)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        with open(f"{output_dir}/report.html", 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"HTML report generated: {html_path}")
        return html_path


def load_ticker_list_from_file(file_path: str) -> List[str]:
    """Load ticker list from a text file (one ticker per line)"""
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading ticker list from {file_path}: {e}")
        return FALLBACK_TICKERS


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Stock Scanner and Prediction Tool")
    parser.add_argument("--predict-only", action="store_true", 
                       help="Run in prediction-only mode (enables ML predictions by default)")
    parser.add_argument("--date-filter", default="latest",
                       help="Date filter: maintained for compatibility (not used with Yahoo data)")
    parser.add_argument("--max-tickers", type=int,
                       help="Maximum number of tickers to process")
    parser.add_argument("--ticker-file", 
                       help="Path to file containing ticker list (one per line)")
    parser.add_argument("--tickers", nargs="+",
                       help="Specific tickers to scan (space-separated)")
    parser.add_argument("--no-database", action="store_true",
                       help="Don't use database for ticker list, use fallback instead")
    parser.add_argument("--output", 
                       help="Output file path for predictions")
    parser.add_argument("--output-dir", default="scanner_reports",
                       help="Output directory for reports (default: scanner_reports)")
    parser.add_argument("--no-html", action="store_true",
                       help="Skip HTML report generation")
    parser.add_argument("--use-ml", action="store_true",
                       help="Use LightGBM ML predictions instead of confidence-based predictions (requires trained models)")
    parser.add_argument("--no-ml", action="store_true",
                       help="Disable ML predictions even in predict-only mode")
    parser.add_argument("--rebuild-models", action="store_true",
                       help="Force rebuild of ML models even if existing models are found")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # Filter arguments
    parser.add_argument("--filter-tickers", nargs="+",
                       help="Only include specific tickers in results (space-separated)")
    parser.add_argument("--min-rr-ratio", type=float,
                       help="Minimum risk/reward ratio filter")
    parser.add_argument("--max-rr-ratio", type=float,
                       help="Maximum risk/reward ratio filter")
    parser.add_argument("--max-atr-price-ratio", type=float, default=0.05,
                       help="Maximum ATR/Price ratio filter (default: 0.05 for 5%%)")
    parser.add_argument("--min-bb-position", type=float,
                       help="Minimum Bollinger Band position filter (0.0-1.0)")
    parser.add_argument("--max-bb-position", type=float,
                       help="Maximum Bollinger Band position filter (0.0-1.0)")
    
    # Volume Ratio Indicator (VRI) filters
    parser.add_argument("--min-vri-20", type=float,
                       help="Minimum VRI_20 filter (volume ratio over 20 periods)")
    parser.add_argument("--max-vri-20", type=float,
                       help="Maximum VRI_20 filter (volume ratio over 20 periods)")
    parser.add_argument("--min-vri-10", type=float,
                       help="Minimum VRI_10 filter (volume ratio over 10 periods)")
    parser.add_argument("--max-vri-10", type=float,
                       help="Maximum VRI_10 filter (volume ratio over 10 periods)")
    
    # Chaikin Money Flow (CMF) filters
    parser.add_argument("--min-cmf-20", type=float,
                       help="Minimum CMF_20 filter (Chaikin Money Flow over 20 periods, range: -1 to +1)")
    parser.add_argument("--max-cmf-20", type=float,
                       help="Maximum CMF_20 filter (Chaikin Money Flow over 20 periods, range: -1 to +1)")
    parser.add_argument("--min-cmf-10", type=float,
                       help="Minimum CMF_10 filter (Chaikin Money Flow over 10 periods, range: -1 to +1)")
    parser.add_argument("--max-cmf-10", type=float,
                       help="Maximum CMF_10 filter (Chaikin Money Flow over 10 periods, range: -1 to +1)")
    
    # Accumulation/Distribution Line filters
    parser.add_argument("--min-ad-momentum", type=float,
                       help="Minimum A/D Line momentum filter (percentage change over 5 periods)")
    parser.add_argument("--max-ad-momentum", type=float,
                       help="Maximum A/D Line momentum filter (percentage change over 5 periods)")
    
    # RSI filters
    parser.add_argument("--min-rsi", type=float,
                       help="Minimum RSI filter (range: 0-100, typical: 30-70)")
    parser.add_argument("--max-rsi", type=float,
                       help="Maximum RSI filter (range: 0-100, typical: 30-70)")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Determine ticker list source
        ticker_list = None
        use_database = not args.no_database
        
        if args.ticker_file:
            ticker_list = load_ticker_list_from_file(args.ticker_file)
        elif args.tickers:
            ticker_list = [ticker.upper() for ticker in args.tickers]
            logger.info(f"Using command-line tickers: {ticker_list}")
        
        # Enable ML by default when called from watchlist.py (predict-only mode)
        # unless explicitly disabled with --no-ml
        use_ml = (args.use_ml or args.predict_only) and not args.no_ml
        
        # Initialize scanner with specified tickers or database/fallback
        scanner = StockScanner(ticker_list, use_database, use_ml_predictions=use_ml, rebuild_models=args.rebuild_models)
        
        # Build filters dictionary
        filters = {}
        if args.filter_tickers:
            filters['ticker'] = [t.upper() for t in args.filter_tickers]
        if args.min_rr_ratio is not None:
            filters['min_rr_ratio'] = args.min_rr_ratio
        if args.max_rr_ratio is not None:
            filters['max_rr_ratio'] = args.max_rr_ratio
        if args.max_atr_price_ratio is not None:
            filters['max_atr_price_ratio'] = args.max_atr_price_ratio
        if args.min_bb_position is not None:
            filters['min_bb_position'] = args.min_bb_position
        if args.max_bb_position is not None:
            filters['max_bb_position'] = args.max_bb_position
        
        # VRI filters
        if args.min_vri_20 is not None:
            filters['min_vri_20'] = args.min_vri_20
        if args.max_vri_20 is not None:
            filters['max_vri_20'] = args.max_vri_20
        if args.min_vri_10 is not None:
            filters['min_vri_10'] = args.min_vri_10
        if args.max_vri_10 is not None:
            filters['max_vri_10'] = args.max_vri_10
        
        # CMF filters
        if args.min_cmf_20 is not None:
            filters['min_cmf_20'] = args.min_cmf_20
        if args.max_cmf_20 is not None:
            filters['max_cmf_20'] = args.max_cmf_20
        if args.min_cmf_10 is not None:
            filters['min_cmf_10'] = args.min_cmf_10
        if args.max_cmf_10 is not None:
            filters['max_cmf_10'] = args.max_cmf_10
        
        # A/D Line filters
        if args.min_ad_momentum is not None:
            filters['min_ad_momentum'] = args.min_ad_momentum
        if args.max_ad_momentum is not None:
            filters['max_ad_momentum'] = args.max_ad_momentum
        
        # RSI filters
        if args.min_rsi is not None:
            filters['min_rsi'] = args.min_rsi
        if args.max_rsi is not None:
            filters['max_rsi'] = args.max_rsi
        
        if filters:
            logger.info(f"Applied filters: {filters}")
        
        # Run scan
        predictions = scanner.scan_stocks(
            date_filter=args.date_filter,
            max_tickers=args.max_tickers,
            predict_only=args.predict_only,
            filters=filters if filters else None
        )
        
        # Save results
        output_file = scanner.save_predictions(predictions, args.output, args.output_dir)
        
        # Generate HTML report (unless disabled)
        html_report_path = None
        if not args.no_html and predictions:
            html_report_path = scanner.generate_html_report(predictions, args.output_dir)
        
        # Print summary
        if predictions:
            print(f"\n📊 SCAN RESULTS SUMMARY")
            print("=" * 60)
            print(f"Total predictions: {len(predictions)}")
            
            buy_signals = [p for p in predictions if p['signal'] == 'BUY']
            sell_signals = [p for p in predictions if p['signal'] == 'SELL']
            hold_signals = [p for p in predictions if p['signal'] in ['HOLD', 'WEAK_HOLD']]
            
            print(f"🚀 BUY signals: {len(buy_signals)}")
            print(f"📉 SELL signals: {len(sell_signals)}")
            print(f"⏸️  HOLD signals: {len(hold_signals)}")
            
            if buy_signals:
                print(f"\n🎯 TOP BUY RECOMMENDATIONS:")
                for pred in buy_signals[:5]:  # Top 5
                    print(f"   {pred['ticker']}: {pred['confidence']:.1f}% confidence")
            
            print(f"\n📁 Reports generated:")
            print(f"   📄 CSV: {output_file}")
            if html_report_path:
                print(f"   🌐 HTML: {html_report_path}")
            
            if html_report_path:
                print(f"\n💡 Open the HTML report in your browser for interactive analysis!")
        else:
            print("❌ No predictions generated")
    
    except Exception as e:
        logger.error(f"Scanner failed: {e}")
        raise


if __name__ == "__main__":
    main()