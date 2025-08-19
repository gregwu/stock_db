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
from util import calculate_all_technical_indicators

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
    
    def __init__(self, ticker_list: List[str] = None, use_database: bool = True, use_ml_predictions: bool = False):
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
            data = yf.download(ticker, period=period, progress=False)
            
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
            
            # Calculate all technical indicators
            df = calculate_all_technical_indicators(data)
            
            # Convert columns to lowercase for compatibility
            df.columns = [col.lower() for col in df.columns]
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Ensure we have the basic OHLCV columns in both cases
            df['OPEN'] = df['open']
            df['HIGH'] = df['high']
            df['LOW'] = df['low']
            df['CLOSE'] = df['close']
            df['VOL'] = df['vol']
            
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
        if 'RSI_14' not in df.columns:
            df['RSI_14'] = TechnicalIndicators.rsi(df['CLOSE'], 14)
        
        if 'BB_Middle' not in df.columns:
            df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = TechnicalIndicators.bollinger_bands(df['CLOSE'])
        
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
        
        # Volume analysis
        if 'vol' in latest and 'vol' in df.columns:
            current_volume = latest['vol']
            avg_volume = df['vol'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            scores['volume_ratio'] = volume_ratio
            scores['volume_score'] = min(volume_ratio / 2.0, 1.0)  # Cap at 1.0
        
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
            # This assumes the model was trained with the same feature set
            from config import features
            
            # Get latest row for prediction
            latest_data = df.iloc[-1:].copy()
            
            # Select features that exist in the data
            available_features = [f for f in features if f.lower() in latest_data.columns]
            
            if len(available_features) < 10:  # Minimum feature threshold
                logger.warning(f"Insufficient features for ML prediction: {len(available_features)} available")
                return None
            
            # Prepare feature matrix
            X = latest_data[available_features].values.reshape(1, -1)
            
            # Handle any NaN values
            if np.isnan(X).any():
                X = np.nan_to_num(X)
            
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
            logger.error(f"ML prediction failed for {ticker}: {e}")
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
            'technical_scores': scores
        }
        
        return prediction
    
    def scan_stocks(self, date_filter: str = "latest", max_tickers: Optional[int] = None, 
                   predict_only: bool = False) -> List[Dict]:
        """
        Scan stocks and generate predictions
        
        Args:
            date_filter: Date filter for ticker selection
            max_tickers: Maximum number of tickers to scan
            predict_only: Only generate predictions, don't run full scan
            
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
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
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
        
        function filterTable(signal) {{
            currentFilter = signal;
            const rows = document.querySelectorAll('#resultsTable tbody tr');
            
            // Update button states
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            const targetButton = document.querySelector('.btn-' + signal);
            if (targetButton) {{
                targetButton.classList.add('active');
            }}
            
            rows.forEach(row => {{
                const signalCell = row.getElementsByTagName('td')[2];
                if (signalCell) {{
                    const signalText = signalCell.textContent.trim();
                    if (signal === 'all' || signalText === signal || 
                        (signal === 'hold' && (signalText === 'HOLD' || signalText === 'WEAK_HOLD'))) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }});
        }}
        
        window.onload = function() {{
            // Set initial sort
            sortTable(12, 'ml_class');
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
    </div>
    
    <table id="resultsTable">
        <thead>
            <tr>
                <th onclick="sortTable(0, 'rank')">#</th>
                <th onclick="sortTable(1, 'ticker')">Ticker</th>
                <th onclick="sortTable(2, 'signal')">Signal</th>
                <th onclick="sortTable(3, 'current_price')">Current Price</th>
                <th onclick="sortTable(4, 'predicted_price')">Predicted Price</th>
                <th onclick="sortTable(5, 'risk_amount')">Risk</th>
                <th onclick="sortTable(6, 'reward_amount')">Reward</th>
                <th onclick="sortTable(7, 'rr_ratio')">R/R Ratio</th>
                <th onclick="sortTable(8, 'rsi')">RSI</th>
                <th onclick="sortTable(9, 'trend_score')">Trend</th>
                <th onclick="sortTable(10, 'ml_negative')">ML Negative</th>
                <th onclick="sortTable(11, 'ml_positive')">ML Positive</th>
                <th onclick="sortTable(12, 'ml_class')">ML Class</th>
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
            
            # Get technical scores
            tech_scores = pred.get('technical_scores', {})
            rsi_value = tech_scores.get('rsi_value', 0)
            trend_score = tech_scores.get('trend_score', 0)
            risk_amount = tech_scores.get('risk_amount', 0)
            reward_amount = tech_scores.get('reward_amount', 0)
            rr_ratio = tech_scores.get('risk_reward_ratio', 0)
            rr_status = tech_scores.get('rr_status', 'Unknown')
            
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
                ml_negative_pct = "N/A"
                ml_positive_pct = "N/A"
            
            # Apply CSS classes based on values
            signal_class = f"signal-{signal}"
            confidence_class = "confidence-high" if confidence >= 70 else "confidence-medium" if confidence >= 50 else "confidence-low"
            rsi_class = "rsi-oversold" if rsi_value < 30 else "rsi-overbought" if rsi_value > 70 else "rsi-neutral"
            
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
                <td class="ticker"><a href="/ypredict/?ticker={ticker}&pattern_length=60" target="_blank">{ticker}</a></td>
                <td class="{signal_class}">{signal}</td>
                <td>${current_price:.2f}</td>
                <td class="{price_change_class}">${predicted_price:.2f}<br><small>({price_change:+.1f}%)</small></td>
                <td class="risk-amount">${risk_amount:.2f}<br><small>({risk_percentage:.1f}%)</small></td>
                <td class="reward-amount">${reward_amount:.2f}<br><small>({reward_percentage:.1f}%)</small></td>
                <td class="{rr_class}">{rr_ratio:.2f} ({rr_status})</td>
                <td class="{rsi_class}">{rsi_value:.1f}</td>
                <td>{trend_score:.1%}</td>
                <td><span class="ml-negative">{ml_negative_pct}</span></td>
                <td><span class="ml-positive">{ml_positive_pct}</span></td>
                <td class="{ml_class_css}">
                    {confidence:.1f}%
                    {f'<br><small class="ml-class-small">{ml_class}</small>' if has_ml_prediction and ml_class else ''}
                </td>
            </tr>"""
        
        html_report += """
        </tbody>
    </table>
    
    <div class="footer">
        <p><strong>Stock Scanner Report</strong> - Technical analysis powered by Yahoo Finance data</p>
        <p>🔍 Scanner uses RSI, MACD, Bollinger Bands, Volume, and Trend analysis for predictions</p>
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
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
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
        scanner = StockScanner(ticker_list, use_database, use_ml_predictions=use_ml)
        
        # Run scan
        predictions = scanner.scan_stocks(
            date_filter=args.date_filter,
            max_tickers=args.max_tickers,
            predict_only=args.predict_only
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