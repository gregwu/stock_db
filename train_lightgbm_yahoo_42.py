#!/usr/bin/env python3
"""
train_lightgbm_yahoo_42.py - LightGBM Training using Yahoo Finance Data with 42 Features

This script trains a LightGBM model using data fetched directly from Yahoo Finance,
with the complete 42-feature set including all derived features.

Usage:
    python train_lightgbm_yahoo_42.py <ticker>
    
Example:
    python train_lightgbm_yahoo_42.py STOK
"""

import pandas as pd
import lightgbm as lgb
import yfinance as yf
import numpy as np
import os
import sys
import warnings
import argparse
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import utility functions for technical indicators
from util import calculate_all_technical_indicators
from config import features

def fetch_yahoo_data(ticker, period='max'):
    """
    Fetch stock data from Yahoo Finance and calculate technical indicators.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period ('1y', '2y', '5y', 'max', etc.)
        
    Returns:
        DataFrame with stock data and technical indicators
    """
    
    print(f"📊 Fetching data for {ticker} from Yahoo Finance...")
    
    try:
        # Create ticker object and fetch data
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period)
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return pd.DataFrame()
        
        # Convert to expected format for technical indicators
        df = data.reset_index()
        
        # Rename columns to match expected format
        column_mapping = {
            'Date': 'DATE',
            'Open': 'OPEN',
            'High': 'HIGH', 
            'Low': 'LOW',
            'Close': 'CLOSE',
            'Volume': 'VOL'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add required columns for compatibility
        df['PER'] = 1  # Daily data
        df['TIME'] = '0000'  # Default time
        df['OPENINT'] = 0  # Open interest (not available for stocks)
        
        # Sort by date
        df = df.sort_values('DATE').reset_index(drop=True)
        
        print(f"✅ Fetched {len(df)} rows of data")
        print(f"📅 Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        
        # Calculate technical indicators
        df = calculate_all_technical_indicators(df)
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def create_prediction_labels(df):
    """
    Create 6-class prediction labels based on next day price change.
    
    Args:
        df: DataFrame with CLOSE prices
        
    Returns:
        DataFrame with added label columns
    """
    
    print("🏷️ Creating prediction labels...")
    
    # Calculate next day percentage change
    df['change_pct_1d'] = (df['CLOSE'].shift(-1) - df['CLOSE']) / df['CLOSE'] * 100
    
    # Create 6-class labels based on percentage change
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
    
    df['label_class'] = df['change_pct_1d'].apply(classify_change)
    
    # Remove rows with null labels (last row due to shift)
    df = df.dropna(subset=['label_class'])
    df['label_class'] = df['label_class'].astype(int)
    
    print(f"✅ Created labels for {len(df)} samples")
    print("📊 Class distribution:")
    class_labels = {
        0: "Drop >10%",
        1: "Drop 5-10%", 
        2: "Drop 0-5%",
        3: "Gain 0-5%",
        4: "Gain 5-10%",
        5: "Gain >10%"
    }
    
    for class_id in sorted(df['label_class'].unique()):
        count = (df['label_class'] == class_id).sum()
        percentage = (count / len(df)) * 100
        print(f"  Class {class_id} ({class_labels[class_id]}): {count} samples ({percentage:.1f}%)")
    
    return df

def create_derived_features(df):
    """
    Create all 14 derived features to match the 42-feature set.
    
    Args:
        df: DataFrame with base technical indicators
        
    Returns:
        DataFrame with added derived features
    """
    
    print("🔧 Creating 14 derived features...")
    
    # Convert columns to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # Add missing core features that might be named differently
    if 'price_to_sma_5' not in df.columns and 'sma_5' in df.columns:
        df['price_to_sma_5'] = df['close'] / df['sma_5']
    
    if 'price_to_sma_20' not in df.columns and 'sma_20' in df.columns:
        df['price_to_sma_20'] = df['close'] / df['sma_20']
        
    if 'sma_5_slope' not in df.columns and 'sma_5' in df.columns:
        df['sma_5_slope'] = df['sma_5'].pct_change()
        
    if 'sma_20_slope' not in df.columns and 'sma_20' in df.columns:
        df['sma_20_slope'] = df['sma_20'].pct_change()
        
    if 'macd_signal' not in df.columns and 'macd_signal_line' in df.columns:
        df['macd_signal'] = df['macd_signal_line']
        
    if 'macd_hist' not in df.columns and 'macd_histogram' in df.columns:
        df['macd_hist'] = df['macd_histogram']
        
    if 'volume_ratio' not in df.columns and 'vol' in df.columns and 'volume_sma' in df.columns:
        df['volume_ratio'] = np.where(df['volume_sma'] != 0, df['vol'] / df['volume_sma'], 0)
        
    if 'resistance_distance' not in df.columns and 'resistance_20' in df.columns:
        df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close']
    
    # === Create all 14 derived features ===
    
    # Relative Volatility Ratios
    df['bb_to_volatility'] = np.where(df['price_volatility'] != 0, 
                                      df['bb_std'] / df['price_volatility'], 0)
    
    df['bb_width_to_volatility20'] = np.where(df['volatility_20'] != 0, 
                                              df['bb_width'] / df['volatility_20'], 0)
    
    df['price_vol_to_vol20'] = np.where(df['volatility_20'] != 0, 
                                        df['price_volatility'] / df['volatility_20'], 0)
    
    # MACD Divergence Strength
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['macd_trend_strength'] = df['macd_hist'] * df['macd_momentum']
    
    # SMA Cross Features
    df['sma_diff_5_20'] = df['price_to_sma_20'] - df['price_to_sma_5']
    df['sma_slope_diff'] = df['sma_5_slope'] - df['sma_20_slope']
    
    # Volume + Price Fusion
    df['vol_price_momentum'] = df['volume_ratio'] * df['price_change_abs']
    df['obv_macd_interact'] = df['obv_momentum'] * df['macd_momentum']
    
    # Stochastic Spread
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    
    # Support/Resistance Distance Ratios
    df['sr_ratio'] = np.where(df['resistance_distance'] != 0, 
                              df['support_distance'] / df['resistance_distance'], 0)
    
    df['move_to_support_ratio'] = np.where(df['support_distance'] != 0, 
                                           df['price_change_abs'] / df['support_distance'], 0)
    
    # Trend Persistence
    df['trend_consistency_2d'] = df['price_change_lag_1'] * df['price_change_lag_2']
    df['trend_consistency_3d'] = df['price_change_lag_1'] * df['price_change_lag_3']
    
    print("✅ All 14 derived features created successfully!")
    
    return df

def prepare_features(df):
    """
    Prepare the complete 42-feature set matching config.py
    
    Args:
        df: DataFrame with technical indicators and derived features
        
    Returns:
        list: Available features from the 42-feature set
    """
    
    print("🔧 Preparing 42-feature set...")
    
    # Create derived features
    df = create_derived_features(df)
    
    # Define the complete 42-feature set from config.py (lowercase)
    target_features = [f.lower() for f in features]
    
    # Check which features are available
    available_features = []
    missing_features = []
    
    for feature in target_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    print(f"✅ Using {len(available_features)} out of {len(target_features)} features (target: 42)")
    
    return available_features

def train_lightgbm_model(ticker, period='max', use_smote=True, use_class_weights=True):
    """
    Train LightGBM model using Yahoo Finance data with 42 features.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period to fetch
        use_smote: Whether to apply SMOTE for class balancing
        use_class_weights: Whether to use class weights
        
    Returns:
        tuple: (model, results_dict)
    """
    
    print(f"💡 Training LightGBM model for {ticker} with 42 features")
    print("=" * 60)
    
    # Fetch data
    df = fetch_yahoo_data(ticker, period)
    
    if df.empty:
        print("❌ No data fetched. Exiting...")
        return None, None
    
    # Create labels
    df = create_prediction_labels(df)
    
    if len(df) == 0:
        print("❌ No data after creating labels. Exiting...")
        return None, None
    
    # Prepare features
    available_features = prepare_features(df)
    
    # Check for null values and clean data
    print("🧹 Cleaning data...")
    initial_rows = len(df)
    
    # Check null counts for available features
    null_counts = df[available_features + ['label_class']].isnull().sum()
    features_with_nulls = null_counts[null_counts > 0]
    
    if len(features_with_nulls) > 0:
        print(f"🔍 Features with null values:")
        for feature, count in features_with_nulls.items():
            print(f"  {feature}: {count} nulls")
    
    # Remove rows with null values
    df_clean = df.dropna(subset=available_features + ['label_class'])
    final_rows = len(df_clean)
    
    print(f"📊 Data after cleaning: {final_rows} rows (removed {initial_rows - final_rows} rows with nulls)")
    
    if final_rows < 100:
        print("❌ Insufficient data after cleaning. Need at least 100 samples.")
        return None, None
    
    # Prepare X and y
    X = df_clean[available_features]
    y = df_clean['label_class'].astype(int)
    
    print(f"📊 Final dataset: {len(X)} samples, {len(available_features)} features")
    
    # Chronological split (80% train, 20% test)
    print("📅 Splitting data chronologically...")
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    train_start = df_clean['date'].iloc[0]
    train_end = df_clean['date'].iloc[split_idx - 1]
    test_start = df_clean['date'].iloc[split_idx]
    test_end = df_clean['date'].iloc[-1]
    
    print(f"Training set: {train_start} to {train_end} ({len(X_train)} samples)")
    print(f"Test set: {test_start} to {test_end} ({len(X_test)} samples)")
    
    # Apply SMOTE if enabled
    if use_smote:
        print("\n🔄 Applying SMOTE to training data...")
        
        # Clean data for SMOTE
        X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
        nan_mask = X_train_clean.isnull().any(axis=1)
        
        if nan_mask.sum() > 0:
            print(f"⚠️ Removing {nan_mask.sum()} rows with NaN/inf values for SMOTE")
            X_train_clean = X_train_clean[~nan_mask]
            y_train_clean = y_train[~nan_mask]
        else:
            y_train_clean = y_train
        
        # Check class distribution
        class_counts = y_train_clean.value_counts()
        min_class_size = class_counts.min()
        
        if min_class_size >= 6:  # SMOTE requires at least 6 samples
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_clean, y_train_clean)
            print(f"✅ SMOTE applied: {len(y_train_clean)} → {len(y_train_balanced)} samples")
            X_train, y_train = X_train_balanced, y_train_balanced
        else:
            print(f"⚠️ Skipping SMOTE: minimum class size ({min_class_size}) too small")
    
    # Compute class weights
    class_weights = None
    if use_class_weights:
        print("⚖️ Computing class weights...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Class weights: {class_weight_dict}")
    
    # Train LightGBM model
    print("\n💡 Training LightGBM model...")
    model_params = {
        'objective': 'multiclass',
        'num_class': 6,
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbosity': -1,
        'class_weight': 'balanced' if use_class_weights else None
    }
    
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    # Class labels for reporting
    class_labels = {
        0: "Drop >10%",
        1: "Drop 5-10%", 
        2: "Drop 0-5%",
        3: "Gain 0-5%",
        4: "Gain 5-10%",
        5: "Gain >10%"
    }
    
    print("\nClassification Report:")
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [class_labels[i] for i in unique_classes]
    print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names, zero_division=0))
    
    # Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # Prepare results
    results = {
        'ticker': ticker,
        'period': period,
        'accuracy': accuracy,
        'total_samples': len(df_clean),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': available_features,
        'class_distribution': y.value_counts().to_dict(),
        'feature_importance': feature_importance.to_dict('records'),
        'model_params': model_params,
        'use_smote': use_smote,
        'use_class_weights': use_class_weights
    }
    
    return model, results

def save_model_and_results(model, results, ticker):
    """
    Save the trained model and results.
    
    Args:
        model: Trained LightGBM model
        results: Results dictionary
        ticker: Stock ticker symbol
    """
    
    print(f"\n💾 Saving model and results for {ticker}...")
    
    # Create model directory
    model_dir = f"models/{ticker}_yahoo"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = f"{model_dir}/lightgbm_model_yahoo.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save feature importance
    feature_importance_df = pd.DataFrame(results['feature_importance'])
    feature_importance_path = f"{model_dir}/feature_importance_yahoo.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"✅ Feature importance saved to {feature_importance_path}")
    
    # Save feature list
    features_path = f"{model_dir}/features_used_yahoo.json"
    with open(features_path, 'w') as f:
        json.dump(results['features_used'], f, indent=2)
    print(f"✅ Feature list saved to {features_path}")
    
    # Save training summary
    summary_path = f"{model_dir}/training_summary_yahoo.txt"
    with open(summary_path, 'w') as f:
        f.write(f"LIGHTGBM TRAINING SUMMARY - YAHOO FINANCE DATA (42 FEATURES)\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Ticker: {results['ticker']}\n")
        f.write(f"Data Period: {results['period']}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset:\n")
        f.write(f"- Total samples: {results['total_samples']}\n")
        f.write(f"- Training samples: {results['train_samples']}\n")
        f.write(f"- Test samples: {results['test_samples']}\n")
        f.write(f"- Features used: {len(results['features_used'])}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Use SMOTE: {results['use_smote']}\n")
        f.write(f"- Use Class Weights: {results['use_class_weights']}\n\n")
        f.write(f"Performance:\n")
        f.write(f"- Test Accuracy: {results['accuracy']:.4f}\n\n")
        f.write(f"Top 5 Most Important Features:\n")
        for i, feature_info in enumerate(results['feature_importance'][:5]):
            f.write(f"- {feature_info['feature']}: {feature_info['importance']:.4f}\n")
    
    print(f"✅ Training summary saved to {summary_path}")

def main():
    """Main function to handle command line arguments and run training."""
    
    if len(sys.argv) < 2:
        print("Usage: python train_lightgbm_yahoo_42.py <ticker>")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    print(f"🚀 LightGBM Training Script - Yahoo Finance Data (42 Features)")
    print(f"📊 Ticker: {ticker}")
    print()
    
    # Train model
    model, results = train_lightgbm_model(ticker)
    
    if model is not None:
        # Save model and results
        save_model_and_results(model, results, ticker)
        
        print(f"\n🎉 Training completed successfully for {ticker}!")
        print(f"📈 Final accuracy: {results['accuracy']:.4f}")
        print(f"📁 Model saved in: models/{ticker}_yahoo/")
        print(f"🔧 Features used: {len(results['features_used'])}/42")
    else:
        print("❌ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()