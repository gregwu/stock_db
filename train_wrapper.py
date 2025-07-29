#!/usr/bin/env python3
"""
train_wrapper.py - Wrapper for training LightGBM models with proper data format

This wrapper ensures data compatibility between the Streamlit app and training script
by preparing data in the format expected by train_lightgbm_yahoo_42.py
"""

import sys
import tempfile
import os
import pandas as pd
from pathlib import Path

# Import the data loading function from fractal_predict
sys.path.append('.')
from fractal_predict import load_stock_data
from train_lightgbm_yahoo_42 import train_lightgbm_model, save_model_and_results

def prepare_training_data(ticker):
    """
    Prepare training data with uppercase columns for compatibility.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        tuple: (success, message)
    """
    print(f"🔧 Preparing training data for {ticker}...")
    
    try:
        # Load data with uppercase columns for training compatibility
        df, error = load_stock_data(ticker, keep_uppercase=True)
        
        if error:
            return False, f"Error loading data: {error}"
        
        if df is None or len(df) < 100:
            return False, "Insufficient data for training"
        
        print(f"✅ Data prepared: {len(df)} rows with {len(df.columns)} columns")
        print(f"📊 Column format check - Has 'CLOSE': {'CLOSE' in df.columns}")
        
        return True, "Data preparation successful"
        
    except Exception as e:
        return False, f"Data preparation failed: {str(e)}"

def main():
    """Main function to handle training with proper data format."""
    
    if len(sys.argv) < 2:
        print("Usage: python train_wrapper.py <ticker>")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    print(f"🚀 Training Wrapper for {ticker}")
    print("=" * 50)
    
    # Step 1: Prepare data
    success, message = prepare_training_data(ticker)
    
    if not success:
        print(f"❌ {message}")
        sys.exit(1)
    
    print(f"✅ {message}")
    
    # Step 2: Train model using the existing training function
    print(f"\n💡 Starting model training for {ticker}...")
    
    try:
        model, results = train_lightgbm_model(ticker)
        
        if model is not None and results is not None:
            # Step 3: Save model and results
            save_model_and_results(model, results, ticker)
            
            print(f"\n🎉 Training completed successfully for {ticker}!")
            print(f"📈 Final accuracy: {results['accuracy']:.4f}")
            print(f"📁 Model saved in: models/{ticker}_yahoo/")
            print(f"🔧 Features used: {len(results['features_used'])}")
            
        else:
            print("❌ Training failed - no model returned")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()