#!/usr/bin/env python3
"""
Generate Cached Stock Data
========================
Create cached_stock_data.pkl with stock data from the database
for use in training scripts.

This script:
1. Loads stock data from PostgreSQL database (with pre-calculated indicators)
2. Saves processed data to cached_stock_data.pkl
3. Provides data quality validation and statistics

Note: Technical indicators are already calculated by savedb.py pipeline.
"""

import pandas as pd
import numpy as np
import os
import time
from sqlalchemy import create_engine
from dotenv import load_dotenv
import psutil

# Load environment variables
load_dotenv()

def get_memory_status():
    """Get current memory status."""
    mem = psutil.virtual_memory()
    return {
        'available_gb': mem.available / (1024**3),
        'percent_used': mem.percent,
        'total_gb': mem.total / (1024**3)
    }

def load_sql(filename):
    """Load SQL query from file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"SQL file not found: {filename}")
    
    with open(filename, 'r') as file:
        return file.read()

def save_data_in_chunks(df, cache_file, chunk_size=10000):
    """Save DataFrame to pickle file in chunks to reduce memory usage."""
    print(f"💾 Saving data in chunks of {chunk_size:,} rows...")
    
    # Split the dataframe into chunks
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    print(f"📊 Split data into {len(chunks)} chunks")
    
    # Save chunks to temporary files first
    temp_files = []
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        temp_file = f"{cache_file}_chunk_{i}.pkl"
        chunk.to_pickle(temp_file)
        temp_files.append(temp_file)
        print(f"   ✅ Chunk {i+1}/{len(chunks)} saved ({len(chunk):,} rows)")
        
        # Clear chunk from memory
        del chunk
        import gc
        gc.collect()
    
    # Combine chunks into final file
    print(f"🔗 Combining chunks into final cache file...")
    combined_chunks = []
    
    for i, temp_file in enumerate(temp_files):
        chunk = pd.read_pickle(temp_file)
        combined_chunks.append(chunk)
        print(f"   📥 Loaded chunk {i+1}/{len(temp_files)}")
        
        # Remove temporary file
        os.remove(temp_file)
    
    # Concatenate and save final file
    final_df = pd.concat(combined_chunks, ignore_index=True)
    final_df.to_pickle(cache_file)
    
    # Clean up
    del combined_chunks, final_df
    import gc
    gc.collect()
    
    save_time = time.time() - start_time
    print(f"✅ Chunked save completed in {save_time:.1f} seconds")
    
    return save_time

def load_data_in_chunks(engine, query, chunk_size=10000):
    """Load data from database in chunks to reduce memory usage."""
    print(f"📊 Loading data in chunks of {chunk_size:,} rows...")
    
    # First, get total count
    count_query = f"SELECT COUNT(*) FROM ({query}) as subquery"
    try:
        total_rows = pd.read_sql(count_query, engine).iloc[0, 0]
        print(f"📊 Total rows to load: {total_rows:,}")
    except:
        print("📊 Could not determine total row count, proceeding with chunked loading...")
        total_rows = None
    
    # Load data in chunks
    chunks = []
    chunk_num = 0
    
    for chunk in pd.read_sql(query, engine, chunksize=chunk_size):
        chunk_num += 1
        chunks.append(chunk)
        
        if total_rows:
            progress = min(chunk_num * chunk_size, total_rows)
            print(f"   📥 Loaded chunk {chunk_num} ({len(chunk):,} rows, {progress:,}/{total_rows:,} total)")
        else:
            print(f"   📥 Loaded chunk {chunk_num} ({len(chunk):,} rows)")
        
        # Memory check after each chunk
        mem_status = get_memory_status()
        if mem_status['available_gb'] < 0.5:
            print(f"⚠️  Low memory warning: {mem_status['available_gb']:.1f}GB available")
            break
    
    # Combine chunks
    print(f"🔗 Combining {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    # Clean up chunk list
    del chunks
    import gc
    gc.collect()
    
    print(f"✅ Combined data: {len(df):,} rows, {len(df.columns)} columns")
    return df

def validate_data_quality(df):
    """Validate the quality of the processed data."""
    print("\n🔍 Validating data quality...")
    
    # Basic stats
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📊 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check for essential columns
    essential_cols = ['close_price', 'high_price', 'low_price', 'open_price', 'volume', 'ticker', 'date']
    missing_essential = [col for col in essential_cols if col not in df.columns]
    if missing_essential:
        print(f"❌ Missing essential columns: {missing_essential}")
        return False
    
    # Check for technical indicator columns (should be pre-calculated)
    indicator_cols = [col for col in df.columns if any(x in col.lower() for x in ['sma', 'rsi', 'macd', 'bb_', 'ema'])]
    print(f"📊 Found {len(indicator_cols)} technical indicator columns")
    if len(indicator_cols) > 0:
        print(f"   Examples: {indicator_cols[:5]}...")
    
    # Check for label columns
    label_cols = [col for col in df.columns if col.startswith('label_') or 'close_' in col]
    print(f"📊 Found {len(label_cols)} label/target columns: {label_cols[:5]}...")
    
    # Check for excessive NaN values in key columns
    for col in essential_cols:
        if col in df.columns:
            nan_pct = df[col].isnull().sum() / len(df) * 100
            if nan_pct > 10:
                print(f"⚠️  Column '{col}' has {nan_pct:.1f}% NaN values")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024*1024)
    print(f"💾 Memory usage: {memory_mb:.1f} MB")
    
    return True

def save_data_hdf5(df, cache_file, chunk_size=10000):
    """Save DataFrame to HDF5 format for better memory efficiency with large datasets."""
    try:
        import tables  # PyTables for HDF5 support
    except ImportError:
        print("⚠️  PyTables not available, falling back to chunked pickle...")
        return save_data_in_chunks(df, cache_file, chunk_size)
    
    print(f"💾 Saving data to HDF5 format (more memory efficient)...")
    
    # Convert to HDF5 format
    hdf5_file = cache_file.replace('.pkl', '.h5')
    start_time = time.time()
    
    # Save to HDF5 with compression
    df.to_hdf(hdf5_file, key='stock_data', mode='w', complib='zlib', complevel=9)
    
    save_time = time.time() - start_time
    print(f"✅ HDF5 save completed in {save_time:.1f} seconds")
    
    # Also save as pickle for compatibility
    print(f"💾 Creating pickle copy for compatibility...")
    df.to_pickle(cache_file)
    
    # Compare file sizes
    hdf5_size = os.path.getsize(hdf5_file) / (1024*1024)
    pickle_size = os.path.getsize(cache_file) / (1024*1024)
    
    print(f"📁 HDF5 file size: {hdf5_size:.1f} MB")
    print(f"📁 Pickle file size: {pickle_size:.1f} MB")
    print(f"💾 Space savings: {((pickle_size - hdf5_size) / pickle_size * 100):.1f}%")
    
    return save_time

def main(chunk_size=10000):
    """Main function to generate cached stock data with chunked processing."""
    print("🚀 Generating Cached Stock Data (Memory-Optimized)")
    print("="*50)
    
    # Check memory
    mem_status = get_memory_status()
    print(f"💾 Initial memory: {mem_status['available_gb']:.1f}GB available ({mem_status['percent_used']:.1f}% used)")
    print(f"📊 Chunk size: {chunk_size:,} rows")
    
    # Database configuration
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD")
    }
    
    DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    
    try:
        # Load data from database
        print("\n📊 Loading data from PostgreSQL database...")
        engine = create_engine(DATABASE_URL)
        
        # Load SQL query
        query = load_sql('multilabel_cls.sql')
        print(f"📄 Loaded SQL query from multilabel_cls.sql")
        
        # Execute query and load data in chunks
        start_time = time.time()
        df = load_data_in_chunks(engine, query, chunk_size)
        load_time = time.time() - start_time
        
        print(f"✅ Loaded {len(df):,} rows in {load_time:.1f} seconds")
        print(f"📊 Final columns: {len(df.columns)}")
        
        # Check memory after loading
        mem_status = get_memory_status()
        print(f"💾 Memory after loading: {mem_status['available_gb']:.1f}GB available")
        
        if mem_status['available_gb'] < 0.5:
            print("❌ Insufficient memory for processing")
            return False
        
        # Validate data quality
        if not validate_data_quality(df):
            print("❌ Data quality validation failed")
            return False
        
        # Save to cache using the most appropriate method based on size
        cache_file = 'cached_stock_data.pkl'
        print(f"\n💾 Saving data to {cache_file}...")
        
        # Choose saving strategy based on dataset size
        if len(df) > 100000:  # Very large datasets (>100k rows)
            print(f"📊 Large dataset detected ({len(df):,} rows)")
            try:
                save_time = save_data_hdf5(df, cache_file, chunk_size)
            except:
                print("⚠️  HDF5 save failed, using chunked pickle...")
                save_time = save_data_in_chunks(df, cache_file, chunk_size)
        elif len(df) > 50000:  # Medium datasets (50k-100k rows)
            print(f"📊 Medium dataset detected ({len(df):,} rows)")
            save_time = save_data_in_chunks(df, cache_file, chunk_size)
        else:  # Smaller datasets (<50k rows)
            print(f"📊 Small dataset detected ({len(df):,} rows)")
            start_time = time.time()
            df.to_pickle(cache_file)
            save_time = time.time() - start_time
            print(f"✅ Standard save completed in {save_time:.1f} seconds")
        
        # Final stats
        file_size_mb = os.path.getsize(cache_file) / (1024*1024)
        print(f"📁 Cache file size: {file_size_mb:.1f} MB")
        
        # Final memory check
        mem_status = get_memory_status()
        print(f"💾 Final memory: {mem_status['available_gb']:.1f}GB available")
        
        print(f"\n🎉 Cache generation complete!")
        print(f"📁 File: {cache_file}")
        print(f"📊 Shape: {df.shape}")
        print(f"💾 Size: {file_size_mb:.1f} MB")
        print(f"📊 Technical indicators: Pre-calculated by savedb.py pipeline")
        print(f"⚡ Memory-optimized: Used {chunk_size:,} row chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating cache: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    # Default chunk size
    chunk_size = 10000
    
    # Allow command line argument for chunk size
    if len(sys.argv) > 1:
        try:
            chunk_size = int(sys.argv[1])
            print(f"📊 Using custom chunk size: {chunk_size:,}")
        except ValueError:
            print(f"⚠️  Invalid chunk size '{sys.argv[1]}', using default: {chunk_size:,}")
    
    success = main(chunk_size)
    
    if success:
        print("\n🏆 Cache generation successful!")
        print("🎯 You can now use this cached data in your training scripts")
        print("📝 The cache includes pre-calculated technical indicators from the database")
        print(f"⚡ Memory-optimized with {chunk_size:,} row chunks")
        print("\n💡 Usage in training scripts:")
        print("   df = pd.read_pickle('cached_stock_data.pkl')")
        print("\n💡 Usage with custom chunk size:")
        print("   python generate_cache.py 5000  # Use 5,000 row chunks")
    else:
        print("\n❌ Cache generation failed!")
        print("🔧 Please check your database connection and multilabel_cls.sql file")
