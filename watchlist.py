#!/usr/bin/env python3
"""
Table Management - Streamlit Web App

A user-friendly web interface for managing the database table.
Provides all CRUD operations, statistics, and data management features
through an intuitive web interface.

Run with: streamlit run watchlist.py
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import os
import io
import json
import re
import subprocess
import threading
import webbrowser
import time
from PIL import Image

# Try to import OCR library
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration
from config import DB_CONFIG, DIRECTORIES, PYTHON_CONFIG, FILE_PATTERNS, WEB_CONFIG, PREDICTION_REPORTS

# Database configuration is now imported from config.py

def ensure_directories():
    """Ensure all configured directories exist"""
    for dir_name, dir_path in DIRECTORIES.items():
        # Skip work_dir as it should already exist
        if dir_name == 'work_dir':
            continue
        
        # Create relative paths within work_dir for logs and some reports
        if dir_name in ['logs']:
            full_path = os.path.join(DIRECTORIES['work_dir'], dir_path)
        else:
            full_path = dir_path
            
        try:
            os.makedirs(full_path, exist_ok=True)
        except Exception as e:
            # Don't fail silently, but log the issue
            import streamlit as st
            st.warning(f"⚠️ Could not create directory {full_path}: {e}")

def get_python_executable():
    """Get the Python executable path from config with smart detection"""
    # First try the configured executable
    python_exe = PYTHON_CONFIG['executable']
    if os.path.exists(python_exe):
        return python_exe
    
    # Try common virtual environment paths
    potential_paths = [
        'predictor/bin/python',  # Local development
        'venv/bin/python',       # Standard venv
        'env/bin/python',        # Alternative venv name
        '.venv/bin/python',      # Hidden venv
        'bin/python',            # Direct bin folder
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # Fall back to system python
    return PYTHON_CONFIG['fallback_executable']

def get_web_url_for_report(report_filename):
    """Generate web URL for accessing HTML reports"""
    if WEB_CONFIG['enable_web_links']:
        base_url = WEB_CONFIG['base_url'].rstrip('/')
        web_path = WEB_CONFIG['reports_web_path'].strip('/')
        return f"{base_url}/{web_path}/{report_filename}"
    return None

def create_report_link_or_download(html_file, report_type="HTML Report"):
    """Create either a web link or download button for reports"""
    if not os.path.exists(html_file):
        st.warning(f"⚠️ {report_type} not found: {html_file}")
        return
    
    filename = os.path.basename(html_file)
    web_url = get_web_url_for_report(filename)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if web_url and WEB_CONFIG['enable_web_links']:
            st.markdown(f"🌐 **[Open {report_type}]({web_url})**")
        else:
            st.info(f"📄 {report_type}: {filename}")
    
    with col2:
        if WEB_CONFIG['enable_file_download']:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.download_button(
                    label=f"📥 Download {report_type}",
                    data=html_content,
                    file_name=filename,
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"❌ Error reading {report_type}: {e}")

class StreamlitSeekingAlphaManager:
    """Streamlit-optimized manager for table operations"""
    
    def __init__(self):
        # Fix database config key name
        self.db_config = DB_CONFIG.copy()
        if 'dbname' in self.db_config:
            self.db_config['database'] = self.db_config.pop('dbname')
    
    @st.cache_resource
    def get_connection(_self):
        """Get cached database connection"""
        return psycopg2.connect(**_self.db_config)
    
    def create_table(self):
        """Create the seekingalpha table if it doesn't exist"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS seekingalpha (
                            id SERIAL PRIMARY KEY,
                            ticker VARCHAR(20) NOT NULL,
                            date_added DATE NOT NULL DEFAULT CURRENT_DATE,
                            pdf_source VARCHAR(255),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(ticker, date_added)
                        )
                    """)
                    
                    # Create index for better performance
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_seekingalpha_ticker_date 
                        ON seekingalpha(ticker, date_added)
                    """)
                    
                    conn.commit()
                    return True
        except Exception as e:
            st.error(f"❌ Error creating table: {e}")
            return False
    
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_records(_self, date_filter: str = None, ticker_filter: str = None, 
                   limit: int = None) -> pd.DataFrame:
        """
        Get records as DataFrame with optional filters
        """
        try:
            with _self.get_connection() as conn:
                query = "SELECT * FROM seekingalpha"
                params = []
                conditions = []
                
                if date_filter == 'today':
                    conditions.append("date_added = CURRENT_DATE")
                elif date_filter and date_filter != 'all':
                    conditions.append("date_added = %s")
                    params.append(date_filter)
                
                if ticker_filter:
                    conditions.append("ticker ILIKE %s")
                    params.append(f"%{ticker_filter}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY date_added DESC, ticker ASC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=params)
                return df
                
        except Exception as e:
            st.error(f"❌ Error fetching records: {e}")
            return pd.DataFrame()
    
    def add_ticker(self, ticker: str, pdf_source: str = None, 
                   date_added: date = None) -> bool:
        """Add a single ticker to the database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO seekingalpha (ticker, date_added, pdf_source)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (ticker, date_added) DO NOTHING
                    """, (ticker.upper().strip(), date_added or datetime.now().date(), pdf_source))
                    
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    else:
                        return False  # Already exists
                        
        except psycopg2.IntegrityError as e:
            if "duplicate key value violates unique constraint" in str(e) and "pkey" in str(e):
                # Fix sequence issue and retry once
                self._fix_sequence()
                try:
                    with self.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO seekingalpha (ticker, date_added, pdf_source)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (ticker, date_added) DO NOTHING
                            """, (ticker.upper().strip(), date_added or datetime.now().date(), pdf_source))
                            
                            if cur.rowcount > 0:
                                conn.commit()
                                return True
                            else:
                                return False
                except Exception as retry_e:
                    st.error(f"❌ Error adding ticker after sequence fix: {retry_e}")
                    return False
            else:
                st.error(f"❌ Error adding ticker: {e}")
                return False
        except Exception as e:
            st.error(f"❌ Error adding ticker: {e}")
            return False
    
    def delete_record(self, record_id: int) -> bool:
        """Delete a record by ID"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM seekingalpha WHERE id = %s", (record_id,))
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    return False
        except Exception as e:
            st.error(f"❌ Error deleting record: {e}")
            return False
    
    def update_record(self, record_id: int, ticker: str = None, 
                     pdf_source: str = None, date_added: date = None) -> bool:
        """Update an existing record"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    updates = []
                    params = []
                    
                    if ticker:
                        updates.append("ticker = %s")
                        params.append(ticker.upper())
                    if pdf_source is not None:  # Allow empty string
                        updates.append("pdf_source = %s")
                        params.append(pdf_source if pdf_source else None)
                    if date_added:
                        updates.append("date_added = %s")
                        params.append(date_added)
                    
                    if not updates:
                        return False
                    
                    params.append(record_id)
                    query = f"UPDATE seekingalpha SET {', '.join(updates)} WHERE id = %s"
                    
                    cur.execute(query, params)
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    return False
                    
        except Exception as e:
            st.error(f"❌ Error updating record: {e}")
            return False
    
    @st.cache_data(ttl=60)
    def get_statistics(_self) -> Dict:
        """Get comprehensive statistics"""
        try:
            with _self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    stats = {}
                    
                    # Total records
                    cur.execute("SELECT COUNT(*) as total FROM seekingalpha")
                    stats['total_records'] = cur.fetchone()['total']
                    
                    # Unique tickers
                    cur.execute("SELECT COUNT(DISTINCT ticker) as unique_tickers FROM seekingalpha")
                    stats['unique_tickers'] = cur.fetchone()['unique_tickers']
                    
                    # Records by date (last 14 days)
                    cur.execute("""
                        SELECT date_added, COUNT(*) as count 
                        FROM seekingalpha 
                        WHERE date_added >= CURRENT_DATE - INTERVAL '14 days'
                        GROUP BY date_added 
                        ORDER BY date_added DESC
                    """)
                    stats['recent_dates'] = [dict(row) for row in cur.fetchall()]
                    
                    # Top PDF sources
                    cur.execute("""
                        SELECT 
                            COALESCE(pdf_source, 'No source') as pdf_source, 
                            COUNT(*) as count 
                        FROM seekingalpha 
                        GROUP BY pdf_source 
                        ORDER BY count DESC 
                        LIMIT 10
                    """)
                    stats['top_sources'] = [dict(row) for row in cur.fetchall()]
                    
                    # Most frequent tickers
                    cur.execute("""
                        SELECT ticker, COUNT(*) as count 
                        FROM seekingalpha 
                        GROUP BY ticker 
                        ORDER BY count DESC 
                        LIMIT 10
                    """)
                    stats['top_tickers'] = [dict(row) for row in cur.fetchall()]
                    
                    # Today's stats
                    cur.execute("SELECT COUNT(*) as today_count FROM seekingalpha WHERE date_added = CURRENT_DATE")
                    stats['today_count'] = cur.fetchone()['today_count']
                    
                    return stats
                    
        except Exception as e:
            st.error(f"❌ Error getting statistics: {e}")
            return {}
    
    def cleanup_old_records(self, days: int) -> int:
        """Remove records older than specified days"""
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM seekingalpha WHERE date_added < %s", (cutoff_date,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    return deleted_count
        except Exception as e:
            st.error(f"❌ Error during cleanup: {e}")
            return 0
    
    def _fix_sequence(self):
        """Fix the auto-increment sequence for the ID column"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get the maximum ID from the table
                    cur.execute("SELECT COALESCE(MAX(id), 0) FROM seekingalpha")
                    max_id = cur.fetchone()[0]
                    
                    # Set the sequence to max_id + 1
                    cur.execute(f"SELECT setval('seekingalpha_id_seq', {max_id + 1})")
                    conn.commit()
                    
        except Exception as e:
            st.warning(f"⚠️ Could not fix sequence: {e}")

# Initialize the manager
@st.cache_resource
def get_manager():
    return StreamlitSeekingAlphaManager()

def extract_symbols_from_image(image_file) -> List[str]:
    """Extract stock symbols from an uploaded image using OCR"""
    if not OCR_AVAILABLE:
        st.error("❌ OCR libraries not available. Please install: `pip install pytesseract` and `brew install tesseract` (macOS)")
        return []
    
    try:
        # Open the image
        image = Image.open(image_file)
        
        # Perform OCR to extract text
        try:
            text = pytesseract.image_to_string(image)
        except Exception as ocr_error:
            st.error(f"❌ OCR Error: {ocr_error}")
            st.info("💡 Make sure Tesseract is installed: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)")
            return []
        
        # Extract potential stock symbols from text
        symbols = extract_symbols_from_text(text)
        
        return symbols
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return []

def extract_symbols_from_text(text: str) -> List[str]:
    """Extract stock symbols from text using regex patterns"""
    symbols = set()
    
    # Common stock symbol patterns
    patterns = [
        # Standard 1-5 letter symbols (most common)
        r'\b[A-Z]{1,5}\b',
        # Symbols with numbers (like BRK.A, BRK.B)
        r'\b[A-Z]{1,4}\.[A-Z]\b',
        # Symbols with hyphens
        r'\b[A-Z]{1,4}-[A-Z]\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        symbols.update(matches)
    
    # Filter out common words that aren't stock symbols
    common_words = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR',
        'HAD', 'BUT', 'HAS', 'HIS', 'HIM', 'HIT', 'HOT', 'HOW', 'ITS', 'LET', 'MAY', 'NEW', 'NOW',
        'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'GET', 'LET', 'MAN', 'OFF', 'PUT', 'SAY', 'SHE',
        'TOO', 'USE', 'WIN', 'YES', 'YET', 'BIG', 'BOX', 'CAR', 'CUT', 'DOG', 'EAT', 'END', 'FAR',
        'FEW', 'GOT', 'GUN', 'JOB', 'LOT', 'OWN', 'RUN', 'SET', 'SIT', 'TOP', 'TRY', 'WAR', 'WAY',
        'STOCK', 'PRICE', 'MARKET', 'TRADE', 'BUY', 'SELL', 'HOLD', 'SHARE', 'VOLUME', 'HIGH', 'LOW',
        'OPEN', 'CLOSE', 'PERCENT', 'CHANGE', 'NASDAQ', 'NYSE', 'DOW', 'INDEX', 'FUND', 'ETF'
    }
    
    # Keep only likely stock symbols (1-5 chars, not common words)
    filtered_symbols = [
        symbol for symbol in symbols 
        if 1 <= len(symbol.replace('.', '').replace('-', '')) <= 5 
        and symbol not in common_words
        and not symbol.isdigit()  # Remove pure numbers
    ]
    
    return sorted(list(set(filtered_symbols)))

def run_predictions(date_filter: str = "latest", max_tickers: int = None, 
                   background: bool = True) -> Dict:
    """
    Run scanner.py predictions on the database tickers
    
    Args:
        date_filter: Date filter for predictions ("latest", "today", specific date)
        max_tickers: Maximum number of tickers to process
        background: Whether to run in background thread
        
    Returns:
        Dictionary with success status and results
    """
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Python executable path from config
        python_exe = get_python_executable()
        
        # Build command
        cmd = [
            python_exe, 
            "scanner.py", 
            "--predict-only",
            "--date-filter", date_filter
        ]
        
        if max_tickers:
            cmd.extend(["--max-tickers", str(max_tickers)])
        
        # Determine the correct working directory
        work_dir = DIRECTORIES['work_dir']
        
        # Check if scanner.py exists in current directory or work_dir
        scanner_script = "scanner.py"
        if not os.path.exists(os.path.join(work_dir, scanner_script)):
            # If not in work_dir, check if it's in current directory
            if os.path.exists(scanner_script):
                work_dir = "."
            else:
                # Last resort: check common locations
                for potential_dir in [".", "predictor", "../"]:
                    if os.path.exists(os.path.join(potential_dir, scanner_script)):
                        work_dir = potential_dir
                        break
        
        # Create log file path with timestamp using config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Fix: logs directory should be relative to current directory, not work_dir
        log_file = os.path.join(
            DIRECTORIES['logs'], 
            f"{FILE_PATTERNS['prediction_log_prefix']}{timestamp}{FILE_PATTERNS['log_extension']}"
        )
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if background:
            # Run in background and return immediately
            def run_scanner():
                try:
                    # Redirect stdout and stderr to log file
                    with open(log_file, 'w') as f:
                        f.write(f"=== Prediction Log ===\n")
                        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Command: {' '.join(cmd)}\n")
                        f.write(f"Working Directory: {work_dir}\n")
                        f.write("=" * 50 + "\n\n")
                        f.flush()
                        
                        result = subprocess.run(
                            cmd, 
                            cwd=work_dir,
                            stdout=f,
                            stderr=subprocess.STDOUT,  # Merge stderr with stdout
                            text=True, 
                            timeout=1800  # 30 minutes timeout
                        )
                    
                    # Store results in session state
                    st.session_state.prediction_result = {
                        'success': result.returncode == 0,
                        'output': f"Logged to: {log_file}",
                        'error': None if result.returncode == 0 else f"Process failed with exit code {result.returncode}",
                        'command': ' '.join(cmd),
                        'timestamp': datetime.now().isoformat(),
                        'log_file': log_file
                    }
                except Exception as e:
                    st.session_state.prediction_result = {
                        'success': False,
                        'output': '',
                        'error': str(e),
                        'command': ' '.join(cmd),
                        'timestamp': datetime.now().isoformat(),
                        'log_file': log_file
                    }
            
            # Start background thread
            thread = threading.Thread(target=run_scanner)
            thread.daemon = True
            thread.start()
            
            return {
                'success': True,
                'message': 'Predictions started in background',
                'command': ' '.join(cmd),
                'log_file': log_file
            }
        else:
            # Run synchronously and wait for results
            with open(log_file, 'w') as f:
                f.write(f"=== Prediction Log ===\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Working Directory: {work_dir}\n")
                f.write("=" * 50 + "\n\n")
                f.flush()
                
                result = subprocess.run(
                    cmd, 
                    cwd=work_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,  # Merge stderr with stdout
                    text=True, 
                    timeout=1800  # 30 minutes timeout
                )
            
            return {
                'success': result.returncode == 0,
                'output': f"Logged to: {log_file}",
                'error': result.stderr if result.returncode != 0 else None,
                'command': ' '.join(cmd),
                'returncode': result.returncode,
                'log_file': log_file
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'command': ' '.join(cmd) if 'cmd' in locals() else 'unknown',
            'log_file': log_file if 'log_file' in locals() else None
        }

def get_log_tail(log_file: str, lines: int = 50) -> str:
    """
    Get the last N lines from a log file
    
    Args:
        log_file: Path to the log file
        lines: Number of lines to retrieve from the end
        
    Returns:
        String containing the last N lines of the log file
    """
    try:
        if not os.path.exists(log_file):
            return f"Log file not found: {log_file}"
        
        # Use tail command for efficiency on large files
        result = subprocess.run(
            ['tail', '-n', str(lines), log_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            # Fallback to Python implementation
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
                
    except Exception as e:
        return f"Error reading log file: {e}"

def get_log_size(log_file: str) -> str:
    """Get human-readable file size"""
    try:
        if not os.path.exists(log_file):
            return "0 B"
        
        size = os.path.getsize(log_file)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "Unknown"

def is_process_running(log_file: str) -> bool:
    """
    Check if prediction process is still running by looking for recent log updates
    """
    try:
        if not os.path.exists(log_file):
            return False
        
        # Check if file was modified in the last 60 seconds (more generous)
        mod_time = os.path.getmtime(log_file)
        current_time = datetime.now().timestamp()
        
        # Also check if the log contains a completion marker
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if any(marker in content.lower() for marker in [
                    'completed successfully', 'process completed', 'prediction process completed',
                    'error:', 'failed:', 'exception:', 'traceback'
                ]):
                    return False
        except:
            pass
        
        return (current_time - mod_time) < 60
    except:
        return False

def get_process_status(log_file: str) -> str:
    """
    Get detailed process status from log file
    """
    try:
        if not os.path.exists(log_file):
            return "❓ Log file not found"
        
        with open(log_file, 'r') as f:
            content = f.read().lower()
        
        # Check for completion indicators
        if 'completed successfully' in content or 'prediction process completed' in content:
            return "✅ Completed Successfully"
        elif any(error in content for error in ['error:', 'failed:', 'exception:', 'traceback']):
            return "❌ Failed with Error"
        elif 'processing' in content or 'starting' in content:
            if is_process_running(log_file):
                return "🔄 Running"
            else:
                return "⏸️ Stopped/Stalled"
        else:
            return "📝 Starting"
            
    except Exception as e:
        return f"❓ Unknown ({e})"

def show_prediction_controls(added_count: int = 0):
    """
    Show prediction controls after successful ticker addition
    
    Args:
        added_count: Number of tickers that were just added
    """
    if added_count > 0:
        st.success(f"✅ Successfully added {added_count} tickers to database!")
        
        st.divider()
        st.subheader("🔮 Run Predictions")
        st.markdown(f"""
        **{added_count} new tickers** have been added to the database. 
        You can now run predictions on these tickers using the scanner.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_date_filter = st.selectbox(
                "Date Filter",
                ["latest", "today"],
                help="Which tickers to predict on"
            )
            
        with col2:
            pred_max_tickers = st.number_input(
                "Max Tickers", 
                min_value=1, 
                value=min(added_count, 10),
                help="Number of tickers to process (no upper limit)"
            )
            
        with col3:
            pred_background = st.checkbox(
                "Run in Background", 
                value=True,
                help="Run predictions in background (recommended)"
            )
        
        col_run, col_status = st.columns([1, 3])
        
        with col_run:
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner("Starting predictions..."):
                    result = run_predictions(
                        date_filter=pred_date_filter,
                        max_tickers=pred_max_tickers,
                        background=pred_background
                    )
                    
                    if result['success']:
                        if pred_background:
                            st.success("✅ Predictions started in background!")
                            st.info("💡 Check the status below or refresh the page to see results")
                            
                            # Show log file info if available
                            if result.get('log_file'):
                                st.info(f"📄 **Log file:** `{os.path.basename(result['log_file'])}`")
                        else:
                            st.success("✅ Predictions completed!")
                            
                            # Show log file info if available
                            if result.get('log_file'):
                                st.info(f"📄 **Log file:** `{os.path.basename(result['log_file'])}`")
                                
                                # Option to view log immediately
                                if st.button("� View Log", key="view_sync_log"):
                                    st.session_state.viewing_log = result['log_file']
                                    st.rerun()
                    else:
                        st.error(f"❌ Failed to start predictions: {result.get('error', 'Unknown error')}")
        
        with col_status:
            # Show prediction status if available
            if 'prediction_result' in st.session_state:
                pred_result = st.session_state.prediction_result
                
                if pred_result['success']:
                    st.success("✅ Background predictions completed!")
                    
                    # Check for reports directory from config
                    reports_dir = DIRECTORIES['reports']
                    if os.path.exists(reports_dir):
                        html_file = os.path.join(reports_dir, FILE_PATTERNS['report_html'])
                        json_file = os.path.join(reports_dir, FILE_PATTERNS['report_json'])
                        csv_file = os.path.join(reports_dir, FILE_PATTERNS['report_csv'])
                        
                        st.markdown("**📁 Generated Reports:**")
                        if os.path.exists(html_file):
                            create_report_link_or_download(html_file, "HTML Report")
                        if os.path.exists(json_file):
                            filename = os.path.basename(json_file)
                            with open(json_file, 'r') as f:
                                json_content = f.read()
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_content,
                                file_name=filename,
                                mime="application/json"
                            )
                        if os.path.exists(csv_file):
                            df = pd.read_csv(csv_file)
                            st.download_button(
                                label="� Download CSV Report",
                                data=df.to_csv(index=False),
                                file_name=os.path.basename(csv_file),
                                mime="text/csv"
                            )
                else:
                    st.error("❌ Background predictions failed!")
                    
                with st.expander("📝 Show Prediction Details"):
                    st.code(f"Command: {pred_result['command']}")
                    st.code(f"Timestamp: {pred_result['timestamp']}")
                    
                    # Show log file if available
                    if pred_result.get('log_file'):
                        st.code(f"Log file: {pred_result['log_file']}")
                        
                        # Quick log preview
                        if os.path.exists(pred_result['log_file']):
                            log_preview = get_log_tail(pred_result['log_file'], 10)
                            st.text_area("Log Preview (last 10 lines)", log_preview, height=150)
                        else:
                            st.warning("📄 Log file not found")
                    
                    if pred_result.get('output'):
                        st.text_area("Output", pred_result['output'], height=200)
                    if pred_result.get('error'):
                        st.text_area("Error", pred_result['error'], height=100)
                
                # Clear button
                if st.button("🔄 Clear Status"):
                    if 'prediction_result' in st.session_state:
                        del st.session_state.prediction_result
                    st.rerun()

def main():
    st.set_page_config(
        page_title="Table Manager",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ensure configured directories exist
    ensure_directories()
    
    st.title("📊 Table Manager")
    st.markdown("*Comprehensive database management for ticker records*")
    
    manager = get_manager()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Pages",
        ["📋 View Records", "➕ Add Records", "🔮 Run Predictions", "📊 Statistics", "🔧 Management", "📤 Export/Import"],
        label_visibility="collapsed"
    )
    
    # Prediction Reports Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Prediction Reports")
    st.sidebar.markdown(f"""
    **🔍 Predictions:**  
    [📈 View HTML Report]({PREDICTION_REPORTS['seekingalpha_report']})
    
    **📈 Stock Predictions:**  
    [📊 View Stock Report]({PREDICTION_REPORTS['stock_predictions_report']})
    """)
    
    # Create table if needed
    if not hasattr(st.session_state, 'table_created'):
        if manager.create_table():
            st.session_state.table_created = True
    
    # Main content based on selected page
    if page == "📋 View Records":
        view_records_page(manager)
    elif page == "➕ Add Records":
        add_records_page(manager)
    elif page == "🔮 Run Predictions":
        predictions_page(manager)
    elif page == "📊 Statistics":
        statistics_page(manager)
    elif page == "🔧 Management":
        management_page(manager)
    elif page == "📤 Export/Import":
        export_import_page(manager)

def view_records_page(manager):
    """Page for viewing and filtering records with inline editing and delete buttons"""
    st.header("📋 View & Edit Records")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        date_filter = st.selectbox(
            "Date Filter",
            ["all", "today"] + [(datetime.now().date() - timedelta(days=i)).strftime("%Y-%m-%d") 
                               for i in range(1, 8)]
        )
    
    with col2:
        ticker_filter = st.text_input("Ticker Filter", placeholder="e.g., AAPL")
    
    with col3:
        limit = st.number_input("Limit Results", min_value=0, max_value=1000, value=50)
    
    with col4:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Get and display records
    df = manager.get_records(
        date_filter=date_filter if date_filter != "all" else None,
        ticker_filter=ticker_filter if ticker_filter else None,
        limit=limit if limit > 0 else None
    )
    
    if not df.empty:
        # Initialize session state for sorting and changes
        if 'table_sort_by' not in st.session_state:
            st.session_state.table_sort_by = 'date_added'
        if 'table_sort_order' not in st.session_state:
            st.session_state.table_sort_order = 'desc'
        if 'pending_updates' not in st.session_state:
            st.session_state.pending_updates = {}
        if 'pending_deletes' not in st.session_state:
            st.session_state.pending_deletes = set()
        
        # Apply sorting to dataframe
        sort_by = st.session_state.table_sort_by
        sort_order = st.session_state.table_sort_order
        ascending = sort_order == "asc"
        
        if sort_by in df.columns:
            # Handle different data types for sorting
            if sort_by in ['date_added', 'created_at']:
                df = df.sort_values(by=sort_by, ascending=ascending, na_position='last').reset_index(drop=True)
            elif sort_by == 'ticker':
                df = df.sort_values(by=sort_by, ascending=ascending, key=lambda x: x.str.upper()).reset_index(drop=True)
            else:
                df = df.sort_values(by=sort_by, ascending=ascending, na_position='last').reset_index(drop=True)
        
        # Show current sort status
        sort_emoji = "🔽" if sort_order == "desc" else "🔼"
        st.subheader(f"📊 Found {len(df)} record(s) - Sorted by **{sort_by.replace('_', ' ').title()}** {sort_emoji}")
        
        # Display editable table with delete buttons
        st.markdown("### 📝 Interactive Table (Click Headers to Sort, Fields to Edit)")
        
        # Table header with clickable sorting buttons
        header_cols = st.columns([1, 2, 2, 4, 3, 2])
        
        # Sort indicators and button handlers
        def get_sort_indicator(column):
            if st.session_state.table_sort_by == column:
                return ' 🔽' if st.session_state.table_sort_order == 'desc' else ' 🔼'
            return ''
        
        def handle_sort_click(column):
            if st.session_state.table_sort_by == column:
                # Toggle order if same column
                st.session_state.table_sort_order = 'asc' if st.session_state.table_sort_order == 'desc' else 'desc'
            else:
                # New column, default to desc
                st.session_state.table_sort_by = column
                st.session_state.table_sort_order = 'desc'
        
        with header_cols[0]:
            if st.button(f"**ID**{get_sort_indicator('id')}", key="sort_id", help="Click to sort by ID", use_container_width=True):
                handle_sort_click('id')
                st.rerun()
            
        with header_cols[1]:
            if st.button(f"**Ticker**{get_sort_indicator('ticker')}", key="sort_ticker", help="Click to sort by Ticker", use_container_width=True):
                handle_sort_click('ticker')
                st.rerun()
                
        with header_cols[2]:
            if st.button(f"**Date Added**{get_sort_indicator('date_added')}", key="sort_date", help="Click to sort by Date", use_container_width=True):
                handle_sort_click('date_added')
                st.rerun()
                
        with header_cols[3]:
            if st.button(f"**PDF Source**{get_sort_indicator('pdf_source')}", key="sort_source", help="Click to sort by PDF Source", use_container_width=True):
                handle_sort_click('pdf_source')
                st.rerun()
                
        with header_cols[4]:
            if st.button(f"**Created At**{get_sort_indicator('created_at')}", key="sort_created", help="Click to sort by Created At", use_container_width=True):
                handle_sort_click('created_at')
                st.rerun()
                
        with header_cols[5]:
            col_action1, col_action2 = st.columns([1, 1])
            with col_action1:
                st.markdown("**Actions**")
            with col_action2:
                if st.button("🔄", key="reset_sort", help="Reset to default sort (Date Added desc)", use_container_width=True):
                    st.session_state.table_sort_by = 'date_added'
                    st.session_state.table_sort_order = 'desc'
                    st.rerun()
        
        st.divider()
        
        # Track if any changes were made
        changes_made = False
        
        # Display each record as an editable row
        for idx, record in df.iterrows():
            record_id = record['id']
            
            # Skip if marked for deletion
            if record_id in st.session_state.pending_deletes:
                continue
            
            # Create columns for each field
            cols = st.columns([1, 2, 2, 4, 3, 2])
            
            with cols[0]:
                st.write(f"`{record_id}`")
            
            with cols[1]:
                # Editable ticker
                new_ticker = st.text_input(
                    "ticker",
                    value=record['ticker'],
                    key=f"ticker_{record_id}",
                    label_visibility="collapsed"
                )
                if new_ticker != record['ticker']:
                    if record_id not in st.session_state.pending_updates:
                        st.session_state.pending_updates[record_id] = {}
                    st.session_state.pending_updates[record_id]['ticker'] = new_ticker.upper()
                    changes_made = True
            
            with cols[2]:
                # Editable date
                new_date = st.date_input(
                    "date",
                    value=record['date_added'],
                    key=f"date_{record_id}",
                    label_visibility="collapsed"
                )
                if new_date != record['date_added']:
                    if record_id not in st.session_state.pending_updates:
                        st.session_state.pending_updates[record_id] = {}
                    st.session_state.pending_updates[record_id]['date_added'] = new_date
                    changes_made = True
            
            with cols[3]:
                # Editable PDF source
                new_source = st.text_input(
                    "source",
                    value=record['pdf_source'] or "",
                    key=f"source_{record_id}",
                    label_visibility="collapsed",
                    placeholder="PDF source..."
                )
                if new_source != (record['pdf_source'] or ""):
                    if record_id not in st.session_state.pending_updates:
                        st.session_state.pending_updates[record_id] = {}
                    st.session_state.pending_updates[record_id]['pdf_source'] = new_source if new_source else None
                    changes_made = True
            
            with cols[4]:
                # Display created_at (not editable)
                created_at_str = record['created_at'].strftime('%Y-%m-%d %H:%M') if record['created_at'] else 'N/A'
                st.write(f"`{created_at_str}`")
            
            with cols[5]:
                # Delete button
                delete_col1, delete_col2 = st.columns(2)
                with delete_col1:
                    if st.button("🗑️", key=f"delete_{record_id}", help="Delete this record"):
                        st.session_state.pending_deletes.add(record_id)
                        st.rerun()
                
                with delete_col2:
                    # Show if record has pending changes
                    if record_id in st.session_state.pending_updates:
                        st.markdown("✏️", help="Has changes")
        
        # Action buttons
        if st.session_state.pending_updates or st.session_state.pending_deletes:
            st.markdown("### 💾 Pending Changes")
            
            col1, col2, col3 = st.columns([2, 2, 6])
            
            with col1:
                if st.button("✅ Save All Changes", type="primary"):
                    success_count = 0
                    error_count = 0
                    
                    # Process updates
                    for record_id, updates in st.session_state.pending_updates.items():
                        try:
                            if manager.update_record(
                                record_id=record_id,
                                ticker=updates.get('ticker'),
                                pdf_source=updates.get('pdf_source'),
                                date_added=updates.get('date_added')
                            ):
                                success_count += 1
                            else:
                                error_count += 1
                        except Exception as e:
                            st.error(f"Error updating record {record_id}: {e}")
                            error_count += 1
                    
                    # Process deletions
                    for record_id in st.session_state.pending_deletes:
                        try:
                            if manager.delete_record(record_id):
                                success_count += 1
                            else:
                                error_count += 1
                        except Exception as e:
                            st.error(f"Error deleting record {record_id}: {e}")
                            error_count += 1
                    
                    # Clear pending changes
                    st.session_state.pending_updates = {}
                    st.session_state.pending_deletes = set()
                    st.cache_data.clear()
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully processed {success_count} changes!")
                    if error_count > 0:
                        st.error(f"❌ {error_count} operations failed")
                    
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel Changes", type="secondary"):
                    st.session_state.pending_updates = {}
                    st.session_state.pending_deletes = set()
                    st.rerun()
            
            with col3:
                # Show summary of pending changes
                update_count = len(st.session_state.pending_updates)
                delete_count = len(st.session_state.pending_deletes)
                st.info(f"📝 {update_count} updates, 🗑️ {delete_count} deletions pending")
        
        # Display preview of deleted records
        if st.session_state.pending_deletes:
            with st.expander(f"🗑️ Records to be deleted ({len(st.session_state.pending_deletes)})"):
                deleted_df = df[df['id'].isin(st.session_state.pending_deletes)]
                st.dataframe(deleted_df, use_container_width=True)
        
        # Bulk operations
        st.markdown("### 🔧 Bulk Operations")
        bulk_col1, bulk_col2 = st.columns(2)
        
        with bulk_col1:
            if st.button("🗑️ Delete All Displayed Records", type="secondary"):
                if st.checkbox("Confirm bulk deletion", key="bulk_delete_confirm"):
                    deleted_count = 0
                    for _, record in df.iterrows():
                        if manager.delete_record(record['id']):
                            deleted_count += 1
                    
                    st.success(f"✅ Deleted {deleted_count} records")
                    st.cache_data.clear()
                    st.rerun()
        
        with bulk_col2:
            pass
    
    else:
        st.info("📭 No records found matching the criteria")
        
        # Show quick add if no records
        with st.expander("➕ Quick Add Record"):
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            with quick_col1:
                quick_ticker = st.text_input("Ticker", placeholder="AAPL")
            with quick_col2:
                quick_source = st.text_input("Source", placeholder="manual")
            with quick_col3:
                if st.button("➕ Add", disabled=not quick_ticker):
                    if manager.add_ticker(quick_ticker, quick_source):
                        st.success(f"✅ Added {quick_ticker}")
                        st.cache_data.clear()
                        st.rerun()

def add_records_page(manager):
    """Page for adding new records"""
    st.header("➕ Add Records")
    
    # Single ticker addition
    st.subheader("📝 Add Single Ticker")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL").upper()
    with col2:
        pdf_source = st.text_input("PDF Source", placeholder="source.pdf")
    with col3:
        date_added = st.date_input("Date Added", value=datetime.now().date())
    
    if st.button("➕ Add Ticker", type="primary", disabled=not ticker):
        if manager.add_ticker(ticker, pdf_source, date_added):
            st.success(f"✅ Added {ticker} successfully!")
            st.cache_data.clear()
            
            # Show prediction controls for single ticker
            show_prediction_controls(added_count=1)
        else:
            st.warning(f"⚠️ {ticker} already exists for {date_added}")
    
    st.divider()
    
    # Image upload and symbol extraction
    st.subheader("📷 Extract Symbols from Image")
    
    if not OCR_AVAILABLE:
        st.warning("⚠️ OCR functionality not available. To enable image symbol extraction, install: `pip install pytesseract` and system Tesseract OCR")
        st.info("📝 You can still use manual text input below or bulk import from CSV files.")
    else:
        st.markdown("""
        Upload an image containing stock symbols (screenshots, photos of screens, documents, etc.) 
        and we'll automatically extract the symbols for you using OCR technology.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image containing stock symbols"
            )
            
            if uploaded_image is not None:
                # Display the uploaded image
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Extract symbols button (moved here for better flow)
            if uploaded_image is not None:
                if st.button("🔍 Extract Symbols", type="primary"):
                    with st.spinner("Extracting symbols from image..."):
                        extracted_symbols = extract_symbols_from_image(uploaded_image)
                        
                        if extracted_symbols:
                            st.session_state.extracted_symbols = extracted_symbols
                            st.success(f"✅ Found {len(extracted_symbols)} potential symbols!")
                        else:
                            st.warning("⚠️ No symbols found in the image. Try a clearer image or check OCR setup.")
            
            # Display and manage extracted symbols
            if 'extracted_symbols' in st.session_state and st.session_state.extracted_symbols:
                st.write("**Extracted Symbols:**")
                
                # Allow user to edit the extracted symbols
                symbols_text = st.text_area(
                    "Edit symbols (one per line)",
                    value='\n'.join(st.session_state.extracted_symbols),
                    height=200,
                    help="Review and edit the extracted symbols. Remove any that aren't stock symbols."
                )
                
                # Parse the edited symbols
                final_symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    image_source = st.text_input("PDF Source (for all)", placeholder="image_extract.pdf", value="image_extracted")
                with col_b:
                    image_date = st.date_input("Date Added (for all)", value=datetime.now().date(), key="image_date")
                
                st.write(f"**Ready to import:** {len(final_symbols)} symbols")
                
                # Import the symbols
                if st.button("📥 Import Extracted Symbols", type="primary", disabled=not final_symbols):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    added_count = 0
                    total_count = len(final_symbols)
                    
                    for i, symbol in enumerate(final_symbols):
                        status_text.text(f"Adding {symbol}... ({i+1}/{total_count})")
                        if manager.add_ticker(symbol, image_source, image_date):
                            added_count += 1
                        progress_bar.progress((i + 1) / total_count)
                    
                    st.cache_data.clear()
                    
                    # Clear the session state
                    if 'extracted_symbols' in st.session_state:
                        del st.session_state.extracted_symbols
                    
                    # Show prediction controls for extracted symbols
                    show_prediction_controls(added_count=added_count)
            else:
                st.info("📷 Upload an image above to extract symbols")
                st.markdown("""
                **Supported formats:**
                - PNG, JPG, JPEG, BMP, TIFF
                - Screenshots of trading platforms
                - Photos of screens or documents
                - Scanned documents with symbol lists
                
                **Tips for better results:**
                - Use high contrast images
                - Ensure text is clearly readable
                - Avoid blurry or distorted images
                """)
    
    st.divider()
    
    # Bulk addition
    st.subheader("📋 Bulk Add Tickers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bulk_tickers = st.text_area(
            "Ticker Symbols (one per line)",
            placeholder="AAPL\nMSFT\nGOOGL\nTSLA",
            height=150
        )
        bulk_source = st.text_input("PDF Source (for all)", placeholder="bulk_import.pdf")
        bulk_date = st.date_input("Date Added (for all)", value=datetime.now().date())
    
    with col2:
        st.write("**Preview:**")
        if bulk_tickers:
            tickers_list = [t.strip().upper() for t in bulk_tickers.split('\n') if t.strip()]
            for ticker in tickers_list[:10]:  # Show first 10
                st.write(f"• {ticker}")
            if len(tickers_list) > 10:
                st.write(f"... and {len(tickers_list) - 10} more")
            
            st.write(f"**Total:** {len(tickers_list)} tickers")
    
    if st.button("📥 Bulk Add Tickers", type="primary", disabled=not bulk_tickers):
        tickers_list = [t.strip().upper() for t in bulk_tickers.split('\n') if t.strip()]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        added_count = 0
        total_count = len(tickers_list)
        
        for i, ticker in enumerate(tickers_list):
            status_text.text(f"Adding {ticker}... ({i+1}/{total_count})")
            if manager.add_ticker(ticker, bulk_source, bulk_date):
                added_count += 1
            progress_bar.progress((i + 1) / total_count)
        
        st.cache_data.clear()
        
        # Show prediction controls for bulk add
        show_prediction_controls(added_count=added_count)

def predictions_page(manager):
    """Dedicated page for running predictions"""
    st.header("🔮 Run Stock Predictions")
    st.markdown("""
    Run predictions on tickers in the database using the scanner. 
    This will analyze tickers and generate buy/sell predictions with confidence scores.
    """)
    
    # Get database statistics
    stats = manager.get_statistics()
    
    if stats['total_records'] == 0:
        st.warning("⚠️ No tickers found in database. Add some tickers first!")
        return
    
    # Show database overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tickers", stats['total_records'])
    with col2:
        st.metric("Unique Tickers", stats['unique_tickers'])
    with col3:
        st.metric("Today's Tickers", stats['today_count'])
    
    st.divider()
    
    # Prediction configuration
    st.subheader("⚙️ Prediction Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.selectbox(
            "📅 Date Filter",
            ["latest", "today"] + [(datetime.now().date() - timedelta(days=i)).strftime("%Y-%m-%d") 
                                  for i in range(1, 8)],
            help="Which tickers to run predictions on"
        )
    
    with col2:
        max_tickers = st.number_input(
            "🔢 Max Tickers",
            min_value=1,
            value=100,
            help="Number of tickers to process (no upper limit)"
        )
    
    with col3:
        background_mode = st.checkbox(
            "🔄 Background Mode",
            value=True,
            help="Run predictions in background (recommended for large batches)"
        )
    
    # Show preview of what will be processed
    st.subheader("📋 Prediction Preview")
    try:
        preview_df = manager.get_records(
            date_filter=date_filter if date_filter not in ["latest", "today"] else (date_filter if date_filter == "today" else None),
            limit=max_tickers
        )
        
        if not preview_df.empty:
            st.info(f"📊 Will process **{len(preview_df)}** tickers")
            
            # Show preview table
            with st.expander("👀 Preview Tickers"):
                st.dataframe(preview_df[['ticker', 'date_added', 'pdf_source']], use_container_width=True)
        else:
            st.warning("⚠️ No tickers found matching the selected criteria")
            return
            
    except Exception as e:
        st.error(f"❌ Error loading preview: {e}")
        return
    
    st.divider()
    
    # Run predictions
    st.subheader("🚀 Execute Predictions")
    
    col_run, col_status = st.columns([1, 2])
    
    with col_run:
        if st.button("🚀 Start Predictions", type="primary", disabled=len(preview_df) == 0):
            with st.spinner("Starting predictions..."):
                result = run_predictions(
                    date_filter=date_filter,
                    max_tickers=max_tickers,
                    background=background_mode
                )
                
                if result['success']:
                    if background_mode:
                        st.success("✅ Predictions started in background!")
                        st.info("💡 Refresh this page to see results when complete")
                    else:
                        st.success("✅ Predictions completed!")
                        
                        # Show output
                        if 'output' in result:
                            with st.expander("📊 Prediction Output"):
                                st.code(result['output'])
                else:
                    st.error(f"❌ Failed to start predictions: {result.get('error', 'Unknown error')}")
    
    with col_status:
        # Show command that will be executed using config
        python_exe = get_python_executable()
        
        cmd_parts = [
            python_exe,
            "scanner.py",
            "--predict-only",
            "--date-filter", date_filter,
            "--max-tickers", str(max_tickers)
        ]
        
        st.info("**Command to execute:**")
        st.code(" ".join(cmd_parts))
    
    st.divider()
    
    # Show prediction status and results
    st.subheader("📊 Current Prediction Status & Results")
    
    # Debug info
    if st.checkbox("🐛 Show Debug Info", help="Show debugging information"):
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        if 'prediction_result' in st.session_state:
            st.write("**Prediction Result:**", st.session_state.prediction_result)
        
        # Test log tailing with existing log files
        if st.button("🧪 Test Log Tailing", help="Create test prediction result for debugging"):
            # Find a recent log file for testing
            logs_dir = DIRECTORIES['logs']
            if os.path.exists(logs_dir):
                log_files = [f for f in os.listdir(logs_dir) if f.endswith(FILE_PATTERNS['log_extension'])]
                if log_files:
                    test_log = os.path.join(logs_dir, sorted(log_files)[-1])  # Get most recent
                    st.session_state.prediction_result = {
                        'success': True,
                        'output': f"Test log: {test_log}",
                        'error': None,
                        'command': 'test command',
                        'timestamp': datetime.now().isoformat(),
                        'log_file': test_log
                    }
                    st.success(f"✅ Test prediction result created with log: {test_log}")
                    st.rerun()
                else:
                    st.warning("No log files found for testing")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("🔄 Auto-refresh status (every 30 seconds)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Check for prediction results in session state
    if 'prediction_result' in st.session_state:
        pred_result = st.session_state.prediction_result
        
        # Status header with timestamp
        status_col1, status_col2, status_col3 = st.columns([2, 2, 1])
        
        with status_col1:
            if pred_result['success']:
                st.success("✅ **Status:** Predictions Completed")
            else:
                st.error("❌ **Status:** Predictions Failed")
        
        with status_col2:
            completion_time = pred_result.get('timestamp', 'Unknown')
            if completion_time != 'Unknown':
                try:
                    completed_dt = datetime.fromisoformat(completion_time)
                    time_diff = datetime.now() - completed_dt
                    if time_diff.days > 0:
                        time_str = f"{time_diff.days} days ago"
                    elif time_diff.seconds > 3600:
                        time_str = f"{time_diff.seconds // 3600} hours ago"
                    elif time_diff.seconds > 60:
                        time_str = f"{time_diff.seconds // 60} minutes ago"
                    else:
                        time_str = "Just now"
                    st.info(f"⏰ **Completed:** {time_str}")
                except:
                    st.info(f"⏰ **Completed:** {completion_time}")
            else:
                st.info("⏰ **Completed:** Unknown")
        
        with status_col3:
            if st.button("🔄 Clear Status", help="Clear prediction status"):
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                st.rerun()
        
        st.divider()
        
        # Show log tailing section
        if pred_result.get('log_file'):
            log_file = pred_result['log_file']
            
            st.markdown("### 📄 Live Prediction Log")
            
            # Log status and controls
            log_col1, log_col2, log_col3, log_col4 = st.columns([2, 2, 2, 1])
            
            with log_col1:
                if os.path.exists(log_file):
                    log_size = get_log_size(log_file)
                    status = get_process_status(log_file)
                    
                    # Show status with appropriate color
                    if "Running" in status:
                        st.success(f"{status} ({log_size})")
                    elif "Completed" in status:
                        st.info(f"{status} ({log_size})")
                    elif "Failed" in status or "Error" in status:
                        st.error(f"{status} ({log_size})")
                    else:
                        st.warning(f"{status} ({log_size})")
                else:
                    st.warning("📄 **Log file not found**")
            
            with log_col2:
                log_lines = st.selectbox(
                    "Lines to show",
                    [25, 50, 100, 200, 500],
                    index=1,
                    key="log_lines_select"
                )
            
            with log_col3:
                auto_scroll = st.checkbox(
                    "📜 Auto-scroll",
                    value=True,
                    help="Automatically scroll to bottom of log"
                )
            
            with log_col4:
                if st.button("🔄", help="Refresh log", key="refresh_log"):
                    st.rerun()
            
            # Display log content
            if os.path.exists(log_file):
                log_content = get_log_tail(log_file, log_lines)
                
                if log_content:
                    # Create container for log display
                    log_container = st.container()
                    with log_container:
                        st.code(log_content, language="text")
                        
                        if auto_scroll:
                            # Add some JavaScript to scroll to bottom (will work in some cases)
                            st.markdown("""
                            <script>
                            setTimeout(function() {
                                var element = document.querySelector('[data-testid="stCodeBlock"]');
                                if (element) {
                                    element.scrollTop = element.scrollHeight;
                                }
                            }, 100);
                            </script>
                            """, unsafe_allow_html=True)
                else:
                    st.info("📝 Log file is empty or unreadable")
            else:
                st.warning(f"📄 Log file not found: {log_file}")
            
            # Log file actions
            log_actions_col1, log_actions_col2, log_actions_col3, log_actions_col4 = st.columns(4)
            
            with log_actions_col1:
                if os.path.exists(log_file):
                    # Download log file
                    try:
                        with open(log_file, 'rb') as f:
                            st.download_button(
                                "📥 Download Full Log",
                                f.read(),
                                file_name=os.path.basename(log_file),
                                mime="text/plain"
                            )
                    except:
                        pass
            
            with log_actions_col2:
                if os.path.exists(log_file):
                    if st.button("🗑️ Delete Log", help="Delete this log file"):
                        try:
                            os.remove(log_file)
                            st.success("✅ Log file deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting log: {e}")
            
            with log_actions_col3:
                # Show all log files using config
                logs_dir = DIRECTORIES['logs']
                if os.path.exists(logs_dir):
                    log_files = [f for f in os.listdir(logs_dir) if f.endswith(FILE_PATTERNS['log_extension'])]
                    if log_files:
                        if st.button(f"📂 View All Logs ({len(log_files)})", help="Show all prediction logs"):
                            st.session_state.show_all_logs = True
                            st.rerun()
            
            with log_actions_col4:
                # Clean old logs using config
                logs_dir = DIRECTORIES['logs']
                if os.path.exists(logs_dir):
                    log_files = [f for f in os.listdir(logs_dir) if f.endswith(FILE_PATTERNS['log_extension'])]
                    if len(log_files) > 1:
                        if st.button("🧹 Clean Old Logs", help="Delete logs older than 7 days"):
                            try:
                                cutoff_time = datetime.now().timestamp() - (7 * 24 * 3600)  # 7 days
                                deleted_count = 0
                                
                                for log_name in log_files:
                                    log_path = os.path.join(logs_dir, log_name)
                                    if os.path.getmtime(log_path) < cutoff_time:
                                        os.remove(log_path)
                                        deleted_count += 1
                                
                                if deleted_count > 0:
                                    st.success(f"✅ Deleted {deleted_count} old logs")
                                else:
                                    st.info("📝 No old logs to delete")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error cleaning logs: {e}")
            
            # Show all logs if requested
            if st.session_state.get('show_all_logs', False):
                with st.expander("📂 All Prediction Logs", expanded=True):
                    logs_dir = DIRECTORIES['logs']
                    if os.path.exists(logs_dir):
                        log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith(FILE_PATTERNS['log_extension'])], reverse=True)
                        
                        if log_files:
                            for log_file_name in log_files:
                                log_path = os.path.join(logs_dir, log_file_name)
                                mod_time = datetime.fromtimestamp(os.path.getmtime(log_path))
                                file_size = get_log_size(log_path)
                                
                                cols = st.columns([3, 2, 2, 1])
                                with cols[0]:
                                    st.write(f"📄 {log_file_name}")
                                with cols[1]:
                                    st.write(f"🕒 {mod_time.strftime('%Y-%m-%d %H:%M')}")
                                with cols[2]:
                                    st.write(f"📊 {file_size}")
                                with cols[3]:
                                    if st.button("👀", key=f"view_{log_file_name}", help="View this log"):
                                        st.session_state.viewing_log = log_path
                                        st.rerun()
                        else:
                            st.info("📭 No log files found")
                    
                    if st.button("❌ Close", key="close_all_logs"):
                        st.session_state.show_all_logs = False
                        st.rerun()
            
            # Show specific log if requested
            if st.session_state.get('viewing_log'):
                viewing_log = st.session_state.viewing_log
                with st.expander(f"📄 Viewing: {os.path.basename(viewing_log)}", expanded=True):
                    if os.path.exists(viewing_log):
                        content = get_log_tail(viewing_log, 200)
                        st.code(content, language="text")
                        
                        if st.button("❌ Close Log View", key="close_log_view"):
                            del st.session_state.viewing_log
                            st.rerun()
                    else:
                        st.error("📄 Log file no longer exists")
                        del st.session_state.viewing_log
                        st.rerun()
            
            st.divider()
        
        if pred_result['success']:
            # Enhanced results display
            col_reports, col_metrics = st.columns([2, 1])
            
            with col_reports:
                st.markdown("### 📁 Generated Reports")
                
                # Check for generated reports using config
                reports_dir = DIRECTORIES['reports']
                if os.path.exists(reports_dir):
                    html_file = os.path.join(reports_dir, FILE_PATTERNS['report_html'])
                    json_file = os.path.join(reports_dir, FILE_PATTERNS['report_json'])
                    csv_file = os.path.join(reports_dir, FILE_PATTERNS['report_csv'])
                    
                    report_cols = st.columns(3)
                    
                    with report_cols[0]:
                        if os.path.exists(html_file):
                            file_size = os.path.getsize(html_file) / 1024  # KB
                            st.success(f"📊 **HTML Report**\n{file_size:.1f} KB")
                            
                            # Use web-compatible report access
                            web_url = get_web_url_for_report(FILE_PATTERNS['report_html'])
                            if web_url and WEB_CONFIG['enable_web_links']:
                                if st.button("🌐 Open Web Report"):
                                    st.markdown(f"**[🔗 Open HTML Report]({web_url})**")
                            
                            if WEB_CONFIG['enable_file_download']:
                                try:
                                    with open(html_file, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    st.download_button(
                                        label="📥 Download HTML",
                                        data=html_content,
                                        file_name=FILE_PATTERNS['report_html'],
                                        mime="text/html"
                                    )
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                            
                            if not web_url:
                                st.info(f"📂 File: {html_file}")
                        else:
                            st.warning("📊 **HTML Report**\n(Not found)")
                    
                    with report_cols[1]:
                        if os.path.exists(csv_file):
                            file_size = os.path.getsize(csv_file) / 1024  # KB
                            st.success(f"📈 **CSV Report**\n{file_size:.1f} KB")
                            
                            # Offer download
                            try:
                                with open(csv_file, 'rb') as f:
                                    st.download_button(
                                        "📥 Download CSV",
                                        f.read(),
                                        file_name="predictions.csv",
                                        mime="text/csv"
                                    )
                            except:
                                pass
                        else:
                            st.warning("📈 **CSV Report**\n(Not found)")
                    
                    with report_cols[2]:
                        if os.path.exists(json_file):
                            file_size = os.path.getsize(json_file) / 1024  # KB
                            st.success(f"📋 **JSON Report**\n{file_size:.1f} KB")
                        else:
                            st.warning("📋 **JSON Report**\n(Not found)")
            
            with col_metrics:
                st.markdown("### 📈 Prediction Summary")
                
                # Load and display JSON metrics
                json_file = os.path.join(reports_dir, "report.json")
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                            
                        # Key metrics
                        st.metric("🎯 Total Processed", json_data.get('total_tickers_processed', 0))
                        st.metric("✅ Successful", json_data.get('successful_predictions', 0))
                        st.metric("❌ Failed", json_data.get('failed_predictions', 0))
                        st.metric("📊 Total Predictions", json_data.get('total_predictions_found', 0))
                        
                        # Prediction breakdown
                        st.markdown("**🔍 Prediction Breakdown:**")
                        col_buy, col_sell = st.columns(2)
                        with col_buy:
                            st.metric("🟢 Buy", json_data.get('buy_signals', 0))
                        with col_sell:
                            st.metric("🔴 Sell", json_data.get('sell_signals', 0))
                        
                        # Performance metrics
                        avg_confidence = json_data.get('average_confidence', 0) * 100
                        avg_accuracy_raw = json_data.get('average_accuracy')
                        if avg_accuracy_raw is not None:
                            avg_accuracy = avg_accuracy_raw * 100
                            accuracy_display = f"{avg_accuracy:.1f}%"
                        else:
                            accuracy_display = "N/A"
                        
                        st.markdown("**� Performance:**")
                        st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
                        st.metric("🎯 Avg Accuracy", accuracy_display)
                        
                    except Exception as e:
                        st.warning(f"Could not parse metrics: {e}")
                else:
                    st.info("📊 No metrics available")
            
            # Show detailed output in expander
            with st.expander("📝 Detailed Output & Logs"):
                if pred_result.get('output'):
                    st.text_area("📤 Prediction Output", pred_result['output'], height=300)
                if pred_result.get('error'):
                    st.text_area("❌ Error Output", pred_result['error'], height=150)
                
                st.code(f"Command executed: {pred_result.get('command', 'unknown')}")
        
        else:
            # Failed prediction display
            st.error("❌ **Prediction execution failed!**")
            st.error(f"**Error:** {pred_result.get('error', 'Unknown error')}")
            
            # Show error details
            with st.expander("� Error Details"):
                if pred_result.get('error'):
                    st.text_area("Error Message", pred_result['error'], height=150)
                if pred_result.get('output'):
                    st.text_area("Partial Output", pred_result['output'], height=200)
                st.code(f"Failed command: {pred_result.get('command', 'unknown')}")
    
    else:
        # No recent predictions
        st.info("🔄 **Status:** No recent prediction results")
        
        # Check for existing reports from previous runs using config
        reports_dir = DIRECTORIES['reports']
        if os.path.exists(reports_dir):
            html_file = os.path.join(reports_dir, FILE_PATTERNS['report_html'])
            csv_file = os.path.join(reports_dir, FILE_PATTERNS['report_csv'])
            json_file = os.path.join(reports_dir, FILE_PATTERNS['report_json'])
            
            if any(os.path.exists(f) for f in [html_file, csv_file, json_file]):
                st.markdown("### 📂 Previous Reports Found")
                
                prev_cols = st.columns(3)
                
                with prev_cols[0]:
                    if os.path.exists(html_file):
                        mod_time = os.path.getmtime(html_file)
                        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
                        file_size = os.path.getsize(html_file) / 1024
                        st.info(f"📊 **HTML Report**\n{mod_time_str}\n({file_size:.1f} KB)")
                        # Display web link for previous report
                        web_url = get_web_url_for_report(FILE_PATTERNS['report_html'])
                        if web_url and WEB_CONFIG['enable_web_links']:
                            st.markdown(f"**[🔗 Open Previous Report]({web_url})**")
                        
                        # Add download button for previous HTML report
                        if WEB_CONFIG['enable_file_download']:
                            try:
                                with open(html_file, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                st.download_button(
                                    label="📥 Download Previous HTML",
                                    data=html_content,
                                    file_name=FILE_PATTERNS['report_html'],
                                    mime="text/html",
                                    key="download_prev_html"
                                )
                            except Exception as e:
                                st.error(f"❌ Error reading HTML: {e}")
                
                with prev_cols[1]:
                    if os.path.exists(csv_file):
                        mod_time = os.path.getmtime(csv_file)
                        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
                        file_size = os.path.getsize(csv_file) / 1024
                        st.info(f"📈 **CSV Report**\n{mod_time_str}\n({file_size:.1f} KB)")
                        try:
                            with open(csv_file, 'rb') as f:
                                st.download_button(
                                    "📥 Download Previous CSV",
                                    f.read(),
                                    file_name=f"previous_predictions.csv",
                                    mime="text/csv",
                                    key="download_prev_csv"
                                )
                        except:
                            pass
                
                with prev_cols[2]:
                    if os.path.exists(json_file):
                        mod_time = os.path.getmtime(json_file)
                        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
                        file_size = os.path.getsize(json_file) / 1024
                        st.info(f"� **JSON Report**\n{mod_time_str}\n({file_size:.1f} KB)")
                        
                        # Show quick metrics from previous run
                        try:
                            with open(json_file, 'r') as f:
                                prev_data = json.load(f)
                            
                            st.markdown("**Previous Results:**")
                            st.write(f"• Buy signals: {prev_data.get('buy_signals', 0)}")
                            st.write(f"• Sell signals: {prev_data.get('sell_signals', 0)}")
                            st.write(f"• Total processed: {prev_data.get('total_tickers_processed', 0)}")
                        except:
                            pass
            else:
                st.info("💡 Run predictions above to generate reports and see results here")
        else:
            st.info("💡 Run predictions above to generate reports and see results here")

def statistics_page(manager):
    """Page for displaying statistics and analytics"""
    st.header("📊 Statistics & Analytics")
    
    stats = manager.get_statistics()
    
    if not stats:
        st.error("❌ Unable to load statistics")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("Unique Tickers", f"{stats['unique_tickers']:,}")
    with col3:
        st.metric("Today's Records", f"{stats['today_count']:,}")
    with col4:
        diversity = (stats['unique_tickers'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
        st.metric("Diversity %", f"{diversity:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Records by date chart
        if stats['recent_dates']:
            st.subheader("📅 Records by Date (Last 14 days)")
            dates_df = pd.DataFrame(stats['recent_dates'])
            
            fig = px.bar(
                dates_df, 
                x='date_added', 
                y='count',
                title="Daily Record Count",
                labels={'count': 'Number of Records', 'date_added': 'Date'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top tickers chart
        if stats['top_tickers']:
            st.subheader("🏆 Most Frequent Tickers")
            tickers_df = pd.DataFrame(stats['top_tickers'])
            
            fig = px.pie(
                tickers_df, 
                values='count', 
                names='ticker',
                title="Ticker Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tables
    col1, col2 = st.columns(2)
    
    with col1:
        if stats['top_tickers']:
            st.subheader("📈 Top Tickers")
            tickers_df = pd.DataFrame(stats['top_tickers'])
            st.dataframe(tickers_df, use_container_width=True)
    
    with col2:
        if stats['top_sources']:
            st.subheader("📄 Top PDF Sources")
            sources_df = pd.DataFrame(stats['top_sources'])
            st.dataframe(sources_df, use_container_width=True)

def management_page(manager):
    """Page for database maintenance operations"""
    st.header("🔧 Database Management")
    
    # Cleanup old records
    st.subheader("🧹 Cleanup Old Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cleanup_days = st.number_input(
            "Remove records older than (days)", 
            min_value=1, 
            max_value=365, 
            value=30
        )
        
        cutoff_date = datetime.now().date() - timedelta(days=cleanup_days)
        st.write(f"**Cutoff Date:** {cutoff_date}")
        
        # Preview what will be deleted
        df_old = manager.get_records()
        if not df_old.empty:
            old_records = df_old[pd.to_datetime(df_old['date_added']).dt.date < cutoff_date]
            st.write(f"**Records to delete:** {len(old_records)}")
        
        if st.button("🗑️ Cleanup Old Records", type="secondary"):
            deleted_count = manager.cleanup_old_records(cleanup_days)
            if deleted_count > 0:
                st.success(f"✅ Deleted {deleted_count} old records")
                st.cache_data.clear()
            else:
                st.info("📭 No old records found to delete")
    
    with col2:
        st.subheader("🔄 Database Operations")
        
        if st.button("🔄 Clear Cache", help="Clear cached data"):
            st.cache_data.clear()
            st.success("✅ Cache cleared")
        
        if st.button("📊 Refresh Statistics", help="Force refresh statistics"):
            st.cache_data.clear()
            st.success("✅ Statistics refreshed")
        
        st.subheader("ℹ️ Database Info")
        
        # Connection status
        try:
            with manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    st.success("✅ Database Connected")
                    st.text(f"PostgreSQL Version: {version.split()[1]}")
        except Exception as e:
            st.error(f"❌ Database Connection Error: {e}")

def export_import_page(manager):
    """Page for data export and import operations"""
    st.header("📤 Export & Import Data")
    
    # Export section
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_filter = st.selectbox(
            "Export Filter",
            ["all", "today", "last_7_days", "last_30_days", "custom_date"]
        )
        
        if export_filter == "custom_date":
            export_date = st.date_input("Select Date")
            date_filter = export_date.strftime("%Y-%m-%d")
        elif export_filter == "today":
            date_filter = "today"
        elif export_filter == "last_7_days":
            date_filter = (datetime.now().date() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif export_filter == "last_30_days":
            date_filter = (datetime.now().date() - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            date_filter = None
        
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
    
    with col2:
        # Preview export data
        df_export = manager.get_records(date_filter=date_filter)
        st.write(f"**Records to export:** {len(df_export)}")
        
        if not df_export.empty:
            st.dataframe(df_export.head(), use_container_width=True)
    
    if not df_export.empty:
        if export_format == "CSV":
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            json_data = df_export.to_json(orient='records', date_format='iso', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.divider()
    
    # Import section
    st.subheader("📥 Import Data")
    
    # Create tabs for different import methods
    import_tab1, import_tab2, import_tab3, import_tab4 = st.tabs(["📄 CSV File", "📷 Image OCR", "📝 Text Input", "🔧 Bulk Update"])
    
    with import_tab1:
        # CSV file import (existing functionality)
        uploaded_file = st.file_uploader(
            "Choose a CSV file to import",
            type=['csv'],
            help="CSV should contain columns: ticker, date_added, pdf_source",
            key="csv_uploader"
        )
        
        if uploaded_file is not None:
            try:
                df_import = pd.read_csv(uploaded_file)
                
                st.write("**File Preview:**")
                st.dataframe(df_import.head(), use_container_width=True)
                
                # Validate columns
                required_columns = ['ticker', 'date_added']
                missing_columns = [col for col in required_columns if col not in df_import.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                else:
                    st.success("✅ File format is valid")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Total rows:** {len(df_import)}")
                    with col2:
                        st.write(f"**Columns:** {list(df_import.columns)}")
                    
                    if st.button("📥 Import CSV Data", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        imported_count = 0
                        total_rows = len(df_import)
                        
                        for i, row in df_import.iterrows():
                            status_text.text(f"Importing row {i+1}/{total_rows}...")
                            try:
                                ticker = str(row['ticker']).upper().strip()
                                date_added = pd.to_datetime(row['date_added']).date()
                                pdf_source = row.get('pdf_source', None)
                                
                                if manager.add_ticker(ticker, pdf_source, date_added):
                                    imported_count += 1
                            except Exception as e:
                                st.warning(f"⚠️ Error importing row {i+1}: {e}")
                            
                            progress_bar.progress((i + 1) / total_rows)
                        
                        st.success(f"✅ Successfully imported {imported_count} out of {total_rows} records!")
                        st.cache_data.clear()
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with import_tab2:
        # Image OCR import
        if not OCR_AVAILABLE:
            st.warning("⚠️ OCR functionality not available. To enable image symbol extraction, install: `pip install pytesseract` and system Tesseract OCR")
            st.info("📝 You can still use CSV file import or manual text input.")
        else:
            st.markdown("""
            Upload an image containing stock symbols and extract them automatically using OCR technology.
            Perfect for importing from screenshots, photos, or scanned documents.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_image = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    help="Upload an image containing stock symbols",
                    key="image_uploader_import"
                )
                
                if uploaded_image is not None:
                    # Display the uploaded image
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Extract symbols button (moved here for better flow)
                if uploaded_image is not None:
                    if st.button("🔍 Extract Symbols from Image", type="primary", key="extract_import"):
                        with st.spinner("Extracting symbols from image..."):
                            extracted_symbols = extract_symbols_from_image(uploaded_image)
                            
                            if extracted_symbols:
                                st.session_state.extracted_symbols_import = extracted_symbols
                                st.success(f"✅ Found {len(extracted_symbols)} potential symbols!")
                            else:
                                st.warning("⚠️ No symbols found in the image. Try a clearer image or check OCR setup.")
                
                # Display and manage extracted symbols
                if 'extracted_symbols_import' in st.session_state and st.session_state.extracted_symbols_import:
                    st.write("**Extracted Symbols:**")
                    
                    # Allow user to edit the extracted symbols
                    symbols_text = st.text_area(
                        "Edit symbols (one per line)",
                        value='\n'.join(st.session_state.extracted_symbols_import),
                        height=200,
                        help="Review and edit the extracted symbols. Remove any that aren't stock symbols.",
                        key="symbols_text_import"
                    )
                    
                    # Parse the edited symbols
                    final_symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        image_source = st.text_input("PDF Source (for all)", placeholder="image_import.pdf", value="image_imported", key="image_source_import")
                    with col_b:
                        image_date = st.date_input("Date Added (for all)", value=datetime.now().date(), key="image_date_import")
                    
                    st.write(f"**Ready to import:** {len(final_symbols)} symbols")
                    
                    # Import the symbols
                    if st.button("📥 Import Extracted Symbols", type="primary", disabled=not final_symbols, key="import_extracted"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        added_count = 0
                        total_count = len(final_symbols)
                        
                        for i, symbol in enumerate(final_symbols):
                            status_text.text(f"Adding {symbol}... ({i+1}/{total_count})")
                            if manager.add_ticker(symbol, image_source, image_date):
                                added_count += 1
                            progress_bar.progress((i + 1) / total_count)
                        
                        st.success(f"✅ Successfully imported {added_count} out of {total_count} symbols from image!")
                        st.cache_data.clear()
                        
                        # Clear the session state
                        if 'extracted_symbols_import' in st.session_state:
                            del st.session_state.extracted_symbols_import
                else:
                    st.info("📷 Upload an image above to extract symbols")
                    st.markdown("""
                    **Supported formats:**
                    - PNG, JPG, JPEG, BMP, TIFF
                    - Screenshots of trading platforms
                    - Photos of screens or documents
                    - Scanned documents with symbol lists
                    
                    **Tips for better results:**
                    - Use high contrast images
                    - Ensure text is clearly readable
                    - Avoid blurry or distorted images
                    """)
    
    with import_tab3:
        # Manual text input for symbols
        st.markdown("""
        Paste or type stock symbols directly. This is useful when you have symbols from 
        other sources or want to manually input a list.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            manual_symbols = st.text_area(
                "Enter symbols (one per line or comma/space separated)",
                placeholder="AAPL\nMSFT GOOGL TSLA\nAMZN, NVDA, META",
                height=200,
                help="Enter stock symbols in any format - one per line, comma separated, or space separated"
            )
            
            manual_source = st.text_input("PDF Source (for all)", placeholder="manual_import.pdf", value="manual_import")
            manual_date = st.date_input("Date Added (for all)", value=datetime.now().date(), key="manual_date_import")
        
        with col2:
            if manual_symbols:
                # Parse symbols from various formats
                import re
                # Replace commas and multiple spaces with single spaces, then split
                normalized_text = re.sub(r'[,\s]+', ' ', manual_symbols.strip())
                parsed_symbols = [s.strip().upper() for s in normalized_text.split() if s.strip()]
                
                st.write("**Parsed Symbols:**")
                for i, symbol in enumerate(parsed_symbols[:20], 1):  # Show first 20
                    st.write(f"{i}. {symbol}")
                
                if len(parsed_symbols) > 20:
                    st.write(f"... and {len(parsed_symbols) - 20} more")
                
                st.write(f"**Total symbols:** {len(parsed_symbols)}")
                
                if st.button("📥 Import Manual Symbols", type="primary", disabled=not parsed_symbols):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    added_count = 0
                    total_count = len(parsed_symbols)
                    
                    for i, symbol in enumerate(parsed_symbols):
                        status_text.text(f"Adding {symbol}... ({i+1}/{total_count})")
                        if manager.add_ticker(symbol, manual_source, manual_date):
                            added_count += 1
                        progress_bar.progress((i + 1) / total_count)
                    
                    st.success(f"✅ Successfully imported {added_count} out of {total_count} symbols!")
                    st.cache_data.clear()
            else:
                st.info("📝 Enter symbols above to see the preview")
                st.markdown("""
                **Supported formats:**
                - One symbol per line: `AAPL\\nMSFT\\nGOOGL`
                - Space separated: `AAPL MSFT GOOGL TSLA`
                - Comma separated: `AAPL, MSFT, GOOGL, TSLA`
                - Mixed format: `AAPL\\nMSFT GOOGL\\nTSLA, AMZN`
                """)
    
    with import_tab4:
        # Bulk update operations
        st.markdown("""
        Perform bulk operations on existing records. These operations affect records 
        that are already in the database.
        """)
        
        st.subheader("📝 Bulk Update PDF Sources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter options for bulk update
            st.write("**Select Records to Update:**")
            
            bulk_date_filter = st.selectbox(
                "Date Filter",
                ["all", "today"] + [(datetime.now().date() - timedelta(days=i)).strftime("%Y-%m-%d") 
                                   for i in range(1, 8)],
                help="Choose which records to update based on date",
                key="bulk_date_filter"
            )
            
            bulk_ticker_filter = st.text_input(
                "Ticker Filter", 
                placeholder="e.g., AAPL (leave empty for all)",
                help="Filter by ticker symbol (partial matches allowed)",
                key="bulk_ticker_filter"
            )
            
            bulk_source_filter = st.text_input(
                "Current PDF Source Filter",
                placeholder="e.g., old_source.pdf (leave empty for all)",
                help="Filter by current PDF source (partial matches allowed)",
                key="bulk_source_filter"
            )
            
            # New PDF source
            new_bulk_source = st.text_input(
                "New PDF Source",
                placeholder="new_source.pdf",
                help="The new PDF source to apply to filtered records",
                key="new_bulk_source"
            )
            
        with col2:
            # Preview affected records
            st.write("**Preview Affected Records:**")
            
            # Get records based on filters
            preview_df = manager.get_records(
                date_filter=bulk_date_filter if bulk_date_filter != "all" else None,
                ticker_filter=bulk_ticker_filter if bulk_ticker_filter else None,
                limit=100  # Limit preview to 100 records
            )
            
            # Additional filtering by PDF source if specified
            if bulk_source_filter and not preview_df.empty:
                preview_df = preview_df[
                    preview_df['pdf_source'].fillna('').str.contains(bulk_source_filter, case=False, na=False)
                ]
            
            if not preview_df.empty:
                st.write(f"**Records to update:** {len(preview_df)}")
                
                # Show preview of first few records
                display_df = preview_df[['id', 'ticker', 'date_added', 'pdf_source']].head(10)
                st.dataframe(display_df, use_container_width=True)
                
                if len(preview_df) > 10:
                    st.write(f"... and {len(preview_df) - 10} more records")
                
                # Show current PDF sources summary
                if 'pdf_source' in preview_df.columns:
                    source_counts = preview_df['pdf_source'].fillna('(empty)').value_counts()
                    st.write("**Current PDF Sources:**")
                    for source, count in source_counts.head(5).items():
                        st.write(f"• {source}: {count} records")
                    if len(source_counts) > 5:
                        st.write(f"• ... and {len(source_counts) - 5} more sources")
                
            else:
                st.info("📭 No records match the current filters")
        
        # Bulk update action
        st.divider()
        
        if not preview_df.empty and new_bulk_source:
            col_a, col_b, col_c = st.columns([2, 2, 6])
            
            with col_a:
                if st.button("📝 Apply Bulk Update", type="primary", key="apply_bulk_update"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    updated_count = 0
                    total_count = len(preview_df)
                    
                    for i, (_, record) in enumerate(preview_df.iterrows()):
                        status_text.text(f"Updating {record['ticker']}... ({i+1}/{total_count})")
                        if manager.update_record(record['id'], pdf_source=new_bulk_source):
                            updated_count += 1
                        progress_bar.progress((i + 1) / total_count)
                    
                    st.success(f"✅ Successfully updated {updated_count} out of {total_count} records!")
                    st.cache_data.clear()
            
            with col_b:
                st.warning(f"⚠️ This will update {len(preview_df)} records")
            
            with col_c:
                st.info(f"💡 New PDF source: '{new_bulk_source}' will be applied to all filtered records")
        
        elif preview_df.empty:
            st.info("📋 Apply filters above to see records that would be updated")
        elif not new_bulk_source:
            st.info("📝 Enter a new PDF source to enable bulk update")
        
        # Additional bulk operations
        st.divider()
        st.subheader("🗑️ Bulk Delete Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("⚠️ **Danger Zone** - These operations cannot be undone!")
            
            if st.button("🗑️ Delete All Records with Empty PDF Source", type="secondary"):
                if st.checkbox("Confirm deletion of records with empty PDF source", key="confirm_empty_delete"):
                    empty_source_df = manager.get_records()
                    if not empty_source_df.empty:
                        empty_records = empty_source_df[empty_source_df['pdf_source'].isna() | (empty_source_df['pdf_source'] == '')]
                        
                        deleted_count = 0
                        for _, record in empty_records.iterrows():
                            if manager.delete_record(record['id']):
                                deleted_count += 1
                        
                        st.success(f"✅ Deleted {deleted_count} records with empty PDF source")
                        st.cache_data.clear()
        
        with col2:
            # Show statistics about empty PDF sources
            all_records = manager.get_records()
            if not all_records.empty:
                empty_count = len(all_records[all_records['pdf_source'].isna() | (all_records['pdf_source'] == '')])
                total_count = len(all_records)
                
                st.metric("Records with Empty PDF Source", f"{empty_count:,}")
                st.metric("Total Records", f"{total_count:,}")
                
                if empty_count > 0:
                    percentage = (empty_count / total_count) * 100
                    st.metric("Percentage Empty", f"{percentage:.1f}%")

if __name__ == "__main__":
    main()
