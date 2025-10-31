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

def parse_number_with_suffix(value_str):
    """Parse number with K/M suffix (e.g., '1.5K' -> 1500.0, '2.3M' -> 2300000.0)"""
    if not value_str or value_str == "":
        return 0.0
    
    # Convert to string and strip whitespace
    value_str = str(value_str).strip().upper()
    
    try:
        # Check for K suffix
        if value_str.endswith('K'):
            number = float(value_str[:-1])
            return number * 1000
        # Check for M suffix
        elif value_str.endswith('M'):
            number = float(value_str[:-1])
            return number * 1000000
        # No suffix, just convert to float
        else:
            return float(value_str)
    except ValueError:
        # If parsing fails, return 0
        return 0.0

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
                            price DECIMAL(10,2),
                            exp DATE,
                            premiums DECIMAL(10,2),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(ticker, date_added)
                        )
                    """)
                    
                    # Create index for better performance
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_seekingalpha_ticker_date 
                        ON seekingalpha(ticker, date_added)
                    """)
                    
                    # Add missing columns if they don't exist (for existing tables)
                    cur.execute("""
                        ALTER TABLE seekingalpha 
                        ADD COLUMN IF NOT EXISTS price DECIMAL(10,2)
                    """)
                    cur.execute("""
                        ALTER TABLE seekingalpha 
                        ADD COLUMN IF NOT EXISTS exp DATE
                    """)
                    cur.execute("""
                        ALTER TABLE seekingalpha 
                        ADD COLUMN IF NOT EXISTS premiums DECIMAL(10,2)
                    """)
                    cur.execute("""
                        ALTER TABLE seekingalpha 
                        ADD COLUMN IF NOT EXISTS notional_value DECIMAL(15,2)
                    """)
                    cur.execute("""
                        ALTER TABLE seekingalpha 
                        ADD COLUMN IF NOT EXISTS strike DECIMAL(10,2)
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
                   date_added: date = None, price: float = None, exp: date = None, 
                   premiums: float = None, strike: float = None, notional_value: float = None) -> bool:
        """Add a single ticker to the database"""
        # Check if ticker already exists in database
        if self.ticker_exists(ticker):
            return False  # Ticker already exists
            
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO seekingalpha (ticker, date_added, pdf_source, price, exp, premiums, strike, notional_value)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date_added) DO NOTHING
                    """, (ticker.upper().strip(), date_added or datetime.now().date(), pdf_source, price, exp, premiums, strike, notional_value))
                    
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    else:
                        return False  # Already exists for this date
                        
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
    
    def ticker_exists(self, ticker: str) -> bool:
        """Check if a ticker already exists in the database (regardless of date)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM seekingalpha 
                        WHERE ticker = %s
                    """, (ticker.upper().strip(),))
                    
                    count = cur.fetchone()[0]
                    return count > 0
                    
        except Exception as e:
            st.error(f"❌ Error checking ticker existence: {e}")
            return False
    
    def delete_record(self, record_id: int) -> bool:
        """Delete a record by ID"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # First check if record exists
                    cur.execute("SELECT id FROM seekingalpha WHERE id = %s", (record_id,))
                    if not cur.fetchone():
                        st.warning(f"Record with ID {record_id} not found")
                        return False
                    
                    # Perform the delete
                    cur.execute("DELETE FROM seekingalpha WHERE id = %s", (record_id,))
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    else:
                        st.error(f"Delete operation affected 0 rows for ID {record_id}")
                        return False
        except Exception as e:
            st.error(f"❌ Error deleting record {record_id}: {e}")
            return False
    
    def update_record(self, record_id: int, ticker: str = None, 
                     pdf_source: str = None, date_added: date = None,
                     price: float = None, exp: date = None, premiums: float = None,
                     strike: float = None, notional_value: float = None) -> bool:
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
                    if price is not None:  # Allow 0 as valid price
                        updates.append("price = %s")
                        params.append(price)
                    if exp:
                        updates.append("exp = %s")
                        params.append(exp)
                    if premiums is not None:  # Allow 0 as valid premiums
                        updates.append("premiums = %s")
                        params.append(premiums)
                    if strike is not None:
                        updates.append("strike = %s")
                        params.append(strike)
                    if notional_value is not None:
                        updates.append("notional_value = %s")
                        params.append(notional_value)
                    
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
    
    def upsert_record(self, ticker: str, pdf_source: str = None, 
                     date_added: date = None, price: float = None, exp: date = None, 
                     premiums: float = None, strike: float = None, notional_value: float = None) -> bool:
        """Create or update a record based on ticker and date_added"""
        date_added = date_added or datetime.now().date()
        ticker = ticker.upper().strip()
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Try to update existing record first
                    cur.execute("""
                        UPDATE seekingalpha 
                        SET pdf_source = COALESCE(%s, pdf_source),
                            price = COALESCE(%s, price),
                            exp = COALESCE(%s, exp),
                            premiums = COALESCE(%s, premiums),
                            strike = COALESCE(%s, strike),
                            notional_value = COALESCE(%s, notional_value)
                        WHERE ticker = %s AND date_added = %s
                    """, (pdf_source, price, exp, premiums, strike, notional_value, ticker, date_added))
                    
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    
                    # If no update, insert new record
                    cur.execute("""
                        INSERT INTO seekingalpha 
                        (ticker, date_added, pdf_source, price, exp, premiums, strike, notional_value)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date_added) 
                        DO UPDATE SET
                            pdf_source = COALESCE(EXCLUDED.pdf_source, seekingalpha.pdf_source),
                            price = COALESCE(EXCLUDED.price, seekingalpha.price),
                            exp = COALESCE(EXCLUDED.exp, seekingalpha.exp),
                            premiums = COALESCE(EXCLUDED.premiums, seekingalpha.premiums),
                            strike = COALESCE(EXCLUDED.strike, seekingalpha.strike),
                            notional_value = COALESCE(EXCLUDED.notional_value, seekingalpha.notional_value)
                    """, (ticker, date_added, pdf_source, price, exp, premiums, strike, notional_value))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            st.error(f"❌ Error upserting record: {e}")
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

def parse_date(date_str: str) -> Optional[date]:
    """Parse date string in various formats"""
    if not date_str or not date_str.strip():
        return None
    
    date_str = date_str.strip()
    
    # Common date formats
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%m-%d-%Y',
        '%d-%m-%Y',
        '%Y.%m.%d',
        '%m.%d.%Y',
        '%B %d, %Y',  # January 15, 2024
        '%b %d, %Y',  # Jan 15, 2024
        '%d %B %Y',   # 15 January 2024
        '%d %b %Y',   # 15 Jan 2024
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Try parsing common variations
    # Handle dates like "01/15/2024" or "15/01/2024"
    try:
        parts = re.split(r'[/\-\.]', date_str)
        if len(parts) == 3:
            # Try MM/DD/YYYY first
            try:
                month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return date(year if year > 1900 else year + 2000, month, day)
            except:
                pass
            # Try DD/MM/YYYY
            try:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return date(year if year > 1900 else year + 2000, month, day)
            except:
                pass
    except:
        pass
    
    return None

def parse_number(num_str: str) -> Optional[float]:
    """Parse number string, handling K/M suffixes and commas"""
    if not num_str or not num_str.strip():
        return None
    
    num_str = num_str.strip().replace(',', '').replace('$', '').replace(' ', '')
    
    try:
        # Handle K/M suffixes
        if num_str.upper().endswith('K'):
            return float(num_str[:-1]) * 1000
        elif num_str.upper().endswith('M'):
            return float(num_str[:-1]) * 1000000
        elif num_str.upper().endswith('B'):
            return float(num_str[:-1]) * 1000000000
        else:
            return float(num_str)
    except ValueError:
        return None

def extract_table_data_from_image(image_file) -> List[Dict]:
    """Extract structured table data (symbol, strike, expiration date, notional value) from image"""
    if not OCR_AVAILABLE:
        st.error("❌ OCR libraries not available. Please install: `pip install pytesseract` and `brew install tesseract` (macOS)")
        return []
    
    try:
        # Open the image
        image = Image.open(image_file)
        
        # Try structured data extraction first (tesseract with table structure)
        # Store OCR data with coordinates for position-based alignment
        ocr_data_dict = None
        text_lines = []
        
        try:
            # Use pytesseract with structured output
            ocr_data_dict = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract all text with positions
            current_line = []
            prev_y = None
            
            for i in range(len(ocr_data_dict['text'])):
                text = ocr_data_dict['text'][i].strip()
                if text:
                    y = ocr_data_dict['top'][i]
                    if prev_y is None or abs(y - prev_y) < 5:  # Same line
                        current_line.append(text)
                    else:  # New line
                        if current_line:
                            text_lines.append(' '.join(current_line))
                        current_line = [text]
                    prev_y = y
            
            if current_line:
                text_lines.append(' '.join(current_line))
            
            # If structured extraction didn't work well, fallback to simple text extraction
            if not text_lines or len(text_lines) < 2:
                full_text = pytesseract.image_to_string(image)
                text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                ocr_data_dict = None  # Clear coordinate data if we fall back
            
        except Exception as ocr_error:
            st.error(f"❌ OCR Error: {ocr_error}")
            st.info("💡 Make sure Tesseract is installed: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)")
            return []
        
        # Parse table - handle both columnar (vertical) and row-based formats
        records = []
        
        # First, try to detect if data is in columnar format (vertical columns)
        # This handles OCR output where columns are listed vertically
        ticker_values = []
        strike_values = []
        expiration_values = []
        notional_values = []
        
        current_column = None
        parsing_data = False
        
        for i, line in enumerate(text_lines):
            line = line.strip()
            if not line:
                continue
            
            line_upper = line.upper()
            
            # Detect column headers
            if 'TICKER' in line_upper or 'SYMBOL' in line_upper:
                current_column = 'ticker'
                parsing_data = True
                continue
            elif 'STRIKE' in line_upper:
                current_column = 'strike'
                parsing_data = True
                continue
            elif 'EXPIR' in line_upper or ('EXP' in line_upper and 'DATE' in line_upper):
                current_column = 'expiration'
                parsing_data = True
                continue
            elif 'NOTIONAL' in line_upper or ('VALUE' in line_upper and 'NOTIONAL' in line_upper):
                current_column = 'notional'
                parsing_data = True
                continue
            elif any(keyword in line_upper for keyword in ['TYPE', 'PUT', 'CALL']):
                # Skip option type column
                current_column = None
                parsing_data = True
                continue
            
            # Skip empty lines or header-like lines
            if len(line) < 2:
                continue
            
            # Collect data based on current column
            if current_column == 'ticker':
                # Handle $ prefix: $AAL -> AAL
                cleaned = line.replace('$', '').strip()
                if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', cleaned.upper()):
                    ticker_values.append(cleaned.upper())
            elif current_column == 'strike':
                # Parse strike price, be more lenient to catch missing values
                # Try multiple extraction methods
                strike_val = None
                
                # Method 1: Direct number match
                clean_line = line.replace(',', '').replace('$', '').strip()
                if re.match(r'^\d+\.?\d*$', clean_line):
                    strike_val = parse_number(clean_line)
                    if strike_val and 1 <= strike_val <= 10000:
                        strike_values.append(strike_val)
                        continue
                
                # Method 2: Extract first valid number from line (handles "13 35" or "40 525")
                number_match = re.search(r'\b(\d{1,2}\.?\d*)\b', clean_line)
                if number_match:
                    try:
                        potential_strike = float(number_match.group(1))
                        if 1 <= potential_strike <= 10000:
                            strike_val = potential_strike
                            strike_values.append(strike_val)
                            continue
                    except:
                        pass
                
                # Method 3: Look for numbers that might be split across characters (OCR errors)
                # Extract all digit sequences
                all_numbers = re.findall(r'\d+\.?\d*', clean_line)
                for num_str in all_numbers:
                    try:
                        num_val = float(num_str)
                        if 1 <= num_val <= 10000:
                            strike_val = num_val
                            strike_values.append(strike_val)
                            break
                    except:
                        continue
            elif current_column == 'expiration':
                parsed_date = parse_date(line)
                if parsed_date:
                    expiration_values.append(parsed_date)
            elif current_column == 'notional':
                notional_val = parse_number(line)
                if notional_val and notional_val >= 100:
                    notional_values.append(notional_val)
        
        # Post-process: If we have tickers but fewer strikes, try to find missing strikes
        # by scanning all text lines for potential strike values
        if ticker_values and len(strike_values) < len(ticker_values):
            # Collect all potential strikes from the entire OCR text
            all_potential_strikes = []
            
            # Re-scan text lines looking for numbers in the strike range
            in_strike_section = False
            for line in text_lines:
                line_upper = line.upper().strip()
                if 'STRIKE' in line_upper:
                    in_strike_section = True
                    continue
                elif in_strike_section and any(keyword in line_upper for keyword in ['EXPIR', 'NOTIONAL', 'TICKER', 'TYPE']):
                    in_strike_section = False
                
                if in_strike_section or 'STRIKE' in line_upper:
                    # Extract all numbers from this line (including lines with partial OCR)
                    # Look for standalone numbers or numbers that might be part of a sequence
                    numbers = re.findall(r'\b(\d{1,4}\.?\d*)\b', line.replace(',', '').replace('$', '').replace("'", ''))
                    for num_str in numbers:
                        try:
                            num_val = float(num_str)
                            if 1 <= num_val <= 10000:  # Reasonable strike range
                                all_potential_strikes.append(num_val)
                        except:
                            continue
                    
                    # Also check for numbers that might be in OCR gibberish (like "Q91'90'")
                    # Extract any 2-digit sequences that could be strikes
                    two_digit_patterns = re.findall(r'\b(\d{2})\b', line.replace(',', '').replace('$', ''))
                    for num_str in two_digit_patterns:
                        try:
                            num_val = float(num_str)
                            if 10 <= num_val <= 1000:  # Two-digit strikes like 13, 40
                                all_potential_strikes.append(num_val)
                        except:
                            continue
                    
                    # Also check single digits that might be part of larger numbers
                    # (like "4" could be "40" or part of "13")
                    single_digits = re.findall(r'\b(\d)\b(?!\.)', line.replace(',', '').replace('$', ''))
                    # Look for context - if there's a single digit followed by something, might be part of a number
                    for match in re.finditer(r'(\d)', line):
                        digit = match.group(1)
                        # Check surrounding context
                        start = max(0, match.start() - 2)
                        end = min(len(line), match.end() + 2)
                        context = line[start:end]
                        # If it's a standalone digit (not part of a larger number), consider it
                        if re.search(r'[^\d]' + re.escape(digit) + r'[^\d]', context):
                            try:
                                num_val = float(digit)
                                if 1 <= num_val <= 9:  # Single digit strikes (rare but possible)
                                    all_potential_strikes.append(num_val)
                            except:
                                pass
            
            # Also scan entire text for any numbers that could be missing strikes
            # Sometimes OCR misaligns columns
            for line in text_lines:
                # Look for patterns that might indicate a strike value
                # Check for numbers that appear near ticker symbols or in isolation
                numbers = re.findall(r'\b(\d{1,3}\.?\d*)\b', line.replace(',', '').replace('$', ''))
                for num_str in numbers:
                    try:
                        num_val = float(num_str)
                        # Look for values that are likely strikes (small to medium numbers)
                        # and not likely to be dates (not in common date ranges) or notional (too small)
                        if 1 <= num_val <= 1000 and num_val not in all_potential_strikes:
                            # Check if it's not part of a date (avoid matching year parts)
                            if not re.search(r'\d{4}', line):  # Not near a 4-digit number (year)
                                all_potential_strikes.append(num_val)
                    except:
                        continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_strikes = []
            for strike in all_potential_strikes:
                if strike not in seen:
                    seen.add(strike)
                    unique_strikes.append(strike)
            
            # If we found more strikes than before, try to intelligently merge
            # Prioritize known good strikes (those already captured), then add missing ones
            if len(unique_strikes) >= len(ticker_values):
                # We have enough strikes - merge intelligently
                # Start with known good strikes
                merged_strikes = []
                strike_set = set(strike_values)
                
                # Add original strikes first
                for orig_strike in strike_values:
                    merged_strikes.append(orig_strike)
                
                # Then add missing ones from unique_strikes
                for potential_strike in unique_strikes:
                    if potential_strike not in strike_set and len(merged_strikes) < len(ticker_values):
                        merged_strikes.append(potential_strike)
                
                # Fill any remaining slots
                while len(merged_strikes) < len(ticker_values) and len(unique_strikes) > len(merged_strikes):
                    for potential_strike in unique_strikes:
                        if potential_strike not in merged_strikes:
                            merged_strikes.append(potential_strike)
                            break
                
                strike_values = merged_strikes[:len(ticker_values)]
            elif len(unique_strikes) > len(strike_values):
                # Merge: preserve order of original strikes, add missing ones
                merged_strikes = []
                strike_set = set(strike_values)
                
                # Build merged list maintaining relative order where possible
                for orig_strike in strike_values:
                    merged_strikes.append(orig_strike)
                
                # Add missing strikes from unique_strikes
                for potential_strike in unique_strikes:
                    if potential_strike not in strike_set and len(merged_strikes) < len(ticker_values):
                        merged_strikes.append(potential_strike)
                
                strike_values = merged_strikes
        
        # If we found columnar data, match by index with offset detection
        if ticker_values:
            max_rows = len(ticker_values)
            
            # Use coordinate-based alignment if we have OCR coordinate data
            ticker_rows = []
            strike_rows = []
            rows_dict = {}
            sorted_rows = []
            
            if ocr_data_dict:
                # Group OCR words by their Y position (rows) and X position (columns)
                rows_dict = {}  # {y_position: {column_x: value}}
                
                for i in range(len(ocr_data_dict['text'])):
                    text = ocr_data_dict['text'][i].strip()
                    if not text:
                        continue
                    
                    x = ocr_data_dict['left'][i]
                    y = ocr_data_dict['top'][i]
                    conf = ocr_data_dict['conf'][i]
                    
                    # Skip low confidence results
                    if conf < 30:
                        continue
                    
                    # Round Y to group into rows (within 5 pixels = same row)
                    y_rounded = round(y / 5) * 5
                    
                    if y_rounded not in rows_dict:
                        rows_dict[y_rounded] = {}
                    
                    rows_dict[y_rounded][x] = text
                
                # Find the row positions for tickers and strikes
                sorted_rows = sorted(rows_dict.keys())
                
                for row_y in sorted_rows:
                    row_data = rows_dict[row_y]
                    row_values = sorted(row_data.items())  # Sort by X position
                    
                    for x, text in row_values:
                        text_upper = text.upper().replace('$', '')
                        # Check if this row contains a ticker
                        if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', text_upper):
                            ticker_rows.append((row_y, text_upper))
                        # Check if this row contains a strike (number in valid range)
                        elif re.match(r'^\d+\.?\d*$', text.replace(',', '').replace('$', '')):
                            try:
                                strike_val = float(text.replace(',', '').replace('$', ''))
                                if 1 <= strike_val <= 10000:
                                    strike_rows.append((row_y, strike_val))
                            except:
                                pass
                
                # Sort both by Y position
                ticker_rows.sort(key=lambda x: x[0])
                strike_rows.sort(key=lambda x: x[0])
                
                # Detect missing first strike - ALWAYS check if we have fewer strikes than tickers
                if len(ticker_rows) > 0 and len(strike_rows) < len(ticker_values):
                    # First strike is likely missing - scan for it aggressively
                    first_ticker_y = ticker_rows[0][0] if ticker_rows else None
                    first_strike_y = strike_rows[0][0] if strike_rows else None
                    
                    # Identify strike column X position by analyzing where strikes appear
                    strike_x_positions = []
                    for strike_y, strike_val in strike_rows:
                        # Find the X position of this strike in OCR data
                        for i in range(len(ocr_data_dict['text'])):
                            text = ocr_data_dict['text'][i].strip()
                            if not text:
                                continue
                            try:
                                clean_text = text.replace(',', '').replace('$', '').strip()
                                if abs(ocr_data_dict['top'][i] - strike_y) < 10:
                                    if float(clean_text) == strike_val:
                                        strike_x_positions.append(ocr_data_dict['left'][i])
                                        break
                            except:
                                pass
                    
                    # Average strike column X position
                    avg_strike_x = sum(strike_x_positions) / len(strike_x_positions) if strike_x_positions else None
                    
                    # Look for a number that matches the first ticker's row
                    best_candidate = None
                    best_y = None
                    best_score = float('inf')
                    
                    for i in range(len(ocr_data_dict['text'])):
                        text = ocr_data_dict['text'][i].strip()
                        if not text:
                            continue
                        
                        y = ocr_data_dict['top'][i]
                        x = ocr_data_dict['left'][i]
                        conf = ocr_data_dict['conf'][i]
                        
                        if conf < 30:
                            continue
                        
                        # Check if it's a valid strike number (broader range for first strike)
                        clean_text = text.replace(',', '').replace('$', '').strip()
                        if re.match(r'^\d{1,4}\.?\d*$', clean_text):
                            try:
                                strike_val = float(clean_text)
                                # First strike is usually smaller, but allow up to 1000
                                if 1 <= strike_val <= 1000 and strike_val not in strike_values:
                                    score = float('inf')
                                    
                                    # Prefer numbers that:
                                    # 1. Are on the same row as first ticker (Y position match)
                                    if first_ticker_y:
                                        y_distance = abs(y - first_ticker_y)
                                        if y_distance < 30:  # Within 30 pixels = same row
                                            score = y_distance
                                    
                                    # 2. Are in the strike column area (X position match)
                                    if avg_strike_x:
                                        x_distance = abs(x - avg_strike_x)
                                        if score != float('inf'):
                                            score += x_distance * 0.1  # Prefer same column but not critical
                                        elif abs(x - avg_strike_x) < 100:  # If no Y match, use X
                                            score = abs(x - avg_strike_x) + 1000
                                    
                                    # 3. Are positioned before the first known strike
                                    if first_strike_y and y < first_strike_y:
                                        if score != float('inf'):
                                            score -= 50  # Bonus for being before first strike
                                        else:
                                            score = abs(y - first_ticker_y) + 100 if first_ticker_y else 200
                                    
                                    # Pick the best candidate (lowest score)
                                    if score < best_score:
                                        best_candidate = strike_val
                                        best_y = y
                                        best_score = score
                            except:
                                pass
                    
                    # If we found a candidate, add it
                    if best_candidate is not None:
                        strike_values.insert(0, best_candidate)
                        strike_rows.insert(0, (best_y, best_candidate))
                        # Re-sort strike_rows after insertion
                        strike_rows.sort(key=lambda x: x[0])
                    
                    # Final fallback: If still missing, do an even more aggressive search
                    if len(strike_values) < len(ticker_values) and first_ticker_y:
                        # Scan ALL numbers near the first ticker row, regardless of column
                        for i in range(len(ocr_data_dict['text'])):
                            text = ocr_data_dict['text'][i].strip()
                            if not text:
                                continue
                            
                            y = ocr_data_dict['top'][i]
                            conf = ocr_data_dict['conf'][i]
                            
                            if conf < 20:  # Even lower confidence threshold
                                continue
                            
                            # If it's very close to first ticker row, check if it's a number
                            if abs(y - first_ticker_y) < 40:
                                clean_text = text.replace(',', '').replace('$', '').strip()
                                if re.match(r'^\d{1,3}\.?\d*$', clean_text):
                                    try:
                                        num_val = float(clean_text)
                                        if 1 <= num_val <= 500 and num_val not in strike_values:
                                            # Found a number on first ticker row - likely the missing strike
                                            strike_values.insert(0, num_val)
                                            strike_rows.insert(0, (y, num_val))
                                            strike_rows.sort(key=lambda x: x[0])
                                            break  # Found it, stop searching
                                    except:
                                        pass
            
            # Fallback: If we still have fewer strikes and no coordinate data, try text-based search
            if len(strike_values) < len(ticker_values) and not ocr_data_dict:
                # Scan for potential first strike values
                potential_first_strikes = []
                for line in text_lines[:50]:
                    numbers = re.findall(r'\b(\d{1,2}\.?\d*)\b', line.replace(',', '').replace('$', ''))
                    for num_str in numbers:
                        try:
                            num_val = float(num_str)
                            if 1 <= num_val <= 100 and num_val not in strike_values:
                                potential_first_strikes.append(num_val)
                        except:
                            continue
                
                if potential_first_strikes and len(strike_values) == len(ticker_values) - 1:
                    first_strike_candidate = min([s for s in potential_first_strikes if 1 <= s <= 100])
                    if first_strike_candidate:
                        strike_values.insert(0, first_strike_candidate)
            
            # Match records with proper alignment using coordinate-based matching if available
            # Use coordinate matching if we have OCR data AND found tickers/strikes in rows
            # Otherwise, fall back to index-based matching
            use_coordinate_matching = ocr_data_dict and len(ticker_rows) > 0 and len(strike_rows) > 0
            
            if use_coordinate_matching:
                # Use coordinate-based matching: match tickers and strikes by row Y position
                matched_records = {}
                
                # Create maps of ticker and strike positions
                for ticker_y, ticker_name in ticker_rows:
                    # Find the strike on the same row (or nearest row)
                    best_strike = None
                    best_distance = float('inf')
                    
                    for strike_y, strike_val in strike_rows:
                        distance = abs(strike_y - ticker_y)
                        if distance < best_distance and distance < 20:  # Within 20 pixels = same row
                            best_distance = distance
                            best_strike = strike_val
                    
                    matched_records[ticker_name] = best_strike
                
                # Also match expiration and notional by row
                expiration_rows = []
                notional_rows = []
                
                for row_y in sorted_rows:
                    row_data = rows_dict[row_y]
                    row_values = sorted(row_data.items())
                    
                    for x, text in row_values:
                        # Check for expiration date
                        parsed_date = parse_date(text)
                        if parsed_date:
                            expiration_rows.append((row_y, parsed_date))
                        # Check for notional value
                        notional_val = parse_number(text)
                        if notional_val and notional_val >= 100:
                            notional_rows.append((row_y, notional_val))
                
                # Build records using coordinate-matched values
                for i, ticker in enumerate(ticker_values):
                    # Find strike for this ticker using coordinate matching
                    strike_val = matched_records.get(ticker.upper())
                    
                    # If first ticker has no strike and we have one fewer strikes, try to find it
                    if i == 0 and strike_val is None and len(ticker_values) > len(strike_rows):
                        # Last resort: find any number on the first ticker's row
                        first_ticker_y = ticker_rows[0][0] if ticker_rows else None
                        if first_ticker_y:
                            for j in range(len(ocr_data_dict['text'])):
                                text = ocr_data_dict['text'][j].strip()
                                if not text:
                                    continue
                                y = ocr_data_dict['top'][j]
                                conf = ocr_data_dict['conf'][j]
                                if conf < 20:
                                    continue
                                if abs(y - first_ticker_y) < 40:
                                    clean_text = text.replace(',', '').replace('$', '').strip()
                                    if re.match(r'^\d{1,3}\.?\d*$', clean_text):
                                        try:
                                            num_val = float(clean_text)
                                            if 1 <= num_val <= 500:
                                                strike_val = num_val
                                                break
                                        except:
                                            pass
                    
                    # Fallback: if coordinate matching didn't find strike, use index-based matching
                    if strike_val is None and i < len(strike_values):
                        strike_val = strike_values[i]
                    
                    # Find expiration and notional on same row
                    ticker_pos = None
                    for ticker_y, ticker_name in ticker_rows:
                        if ticker_name == ticker.upper():
                            ticker_pos = ticker_y
                            break
                    
                    exp_val = None
                    notional_val = None
                    if ticker_pos:
                        # Find closest expiration and notional on similar row
                        for exp_y, exp_date in expiration_rows:
                            if abs(exp_y - ticker_pos) < 20:
                                exp_val = exp_date
                                break
                        
                        for not_y, not_val in notional_rows:
                            if abs(not_y - ticker_pos) < 20:
                                notional_val = not_val
                                break
                    
                    # Fallback: if coordinate matching didn't find these, use index-based
                    if exp_val is None and i < len(expiration_values):
                        exp_val = expiration_values[i]
                    if notional_val is None and i < len(notional_values):
                        notional_val = notional_values[i]
                    
                    record = {
                        'symbol': ticker,
                        'strike': strike_val,
                        'expiration': exp_val,
                        'notional_value': notional_val
                    }
                    records.append(record)
            else:
                # Fallback to index-based matching
                for i in range(max_rows):
                    if i < len(ticker_values):
                        # Calculate strike index - handle offset if strikes start from index 1
                        strike_idx = i
                        # If we have one fewer strike than tickers, and we're at index 0, strike might be None
                        if len(strike_values) == len(ticker_values) - 1 and i == 0:
                            strike_val = None
                        elif i < len(strike_values):
                            strike_idx = i
                            strike_val = strike_values[strike_idx]
                        else:
                            strike_val = None
                        
                        record = {
                            'symbol': ticker_values[i],
                            'strike': strike_val,
                            'expiration': expiration_values[i] if i < len(expiration_values) else None,
                            'notional_value': notional_values[i] if i < len(notional_values) else None
                        }
                        records.append(record)
        
        # If columnar parsing didn't work, try row-based parsing
        if not records:
            # Look for header row to identify column positions
            header_found = False
            header_line_idx = -1
            column_order = []  # Track column order from header
            
            for i, line in enumerate(text_lines):
                line_upper = line.upper()
                # Check if this line contains table headers
                if any(keyword in line_upper for keyword in ['TICKER', 'SYMBOL', 'STRIKE', 'EXPIR', 'EXP', 'NOTIONAL', 'VALUE']):
                    header_found = True
                    header_line_idx = i
                    
                    # Try to identify column positions
                    header_parts = re.split(r'\t+|\s{2,}|\||,', line)
                    header_parts = [p.strip().upper() for p in header_parts if p.strip()]
                    
                    for part in header_parts:
                        if 'TICKER' in part or 'SYMBOL' in part:
                            column_order.append('ticker')
                        elif 'STRIKE' in part:
                            column_order.append('strike')
                        elif 'TYPE' in part:
                            column_order.append('type')
                        elif 'EXPIR' in part or 'EXP' in part:
                            column_order.append('expiration')
                        elif 'NOTIONAL' in part or 'VALUE' in part:
                            column_order.append('notional')
                    
                    break
            
            # Start parsing from after header (or from beginning if no header found)
            start_idx = header_line_idx + 1 if header_found else 0
            
            for line in text_lines[start_idx:]:
                line = line.strip()
                if not line or len(line) < 2:
                    continue
                
                # Skip lines that look like headers or separators
                if any(keyword in line.upper() for keyword in ['TICKER', 'SYMBOL', 'STRIKE', 'EXPIR', 'TYPE', 'NOTIONAL', 'VALUE', '---', '===']):
                    continue
                
                # Try to split line into columns (handle various separators)
                # Common separators: tabs, multiple spaces, pipes, commas
                parts = re.split(r'\t+|\s{2,}|\|', line)
                # If split didn't work well, try splitting on single space but only if we have many parts
                if len(parts) < 3:
                    # Try splitting on any whitespace but preserve multi-space separators
                    parts = re.split(r'\s{2,}|(?<=\$)(?=\w)|(?<=\d)(?=\$)', line)
                
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) < 2:  # Need at least symbol and one other field
                    continue
                
                record = {}
                
                # If we have column order from header, use positional mapping
                if column_order and len(parts) >= len(column_order):
                    for idx, col_type in enumerate(column_order):
                        if idx < len(parts):
                            part = parts[idx].strip()
                            if col_type == 'ticker':
                                # Handle $ prefix: $AAL -> AAL
                                part = part.replace('$', '').strip()
                                if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', part.upper()):
                                    record['symbol'] = part.upper()
                            elif col_type == 'strike':
                                strike_val = parse_number(part)
                                if strike_val and strike_val > 0:
                                    record['strike'] = strike_val
                            elif col_type == 'expiration':
                                parsed_date = parse_date(part)
                                if parsed_date:
                                    record['expiration'] = parsed_date
                            elif col_type == 'notional':
                                notional_val = parse_number(part)
                                if notional_val and notional_val >= 100:  # Allow values >= 100 (could be 100K = 100000)
                                    record['notional'] = notional_val
                            # Skip 'type' column
                else:
                    # Fallback: pattern-based identification
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        # Symbol/ticker (handle $ prefix: $AAL -> AAL)
                        if not record.get('symbol'):
                            cleaned_part = part.replace('$', '').strip()
                            if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', cleaned_part.upper()):
                                record['symbol'] = cleaned_part.upper()
                                continue
                        
                        # Strike price (number, possibly with decimals)
                        if not record.get('strike'):
                            # Check if it's a number (but not a date or notional value)
                            if re.match(r'^\d+\.?\d*$', part.replace(',', '').replace('$', '')):
                                strike_val = parse_number(part)
                                # Strike should be reasonable (between 1 and 10000 typically)
                                if strike_val and 1 <= strike_val <= 10000:
                                    record['strike'] = strike_val
                                    continue
                        
                        # Skip single letter P or C (option type)
                        if part.upper() in ['P', 'C', 'PUT', 'CALL']:
                            continue
                        
                        # Expiration date (contains date-like patterns)
                        if not record.get('expiration'):
                            parsed_date = parse_date(part)
                            if parsed_date:
                                record['expiration'] = parsed_date
                                continue
                        
                        # Notional value (large numbers with K/M/B, usually > 1000)
                        if not record.get('notional'):
                            notional_val = parse_number(part)
                            if notional_val and notional_val >= 1000:  # Notional values are usually in thousands or millions
                                record['notional'] = notional_val
                
                # Only add record if we have at least a symbol
                if record.get('symbol'):
                    records.append({
                        'symbol': record.get('symbol'),
                        'strike': record.get('strike'),
                        'expiration': record.get('expiration'),
                        'notional_value': record.get('notional')
                    })
        
        # Final safety check: If we have ticker_values but no records were created,
        # create records from the columnar data using index-based matching
        if not records and ticker_values:
            # Debug: Log what we found (only if records is still empty)
            # This should have been handled earlier, but add as final fallback
            for i in range(len(ticker_values)):
                record = {
                    'symbol': ticker_values[i],
                    'strike': strike_values[i] if i < len(strike_values) else None,
                    'expiration': expiration_values[i] if i < len(expiration_values) else None,
                    'notional_value': notional_values[i] if i < len(notional_values) else None
                }
                records.append(record)
        
        return records
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
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
                   background: bool = True, rebuild_models: bool = False, use_ml: bool = True) -> Dict:
    """
    Run scanner.py predictions on the database tickers
    
    Args:
        date_filter: Date filter for predictions ("latest", "today", specific date)
        max_tickers: Maximum number of tickers to process (None = process all)
        background: Whether to run in background thread
        rebuild_models: Force rebuild of ML models even if existing ones are found
        use_ml: Use ML predictions instead of confidence-based predictions
        
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
        
        # Add ML prediction options
        if use_ml:
            cmd.append("--use-ml")
        else:
            cmd.append("--no-ml")
        
        # Add rebuild models option
        if rebuild_models:
            cmd.append("--rebuild-models")
        
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
            st.session_state.table_sort_by = 'id'
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
        header_cols = st.columns([1, 3, 2, 2, 2, 2])
        
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
            if st.button(f"**Price**{get_sort_indicator('price')}", key="sort_price", help="Click to sort by Price", use_container_width=True):
                handle_sort_click('price')
                st.rerun()
                
        with header_cols[3]:
            if st.button(f"**Exp Date**{get_sort_indicator('exp')}", key="sort_exp", help="Click to sort by Exp Date", use_container_width=True):
                handle_sort_click('exp')
                st.rerun()
                
        with header_cols[4]:
            if st.button(f"**Premiums**{get_sort_indicator('premiums')}", key="sort_premiums", help="Click to sort by Premiums", use_container_width=True):
                handle_sort_click('premiums')
                st.rerun()
                
        with header_cols[5]:
            col_action1, col_action2 = st.columns([1, 1])
            with col_action1:
                st.markdown("**Actions**")
            with col_action2:
                if st.button("🔄", key="reset_sort", help="Reset to default sort (ID desc)", use_container_width=True):
                    st.session_state.table_sort_by = 'id'
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
            cols = st.columns([1, 3, 2, 2, 2, 2])
            
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
                # Editable price
                current_price = record.get('price', 0.0) if record.get('price') is not None else 0.0
                new_price = st.number_input(
                    "price",
                    value=float(current_price),
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"price_{record_id}",
                    label_visibility="collapsed",
                    placeholder="Price..."
                )
                if new_price != current_price:
                    if record_id not in st.session_state.pending_updates:
                        st.session_state.pending_updates[record_id] = {}
                    st.session_state.pending_updates[record_id]['price'] = new_price
                    changes_made = True
            
            with cols[3]:
                # Editable exp date
                current_exp = record.get('exp')
                new_exp = st.date_input(
                    "exp",
                    value=current_exp,
                    key=f"exp_{record_id}",
                    label_visibility="collapsed"
                )
                if new_exp != current_exp:
                    if record_id not in st.session_state.pending_updates:
                        st.session_state.pending_updates[record_id] = {}
                    st.session_state.pending_updates[record_id]['exp'] = new_exp
                    changes_made = True
            
            with cols[4]:
                # Editable premiums
                current_premiums = record.get('premiums', 0.0) if record.get('premiums') is not None else 0.0
                # Format current value for display
                if current_premiums >= 1000000:
                    display_value = f"{current_premiums/1000000:.1f}M"
                elif current_premiums >= 1000:
                    display_value = f"{current_premiums/1000:.1f}K"
                else:
                    display_value = f"{current_premiums:.2f}" if current_premiums > 0 else ""
                
                new_premiums_str = st.text_input(
                    "premiums",
                    value=display_value,
                    key=f"premiums_{record_id}",
                    label_visibility="collapsed",
                    placeholder="e.g. 1.5K, 2.3M, 150"
                )
                new_premiums = parse_number_with_suffix(new_premiums_str)
                if new_premiums != current_premiums:
                    if record_id not in st.session_state.pending_updates:
                        st.session_state.pending_updates[record_id] = {}
                    st.session_state.pending_updates[record_id]['premiums'] = new_premiums
                    changes_made = True
            
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
                    update_success = 0
                    delete_success = 0
                    error_count = 0
                    
                    # Process updates
                    for record_id, updates in st.session_state.pending_updates.items():
                        try:
                            if manager.update_record(
                                record_id=record_id,
                                ticker=updates.get('ticker'),
                                pdf_source=updates.get('pdf_source'),
                                date_added=updates.get('date_added'),
                                price=updates.get('price'),
                                exp=updates.get('exp'),
                                premiums=updates.get('premiums')
                            ):
                                update_success += 1
                            else:
                                st.error(f"Failed to update record {record_id}")
                                error_count += 1
                        except Exception as e:
                            st.error(f"Error updating record {record_id}: {e}")
                            error_count += 1
                    
                    # Process deletions
                    if st.session_state.pending_deletes:
                        st.info(f"Attempting to delete {len(st.session_state.pending_deletes)} records...")
                    
                    for record_id in st.session_state.pending_deletes:
                        try:
                            st.info(f"Deleting record ID: {record_id}")
                            if manager.delete_record(record_id):
                                delete_success += 1
                                st.success(f"Successfully deleted record {record_id}")
                            else:
                                st.error(f"Failed to delete record {record_id}")
                                error_count += 1
                        except Exception as e:
                            st.error(f"Error deleting record {record_id}: {e}")
                            error_count += 1
                    
                    # Clear pending changes
                    st.session_state.pending_updates = {}
                    st.session_state.pending_deletes = set()
                    st.cache_data.clear()
                    
                    # Show results
                    if update_success > 0:
                        st.success(f"✅ Updated {update_success} records!")
                    if delete_success > 0:
                        st.success(f"✅ Deleted {delete_success} records!")
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
                    else:
                        st.warning(f"⚠️ {quick_ticker} already exists in database")

def add_records_page(manager):
    """Page for adding new records"""
    st.header("➕ Add Records")
    
    # Single ticker addition
    st.subheader("📝 Add Single Ticker")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL").upper()
    with col2:
        pdf_source = st.text_input("PDF Source", placeholder="source.pdf")
    with col3:
        date_added = st.date_input("Date Added", value=datetime.now().date())
    with col4:
        price = st.number_input("Price", min_value=0.0, step=0.01, format="%.2f", placeholder="0.00")
    with col5:
        exp = st.date_input("Exp Date", value=None)
    with col6:
        premiums_str = st.text_input("Premiums", placeholder="e.g. 1.5K, 2.3M, 150")
        premiums = parse_number_with_suffix(premiums_str) if premiums_str else 0.0
    
    # Show help text for premiums format
    st.info("💡 Premiums format: Use K for thousands (1.5K = $1,500) or M for millions (2.3M = $2,300,000)")
    
    if st.button("➕ Add Ticker", type="primary", disabled=not ticker):
        if manager.add_ticker(ticker, pdf_source, date_added, price if price > 0 else None, exp, premiums if premiums > 0 else None):
            st.success(f"✅ Added {ticker} successfully!")
            st.cache_data.clear()
        else:
            st.warning(f"⚠️ {ticker} already exists in database")
    
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
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
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
                    duplicate_count = 0
                    total_count = len(final_symbols)
                    
                    for i, symbol in enumerate(final_symbols):
                        status_text.text(f"Adding {symbol}... ({i+1}/{total_count})")
                        if manager.add_ticker(symbol, image_source, image_date):
                            added_count += 1
                        else:
                            duplicate_count += 1
                        progress_bar.progress((i + 1) / total_count)
                    
                    status_text.text(f"✅ Complete! Added {added_count} new symbols. {duplicate_count} duplicates skipped.")
                    if duplicate_count > 0:
                        st.info(f"ℹ️ {duplicate_count} symbols were already in the database and were skipped.")
                    
                    st.cache_data.clear()
                    
                    # Clear the session state
                    if 'extracted_symbols' in st.session_state:
                        del st.session_state.extracted_symbols
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
    
    # Enhanced table data extraction
    st.subheader("📊 Extract Table Data from Image (Symbol, Strike, Expiration, Notional Value)")
    
    if not OCR_AVAILABLE:
        st.warning("⚠️ OCR functionality not available. To enable table data extraction, install: `pip install pytesseract` and system Tesseract OCR")
    else:
        st.markdown("""
        Upload an image containing a **table** with headers: Ticker/Symbol, Strike, Expiration Date, and Notional Value.
        The system will automatically extract all fields and create or update records.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_table_image = st.file_uploader(
                "Choose an image file with table data",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image containing a table with ticker, strike, expiration date, and notional value",
                key="table_image_uploader"
            )
            
            if uploaded_table_image is not None:
                # Display the uploaded image
                st.image(uploaded_table_image, caption="Uploaded Table Image", use_container_width=True)
        
        with col2:
            # Extract table data button
            if uploaded_table_image is not None:
                if st.button("🔍 Extract Table Data", type="primary", key="extract_table_btn"):
                    with st.spinner("Extracting table data from image..."):
                        # Reset file pointer for multiple reads
                        uploaded_table_image.seek(0)
                        
                        # Show OCR raw text for debugging
                        if OCR_AVAILABLE:
                            try:
                                image = Image.open(uploaded_table_image)
                                raw_text = pytesseract.image_to_string(image)
                                
                                with st.expander("🔍 View OCR Raw Text (for debugging)"):
                                    st.text_area("Raw OCR Output", raw_text, height=200, key="ocr_debug")
                                
                                # Reset file pointer again for extraction
                                uploaded_table_image.seek(0)
                            except Exception as e:
                                st.warning(f"Could not show OCR debug text: {e}")
                        
                        extracted_records = extract_table_data_from_image(uploaded_table_image)
                        
                        if extracted_records:
                            st.session_state.extracted_table_records = extracted_records
                            st.success(f"✅ Found {len(extracted_records)} records!")
                        else:
                            st.warning("⚠️ No table data found in the image. Make sure the image contains a table with headers.")
                            st.info("💡 Try checking the OCR Raw Text above to see what was extracted. You may need to improve image quality.")
            
            # Display and manage extracted table records
            if 'extracted_table_records' in st.session_state and st.session_state.extracted_table_records:
                st.write("**Extracted Table Records (Editable):**")
                st.caption("💡 Review and edit the extracted data below. Fix any errors before importing.")
                
                # Initialize edited records in session state if not exists
                if 'edited_table_records' not in st.session_state:
                    st.session_state.edited_table_records = st.session_state.extracted_table_records.copy()
                
                # Create editable table interface
                with st.expander("📋 Edit Records", expanded=True):
                    edited_records = []
                    
                    # Table header
                    header_cols = st.columns([2, 2, 2, 2, 1])
                    with header_cols[0]:
                        st.markdown("**Symbol**")
                    with header_cols[1]:
                        st.markdown("**Strike**")
                    with header_cols[2]:
                        st.markdown("**Expiration Date**")
                    with header_cols[3]:
                        st.markdown("**Notional Value**")
                    with header_cols[4]:
                        st.markdown("**Actions**")
                    
                    st.divider()
                    
                    # Editable rows
                    for idx, record in enumerate(st.session_state.edited_table_records):
                        row_cols = st.columns([2, 2, 2, 2, 1])
                        
                        # Symbol input
                        with row_cols[0]:
                            symbol = st.text_input(
                                "Symbol",
                                value=str(record.get('symbol', '')),
                                key=f"edit_symbol_{idx}",
                                label_visibility="collapsed"
                            )
                        
                        # Strike input
                        with row_cols[1]:
                            strike_val = record.get('strike')
                            strike_str = str(strike_val) if strike_val is not None else ""
                            strike = st.text_input(
                                "Strike",
                                value=strike_str,
                                key=f"edit_strike_{idx}",
                                label_visibility="collapsed",
                                placeholder="e.g., 13"
                            )
                            strike_parsed = None
                            if strike.strip():
                                try:
                                    strike_parsed = float(strike.strip())
                                except:
                                    pass
                        
                        # Expiration date input
                        with row_cols[2]:
                            exp_val = record.get('expiration')
                            exp_str = ""
                            if exp_val:
                                if isinstance(exp_val, date):
                                    exp_str = exp_val.strftime('%Y-%m-%d')
                                else:
                                    exp_str = str(exp_val)
                            exp = st.text_input(
                                "Expiration",
                                value=exp_str,
                                key=f"edit_exp_{idx}",
                                label_visibility="collapsed",
                                placeholder="MM/DD/YYYY"
                            )
                            exp_parsed = None
                            if exp.strip():
                                exp_parsed = parse_date(exp.strip())
                        
                        # Notional value input
                        with row_cols[3]:
                            notional_val = record.get('notional_value')
                            notional_str = ""
                            if notional_val is not None:
                                notional_str = str(notional_val)
                            notional = st.text_input(
                                "Notional",
                                value=notional_str,
                                key=f"edit_notional_{idx}",
                                label_visibility="collapsed",
                                placeholder="e.g., 1.2M"
                            )
                            notional_parsed = None
                            if notional.strip():
                                notional_parsed = parse_number(notional.strip())
                        
                        # Delete button
                        with row_cols[4]:
                            if st.button("🗑️", key=f"delete_row_{idx}", help="Delete this record"):
                                # Mark for deletion
                                st.session_state.edited_table_records.pop(idx)
                                st.rerun()
                        
                        # Store edited record
                        edited_records.append({
                            'symbol': symbol.strip().upper() if symbol.strip() else None,
                            'strike': strike_parsed,
                            'expiration': exp_parsed,
                            'notional_value': notional_parsed
                        })
                    
                    # Update session state with edited records
                    st.session_state.edited_table_records = edited_records
                    
                    # Action buttons
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        if st.button("➕ Add New Row", key="add_row_btn", use_container_width=True):
                            st.session_state.edited_table_records.append({
                                'symbol': None,
                                'strike': None,
                                'expiration': None,
                                'notional_value': None
                            })
                            st.rerun()
                    
                    with action_col2:
                        if st.button("🔄 Reset to Original", key="reset_edited_btn", use_container_width=True, help="Reset all edits back to original extracted data"):
                            if 'extracted_table_records' in st.session_state:
                                st.session_state.edited_table_records = st.session_state.extracted_table_records.copy()
                            st.rerun()
                
                # Show summary of editable records
                valid_records = [r for r in st.session_state.edited_table_records if r.get('symbol')]
                st.write(f"**Ready to import:** {len(valid_records)} valid records (out of {len(st.session_state.edited_table_records)} total)")
                
                # Show preview of changes (simplified check)
                if 'extracted_table_records' in st.session_state:
                    original_count = len(st.session_state.extracted_table_records)
                    edited_count = len(st.session_state.edited_table_records)
                    if original_count != edited_count:
                        st.info(f"⚠️ Records changed: {original_count} → {edited_count}. Review before importing.")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    table_source = st.text_input("PDF Source (for all)", placeholder="table_extract.pdf", value="table_extracted", key="table_source")
                with col_b:
                    table_date = st.date_input("Date Added (for all)", value=datetime.now().date(), key="table_date")
                
                # Import the table records (use edited records)
                if st.button("📥 Import Table Records", type="primary", key="import_table_btn"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Track detailed results
                    created_records = []
                    updated_records = []
                    error_records = []
                    skipped_records = []
                    
                    # Use edited records for import
                    records_to_import = st.session_state.edited_table_records
                    total_count = len(records_to_import)
                    
                    for i, record in enumerate(records_to_import):
                        symbol = record.get('symbol', '').strip() if record.get('symbol') else ''
                        
                        # Skip records without symbols
                        if not symbol:
                            skipped_records.append({
                                'symbol': symbol or '(empty)',
                                'reason': 'Missing symbol',
                                'strike': record.get('strike'),
                                'expiration': record.get('expiration'),
                                'notional_value': record.get('notional_value')
                            })
                            continue
                        
                        status_text.text(f"Processing {symbol}... ({i+1}/{total_count})")
                        
                        try:
                            # Check if record exists before upsert
                            existed_before = manager.ticker_exists(symbol)
                            
                            if manager.upsert_record(
                                ticker=symbol,
                                pdf_source=table_source,
                                date_added=table_date,
                                price=record.get('strike'),  # Using strike as price
                                exp=record.get('expiration'),
                                strike=record.get('strike'),
                                notional_value=record.get('notional_value')
                            ):
                                record_info = {
                                    'symbol': symbol,
                                    'strike': record.get('strike'),
                                    'expiration': record.get('expiration'),
                                    'notional_value': record.get('notional_value'),
                                    'pdf_source': table_source,
                                    'date_added': table_date
                                }
                                if existed_before:
                                    updated_records.append(record_info)
                                else:
                                    created_records.append(record_info)
                            else:
                                error_records.append({
                                    'symbol': symbol,
                                    'reason': 'Upsert returned False',
                                    'strike': record.get('strike'),
                                    'expiration': record.get('expiration'),
                                    'notional_value': record.get('notional_value')
                                })
                        except Exception as e:
                            error_records.append({
                                'symbol': symbol,
                                'reason': str(e),
                                'strike': record.get('strike'),
                                'expiration': record.get('expiration'),
                                'notional_value': record.get('notional_value')
                            })
                        
                        progress_bar.progress((i + 1) / total_count)
                    
                    # Clear progress bar
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display detailed execution report
                    st.success("✅ Import Complete!")
                    st.divider()
                    
                    # Summary statistics
                    summary_cols = st.columns(4)
                    with summary_cols[0]:
                        st.metric("✅ Created", len(created_records))
                    with summary_cols[1]:
                        st.metric("🔄 Updated", len(updated_records))
                    with summary_cols[2]:
                        st.metric("⚠️ Errors", len(error_records))
                    with summary_cols[3]:
                        st.metric("⏭️ Skipped", len(skipped_records))
                    
                    st.divider()
                    
                    # Detailed results in expandable sections
                    if created_records:
                        with st.expander(f"✅ Created Records ({len(created_records)})", expanded=True):
                            df_created = pd.DataFrame(created_records)
                            st.dataframe(df_created, use_container_width=True, hide_index=True)
                    
                    if updated_records:
                        with st.expander(f"🔄 Updated Records ({len(updated_records)})", expanded=True):
                            df_updated = pd.DataFrame(updated_records)
                            st.dataframe(df_updated, use_container_width=True, hide_index=True)
                    
                    if error_records:
                        with st.expander(f"⚠️ Error Records ({len(error_records)})", expanded=True):
                            df_errors = pd.DataFrame(error_records)
                            st.dataframe(df_errors, use_container_width=True, hide_index=True)
                            st.warning("Please review these records and fix any issues before re-importing.")
                    
                    if skipped_records:
                        with st.expander(f"⏭️ Skipped Records ({len(skipped_records)})", expanded=False):
                            df_skipped = pd.DataFrame(skipped_records)
                            st.dataframe(df_skipped, use_container_width=True, hide_index=True)
                            st.info("These records were skipped because they don't have valid symbols.")
                    
                    # Overall summary
                    total_processed = len(created_records) + len(updated_records)
                    total_records = len(records_to_import)
                    
                    if total_processed > 0:
                        success_rate = (total_processed / total_records) * 100
                        st.info(f"📊 **Summary**: {total_processed} out of {total_records} records processed successfully ({success_rate:.1f}% success rate)")
                    
                    st.cache_data.clear()
                    
                    # Clear the session state
                    if 'extracted_table_records' in st.session_state:
                        del st.session_state.extracted_table_records
                    if 'edited_table_records' in st.session_state:
                        del st.session_state.edited_table_records
                    
                    # Add option to continue or view database
                    st.divider()
                    col_view, col_clear = st.columns(2)
                    with col_view:
                        if st.button("👁️ View Database", key="view_db_after_import"):
                            st.session_state.page = 'view'
                            st.rerun()
                    with col_clear:
                        if st.button("🔄 Import More", key="import_more_btn"):
                            st.rerun()
            else:
                st.info("📷 Upload a table image above to extract structured data")
                st.markdown("""
                **Table format requirements:**
                - Headers: Ticker/Symbol, Strike, Expiration Date, Notional Value
                - Clear table structure (rows and columns)
                - High quality image for best OCR results
                
                **Tips:**
                - Ensure table headers are clearly visible
                - Use high contrast between text and background
                - Align the table properly in the image
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
        duplicate_count = 0
        total_count = len(tickers_list)
        
        for i, ticker in enumerate(tickers_list):
            status_text.text(f"Adding {ticker}... ({i+1}/{total_count})")
            if manager.add_ticker(ticker, bulk_source, bulk_date):
                added_count += 1
            else:
                duplicate_count += 1
            progress_bar.progress((i + 1) / total_count)
        
        status_text.text(f"✅ Complete! Added {added_count} new tickers. {duplicate_count} duplicates skipped.")
        if duplicate_count > 0:
            st.info(f"ℹ️ {duplicate_count} tickers were already in the database and were skipped.")
        
        st.cache_data.clear()

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_filter = st.selectbox(
            "📅 Date Filter",
            ["latest", "today"] + [(datetime.now().date() - timedelta(days=i)).strftime("%Y-%m-%d") 
                                  for i in range(1, 8)],
            help="Which tickers to run predictions on"
        )
    
    with col2:
        background_mode = st.checkbox(
            "🔄 Background Mode",
            value=True,
            help="Run predictions in background (recommended for large batches)"
        )
    
    # Add rebuild models option
    col3, col4 = st.columns(2)
    
    with col3:
        rebuild_models = st.checkbox(
            "🔨 Force Rebuild Models",
            value=False,
            help="Force rebuild of all ML models even if existing models are found"
        )
    
    with col4:
        use_ml = st.checkbox(
            "🤖 Use ML Predictions",
            value=True,
            help="Use LightGBM ML predictions instead of confidence-based predictions"
        )
    
    # Show status indicators
    if rebuild_models:
        st.info("🔨 **Rebuild Mode Active** - All ML models will be retrained from scratch")
    
    if not use_ml:
        st.warning("⚠️ **ML Predictions Disabled** - Using confidence-based predictions instead")
    
    # Show preview of what will be processed
    st.subheader("📋 Prediction Preview")
    try:
        preview_df = manager.get_records(
            date_filter=date_filter if date_filter not in ["latest", "today"] else (date_filter if date_filter == "today" else None)
        )
        
        if not preview_df.empty:
            st.info(f"📊 Will process **ALL {len(preview_df)}** tickers")
            
            # Show preview table (limit display to first 50 for performance)
            with st.expander("👀 Preview Tickers"):
                display_df = preview_df[['ticker', 'date_added', 'pdf_source']]
                if len(display_df) > 50:
                    st.dataframe(display_df.head(50), use_container_width=True)
                    st.info(f"Showing first 50 of {len(display_df)} tickers")
                else:
                    st.dataframe(display_df, use_container_width=True)
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
                    max_tickers=None,
                    background=background_mode,
                    rebuild_models=rebuild_models,
                    use_ml=use_ml
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
            "--date-filter", date_filter
        ]
        
        # Add ML prediction options to display
        if use_ml:
            cmd_parts.append("--use-ml")
        else:
            cmd_parts.append("--no-ml")
        
        # Add rebuild models option to display
        if rebuild_models:
            cmd_parts.append("--rebuild-models")
        
        st.info("**Command to execute (ALL tickers):**")
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
                        duplicate_count = 0
                        total_rows = len(df_import)
                        
                        for i, row in df_import.iterrows():
                            status_text.text(f"Importing row {i+1}/{total_rows}...")
                            try:
                                ticker = str(row['ticker']).upper().strip()
                                date_added = pd.to_datetime(row['date_added']).date()
                                pdf_source = row.get('pdf_source', None)
                                
                                if manager.add_ticker(ticker, pdf_source, date_added):
                                    imported_count += 1
                                else:
                                    duplicate_count += 1
                            except Exception as e:
                                st.warning(f"⚠️ Error importing row {i+1}: {e}")
                            
                            progress_bar.progress((i + 1) / total_rows)
                        
                        status_text.text(f"✅ Complete! Imported {imported_count} new records. {duplicate_count} duplicates skipped.")
                        if duplicate_count > 0:
                            st.info(f"ℹ️ {duplicate_count} tickers were already in the database and were skipped.")
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
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
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
                        duplicate_count = 0
                        total_count = len(final_symbols)
                        
                        for i, symbol in enumerate(final_symbols):
                            status_text.text(f"Adding {symbol}... ({i+1}/{total_count})")
                            if manager.add_ticker(symbol, image_source, image_date):
                                added_count += 1
                            else:
                                duplicate_count += 1
                            progress_bar.progress((i + 1) / total_count)
                        
                        status_text.text(f"✅ Complete! Added {added_count} new symbols. {duplicate_count} duplicates skipped.")
                        if duplicate_count > 0:
                            st.info(f"ℹ️ {duplicate_count} symbols were already in the database and were skipped.")
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
                    duplicate_count = 0
                    total_count = len(parsed_symbols)
                    
                    for i, symbol in enumerate(parsed_symbols):
                        status_text.text(f"Adding {symbol}... ({i+1}/{total_count})")
                        if manager.add_ticker(symbol, manual_source, manual_date):
                            added_count += 1
                        else:
                            duplicate_count += 1
                        progress_bar.progress((i + 1) / total_count)
                    
                    status_text.text(f"✅ Complete! Added {added_count} new symbols. {duplicate_count} duplicates skipped.")
                    if duplicate_count > 0:
                        st.info(f"ℹ️ {duplicate_count} symbols were already in the database and were skipped.")
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
