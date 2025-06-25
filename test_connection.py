#!/usr/bin/env python3
"""
Test database connection with different configurations
"""

import os
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'gangwu'),
    'user': os.getenv('DB_USER', 'gangwu'),
    'password': os.getenv('DB_PASSWORD', 'gangwu')
}

def test_psycopg2_connection():
    """Test direct psycopg2 connection"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',  # Try connecting to default database first
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print("✅ psycopg2 connection to 'postgres' database successful")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ psycopg2 connection failed: {e}")
        return False

def test_sqlalchemy_connection():
    """Test SQLAlchemy connection"""
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ SQLAlchemy connection successful: {version}")
        return True
    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {e}")
        return False

def test_target_database():
    """Test connection to target database"""
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT current_database()"))
            db_name = result.fetchone()[0]
            print(f"✅ Connected to target database: {db_name}")
        return True
    except Exception as e:
        print(f"❌ Target database connection failed: {e}")
        return False

def main():
    print("🔍 Testing database connections...")
    print(f"Configuration: {DB_CONFIG}")
    print()
    
    # Test basic connection
    if test_psycopg2_connection():
        print("✅ Basic connection works")
    else:
        print("❌ Basic connection failed")
        return
    
    if test_sqlalchemy_connection():
        print("✅ SQLAlchemy connection works")
    else:
        print("❌ SQLAlchemy connection failed")
        return
    
    if test_target_database():
        print("✅ Target database connection works")
    else:
        print("❌ Target database connection failed - database may not exist")
        print("💡 Try running savedb.py first to create the database")

if __name__ == "__main__":
    main()
