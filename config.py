from dotenv import load_dotenv
import os
load_dotenv()# Placeholder for any shared configurations in future
features = [
    # Core features
    "sma_144_dist",
    "sma_144_slope",
    "rsi_14",
    "macd",
    "macd_momentum",
    "price_change_abs",
    "high_low_ratio",
    "open_close_ratio",
    "price_volatility",
    "bb_width",
    "bb_position",
    "volume_sma",
    "volume_momentum",
    "price_volume",
    "obv",
    "obv_momentum",
    "volatility_20",
    "volatility_momentum",
    "resistance_20",
    "support_distance",
    "price_change_lag_1",
    "price_change_lag_2",
    "price_change_lag_3",
    "price_change_lag_5",
    "volume_ratio_lag_1",
    "volume_ratio_lag_2",
    "volume_ratio_lag_3",
    "volume_ratio_lag_5",

    # === Derived/Composite Features ===
    # These advanced features combine multiple indicators for better predictions
    "bb_to_volatility",                # bb_std / price_volatility
    "bb_width_to_volatility20",        # bb_width / volatility_20
    "price_vol_to_vol20",              # price_volatility / volatility_20
    "macd_diff",                       # macd - macd_signal
    "macd_trend_strength",            # macd_hist * macd_momentum
    "sma_diff_5_20",                  # price_to_sma_20 - price_to_sma_5
    "sma_slope_diff",                # sma_5_slope - sma_20_slope
    "vol_price_momentum",             # volume_ratio * price_change_abs
    "obv_macd_interact",              # obv_momentum * macd_momentum
    "stoch_diff",                     # stoch_k - stoch_d
    "sr_ratio",                       # support_distance / resistance_distance
    "move_to_support_ratio",         # price_change_abs / support_distance
    "trend_consistency_2d",          # price_change_lag_1 * price_change_lag_2
    "trend_consistency_3d"           # price_change_lag_1 * price_change_lag_3
]


cat_features = ['ticker', 'sector']

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'dbname': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Directory configurations
DIRECTORIES = {
    'logs': os.getenv('LOGS_DIR', 'logs'),
    'reports': os.getenv('REPORTS_DIR', 'seekingalpha_reports'),
    'models': os.getenv('MODELS_DIR', 'models'),
    'prediction_reports': os.getenv('PREDICTION_REPORTS_DIR', 'prediction_reports'),
    'enhanced_reports': os.getenv('ENHANCED_REPORTS_DIR', 'enhanced_professional_reports'),
    'backtest': os.getenv('BACKTEST_DIR', 'backtest'),
    'model_comparison': os.getenv('MODEL_COMPARISON_DIR', 'model_comparison'),
    'work_dir': os.getenv('WORK_DIR', '.')  # Changed from 'predictor' to current directory
}

# Python executable configuration
PYTHON_CONFIG = {
    'executable': os.getenv('PYTHON_EXECUTABLE', 'python'),  # Changed from 'predictor/bin/python'
    'fallback_executable': os.getenv('PYTHON_FALLBACK', 'python')
}

# File patterns and extensions
FILE_PATTERNS = {
    'log_extension': '.log',
    'prediction_log_prefix': 'prediction_',
    'report_html': 'seekingalpha_report.html',
    'report_json': 'seekingalpha_report.json',
    'report_csv': 'seekingalpha_predictions.csv'
}

# Web serving configuration
WEB_CONFIG = {
    'base_url': os.getenv('WEB_BASE_URL', 'https://www.stargate.tel'),
    'reports_web_path': os.getenv('REPORTS_WEB_PATH', '/seekingalpha_reports'),
    'enable_web_links': os.getenv('ENABLE_WEB_LINKS', 'true').lower() == 'true',
    'enable_file_download': os.getenv('ENABLE_FILE_DOWNLOAD', 'false').lower() == 'true'
}

# Prediction Report URLs
PREDICTION_REPORTS = {
    'seekingalpha_report': 'https://www.stargate.tel/seekingalpha_reports/seekingalpha_report.html',
    'stock_predictions_report': 'https://www.stargate.tel/reports/stock_predictions_report.html'
}
