import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

# 1. Fetch data (single consistent filter)
engine = create_engine(DB_URL)
sql = """
    SELECT ticker, date, change_pct, close_price,
           LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) AS high_1d
    FROM stock_data
    WHERE close_price > 0
      AND change_pct < 0
      AND change_pct >= -50
"""
df = pd.read_sql_query(sql, engine)
df = df.dropna(subset=['high_1d', 'change_pct', 'close_price'])

# 2. Compute rebounce
df['rebounce'] = (df['high_1d'] - df['close_price']) * 100 / df['close_price']

# 3. Bin change_pct into your specified drop ranges
bins = np.arange(-50, 1, 5)  # [-50, -45, -40, ..., -5, 0]
labels = [f"{int(b)} to {int(b+5)}" for b in bins[:-1]]
df['drop_bin'] = pd.cut(df['change_pct'], bins=bins, labels=labels, right=False)

# 4. Calculate percent of positive rebounce in each bin
bin_stats = (
    df.groupby('drop_bin')
      .agg(
          total=('rebounce', 'count'),
          positive=('rebounce', lambda x: (x > 0).sum())
      )
      .assign(percent_positive=lambda d: 100 * d['positive'] / d['total'])
      .reset_index()
)

print(bin_stats[['drop_bin', 'total', 'positive', 'percent_positive']])

# 5. Plot percent of positive rebounce by drop_bin
plt.figure(figsize=(10, 5))
plt.bar(bin_stats['drop_bin'], bin_stats['percent_positive'], color='skyblue')
for i, pct in enumerate(bin_stats['percent_positive']):
    plt.text(i, pct + 1, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)
plt.ylim(0, 100)
plt.title("Percent of Positive Rebound by Drop Range")
plt.xlabel("Drop Range (change_pct)")
plt.ylabel("Percent of Positive Rebound (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
