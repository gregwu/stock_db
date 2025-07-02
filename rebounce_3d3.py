import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# --- DB Config ---
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

# --- Dynamic drop ranges ---
step = 3
drop_ends = np.arange(0, -105, -step)  # [0, -5, -10, ..., -100]
drop_ranges = [(drop_ends[i], drop_ends[i+1]) for i in range(len(drop_ends)-1)]
drop_labels = [str(int(b)) for a, b in drop_ranges]

# --- Fine rebounce buckets (every 5%) ---
rebounce_edges = list(range(-100, 105, 5))  # -100 to 100, step 5
rebounce_buckets = []
for i in range(len(rebounce_edges)-1):
    left = rebounce_edges[i]
    right = rebounce_edges[i+1]
    if i == 0:
        rebounce_buckets.append(f"<= {right}")
    elif i == len(rebounce_edges)-2:
        rebounce_buckets.append(f">= {left}")
    else:
        rebounce_buckets.append(f"{left} ~ {right}")

# --- Prepare results ---
all_distributions = []

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")

for (a, b), label in zip(drop_ranges, drop_labels):
    sql = f"""
        WITH base AS (
            SELECT
                (LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) - close_price) * 100.0 / close_price AS rebounce
            FROM stock_data
            WHERE close_price > 0 AND change_pct <= {a} AND change_pct > {b}
        )
        SELECT * FROM (
            SELECT
                CASE
                    WHEN rebounce <= -100 THEN '<= -100'
                    WHEN rebounce > 100 THEN '>= 100'
                    ELSE
                        CONCAT(
                            CAST(FLOOR(rebounce/5)*5 AS INT),
                            ' ~ ',
                            CAST(FLOOR(rebounce/5)*5 + 5 AS INT)
                        )
                END AS bucket,
                COUNT(*) AS cnt
            FROM base
            WHERE rebounce IS NOT NULL
            GROUP BY bucket
        ) x
    """
    df = pd.read_sql_query(sql, engine)
    # Ensure all buckets appear
    df = df.set_index('bucket').reindex(rebounce_buckets, fill_value=0).reset_index()
    total = df['cnt'].sum()
    df['percent'] = df['cnt'] / total * 100 if total else 0
    all_distributions.append(df['percent'].values)

# --- 3D Line Plot ---
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

_x = np.arange(len(rebounce_buckets))
_y = np.arange(len(drop_labels))
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
z = np.array(all_distributions)
z = z.reshape(len(drop_labels), len(rebounce_buckets))

# Plot each drop range as a separate line
for i, row in enumerate(z):
    ax.plot(_x, [i]*len(_x), row, label=f"{drop_labels[i]}")

ax.set_xticks(_x)
ax.set_xticklabels(rebounce_buckets, rotation=80, ha='right', fontsize=8)
ax.set_yticks(_y)
ax.set_yticklabels(drop_labels, fontsize=9)
ax.set_zlabel('Percent (%)')
ax.set_ylabel('Drop Range (change_pct)')
ax.set_xlabel('Rebounce Bucket (%)')
ax.set_title('3D Rebounce Distribution by Drop Range and Rebounce Bucket')

plt.tight_layout()
plt.show()
