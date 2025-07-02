
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from dotenv import load_dotenv
import sys


load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

# Define rebounce buckets
all_buckets = [
    '<= -100','-100 ~ -50', '-50 ~ -20', '-20 ~ -10', '-10 ~ -5', '-5 ~ 0', '0 ~ 5',
    '5 ~ 10', '10 ~ 20', '20 ~ 50', '50 ~ 100', '>= 100'
]


# Define change_pct drop ranges
drop_ranges = [(0, -5), (-5, -10), (-10, -15), (-15, -20), (-20, -25),
               (-25, -30), (-30, -35), (-35, -40), (-40, -45), (-45, -50),
               (-50, -55), (-55, -60), (-60, -65), (-65, -70), (-70, -75),
               (-75, -80), (-80, -85), (-85, -90), (-90, -95), (-95, -100)]

# --- Command line argument for file ---
if len(sys.argv) > 1:
    npz_file = sys.argv[1]
else:
    npz_file = "rebounce_bar_data.npz"

if os.path.exists(npz_file):
    # Load from file
    with np.load(npz_file, allow_pickle=True) as data:
        percent_matrix = data['percent_matrix']
        count_matrix = data['count_matrix']
        all_buckets = data['all_buckets']
        drop_ranges = data['drop_ranges']
    print(f"Loaded data from {npz_file}.")
else:
    # Do DB query and save
    conn = psycopg2.connect(**DB_CONFIG)
    percent_matrix = []
    count_matrix = []
    for upper, lower in drop_ranges:
        SQL = f"""
        WITH base AS (
            SELECT
                (LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) - close_price) * 100.0 / close_price AS rebounce
            FROM stock_data
            WHERE close_price > 0 AND change_pct <= {upper} AND change_pct > {lower}
        )
        SELECT bucket, COUNT(*) AS cnt
        FROM (
            SELECT
                CASE
                    WHEN rebounce <= -100 THEN '<= -100'
                    WHEN rebounce > -100 AND rebounce <= -50 THEN '-100 ~ -50'
                    WHEN rebounce > -50 AND rebounce <= -20 THEN '-50 ~ -20'
                    WHEN rebounce > -20 AND rebounce <= -10 THEN '-20 ~ -10'
                    WHEN rebounce > -10 AND rebounce <= -5 THEN '-10 ~ -5'
                    WHEN rebounce > -5 AND rebounce < 0 THEN '-5 ~ 0'
                    WHEN rebounce >= 0 AND rebounce < 5 THEN '0 ~ 5'
                    WHEN rebounce >= 5 AND rebounce < 10 THEN '5 ~ 10'
                    WHEN rebounce >= 10 AND rebounce < 20 THEN '10 ~ 20'
                    WHEN rebounce >= 20 AND rebounce < 50 THEN '20 ~ 50'
                    WHEN rebounce >= 50 AND rebounce < 100 THEN '50 ~ 100'
                    WHEN rebounce >= 100 THEN '>= 100'
                END AS bucket
            FROM base
        ) t
        WHERE bucket IS NOT NULL
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '<= -100' THEN 1
                WHEN '-100 ~ -50' THEN 2
                WHEN '-50 ~ -20' THEN 3
                WHEN '-20 ~ -10' THEN 4
                WHEN '-10 ~ -5' THEN 5
                WHEN '-5 ~ 0' THEN 6
                WHEN '0 ~ 5' THEN 7
                WHEN '5 ~ 10' THEN 8
                WHEN '10 ~ 20' THEN 9
                WHEN '20 ~ 50' THEN 10
                WHEN '50 ~ 100' THEN 11
                WHEN '>= 100' THEN 12
            END
        ;
        """
        df = pd.read_sql_query(SQL, conn)
        df = df.set_index('bucket').reindex(all_buckets, fill_value=0).reset_index()
        total = df['cnt'].sum()
        percent = (df['cnt'] / total * 100) if total > 0 else [0] * len(all_buckets)
        percent_matrix.append(percent)
        count_matrix.append(df['cnt'])
    conn.close()
    percent_matrix = np.array(percent_matrix)
    count_matrix = np.array(count_matrix)
    # --- Save to npz file for later use ---
    np.savez_compressed(
        npz_file,
        percent_matrix=percent_matrix,
        count_matrix=count_matrix,
        all_buckets=np.array(all_buckets),
        drop_ranges=np.array(drop_ranges)
    )
    print(f"Saved 3D rebounce bar data to {npz_file}.")









# Plot 3D bar chart
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

_x = np.arange(len(all_buckets))       # rebounce buckets (x)
_y = np.arange(len(drop_ranges))       # drop ranges (y)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
z = np.zeros_like(x)
dz = percent_matrix.ravel()

dx = 0.7 * np.ones_like(z)
dy = 0.7 * np.ones_like(z)

ax.bar3d(x, y, z, dx, dy, dz, shade=True, color='skyblue')

ax.set_xlabel('Rebounce Bucket (%)')
ax.set_ylabel('Drop Range (change_pct)')
ax.set_zlabel('Percent (%)')
ax.set_xticks(_x + 0.35)
ax.set_xticklabels(all_buckets, rotation=35, ha='right')
ax.set_yticks(np.arange(len(drop_ranges)) + 0.35)
ax.set_yticklabels([f"{u} to {l}" for u, l in drop_ranges])

plt.title('3D Rebounce Distribution by Drop Range and Rebounce Bucket')
plt.tight_layout()
plt.show()
