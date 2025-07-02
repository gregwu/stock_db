import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Database Config ----
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

# ---- Ranges ----
drop_ranges = [
    ("0", "-5"), ("-5", "-10"), ("-10", "-15"), ("-15", "-20"), ("-20", "-25"), ("-25", "-30"),
    ("-30", "-35"), ("-35", "-40"), ("-40", "-45"), ("-45", "-50"), ("-50", "-55"), ("-55", "-60"),
    ("-60", "-65"), ("-65", "-70"), ("-70", "-75"), ("-75", "-80"), ("-80", "-85"), ("-85", "-90"),
    ("-90", "-95"), ("-95", "-100")
]
drop_labels = [f"{a} to {b}" for a, b in drop_ranges]
rebounce_buckets = [
    '<= -100','-100 ~ -50', '-50 ~ -20', '-20 ~ -10', '-10 ~ -5', '-5 ~ 0', '0 ~ 5',
    '5 ~ 10', '10 ~ 20', '20 ~ 50', '50 ~ 100', '>= 100'
]

# ---- Fetch data for all drop ranges ----
all_percents = []

conn = psycopg2.connect(**DB_CONFIG)

for a, b in drop_ranges:
    SQL = f"""
    WITH base AS (
        SELECT
            (LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) - close_price) * 100.0 / close_price AS rebounce
        FROM stock_data
        WHERE close_price > 0 AND change_pct <= {a} AND change_pct > {b}
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
    # Ensure all buckets are present
    df = df.set_index('bucket').reindex(rebounce_buckets, fill_value=0).reset_index()
    total = df['cnt'].sum()
    percent = df['cnt'] / total * 100 if total else [0]*len(df)
    all_percents.append(percent.values)

conn.close()

percent_matrix = np.array(all_percents)  # shape: (len(drop_ranges), len(rebounce_buckets))

# ---- 3D Line Plot ----
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(range(len(rebounce_buckets)), range(len(drop_ranges)))
# Plot lines for each drop range
for i in range(len(drop_ranges)):
    ax.plot(X[i], Y[i], percent_matrix[i], marker='o', label=drop_labels[i])

ax.set_xticks(range(len(rebounce_buckets)))
ax.set_xticklabels(rebounce_buckets, rotation=30, ha='right')
ax.set_yticks(range(len(drop_ranges)))
ax.set_yticklabels(drop_labels)
ax.set_xlabel("Rebounce Bucket (%)")
ax.set_ylabel("Drop Range (change_pct)")
ax.set_zlabel("Percent (%)")
ax.set_title("3D Rebounce Distribution by Drop Range and Rebounce Bucket (Line Chart)")

plt.tight_layout()
plt.show()
