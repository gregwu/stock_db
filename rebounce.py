import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import argparse

load_dotenv()
# Database config - use environment variables or replace directly
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

parser = argparse.ArgumentParser(description="Analyze rebounce distribution with change_pct thresholds.")
parser.add_argument('a', type=float, help='Upper threshold for change_pct (A)')
parser.add_argument('b', type=float, help='Lower threshold for change_pct (B)')
args = parser.parse_args()

A = args.a
B = args.b
# SQL for rebounce bucket counts
SQL = f"""
WITH base AS (
    SELECT
        (LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) - close_price) * 100.0 / close_price AS rebounce
    FROM stock_data
    WHERE  close_price > 0 AND change_pct <= {A} AND change_pct > {B}
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

# Connect and fetch data
conn = psycopg2.connect(**DB_CONFIG)
df = pd.read_sql_query(SQL, conn)
conn.close()

# Fill in any missing buckets as 0 for plotting consistency
all_buckets = [
    '<= -100','-100 ~ -50', '-50 ~ -20', '-20 ~ -10', '-10 ~ -5', '-5 ~ 0', '0 ~ 5',
    '5 ~ 10', '10 ~ 20', '20 ~ 50', '50 ~ 100', '>= 100'
]
df = df.set_index('bucket').reindex(all_buckets, fill_value=0).reset_index()

# Calculate percentages
df['percent'] = df['cnt'] / df['cnt'].sum() * 100

# Print distribution table
print(df[['bucket', 'cnt', 'percent']])

# Up/Down split
up_buckets = ['0 ~ 5', '5 ~ 10', '10 ~ 20', '20 ~ 50', '50 ~ 100', '>= 100']
down_buckets = ['<= -100','-100 ~ -50', '-50 ~ -20', '-20 ~ -10', '-10 ~ -5', '-5 ~ 0']

up_count = df[df['bucket'].isin(up_buckets)]['cnt'].sum()
down_count = df[df['bucket'].isin(down_buckets)]['cnt'].sum()
total = up_count + down_count

up_pct = up_count / total * 100 if total else 0
down_pct = down_count / total * 100 if total else 0

print(f"\nUP: {up_count} ({up_pct:.1f}%)")
print(f"DOWN: {down_count} ({down_pct:.1f}%)")
print(f"Total: {total}")



# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(df['bucket'], df['percent'], color=['red']*5 + ['gray']*2 + ['green']*4 + ['blue'])

plt.title(f"Rebounce Distribution by Bucket ( {A} to {B}, Total: {total})")
plt.xlabel("Rebounce Bucket (%)")
plt.ylabel("Percentage of Occurrences (%)")
plt.xticks(rotation=30)
plt.ylim(0, df['percent'].max() * 1.2)

# Annotate bars
for i, (cnt, pct) in enumerate(zip(df['cnt'], df['percent'])):
    plt.text(i, pct + 0.3, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)

# Annotate up/down
plt.figtext(0.85, 0.88, f"UP: {up_pct:.1f}%", color='green', fontsize=14, ha='center')
plt.figtext(0.15, 0.88, f"DOWN: {down_pct:.1f}%", color='red', fontsize=14, ha='center')

plt.tight_layout()
plt.show()
