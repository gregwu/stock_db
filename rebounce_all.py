import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from tqdm import tqdm
pd.set_option('display.max_rows', None) 
load_dotenv()
# Database config - use environment variables or replace directly
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

# Buckets for rebounce
all_buckets = [
    '<= -100','-100 ~ -50', '-50 ~ -20', '-20 ~ -10', '-10 ~ -5', '-5 ~ 0', '0 ~ 5',
    '5 ~ 10', '10 ~ 20', '20 ~ 50', '50 ~ 100', '>= 100'
]
up_buckets = ['0 ~ 5', '5 ~ 10', '10 ~ 20', '20 ~ 50', '50 ~ 100', '>= 100']
down_buckets = ['<= -100','-100 ~ -50', '-50 ~ -20', '-20 ~ -10', '-10 ~ -5', '-5 ~ 0']

def fetch():
    results = []
    drop_steps = list(range(100, -100, -1))  # 0, -1, -2, ..., -99

    conn = psycopg2.connect(**DB_CONFIG)
    for a, b in tqdm(zip(drop_steps[:-1], drop_steps[1:]), total=len(drop_steps)-1, desc="Drop Ranges"):
        SQL = f"""
    WITH base AS (
        SELECT
            (LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) - close_price) * 100.0 / close_price AS rebounce,
            change_pct_1d AS rebounce1
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
    ;
    """
        df = pd.read_sql_query(SQL, conn)
        # Ensure all buckets present
        df = df.set_index('bucket').reindex(all_buckets, fill_value=0).reset_index()
        up_count = df[df['bucket'].isin(up_buckets)]['cnt'].sum()
        down_count = df[df['bucket'].isin(down_buckets)]['cnt'].sum()
        total = up_count + down_count
        up_pct = up_count / total * 100 if total else 0
        down_pct = down_count / total * 100 if total else 0
        results.append({'drop_range': f"{a} to {b}", 'up_count': up_count, 'down_count': down_count, 'up_pct': up_pct, 'down_pct': down_pct, 'total': total})

    conn.close()

    # Make DataFrame
    results_df = pd.DataFrame(results)
    return results_df

filename = 'rebounce_data4.csv'
if os.path.exists(filename):
    print( "Load from pickle ")
    #results_df = pd.read_pickle(filename)
    results_df = pd.read_csv(filename)
else:
    print("Fetch data from database") 
    results_df = fetch()
    #results_df.to_csv(filename)

print(results_df)
# Assuming 'df' is your summary table as above
best_row = results_df.loc[results_df['up_pct'].idxmax()]
print("Best drop range for positive rebounce:")
print(best_row)


# Optional: Plot
plt.figure(figsize=(14,6))
plt.plot(results_df['drop_range'], results_df['up_count'], label='UP %', color='green')
plt.plot(results_df['drop_range'], results_df['down_count'], label='DOWN %', color='red')
#plt.plot(results_df['drop_range'], results_df['up_pct'], label='UP %', color='green')
#plt.plot(results_df['drop_range'], results_df['down_pct'], label='DOWN %', color='red')
plt.xticks(rotation=90, fontsize=7)
plt.title("Positive/Negative Rebounce Percent by Drop Range (change_pct)")
plt.xlabel("Drop Range (change_pct)")
plt.ylabel("Percent (%)")
# Find x position for vertical line
try:
    idx1 = results_df.index[results_df['drop_range'] == '1 to 0'][0]
    idx2 = results_df.index[results_df['drop_range'] == '0 to -1'][0]
    # The line is between idx1 and idx2, so place it at idx1 + 0.5
    plt.axvline(x=idx1 + 0.5, color='blue', linestyle='--', label='Break: 0')
    plt.axhline(y=50, color='blue', linestyle='--', label='50%')  # Horizontal line at y=50%
    # Optionally annotate
    plt.text(idx1 + 0.5, plt.ylim()[1]*0.95, '0%', color='blue', ha='center')
except IndexError:
    print("Couldn't find the required ranges for the vertical line.")

plt.legend()
plt.tight_layout()
plt.show()
