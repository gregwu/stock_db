import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
pd.set_option('display.max_rows', None)
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

# --- 1. Fetch the Data from DB ---
engine = create_engine(DB_URL)
sql = """
    SELECT ticker, date, change_pct, close_price,
        LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) AS high_1d
    FROM stock_data
    WHERE change_pct <= -5 AND change_pct >= -100 AND close_price > 0
"""
df = pd.read_sql_query(sql, engine)
df = df.dropna(subset=['high_1d', 'change_pct', 'close_price'])

# --- 2. Compute rebounce ---
df['rebounce'] = (df['high_1d'] - df['close_price']) * 100 / df['close_price']

# --- 3. Grid Search for Best a, b (non-overlapping, meaningful windows only) ---
a_range = np.arange(-5, -50, -1)   # e.g. -5 to -49 (upper threshold)
#a_range = [-5, -6, -7]

results = []
for a in a_range:
    # For each a, set b at least 5% lower (so window is minimum width)
    b_range = np.arange(a - 5, -101, -5)  # E.g. -10, -15, ..., -100, in steps of -5
    ##b_range = [-15, -20, -25, -30]
    for b in b_range:
        if b < a:
            subset = df[(df['change_pct'] < a) & (df['change_pct'] > b)]
            n = len(subset)
            if n > 20:
                avg_reb = subset['rebounce'].mean()
                med_reb = subset['rebounce'].median()
                prob_gt10 = (subset['rebounce'] > 10).mean()
                count_pos_reb = (subset['rebounce'] > 0).sum()
                pct_pos_reb = count_pos_reb / n if n > 0 else 0
                results.append({
                    'a': a,
                    'b': b,
                    'count': n,
                    'pct_positive_rebounce': pct_pos_reb,
                    'avg_rebounce': avg_reb,
                    'median_rebounce': med_reb,
                    'prob_rebounce_gt10': prob_gt10
                })
results_df = pd.DataFrame(results)


# Show top results for positive rebounce percentage
if not results_df.empty:
    print("Highest percentage of positive rebounce (rebounce > 0):")
    top10 = results_df.sort_values('pct_positive_rebounce', ascending=False)
    print(top10)
    best_row = top10.iloc[0]
    print(f"\nBest interval for positive rebounce: change_pct between {best_row['a']}% and {best_row['b']}%")
    print(f"Count: {best_row['count']}")
    print(f"Pct Positive Rebounce: {best_row['pct_positive_rebounce']:.2%}")
    print(f"Avg Rebounce: {best_row['avg_rebounce']:.2f}%")
    print(f"Median Rebounce: {best_row['median_rebounce']:.2f}%")
    print(f"P(rebounce > 10%): {best_row['prob_rebounce_gt10']:.2%}")
    exit()
    # Other best metrics...
    print("\nBest average rebounce:")
    print(results_df.sort_values('avg_rebounce', ascending=False).head(5))
    print("\nBest median rebounce:")
    print(results_df.sort_values('median_rebounce', ascending=False).head(5))
    print("\nBest chance of rebounce > 10%:")
    print(results_df.sort_values('prob_rebounce_gt10', ascending=False).head(5))
else:
    print("No results found in the searched parameter ranges.")
