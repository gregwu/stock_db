import psycopg2
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

def get_rebounce_stats(A, B, conn):
    sql = f"""
        WITH base AS (
            SELECT
                (LEAD(high_price, 1) OVER (PARTITION BY ticker ORDER BY date) - close_price) * 100.0 / close_price AS rebounce
            FROM stock_data
            WHERE close_price > 0 AND change_pct <= {A} AND change_pct >= {B}
        ),
        bucketed AS (
            SELECT
                rebounce,
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
        ),
        summary AS (
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN rebounce >= 0 THEN 1 ELSE 0 END) AS up_count,
                SUM(CASE WHEN rebounce < 0 THEN 1 ELSE 0 END) AS down_count,
                AVG(rebounce) AS avg_rebounce,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rebounce) AS median_rebounce,
                SUM(CASE WHEN rebounce > 10 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*),0) AS prob_rebounce_gt10
            FROM bucketed
            WHERE rebounce IS NOT NULL
        )
        SELECT
            {A}::float AS a,
            {B}::float AS b,
            s.total,
            s.up_count,
            s.down_count,
            s.up_count * 1.0 / NULLIF(s.total,0) AS pct_positive_rebounce,
            s.avg_rebounce,
            s.median_rebounce,
            s.prob_rebounce_gt10
        FROM summary s
    """
    df = pd.read_sql_query(sql, conn)
    return df.iloc[0] if not df.empty else None

if __name__ == "__main__":
    # Grid search parameters (window at least 5%)
    a_range = np.arange(-3, -50, -1)   # upper threshold, -5, -6, ..., -49
    results = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        for a in a_range:
            # b at least 5% lower than a
            b_range = np.arange(a-2, -101, -3)
            for b in b_range:
                if b < a:
                    stats = get_rebounce_stats(a, b, conn)
                    if stats is not None and stats['total'] > 20:
                        results.append(stats)
                        print(f"A={a}, B={b}, pct_positive_rebounce={stats['pct_positive_rebounce']:.2%}, count={stats['total']}")

    results_df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    # Show top by positive rebounce
    if not results_df.empty:
        top = results_df.sort_values('pct_positive_rebounce', ascending=False)
        print("\nTop intervals for positive rebounce:")
        print(top[['a','b','total','pct_positive_rebounce','avg_rebounce','median_rebounce','prob_rebounce_gt10']].head(10))
        best = top.iloc[0]
        print(f"\nBest interval: change_pct between {best['a']}% and {best['b']}%")
        print(f"Count: {best['total']}")
        print(f"Pct Positive Rebounce: {best['pct_positive_rebounce']:.2%}")
        print(f"Avg Rebounce: {best['avg_rebounce']:.2f}%")
        print(f"Median Rebounce: {best['median_rebounce']:.2f}%")
        print(f"P(rebounce > 10%): {best['prob_rebounce_gt10']:.2%}")
    else:
        print("No results in grid search.")
