import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm

def get_data_and_save_npz(filename="rebounce_surface_data.npz"):
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
    step = 1
    drop_ends = np.arange(0, -105, -step)
    drop_ranges = [(drop_ends[i], drop_ends[i+1]) for i in range(len(drop_ends)-1)]
    drop_labels = [str(int(b)) for a, b in drop_ranges]

    # --- Fine rebounce buckets (every 1%) ---
    rebounce_edges = list(range(-100, 105, 1))
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
    conn = create_engine(
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
    for (a, b), label in tqdm(list(zip(drop_ranges, drop_labels)), desc="Processing drop ranges"):
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
                                CAST(FLOOR(rebounce/1)*1 AS INT),
                                ' ~ ',
                                CAST(FLOOR(rebounce/1)*1 + 1 AS INT)
                            )
                    END AS bucket,
                    COUNT(*) AS cnt
                FROM base
                WHERE rebounce IS NOT NULL
                GROUP BY bucket
            ) x
        """
        df = pd.read_sql_query(sql, conn)
        # Ensure all buckets appear
        df = df.set_index('bucket').reindex(rebounce_buckets, fill_value=0).reset_index()
        total = df['cnt'].sum()
        df['percent'] = df['cnt'] / total * 100 if total else 0
        all_distributions.append(df['percent'].values)

    # Convert to numpy array for saving
    Z = np.array(all_distributions)

    # --- Save to npz file for later use ---
    np.savez_compressed(filename,
                        Z=Z,
                        drop_labels=np.array(drop_labels),
                        rebounce_buckets=np.array(rebounce_buckets))
    print(f"Saved 3D rebounce data to {filename}.")
    return Z, drop_labels, rebounce_buckets

file = 'rebounce_surface_data.npz'
if not os.path.exists(file):
    Z, drop_labels, rebounce_buckets = get_data_and_save_npz(file)
else:
    with np.load(file, allow_pickle=True) as data:
        Z = data['Z']
        drop_labels = data['drop_labels']
        rebounce_buckets = data['rebounce_buckets']

# --- 3D Surface Plot ---
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

X = np.arange(len(rebounce_buckets))
Y = np.arange(len(drop_labels))
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)

ax.set_xticks(np.arange(len(rebounce_buckets))[::10])
ax.set_xticklabels(rebounce_buckets[::10], rotation=80, ha='right', fontsize=8)
ax.set_yticks(np.arange(len(drop_labels))[::20])
ax.set_yticklabels(drop_labels[::20], fontsize=9)
ax.set_zlabel('Percent (%)')
ax.set_ylabel('Drop Range (change_pct)')
ax.set_xlabel('Rebounce Bucket (%)')
ax.set_title('3D Rebounce Distribution (Surface) by Drop Range and Rebounce Bucket')

fig.colorbar(surf, shrink=0.5, aspect=10, label='Percent (%)')
plt.tight_layout()
plt.show()
