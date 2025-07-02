import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm
import sys
import argparse

def get_data_and_save_npz(filename="rebounce_surface_data.npz", save=False, drop_start=105, drop_end=-105):
    # --- DB Config ---
    load_dotenv()
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
    }

    # --- Parse drop range from command line or use defaults ---
    # Defaults: start=105, end=-105, step=1

    step = 1
    drop_ends = np.arange(drop_start, drop_end, -step)
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
    if save:
        np.savez_compressed(filename,
                            Z=Z,
                            drop_labels=np.array(drop_labels),
                            rebounce_buckets=np.array(rebounce_buckets))
        print(f"Saved 3D rebounce data to {filename}.")
    return Z, drop_labels, rebounce_buckets, drop_ranges

parser = argparse.ArgumentParser(description="3D Rebounce Distribution Plotter")
parser.add_argument('--file', type=str, default='rebounce_surface_data.npz', help='Path to npz data file')
parser.add_argument('--plot', type=str, default='surface', help='Plot type (surface or line)')
parser.add_argument('--start', type=int, default=105)
parser.add_argument('--end', type=int, default=-105)
parser.add_argument('--save', type=bool, default=False, help='Save the data to npz file')
args = parser.parse_args()


drop_start = args.start
drop_end = args.end
save = args.save
file = args.file
plot_type = args.plot
if not os.path.exists(file):
    Z, drop_labels, rebounce_buckets, drop_ranges = get_data_and_save_npz(file, save=save, drop_start=drop_start, drop_end=drop_end)
else:
    with np.load(file, allow_pickle=True) as data:
        Z = data['Z']
        drop_labels = data['drop_labels']
        rebounce_buckets = data['rebounce_buckets']
    # Reconstruct drop_ranges from drop_labels and drop_start/drop_end
    step = 1
    drop_ends = np.arange(drop_start, drop_end, -step)
    drop_ranges = [(drop_ends[i], drop_ends[i+1]) for i in range(len(drop_ends)-1)]
    drop_labels = [str(int(b)) for a, b in drop_ranges]

def plot_surface():
    # --- 3D Surface Plot ---
    fig = plt.figure(figsize=(20, 12))
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

def plot_line():
    percent_matrix = np.array(Z)  # Use Z as the percent matrix
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(len(rebounce_buckets)), range(len(drop_ranges)))
    # Plot lines for each drop range, each with a different color, no dots
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(drop_ranges)))
    for i in range(len(drop_ranges)):
        ax.plot(X[i], Y[i], percent_matrix[i], color=colors[i])

    ax.set_xticks(range(0, len(rebounce_buckets), max(1, len(rebounce_buckets)//10)))
    ax.set_xticklabels(rebounce_buckets[::max(1, len(rebounce_buckets)//10)], rotation=30, ha='right')
    ax.set_yticks(range(0, len(drop_labels), max(1, len(drop_labels)//10)))
    ax.set_yticklabels(drop_labels[::max(1, len(drop_labels)//10)])
    ax.set_xlabel("Rebounce Bucket (%)")
    ax.set_ylabel("Drop Range (change_pct)")
    ax.set_zlabel("Percent (%)")
    ax.set_title("3D Rebounce Distribution by Drop Range and Rebounce Bucket (Line Chart)")

    plt.tight_layout()
    plt.show()



def plot_bar():
    percent_matrix = np.array(Z)  # Use Z as the percent matrix

    # Plot 3D bar chart
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(len(rebounce_buckets))       # rebounce buckets (x)
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
    ax.set_xticklabels(rebounce_buckets, rotation=35, ha='right')
    ax.set_yticks(np.arange(len(drop_ranges)) + 0.35)
    ax.set_yticklabels([f"{u} to {l}" for u, l in drop_ranges])

    plt.title('3D Rebounce Distribution by Drop Range and Rebounce Bucket')
    plt.tight_layout()
    plt.show()


if plot_type == 'bar':
    plot_bar()
elif plot_type == 'line':   
    plot_line()
else:
    plot_surface()