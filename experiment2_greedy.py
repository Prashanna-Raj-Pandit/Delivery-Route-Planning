# experiment2_greedy.py — Greedy-only with per-pair distance budget
# - Uses your exact depot pairs
# - Budget = BASELINE_MULT × baseline(start→end); reserve keeps room to finish
# - Saves route PNGs, coverage & distance heatmaps, covered-by-pair bar, and a performance heatmap

import time
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from greedy import GreedyRouter

# ---------- EXACT ordered depot pairs ----------
PAIRS = [
    ('S','E'),
    ('C','E'),
    ('W','S'),
    ('E','W'),
    ('N','S'),
]

# ---------- Greedy options ----------
COUNT_ON_FULL_SET = True   # score coverage vs the full 1000 rows
BASELINE_MULT     = 3.0    # try 2.0–3.5; lower => shorter routes, less coverage
RESERVE_PCT       = 0.10   # keep 10% of budget to guarantee reaching the end

# ---------------------------- helpers ----------------------------
def _baseline_km(router: GreedyRouter, s: str, e: str) -> float:
    """Shortest-path distance (km) between depots s->e on the road graph."""
    u = router.depot_nodes[s]; v = router.depot_nodes[e]
    return nx.shortest_path_length(router.G, u, v, weight='length') / 1000.0

def _save_heatmap_matrix(matrix_df, title, outfile):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    im = ax.imshow(matrix_df.values, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(matrix_df.shape[1])); ax.set_xticklabels(matrix_df.columns, rotation=0)
    ax.set_yticks(range(matrix_df.shape[0])); ax.set_yticklabels(matrix_df.index)
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            val = matrix_df.iloc[i, j]
            s = f"{val:.1f}" if isinstance(val, float) else str(val)
            ax.text(j, i, s, ha='center', va='center', fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(outfile, dpi=240); plt.close()

def _save_covered_count_bar(df, outfile):
    df = df.copy()
    df['pair'] = df['start'] + '→' + df['end']
    width = max(9, 1 + 0.6*len(df))  # must be a (w,h) tuple
    fig, ax = plt.subplots(figsize=(width, 5), dpi=160)
    ax.bar(df['pair'], df['covered_count'])
    ax.set_xlabel("Depot Pair")
    ax.set_ylabel("Deliveries Covered (count)")
    ax.set_title("Deliveries Covered by Depot Pair (n=1000)")
    ax.grid(True, axis='y', alpha=0.3)
    if len(df):
        ymax = df['covered_count'].max()
        for i, v in enumerate(df['covered_count']):
            ax.text(i, v + 0.01*ymax, str(v), ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(outfile, dpi=240); plt.close()

def _save_pair_bar(df, value_col, ylabel, title, outfile, decimals=2):
    df = df.copy()
    df['pair'] = df['start'] + '→' + df['end']
    width = max(9, 1 + 0.6*len(df))  # dynamic width
    fig, ax = plt.subplots(figsize=(width, 5), dpi=160)
    bars = ax.bar(df['pair'], df[value_col])
    ax.set_xlabel("Depot Pair")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    if len(df):
        ymax = df[value_col].max()
        for i, b in enumerate(bars):
            h = b.get_height()
            label = f"{h:.{decimals}f}" if isinstance(h, float) else str(h)
            ax.text(b.get_x() + b.get_width()/2, h + 0.01*ymax, label,
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(outfile, dpi=240); plt.close()


def _save_heatmap(df, value_col, title, outfile, order):
    piv = (df.pivot(index='start', columns='end', values=value_col)
             .reindex(index=order, columns=order))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    im = ax.imshow(piv.to_numpy(), aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.to_numpy()[i, j]
            if val is not None and not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(outfile, dpi=240); plt.close()

# ---------------------------- main ----------------------------
def run_experiment_2_greedy():
    print("="*64)
    print("MULTI-DEPOT CLEAN SINGLE-PATH — GREEDY (1000 pts, baseline-budgeted)")
    print("="*64)

    router = GreedyRouter("chicago_street_network.graphml")
    order = list(router.depots.keys())

    # validate pairs against available depots
    pairs = [(a,b) for (a,b) in PAIRS if a in router.depots and b in router.depots and a != b]
    if not pairs:
        raise ValueError("No valid depot pairs after validation against router.depots. Check PAIRS and self.depots.")

    # load full dataset
    points_all = router.load_and_preprocess_points("delivery_points_1000.csv")
    print("Loaded delivery_points_1000.csv")

    rows = []
    perf_rows = []  # for the multi-metric heatmap
    for s, e in pairs:
        base = _baseline_km(router, s, e)
        budget_km = base * BASELINE_MULT
        print(f"{s}→{e}: baseline {base:.1f} km | budget {budget_km:.1f} km (reserve {RESERVE_PCT:.0%})")

        tracemalloc.start()
        t0 = time.time()
        route, dist_m, _rs, visited = router.solve_greedy_single_path(
            points_all, start_depot=s, end_depot=e,
            budget_km=budget_km, reserve_pct=RESERVE_PCT
        )
        elapsed = time.time() - t0
        km = dist_m / 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # row-based coverage
        universe = points_all if COUNT_ON_FULL_SET else points_all
        covered_rows = universe[universe['nearest_node'].isin(visited)]
        covered_count = len(covered_rows)
        cov_pct = 100.0 * covered_count / len(universe)
        eff = (covered_count / km) if km > 0 else 0.0

        print(f"  result: {km:.1f} km | covered {covered_count} ({cov_pct:.1f}%) in {elapsed:.2f}s")

        # route map
        router.visualize_route(
            route,
            universe,
            covered_rows['nearest_node'].unique().tolist(),
            f"Greedy_{s}_to_{e}",
            start_depot=s, end_depot=e
        )

        rows.append({
            'start': s, 'end': e,
            'distance_km': km,
            'covered_count': covered_count,
            'coverage_pct': cov_pct,
            'time_s': elapsed,
            'peak_mem_bytes': int(peak),
            'eff_deliveries_per_km': eff,
            'total_points': len(universe),
        })

        perf_rows.append({
            'pair': f"{s}→{e}",
            'Distance (km)': km,
            'Time (s)': elapsed,
            'Efficiency (deliv/km)': eff,
            'Covered (rows)': covered_count
        })

    # summary CSV
    df = pd.DataFrame(rows)
    df.to_csv("experiment_2_results_greedy.csv", index=False)

    # Prep memory in MB
    df['peak_mem_mb'] = df['peak_mem_bytes'] / (1024 * 1024)


    # --- Bar charts by depot pair ---
    _save_pair_bar(df, 'time_s',
               "Time (s)", "Computation Time by Depot Pair",
               "Experiment2_Time_By_Pair.png", decimals=2)

    _save_pair_bar(df, 'peak_mem_mb',
               "Peak Memory (MB)", "Peak Memory by Depot Pair",
               "Experiment2_PeakMem_By_Pair.png", decimals=1)


    # coverage & distance heatmaps on Start×End grid
    _save_heatmap(df, 'coverage_pct', "Coverage (%) — Start×End", "Experiment2_Heatmap_Coverage.png", order)
    _save_heatmap(df, 'distance_km', "Distance (km) — Start×End", "Experiment2_Heatmap_Distance.png", order)

    # covered-by-pair bar
    _save_covered_count_bar(df, "Experiment2_Covered_By_Pair.png")

    # multi-metric performance heatmap (rows: pairs, cols: metrics)
    perf_df = pd.DataFrame(perf_rows).set_index('pair')
    _save_heatmap_matrix(perf_df, "Greedy Multi-Depot — Performance Heatmap", "Experiment2_Performance_Heatmap.png")

    print("\nSaved:")
    print("  - experiment_2_results_greedy.csv")
    print("  - Experiment2_Heatmap_Coverage.png")
    print("  - Experiment2_Heatmap_Distance.png")
    print("  - Experiment2_Covered_By_Pair.png")
    print("  - Experiment2_Performance_Heatmap.png")
    print("  - Experiment2_Time_By_Pair.png")
    print("  - Experiment2_PeakMem_By_Pair.png")
 
    print("  - Route PNGs per depot pair (Greedy_<S>_to_<E>.png)")

if __name__ == "__main__":
    run_experiment_2_greedy()
