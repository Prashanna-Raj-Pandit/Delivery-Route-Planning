# experiment3_greedy.py — Priority-aware Greedy (S->N), NO BUDGET
# Outputs per dataset:
#   Greedy_Priority_<Dataset>.png
#   Greedy_Priority_<Dataset>_P1.png / _P2.png / _P3.png
# Summary CSVs & plots:
#   experiment3_priority_summary_greedy.csv
#   experiment3_priority_totals_vs_covered.csv
#   Greedy_Priority_Summary_Coverage.png
#   Greedy_Priority_Summary_Distance.png
#   Greedy_Priority_Total_vs_Covered.png
#   Greedy_Priority_Total_vs_Covered_ByPriority.png

import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from greedy import GreedyRouter

# -------- datasets (CSV must include a 'priority' column with values {1,2,3}) --------
DATASETS = {
    "Equal":        "delivery_points_1000_equal.csv",
    "HighMajority": "delivery_points_1000_high_majority.csv",
    "LowMajority":  "delivery_points_1000_low_majority.csv",
}

# -------- routing config --------
START, END = "S", "N"           # South -> North

# Priority weights (same as D&C): focus on P1/P2; ignore P3 in scoring
W1, W2, W3 = 50, 30, 0
PRIORITY_WEIGHTS = {1: W1, 2: W2, 3: W3}

# Optional runtime guard: set to an int to cap how many delivery nodes we visit
# None => pure greedy until no candidates remain
MAX_VISITS = None

# ---------- priority-aware greedy (NO budget) ----------
def priority_greedy_route(router: GreedyRouter, points: pd.DataFrame,
                          start_depot: str, end_depot: str,
                          weights: dict,
                          max_visits: int | None = None):
    """
    Pure greedy, speed-first (NO budget):
      - Drops only unreachable candidates (must have paths current→n and n→end).
      - Scores nodes by priority×rows divided by a smooth detour penalty.
      - Strongly prefers forward progress along S→N and tiny detours.
    """
    import networkx as nx

    # --- knobs for the soft penalty (tune if you like) ---
    ALPHA_DETOUR = 2.0   # exponent; larger => harsher on big detours
    LAMBDA_HOP   = 0.05  # tiny cost on hop distance current→n (stabilizes)
    BACK_PENALTY = 5.0   # extra penalty if n is *behind* current relative to end
    EPS          = 1e-6

    G = router.G
    start = router.depot_nodes[start_depot]
    end   = router.depot_nodes[end_depot]

    # rows / priority per nearest_node
    rows_at_node = points.groupby('nearest_node').size().to_dict()
    pri_at_node  = points.groupby('nearest_node')['priority'].first().to_dict()
    candidates   = set(rows_at_node.keys())

    # Precompute dist-to-end on reversed graph; keep only nodes that can reach end
    try:
        revG = G.reverse(copy=False)
        to_end_len = nx.single_source_dijkstra_path_length(revG, end, weight='length')
    except Exception:
        to_end_len = {}
    candidates &= set(to_end_len.keys())

    def dist_to_end(n):  # meters
        return float(to_end_len.get(n, float('inf')))

    current = start
    route   = [current]
    visited_nodes = []
    dist_so_far = 0.0

    # If start itself can't reach end, try direct; otherwise stop.
    if dist_to_end(current) == float('inf'):
        try:
            path = nx.shortest_path(G, current, end, weight='length')
            length = nx.shortest_path_length(G, current, end, weight='length')
            return path, float(length), []
        except nx.NetworkXNoPath:
            return [current], 0.0, []

    while candidates and (max_visits is None or len(visited_nodes) < max_visits):
        # distances from current to all reachable nodes
        d_from_current = nx.single_source_dijkstra_path_length(G, current, weight='length')
        d_direct = dist_to_end(current)  # current -> end

        # Prefer P1/P2 first; if none reachable, consider any
        preferred = [n for n in candidates if int(pri_at_node.get(n, 3)) <= 2]
        consider  = preferred if preferred else list(candidates)

        best = None
        best_score = -1.0
        best_d = None

        for n in consider:
            d = d_from_current.get(n)
            if d is None:
                continue  # not visitable from current
            d_end = dist_to_end(n)
            if not np.isfinite(d_end):
                continue  # can't finish after visiting n

            # detour vs going straight to the end
            detour = (float(d) + d_end) - d_direct
            if detour < 0:
                detour = 0.0  # numerical guard

            # soft, speed-first penalty: tiny detours dominate, big ones crushed
            denom = (detour + EPS)**ALPHA_DETOUR + LAMBDA_HOP*float(d)

            # extra penalty if this move goes "backwards" (increases distance to end)
            if d_end >= d_direct:
                denom *= BACK_PENALTY

            p = int(pri_at_node.get(n, 3))
            w = float(weights.get(p, 0.0))       # P3 can be 0
            rows_here = int(rows_at_node.get(n, 1))

            score = (w * rows_here) / denom

            if score > best_score:
                best, best_score, best_d = n, score, float(d)

        if best is None:
            break  # nothing reachable that improves score

        path = nx.shortest_path(G, current, best, weight='length')
        route.extend(path[1:])
        dist_so_far += best_d
        visited_nodes.append(best)
        candidates.discard(best)
        current = best

    # Always connect to end (no budget)
    try:
        tail = nx.shortest_path(G, current, end, weight='length')
        tail_d = nx.shortest_path_length(G, current, end, weight='length')
        route.extend(tail[1:])
        dist_so_far += float(tail_d)
    except nx.NetworkXNoPath:
        pass

    return route, dist_so_far, visited_nodes


# ---------- plotting helper ----------
def _bar_with_labels(ax, x, y, w, label, ymax):
    bars = ax.bar(x, y, width=w, label=label)
    for xi, v in zip(x, y):
        ax.text(xi, v + 0.01*max(1, ymax), str(int(v)), ha='center', va='bottom', fontsize=9)
    return bars

# ---------------------------- main ----------------------------
def run_experiment_3_greedy():
    print("="*64)
    print("PRIORITY FOCUS — GREEDY (pure, no budget) S → N")
    print("="*64)

    router = GreedyRouter("chicago_street_network.graphml")

    rows = []
    for name, csv_path in DATASETS.items():
        try:
            pts = router.load_and_preprocess_points(csv_path)
            print(f"Loaded {csv_path} ({name})")
        except FileNotFoundError:
            print(f"Missing {csv_path}; skipping {name}.")
            continue

        if 'priority' not in pts.columns:
            raise ValueError(f"{csv_path} must include a 'priority' column with values 1,2,3.")
        pts['priority'] = pts['priority'].astype(int).clip(1, 3)

        tracemalloc.start()
        t0 = time.time()
        route, dist_m, visited = priority_greedy_route(
            router, pts, START, END, PRIORITY_WEIGHTS, max_visits=MAX_VISITS
        )
        elapsed = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        km = float(dist_m) / 1000.0

        # row-based coverage on the full dataset
        covered_rows = pts[pts['nearest_node'].isin(visited)]
        coverage_pct = 100.0 * len(covered_rows) / len(pts) if len(pts) else 0.0
        eff = (len(covered_rows) / km) if km > 0 else 0.0

        # overall map
        router.visualize_route(
            route, pts, covered_rows['nearest_node'].unique().tolist(),
            f"Greedy_Priority_{name}", start_depot=START, end_depot=END
        )
        # per-priority overlays
        for p in (1, 2, 3):
            pts_p = pts[pts['priority'] == p].copy()
            visited_p = pts_p[pts_p['nearest_node'].isin(visited)]['nearest_node'].unique().tolist()
            router.visualize_route(
                route, pts_p, visited_p, f"Greedy_Priority_{name}_P{p}",
                start_depot=START, end_depot=END
            )

        # per-priority covered counts
        per_p = (covered_rows.groupby('priority').size()
                 .reindex([1,2,3], fill_value=0)
                 .rename({1:'P1',2:'P2',3:'P3'})).to_dict()

        # per-priority TOTAL counts
        tot_p = (pts['priority'].value_counts()
                 .reindex([1,2,3], fill_value=0)
                 .rename({1:'P1',2:'P2',3:'P3'})).to_dict()

        print(f"{name}: {km:.1f} km | {coverage_pct:.1f}% covered "
              f"(overall {len(covered_rows)}/{len(pts)}; "
              f"P1={per_p.get('P1',0)}/{tot_p.get('P1',0)}, "
              f"P2={per_p.get('P2',0)}/{tot_p.get('P2',0)}, "
              f"P3={per_p.get('P3',0)}/{tot_p.get('P3',0)}) in {elapsed:.2f}s")

        rows.append({
            'dataset': name,
            'distance_km': km,
            'coverage_pct': coverage_pct,
            'efficiency_deliveries_per_km': eff,
            'time_s': elapsed,
            'peak_mem_bytes': int(peak),
            # covered (per priority)
            'covered_P1': per_p.get('P1', 0),
            'covered_P2': per_p.get('P2', 0),
            'covered_P3': per_p.get('P3', 0),
            # totals (per priority + overall)
            'total_P1': tot_p.get('P1', 0),
            'total_P2': tot_p.get('P2', 0),
            'total_P3': tot_p.get('P3', 0),
            'total_rows': len(pts),
        })

    if not rows:
        print("No datasets found; nothing saved.")
        return

    df = pd.DataFrame(rows).sort_values('dataset')
    df.to_csv("experiment3_priority_summary_greedy.csv", index=False)

    # ---- Totals vs Covered (overall + per priority) ----
    df['covered_total'] = df[['covered_P1','covered_P2','covered_P3']].sum(axis=1)
    df['total_overall'] = df[['total_P1','total_P2','total_P3']].sum(axis=1)
    out_counts = df[['dataset',
                     'total_overall', 'covered_total',
                     'total_P1','covered_P1',
                     'total_P2','covered_P2',
                     'total_P3','covered_P3']].copy()
    out_counts.to_csv("experiment3_priority_totals_vs_covered.csv", index=False)

    # Summary plots (coverage %, distance)
    plt.figure(figsize=(9,6))
    plt.bar(df['dataset'], df['coverage_pct'])
    plt.xlabel("Dataset"); plt.ylabel("Coverage (%)")
    plt.title("Greedy Priority — Coverage by Dataset (No Budget)")
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig("Greedy_Priority_Summary_Coverage.png", dpi=220); plt.close()

    plt.figure(figsize=(9,6))
    plt.bar(df['dataset'], df['distance_km'])
    plt.xlabel("Dataset"); plt.ylabel("Distance (km)")
    plt.title("Greedy Priority — Distance by Dataset (No Budget)")
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig("Greedy_Priority_Summary_Distance.png", dpi=220); plt.close()

    # Plot A: Total vs Covered (overall) per dataset
    plt.figure(figsize=(10,6))
    x = np.arange(len(df)); w = 0.35
    ymax = max(df['total_overall'].max(), df['covered_total'].max())
    _bar_with_labels(plt.gca(), x - w/2, df['total_overall'], w, 'Total', ymax)
    _bar_with_labels(plt.gca(), x + w/2, df['covered_total'], w, 'Covered', ymax)
    plt.xticks(x, df['dataset'])
    plt.ylabel("Deliveries (rows)")
    plt.title("Greedy Priority — Total vs Covered (No Budget)")
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Greedy_Priority_Total_vs_Covered.png", dpi=220)
    plt.close()

    # Plot B: Total vs Covered by Priority (groups of P1/P2/P3 per dataset)
    plt.figure(figsize=(12,6))
    groups = ['P1','P2','P3']
    x0 = np.arange(len(df)); wb = 0.18
    ymax_p = max(df[['total_P1','total_P2','total_P3']].max())
    for gi, g in enumerate(groups):
        xg = x0 + (gi-1)*wb*2
        _bar_with_labels(plt.gca(), xg - wb/2, df[f'total_{g}'],   wb, None, ymax_p)
        _bar_with_labels(plt.gca(), xg + wb/2, df[f'covered_{g}'], wb, None, ymax_p)
    plt.bar([], [], label='Total'); plt.bar([], [], label='Covered'); plt.legend(['Total', 'Covered'])
    plt.xticks(x0, df['dataset']); plt.ylabel("Deliveries (rows)")
    plt.title("Greedy Priority — Total vs Covered by Priority (No Budget)")
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig("Greedy_Priority_Total_vs_Covered_ByPriority.png", dpi=220); plt.close()

        # ---- Time & Memory (by dataset) ----
    # Present memory in MB for readability
    df['peak_mem_mb'] = df['peak_mem_bytes'] / (1024 * 1024)

    # Bar: Time (s) by dataset
    plt.figure(figsize=(9,6))
    bars = plt.bar(df['dataset'], df['time_s'])
    plt.xlabel("Dataset"); plt.ylabel("Time (s)")
    plt.title("Greedy Priority — Time by Dataset (No Budget)")
    plt.grid(True, axis='y', alpha=0.3)
    ymax = max(df['time_s'].max(), 1.0)
    for b in bars:
        h = b.get_height()
        plt.annotate(f"{h:.2f}", (b.get_x()+b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig("Greedy_Priority_Time_By_Dataset.png", dpi=220)
    plt.close()

    # Bar: Peak memory (MB) by dataset
    plt.figure(figsize=(9,6))
    bars = plt.bar(df['dataset'], df['peak_mem_mb'])
    plt.xlabel("Dataset"); plt.ylabel("Peak memory (MB)")
    plt.title("Greedy Priority — Peak Memory by Dataset (No Budget)")
    plt.grid(True, axis='y', alpha=0.3)
    ymax = max(df['peak_mem_mb'].max(), 1.0)
    for b in bars:
        h = b.get_height()
        plt.annotate(f"{h:.1f}", (b.get_x()+b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig("Greedy_Priority_PeakMem_By_Dataset.png", dpi=220)
    plt.close()

   
    print("\nSaved:")
    print("  - experiment3_priority_summary_greedy.csv")
    print("  - experiment3_priority_totals_vs_covered.csv")
    print("  - Greedy_Priority_Summary_Coverage.png")
    print("  - Greedy_Priority_Summary_Distance.png")
    print("  - Greedy_Priority_Total_vs_Covered.png")
    print("  - Greedy_Priority_Total_vs_Covered_ByPriority.png")
    print("  - Per-dataset maps: Greedy_Priority_<Dataset>.png and _P1/_P2/_P3")
    print("  - Greedy_Priority_Time_By_Dataset.png")
    print("  - Greedy_Priority_PeakMem_By_Dataset.png")


if __name__ == "__main__":
    run_experiment_3_greedy()
