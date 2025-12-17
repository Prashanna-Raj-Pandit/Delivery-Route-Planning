# experiment3.py
# Adaptive Priority-Max Coverage (S→N spine with per-dataset behavior)
# - Divide: Quad-tree regions
# - Corridor: near S→N path; width adapts to dataset composition
# - Conquer: S→N slabs, strict P1→P2→P3 ordering; weights adapt to rarity (bounded)
# - Budget OFF for all datasets (cover all selected regions fully)
# - Memory usage logged (RSS + tracemalloc peak)
#
# Outputs:
#   - experiment3_priority_summary.csv
#   - experiment3_priority_memory.csv
#   - figures: Regions_and_Points_*.png, Priority_Coverage_*.png, Total_vs_Covered.png, Route — *.png

import os
import time
import math
import tracemalloc
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional

from divideconquer import DivideConquerRouter

# --------------------
# STATIC BASE CONFIG (defaults; can be overridden per dataset)
# --------------------
GRAPH_FILE = "chicago_street_network.graphml"

DATASETS = {
    "Equal":        "delivery_points_1000_equal.csv",
    "HighMajority": "delivery_points_1000_high_dominance.csv",
    "LowMajority":  "delivery_points_1000_low_dominance.csv",
}

MAX_POINTS_PER_REGION_DEFAULT = 22
CORRIDOR_WIDTH_KM_DEFAULT     = 6.0
NUM_SLABS_DEFAULT             = 16
PER_SLAB_TOP_K_DEFAULT        = None       # None => take all per slab
MIN_REGION_SIZE_DEFAULT       = 5
MIN_NEW_RATIO_DEFAULT         = 0.15       # relaxed for coverage-first
BUDGET_MODE_DEFAULT           = "none"     # force budget off globally

# --------------------
# HELPERS (priority, spine geometry)
# --------------------
def priority_counts_df(df: pd.DataFrame):
    return (
        int((df['priority'] == 1).sum()),
        int((df['priority'] == 2).sum()),
        int((df['priority'] == 3).sum()),
    )

def build_spine(router: DivideConquerRouter, start='S', end='N'):
    s = router.depot_nodes[start]; e = router.depot_nodes[end]
    nodes = nx.shortest_path(router.G, s, e, weight='length')
    coords = router._coords_for_nodes(nodes)  # [(lon,lat), ...]
    return nodes, coords

def min_dist_region_to_spine_km(router: DivideConquerRouter, region, spine_coords):
    min_d = float('inf')
    for (lon1, lat1), (lon2, lat2) in zip(spine_coords, spine_coords[1:]):
        d = router.point_to_segment_distance(
            region['centroid_lat'], region['centroid_lon'],
            lat1, lon1, lat2, lon2
        )
        if d < min_d:
            min_d = d
    return float(min_d)

def projection_along_spine(router: DivideConquerRouter, region, spine_coords):
    return router._projection_score(
        region,
        spine_coords[0][1], spine_coords[0][0],
        spine_coords[-1][1], spine_coords[-1][0]
    )

def make_slabs_by_projection(ordered_regions, num_slabs):
    if not ordered_regions:
        return [[] for _ in range(num_slabs)]
    per = max(1, len(ordered_regions)//num_slabs)
    slabs, i = [], 0
    for _ in range(num_slabs - 1):
        slabs.append(ordered_regions[i:i+per])
        i += per
    slabs.append(ordered_regions[i:])
    return slabs

# --------------------
# ADAPTIVE PARAMETER LOGIC (per dataset)
# --------------------
def adjust_params(points: pd.DataFrame, router: DivideConquerRouter):
    """Derive weights and exploration shape from dataset composition."""
    total = len(points)
    p1 = int((points['priority']==1).sum())
    p2 = int((points['priority']==2).sum())
    p3 = int((points['priority']==3).sum())
    if total == 0:
        total = 1
    p1r, p2r, p3r = p1/total, p2/total, p3/total

    # Base ladder keeps P1 ≥ P2 ≥ P3; rarity adds a bounded bonus
    BASE_W1, BASE_W2, BASE_W3 = 1.0, 0.6, 0.2
    def rarity_bonus(r):
        r = max(r, 1e-6)
        return min(1.0, 0.5 / r) * 0.25  # small capped boost
    W1 = BASE_W1 + rarity_bonus(p1r)
    W2 = BASE_W2 + rarity_bonus(p2r)
    W3 = BASE_W3 + rarity_bonus(p3r)
    S = W1 + W2 + W3
    W1, W2, W3 = (100*x/S for x in (W1, W2, W3))

    # Exploration knobs based on skew
    corridor_width_km = CORRIDOR_WIDTH_KM_DEFAULT
    per_slab_top_k    = PER_SLAB_TOP_K_DEFAULT
    num_slabs         = NUM_SLABS_DEFAULT
    min_new_ratio     = MIN_NEW_RATIO_DEFAULT
    max_points_per_region = MAX_POINTS_PER_REGION_DEFAULT

    if p1r >= 0.60:            # many highs: tighter & slightly more slabs
        corridor_width_km = 4.0
        per_slab_top_k    = 3
        num_slabs         = 18
        min_new_ratio     = 0.20
    elif p3r >= 0.40:          # many lows: widen to hunt scarce P1 pockets
        corridor_width_km = 9.0
        per_slab_top_k    = None
        num_slabs         = 14
        min_new_ratio     = 0.12
    else:                      # balanced
        corridor_width_km = 6.5
        per_slab_top_k    = None
        num_slabs         = 16
        min_new_ratio     = 0.15

    baseline_km = router.baseline_sn_distance_km('S','N')

    # Budget is OFF globally in this experiment
    budget_mode   = "none"
    max_km_budget = None

    if p3r >= 0.40:
        max_points_per_region = MAX_POINTS_PER_REGION_DEFAULT + 4

    return {
        'weights': (W1, W2, W3),
        'corridor_width_km': corridor_width_km,
        'per_slab_top_k': per_slab_top_k,
        'num_slabs': num_slabs,
        'min_new_ratio': min_new_ratio,
        'max_points_per_region': max_points_per_region,
        'budget_mode': budget_mode,
        'max_km_budget': max_km_budget,
        'composition': {'p1': p1, 'p2': p2, 'p3': p3, 'total': total,
                        'p1r': p1r, 'p2r': p2r, 'p3r': p3r, 'baseline_km': baseline_km}
    }

# --------------------
# REGION SELECTION (strict P1→P2→P3, with adaptive weights as tie-breaker)
# --------------------
def select_regions_priority_first(router: DivideConquerRouter,
                                  points: pd.DataFrame,
                                  *,
                                  start='S', end='N',
                                  max_points_per_region: int,
                                  corridor_width_km: float,
                                  num_slabs: int,
                                  per_slab_top_k,
                                  min_region_size: int,
                                  weights: tuple):
    """Return selected region dicts in S→N order with strict (p1,p2,p3,size) cascade."""
    W1, W2, W3 = weights
    def priority_score(p1,p2,p3):
        return W1*p1 + W2*p2 + W3*p3

    regions = router.quad_tree_decomposition(points, max_points_per_region=max_points_per_region)
    kept = []
    spine_nodes, spine_coords = build_spine(router, start, end)

    for r in regions:
        rp = r['points']
        if len(rp) < min_region_size:
            continue
        min_d = min_dist_region_to_spine_km(router, r, spine_coords)
        if min_d > corridor_width_km:
            continue
        p1, p2, p3 = priority_counts_df(rp)
        proj  = projection_along_spine(router, r, spine_coords)
        kept.append({
            'region': r,
            'proj': proj,
            'min_d_km': min_d,
            'size': len(rp),
            'p1': p1, 'p2': p2, 'p3': p3,
            'score': priority_score(p1, p2, p3)
        })

    kept.sort(key=lambda x: x['proj'])
    slabs = make_slabs_by_projection(kept, num_slabs=num_slabs)

    selected = []
    for slab in slabs:
        if not slab:
            continue
        # STRICT priority cascade; score as final tie-breaker
        slab_sorted = sorted(
            slab,
            key=lambda x: (x['p1'], x['p2'], x['p3'], x['size'], x['score']),
            reverse=True
        )
        if per_slab_top_k is None:
            selected.extend(slab_sorted)
        else:
            selected.extend(slab_sorted[:per_slab_top_k])

    selected.sort(key=lambda x: x['proj'])
    return selected, spine_nodes

# --------------------
# ROUTE STITCHING (budget OFF path is used)
# --------------------
def build_priority_route(router: DivideConquerRouter,
                         selected_regions: list,
                         spine_nodes: list,
                         points: pd.DataFrame,
                         *,
                         start='S', end='N',
                         min_new_ratio: float,
                         budget_mode: str,
                         max_km_budget: Optional[float]):
    """
    budget_mode:
      - "none":    ignore budget, visit all deliveries in each selected region
      - (other modes kept for API compatibility, but not used in this experiment)
    """
    s_node = router.depot_nodes[start]
    e_node = router.depot_nodes[end]

    route = [s_node]
    current = s_node
    visited_deliveries = set()
    total_m = 0.0

    for item in selected_regions:
        reg = item['region']
        rp  = reg['points']
        reg_nodes = set(rp['nearest_node'])

        # ---- FIXED: newness gate uses unique nodes, not raw points ----
        new_nodes = reg_nodes - visited_deliveries
        unique_nodes_in_region = len(reg_nodes)
        min_required_new_nodes = max(1, int(unique_nodes_in_region * min_new_ratio))
        if len(new_nodes) < min_required_new_nodes:
            continue

        dcur = router.dijkstra_cache(current)
        def dist_or_inf(n):
            v = dcur.get(n, float('inf'))
            return float(v) if v is not None else float('inf')
        try:
            region_entry = min(new_nodes, key=lambda n: dist_or_inf(n))
        except ValueError:
            continue

        # Path to region
        try:
            to_region_path = nx.shortest_path(router.G, current, region_entry, weight='length')
            to_region_cost = nx.shortest_path_length(router.G, current, region_entry, weight='length')
        except Exception:
            to_region_path, to_region_cost = [current, region_entry], 0.0

        # budget_mode is "none": visit the full region
        reg_path, reg_cost, reg_visited = router._full_region_path_and_cost(rp, region_entry)

        route.extend(to_region_path[1:])
        route.extend(reg_path[1:])
        total_m += float(to_region_cost) + float(reg_cost)
        visited_deliveries.update(reg_visited)
        current = reg_path[-1] if reg_path else current

    # Finish to end depot
    if current != e_node:
        try:
            final_path = nx.shortest_path(router.G, current, e_node, weight='length')
            final_cost = nx.shortest_path_length(router.G, current, e_node, weight='length')
        except Exception:
            final_path, final_cost = [current, e_node], 0.0
        route.extend(final_path[1:])
        total_m += float(final_cost)

    return route, total_m, visited_deliveries

# --------------------
# VIS
# --------------------
def plot_points_by_priority_with_regions(router: DivideConquerRouter,
                                         points: pd.DataFrame,
                                         selected_regions: list,
                                         name: str):
    fig, ax = plt.subplots(figsize=(14, 16), dpi=180)
    router._get_city_boundary().boundary.plot(ax=ax, linewidth=2.0, edgecolor="#666", alpha=0.9, zorder=0)
    router.edges.plot(ax=ax, linewidth=0.35, edgecolor="#B0B0B0", alpha=0.85, zorder=1)

    cmap = {1: ("#FF3B30", "#8B0000"),
            2: ("#FFA500", "#8B5A00"),
            3: ("#00C853", "#006400")}
    for p in [1, 2, 3]:
        sub = points[points['priority'] == p]
        if not sub.empty:
            face, edge = cmap[p]
            ax.scatter(sub['lon'], sub['lat'], s=22, c=face, edgecolors=edge, linewidths=0.5,
                       alpha=0.95, zorder=4, label=f'Priority {p}')

    for item in selected_regions:
        b = item['region']['bounds']
        rect = plt.Rectangle((b['min_lon'], b['min_lat']),
                             b['max_lon'] - b['min_lon'],
                             b['max_lat'] - b['min_lat'],
                             fill=False, lw=2.0, ec='#1F77B4', alpha=0.95, zorder=6)
        ax.add_patch(rect)
        ax.scatter(item['region']['centroid_lon'], item['region']['centroid_lat'],
                   c='#1F77B4', s=36, marker='x', zorder=7)

    for name_d, d in router.depots.items():
        ax.scatter(d['lon'], d['lat'], s=460, marker='s', c=d['color'],
                   edgecolors='black', linewidth=3, zorder=10, label=f'{d["name"]} Depot')

    router._set_fixed_bounds(ax)
    ax.set_title(f"Regions and Points — {name}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True, alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(f"Regions_and_Points_{name}.png", dpi=320, bbox_inches='tight')
    plt.close()

def plot_total_vs_covered(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(summary_df)); w = 0.35
    ax.bar(x-w/2, summary_df['total_points'], width=w, label='Total')
    bars = ax.bar(x+w/2, summary_df['covered_points'], width=w, label='Covered')
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+3, f"{int(r.get_height())}",
                ha='center', va='bottom', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(summary_df['dataset'])
    ax.set_ylabel("Deliveries"); ax.set_title("Total vs Covered")
    ax.legend(); ax.grid(True, axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig("Total_vs_Covered.png", dpi=300); plt.close()

def plot_priority_coverage(name: str, covered_counts: dict, total_counts: dict):
    fig, ax = plt.subplots(figsize=(8,6))
    labels = ['P1','P2','P3']
    total = [total_counts['p1'], total_counts['p2'], total_counts['p3']]
    cov   = [covered_counts[1], covered_counts[2], covered_counts[3]]
    x = np.arange(3); w = 0.35
    ax.bar(x-w/2, total, width=w, label='Total')
    ax.bar(x+w/2, cov,   width=w, label='Covered')
    for i,v in enumerate(cov):
        ax.text(i+w/2, v+3, f"{int(v)}", ha='center', va='bottom', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Deliveries"); ax.set_title(f"Priority Coverage — {name}")
    ax.legend(); ax.grid(True, axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(f"Priority_Coverage_{name}.png", dpi=300); plt.close()

# --------------------
# MEMORY LOGGING
# --------------------
def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def mem_row(stage, dataset, t_s, rss_now_mb, peak_tracemalloc_mb):
    return {
        'stage': stage,
        'dataset': dataset,
        'time_s': round(t_s, 3),
        'rss_mb': round(rss_now_mb, 2),
        'tracemalloc_peak_mb': round(peak_tracemalloc_mb, 2),
    }

# --------------------
# MAIN
# --------------------
def run_experiment3_priority_focus():
    tracemalloc.start()
    mem_rows = []
    t_start = time.time()

    router = DivideConquerRouter(GRAPH_FILE)
    summary_rows = []
    mem_rows.append(mem_row('router_ready', 'ALL', time.time()-t_start,
                            rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

    for name, csv in DATASETS.items():
        t_dataset0 = time.time()
        print(f"\n=== {name} ===")
        points = router.load_and_preprocess_points(csv)
        mem_rows.append(mem_row('loaded_points', name, time.time()-t_dataset0,
                                rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

        # ---- ADAPT PARAMETERS FROM THIS DATASET ----
        params = adjust_params(points, router)
        W1, W2, W3 = params['weights']
        corridor_width_km = params['corridor_width_km']
        per_slab_top_k    = params['per_slab_top_k']
        num_slabs         = params['num_slabs']
        min_new_ratio     = params['min_new_ratio']
        max_points_per_region = params['max_points_per_region']
        # Force budget off for ALL datasets
        budget_mode       = "none"
        max_km_budget     = None
        comp              = params['composition']

        print(f"  Composition: P1={comp['p1']} ({comp['p1r']:.2%}), "
              f"P2={comp['p2']} ({comp['p2r']:.2%}), "
              f"P3={comp['p3']} ({comp['p3r']:.2%}), baseline={comp['baseline_km']:.1f} km")
        print(f"  Weights: W1={W1:.1f}, W2={W2:.1f}, W3={W3:.1f} | "
              f"Corridor={corridor_width_km:.1f} km | Slabs={num_slabs} | "
              f"TopK={per_slab_top_k} | MinNew={min_new_ratio:.2f} | "
              f"Budget={budget_mode} @ ∞")

        # 1) Select regions (adaptive)
        selected, spine_nodes = select_regions_priority_first(
            router, points,
            start='S', end='N',
            max_points_per_region=max_points_per_region,
            corridor_width_km=corridor_width_km,
            num_slabs=num_slabs,
            per_slab_top_k=per_slab_top_k,
            min_region_size=MIN_REGION_SIZE_DEFAULT,
            weights=(W1, W2, W3)
        )
        print(f"  Selected regions: {len(selected)}")
        mem_rows.append(mem_row('regions_selected', name, time.time()-t_dataset0,
                                rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

        # 2) Visual: regions + points by priority
        plot_points_by_priority_with_regions(router, points, selected, name)
        mem_rows.append(mem_row('plot_regions', name, time.time()-t_dataset0,
                                rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

        # 3) Build route (budget OFF)
        t0 = time.time()
        route, dist_m, visited = build_priority_route(
            router, selected, spine_nodes, points,
            start='S', end='N',
            min_new_ratio=min_new_ratio,
            budget_mode=budget_mode,
            max_km_budget=max_km_budget
        )
        dt = time.time() - t0
        mem_rows.append(mem_row('route_built', name, time.time()-t_dataset0,
                                rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

        # 4) Visualize final route
        router.visualize_route(
            route_nodes=route,
            delivery_points=points,
            visited_deliveries=visited,
            title=f"Route — {name}",
            start_depot='S', end_depot='N'
        )
        mem_rows.append(mem_row('plot_route', name, time.time()-t_dataset0,
                                rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

        # 5) Coverage accounting + plot
        all_counts = {
            'p1': int((points['priority']==1).sum()),
            'p2': int((points['priority']==2).sum()),
            'p3': int((points['priority']==3).sum()),
            'total': len(points)
        }
        covered_counts = {
            1: int((points[points['nearest_node'].isin(visited)]['priority']==1).sum()),
            2: int((points[points['nearest_node'].isin(visited)]['priority']==2).sum()),
            3: int((points[points['nearest_node'].isin(visited)]['priority']==3).sum()),
        }
        plot_priority_coverage(name, covered_counts, all_counts)

        dist_km = router.calculate_route_distance(route)/1000.0
        covered_total = covered_counts[1] + covered_counts[2] + covered_counts[3]
        print(f"  Distance: {dist_km:.1f} km | Covered: {covered_total}/{all_counts['total']} "
              f"(P1={covered_counts[1]}, P2={covered_counts[2]}, P3={covered_counts[3]}) | Time: {dt:.2f}s")

        summary_rows.append({
            'dataset': name,
            'total_points': all_counts['total'],
            'covered_points': covered_total,
            'covered_p1': covered_counts[1],
            'covered_p2': covered_counts[2],
            'covered_p3': covered_counts[3],
            'distance_km': dist_km,
            'regions_selected': len(selected),
            'runtime_s': dt,
            'budget_mode': budget_mode,
            'max_km_budget': None,
            'corridor_km': corridor_width_km,
            'num_slabs': num_slabs,
            'top_k': per_slab_top_k,
            'W1': round(W1,2), 'W2': round(W2,2), 'W3': round(W3,2),
            'p1_ratio': round(comp['p1r'],4),
            'p2_ratio': round(comp['p2r'],4),
            'p3_ratio': round(comp['p3r'],4)
        })

        mem_rows.append(mem_row('dataset_done', name, time.time()-t_dataset0,
                                rss_mb(), tracemalloc.get_traced_memory()[1]/(1024**2)))

    # Save CSVs
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv("experiment3_priority_summary.csv", index=False)
        plot_total_vs_covered(df)
        print("\nSaved: experiment3_priority_summary.csv and Total_vs_Covered.png")

    overall_peak_mb = tracemalloc.get_traced_memory()[1]/(1024**2)
    mem_rows.append(mem_row('overall_done', 'ALL', time.time()-t_start,
                            rss_mb(), overall_peak_mb))
    pd.DataFrame(mem_rows).to_csv("experiment3_priority_memory.csv", index=False)
    print("Saved: experiment3_priority_memory.csv")

    return pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

if __name__ == "__main__":
    _ = run_experiment3_priority_focus()
