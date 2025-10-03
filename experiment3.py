# experiment3.py
# Priority-focused Divide & Conquer (same spine/corridor framework as Exp 1 & 2)
# - Divide: Quad-tree regions
# - Corridor: keep only regions near the S→N shortest path (same idea as Exp 1/2)
# - Conquer: select regions per S→N order, prioritizing High (P1) and Medium (P2)
#             using simple per-slab top-K (to avoid early slabs hogging everything)
# - Distance is NOT optimized; corridor + per-slab cap prevent runaway detours
# - Uses DivideConquerRouter unchanged

import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from divideconquer import DivideConquerRouter

# --------------------
# CONFIG
# --------------------
GRAPH_FILE = "chicago_street_network.graphml"

DATASETS = {
    "Equal":        "delivery_points_1000_equal.csv",
    "HighMajority": "delivery_points_1000_high_majority.csv",
    "LowMajority":  "delivery_points_1000_low_majority.csv",
}

# Quad-tree & corridor (same flavor as Exp 1/2)
MAX_POINTS_PER_REGION = 20
CORRIDOR_WIDTH_KM     = 3.0

# Priority weights (focus on P1/P2)
W1, W2, W3 = 50, 30, 0  # intentionally ignore P3 in scoring

# S→N ordering control (keeps it D&C-ish without heavy DP)
NUM_SLABS        = 12   # split the spine into ordered bands
PER_SLAB_TOP_K   = 2    # pick at most K regions per slab (prevents hogging)
MIN_REGION_SIZE  = 6    # ignore tiny regions
MIN_NEW_RATIO    = 0.30 # skip region if <30% of its points are new vs already visited

# --------------------
# HELPERS
# --------------------
def priority_counts_df(df: pd.DataFrame):
    return (int((df['priority']==1).sum()),
            int((df['priority']==2).sum()),
            int((df['priority']==3).sum()))

def priority_score(p1, p2, p3):
    return W1*p1 + W2*p2 + W3*p3

def build_spine(router: DivideConquerRouter, start='S', end='N'):
    s = router.depot_nodes[start]; e = router.depot_nodes[end]
    nodes = nx.shortest_path(router.G, s, e, weight='length')
    coords = router._coords_for_nodes(nodes)  # [(lon,lat), ...]
    return nodes, coords

def min_dist_region_to_spine_km(router: DivideConquerRouter, region, spine_coords):
    # region centroid to each spine segment
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
    # Use the same projection score logic as your router to order regions S→N
    return router._projection_score(region,
                                    spine_coords[0][1], spine_coords[0][0],
                                    spine_coords[-1][1], spine_coords[-1][0])

def make_slabs_by_projection(ordered_regions, num_slabs=NUM_SLABS):
    """ordered_regions: list of dicts with key 'proj' sorted ascending.
       Returns list of lists (regions per slab)."""
    if not ordered_regions:
        return [[] for _ in range(num_slabs)]
    per = max(1, len(ordered_regions)//num_slabs)
    slabs = []
    i = 0
    for s in range(num_slabs-1):
        slabs.append(ordered_regions[i:i+per])
        i += per
    slabs.append(ordered_regions[i:])  # remainder in last slab
    return slabs

def select_regions_priority_first(router: DivideConquerRouter, points: pd.DataFrame,
                                  start='S', end='N',
                                  max_points_per_region=MAX_POINTS_PER_REGION,
                                  corridor_width_km=CORRIDOR_WIDTH_KM,
                                  num_slabs=NUM_SLABS,
                                  per_slab_top_k=PER_SLAB_TOP_K,
                                  min_region_size=MIN_REGION_SIZE):
    """Return a list of selected region dicts in S→N order, prioritized by P1/P2 within corridor."""
    regions = router.quad_tree_decomposition(points, max_points_per_region=max_points_per_region)

    # Keep only corridor regions; compute projection & priority
    kept = []
    spine_nodes, spine_coords = build_spine(router, start, end)
    for r in regions:
        rp = r['points']
        if len(rp) < min_region_size:
            continue
        # corridor filter (same guardrail as Exp 1/2)
        min_d = min_dist_region_to_spine_km(router, r, spine_coords)
        if min_d > corridor_width_km:
            continue
        p1, p2, p3 = priority_counts_df(rp)
        score = priority_score(p1, p2, p3)  # P1/P2 heavy, P3 ignored
        proj  = projection_along_spine(router, r, spine_coords)
        kept.append({
            'region': r,
            'proj': proj,
            'min_d_km': min_d,
            'size': len(rp),
            'p1': p1, 'p2': p2, 'p3': p3,
            'score': score
        })

    # Order S→N by projection, then slice into slabs & pick top-K per slab by score
    kept.sort(key=lambda x: x['proj'])
    slabs = make_slabs_by_projection(kept, num_slabs=num_slabs)
    selected = []
    for slab in slabs:
        if not slab:
            continue
        slab_sorted = sorted(slab, key=lambda x: (x['score'], x['p1'], x['p2'], x['size']), reverse=True)
        selected.extend(slab_sorted[:per_slab_top_k])

    # Maintain global S→N order
    selected.sort(key=lambda x: x['proj'])
    return selected, spine_nodes

def build_priority_route(router: DivideConquerRouter,
                         selected_regions: list,
                         spine_nodes: list,
                         points: pd.DataFrame,
                         start='S', end='N',
                         min_new_ratio=MIN_NEW_RATIO):
    """Stitch a full route visiting selected regions in S→N order; skip if mostly covered."""
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

        # Skip if <min_new_ratio are new
        new_nodes = reg_nodes - visited_deliveries
        if len(new_nodes) < max(1, int(item['size'] * min_new_ratio)):
            continue

        # Entry = nearest new node to current (graph distance)
        dcur = router.dijkstra_cache(current)
        # safeguard if some nodes missing in dcur
        def dist_or_inf(n): 
            v = dcur.get(n, float('inf'))
            return float(v) if v is not None else float('inf')
        try:
            region_entry = min(new_nodes, key=lambda n: dist_or_inf(n))
        except ValueError:
            # no new nodes
            continue

        # Go current -> entry
        try:
            to_region_path = nx.shortest_path(router.G, current, region_entry, weight='length')
            to_region_cost = nx.shortest_path_length(router.G, current, region_entry, weight='length')
        except Exception:
            to_region_path = [current, region_entry]
            to_region_cost = 0.0

        # Visit all points in region (NN inside region)
        reg_path, reg_cost, reg_visited = router._full_region_path_and_cost(rp, region_entry)

        # Stitch
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
            final_path = [current, e_node]
            final_cost = 0.0
        route.extend(final_path[1:])
        total_m += float(final_cost)

    return route, total_m, visited_deliveries

# --------------------
# VIS (plain names)
# --------------------
def plot_points_by_priority_with_regions(router: DivideConquerRouter,
                                         points: pd.DataFrame,
                                         selected_regions: list,
                                         name: str):
    fig, ax = plt.subplots(figsize=(14, 16), dpi=180)
    router._get_city_boundary().boundary.plot(ax=ax, linewidth=2.0, edgecolor="#666", alpha=0.9, zorder=0)
    router.edges.plot(ax=ax, linewidth=0.35, edgecolor="#B0B0B0", alpha=0.85, zorder=1)

    # points by priority
    cmap = {1: ("#FF3B30", "#8B0000"),  # high
            2: ("#FFA500", "#8B5A00"),  # medium
            3: ("#00C853", "#006400")}  # low
    for p in [1,2,3]:
        sub = points[points['priority'] == p]
        if not sub.empty:
            face, edge = cmap[p]
            ax.scatter(sub['lon'], sub['lat'], s=22, c=face, edgecolors=edge, linewidths=0.5,
                       alpha=0.95, zorder=4, label=f'Priority {p}')

    # draw selected region rectangles
    for item in selected_regions:
        b = item['region']['bounds']
        rect = plt.Rectangle((b['min_lon'], b['min_lat']),
                             b['max_lon'] - b['min_lon'],
                             b['max_lat'] - b['min_lat'],
                             fill=False, lw=2.0, ec='#1F77B4', alpha=0.95, zorder=6)
        ax.add_patch(rect)
        ax.scatter(item['region']['centroid_lon'], item['region']['centroid_lat'],
                   c='#1F77B4', s=36, marker='x', zorder=7)

    # depots
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
# MAIN RUNNER
# --------------------
def run_experiment3_priority_focus():
    router = DivideConquerRouter(GRAPH_FILE)
    summary_rows = []

    for name, csv in DATASETS.items():
        print(f"\n=== {name} ===")
        points = router.load_and_preprocess_points(csv)

        # 1) Select regions: same corridor idea, but prioritize P1/P2 per slab
        selected, spine_nodes = select_regions_priority_first(
            router, points,
            start='S', end='N',
            max_points_per_region=MAX_POINTS_PER_REGION,
            corridor_width_km=CORRIDOR_WIDTH_KM,
            num_slabs=NUM_SLABS,
            per_slab_top_k=PER_SLAB_TOP_K,
            min_region_size=MIN_REGION_SIZE
        )
        print(f"  Selected regions: {len(selected)} (PER_SLAB_TOP_K={PER_SLAB_TOP_K})")

        # 2) Visual: regions + priority-colored points
        plot_points_by_priority_with_regions(router, points, selected, name)

        # 3) Build route by visiting selected regions in S→N order (skip mostly-covered)
        t0 = time.time()
        route, dist_m, visited = build_priority_route(
            router, selected, spine_nodes, points,
            start='S', end='N', min_new_ratio=MIN_NEW_RATIO
        )
        dt = time.time() - t0

        # 4) Visual: standard route with visited vs unvisited (same as Exp 1/2)
        router.visualize_route(
            route_nodes=route,
            delivery_points=points,
            visited_deliveries=visited,
            title=f"Route — {name}",
            start_depot='S', end_depot='N'
        )

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
            'runtime_s': dt
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv("experiment3_priority_summary.csv", index=False)
        plot_total_vs_covered(df)
        print("\nSaved: experiment3_priority_summary.csv and Total_vs_Covered.png")
        return df
    return pd.DataFrame()

if __name__ == "__main__":
    _ = run_experiment3_priority_focus()
