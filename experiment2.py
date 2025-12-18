# experiment2.py (compatible with your current divideconquer.py)
import os, time, gc
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

from divideconquer import DivideConquerRouter

# --- small helper: haversine distance in km ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def run_experiment_2(router: DivideConquerRouter):
    """
    Multi-depot experiment: one clean single-path route per start→end pair.
    Uses the corridor+quad-tree method implemented in DivideConquerRouter.solve_clean_single_path.
    No budget logic here (consistent with your latest preference).
    """
    results = []
    # Try a few interesting pairs covering different directions
    pairs = [('S','E'), ('C','E'), ('W','S'), ('E','W'), ('N','S')]
    names = {k: v['name'] for k, v in router.depots.items()}

    # Load one common dataset
    points = router.load_and_preprocess_points("delivery_points_1000.csv")

    for s, e in pairs:
        print(f"\nPair: {names[s]} → {names[e]}")
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss / (1024*1024)

        # Wrap the call for memory_profiler peak tracking
        def _runner():
            return router.solve_clean_single_path(
                delivery_points=points,
                start_depot=s,
                end_depot=e,
                max_points_per_region=20,
                corridor_width_km=3.0
            )

        t0 = time.time()
        # memory_usage with (func, args, kwargs) returns peak MB
        mem_peak = memory_usage((_runner, (), {}), interval=0.1, max_usage=True)
        route, dist_m, used_regions, visited_deliveries = _runner()
        dt = time.time() - t0

        # Compute actual polyline distance (your helper recomputes exactly)
        actual_dist_m = router.calculate_route_distance(route)
        covered = len(visited_deliveries)

        mem_used = max(mem_peak - mem_before, 0.0)
        results.append({
            'depot_pair': f"{names[s]} to {names[e]}",
            'computation_time_s': dt,
            'memory_used_mb': mem_used,
            'distance_km': actual_dist_m/1000.0,
            'visited_nodes': len(set(route)),
            'delivery_points_covered': covered,
            'coverage_percentage': covered/len(points)*100.0
        })

        # NOTE: visualize_route signature is:
        # visualize_route(route_nodes, delivery_points, visited_deliveries, title, start_depot=None, end_depot=None)
        router.visualize_route(
            route_nodes=route,
            delivery_points=points,
            visited_deliveries=visited_deliveries,
            title=f"Route_{names[s]}_to_{names[e]}",
            start_depot=s,
            end_depot=e
        )

        print(f"  dist={actual_dist_m/1000:.1f} km  "
              f"covered={covered}/{len(points)} "
              f"time={dt:.2f}s  mem≈{mem_used:.1f}MB")

    df = pd.DataFrame(results)
    df.to_csv("experiment_2_results.csv", index=False)
    return df

def create_multi_depot_plots(results: pd.DataFrame):
    # heatmap-style table
    fig, ax = plt.subplots(figsize=(12, 7))
    data = results[['distance_km','computation_time_s','memory_used_mb','delivery_points_covered']].values
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(['Distance (km)','Time (s)','Memory (MB)','Covered'], fontsize=12)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(results['depot_pair'], fontsize=12)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.1f}" if j<3 else f"{int(data[i,j])}",
                    ha='center', va='center', color='w', fontweight='bold')
    ax.set_title("Multi-Depot Performance Comparison", fontweight='bold')
    plt.colorbar(im, ax=ax, label='Value'); plt.tight_layout()
    plt.savefig("Multi_Depot_Performance_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # bar for covered points
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(results['depot_pair'], results['delivery_points_covered'], color='skyblue')
    for rect in ax.patches:
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+5, f"{int(rect.get_height())}",
                ha='center', va='bottom', fontweight='bold')
    ax.set_title("Delivery Points Covered by Depot Pair", fontweight='bold')
    ax.set_ylabel("Points"); ax.set_xlabel("Depot Pair")
    plt.xticks(rotation=30, ha='right'); ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig("Delivery_Points_Covered_by_Depot_Pair.png", dpi=300)
    plt.close()

def assign_nearest_depot_region(points: pd.DataFrame, depots: dict) -> pd.Series:
    """
    Assign each point to its nearest depot by haversine distance.
    Returns a Series of depot keys (e.g., 'S','N','E','W','C').
    """
    depot_keys = list(depots.keys())
    depot_lats = np.array([depots[k]['lat'] for k in depot_keys])
    depot_lons = np.array([depots[k]['lon'] for k in depot_keys])

    # vectorized nearest depot
    lat = points['lat'].values
    lon = points['lon'].values
    # compute distances to all depots: shape (num_points, num_depots)
    dists = np.stack([haversine_km(lat, lon, depot_lats[j], depot_lons[j]) for j in range(len(depot_keys))], axis=1)
    nearest_idx = np.argmin(dists, axis=1)
    return pd.Series([depot_keys[i] for i in nearest_idx], index=points.index)

if __name__ == "__main__":
    print("Experiment 2 — Multi-Depot (Clean Path, No Budget)")
    router = DivideConquerRouter("chicago_street_network.graphml", distance_weight=0.3, delivery_weight=0.7)

    res = run_experiment_2(router)
    if not res.empty:
        create_multi_depot_plots(res)

        # region distribution (nearest depot by distance; replaces missing get_region_for_point)
        pts = router.load_and_preprocess_points("delivery_points_1000.csv")
        pts['region'] = assign_nearest_depot_region(pts, router.depots)

        counts = pts['region'].value_counts()
        fig, ax = plt.subplots(figsize=(10,6))
        labels = [router.depots[k]['name'] for k in counts.index]
        colors = [router.depots[k]['color'] for k in counts.index]
        bars = ax.bar(labels, counts.values, color=colors, alpha=.8)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+5, f"{int(b.get_height())}",
                    ha='center', va='bottom', fontweight='bold')
        ax.set_title("Delivery Points by Nearest Depot", fontweight='bold')
        ax.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig("Regional_Distribution_Analysis.png", dpi=300)
        plt.close()

        print("✓ Saved experiment_2_results.csv and plots.")
