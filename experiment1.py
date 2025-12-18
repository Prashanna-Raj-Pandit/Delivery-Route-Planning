# experiment1.py - UPDATED VERSION
# Uses clean single-path routing that follows S→N shortest path

import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from divideconquer import DivideConquerRouter


def compute_baseline_sn_km(router: DivideConquerRouter, start='S', end='N') -> float:
    s = router.depot_nodes[start]
    e = router.depot_nodes[end]
    d_m = nx.shortest_path_length(router.G, s, e, weight='length')
    return d_m / 1000.0


def dynamic_budget_km(router: DivideConquerRouter, n_points: int,
                      start='S', end='N',
                      per_point_km: float = 0.30,
                      max_margin_km: float = 120.0,
                      reserve_pct: float = 0.15) -> float:
    base = compute_baseline_sn_km(router, start, end)
    margin = min(max_margin_km, per_point_km * n_points)
    reserve = reserve_pct * base
    return base + margin + reserve


def plot_scalability_panels(df: pd.DataFrame, out_png="Scalability_Analysis.png"):
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    ax = ax.ravel()

    ax[0].plot(df['size'], df['computation_time'], 'o-', linewidth=2, color='#2E86AB')
    ax[0].set_title('Computation Time vs n', fontweight='bold', fontsize=14)
    ax[0].set_xlabel('Number of Delivery Points (n)', fontsize=11)
    ax[0].set_ylabel('Time (seconds)', fontsize=11)
    ax[0].grid(True, alpha=.3)

    ax[1].plot(df['size'], df['regions_count'], 's-', linewidth=2, color='#A23B72')
    ax[1].set_title('Number of Regions vs n', fontweight='bold', fontsize=14)
    ax[1].set_xlabel('Number of Delivery Points (n)', fontsize=11)
    ax[1].set_ylabel('Regions Used', fontsize=11)
    ax[1].grid(True, alpha=.3)

    ax[2].plot(df['size'], df['coverage_percentage'], '^-', linewidth=2, color='#F18F01')
    ax[2].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% target')
    ax[2].set_title('Coverage (%) vs n', fontweight='bold', fontsize=14)
    ax[2].set_xlabel('Number of Delivery Points (n)', fontsize=11)
    ax[2].set_ylabel('Coverage (%)', fontsize=11)
    ax[2].set_ylim(0, 105)
    ax[2].grid(True, alpha=.3)
    ax[2].legend()

    ax[3].plot(df['size'], df['efficiency_deliv_per_km'], 'd-', linewidth=2, color='#6A994E')
    ax[3].set_title('Efficiency (deliveries/km) vs n', fontweight='bold', fontsize=14)
    ax[3].set_xlabel('Number of Delivery Points (n)', fontsize=11)
    ax[3].set_ylabel('Deliveries per km', fontsize=11)
    ax[3].grid(True, alpha=.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")


def run_clean_path_scalability_experiment():
    print("="*64)
    print("CLEAN SINGLE-PATH ROUTING — SCALABILITY (South → North)")
    print("="*64)

    router = DivideConquerRouter("chicago_street_network.graphml")
    
    # ADD THE NEW METHOD TO THE ROUTER CLASS
    router.solve_clean_single_path = solve_clean_single_path.__get__(router)
    
    sizes = [100, 200, 300, 500, 1000]
    results = []

    for size in sizes:
        print(f"\n{'='*14} n = {size} {'='*14}")

        csv_path = f"delivery_points_{size}.csv"
        try:
            points = router.load_and_preprocess_points(csv_path)
        except FileNotFoundError:
            print(f"  {csv_path} missing; skipping n={size}.")
            continue

        # Quad-tree decomposition visualization
        regions = router.quad_tree_decomposition(points, max_points_per_region=15)
        router.visualize_quad_tree(points=points, regions=regions, 
                                 title=f"Quad_Tree_Decomposition_Size_{size}")

        # Budget calculation (less restrictive)
        baseline_km = compute_baseline_sn_km(router, 'S', 'N')
        budget_km = dynamic_budget_km(
            router, n_points=size, start='S', end='N',
            per_point_km=0.40,  # Increased margin
            max_margin_km=150.0,  # More generous
            reserve_pct=0.10   # Less reserve
        )
        print(f"  Baseline S->N: {baseline_km:.1f} km")
        print(f"  Budget: {budget_km:.1f} km")

        tracemalloc.start()
        t0 = time.time()

        # USE CLEAN SINGLE-PATH ROUTING
        route, dist_m, region_solutions, visited_deliveries = router.solve_clean_single_path(
            points, 'S', 'N',
            max_points_per_region=15,
            corridor_width_km=3.0  # Narrow corridor for cleaner path
        )

        comp_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        covered = len(visited_deliveries)
        actual_dist_m = router.calculate_route_distance(route)

        results.append({
            'size': size,
            'computation_time': comp_time,
            'total_distance_m': actual_dist_m,
            'distance_km': actual_dist_m / 1000.0,
            'regions_count': len(region_solutions),
            'deliveries_covered': covered,
            'coverage_percentage': (covered/size)*100.0,
            'avg_region_size': (covered/len(region_solutions)) if region_solutions else 0.0,
            'efficiency_deliv_per_km': (covered / (actual_dist_m/1000.0)) if actual_dist_m > 0 else 0.0,
            'memory_peak_mb': peak/1024/1024,
            'route_length': len(route),
            'budget_km': budget_km,
            'budget_utilization_pct': (actual_dist_m/1000.0 / budget_km) * 100.0
        })

        print(f"  Time: {comp_time:.2f}s | Distance: {actual_dist_m/1000.0:.1f} km | "
              f"Covered: {covered}/{size} ({100*covered/size:.1f}%) | Regions: {len(region_solutions)}")
        if actual_dist_m/1000.0 > 0:
            print(f"  Efficiency: {covered/(actual_dist_m/1000.0):.2f} deliveries/km")

        router.visualize_route(
            route_nodes=route,
            delivery_points=points,
            visited_deliveries=visited_deliveries,
            title=f"Route_Size_{size}",
            start_depot='S',
            end_depot='N'
        )

    df = pd.DataFrame(results)
    df.to_csv("scalability_results.csv", index=False)
    print(f"\n{'='*64}")
    print("RESULTS SAVED: scalability_results.csv")
    print("="*64)

    plot_scalability_panels(df, out_png="Scalability_Analysis.png")
    return df


def print_summary_table(df: pd.DataFrame):
    print("\n" + "="*90)
    print("CLEAN PATH ROUTING - SUMMARY")
    print("="*90)
    print(f"{'n':<8} {'Time(s)':<10} {'Dist(km)':<12} {'Regions':<10} {'Covered':<10} {'Coverage%':<12} {'Eff(d/km)':<10}")
    print("-"*90)
    for _, row in df.iterrows():
        eff = row['efficiency_deliv_per_km'] if row['distance_km'] > 0 else 0.0
        print(f"{int(row['size']):<8} "
              f"{row['computation_time']:<10.2f} "
              f"{row['distance_km']:<12.1f} "
              f"{int(row['regions_count']):<10} "
              f"{int(row['deliveries_covered']):<10} "
              f"{row['coverage_percentage']:<12.1f} "
              f"{eff:<10.2f}")
    print("="*90)


# ADD THE NEW ROUTING METHOD HERE
def solve_clean_single_path(self, delivery_points: pd.DataFrame,
                          start_depot: str, end_depot: str,
                          max_points_per_region: int = 15,
                          corridor_width_km: float = 3.0):
    """
    Creates a clean single path from start→end by visiting regions in geographic order
    """
    print("CLEAN PATH: Building route along shortest path...")
    
    # 1. Get the actual shortest path between depots
    s_node = self.depot_nodes[start_depot]
    e_node = self.depot_nodes[end_depot]
    main_path = nx.shortest_path(self.G, s_node, e_node, weight='length')
    main_path_coords = self._coords_for_nodes(main_path)
    
    # 2. Decompose into regions
    regions = self.quad_tree_decomposition(delivery_points, max_points_per_region)
    
    # 3. Find regions near the main path
    path_regions = []
    for region in regions:
        # Calculate minimum distance from region centroid to any segment of main path
        min_dist = float('inf')
        for i in range(len(main_path_coords)-1):
            lon1, lat1 = main_path_coords[i]
            lon2, lat2 = main_path_coords[i+1]
            
            dist = self.point_to_segment_distance(
                region['centroid_lat'], region['centroid_lon'],
                lat1, lon1, lat2, lon2
            )
            min_dist = min(min_dist, dist)
        
        if min_dist <= corridor_width_km:
            # Calculate projection score to order regions along the path
            projection = self._projection_score(region, 
                main_path_coords[0][1], main_path_coords[0][0],  # start
                main_path_coords[-1][1], main_path_coords[-1][0]  # end
            )
            path_regions.append((region, projection, min_dist, len(region['points'])))
    
    # 4. Sort regions by position along the path (projection score)
    path_regions.sort(key=lambda x: x[1])
    print(f"  Found {len(path_regions)} regions along the path")
    
    # 5. Build route by visiting regions in order
    complete_route = [s_node]
    current_node = s_node
    total_m = 0.0
    visited_deliveries = set()
    used_regions = []
    
    for region, projection, distance, point_count in path_regions:
        region_nodes = set(region['points']['nearest_node'])
        
        # Skip if no new deliveries or already mostly covered
        new_points = region_nodes - visited_deliveries
        if len(new_points) < max(1, point_count * 0.3):  # Skip if <30% new points
            continue
            
        try:
            # Route to this region (use nearest point on main path)
            region_entry_node = min(region_nodes, 
                                  key=lambda n: nx.shortest_path_length(self.G, current_node, n, weight='length'))
            
            # Get path to region
            to_region_path = self._shortest_path_nodes(current_node, region_entry_node)
            to_region_cost = nx.shortest_path_length(self.G, current_node, region_entry_node, weight='length')
            
            # Visit all points in region
            region_path, region_cost, region_visited = self._full_region_path_and_cost(
                region['points'], region_entry_node
            )
            
            # Add to complete route
            complete_route.extend(to_region_path[1:])
            complete_route.extend(region_path[1:])
            total_m += to_region_cost + region_cost
            visited_deliveries.update(region_visited)
            current_node = region_path[-1]
            
            used_regions.append(region)
            print(f"  Added region with {len(region_visited)} deliveries")
            
        except Exception as e:
            print(f"  Could not route through region: {e}")
            continue
    
    # 6. Route to final destination
    if current_node != e_node:
        try:
            final_path = self._shortest_path_nodes(current_node, e_node)
            final_cost = nx.shortest_path_length(self.G, current_node, e_node, weight='length')
            complete_route.extend(final_path[1:])
            total_m += final_cost
        except Exception as e:
            print(f"  Warning: Could not route to end depot: {e}")
    
    print(f"CLEAN PATH: distance = {total_m/1000:.2f} km | regions used = {len(used_regions)} | deliveries = {len(visited_deliveries)}")
    return complete_route, total_m, used_regions, visited_deliveries


if __name__ == "__main__":
    df_results = run_clean_path_scalability_experiment()
    print_summary_table(df_results)