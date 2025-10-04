# experiment1_greedy.py
# CLEAN SINGLE-PATH — SCALABILITY (South → North) — GREEDY

import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from greedy import GreedyRouter

def compute_baseline_sn_km(router: GreedyRouter, start='S', end='N') -> float:
    s = router.depot_nodes[start]
    e = router.depot_nodes[end]
    d_m = nx.shortest_path_length(router.G, s, e, weight='length')
    return d_m / 1000.0

def dynamic_budget_km(router: GreedyRouter, n_points: int,
                      start='S', end='N',
                      per_point_km: float = 0.40,
                      max_margin_km: float = 150.0,
                      reserve_pct: float = 0.10) -> float:
    """Same budget formula you used in D&C version."""
    base = compute_baseline_sn_km(router, start, end)
    margin = min(max_margin_km, per_point_km * n_points)
    reserve = reserve_pct * base
    return base + margin + reserve

def run_greedy_scalability_experiment():
    print("="*64)
    print("CLEAN SINGLE-PATH (GREEDY) — SCALABILITY (South → North)")
    print("="*64)

    router = GreedyRouter("chicago_street_network.graphml")

    sizes = [100, 200, 300, 500, 1000]
    results = []

    base_km = compute_baseline_sn_km(router, 'S', 'N')
    print(f"Baseline S->N: {base_km:.1f} km")

    for size in sizes:
        csv_path = f"delivery_points_{size}.csv"
        try:
            points = router.load_and_preprocess_points(csv_path)
        except FileNotFoundError:
            print(f"  {csv_path} missing; skipping n={size}.")
            continue

        budget_km = dynamic_budget_km(router, n_points=size, start='S', end='N',
                                      per_point_km=0.40, max_margin_km=150.0, reserve_pct=0.10)
        print(f"\n============== n = {size} ==============")
        print(f"  Budget: {budget_km:.1f} km")

        tracemalloc.start()
        t0 = time.time()

        route, dist_m, _region_solutions, visited = router.solve_greedy_single_path(
            points=points, start_depot='S', end_depot='N',
            budget_km=budget_km, reserve_pct=0.10
        )

        comp_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        actual_dist_m = router.calculate_route_distance(route)
        covered = len(visited)

        results.append({
            'size': size,
            'computation_time_s': comp_time,
            'peak_mem_bytes': int(peak),
            'total_distance_m': actual_dist_m,
            'distance_km': actual_dist_m/1000.0,
            'deliveries_covered': covered,
            'coverage_pct': (covered/size)*100.0
        })

        print(f"  Time: {comp_time:.2f}s | Distance: {actual_dist_m/1000.0:.1f} km | "
              f"Covered: {covered}/{size} ({100*covered/size:.1f}%)")
        if actual_dist_m > 0:
            print(f"  Efficiency: {covered/(actual_dist_m/1000.0):.2f} deliveries/km")

        router.visualize_route(
            route_nodes=route,
            delivery_points=points,
            visited_deliveries=visited,
            title=f"Greedy_Route_Size_{size}",
            start_depot='S',
            end_depot='N'
        )
        print(f"  Saved: Greedy_Route_Size_{size}.png")

    df = pd.DataFrame(results)
    df.to_csv("scalability_results_greedy.csv", index=False)
    print("\nRESULTS SAVED: scalability_results_greedy.csv")

    # Plot size vs distance and coverage
    plt.figure(figsize=(9,6))
    plt.plot(df['size'], df['distance_km'], marker='o')
    plt.xlabel("Number of delivery points (n)")
    plt.ylabel("Route distance (km)")
    plt.title("Greedy Single-Path Scalability: Distance vs n")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("Greedy_Scalability_Distance_vs_n.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(df['size'], df['coverage_pct'], marker='o')
    plt.xlabel("Number of delivery points (n)")
    plt.ylabel("Coverage (%)")
    plt.title("Greedy Single-Path Scalability: Coverage vs n")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("Greedy_Scalability_Coverage_vs_n.png", dpi=220)
    plt.close()
    print("Saved: Greedy_Scalability_Distance_vs_n.png, Greedy_Scalability_Coverage_vs_n.png")


if __name__ == "__main__":
    run_greedy_scalability_experiment()
