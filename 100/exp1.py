import pandas as pd
from divideconquer import DivideConquerRouter

def run_experiment1_100(graph_file="chicago_street_network.graphml"):
    router = DivideConquerRouter(graph_file)

    # Only the three 100-point datasets
    datasets_100 = [
        "Folyd-Warshall.csv",
        "Nearest Neighbor.csv",
        "Quad-Tree Decomposition.csv"
    ]

    results = []

    for csv_path in datasets_100:
        print(f"\n{'='*14} Running dataset: {csv_path} {'='*14}")
        try:
            points = router.load_and_preprocess_points(csv_path)
        except FileNotFoundError:
            print(f"  {csv_path} missing; skipping.")
            continue

        # Quad-tree decomposition visualization
        regions = router.quad_tree_decomposition(points, max_points_per_region=15)
        router.visualize_quad_tree_decomposition(
            regions,
            points,
            title=f"100_{csv_path.replace('.csv','')}"
        )

        # Clean route visualization
        route, dist, used_regions, visited = router.solve_clean_single_path(
            points, start_depot="S", end_depot="N", max_points_per_region=15
        )
        router.visualize_route(
            route,
            points,
            visited,
            title=f"100_{csv_path.replace('.csv','')}",
            start_depot="S",
            end_depot="N"
        )

        # Store results for CSV
        results.append({
            "dataset": csv_path,
            "regions": len(regions),
            "deliveries": len(points),
            "visited": len(visited),
            "distance_km": dist / 1000.0
        })

    # Save results into a CSV
    pd.DataFrame(results).to_csv("100_results.csv", index=False)
    print("✅ Results saved to experiment1_100_results.csv")


if __name__ == "__main__":
    run_experiment1_100()
