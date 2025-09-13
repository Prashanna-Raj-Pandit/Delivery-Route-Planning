from divideconquer import DivideConquerRouter
import pandas as pd
import psutil
import os
import time

def run_experiment_2(router):
    """
    Run multi-depot experiment
    """
    results = []
    depot_pairs = [('C', 'E'), ('S', 'E'), ('W', 'S'), ('E', 'W'), ('N', 'S')]
    
    delivery_points = router.load_and_preprocess_points("delivery_points_1000.csv")
    for start_depot, end_depot in depot_pairs:
        print(f"Running experiment for {start_depot}-{end_depot}...")
        regions = router.quad_tree_decomposition(delivery_points, max_points_per_region=50)
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        route, distance, region_data = router.solve_region_routes(regions, start_depot, end_depot)
        computation_time = time.time() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        visited_nodes = len(set(route))
        
        delivery_nodes_covered = len([node for node in route if node in
            set(delivery_points['nearest_node'])])
        
        results.append({
            'depot_pair': f"{start_depot}-{end_depot}",
            'computation_time': computation_time,
            'memory_used': memory_used,
            'distance': distance,
            'visited_nodes': visited_nodes,
            'delivery_points_covered': delivery_nodes_covered
        })
        
        router.visualize_route_with_regions(route, delivery_points, region_data,
                                         f"Route_{start_depot}_to_{end_depot}")
        
        print(f"{start_depot}-{end_depot}: {computation_time:.2f}s, {distance:.2f}m")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting Experiment 2: Multi-Depot Analysis")
    print("="*50)
    
    try:
        router = DivideConquerRouter(
            graph_file="chicago_street_network.graphml",
            distance_weight=0.3,
            delivery_weight=0.7
        )
        
        exp2_results = run_experiment_2(router)
        exp2_results.to_csv("experiment_2_results.csv", index=False)
        print("✓ Experiment 2 completed! Results saved to experiment_2_results.csv")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")