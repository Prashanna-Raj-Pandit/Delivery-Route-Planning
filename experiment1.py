from divideconquer import DivideConquerRouter
import pandas as pd
import psutil
import os
import time

def run_experiment_1(router, sizes=[100, 200, 300, 500, 1000]):
    """
    Run scalability experiment with weighted South-to-North route
    """
    results = []
    data_files = [f"delivery_points_{size}.csv" for size in sizes]
    
    for size, data_file in zip(sizes, data_files):
        print(f"Running experiment for size {size}...")
        delivery_points = router.load_and_preprocess_points(data_file)
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        route, distance, points_covered = router.find_optimal_path_with_points(
            delivery_points, router.depot_nodes['S'], router.depot_nodes['N'])
        computation_time = time.time() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        visited_nodes = len(set(route))
        
        # Generate map visualization
        router.visualize_route(route, delivery_points, f"Route_S_to_N_size_{size}")
        
        results.append({
            'size': size,
            'computation_time': computation_time,
            'memory_used': memory_used,
            'distance': distance,
            'visited_nodes': visited_nodes,
            'delivery_points_covered': points_covered,
            'coverage_percentage': points_covered / size * 100
        })
        
        print(f"Size {size}: {computation_time:.2f}s, {distance:.2f}m, {memory_used:.2f}MB, "
              f"{points_covered}/{size} points covered ({points_covered/size*100:.1f}%)")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting Experiment 1: Scalability Test")
    print("="*50)
    
    try:
        router = DivideConquerRouter(
            graph_file="chicago_street_network.graphml",
            distance_weight=0.3,
            delivery_weight=0.7
        )
        
        exp1_results = run_experiment_1(router)
        exp1_results.to_csv("experiment_1_results.csv", index=False)
        print("✓ Experiment 1 completed! Results saved to experiment_1_results.csv")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")