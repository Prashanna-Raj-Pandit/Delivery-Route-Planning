from divideconquer import DivideConquerRouter
import pandas as pd
import psutil
import os
import time
import matplotlib.pyplot as plt
import numpy as np

def run_experiment_1(router, sizes=[100, 200, 300, 500, 1000]):
    """
    Run scalability experiment with weighted South-to-North route
    """
    results = []
    data_files = [f"delivery_points_{size}.csv" for size in sizes]
    
    for size, data_file in zip(sizes, data_files):
        print(f"Running experiment for size {size}...")
        try:
            delivery_points = router.load_and_preprocess_points(data_file)
        except FileNotFoundError:
            print(f"Error: {data_file} not found. Skipping size {size}.")
            continue
        
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
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("experiment_1_results.csv", index=False)
        print("Results saved to experiment_1_results.csv")
    else:
        print("No results to save. Check for errors in data loading or processing.")
    
    return pd.DataFrame(results)

def create_scalability_analysis_plots(exp1_results):
    """
    Create scalability analysis plots for Experiment 1
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(exp1_results['size'], exp1_results['computation_time'], 'o-', linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Number of Delivery Points')
    axes[0, 0].set_ylabel('Computation Time (s)')
    axes[0, 0].set_title('Computation Time vs Problem Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(exp1_results['size'], exp1_results['memory_used'], 's-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Number of Delivery Points')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Usage vs Problem Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(exp1_results['size'], exp1_results['distance'], '^-', linewidth=2, color='green')
    axes[0, 2].set_xlabel('Number of Delivery Points')
    axes[0, 2].set_ylabel('Total Distance (m)')
    axes[0, 2].set_title('Total Distance vs Problem Size')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(exp1_results['size'], exp1_results['delivery_points_covered'], 'v-', linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Number of Delivery Points')
    axes[1, 0].set_ylabel('Points Covered')
    axes[1, 0].set_title('Delivery Points Covered vs Problem Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    efficiency = exp1_results['delivery_points_covered'] / exp1_results['size']
    axes[1, 1].plot(exp1_results['size'], efficiency, '*-', linewidth=2, color='brown')
    axes[1, 1].set_xlabel('Number of Delivery Points')
    axes[1, 1].set_ylabel('Efficiency Ratio')
    axes[1, 1].set_title('Coverage Efficiency vs Problem Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')  # Remove unused subplot
    
    plt.tight_layout()
    plt.savefig('Comprehensive_Scalability_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

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
        if not exp1_results.empty:
            print("✓ Experiment 1 completed! Results saved to experiment_1_results.csv")
            print("Generating scalability analysis plots...")
            create_scalability_analysis_plots(exp1_results)
            print("✓ Comprehensive scalability plots generated!")
        else:
            print("Experiment 1 failed to produce results.")
        
        # Generate sample route visualization
        print("Generating sample route visualization...")
        sample_points = router.load_and_preprocess_points("delivery_points_100.csv")
        route, distance, points_covered = router.find_optimal_path_with_points(
            sample_points, router.depot_nodes['S'], router.depot_nodes['N'])
        router.visualize_route(route, sample_points, "Sample_Delivery_Route_S_to_N")
        print("✓ Sample route visualization generated!")
        
        # Generate quad-tree visualization
        print("Generating quad-tree visualization...")
        regions = router.quad_tree_decomposition(sample_points, max_points_per_region=25)
        router.visualize_quad_tree(sample_points, regions, "Enhanced_Quad-Tree_Decomposition")
        print("✓ Quad-tree visualization generated!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")