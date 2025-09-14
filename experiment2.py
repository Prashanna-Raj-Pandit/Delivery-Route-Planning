from divideconquer import DivideConquerRouter
import pandas as pd
import psutil
import os
import time
import matplotlib.pyplot as plt
import numpy as np

def run_experiment_2(router):
    """
    Run multi-depot experiment
    """
    results = []
    depot_pairs = [('C', 'E'), ('S', 'E'), ('W', 'S'), ('E', 'W'), ('N', 'S')]
    
    try:
        delivery_points = router.load_and_preprocess_points("delivery_points_1000.csv")
    except FileNotFoundError:
        print("Error: delivery_points_1000.csv not found.")
        return pd.DataFrame()
    
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
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("experiment_2_results.csv", index=False)
        print("Results saved to experiment_2_results.csv")
    else:
        print("No results to save. Check for errors in data loading or processing.")
    
    return pd.DataFrame(results)

def create_multi_depot_plots(router, exp2_results):
    """
    Create multi-depot performance and regional distribution plots for Experiment 2
    """
    # Multi-depot performance heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    depot_pairs = [pair.split('-') for pair in exp2_results['depot_pair']]
    performance_data = exp2_results[['distance', 'computation_time', 'memory_used']].values
    
    im = ax.imshow(performance_data, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Distance (m)', 'Time (s)', 'Memory (MB)'])
    ax.set_yticks(range(len(depot_pairs)))
    ax.set_yticklabels(exp2_results['depot_pair'])
    
    for i in range(len(depot_pairs)):
        for j in range(3):
            text = ax.text(j, i, f'{performance_data[i, j]:.1f}',
                           ha="center", va="center", color="w", fontweight='bold')
    
    ax.set_title('Multi-Depot Performance Comparison', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('Multi_Depot_Performance_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Regional distribution analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    delivery_points = router.load_and_preprocess_points("delivery_points_1000.csv")
    delivery_points['region'] = delivery_points.apply(
        lambda x: router.get_region_for_point(x['lat'], x['lon']), axis=1)
    
    region_counts = delivery_points['region'].value_counts()
    colors = [router.depots[r]['color'] for r in region_counts.index]
    
    bars = ax.bar(region_counts.index, region_counts.values, color=colors, alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Delivery Points Distribution by Region', fontsize=16, fontweight='bold')
    ax.set_xlabel('Region')
    ax.set_ylabel('Number of Delivery Points')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Regional_Distribution_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

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
        if not exp2_results.empty:
            print("✓ Experiment 2 completed! Results saved to experiment_2_results.csv")
            print("Generating multi-depot analysis plots...")
            create_multi_depot_plots(router, exp2_results)
            print("✓ Multi-depot and regional distribution plots generated!")
        
        # Generate quad-tree visualization
        print("Generating quad-tree visualization...")
        sample_points = router.load_and_preprocess_points("delivery_points_100.csv")
        regions = router.quad_tree_decomposition(sample_points, max_points_per_region=25)
        router.visualize_quad_tree(sample_points, regions, "Enhanced_Quad-Tree_Decomposition")
        print("✓ Quad-tree visualization generated!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")