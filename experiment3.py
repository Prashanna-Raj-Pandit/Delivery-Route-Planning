from divideconquer import DivideConquerRouter
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import numpy as np

def run_experiment_3(router):
    """
    Run priority-aware delivery experiment
    """
    results = []
    priority_distributions = [
        {'1': 0.33, '2': 0.34, '3': 0.33},
        {'1': 0.6, '2': 0.3, '3': 0.1},
        {'1': 0.1, '2': 0.3, '3': 0.6}
    ]
    
    try:
        delivery_points = router.load_and_preprocess_points("delivery_points_1000.csv")
    except FileNotFoundError:
        print("Error: delivery_points_1000.csv not found.")
        return pd.DataFrame()
    
    for i, dist in enumerate(priority_distributions):
        print(f"Running experiment for distribution {i+1}...")
        n = len(delivery_points)
        n1 = int(n * dist['1'])
        n2 = int(n * dist['2'])
        n3 = n - n1 - n2
        
        priorities = [1] * n1 + [2] * n2 + [3] * n3
        random.shuffle(priorities)
        
        points = delivery_points.copy()
        points['priority'] = priorities[:n]
        
        regions = router.quad_tree_decomposition(points, max_points_per_region=50)
        route, distance, region_data = router.solve_region_routes(regions, 'C')
        
        high_priority = sum(1 for p in points['priority'] if p == 1)
        med_priority = sum(1 for p in points['priority'] if p == 2)
        low_priority = sum(1 for p in points['priority'] if p == 3)
        
        results.append({
            'distribution': f"Dist_{i+1}",
            'distance': distance,
            'high_priority_count': high_priority,
            'med_priority_count': med_priority,
            'low_priority_count': low_priority,
            'high_percentage': high_priority/n*100,
            'med_percentage': med_priority/n*100,
            'low_percentage': low_priority/n*100
        })
        
        router.visualize_priority_distribution(points, f"Priority_Distribution_{i+1}")
        
        print(f"Distribution {i+1}: {distance:.2f}m")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("experiment_3_results.csv", index=False)
        print("Results saved to experiment_3_results.csv")
    else:
        print("No results to save. Check for errors in data loading or processing.")
    
    return pd.DataFrame(results)

def create_priority_analysis_plots(router, exp3_results):
    """
    Create priority analysis and regional distribution plots for Experiment 3
    """
    # Priority analysis radar plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    categories = ['High Priority', 'Medium Priority', 'Low Priority', 'Distance', 'Efficiency']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for i, row in exp3_results.iterrows():
        values = [
            row['high_percentage'],
            row['med_percentage'],
            row['low_percentage'],
            row['distance'] / 1000,
            100
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Distribution {i+1}')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)
    ax.set_title('Priority Distribution Analysis', size=16, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('Priority_Analysis_Radar.png', dpi=300, bbox_inches='tight')
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
    print("Starting Experiment 3: Priority-Aware Delivery")
    print("="*50)
    
    try:
        router = DivideConquerRouter(
            graph_file="chicago_street_network.graphml",
            distance_weight=0.3,
            delivery_weight=0.7
        )
        
        exp3_results = run_experiment_3(router)
        if not exp3_results.empty:
            print("✓ Experiment 3 completed! Results saved to experiment_3_results.csv")
            print("Generating priority analysis plots...")
            create_priority_analysis_plots(router, exp3_results)
            print("✓ Priority and regional distribution plots generated!")
        
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