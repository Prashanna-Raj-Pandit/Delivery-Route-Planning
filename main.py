from divideconquer import DivideConquerRouter
from experiment1 import run_experiment_1
from experiment2 import run_experiment_2
from experiment3 import run_experiment_3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_comprehensive_analysis_plots(router, exp1_results, exp2_results, exp3_results):
    """
    Create comprehensive analysis plots with multiple visualization types
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
    print("Starting Enhanced Divide and Conquer Route Optimization...")
    print("="*60)
    
    try:
        router = DivideConquerRouter(
            graph_file="chicago_street_network.graphml",
            distance_weight=0.3,
            delivery_weight=0.7
        )
        
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 1: SCALABILITY TEST")
        print("="*50)
        exp1_results = run_experiment_1(router)
        exp1_results.to_csv("experiment_1_results.csv", index=False)
        print("✓ Experiment 1 completed! Results saved to experiment_1_results.csv")
        
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 2: MULTI-DEPOT ANALYSIS")
        print("="*50)
        exp2_results = run_experiment_2(router)
        exp2_results.to_csv("experiment_2_results.csv", index=False)
        print("✓ Experiment 2 completed! Results saved to experiment_2_results.csv")
        
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 3: PRIORITY-AWARE DELIVERY")
        print("="*50)
        exp3_results = run_experiment_3(router)
        exp3_results.to_csv("experiment_3_results.csv", index=False)
        print("✓ Experiment 3 completed! Results saved to experiment_3_results.csv")
        
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        
        sample_points = router.load_and_preprocess_points("delivery_points_100.csv")
        regions = router.quad_tree_decomposition(sample_points, max_points_per_region=25)
        
        print("Generating Quad-Tree visualization...")
        router.visualize_quad_tree(sample_points, regions, "Enhanced_Quad-Tree_Decomposition")
        
        print("Generating sample route visualization...")
        route, distance, points_covered = router.find_optimal_path_with_points(
            sample_points, router.depot_nodes['S'], router.depot_nodes['N'])
        router.visualize_route(route, sample_points, "Sample_Delivery_Route_S_to_N")
        
        print("Generating comprehensive analysis plots...")
        create_comprehensive_analysis_plots(router, exp1_results, exp2_results, exp3_results)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGENERATED FILES:")
        print("1. experiment_1_results.csv")
        print("2. experiment_2_results.csv")
        print("3. experiment_3_results.csv")
        print("4. Route_S_to_N_size_100.png, Route_S_to_N_size_200.png, ..., Route_S_to_N_size_1000.png")
        print("5. Enhanced_Quad-Tree_Decomposition.png")
        print("6. Sample_Delivery_Route_S_to_N.png")
        print("7. Route_C_to_E.png, Route_S_to_E.png, etc. (for each depot pair)")
        print("8. Priority_Distribution_1.png, Priority_Distribution_2.png, etc.")
        print("9. Comprehensive_Scalability_Analysis.png")
        print("10. Multi_Depot_Performance_Heatmap.png")
        print("11. Priority_Analysis_Radar.png")
        print("12. Regional_Distribution_Analysis.png")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")