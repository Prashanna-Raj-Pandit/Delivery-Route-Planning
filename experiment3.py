from divideconquer import DivideConquerRouter
import pandas as pd
import random
import time

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
    
    delivery_points = router.load_and_preprocess_points("delivery_points_1000.csv")
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
    
    return pd.DataFrame(results)

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
        exp3_results.to_csv("experiment_3_results.csv", index=False)
        print("✓ Experiment 3 completed! Results saved to experiment_3_results.csv")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")