from divideconquer import DivideConquerRouter
from experiment1 import run_experiment_1, create_scalability_analysis_plots
from experiment2 import run_experiment_2, create_multi_depot_plots
from experiment3 import run_experiment_3, create_priority_analysis_plots
import pandas as pd

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
        if not exp1_results.empty:
            print("Generating scalability analysis plots...")
            create_scalability_analysis_plots(exp1_results)
            print("✓ Experiment 1 completed! Results saved to experiment_1_results.csv")
        else:
            print("Experiment 1 failed to produce results.")
        
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 2: MULTI-DEPOT ANALYSIS")
        print("="*50)
        exp2_results = run_experiment_2(router)
        if not exp2_results.empty:
            print("Generating multi-depot analysis plots...")
            create_multi_depot_plots(router, exp2_results)
            print("✓ Experiment 2 completed! Results saved to experiment_2_results.csv")
        else:
            print("Experiment 2 failed to produce results.")
        
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 3: PRIORITY-AWARE DELIVERY")
        print("="*50)
        exp3_results = run_experiment_3(router)
        if not exp3_results.empty:
            print("Generating priority analysis plots...")
            create_priority_analysis_plots(router, exp3_results)
            print("✓ Experiment 3 completed! Results saved to experiment_3_results.csv")
        else:
            print("Experiment 3 failed to produce results.")
        
        print("\n" + "="*50)
        print("GENERATING ADDITIONAL VISUALIZATIONS")
        print("="*50)
        
        print("Generating sample route visualization...")
        sample_points = router.load_and_preprocess_points("delivery_points_100.csv")
        route, distance, points_covered = router.find_optimal_path_with_points(
            sample_points, router.depot_nodes['S'], router.depot_nodes['N'])
        router.visualize_route(route, sample_points, "Sample_Delivery_Route_S_to_N")
        print("✓ Sample route visualization generated!")
        
        print("Generating quad-tree visualization...")
        regions = router.quad_tree_decomposition(sample_points, max_points_per_region=25)
        router.visualize_quad_tree(sample_points, regions, "Enhanced_Quad-Tree_Decomposition")
        print("✓ Quad-tree visualization generated!")
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGENERATED FILES:")
        print("1. experiment_1_results.csv")
        print("2. experiment_2_results.csv")
        print("3. experiment_3_results.csv")
        print("4. Route_S_to_N_size_100.png, Route_S_to_N_size_200.png, ..., Route_S_to_N_size_1000.png")
        print("5. Route_C_to_E.png, Route_S_to_E.png, etc. (for each depot pair)")
        print("6. Priority_Distribution_1.png, Priority_Distribution_2.png, etc.")
        print("7. Comprehensive_Scalability_Analysis.png")
        print("8. Multi_Depot_Performance_Heatmap.png")
        print("9. Priority_Analysis_Radar.png")
        print("10. Regional_Distribution_Analysis.png")
        print("11. Sample_Delivery_Route_S_to_N.png")
        print("12. Enhanced_Quad-Tree_Decomposition.png")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")