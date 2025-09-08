# Steps to run the code:
# 1. Ensure you have the required libraries installed: osmnx, networkx, pandas, matplotlib.
# 2. Place the chicago_street_network.graphml file in the same directory as this script
# 3. Run the dataset/delivery points.py script once to generate the delivery_points.csv file.
# 4. Run this script to perform the experiments and generate results.

import osmnx as ox
import networkx as nx
import pandas as pd
import random
import numpy as np
import time
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

def load_dataset():
    # Load graph
    G = ox.load_graphml("chicago_street_network.graphml")
    G = G.to_directed()
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Node and edge statistics
    nodes, edges = ox.graph_to_gdfs(G)
    print("\nNode statistics:")
    print(nodes[['y', 'x']].describe())
    print("\nEdge statistics:")
    print(edges[['highway', 'oneway', 'length']].describe())
    print("\nRoad types distribution:")
    print(edges['highway'].value_counts())
    total_length_km = edges['length'].sum() / 1000
    print(f"\nTotal road length: {total_length_km:.2f} km")
    
    # Load or generate delivery points
    try:
        df = pd.read_csv("delivery_points.csv")
        print("\nLoaded delivery_points.csv")
        # Check if 'priority' column exists
        if 'priority' not in df.columns:
            print("Warning: 'priority' column missing. Assigning priorities (329 high, 355 medium, 316 low).")
            num_points = len(df)
            if num_points == 1000:
                priorities = [1] * 329 + [2] * 355 + [3] * 316
                random.shuffle(priorities)
            else:
                priorities = random.choices([1, 2, 3], weights=[0.329, 0.355, 0.316], k=num_points)
            df['priority'] = priorities
            df.to_csv("delivery_points.csv", index=False)
    except FileNotFoundError:
        print("\nGenerating new delivery points...")
        delivery_nodes = random.sample(list(G.nodes()), 1000)
        delivery_data = [{
            "node_id": node,
            "lat": G.nodes[node]['y'],
            "lon": G.nodes[node]['x'],
            "priority": random.choice([1, 2, 3])
        } for node in delivery_nodes]
        df = pd.DataFrame(delivery_data)
        df.to_csv("delivery_points.csv", index=False)
    
    print("\nSample Delivery Points:")
    print(df.head())
    print("\nPriority Distribution:")
    print(df['priority'].value_counts())
    
    return G, df

# --- Plotting (Adapted from plot_map.py) ---
def plot_map(G, df):
    nodes, edges = ox.graph_to_gdfs(G)
    fig, ax = plt.subplots(figsize=(12, 12))
    edges.plot(ax=ax, linewidth=0.5, edgecolor='lightgray')
    nodes.plot(ax=ax, markersize=5, color='blue', alpha=0.6, label='Intersections')
    ax.scatter(df['lon'], df['lat'], c='red', s=15, label='Delivery Points', alpha=0.8)
    ax.set_title("Chicago Street Network with Intersections and Delivery Points", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.savefig("chicago_network.png")
    plt.show()

# --- Utility Functions ---
def short_path_dist(G, u, v):
    try:
        return nx.shortest_path_length(G, u, v, weight='length') / 1000  # Convert to km
    except nx.NetworkXNoPath:
        return float('inf')

def get_path_nodes(G, u, v):
    try:
        return nx.shortest_path(G, u, v, weight='length')
    except nx.NetworkXNoPath:
        return []

def calculate_metrics(G, route):
    total_distance = 0
    all_path_nodes = set()
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        dist = short_path_dist(G, u, v)
        total_distance += dist
        path = get_path_nodes(G, u, v)
        all_path_nodes.update(path)
    return total_distance, len(all_path_nodes)

def calculate_priority_adherence(route, priority_dict):
    delivery_order = route[1:-1]  # Exclude depot
    positions = {p: i for i, p in enumerate(delivery_order)}
    good = total = 0
    for p1 in delivery_order:
        if priority_dict[p1] != 1: continue
        for p2 in delivery_order:
            if priority_dict[p2] > 1:
                total += 1
                if positions[p1] < positions[p2]:
                    good += 1
    return (good / total * 100) if total > 0 else 100.0

# --- Algorithms ---
def dijkstra_greedy(G, depot, delivery_points, priority_dict, use_priority=False):
    start_time = time.time()
    factors = {1: 1.0, 2: 1.5, 3: 2.0} if use_priority else {1: 1.0, 2: 1.0, 3: 1.0}
    
    # Run Dijkstra from depot
    distances = nx.single_source_dijkstra_path_length(G, depot, weight='length')
    distances = {k: v / 1000 for k, v in distances.items()}  # Convert to km
    
    # Sort points by adjusted distance
    sorted_points = sorted(delivery_points, key=lambda p: distances.get(p, float('inf')) * factors[priority_dict[p]])
    route = [depot] + sorted_points + [depot]
    
    comp_time = time.time() - start_time
    memory_approx = sys.getsizeof(route) + sys.getsizeof(distances)
    return route, comp_time, memory_approx

def nearest_neighbor_tsp(G, points, depot, priority_dict, use_priority=False):
    factors = {1: 1.0, 2: 1.5, 3: 2.0} if use_priority else {1: 1.0, 2: 1.0, 3: 1.0}
    points = list(points)
    route = [depot]
    current = depot
    while points:
        min_dist = float('inf')
        next_p = None
        for p in points:
            d = short_path_dist(G, current, p) * factors[priority_dict[p]]
            if d < min_dist:
                min_dist = d
                next_p = p
        route.append(next_p)
        points.remove(next_p)
        current = next_p
    route.append(depot)
    return route

def merge_tours(G, left_route, right_route, depot):
    left_path = left_route[1:-1]
    right_path = right_route[1:-1]
    candidates = []
    for first, second in [(left_path, right_path), (right_path, left_path)]:
        for rev_first in [False, True]:
            for rev_second in [False, True]:
                p1 = first[::-1] if rev_first else first
                p2 = second[::-1] if rev_second else second
                if not p1 or not p2:
                    continue
                connect_dist = short_path_dist(G, p1[-1], p2[0])
                full_route = [depot] + p1 + p2 + [depot]
                candidates.append((connect_dist, full_route))
    if not candidates:
        return left_route if left_route else right_route
    return min(candidates, key=lambda x: x[0])[1]

def divide_conquer_tsp(G, depot, delivery_points, priority_dict, use_priority=False, threshold=10):
    start_time = time.time()
    if len(delivery_points) <= threshold:
        route = nearest_neighbor_tsp(G, delivery_points, depot, priority_dict, use_priority)
        comp_time = time.time() - start_time
        memory_approx = sys.getsizeof(route) + sys.getsizeof(delivery_points)
        return route, comp_time, memory_approx
    
    # Split by median longitude
    coords = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in delivery_points}
    sorted_points = sorted(delivery_points, key=lambda p: coords[p][0])
    mid = len(sorted_points) // 2
    left = sorted_points[:mid]
    right = sorted_points[mid:]
    
    left_route, _, _ = divide_conquer_tsp(G, depot, left, priority_dict, use_priority, threshold)
    right_route, _, _ = divide_conquer_tsp(G, depot, right, priority_dict, use_priority, threshold)
    
    route = merge_tours(G, left_route, right_route, depot)
    comp_time = time.time() - start_time
    memory_approx = sys.getsizeof(route) + sys.getsizeof(left) + sys.getsizeof(right)
    return route, comp_time, memory_approx

def floyd_warshall_heuristic(G, depot, delivery_points, priority_dict, use_priority=False):
    start_time = time.time()
    points = [depot] + delivery_points
    n = len(points)
    # Precompute all-pairs shortest paths for delivery points
    dist_matrix = {p1: {p2: short_path_dist(G, p1, p2) for p2 in points if p1 != p2} for p1 in points}
    
    # Nearest neighbor with priority
    factors = {1: 1.0, 2: 1.5, 3: 2.0} if use_priority else {1: 1.0, 2: 1.0, 3: 1.0}
    factors[depot] = 1.0
    remaining = set(delivery_points)
    route = [depot]
    current = depot
    while remaining:
        min_dist = float('inf')
        next_p = None
        for p in remaining:
            d = dist_matrix[current][p] * factors[priority_dict[p]]
            if d < min_dist:
                min_dist = d
                next_p = p
        route.append(next_p)
        remaining.remove(next_p)
        current = next_p
    route.append(depot)
    
    comp_time = time.time() - start_time
    memory_approx = sys.getsizeof(dist_matrix) + sys.getsizeof(route)
    return route, comp_time, memory_approx

# --- Experiments ---
def run_experiments(G, df):
    results = []
    sizes = [100, 200, 300, 500, 1000]
    priority_dist = {1: 329, 2: 355, 3: 316}  # From proposal for 1000 points
    
    for size in sizes:
        # Sample delivery points
        if size == 1000:
            delivery_df = df
            priority_dict = dict(zip(df['node_id'], df['priority']))
        else:
            delivery_df = df.sample(n=size, random_state=42)
            priority_dict = dict(zip(delivery_df['node_id'], delivery_df['priority']))
        delivery_points = list(delivery_df['node_id'])
        depot = random.choice(list(G.nodes()))  # Fixed depot for consistency
        
        # Exp 1 & 2: Baseline/Scaling (no priority)
        for algo_name, algo_func in [
            ('Dijkstra', dijkstra_greedy),
            ('DivideConquer', divide_conquer_tsp),
            ('FloydHeuristic', floyd_warshall_heuristic)
        ]:
            route, comp_time, memory = algo_func(G, depot, delivery_points, priority_dict, use_priority=False)
            total_dist, nodes_visited = calculate_metrics(G, route)
            results.append({
                'Experiment': 'Baseline/Scaling',
                'Size': size,
                'Algorithm': algo_name,
                'Total Distance (km)': total_dist,
                'Comp Time (s)': comp_time,
                'Memory (MB)': memory / (1024**2),
                'Nodes Visited': nodes_visited,
                'Priority Adherence (%)': '-'
            })
        
        # Exp 3: Priority (for sizes 100-500)
        if size <= 500:
            for algo_name, algo_func in [
                ('Dijkstra', dijkstra_greedy),
                ('DivideConquer', divide_conquer_tsp),
                ('FloydHeuristic', floyd_warshall_heuristic)
            ]:
                route, comp_time, memory = algo_func(G, depot, delivery_points, priority_dict, use_priority=True)
                total_dist, nodes_visited = calculate_metrics(G, route)
                adherence = calculate_priority_adherence(route, priority_dict)
                results.append({
                    'Experiment': 'Priority',
                    'Size': size,
                    'Algorithm': algo_name,
                    'Total Distance (km)': total_dist,
                    'Comp Time (s)': comp_time,
                    'Memory (MB)': memory / (1024**2),
                    'Nodes Visited': nodes_visited,
                    'Priority Adherence (%)': adherence
                })
    
    results_df = pd.DataFrame(results)
    print("\nExperiment Results:")
    print(results_df)
    results_df.to_csv("experiment_results.csv", index=False)
    
    # Generate chart for total distance vs size
    baseline_results = results_df[results_df['Experiment'] == 'Baseline/Scaling']
    chart_data = {
        'type': 'line',
        'data': {
            'labels': sizes,
            'datasets': [
                {
                    'label': algo,
                    'data': baseline_results[baseline_results['Algorithm'] == algo]['Total Distance (km)'].tolist(),
                    'borderColor': color,
                    'fill': False
                } for algo, color in [
                    ('Dijkstra', '#1f77b4'),
                    ('DivideConquer', '#ff7f0e'),
                    ('FloydHeuristic', '#2ca02c')
                ]
            ]
        },
        'options': {
            'title': {'display': True, 'text': 'Total Distance vs Number of Delivery Points'},
            'scales': {
                'x': {'title': {'display': True, 'text': 'Number of Delivery Points'}},
                'y': {'title': {'display': True, 'text': 'Total Distance (km)'}}
            }
        }
    }
    print("\nChart for Total Distance vs Size:")
    print("```chartjs")
    print(chart_data)
    print("```")

# --- Main Execution ---
if __name__ == "__main__":
    G, df = load_dataset()
    plot_map(G, df)
    run_experiments(G, df)