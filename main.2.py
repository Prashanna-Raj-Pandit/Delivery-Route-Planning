import random
import pandas as pd
import networkx as nx
import time
import psutil
import os
import math
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Distribution definitions
DISTRIBUTION = {
    "equal": [33, 33, 34],
    "high_dominance": [50, 25, 25],
    "low_dominance": [25, 25, 50],
    "med_dominance": [25, 50, 25]
}
def generate_delivery_points(G, sample_size=1000, distribution="equal"):
    # Pick random nodes as delivery points
    delivery_nodes = random.sample(list(G.nodes()), sample_size)

    # Convert to dataframe
    delivery_data = []
    for node in delivery_nodes:
        delivery_data.append({
            "node_id": node,
            "lat": G.nodes[node]['y'],
            "lon": G.nodes[node]['x'],
            # Assign random priority: 1 = High, 2 = Medium, 3 = Low
            "priority": random.choices([1, 2, 3], weights=DISTRIBUTION[distribution])[0]
        })

    df = pd.DataFrame(delivery_data)

    # Save to CSV
    df.to_csv("delivery_points.csv", index=False)

    # Print sample data
    print("Sample Delivery Points:")
    print(df.head())

    # Print statistics
    print("\nDataset Statistics:")
    print(df.describe(include='all'))

    # Priority distribution
    print("\nPriority Distribution:")
    print(df['priority'].value_counts())

    return df


def dijkstra_path_length(G, source, target, weight='length'):
    """Calculate shortest path length using Dijkstra's algorithm"""
    try:
        path = nx.shortest_path(G, source, target, weight=weight, method='dijkstra')
        length = nx.shortest_path_length(G, source, target, weight=weight, method='dijkstra')
        return path, length
    except nx.NetworkXNoPath:
        return [], float('inf')


def greedy_nearest_neighbor(G, start, deliveries, exp=1):
    """Greedy algorithm that always visits the nearest unvisited delivery point"""
    unvisited = set(deliveries)
    current = start
    route = [current]
    total_distance = 0
    delivery_points_visited = []

    # For experiment 3, we need to consider priorities
    priority_weights = {1: 0.5, 2: 0.3, 3: 0.2} if exp == 3 else {1: 1, 2: 1, 3: 1}

    while unvisited:
        # Find the nearest unvisited delivery point
        min_distance = float('inf')
        next_point = None

        for point in unvisited:
            # Get priority weight for this point
            priority = int(G.nodes[point].get('priority', 2))
            weight = priority_weights[priority]

            # Calculate weighted distance
            _, dist = dijkstra_path_length(G, current, point)
            weighted_dist = dist * (1 - weight * 0.3)  # Prioritize higher priority points

            if weighted_dist < min_distance:
                min_distance = weighted_dist
                next_point = point

        if next_point is None:
            break

        # Add the path to the next point to our route
        path, dist = dijkstra_path_length(G, current, next_point)
        route.extend(path[1:])  # Skip the first node (current) to avoid duplication
        total_distance += dist

        # Record that we visited this delivery point
        delivery_points_visited.append({
            'node_id': next_point,
            'priority': int(G.nodes[next_point].get('priority', 2))
        })

        # Update current position and remove from unvisited
        current = next_point
        unvisited.remove(next_point)

    # Return to start if needed
    if exp != 2:  # For experiment 2, we end at destination, not start
        path, dist = dijkstra_path_length(G, current, start)
        route.extend(path[1:])
        total_distance += dist

    return route, total_distance, delivery_points_visited


def simple_spatial_clustering(points_data, max_points_per_cluster=20):
    """Simple spatial clustering without using QuadTree (procedural approach)"""
    if not points_data:
        return []

    # Extract coordinates
    xs = [p['x'] for p in points_data]
    ys = [p['y'] for p in points_data]

    # Find bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Simple grid-based clustering
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Determine grid size based on number of points
    grid_size = max(2, int(math.sqrt(len(points_data) / max_points_per_cluster)))

    # Create grid cells
    grid = {}
    for point in points_data:
        x_idx = int((point['x'] - min_x) / x_range * grid_size)
        y_idx = int((point['y'] - min_y) / y_range * grid_size)

        if (x_idx, y_idx) not in grid:
            grid[(x_idx, y_idx)] = []
        grid[(x_idx, y_idx)].append(point)

    # Return clusters
    return list(grid.values())


def quad_tree_decomposition(G, start, deliveries, exp=1):
    """Divide and conquer approach using simple spatial clustering"""
    # Get coordinates for all delivery points
    points_data = []
    for point in deliveries:
        points_data.append({
            'node': point,
            'x': G.nodes[point]['x'],
            'y': G.nodes[point]['y']
        })

    # Use simple spatial clustering
    clusters = simple_spatial_clustering(points_data)

    # For each cluster, find the centroid and solve TSP
    cluster_routes = []
    for cluster in clusters:
        if not cluster:
            continue

        # Extract just the node IDs from the cluster
        cluster_nodes = [point['node'] for point in cluster]

        # Find centroid of cluster
        centroid_x = sum(point['x'] for point in cluster) / len(cluster)
        centroid_y = sum(point['y'] for point in cluster) / len(cluster)

        # Find node closest to centroid
        centroid_node = min(cluster, key=lambda p: (p['x'] - centroid_x) ** 2 + (p['y'] - centroid_y) ** 2)['node']

        # Solve TSP for this cluster using greedy approach
        cluster_route, cluster_dist, _ = greedy_nearest_neighbor(G, centroid_node, cluster_nodes, exp)
        cluster_routes.append((cluster_route, cluster_dist, centroid_node))

    # Connect clusters in order of proximity to start
    unvisited_clusters = cluster_routes.copy()
    current = start
    full_route = [current]
    total_distance = 0
    delivery_points_visited = []

    while unvisited_clusters:
        # Find the nearest cluster centroid
        min_dist = float('inf')
        next_cluster_idx = -1

        for i, (route, dist, centroid) in enumerate(unvisited_clusters):
            _, cluster_dist = dijkstra_path_length(G, current, centroid)
            if cluster_dist < min_dist:
                min_dist = cluster_dist
                next_cluster_idx = i

        if next_cluster_idx == -1:
            break

        # Add path to this cluster
        next_route, next_dist, next_centroid = unvisited_clusters[next_cluster_idx]
        path_to_cluster, dist_to_cluster = dijkstra_path_length(G, current, next_centroid)

        full_route.extend(path_to_cluster[1:])
        total_distance += dist_to_cluster

        # Add the cluster's route
        full_route.extend(next_route[1:])
        total_distance += next_dist

        # Record delivery points visited in this cluster
        for node in next_route:
            if node in deliveries and node not in [d['node_id'] for d in delivery_points_visited]:
                delivery_points_visited.append({
                    'node_id': node,
                    'priority': int(G.nodes[node].get('priority', 2))
                })

        # Update current position and remove cluster
        current = next_route[-1] if next_route else current
        unvisited_clusters.pop(next_cluster_idx)

    # Return to start if needed
    if exp != 2:
        path, dist = dijkstra_path_length(G, current, start)
        full_route.extend(path[1:])
        total_distance += dist

    return full_route, total_distance, delivery_points_visited


def floyd_warshall(G, start, deliveries, exp=1):
    """Floyd-Warshall algorithm for all-pairs shortest paths"""
    # This is a simplified version for demonstration
    # Note: Full Floyd-Warshall would be too expensive for large graphs

    # For large graphs, we'll use a simplified approach
    # that only computes paths between delivery points

    # Create a distance matrix between all delivery points
    n = len(deliveries)
    dist_matrix = np.full((n, n), float('inf'))

    # Initialize diagonal
    for i in range(n):
        dist_matrix[i][i] = 0

    # Calculate distances between delivery points
    for i in range(n):
        for j in range(i + 1, n):
            try:
                dist = nx.shortest_path_length(G, deliveries[i], deliveries[j], weight='length')
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
            except nx.NetworkXNoPath:
                pass

    # Solve TSP using dynamic programming (held-karp algorithm)
    # This is a simplified version for demonstration
    def held_karp(dist_matrix):
        n = len(dist_matrix)
        # memoization table: key=(bitmask of visited nodes, last node index)
        memo = {}

        def dp(mask, pos):
            if (mask, pos) in memo:
                return memo[(mask, pos)]

            if mask == (1 << n) - 1:
                return dist_matrix[pos][0], [pos]  # Return to start

            min_cost = float('inf')
            best_path = []

            for next_pos in range(n):
                if not (mask & (1 << next_pos)):
                    new_mask = mask | (1 << next_pos)
                    cost, path = dp(new_mask, next_pos)
                    total_cost = dist_matrix[pos][next_pos] + cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_path = [pos] + path

            memo[(mask, pos)] = (min_cost, best_path)
            return min_cost, best_path

        return dp(1, 0)  # Start with node 0 visited

    # Get the optimal path
    total_distance, path_indices = held_karp(dist_matrix)

    # Reconstruct the full route
    route = [start]
    delivery_points_visited = []

    # Add path from start to first delivery point
    first_delivery = deliveries[path_indices[0]]
    path_to_first, dist_to_first = dijkstra_path_length(G, start, first_delivery)
    route.extend(path_to_first[1:])

    # Add paths between delivery points
    for i in range(len(path_indices) - 1):
        from_idx = path_indices[i]
        to_idx = path_indices[i + 1]
        from_node = deliveries[from_idx]
        to_node = deliveries[to_idx]

        path, dist = dijkstra_path_length(G, from_node, to_node)
        route.extend(path[1:])

        # Record delivery point
        delivery_points_visited.append({
            'node_id': from_node,
            'priority': int(G.nodes[from_node].get('priority', 2))
        })

    # Record last delivery point
    last_node = deliveries[path_indices[-1]]
    delivery_points_visited.append({
        'node_id': last_node,
        'priority': int(G.nodes[last_node].get('priority', 2))
    })

    # Return to start if needed
    if exp != 2:
        path, dist = dijkstra_path_length(G, last_node, start)
        route.extend(path[1:])
        total_distance += dist + dist_to_first

    return route, total_distance, delivery_points_visited


def filter_deliveries_between(src_node, dst_node, deliveries, G):
    """
    Select deliveries geographically between source and destination.
    Uses bounding box between src and dst.
    """
    src_lat, src_lon = G.nodes[src_node]['y'], G.nodes[src_node]['x']
    dst_lat, dst_lon = G.nodes[dst_node]['y'], G.nodes[dst_node]['x']

    # bounding box
    lat_min, lat_max = min(src_lat, dst_lat), max(src_lat, dst_lat)
    lon_min, lon_max = min(src_lon, dst_lon), max(src_lon, dst_lon)

    deliveries_filtered = [
        d for d in deliveries
        if (lat_min <= G.nodes[d]['y'] <= lat_max) and
           (lon_min <= G.nodes[d]['x'] <= lon_max)
    ]

    return deliveries_filtered


def plot_experiment1_results(df):
    """Plot results for experiment 1"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Runtime vs Number of deliveries
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[0, 0].plot(algo_data['num_deliveries'], algo_data['runtime_s'], 'o-', label=algo)
    axes[0, 0].set_xlabel('Number of Deliveries')
    axes[0, 0].set_ylabel('Runtime (s)')
    axes[0, 0].set_title('Runtime vs Number of Deliveries')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Memory usage vs Number of deliveries
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[0, 1].plot(algo_data['num_deliveries'], algo_data['memory_MB'], 'o-', label=algo)
    axes[0, 1].set_xlabel('Number of Deliveries')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Usage vs Number of Deliveries')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Distance vs Number of deliveries
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[1, 0].plot(algo_data['num_deliveries'], algo_data['total_distance_m'], 'o-', label=algo)
    axes[1, 0].set_xlabel('Number of Deliveries')
    axes[1, 0].set_ylabel('Total Distance (m)')
    axes[1, 0].set_title('Total Distance vs Number of Deliveries')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Delivery coverage vs Number of deliveries
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[1, 1].plot(algo_data['num_deliveries'], algo_data['number_of_DPoints_visited'], 'o-', label=algo)
    axes[1, 1].set_xlabel('Number of Deliveries')
    axes[1, 1].set_ylabel('Delivery Points Visited')
    axes[1, 1].set_title('Delivery Coverage vs Number of Deliveries')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/experiment1_results.png')
    plt.close()


def plot_experiment2_results(df):
    """Plot results for experiment 2"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Runtime by depot pair
    depot_pairs = df['source_depot_name'] + '-' + df['destination_depot_name']
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[0, 0].bar(range(len(algo_data)), algo_data['runtime_s'], label=algo, alpha=0.7)
    axes[0, 0].set_xlabel('Depot Pairs')
    axes[0, 0].set_ylabel('Runtime (s)')
    axes[0, 0].set_title('Runtime by Depot Pair')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Memory usage by depot pair
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[0, 1].bar(range(len(algo_data)), algo_data['memory_MB'], label=algo, alpha=0.7)
    axes[0, 1].set_xlabel('Depot Pairs')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Usage by Depot Pair')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Distance by depot pair
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]
        axes[1, 0].bar(range(len(algo_data)), algo_data['total_distance_m'], label=algo, alpha=0.7)
    axes[1, 0].set_xlabel('Depot Pairs')


##########

# Generic code in process


#########






sys.path.append("dataset")  # folder that extracts csv file dataset from extractdataset.py and deliverypoints.py

# CSV file names created by the scripts
NODES_FILE = "chicago_nodes.csv"
EDGES_FILE = "chicago_edges.csv"
algorithms=['greedy','dnq','dp']

# Load nodes and edges
nodes = pd.read_csv(NODES_FILE)
edges = pd.read_csv(EDGES_FILE)

# Build graph from nodes and edges
G = nx.MultiDiGraph()
for _, row in nodes.iterrows():
    G.add_node(str(row['osmid']), y=row['y'], x=row['x'])

for _, row in edges.iterrows():
    u, v = str(row['u']), str(row['v'])
    length = row['length']
    G.add_edge(u, v, length=length)



#Experiment 1 - Varying delivery points
print("\nRunning Experiment 1: Varying number of delivery points")
delivery_sizes = [100, 200, 300, 500, 1000]
experiment1Results = []

for size in delivery_sizes:
    start_depot = north_node
    end_depot=south_node

    df = generate_delivery_points(sample_size=size, distribution="equal")
    deliveries = [str(d) for d in df['node_id']]
    for algo in algorithms:
        if algo=='greedy':
            start_time = time.time()
            route, totalDistance,delivery_points_visited = greedy_nearest_neighbor(G, start_depot, deliveries,exp=1)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024**2  # in MB
        elif algo=='dnq':
            start_time = time.time()
            route, totalDistance,delivery_points_visited = quad_tree_decompositoin(G, start_depot, deliveriese,exp=1)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
        elif algo=='dp':
            start_time = time.time()
            route, totalDistance,delivery_points_visited = floyd_warshall(G, start_depot, deliveries,exp=1)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
    experiment1Results.append({
        "algo":algo,
        "num_deliveries": size,
        "start_depot": start_depot,
        "end_depot":end_depot,
        "total_distance_m": totalDistance,
        "runtime_s": runtime, #
        "memory_MB": memory,
        "nodes_visited": len(route),
        "number_of_DPoints_visited": len(delivery_points_visited),
        "delivery_points_visited":delivery_points_visited,

    })

exp1_df = pd.DataFrame(experiment1Results)
exp1_df.to_csv("experiment1_results.csv", index=False)
# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
greedyPlot.plot_experiment1_results(exp1_df)
print(exp1_df)



def filter_deliveries_between(src_node, dst_node, deliveries, nodes):
    """
    Select deliveries geographically between source and destination.
    Uses bounding box between src and dst.
    """
    # convert osmid to string for matching
    nodes_str = nodes.copy()
    nodes_str['osmid'] = nodes_str['osmid'].astype(str)

    src_lat, src_lon = nodes_str.loc[nodes_str['osmid'] == str(src_node), ['y', 'x']].values[0]
    dst_lat, dst_lon = nodes_str.loc[nodes_str['osmid'] == str(dst_node), ['y', 'x']].values[0]

    # bounding box
    lat_min, lat_max = min(src_lat, dst_lat), max(src_lat, dst_lat)
    lon_min, lon_max = min(src_lon, dst_lon), max(src_lon, dst_lon)

    deliveries_filtered = [
        d for d in deliveries
        if (lat_min <= nodes_str.loc[nodes_str['osmid'] == str(d), 'y'].values[0] <= lat_max) and
           (lon_min <= nodes_str.loc[nodes_str['osmid'] == str(d), 'x'].values[0] <= lon_max)
    ]

    return deliveries_filtered


# Experiment 2 ---
experiment2results = []

# Select depots (C, N, S, E, W)
depot_nodes = {}
centerlat, centerlon = nodes['y'].mean(), nodes['x'].mean()
central = nodes.iloc[((nodes['y']-centerlat)**2 + (nodes['x']-centerlon)**2).idxmin()]['osmid']
depot_nodes['C'] = str(central)
depot_nodes['N'] = str(nodes.loc[nodes['y'].idxmax(), 'osmid'])
depot_nodes['S'] = str(nodes.loc[nodes['y'].idxmin(), 'osmid'])
depot_nodes['E'] = str(nodes.loc[nodes['x'].idxmax(), 'osmid'])
depot_nodes['W'] = str(nodes.loc[nodes['x'].idxmin(), 'osmid'])

print("Depots selected:", depot_nodes)
print("\nRunning Experiment 2: Different depots")

# Define depot-to-depot pairs
depot_pairs = [("C","E"), ("S","E"), ("W","S"), ("E","W")]

same_depot = False  # round trip vs cross depot

for src, dst in depot_pairs:
    source = depot_nodes[src]
    destination = depot_nodes[dst]

    # NEW: filter deliveries based on depot geography
    deliveries_for_run = filter_deliveries_between(source, destination, deliveries, nodes)

    if not same_depot:
        deliveries_for_run.append(destination)  # force route to end at dst

    if len(deliveries_for_run) == 0:
        print(f"️ No deliveries found between {src} and {dst}, skipping...")
        continue
    for algo in algorithms:
        # Run the algorithm
        if algo=="greedy":
            start_time = time.time()
            route, total_distance,delivery_points_visited = greedy_nearest_neighbor(G, source, deliveries_for_run,exp=2)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024**2
        elif algo=="dp":
            start_time = time.time()
            route, total_distance,delivery_points_visited = floyd_warshall(G, source, deliveries_for_run,exp=2)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 ** 2
        elif algo=="dnq":
            start_time = time.time()
            route, total_distance,delivery_points_visited = quad_tree_decompositoin(G, source, deliveries_for_run,exp=2)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 ** 2
        else:
            print("invalid algorithm")
        experiment2results.append({
        "algo":algo,
        "source_depot_name": src,  # C, N, S, E, W
        "source_node_id": source,  # actual node ID in the graph
        "destination_depot_name": dst if not same_depot else src,
        "destination_node_id": destination,
        "total_distance_m": total_distance,
        "runtime_s": runtime,
        "memory_MB": memory,
        "nodes_visited": len(route),
        "delivery_nodes_covered": len([n for n in route if n in deliveries]),
        "route_node_ids": ",".join(route)  # full route as comma-separated node IDs
        })


# Save results
exp2_df = pd.DataFrame(experiment2results)
exp2_df.to_csv("experiment2_results.csv", index=False)
# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
greedyPlot.plot_experiment2_results(exp2_df)
nodes_df = pd.read_csv("chicago_nodes.csv")

greedyPlot.plot_depot_routes(exp2_df, nodes_df)
print(exp2_df)

#experiment 3
experiment3results = []

priority_distributions = ["equal", "high_dominance", "med_dominance", "low_dominance"]

for dist_name in priority_distributions:
    print(f"\nRunning experiment for distribution: {dist_name}")

    # Generate deliveries with given distribution
    deliveries_df = generate_delivery_points(
        sample_size=1000,
        distribution=dist_name
    )

   # Ensure node IDs are strings
    deliveries_df["node_id"] = deliveries_df["node_id"].astype(str)

    depot = deliveries_df.iloc[0]["node_id"]
    delivery_nodes = [n for n in deliveries_df["node_id"].values if n != depot]

    for algo in algorithms:
        if algo=="greedy":

            route, distance,delivery_points_visited = greedy_nearest_neighbor(G, depot, delivery_nodes,exp=3)
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024**2

        elif algo=="dnq":
            route, distance,delivery_points_visited = greedy_nearest_neighbor(G, depot, delivery_nodes,exp=3 )
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 ** 2
        elif algo=="dp":
            route, distance,delivery_points_visited = greedy_nearest_neighbor(G, depot, delivery_nodes,exp=3 )
            runtime = time.time() - start_time
            memory = psutil.Process().memory_info().rss / 1024 ** 2
        else:
            print("Invalid algorithm selection")
        high_priority = sum(delivery_points_visited["priority"] == 1)
        med_priority = sum(delivery_points_visited["priority"] == 2)
        low_priority = sum(delivery_points_visited["priority"] == 3)
        n = len(deliveries_df)

        experiment3results.append({
            "distribution": dist_name,
            "num_deliveries": 1000,
            "depot": depot,
            "total_distance_m": distance,
            "runtime_s": runtime,
            "memory_MB": memory,
            "high_priority_count": high_priority,
            "med_priority_count": med_priority,
            "low_priority_count": low_priority,
            "high_percentage": high_priority / n * 100,
            "med_percentage": med_priority / n * 100,
            "low_percentage": low_priority / n * 100
        })

    print(f"Distribution {dist_name}: {distance:.2f} m, "
          f"Runtime: {runtime:.2f}s, Memory: {memory:.2f} MB")

df = pd.DataFrame(experiment3results)
df.to_csv("experiment3_results.csv", index=False)

if not os.path.exists('plots'):
    os.makedirs('plots')

greedyPlot.plot_experiment3_results(df)
print(df)