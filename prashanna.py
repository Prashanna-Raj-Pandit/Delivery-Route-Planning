import pandas as pd
import networkx as nx
import time
import psutil
import random
from simran import greedy_nearest_neighbor
import os
import sys
from dataset.delivery_points import *
import sys
from dataset import greedyPlot

# TODO: implement all the algorithm by maximizing the number of delivery points cvered. also keep in mind that the path is not very long.
# TODO: use weighted technique like. distance x 0.3 + delivery points x 0.7 weights
# TODO: use weights like 0.5 x high priority + 0.3 mid priority + 0.2 low priority ( for experiment 3)

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