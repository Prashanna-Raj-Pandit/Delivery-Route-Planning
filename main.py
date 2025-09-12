import pandas as pd
import networkx as nx
import time
import psutil
import random
from simran import greedy_nearest_neighbor  
import os
import sys

sys.path.append("dataset")  # folder that extracts csv file dataset from extractdataset.py and deliverypoints.py

# CSV file names created by the scripts
NODES_FILE = "chicago_nodes.csv"
EDGES_FILE = "chicago_edges.csv"
DELIVERY_FILE = "delivery_points.csv"


#Generates datasets only if needed
if not (os.path.exists(NODES_FILE) and os.path.exists(EDGES_FILE)):
    print("Nodes/edges CSVs not found. Running extractdataset.py...")
    import extractdataset
else:
    print("Nodes/edges CSVs already exist. Skipping extraction.")

if not os.path.exists(DELIVERY_FILE):
    print("Delivery points CSV not found. Running deliverypoints.py...")
    import deliverypoints
else:
    print("Delivery points CSV already exists. Skipping generation.")



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


# Load delivery points
delivery_df = pd.read_csv(DELIVERY_FILE)
# Convert node_ids to strings and keep only those that exist in the graph
deliveries = [str(d) for d in delivery_df['node_id'] if str(d) in G.nodes]


#Experiment 1 - Varying delivery points
print("\nRunning Experiment 1: Varying number of delivery points")
experiment1Results = []
delivery_sizes = [100, 200, 300, 500, 1000]
# delivery_sizes=[100,500,1000,1500,2000]



for n in delivery_sizes:
    trial_deliveries = deliveries[:n]  # first n deliveries
    depot = trial_deliveries[0]

    start_time = time.time()
    route, totalDistance = greedy_nearest_neighbor(G, depot, trial_deliveries[1:])
    # print(f"Route from {depot} to {trial_deliveries[1:]}\n{route}")
    runtime = time.time() - start_time
    memory = psutil.Process().memory_info().rss / 1024**2  # in MB

    experiment1Results.append({
        "num_deliveries": n,
        "depot": depot,
        "total_distance_m": totalDistance,
        "runtime_s": runtime,
        "memory_MB": memory,
        "nodes_visited": len(route)
    })

exp1_df = pd.DataFrame(experiment1Results)
exp1_df.to_csv("experiment1_results.csv", index=False)
print(exp1_df)


# Experiment 2 - Different depots
print("\nRunning Experiment 2: Different depots")
NUM_DELIVERIES = 1000
if len(deliveries) >= NUM_DELIVERIES:
    random.seed(42)  # fixed seed ensures same selection every run
    deliveries = random.sample(deliveries, NUM_DELIVERIES)
else:
    deliveries = deliveries  # use all if less than 1000

print(f"Using {len(deliveries)} delivery points")
experiment2_results = []
# Select depots (C, N, S, E, W)
depot_nodes = {}

# Central = closest to centroid
centerlat, centerlon = nodes['y'].mean(), nodes['x'].mean()
central = nodes.iloc[((nodes['y']-centerlat)**2 + (nodes['x']-centerlon)**2).idxmin()]['osmid']
depot_nodes['C'] = str(central)

# North, South, East, West
depot_nodes['N'] = str(nodes.loc[nodes['y'].idxmax(), 'osmid'])
depot_nodes['S'] = str(nodes.loc[nodes['y'].idxmin(), 'osmid'])
depot_nodes['E'] = str(nodes.loc[nodes['x'].idxmax(), 'osmid'])
depot_nodes['W'] = str(nodes.loc[nodes['x'].idxmin(), 'osmid'])

print("Depots selected:", depot_nodes)
print("\nRunning Experiment 2: Different depots")
experiment2Results = []

# Define depot-to-depot pairs you want to test
depot_pairs = [("C","E"), ("S","E"), ("W","S"), ("E","W")]

# Toggle: same_depot = True runs round trips, False runs cross-depot
same_depot = False

for src, dst in depot_pairs:
    source = depot_nodes[src]
    destination = depot_nodes[dst]

    if same_depot:
        # Case 1: start and end at same depot (round trip)
        deliveries_for_run = [d for d in deliveries if d != source]
    else:
        # Case 2: start at src, end at dst
        deliveries_for_run = [d for d in deliveries if d not in [source, destination]]
        deliveries_for_run.append(destination)  # force route to end at dst

    # Run the algorithm
    start_time = time.time()
    route, total_distance = greedy_nearest_neighbor(G, source, deliveries_for_run)
    # print(f"Route from {source} to {deliveries_for_run}\n{route}")
    runtime = time.time() - start_time
    memory = psutil.Process().memory_info().rss / 1024**2

    # Record results
    experiment2_results.append({
        "source": src,
        "destination": dst if not same_depot else src,
        "total_distance_m": total_distance,
        "runtime_s": runtime,
        "memory_MB": memory,
        "nodes_visited": len(route),
        "delivery_nodes_covered": len([n for n in route if n in deliveries])
    })

# Save results
exp2_df = pd.DataFrame(experiment2_results)
exp2_df.to_csv("experiment2_results.csv", index=False)
print(exp2_df)



#Experiment 3 - Priority delivery
print("\nRunning Experiment 3: Priority delivery")
experiment3_results = []

for n in delivery_sizes:
    trial_deliveries = deliveries[:n]
    depot = trial_deliveries[0]

    # Sort deliveries by priority (1=High first)
    trial_deliveries_sorted = delivery_df[delivery_df['node_id'].isin(trial_deliveries)].sort_values(by='priority')
    trial_deliveries_sorted = list(trial_deliveries_sorted['node_id'].astype(str))

    start_time = time.time()
    route, total_distance = greedy_nearest_neighbor(G, depot, [d for d in trial_deliveries_sorted if d != depot])
    runtime = time.time() - start_time
    memory = psutil.Process().memory_info().rss / 1024**2

    # Measure priority adherence
    high_count = sum(1 for d in route if delivery_df.loc[delivery_df['node_id']==int(d), 'priority'].values[0]==1)
    total_high = sum(delivery_df['priority']==1)
    priority_adherence = high_count / total_high * 100 if total_high > 0 else 0

    experiment3_results.append({
        "num_deliveries": n,
        "depot": depot,
        "total_distance_m": total_distance,
        "runtime_s": runtime,
        "memory_MB": memory,
        "nodes_visited": len(route),
        "priority_adherence_%": priority_adherence
    })

exp3_df = pd.DataFrame(experiment3_results)
exp3_df.to_csv("experiment3_results.csv", index=False)
print(exp3_df)

print("\nAll experiments completed. CSV files saved.")
