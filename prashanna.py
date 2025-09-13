import pandas as pd
import networkx as nx
import osmnx as ox
import numpy as np
import time
import psutil
######## Generating delivery points ###########

import random
import pandas as pd
import osmnx as ox

DISTRIBUTION={
    "equal":[33,33,34],
    "high_dominance":[50,25,25],
    "low_dominance":[25,25,50],
    "med_dominance":[25,50,25]
}

def generate_delivery_points(sample_size=1000,distribution="equal"):
    # Load graph
    G = ox.load_graphml("./chicago_street_network.graphml")

    # Pick 1000 random nodes as delivery points
    delivery_nodes = random.sample(list(G.nodes()), sample_size)

    # Convert to dataframe
    delivery_data = []
    for node in delivery_nodes:
        delivery_data.append({
            "node_id": node,
            "lat": G.nodes[node]['y'],
            "lon": G.nodes[node]['x'],
            # Assign random priority: 1 = High, 2 = Medium, 3 = Low
            "priority": random.choices([1, 2, 3],weights=DISTRIBUTION[distribution])[0]
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

# How to use ?
# generate_delivery_points(sample_size=100,distribution="high_dominance")
def north_south_band_partitioning(G, delivery_df, nodes, depot_south, depot_north, num_bands=4):
    """
    Divide the city into latitude bands and visit as many delivery points as possible
    while moving from South → North depot with minimal distance.
    """

    # Sort deliveries by latitude (South → North)
    delivery_df = delivery_df.copy()
    delivery_df = delivery_df.sort_values("lat")

    # Create latitude bands
    lat_min, lat_max = delivery_df["lat"].min(), delivery_df["lat"].max()
    bands = np.linspace(lat_min, lat_max, num_bands + 1)

    # Assign each delivery to a band
    delivery_df["band"] = pd.cut(delivery_df["lat"], bins=bands, labels=False, include_lowest=True)

    route = [depot_south]   # start at South depot
    total_distance = 0.0
    deliveries_visited = []

    current = depot_south

    # Solve routing within each band in order
    for b in range(num_bands):
        band_nodes = delivery_df[delivery_df["band"] == b]["node_id"].astype(str).tolist()

        if not band_nodes:
            continue

        unvisited = set(band_nodes)

        while unvisited:
            # Dijkstra distances from current node
            dist_dict = nx.single_source_dijkstra_path_length(G, current, weight="length")

            # Pick closest delivery in this band
            reachable = [d for d in unvisited if d in dist_dict]
            if not reachable:
                break

            next_node = min(reachable, key=lambda x: dist_dict[x])
            total_distance += dist_dict[next_node]
            route.append(next_node)
            deliveries_visited.append(next_node)

            current = next_node
            unvisited.remove(next_node)

    # Finally connect to the North depot
    try:
        dist_to_north = nx.dijkstra_path_length(G, current, depot_north, weight="length")
        total_distance += dist_to_north
        route.append(depot_north)
    except nx.NetworkXNoPath:
        print("Warning: North depot unreachable from last node.")

    return route, deliveries_visited, total_distance

size=[100,500,1000,1500,2000]

if __name__ == "__main__":
    # Load graph
    G = ox.load_graphml("./chicago_street_network.graphml")
    results=[]
    # Load CSVs
    nodes = pd.read_csv("chicago_nodes.csv")

    # Build graph (ensure nodes are string)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes})

    # Pick South and North depots
    depot_south = str(nodes.loc[nodes['y'].idxmin(), 'osmid'])
    depot_north = str(nodes.loc[nodes['y'].idxmax(), 'osmid'])

    print("South depot:", depot_south)
    print("North depot:", depot_north)

    for s in size:
        generate_delivery_points(sample_size=s,distribution="equal")
        delivery_df = pd.read_csv("delivery_points.csv")
        # Run experiment
        start_time = time.time()
        route, deliveries_visited, total_distance = north_south_band_partitioning(
            G, delivery_df, nodes, depot_south, depot_north, num_bands=4
        )
        runtime = time.time() - start_time
        memory = psutil.Process().memory_info().rss / 1024**2

        # Results
        metrics = {
            "depot_south": depot_south,
            "depot_north": depot_north,
            "num_deliveries_possible": len(delivery_df),
            "num_deliveries_visited": len(deliveries_visited),
            "total_distance_m": total_distance,
            "runtime_s": runtime,
            "memory_MB": memory,
            "nodes_visited": len(route),
        }
        results.append(metrics)

    pd.DataFrame(results).to_csv("exp1_band_results.csv", index=False)
