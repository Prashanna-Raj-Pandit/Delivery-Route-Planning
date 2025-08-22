import osmnx as ox
import networkx as nx
import pandas as pd

# Load graph
G = ox.load_graphml("chicago_street_network.graphml")

# Get nodes and edges as GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

print("✅ Basic Graph Info")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is directed? {nx.is_directed(G)}")

# --- NODE STATS ---
print("\n📌 Node statistics:")
print(nodes.describe(include="all"))

# --- EDGE STATS ---
print("\n📌 Edge statistics:")
print(edges.describe(include="all"))

# --- EXTRA METRICS ---
total_length_km = edges["length"].sum() / 1000
avg_street_length = edges["length"].mean()
max_street_length = edges["length"].max()

print("\n📌 Road Network Summary:")
print(f"Total road length: {total_length_km:.2f} km")
print(f"Average street segment length: {avg_street_length:.2f} m")
print(f"Longest street segment: {max_street_length:.2f} m")

# Road types distribution
road_type_counts = edges["highway"].value_counts()
print("\n📌 Road types distribution:")
print(road_type_counts)
