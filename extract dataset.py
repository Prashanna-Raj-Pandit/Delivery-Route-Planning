###### Chicago map #######

import osmnx as ox

# Download the drivable road network of Chicago
G = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")

# Save to file if you want
ox.save_graphml(G, "chicago_street_network.graphml")


# Convert to GeoDataFrames (nodes and edges)
nodes, edges = ox.graph_to_gdfs(G)

# Save them as CSVs
nodes.to_csv("chicago_nodes.csv")
edges.to_csv("chicago_edges.csv")

print(" - chicago_nodes.csv")
print(" - chicago_edges.csv")

# Display first few rows
print("\nNodes sample:")
print(nodes.head())

print("\nEdges sample:")
print(edges.head())

