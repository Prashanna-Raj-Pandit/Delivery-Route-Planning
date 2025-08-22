import osmnx as ox

# Load the GraphML
G = ox.load_graphml("chicago_street_network.graphml")
print(G.number_of_nodes(), G.number_of_edges())
