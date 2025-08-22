# # import osmnx as ox
# # G = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")
# # ox.plot_graph(G, node_size=5, edge_linewidth=0.5)
#
# import osmnx as ox
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the street network
# G = ox.load_graphml("chicago_street_network.graphml")
#
# # Load the delivery points CSV
# df = pd.read_csv("delivery_points.csv")
#
# # Plot the street network
# fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='lightgray')
#
# # Overlay delivery points
# ax.scatter(df['lon'], df['lat'], c='red', s=10, label='Delivery Points', alpha=0.7)
#
# # Add title and legend
# ax.set_title("Chicago Street Network with Delivery Points")
# ax.legend()
#
# plt.show()


import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

# Load the street network
G = ox.load_graphml("chicago_street_network.graphml")

# Load delivery points
df = pd.read_csv("delivery_points.csv")

# Convert graph to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# Start plotting
fig, ax = plt.subplots(figsize=(12, 12))

# Plot edges (roads) in light gray
edges.plot(ax=ax, linewidth=0.5, edgecolor='lightgray')

# Plot nodes (intersections) in blue
nodes.plot(ax=ax, markersize=5, color='blue', alpha=0.6, label='Intersections')

# Plot delivery points in red
ax.scatter(df['lon'], df['lat'], c='red', s=15, label='Delivery Points', alpha=0.8)

# Add title and legend
ax.set_title("Chicago Street Network with Intersections and Delivery Points", fontsize=16)
ax.legend()

plt.show()
