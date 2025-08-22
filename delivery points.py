######## Generating delivery points ###########

import random
import pandas as pd
import osmnx as ox

# Load graph
G = ox.load_graphml("chicago_street_network.graphml")

# Pick 1000 random nodes as delivery points
delivery_nodes = random.sample(list(G.nodes()), 1000)

# Convert to dataframe
delivery_data = []
for node in delivery_nodes:
    delivery_data.append({
        "node_id": node,
        "lat": G.nodes[node]['y'],
        "lon": G.nodes[node]['x']
    })

df = pd.DataFrame(delivery_data)

# Save to CSV
df.to_csv("delivery_points.csv", index=False)

print(df.head())
