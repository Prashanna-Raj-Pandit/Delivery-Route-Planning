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
        "lon": G.nodes[node]['x'],
        # Assign random priority: 1 = High, 2 = Medium, 3 = Low
        "priority": random.choice([1, 2, 3])
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

