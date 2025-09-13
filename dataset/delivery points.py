# # ######## Generating delivery points ###########

# # import random
# # import pandas as pd
# # import osmnx as ox

# # # Load graph
# # G = ox.load_graphml("../chicago_street_network.graphml")

# # # Pick 1000 random nodes as delivery points
# # delivery_nodes = random.sample(list(G.nodes()), 1000)

# # # Convert to dataframe
# # delivery_data = []
# # for node in delivery_nodes:
# #     delivery_data.append({
# #         "node_id": node,
# #         "lat": G.nodes[node]['y'],
# #         "lon": G.nodes[node]['x'],
# #         # Assign random priority: 1 = High, 2 = Medium, 3 = Low
# #         "priority": random.choice([1, 2, 3])
# #     })

# # df = pd.DataFrame(delivery_data)

# # # Save to CSV
# # df.to_csv("delivery_points.csv", index=False)

# # # Print sample data
# # print("Sample Delivery Points:")
# # print(df.head())

# # # Print statistics
# # print("\nDataset Statistics:")
# # print(df.describe(include='all'))

# # # Priority distribution
# # print("\nPriority Distribution:")
# # print(df['priority'].value_counts())






# ### Asha delivery points.py ###
# import pandas as pd
# import random

# # Load existing delivery_points.csv
# df = pd.read_csv("../delivery_points.csv")

# # Assign priorities to match proposal distribution
# num_points = len(df)
# priorities = [1] * 329 + [2] * 355 + [3] * 316
# if num_points != 1000:
#     print(f"Warning: CSV has {num_points} points, not 1000. Adjusting priorities.")
#     priorities = random.choices([1, 2, 3], weights=[0.329, 0.355, 0.316], k=num_points)
# else:
#     random.shuffle(priorities)

# # Add priority column
# df['priority'] = priorities[:num_points]

# # Save updated CSV
# df.to_csv("../delivery_points.csv", index=False)
# print("Updated delivery_points.csv with priority column.")
# print("\nSample Delivery Points:")
# print(df.head())
# print("\nPriority Distribution:")
# print(df['priority'].value_counts())


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
    G = ox.load_graphml("chicago_street_network.graphml")

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


if __name__ == "__main__":
    generate_delivery_points(sample_size=500, distribution="equal")