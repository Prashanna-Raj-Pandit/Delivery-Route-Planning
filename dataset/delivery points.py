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

######## Generating delivery points with distributions ###########

# generate_delivery_points_variants.py
# Generates multiple independent 1000-point delivery datasets per distribution.
# Files: delivery_points_1000_<distribution>_v<1..5>.csv
# Also writes generation_summary.csv with counts & percentages.

# import os
# import time
# import random
# import pandas as pd
# import osmnx as ox

# GRAPH_FILE = "chicago_street_network.graphml"

# # Distributions: weights for priorities [P1, P2, P3]
# DISTRIBUTIONS = {
#     "equal":          [33, 33, 34],   # ~33% each
#     "high_dominance": [50, 25, 25],   # P1 ~50%
#     "low_dominance":  [25, 25, 50],   # P3 ~50%
#     # Uncomment if you want a medium-dominance option:
#     # "med_dominance":  [25, 50, 25], # P2 ~50%
# }

# # Which distributions to generate
# DISTRIBUTIONS_TO_MAKE = ["equal", "high_dominance", "low_dominance"]

# def generate_one_dataset(G, sample_size: int, weights: list[int], rng: random.Random):
#     """Return a DataFrame of delivery points (lat, lon, priority) for one draw."""
#     nodes_list = list(G.nodes())
#     if sample_size > len(nodes_list):
#         # Fallback: sample with replacement if graph is tiny (unlikely for Chicago)
#         chosen_nodes = rng.choices(nodes_list, k=sample_size)
#     else:
#         chosen_nodes = rng.sample(nodes_list, sample_size)

#     rows = []
#     for node in chosen_nodes:
#         rows.append({
#             "node_id": node,
#             "lat": G.nodes[node]["y"],
#             "lon": G.nodes[node]["x"],
#             "priority": rng.choices([1, 2, 3], weights=weights, k=1)[0]
#         })
#     return pd.DataFrame(rows)

# def save_with_checks(df: pd.DataFrame, path: str):
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#     df.to_csv(path, index=False)

# def main(sample_size=1000, num_variants=5, out_dir=".", base_seed=20241004):
#     print("Loading graph…")
#     G = ox.load_graphml(GRAPH_FILE)
#     print(f"Graph loaded with {len(G.nodes())} nodes, {len(G.edges())} edges.")

#     summary_rows = []

#     for dist_name in DISTRIBUTIONS_TO_MAKE:
#         weights = DISTRIBUTIONS[dist_name]
#         print(f"\n=== Generating {num_variants} datasets for '{dist_name}' "
#               f"({sample_size} points each) ===")

#         for i in range(1, num_variants + 1):
#             # Reproducible RNG per variant & distribution
#             seed = (base_seed * 1315423911) ^ hash((dist_name, i))
#             rng = random.Random(seed)

#             df = generate_one_dataset(G, sample_size, weights, rng)
#             filename = os.path.join(out_dir, f"delivery_points_{sample_size}_{dist_name}_v{i}.csv")
#             save_with_checks(df, filename)

#             # Stats for summary
#             counts = df["priority"].value_counts().reindex([1,2,3], fill_value=0)
#             pcts = counts / len(df) * 100.0
#             print(f"  [{dist_name} v{i}] saved → {filename} | "
#                   f"P1={counts[1]}({pcts[1]:.1f}%), "
#                   f"P2={counts[2]}({pcts[2]:.1f}%), "
#                   f"P3={counts[3]}({pcts[3]:.1f}%)")

#             summary_rows.append({
#                 "file": os.path.basename(filename),
#                 "distribution": dist_name,
#                 "sample_size": len(df),
#                 "P1_count": int(counts[1]),
#                 "P2_count": int(counts[2]),
#                 "P3_count": int(counts[3]),
#                 "P1_pct": round(pcts[1], 2),
#                 "P2_pct": round(pcts[2], 2),
#                 "P3_pct": round(pcts[3], 2),
#                 "seed": seed,
#             })

#     summary_path = os.path.join(out_dir, "generation_summary.csv")
#     pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
#     print(f"\nWrote summary: {summary_path}")

# if __name__ == "__main__":
#     # Change DISTRIBUTIONS_TO_MAKE above if you want only one distribution.
#     # Example for “just 5 for 1000” total (not per distribution):
#     # DISTRIBUTIONS_TO_MAKE = ["equal"]  # <- then you’ll get exactly 5 datasets.
#     main(sample_size=1000, num_variants=5, out_dir=".")



# generate_random_nodes.py
# Makes 5 independent datasets of 1,000 randomly sampled street nodes from Chicago.
# Columns: node_id, lat, lon  (no priority column)

import os
import random
import pandas as pd
import osmnx as ox

GRAPH_FILE = "chicago_street_network.graphml"

def generate_random_nodes(G, sample_size: int, rng: random.Random) -> pd.DataFrame:
    nodes = list(G.nodes())
    if sample_size > len(nodes):
        chosen = rng.choices(nodes, k=sample_size)       # fallback: with replacement
    else:
        chosen = rng.sample(nodes, sample_size)          # without replacement
    rows = [{"node_id": n, "lat": G.nodes[n]["y"], "lon": G.nodes[n]["x"]} for n in chosen]
    return pd.DataFrame(rows)

def main(sample_size=1000, num_variants=5, out_dir=".", base_seed=20241004):
    os.makedirs(out_dir, exist_ok=True)
    print("loading graph…")
    G = ox.load_graphml(GRAPH_FILE)
    print(f"graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    for i in range(1, num_variants + 1):
        seed = (base_seed * 2654435761) ^ i
        rng = random.Random(seed)
        df = generate_random_nodes(G, sample_size, rng)
        path = os.path.join(out_dir, f"delivery_points_{sample_size}_v{i}.csv")
        df.to_csv(path, index=False)
        print(f"[v{i}] saved {path} (rows={len(df)})")

if __name__ == "__main__":
    main(sample_size=1000, num_variants=5, out_dir=".")
