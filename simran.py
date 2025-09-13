import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

def plot_priority_nodes_with_depots():
    # === Load street network ===
    G = ox.load_graphml("chicago_street_network.graphml")

    # === Load delivery points with priority ===
    df = pd.read_csv("delivery_points.csv")

    # === Convert graph to GeoDataFrames ===
    nodes, edges = ox.graph_to_gdfs(G)

    # === Define colors for priority ===
    priority_colors = {
        1: "red",     # High priority
        2: "orange",  # Medium priority
        3: "green"    # Low priority
    }

    # === Compute depot nodes ===
    depot_nodes = {}
    centerlat, centerlon = nodes.geometry.y.mean(), nodes.geometry.x.mean()
    central = nodes.loc[((nodes.geometry.y - centerlat) ** 2 + (nodes.geometry.x - centerlon) ** 2).idxmin()].name
    depot_nodes["C"] = central
    depot_nodes["N"] = nodes.geometry.y.idxmax()
    depot_nodes["S"] = nodes.geometry.y.idxmin()
    depot_nodes["E"] = nodes.geometry.x.idxmax()
    depot_nodes["W"] = nodes.geometry.x.idxmin()
    print(depot_nodes)
    # === Plotting ===
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot Chicago street network
    edges.plot(ax=ax, linewidth=0.5, edgecolor="lightgray")

    # Plot delivery points by priority
    for priority, color in priority_colors.items():
        subset = df[df["priority"] == priority]
        ax.scatter(
            subset["lon"], subset["lat"],
            c=color, s=20, alpha=0.8,
            label=f"Priority {priority}"
        )

    # Plot depot nodes as black-edged squares with labels
    depot_coords = nodes.loc[list(depot_nodes.values())]
    ax.scatter(
        depot_coords.geometry.x, depot_coords.geometry.y,
        c="blue", s=80, marker="s", edgecolors="black", label="Depots"
    )

    for label, node_id in depot_nodes.items():
        row = nodes.loc[node_id]
        ax.text(
            row.geometry.x, row.geometry.y, label,
            fontsize=10, ha="center", va="center", color="white",
            bbox=dict(facecolor="blue", alpha=0.7, boxstyle="circle,pad=0.2")
        )

    ax.set_title("Chicago Street Network with Priority-Based Delivery Points and Depots", fontsize=16)
    ax.legend(title="Legend", fontsize=10)

    plt.show()

# Run the function
plot_priority_nodes_with_depots()
