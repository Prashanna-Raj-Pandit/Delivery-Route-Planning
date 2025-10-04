import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt


def chicago_with_delivery_points():
    # Load the street network
    G = ox.load_graphml("../chicago_street_network.graphml")

    # Load delivery points
    df = pd.read_csv("../delivery_points.csv")

    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot edges (roads) in light gray
    edges.plot(ax=ax, linewidth=0.5, edgecolor='lightgray',label="Edges")

    # Plot nodes (intersections) in blue
    nodes.plot(ax=ax, markersize=5, color='blue', alpha=0.6, label='Nodes')

    # Plot delivery points in red
    ax.scatter(df['lon'], df['lat'], c='red', s=15, label='Delivery Points', alpha=0.8)

    # Add title and legend
    ax.set_title("Chicago Street Network with Intersections and Delivery Points", fontsize=16)
    ax.legend()

    plt.show()


def plot_priority_nodes():
    # Load the street network
    G = ox.load_graphml("../chicago_street_network.graphml")

    # Load delivery points with priority
    df = pd.read_csv("../delivery_points.csv")

    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    priority_colors = {
        1: "red",  # High priority
        2: "orange",  # Medium priority
        3: "green"  # Low priority
    }

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot edges (roads) in light gray
    edges.plot(ax=ax, linewidth=0.5, edgecolor="lightgray")

    # Plot nodes (intersections) in blue
    # nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.5, label="Intersections")

    # Plot delivery points by priority
    for priority, color in priority_colors.items():
        subset = df[df["priority"] == priority]
        ax.scatter(
            subset["lon"], subset["lat"],
            c=color, s=20, alpha=0.8,
            label=f"Priority {priority}"
        )

    ax.set_title("Chicago Street Network with Priority-Based Delivery Points", fontsize=16)
    ax.legend(title="Legend", fontsize=10)

    plt.show()


# Define depot locations
depots = {
    'C': {'lat': 41.85, 'lon': -87.68, 'color': 'red', 'name': 'Central'},
    'N': {'lat': 42.00, 'lon': -87.70, 'color': 'purple', 'name': 'North'},
    'S': {'lat': 41.70, 'lon': -87.63, 'color': 'orange', 'name': 'South'},
    'E': {'lat': 41.8914, 'lon': -87.6099, 'color': 'blue', 'name': 'East'},
    'W': {'lat': 41.8800, 'lon': -87.7500, 'color': 'green', 'name': 'West'},
}
import matplotlib.patches as mpatches

def chicago_with_delivery_points_and_depots():
    # Load the street network
    G = ox.load_graphml("../chicago_street_network.graphml")

    # Load delivery points
    df = pd.read_csv("delivery_points.csv")

    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    # Map depots to nearest nodes in the graph
    depot_nodes = {}
    for label, depot in depots.items():
        node = ox.distance.nearest_nodes(G, depot['lon'], depot['lat'])
        depot_nodes[label] = node

    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot edges (roads) in light gray
    edges.plot(ax=ax, linewidth=0.5, edgecolor='lightgray')

    # Plot delivery points (all red, no priority)
    ax.scatter(df['lon'], df['lat'], c='red', s=15, label='Delivery Points', alpha=0.8)

    # Plot depots with their colors
    depot_coords = nodes.loc[list(depot_nodes.values())]
    ax.scatter(
        depot_coords.geometry.x, depot_coords.geometry.y,
        c=[depots[label]['color'] for label in depot_nodes.keys()],
        s=80, marker='s', edgecolors='black'
    )

    # Add depot labels (short label: C, N, S, etc.)
    for label, node_id in depot_nodes.items():
        row = nodes.loc[node_id]
        ax.text(
            row.geometry.x, row.geometry.y, label,
            fontsize=10, ha="center", va="center", color="white",
            bbox=dict(facecolor=depots[label]['color'], alpha=0.7, boxstyle="circle,pad=0.2")
        )

    # Create custom legend handles using depot names
    depot_handles = [
        mpatches.Patch(color=depot['color'], label=depot['name'])
        for depot in depots.values()
    ]

    # Add title and legend
    ax.set_title("Chicago Street Network with Intersections, Delivery Points, and Depots", fontsize=16)
    ax.legend(
        handles=[plt.Line2D([0], [0], color='red', marker='o', linestyle='', markersize=6, label='Delivery Points')]
        + depot_handles
    )

    plt.show()


# Run it
chicago_with_delivery_points_and_depots()
# chicago_with_delivery_points()
# plot_priority_nodes()