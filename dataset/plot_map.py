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
    edges.plot(ax=ax, linewidth=0.5, edgecolor='lightgray')

    # Plot nodes (intersections) in blue
    nodes.plot(ax=ax, markersize=5, color='blue', alpha=0.6, label='Intersections')

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

# chicago_with_delivery_points()
plot_priority_nodes()