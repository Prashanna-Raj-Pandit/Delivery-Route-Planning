import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import random
import os

# =========================
# Load Data
# =========================
edges = pd.read_csv("chicago_edges.csv")
nodes = pd.read_csv("chicago_nodes.csv")
deliveries = pd.read_csv("delivery_points.csv")

# Build Graph
G = nx.from_pandas_edgelist(
    edges, source="u", target="v", edge_attr=["length"], create_using=nx.DiGraph()
)

# pos = {row["node_id"]: (row["lon"], row["lat"]) for _, row in nodes.iterrows()}
pos = {row["osmid"]: (row["x"], row["y"]) for _, row in nodes.iterrows()}
# =========================
# Algorithms
# =========================
def greedy_dijkstra(G, source, target):
    return nx.dijkstra_path(G, source=source, target=target, weight="length")

def divide_and_conquer_kmeans(G, delivery_nodes, k=5):
    coords = np.array([[pos[n][0], pos[n][1]] for n in delivery_nodes])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    clusters = {i: [] for i in range(k)}
    for node, label in zip(delivery_nodes, kmeans.labels_):
        clusters[label].append(node)
    routes = []
    for cluster_nodes in clusters.values():
        if len(cluster_nodes) > 1:
            subgraph = G.subgraph(cluster_nodes)
            mst = nx.minimum_spanning_tree(subgraph.to_undirected())
            routes.append(list(mst.nodes))
    return routes

def floyd_warshall_dp(G):
    return dict(nx.floyd_warshall(G, weight="length"))

# =========================
# Experiments
# =========================
def experiment_scalability(G, deliveries):
    sizes = [100, 200, 300, 500, 1000]
    results = []
    for n in sizes:
        subset = deliveries.sample(n)
        source, target = random.sample(list(G.nodes), 2)
        # Greedy
        path = greedy_dijkstra(G, source, target)
        results.append({"algo": "Greedy", "size": n, "length": len(path)})
        # Divide & Conquer
        routes = divide_and_conquer_kmeans(G, subset["node_id"].tolist(), k=5)
        results.append({"algo": "Divide-Conquer", "size": n, "length": sum(len(r) for r in routes)})
        # DP
        dist = floyd_warshall_dp(G)
        results.append({"algo": "DP", "size": n, "length": dist[source][target]})
    return pd.DataFrame(results)

def experiment_multidepot(G, deliveries):
    depots = random.sample(list(G.nodes), 5)  # C,E,W,N,S (placeholder)
    pairs = [(depots[0], depots[1]), (depots[1], depots[2]), (depots[2], depots[3])]
    results = []
    for (src, dst) in pairs:
        path = greedy_dijkstra(G, src, dst)
        results.append({"algo": "Greedy", "pair": f"{src}-{dst}", "length": len(path)})
        routes = divide_and_conquer_kmeans(G, deliveries["node_id"].sample(500).tolist(), k=5)
        results.append({"algo": "Divide-Conquer", "pair": f"{src}-{dst}", "length": sum(len(r) for r in routes)})
        dist = floyd_warshall_dp(G)
        results.append({"algo": "DP", "pair": f"{src}-{dst}", "length": dist[src][dst]})
    return pd.DataFrame(results)

def experiment_priority(G, deliveries):
    results = []
    source, target = random.sample(list(G.nodes), 2)
    for algo in ["Greedy", "Divide-Conquer", "DP"]:
        for priority in [1, 2, 3]:
            subset = deliveries[deliveries["priority"] == priority]
            count = len(subset)
            results.append({"algo": algo, "priority": priority, "count": count})
    return pd.DataFrame(results)

# =========================
# Run Experiments
# =========================
scalability_df = experiment_scalability(G, deliveries)
# multidepot_df = experiment_multidepot(G, deliveries)
# priority_df = experiment_priority(G, deliveries)

# =========================
# Plots
# =========================
def plot_scalability(df):
    plt.figure(figsize=(8,6))
    sns.lineplot(data=df, x="size", y="length", hue="algo", marker="o")
    plt.title("Scalability of Algorithms")
    plt.xlabel("Number of Delivery Points")
    plt.ylabel("Path/Route Length")
    plt.show()

def plot_multidepot(df):
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x="pair", y="length", hue="algo")
    plt.title("Multi-Depot Route Comparison")
    plt.xlabel("Depot Pairs")
    plt.ylabel("Path/Route Length")
    plt.show()

def plot_priority(df):
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x="priority", y="count", hue="algo")
    plt.title("Priority-Aware Delivery")
    plt.xlabel("Priority Level")
    plt.ylabel("Deliveries Served")
    plt.show()

# Run plots
plot_scalability(scalability_df)
# plot_multidepot(multidepot_df)
# plot_priority(priority_df)

# =========================
# Plot Map with Priorities
# =========================
def plot_delivery_map(G, nodes, deliveries):
    fig, ax = plt.subplots(figsize=(12,12))
    edges.plot(ax=ax, linewidth=0.5, edgecolor="lightgray")
    # Plot nodes
    ax.scatter(nodes["lon"], nodes["lat"], c="blue", s=2, alpha=0.3, label="Intersections")
    # Plot priorities
    colors = {1: "red", 2: "orange", 3: "green"}
    for p, c in colors.items():
        subset = deliveries[deliveries["priority"] == p]
        ax.scatter(subset["lon"], subset["lat"], c=c, s=20, alpha=0.8, label=f"Priority {p}")
    ax.set_title("Chicago Delivery Points with Priority Levels")
    ax.legend()
    plt.show()

plot_delivery_map(G, nodes, deliveries)
