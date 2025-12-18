# Floyd–Warshall delivery routing (Jupyter-friendly)
# --------------------------------------------------
# Requirements: osmnx, networkx, pandas, numpy, matplotlib
# Files expected:
#   - chicago_street_network.graphml  (street graph)
#   - delivery_points.csv             (node_id,lat,lon,priority)
#
# Outputs:
#   - fw_route_legs.csv  (per-leg metrics)
#   - fw_summary.csv     (aggregate metrics)
#   - fw_route.png       (optional plot)

import os
import time
import tracemalloc
import math
import random
import sys
from collections import defaultdict

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (adjust as needed)
# -----------------------------
GRAPH_PATH = "../chicago_street_network.graphml"
DELIVERIES_CSV = "delivery_points.csv"
N_DELIVERIES   = 100          # number of delivery points to include (subset of CSV)
SEED           = 42
PLOT_ROUTE     = True         # set False to skip plotting
WEIGHT_ATTR    = "length"     # OSMnx edge length in meters

random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# Helpers / I/O routines
# -----------------------
def ensure_delivery_points(G, csv_path=DELIVERIES_CSV, n_default=1000):
    """
    Load delivery_points.csv if present; else generate n_default random graph nodes with priority.
    Returns DataFrame with columns: node_id, lat, lon, priority
    """
    try:
        df = pd.read_csv(csv_path)
        if "priority" not in df.columns:
            # assign priorities if missing (approx Chicago proposal split)
            num_points = len(df)
            if num_points == 1000:
                priorities = [1]*329 + [2]*355 + [3]*316
                random.shuffle(priorities)
            else:
                priorities = random.choices([1,2,3], weights=[0.329,0.355,0.316], k=num_points)
            df["priority"] = priorities
            df.to_csv(csv_path, index=False)
        return df
    except FileNotFoundError:
        # generate
        nodes = list(G.nodes())
        picks = random.sample(nodes, n_default)
        data = []
        for n in picks:
            data.append({
                "node_id": n,
                "lat": G.nodes[n]["y"],
                "lon": G.nodes[n]["x"],
                "priority": random.choice([1,2,3])
            })
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return df

def pick_depot(G):
    """Pick a random depot node (could be replaced by a known depot)."""
    return random.choice(list(G.nodes()))

def osm_shortest_path_and_length(G, u, v, weight=WEIGHT_ATTR):
    """Return (path_nodes_list, length_km). If no path, returns ([], inf)."""
    try:
        path = nx.shortest_path(G, u, v, weight=weight)
        length_m = nx.path_weight(G, path, weight=weight)
        return path, length_m / 1000.0
    except nx.NetworkXNoPath:
        return [], float("inf")

# ----------------------------------------------
# Build pairwise distance matrix over deliveries
# ----------------------------------------------
def build_pairwise_matrix(G, nodes_list, weight=WEIGHT_ATTR):
    """
    For the given list of OSM node ids (delivery nodes + depot),
    compute an |V|x|V| distance matrix using OSM shortest paths.
    Also cache each leg's OSM path for later metrics.
    """
    n = len(nodes_list)
    D = np.full((n, n), np.inf, dtype=float)
    next_idx = [[None]*n for _ in range(n)]  # for FW path reconstruction on delivery graph
    path_cache = {}  # (i,j) -> list of OSM nodes along that leg

    for i in range(n):
        D[i, i] = 0.0
        next_idx[i][i] = i

    # Compute direct distances between every pair using OSM shortest paths
    for i in range(n):
        ui = nodes_list[i]
        for j in range(i+1, n):
            vj = nodes_list[j]
            path, dist_km = osm_shortest_path_and_length(G, ui, vj, weight=weight)
            if math.isfinite(dist_km):
                D[i, j] = dist_km
                D[j, i] = dist_km
                next_idx[i][j] = j
                next_idx[j][i] = i
                path_cache[(i, j)] = path
                path_cache[(j, i)] = list(reversed(path))
            # if no path, remains inf and next_idx stays None
    return D, next_idx, path_cache

# -----------------------
# Floyd–Warshall routine
# -----------------------
def floyd_warshall(D, next_idx):
    """
    Standard Floyd–Warshall on the delivery-node distance matrix.
    D is modified in place; next_idx is predecessor/next matrix for path reconstruction.
    """
    n = D.shape[0]
    for k in range(n):
        Dik = D[:, k]
        Dkj = D[k, :]
        for i in range(n):
            # skip if inf on left
            di = Dik[i]
            if not math.isfinite(di):
                continue
            # vectorized-ish update
            cand = di + Dkj
            better = cand < D[i, :]
            if np.any(better):
                for j in np.where(better)[0]:
                    D[i, j] = cand[j]
                    next_idx[i][j] = next_idx[i][k]  # first hop from i toward k
    return D, next_idx

def reconstruct_fw_sequence(next_idx, i, j):
    """
    Reconstruct sequence of delivery-node indices from i to j using next matrix.
    Returns list of delivery indices [i, ..., j]; empty if no path.
    """
    if next_idx[i][j] is None:
        return []
    seq = [i]
    while i != j:
        i = next_idx[i][j]
        if i is None:  # defensive
            return []
        seq.append(i)
    return seq

# -----------------------------------
# Build a route using FW distances
# -----------------------------------
def build_route_nearest_neighbor(D, start_idx=0):
    """
    Simple TSP heuristic over delivery indices, guided by FW distances.
    Returns a cycle starting and ending at start_idx.
    """
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start_idx)
    route = [start_idx]
    curr = start_idx
    while unvisited:
        # choose nearest by D
        next_i = min(unvisited, key=lambda j: D[curr, j])
        route.append(next_i)
        unvisited.remove(next_i)
        curr = next_i
    route.append(start_idx)  # return to depot
    return route

# ----------------------------------------------------------------
# Convert delivery-index route into actual OSM path and statistics
# ----------------------------------------------------------------
def route_osm_paths_and_metrics(G, nodes_list, route_idx, D, next_idx, path_cache):
    """
    For a delivery-index route (e.g., [0, 5, 2, 0]), stitch together the OSM paths.
    Returns:
      legs_df: per-leg details (from_node, to_node, distance_km, nodes_in_leg, edges_in_leg)
      summary: dict of totals (nodes_visited, edges_visited, total_distance_km)
    """
    # accumulate unique nodes/edges visited (in OSM graph space)
    visited_nodes = set()
    visited_edges = set()

    legs = []
    for a, b in zip(route_idx[:-1], route_idx[-1:0:-1]):  # wrong: fix below
        pass

# (Continue the cell — fixed stitching + metrics)

def route_osm_paths_and_metrics(G, nodes_list, route_idx, D, next_idx, path_cache):
    visited_nodes = set()
    visited_edges = set()
    legs = []

    # Helper to add an OSM node path into visited sets & return counts
    def add_osm_path(path_nodes):
        for n in path_nodes:
            visited_nodes.add(n)
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            visited_edges.add((u, v))
        return len(path_nodes), max(0, len(path_nodes) - 1)

    # For each hop in delivery index space, we might need to pass through intermediate delivery indices
    # according to FW's next matrix. Reconstruct the sequence of delivery indices, then stitch the cached OSM legs.
    total_distance_km = 0.0
    legs_records = []

    for i_idx, j_idx in zip(route_idx[:-1], route_idx[1:]):
        # Sequence of delivery indices per FW (e.g., [i, k, ..., j])
        seq = reconstruct_fw_sequence(next_idx, i_idx, j_idx)
        if not seq:
            # Fallback: direct cached leg (if exists)
            seq = [i_idx, j_idx]

        # Now stitch OSM legs along the seq
        osm_nodes_full = []
        leg_distance_km = 0.0

        for s, t in zip(seq[:-1], seq[1:]):
            # Try to fetch cached OSM path between delivery indices (s,t)
            path = path_cache.get((s, t))
            if path is None:
                # As a fallback, compute now
                path, _dist_km = osm_shortest_path_and_length(G, nodes_list[s], nodes_list[t])
                # cache it
                path_cache[(s, t)] = path
                path_cache[(t, s)] = list(reversed(path)) if path else []
                # distance:
                try:
                    _len_m = nx.path_weight(G, path, weight=WEIGHT_ATTR)
                    _dist_km = _len_m / 1000.0
                except Exception:
                    _dist_km = float("inf")
            else:
                # compute distance from path to be precise
                try:
                    _len_m = nx.path_weight(G, path, weight=WEIGHT_ATTR)
                    _dist_km = _len_m / 1000.0
                except Exception:
                    _dist_km = float("inf")

            leg_distance_km += _dist_km

            if not osm_nodes_full:
                osm_nodes_full.extend(path)
            else:
                # avoid duplicating the junction node
                if path and osm_nodes_full and path[0] == osm_nodes_full[-1]:
                    osm_nodes_full.extend(path[1:])
                else:
                    osm_nodes_full.extend(path)

        # Update global visited sets and record per-hop metrics
        nodes_in_leg, edges_in_leg = add_osm_path(osm_nodes_full)
        legs_records.append({
            "from_delivery_idx": i_idx,
            "to_delivery_idx": j_idx,
            "from_node_id": nodes_list[i_idx],
            "to_node_id": nodes_list[j_idx],
            "distance_km": leg_distance_km,
            "nodes_in_leg": nodes_in_leg,
            "edges_in_leg": edges_in_leg
        })
        total_distance_km += leg_distance_km

    legs_df = pd.DataFrame(legs_records)
    summary = {
        "total_distance_km": total_distance_km,
        "nodes_visited": len(visited_nodes),
        "edges_visited": len(visited_edges),
    }
    return legs_df, summary

# ----------------
# Optional plotting
# ----------------
def plot_route(G, deliveries_df, nodes_list, route_idx, path_cache, out_path="fw_route.png"):
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    # Build full node list along route for plotting (using cached legs)
    plot_nodes = []
    for i_idx, j_idx in zip(route_idx[:-1], route_idx[1:]):
        path = path_cache.get((i_idx, j_idx))
        if path is None:
            path, _ = osm_shortest_path_and_length(G, nodes_list[i_idx], nodes_list[j_idx])
        if not plot_nodes:
            plot_nodes.extend(path)
        else:
            if path and plot_nodes and path[0] == plot_nodes[-1]:
                plot_nodes.extend(path[1:])
            else:
                plot_nodes.extend(path)

    # Extract coordinates
    xs = [G.nodes[n]["x"] for n in plot_nodes if n in G.nodes]
    ys = [G.nodes[n]["y"] for n in plot_nodes if n in G.nodes]

    fig, ax = plt.subplots(figsize=(12, 12))
    edges_gdf.plot(ax=ax, linewidth=0.5, alpha=0.4)
    nodes_gdf.plot(ax=ax, markersize=1, alpha=0.4)

    # delivery points
    ax.scatter(deliveries_df["lon"], deliveries_df["lat"], s=10, alpha=0.9, label="Deliveries")
    # route polyline
    ax.plot(xs, ys, linewidth=2, label="FW Route")

    ax.set_title("Route (Floyd–Warshall distance matrix + NN tour)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

# -----------------------
# Main notebook workflow
# -----------------------
# 1) Load graph
print("Loading graph...")
G = ox.load_graphml(GRAPH_PATH)
G = G.to_directed()
print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# 2) Load/generate deliveries and subset
deliveries_all = ensure_delivery_points(G, DELIVERIES_CSV)
if N_DELIVERIES > len(deliveries_all):
    N_DELIVERIES = len(deliveries_all)

deliveries = deliveries_all.sample(n=N_DELIVERIES, random_state=SEED).reset_index(drop=True)
delivery_nodes = deliveries["node_id"].tolist()
depot = pick_depot(G)
nodes_list = [depot] + delivery_nodes  # index 0 is depot

print(f"Depot (OSM node): {depot}")
print(f"Using {len(delivery_nodes)} delivery points.")

# 3) Build pairwise OSM distances + FW
print("Computing pairwise OSM shortest-path distances and running Floyd–Warshall...")
tracemalloc.start()
t0 = time.perf_counter()

D, next_idx, path_cache = build_pairwise_matrix(G, nodes_list, weight=WEIGHT_ATTR)
t1 = time.perf_counter()
D_fw, next_fw = floyd_warshall(D, next_idx)
t2 = time.perf_counter()

# 4) Build a route (NN using FW distances)
route_idx = build_route_nearest_neighbor(D_fw, start_idx=0)
t3 = time.perf_counter()

# 5) Convert to OSM paths & compute metrics
legs_df, summary = route_osm_paths_and_metrics(G, nodes_list, route_idx, D_fw, next_fw, path_cache)
t4 = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# 6) Execution time / memory
timings = {
    "pairwise_osm_dist_sec": t1 - t0,
    "floyd_warshall_sec": t2 - t1,
    "route_construction_sec": t3 - t2,
    "osm_metrics_sec": t4 - t3,
    "total_runtime_sec": t4 - t0
}
memory_mb = peak / (1024**2)

# 7) Aggregate metrics
summary_out = {
    "n_deliveries": len(delivery_nodes),
    "nodes_visited": summary["nodes_visited"],
    "edges_visited": summary["edges_visited"],
    "total_distance_km": summary["total_distance_km"],
    "pairwise_osm_dist_sec": timings["pairwise_osm_dist_sec"],
    "floyd_warshall_sec": timings["floyd_warshall_sec"],
    "route_construction_sec": timings["route_construction_sec"],
    "osm_metrics_sec": timings["osm_metrics_sec"],
    "total_runtime_sec": timings["total_runtime_sec"],
    "peak_memory_mb": memory_mb
}

# 8) Save outputs
legs_df.to_csv("fw_route_legs.csv", index=False)
pd.DataFrame([summary_out]).to_csv("fw_summary.csv", index=False)

print("\n=== Floyd–Warshall Results ===")
print(pd.DataFrame([summary_out]).T)

# 9) Optional plot
if PLOT_ROUTE:
    plot_route(G, deliveries, nodes_list, route_idx, path_cache, out_path="fw_route.png")
