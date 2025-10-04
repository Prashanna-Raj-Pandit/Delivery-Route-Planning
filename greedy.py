# greedy.py
# Greedy nearest-neighbor single-path router on a road network
# - Follows a simple heuristic: from the current node, visit the nearest unvisited delivery
#   that still allows you to finish at the END within the distance budget (if provided).
# - Expands every hop to real street paths using NetworkX shortest paths (weight='length').
# - Exposes a compatible API subset with your divideconquer.DivideConquerRouter used by your experiments.

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional

import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd


class GreedyRouter:
    def __init__(self, graphml_path_or_graph):
        """
        Parameters
        ----------
        graphml_path_or_graph : str | networkx.MultiDiGraph
            Path to a GraphML file (e.g., 'chicago_street_network.graphml') OR
            a preloaded osmnx/NetworkX graph.
        """
        if isinstance(graphml_path_or_graph, str):
            self.G = ox.load_graphml(graphml_path_or_graph)
        else:
            self.G = graphml_path_or_graph

        # Node & edge GeoDataFrames for spatial ops & plotting
        self.nodes, self.edges = ox.graph_to_gdfs(self.G)

        # Build KDTree on (lat, lon) for fast nearest-node lookup
        self.nodes = self.nodes.to_crs(epsg=4326) if self.nodes.crs is not None else self.nodes
        self.nodes["lat"] = self.nodes["y"].astype(float)
        self.nodes["lon"] = self.nodes["x"].astype(float)
        self.kdtree = KDTree(self.nodes[["lat", "lon"]].to_numpy(), metric="euclidean")

        # Average-lat scaling for rough lon-distance scaling
        self.cos_avg_lat = math.cos(np.deg2rad(self.nodes["lat"].mean()))

        # Chicago-ish depot anchors (consistent with your D&C code)
        self.depots = {
            'C': {'lat': 41.85,   'lon': -87.68,   'color': 'red',    'name': 'Central'},
            'N': {'lat': 42.00,   'lon': -87.70,   'color': 'purple', 'name': 'North'},
            'S': {'lat': 41.70,   'lon': -87.63,   'color': 'orange', 'name': 'South'},
            'E': {'lat': 41.8914, 'lon': -87.6099, 'color': 'blue',   'name': 'East'},
            'W': {'lat': 41.8800, 'lon': -87.7500, 'color': 'green',  'name': 'West'},
        }
        self.depot_nodes: Dict[str, int] = {}
        for k, d in self.depots.items():
            _, ind = self.kdtree.query([[d['lat'], d['lon']]], k=1)
            self.depot_nodes[k] = int(self.nodes.index[ind[0,0]])
            print(f"Depot {k} ({d['name']}) -> node {self.depot_nodes[k]}")

        # Cache for single-source Dijkstra distances
        self._cache = {}

        print("GreedyRouter ready.")

    # ---------------------------- helpers ----------------------------
    def _dijkstra_lengths(self, source: int) -> Dict[int, float]:
        """Shortest-path length (meters) from source to all nodes (cached)."""
        if source not in self._cache:
            self._cache[source] = nx.single_source_dijkstra_path_length(self.G, source, weight='length')
        return self._cache[source]

    def find_nearest_node(self, lat: float, lon: float) -> int:
        _, ind = self.kdtree.query([[lat, lon]], k=1)
        return int(self.nodes.index[ind[0,0]])

    def load_and_preprocess_points(self, csv_path: str) -> pd.DataFrame:
        """Load delivery points; expects columns: lat, lon. Optional: priority."""
        df = pd.read_csv(csv_path)
        if 'priority' not in df.columns:
            df['priority'] = 2
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df = df.dropna(subset=['lat', 'lon']).copy()
        df['nearest_node'] = df.apply(lambda r: self.find_nearest_node(r['lat'], r['lon']), axis=1)
        return df

    def baseline_sn_distance_km(self, start_depot: str='S', end_depot: str='N') -> float:
        s = self.depot_nodes[start_depot]; e = self.depot_nodes[end_depot]
        d_m = nx.shortest_path_length(self.G, s, e, weight='length')
        return d_m / 1000.0

    def calculate_route_distance(self, route_nodes: List[int]) -> float:
        tot = 0.0
        for a, b in zip(route_nodes, route_nodes[1:]):
            try:
                tot += nx.shortest_path_length(self.G, a, b, weight='length')
            except Exception:
                pass
        return tot

    # ---------------------------- greedy core ----------------------------
    def solve_greedy_single_path(self,
                                 points: pd.DataFrame,
                                 start_depot: str='S',
                                 end_depot: str='N',
                                 budget_km: Optional[float]=None,
                                 reserve_pct: float = 0.10
                                 ) -> Tuple[List[int], float, List[dict], List[int]]:
        """
        Greedy NN with budget-feasibility: at each step choose the nearest unvisited
        delivery (by network distance) that still allows finishing at END with a
        reserve on the end-leg. If budget_km is None, visit all by pure NN.

        Returns
        -------
        route_nodes : list[int]
            Full list of graph nodes along the stitched path (start → ... → end)
        total_dist_m : float
            Total length of the route in meters
        region_solutions : list[dict]
            Placeholder for compatibility; empty list here.
        visited_deliveries : list[int]
            Node IDs of deliveries that were actually visited
        """
        start_node = self.depot_nodes[start_depot]
        end_node   = self.depot_nodes[end_depot]

        # Delivery node IDs
        candidates = list(map(int, points['nearest_node'].unique().tolist()))
        remaining = set(candidates)

        route_nodes: List[int] = [start_node]
        visited: List[int] = []

        total_m = 0.0
        cur = start_node

        # Precompute end-legs once
        end_dists = self._dijkstra_lengths(end_node)

        # Convert budget to meters and compute reserve on the end-leg
        budget_m: Optional[float] = None if budget_km is None else budget_km * 1000.0
        def feasible(next_node: int, additional_leg_m: float) -> bool:
            if budget_m is None:
                return True
            # require we can still reach END with a reserve on that leg
            end_leg = end_dists.get(next_node, float('inf'))
            reserve = reserve_pct * end_dists.get(start_node, end_dists.get(cur, 0.0))
            return (total_m + additional_leg_m + end_leg + reserve) <= budget_m

        while remaining:
            # For current position, find nearest feasible next
            dists = self._dijkstra_lengths(cur)
            best_node = None
            best_leg_m = float('inf')
            for node in remaining:
                leg = dists.get(node, float('inf'))
                if leg < best_leg_m:
                    # check feasibility against budget
                    if feasible(node, leg):
                        best_leg_m = leg
                        best_node = node

            if best_node is None:
                # cannot add more (budget or disconnected); go to END
                break

            # Append the street path for cur → best_node
            try:
                path = nx.shortest_path(self.G, cur, best_node, weight='length')
            except nx.NetworkXNoPath:
                # Skip unreachable nodes
                remaining.remove(best_node)
                continue

            # Stitch, avoiding duplicate of current
            route_nodes.extend(path[1:])
            total_m += best_leg_m
            cur = best_node
            visited.append(best_node)
            remaining.remove(best_node)

        # Finally go to END
        try:
            tail = nx.shortest_path(self.G, cur, end_node, weight='length')
            route_nodes.extend(tail[1:])
            total_m += nx.shortest_path_length(self.G, cur, end_node, weight='length')
        except nx.NetworkXNoPath:
            # If unreachable, just keep what we have
            pass

        return route_nodes, total_m, [], visited

    # ---------------------------- plotting ----------------------------
    def _get_city_boundary(self):
        """Build (or return cached) convex hull of nodes to use as a clean boundary box."""
        if getattr(self, "_city_boundary", None) is None:
            gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in self.nodes[['lon','lat']].to_numpy()],
                                   crs="EPSG:4326").to_crs(epsg=3857)
            self._city_boundary = gdf.unary_union.convex_hull
            self._city_boundary = gpd.GeoSeries([self._city_boundary], crs="EPSG:3857").to_crs(epsg=4326)
        return self._city_boundary

    def visualize_route(self, route_nodes, delivery_points, visited_deliveries, title,
                        start_depot=None, end_depot=None):
        fig, ax = plt.subplots(figsize=(14, 16), dpi=180)
        self._get_city_boundary().boundary.plot(ax=ax, linewidth=2.0, edgecolor='#7B7A7A', alpha=0.9, zorder=0)
        self.edges.plot(ax=ax, linewidth=0.35, edgecolor='#B0B0B0', alpha=0.85, zorder=1)

        # Unvisited
        unv = delivery_points[~delivery_points['nearest_node'].isin(visited_deliveries)]
        ax.scatter(unv['lon'], unv['lat'], s=16, c='#E6F0FF', edgecolors="#2367AC", linewidths=0.5,
                   alpha=0.6, zorder=2, label='Unvisited')

        # Visited
        vis = delivery_points[delivery_points['nearest_node'].isin(visited_deliveries)]
        ax.scatter(vis['lon'], vis['lat'], s=20, c='#1F77B4', edgecolors="#114E7A", linewidths=0.6,
                   alpha=0.95, zorder=3, label='Visited')

        # Draw the route polyline from node geometries
        coords = [(self.nodes.loc[n, 'lon'], self.nodes.loc[n, 'lat']) for n in route_nodes if n in self.nodes.index]
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, linewidth=1.2, alpha=0.9, zorder=4)

        # Depots
        for code, meta in self.depots.items():
            ax.scatter([meta['lon']], [meta['lat']], s=90, c=meta['color'], alpha=0.9, zorder=5)
            ax.text(meta['lon']+0.002, meta['lat']+0.002, meta['name'], fontsize=10, weight='bold')

        # Title & save
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(f"{title}.png", dpi=320, bbox_inches='tight')
        plt.close()
