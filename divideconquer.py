import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import psutil
import os
import seaborn as sns
from functools import lru_cache

# Set up plotting style
plt.style.use('default')
ox.settings.use_cache = True
ox.settings.log_console = False

class DivideConquerRouter:
    def __init__(self, graph_file, distance_weight=0.3, delivery_weight=0.7):
        """
        Initialize the Divide and Conquer Router with weighted objectives
        """
        self.distance_weight = distance_weight
        self.delivery_weight = delivery_weight
        
        print("Loading graph...")
        self.G = ox.load_graphml(graph_file)
        
        print("Converting graph to GeoDataFrames...")
        self.nodes, self.edges = ox.graph_to_gdfs(self.G)
        
        # Ensure 'x' and 'y' are numeric
        print("Nodes head:\n", self.nodes[['x', 'y']].head())
        print("Nodes dtypes:\n", self.nodes[['x', 'y']].dtypes)
        self.nodes['x'] = pd.to_numeric(self.nodes['x'], errors='coerce')
        self.nodes['y'] = pd.to_numeric(self.nodes['y'], errors='coerce')
        self.nodes = self.nodes.dropna(subset=['x', 'y']).copy()
        print("Cleaned nodes shape:", self.nodes.shape)
        
        print("Creating KDTree for nearest node search...")
        self.kdtree = KDTree(self.nodes[['y', 'x']])
        
        # Define depot locations (Central, Eastern, Western, Northern, Southern)
        self.depots = {
            'C': {'lat': 41.85, 'lon': -87.68, 'color': 'red'},
            'N': {'lat': 42.00, 'lon': -87.70, 'color': 'purple'},
            'S': {'lat': 41.70, 'lon': -87.63, 'color': 'orange'},
            'E': {'lat': 41.8914, 'lon': -87.6099, 'color': 'blue'},
            'W': {'lat': 41.8800, 'lon': -87.7500, 'color': 'green'},
        }
        
        # Assign each depot to the nearest node in the graph
        self.depot_nodes = {}
        for name, depot_info in self.depots.items():
            lat = depot_info['lat']
            lon = depot_info['lon']
            dist, idx = self.kdtree.query([[lat, lon]])
            nearest_node = self.nodes.iloc[idx[0]].name
            self.depot_nodes[name] = nearest_node
            print(f"Depot {name} assigned to node: {nearest_node}")
        
        # Cache for Dijkstra results
        self.dijkstra_cache = lru_cache(maxsize=1000)(self._cached_dijkstra_distances)
        
        print("Router initialized successfully!")

    def load_and_preprocess_points(self, delivery_points_file):
        """
        Load and preprocess delivery points
        """
        print(f"Loading delivery points from {delivery_points_file}...")
        delivery_points = pd.read_csv(delivery_points_file)
        delivery_points['lat'] = pd.to_numeric(delivery_points['lat'], errors='coerce')
        delivery_points['lon'] = pd.to_numeric(delivery_points['lon'], errors='coerce')
        delivery_points = delivery_points.dropna(subset=['lat', 'lon'])
        print("Delivery points shape after cleaning:", delivery_points.shape)
        print("Delivery points dtypes:\n", delivery_points[['lat', 'lon']].dtypes)
        
        print("Precomputing nearest nodes for delivery points...")
        delivery_points['nearest_node'] = delivery_points.apply(
            lambda row: self.find_nearest_node(row['lat'], row['lon']), axis=1)
        
        return delivery_points

    def _cached_dijkstra_distances(self, source_node):
        """
        Compute and cache Dijkstra distances from a source node to all other nodes
        """
        return nx.single_source_dijkstra_path_length(self.G, source_node, weight='length')

    def get_region_for_point(self, lat, lon):
        """
        Determine which region a point belongs to based on coordinates
        """
        if lat > 41.90: return 'N'
        elif lat < 41.80: return 'S'
        elif lon < -87.70: return 'W'
        elif lon > -87.60: return 'E'
        else: return 'C'

    def find_nearest_node(self, lat, lon):
        """
        Find the nearest graph node to given coordinates
        """
        dist, idx = self.kdtree.query([[lat, lon]])
        return self.nodes.iloc[idx[0]].name

    def quad_tree_decomposition(self, points, max_points_per_region=25):
        """
        Perform quad-tree decomposition to partition delivery points
        """
        if len(points) <= max_points_per_region:
            return [points]
        
        min_lat, max_lat = points['lat'].min(), points['lat'].max()
        min_lon, max_lon = points['lon'].min(), points['lon'].max()
        
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2
        
        quadrants = [
            points[(points['lat'] <= mid_lat) & (points['lon'] <= mid_lon)],
            points[(points['lat'] <= mid_lat) & (points['lon'] > mid_lon)],
            points[(points['lat'] > mid_lat) & (points['lon'] <= mid_lon)],
            points[(points['lat'] > mid_lat) & (points['lon'] > mid_lon)]
        ]
        
        regions = []
        for quadrant in quadrants:
            if len(quadrant) > 0:
                regions.extend(self.quad_tree_decomposition(quadrant, max_points_per_region))
        
        return regions

    def find_optimal_path_with_points(self, points, start_node, end_node, max_detour_ratio=0.5):
        """
        Find optimal path from start_node to end_node considering both distance and delivery points
        Uses weighted objective: distance_weight * distance + delivery_weight * (1 - delivery_coverage)
        """
        # Compute shortest path from start to end
        try:
            shortest_path = nx.shortest_path(self.G, start_node, end_node, weight='length')
            shortest_dist = nx.shortest_path_length(self.G, start_node, end_node, weight='length')
        except nx.NetworkXNoPath:
            return [start_node], 0, 0
        
        # Get delivery point nodes
        point_nodes = set(points['nearest_node'].unique())
        total_points = len(point_nodes)
        
        # Calculate maximum allowed detour
        max_detour = shortest_dist * max_detour_ratio
        
        # Find candidate nodes within detour distance of shortest path nodes
        candidate_nodes = set()
        for path_node in shortest_path:
            try:
                distances = self.dijkstra_cache(path_node)
                for node in point_nodes:
                    if node in distances and distances[node] <= max_detour:
                        candidate_nodes.add(node)
            except nx.NetworkXNoPath:
                continue
        
        # Initialize route with start node
        route = [start_node]
        total_distance = 0
        covered_points = set()
        
        # If start node is a delivery point, mark it as covered
        if start_node in point_nodes:
            covered_points.add(start_node)
        
        current = start_node
        
        # Continue until we reach the end node
        while current != end_node:
            # Get distances from current position
            try:
                distances_from_current = self.dijkstra_cache(current)
                distances_to_end = self.dijkstra_cache(end_node)
            except nx.NetworkXNoPath:
                break
            
            # Find the best next node based on weighted objective
            best_node = None
            best_score = float('-inf')
            
            # Consider all candidate nodes and the end node
            candidates = list(candidate_nodes - covered_points) + [end_node]
            
            for node in candidates:
                if node not in distances_from_current or node not in distances_to_end:
                    continue
                
                # Calculate detour for this node
                detour = (distances_from_current[node] +
                          distances_to_end[node] -
                          distances_from_current[end_node])
                
                # Skip if detour is too large
                if detour > max_detour and node != end_node:
                    continue
                
                # Calculate the objective function value
                new_points_covered = 1 if node in point_nodes and node not in covered_points else 0
                
                # Normalize distance and delivery components
                norm_distance = distances_from_current[node] / shortest_dist
                norm_delivery = new_points_covered / total_points if total_points > 0 else 0
                
                # Calculate weighted score (we want to minimize distance and maximize delivery)
                score = (self.delivery_weight * norm_delivery -
                         self.distance_weight * norm_distance)
                
                if score > best_score:
                    best_score = score
                    best_node = node
            
            # If no suitable node found, go directly to end
            if best_node is None:
                try:
                    path = nx.shortest_path(self.G, current, end_node, weight='length')
                    path_dist = nx.shortest_path_length(self.G, current, end_node, weight='length')
                    route.extend(path[1:])
                    total_distance += path_dist
                    break
                except nx.NetworkXNoPath:
                    break
            
            # Add best node to route
            try:
                path = nx.shortest_path(self.G, current, best_node, weight='length')
                path_dist = nx.shortest_path_length(self.G, current, best_node, weight='length')
                route.extend(path[1:])
                total_distance += path_dist
                
                # Update covered points if this is a delivery node
                if best_node in point_nodes and best_node not in covered_points:
                    covered_points.add(best_node)
                
                # Remove from candidate nodes if we've visited it
                if best_node in candidate_nodes:
                    candidate_nodes.discard(best_node)
                
                current = best_node
            except nx.NetworkXNoPath:
                if best_node in candidate_nodes:
                    candidate_nodes.discard(best_node)
                continue
        
        return route, total_distance, len(covered_points)

    def solve_tsp_nearest_neighbor(self, points, start_node, end_node=None):
        """
        Solve TSP using nearest neighbor (used for Experiments 2, 3, and sample viz)
        """
        if end_node is None:
            end_node = start_node
        
        point_nodes = set(points['nearest_node'].unique())
        
        route = [start_node]
        unvisited = point_nodes.copy()
        total_distance = 0
        
        if start_node != end_node:
            unvisited.add(end_node)
        
        current = start_node
        while unvisited:
            try:
                distances = self.dijkstra_cache(current)
            except nx.NetworkXNoPath:
                break
            
            nearest = None
            min_score = float('inf')
            for node in unvisited:
                if node in distances:
                    dist_to_node = distances[node]
                    node_lat = self.nodes.loc[node]['y']
                    score = dist_to_node - 0.5 * node_lat
                    if score < min_score:
                        min_score = score
                        nearest = node
            
            if nearest is None:
                break
                
            route.append(nearest)
            unvisited.remove(nearest)
            total_distance += distances[nearest]
            current = nearest
        
        return route, total_distance

    def solve_region_routes(self, regions, start_depot, end_depot=None):
        """
        Solve routes for each region and connect them (used for Experiments 2 and 3)
        """
        if end_depot is None:
            end_depot = start_depot
        
        start_node = self.depot_nodes[start_depot]
        end_node = self.depot_nodes[end_depot]
        
        region_routes = []
        for region in regions:
            centroid_lat = region['lat'].mean()
            centroid_lon = region['lon'].mean()
            
            route, distance = self.solve_tsp_nearest_neighbor(region, start_node, end_node)
            region_routes.append({
                'route': route,
                'distance': distance,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'points': region
            })
        
        region_order = self.order_regions(region_routes, start_node, end_node)
        
        complete_route = [start_node]
        total_distance = 0
        
        for i, idx in enumerate(region_order):
            region = region_routes[idx]
            
            if i == 0:
                try:
                    path = nx.shortest_path(self.G, complete_route[-1], region['route'][0], weight='length')
                    dist = nx.shortest_path_length(self.G, complete_route[-1], region['route'][0], weight='length')
                    complete_route.extend(path[1:])
                    total_distance += dist
                except:
                    pass
            else:
                prev_region = region_routes[region_order[i-1]]
                try:
                    path = nx.shortest_path(self.G, prev_region['route'][-1], region['route'][0], weight='length')
                    dist = nx.shortest_path_length(self.G, prev_region['route'][-1], region['route'][0], weight='length')
                    complete_route.extend(path[1:])
                    total_distance += dist
                except:
                    pass
            
            complete_route.extend(region['route'][1:])
            total_distance += region['distance']
        
        return complete_route, total_distance, region_routes

    def order_regions(self, regions, start_node, end_node):
        """
        Order regions for South-to-North progression (used for Experiments 2 and 3)
        """
        region_indices = list(range(len(regions)))
        region_indices.sort(key=lambda i: regions[i]['centroid_lat'])
        return region_indices

    def calculate_route_distance(self, route):
        """
        Calculate total distance of a route
        """
        total_distance = 0
        for i in range(len(route) - 1):
            try:
                dist = nx.shortest_path_length(self.G, route[i], route[i+1], weight='length')
                total_distance += dist
            except:
                continue
        return total_distance

    def visualize_quad_tree(self, points, regions, title="Quad-Tree Decomposition"):
        """
        Visualize quad-tree decomposition with regional colors and depot letters
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        self.edges.plot(ax=ax, linewidth=0.3, edgecolor='lightgray', alpha=0.5)
        
        region_colors = {
            'N': 'purple', 'S': 'orange', 'E': 'blue',
            'W': 'green', 'C': 'red'
        }
        
        for _, point in points.iterrows():
            region = self.get_region_for_point(point['lat'], point['lon'])
            point_color = region_colors.get(region, 'gray')
            ax.scatter(point['lon'], point['lat'], c=point_color, s=30, alpha=0.7)
        
        region_colors_list = ['cyan', 'magenta', 'yellow', 'lime', 'pink', 'brown', 'olive', 'navy']
        for i, region in enumerate(regions):
            min_lat, max_lat = region['lat'].min(), region['lat'].max()
            min_lon, max_lon = region['lon'].min(), region['lon'].max()
            
            rect = patches.Rectangle(
                (min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                linewidth=2, edgecolor=region_colors_list[i % len(region_colors_list)],
                facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
        
        for name, depot_info in self.depots.items():
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax.scatter(lon, lat, c=depot_color, s=200, marker='o', edgecolors='black',
                       linewidth=2)
            ax.text(lon, lat, name[0], fontsize=12, ha='center', va='center',
                    color='white', weight='bold')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='North'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='South'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='East'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='West'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Central')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_route(self, route, points, title="Delivery Route"):
        """
        Visualize the single computed route with depot letters and legend
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        self.edges.plot(ax=ax, linewidth=0.3, edgecolor='lightgray', alpha=0.4)
        
        ax.scatter(points['lon'], points['lat'], c='blue', s=40, alpha=0.8, label='Delivery Points')
        
        route_coords = []
        for node in route:
            if node in self.nodes.index:
                node_data = self.nodes.loc[node]
                route_coords.append((node_data['x'], node_data['y']))
        
        if route_coords:
            lons, lats = zip(*route_coords)
            ax.plot(lons, lats, 'k-', linewidth=2, alpha=0.8, label='Direct Route')
        
        for name in ['S', 'N']:
            depot_info = self.depots[name]
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax.scatter(lon, lat, c=depot_color, s=200, marker='o', edgecolors='black',
                       linewidth=2, label=f"{name} Depot")
            ax.text(lon, lat, name[0], fontsize=12, ha='center', va='center',
                    color='white', weight='bold')
        
        ax.set_title(f"{title}\nTotal Distance: {self.calculate_route_distance(route):.2f}m",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_route_with_regions(self, route, points, region_data, title="Delivery Route"):
        """
        Visualize the computed route with regional coloring
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        self.edges.plot(ax=ax, linewidth=0.3, edgecolor='lightgray', alpha=0.4)
        
        region_colors = plt.cm.tab10(np.linspace(0, 1, len(region_data)))
        
        for i, region_info in enumerate(region_data):
            region_points = region_info['points']
            for _, point in region_points.iterrows():
                ax.scatter(point['lon'], point['lat'], c=[region_colors[i]], s=40, alpha=0.8)
        
        route_coords = []
        for node in route:
            if node in self.nodes.index:
                node_data = self.nodes.loc[node]
                route_coords.append((node_data['x'], node_data['y']))
        
        if route_coords:
            lons, lats = zip(*route_coords)
            ax.plot(lons, lats, 'k-', linewidth=2, alpha=0.8, label='Route')
        
        for name, depot_info in self.depots.items():
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax.scatter(lon, lat, c=depot_color, s=200, marker='o', edgecolors='black',
                       linewidth=2)
            ax.text(lon, lat, name[0], fontsize=12, ha='center', va='center',
                    color='white', weight='bold')
        
        ax.set_title(f"{title}\nTotal Distance: {self.calculate_route_distance(route):.2f}m",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_priority_distribution(self, points, title="Priority Distribution"):
        """
        Visualize priority distribution across regions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
        points['region'] = points.apply(lambda x: self.get_region_for_point(x['lat'], x['lon']), axis=1)
    
        priority_data = points.groupby(['region', 'priority']).size().unstack(fill_value=0)
        priority_data.plot(kind='bar', ax=ax1, color=['red', 'orange', 'green'])
        ax1.set_title('Priority Distribution by Region', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Number of Deliveries')
        ax1.legend(['High', 'Medium', 'Low'])
        ax1.tick_params(axis='x', rotation=45)
    
        self.edges.plot(ax=ax2, linewidth=0.3, edgecolor='lightgray', alpha=0.5)
        
        colors = ['red', 'orange', 'green']
        for priority, color in zip([1, 2, 3], colors):
            priority_points = points[points['priority'] == priority]
            ax2.scatter(priority_points['lon'], priority_points['lat'],
                        c=color, s=20, label=f'Priority {priority}', alpha=0.7)
    
        for name, depot_info in self.depots.items():
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax2.scatter(lon, lat, c=depot_color, s=100, marker='o', edgecolors='black')
            ax2.text(lon, lat, name[0], fontsize=10, ha='center', va='center',
                     color='white', weight='bold')
    
        ax2.set_title('Geographic Distribution of Priorities', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.legend()
    
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()