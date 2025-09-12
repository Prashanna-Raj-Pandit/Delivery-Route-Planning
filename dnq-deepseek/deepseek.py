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

# Set up plotting style
plt.style.use('default')
ox.settings.use_cache = True
ox.settings.log_console = False

class DivideConquerRouter:
    def __init__(self, graph_file, delivery_points_file):
        """
        Initialize the Divide and Conquer Router
        """
        print("Loading graph...")
        self.G = ox.load_graphml(graph_file)
       
        print("Loading delivery points...")
        self.delivery_points = pd.read_csv(delivery_points_file)
       
        # Clean delivery points lat/lon to numeric
        self.delivery_points['lat'] = pd.to_numeric(self.delivery_points['lat'], errors='coerce')
        self.delivery_points['lon'] = pd.to_numeric(self.delivery_points['lon'], errors='coerce')
        self.delivery_points = self.delivery_points.dropna(subset=['lat', 'lon'])
        print("Delivery points shape after cleaning:", self.delivery_points.shape)
        print("Delivery points dtypes:\n", self.delivery_points[['lat', 'lon']].dtypes)
       
        print("Converting graph to GeoDataFrames...")
        self.nodes, self.edges = ox.graph_to_gdfs(self.G)
        
        # Ensure 'x' and 'y' are numeric
        print("Nodes head:\n", self.nodes[['x', 'y']].head())
        print("Nodes dtypes:\n", self.nodes[['x', 'y']].dtypes)
        self.nodes['x'] = pd.to_numeric(self.nodes['x'], errors='coerce')
        self.nodes['y'] = pd.to_numeric(self.nodes['y'], errors='coerce')
        self.nodes = self.nodes.dropna(subset=['x', 'y']).copy()  # Drop invalid rows
        print("Cleaned nodes shape:", self.nodes.shape)
       
        print("Creating KDTree for nearest node search...")
        self.kdtree = KDTree(self.nodes[['y', 'x']])
       
        # Define depot locations (Central, Eastern, Western, Northern, Southern)
        self.depots = {
            'C': {'lat': 41.8781, 'lon': -87.6600, 'color': 'red'},
            'E': {'lat': 41.8914, 'lon': -87.6099, 'color': 'blue'},
            'W': {'lat': 41.8800, 'lon': -87.7500, 'color': 'green'},
            'N': {'lat': 41.9476, 'lon': -87.7000, 'color': 'purple'},
            'S': {'lat': 41.7000, 'lon': -87.6420, 'color': 'orange'}
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
       
        print("Router initialized successfully!")
   
    def get_region_for_point(self, lat, lon):
        """
        Determine which region a point belongs to based on coordinates
        """
        if lat > 41.90: return 'N'  # North
        elif lat < 41.80: return 'S'  # South
        elif lon < -87.70: return 'W'  # West
        elif lon > -87.60: return 'E'  # East
        else: return 'C'  # Central
   
    def find_nearest_node(self, lat, lon):
        """
        Find the nearest graph node to given coordinates
        """
        dist, idx = self.kdtree.query([[lat, lon]])
        return self.nodes.iloc[idx[0]].name
   
    def quad_tree_decomposition(self, points, max_points_per_region=50):
        """
        Perform quad-tree decomposition to partition delivery points
        """
        if len(points) <= max_points_per_region:
            return [points]
       
        # Find bounding box of points
        min_lat, max_lat = points['lat'].min(), points['lat'].max()
        min_lon, max_lon = points['lon'].min(), points['lon'].max()
       
        # Calculate midpoints
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2
       
        # Divide into quadrants
        quadrants = [
            points[(points['lat'] <= mid_lat) & (points['lon'] <= mid_lon)],  # SW
            points[(points['lat'] <= mid_lat) & (points['lon'] > mid_lon)],  # SE
            points[(points['lat'] > mid_lat) & (points['lon'] <= mid_lon)],  # NW
            points[(points['lat'] > mid_lat) & (points['lon'] > mid_lon)]  # NE
        ]
       
        # Recursively decompose each quadrant
        regions = []
        for quadrant in quadrants:
            if len(quadrant) > 0:
                regions.extend(self.quad_tree_decomposition(quadrant, max_points_per_region))
       
        return regions
   
    def solve_tsp_nearest_neighbor(self, points, start_node, end_node=None):
        """
        Solve TSP using nearest neighbor heuristic
        """
        if end_node is None:
            end_node = start_node
       
        # Convert delivery points to graph nodes
        point_nodes = []
        for _, point in points.iterrows():
            node_id = self.find_nearest_node(point['lat'], point['lon'])
            point_nodes.append(node_id)
       
        # Initialize route with start node
        route = [start_node]
        unvisited = set(point_nodes)
        total_distance = 0
       
        if start_node != end_node:
            unvisited.add(end_node)
       
        # Build route using nearest neighbor
        current = start_node
        while unvisited:
            nearest = None
            min_dist = float('inf')
           
            for node in unvisited:
                try:
                    dist = nx.shortest_path_length(self.G, current, node, weight='length')
                    if dist < min_dist:
                        min_dist = dist
                        nearest = node
                except:
                    continue
           
            if nearest is None:
                break
               
            route.append(nearest)
            unvisited.remove(nearest)
            total_distance += min_dist
            current = nearest
       
        return route, total_distance
   
    def solve_region_routes(self, regions, start_depot, end_depot=None):
        """
        Solve routes for each region and connect them
        """
        if end_depot is None:
            end_depot = start_depot
       
        start_node = self.depot_nodes[start_depot]
        end_node = self.depot_nodes[end_depot]
       
        # Solve TSP for each region
        region_routes = []
        for region in regions:
            centroid_lat = region['lat'].mean()
            centroid_lon = region['lon'].mean()
           
            route, distance = self.solve_tsp_nearest_neighbor(region, start_node)
            region_routes.append({
                'route': route,
                'distance': distance,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'points': region
            })
       
        # Order regions by proximity to start depot
        region_order = self.order_regions(region_routes, start_node, end_node)
       
        # Connect regions
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
        Order regions for efficient traversal
        """
        centroids = [(r['centroid_lat'], r['centroid_lon']) for r in regions]
        centroid_nodes = []
       
        for lat, lon in centroids:
            node_id = self.find_nearest_node(lat, lon)
            centroid_nodes.append(node_id)
       
        ordered_indices = []
        unvisited = set(range(len(regions)))
        current = start_node
       
        while unvisited:
            nearest_idx = None
            min_dist = float('inf')
           
            for idx in unvisited:
                try:
                    dist = nx.shortest_path_length(self.G, current, centroid_nodes[idx], weight='length')
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx
                except:
                    continue
           
            if nearest_idx is None:
                break
               
            ordered_indices.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current = centroid_nodes[nearest_idx]
       
        return ordered_indices
   
    def run_experiment_1(self, sizes=[100, 200, 300, 500]):
        """
        Run scalability experiment
        """
        results = []
       
        for size in sizes:
            print(f"Running experiment for size {size}...")
            sample_points = self.delivery_points.sample(n=size, random_state=42)
            regions = self.quad_tree_decomposition(sample_points, max_points_per_region=25)
           
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
           
            start_time = time.time()
            route, distance, region_data = self.solve_region_routes(regions, 'C')
            computation_time = time.time() - start_time
           
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
           
            visited_nodes = len(set(route))
           
            delivery_nodes_covered = len([node for node in route if node in
                set(self.find_nearest_node(p['lat'], p['lon']) for _, p in sample_points.iterrows())])
           
            results.append({
                'size': size,
                'computation_time': computation_time,
                'memory_used': memory_used,
                'distance': distance,
                'visited_nodes': visited_nodes,
                'delivery_points_covered': delivery_nodes_covered,
                'regions': len(regions)
            })
           
            print(f"Size {size}: {computation_time:.2f}s, {distance:.2f}m, {memory_used:.2f}MB")
       
        return pd.DataFrame(results)
   
    def run_experiment_2(self):
        """
        Run multi-depot experiment
        """
        results = []
        depot_pairs = [('C', 'E'), ('S', 'E'), ('W', 'S'), ('E', 'W'), ('N', 'S')]
       
        for start_depot, end_depot in depot_pairs:
            print(f"Running experiment for {start_depot}-{end_depot}...")
            regions = self.quad_tree_decomposition(self.delivery_points, max_points_per_region=50)
           
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
           
            start_time = time.time()
            route, distance, region_data = self.solve_region_routes(regions, start_depot, end_depot)
            computation_time = time.time() - start_time
           
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
           
            visited_nodes = len(set(route))
           
            delivery_nodes_covered = len([node for node in route if node in
                set(self.find_nearest_node(p['lat'], p['lon']) for _, p in self.delivery_points.iterrows())])
           
            results.append({
                'depot_pair': f"{start_depot}-{end_depot}",
                'computation_time': computation_time,
                'memory_used': memory_used,
                'distance': distance,
                'visited_nodes': visited_nodes,
                'delivery_points_covered': delivery_nodes_covered
            })
           
            # Generate route visualization for each depot pair
            self.visualize_route_with_regions(route, self.delivery_points, region_data,
                                            f"Route_{start_depot}_to_{end_depot}")
           
            print(f"{start_depot}-{end_depot}: {computation_time:.2f}s, {distance:.2f}m")
       
        return pd.DataFrame(results)
   
    def run_experiment_3(self):
        """
        Run priority-aware delivery experiment
        """
        results = []
        priority_distributions = [
            {'1': 0.33, '2': 0.34, '3': 0.33},
            {'1': 0.6, '2': 0.3, '3': 0.1},
            {'1': 0.1, '2': 0.3, '3': 0.6}
        ]
       
        for i, dist in enumerate(priority_distributions):
            print(f"Running experiment for distribution {i+1}...")
            n = len(self.delivery_points)
            n1 = int(n * dist['1'])
            n2 = int(n * dist['2'])
            n3 = n - n1 - n2
           
            priorities = [1] * n1 + [2] * n2 + [3] * n3
            random.shuffle(priorities)
           
            points = self.delivery_points.copy()
            points['priority'] = priorities[:n]
           
            regions = self.quad_tree_decomposition(points, max_points_per_region=50)
            route, distance, region_data = self.solve_region_routes(regions, 'C')
           
            # Priority analysis
            high_priority = sum(1 for p in points['priority'] if p == 1)
            med_priority = sum(1 for p in points['priority'] if p == 2)
            low_priority = sum(1 for p in points['priority'] if p == 3)
           
            results.append({
                'distribution': f"Dist_{i+1}",
                'distance': distance,
                'high_priority_count': high_priority,
                'med_priority_count': med_priority,
                'low_priority_count': low_priority,
                'high_percentage': high_priority/n*100,
                'med_percentage': med_priority/n*100,
                'low_percentage': low_priority/n*100
            })
           
            # Generate visualization for each distribution
            self.visualize_priority_distribution(points, f"Priority_Distribution_{i+1}")
           
            print(f"Distribution {i+1}: {distance:.2f}m")
       
        return pd.DataFrame(results)
   
    def visualize_quad_tree(self, points, regions, title="Quad-Tree Decomposition"):
        """
        Visualize quad-tree decomposition with regional colors
        """
        fig, ax = plt.subplots(figsize=(14, 12))
       
        # Plot the street network
        self.edges.plot(ax=ax, linewidth=0.3, edgecolor='lightgray', alpha=0.5)
       
        # Define region colors
        region_colors = {
            'N': 'purple', 'S': 'orange', 'E': 'blue',
            'W': 'green', 'C': 'red'
        }
       
        # Plot delivery points with color by region
        for _, point in points.iterrows():
            region = self.get_region_for_point(point['lat'], point['lon'])
            point_color = region_colors.get(region, 'gray')
            ax.scatter(point['lon'], point['lat'], c=point_color, s=30, alpha=0.7)
       
        # Draw region boundaries with different colors
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
       
        # Add depots with their colors (FIXED: Use explicit dict access)
        for name in self.depots:
            depot_info = self.depots[name]
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax.scatter(lon, lat, c=depot_color, s=200, marker='s', edgecolors='black',
                       linewidth=2, label=f'Depot {name}')
       
        # Add legend for regions
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
   
    def visualize_route_with_regions(self, route, points, region_data, title="Delivery Route"):
        """
        Visualize the computed route with regional coloring
        """
        fig, ax = plt.subplots(figsize=(14, 12))
       
        # Plot the street network
        self.edges.plot(ax=ax, linewidth=0.3, edgecolor='lightgray', alpha=0.4)
       
        # Define region colors
        region_colors = plt.cm.tab10(np.linspace(0, 1, len(region_data)))
       
        # Plot each region with different color
        for i, region_info in enumerate(region_data):
            region_points = region_info['points']
            for _, point in region_points.iterrows():
                ax.scatter(point['lon'], point['lat'], c=[region_colors[i]], s=40, alpha=0.8)
       
        # Plot the route
        route_coords = []
        for node in route:
            if node in self.nodes.index:
                node_data = self.nodes.loc[node]
                route_coords.append((node_data['x'], node_data['y']))
       
        if route_coords:
            lons, lats = zip(*route_coords)
            ax.plot(lons, lats, 'k-', linewidth=2, alpha=0.8, label='Route')
       
        # Add depots (FIXED: Use explicit dict access to avoid unpacking issue)
        for name in self.depots:
            depot_info = self.depots[name]
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax.scatter(lon, lat, c=depot_color, s=200, marker='s', edgecolors='black',
                       linewidth=2, label=f'Depot {name}')
       
        ax.set_title(f"{title}\nTotal Distance: {self.calculate_route_distance(route):.2f}m",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.legend()
       
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()
   
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
   
    def visualize_priority_distribution(self, points, title="Priority Distribution"):
        """
        Visualize priority distribution across regions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
        # Plot 1: Priority distribution by region
        points['region'] = points.apply(lambda x: self.get_region_for_point(x['lat'], x['lon']), axis=1)
    
        priority_data = points.groupby(['region', 'priority']).size().unstack(fill_value=0)
        priority_data.plot(kind='bar', ax=ax1, color=['red', 'orange', 'green'])
        ax1.set_title('Priority Distribution by Region', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Number of Deliveries')
        ax1.legend(['High', 'Medium', 'Low'])
        ax1.tick_params(axis='x', rotation=45)
    
        # Plot 2: Geographic distribution of priorities
        # Add street network as background
        self.edges.plot(ax=ax2, linewidth=0.3, edgecolor='lightgray', alpha=0.5)
        
        colors = ['red', 'orange', 'green']
        for priority, color in zip([1, 2, 3], colors):
            priority_points = points[points['priority'] == priority]
            ax2.scatter(priority_points['lon'], priority_points['lat'],
                        c=color, s=20, label=f'Priority {priority}', alpha=0.7)
    
        # Add depots
        for name in self.depots:
            depot_info = self.depots[name]
            lat = depot_info['lat']
            lon = depot_info['lon']
            depot_color = depot_info['color']
            ax2.scatter(lon, lat, c=depot_color, s=100, marker='s', edgecolors='black', label=f'Depot {name}')
    
        ax2.set_title('Geographic Distribution of Priorities', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.legend()
    
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()
   
    def create_comprehensive_analysis_plots(self, exp1_results, exp2_results, exp3_results):
        """
        Create comprehensive analysis plots with multiple visualization types
        """
        # 1. Scalability Analysis (Multiple subplots)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       
        # Computation time
        axes[0, 0].plot(exp1_results['size'], exp1_results['computation_time'], 'o-', linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Number of Delivery Points')
        axes[0, 0].set_ylabel('Computation Time (s)')
        axes[0, 0].set_title('Computation Time vs Problem Size')
        axes[0, 0].grid(True, alpha=0.3)
       
        # Memory usage
        axes[0, 1].plot(exp1_results['size'], exp1_results['memory_used'], 's-', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Number of Delivery Points')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage vs Problem Size')
        axes[0, 1].grid(True, alpha=0.3)
       
        # Distance
        axes[0, 2].plot(exp1_results['size'], exp1_results['distance'], '^-', linewidth=2, color='green')
        axes[0, 2].set_xlabel('Number of Delivery Points')
        axes[0, 2].set_ylabel('Total Distance (m)')
        axes[0, 2].set_title('Total Distance vs Problem Size')
        axes[0, 2].grid(True, alpha=0.3)
       
        # Regions created
        axes[1, 0].plot(exp1_results['size'], exp1_results['regions'], 'd-', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Number of Delivery Points')
        axes[1, 0].set_ylabel('Number of Regions')
        axes[1, 0].set_title('Regions Created vs Problem Size')
        axes[1, 0].grid(True, alpha=0.3)
       
        # Points covered
        axes[1, 1].plot(exp1_results['size'], exp1_results['delivery_points_covered'], 'v-', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Number of Delivery Points')
        axes[1, 1].set_ylabel('Points Covered')
        axes[1, 1].set_title('Delivery Points Covered vs Problem Size')
        axes[1, 1].grid(True, alpha=0.3)
       
        # Efficiency ratio
        efficiency = exp1_results['delivery_points_covered'] / exp1_results['size']
        axes[1, 2].plot(exp1_results['size'], efficiency, '*-', linewidth=2, color='brown')
        axes[1, 2].set_xlabel('Number of Delivery Points')
        axes[1, 2].set_ylabel('Efficiency Ratio')
        axes[1, 2].set_title('Coverage Efficiency vs Problem Size')
        axes[1, 2].grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.savefig('Comprehensive_Scalability_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
       
        # 2. Multi-Depot Performance Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
       
        # Create performance matrix
        depot_pairs = [pair.split('-') for pair in exp2_results['depot_pair']]
        performance_data = exp2_results[['distance', 'computation_time', 'memory_used']].values
       
        im = ax.imshow(performance_data, cmap='viridis', aspect='auto')
       
        # Set labels
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Distance (m)', 'Time (s)', 'Memory (MB)'])
        ax.set_yticks(range(len(depot_pairs)))
        ax.set_yticklabels(exp2_results['depot_pair'])
       
        # Add text annotations
        for i in range(len(depot_pairs)):
            for j in range(3):
                text = ax.text(j, i, f'{performance_data[i, j]:.1f}',
                               ha="center", va="center", color="w", fontweight='bold')
       
        ax.set_title('Multi-Depot Performance Comparison', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('Multi_Depot_Performance_Heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
       
        # 3. Priority Analysis Radar Chart
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
       
        categories = ['High Priority', 'Medium Priority', 'Low Priority', 'Distance', 'Efficiency']
        N = len(categories)
       
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
       
        for i, row in exp3_results.iterrows():
            values = [
                row['high_percentage'],
                row['med_percentage'],
                row['low_percentage'],
                row['distance'] / 1000,  # Scale distance for better visualization
                100  # Fixed efficiency for reference
            ]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Distribution {i+1}')
            ax.fill(angles, values, alpha=0.1)
       
        ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)
        ax.set_title('Priority Distribution Analysis', size=16, fontweight='bold')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('Priority_Analysis_Radar.png', dpi=300, bbox_inches='tight')
        plt.show()
       
        # 4. Regional Distribution Analysis
        fig, ax = plt.subplots(figsize=(12, 8))
       
        # Analyze regional distribution
        self.delivery_points['region'] = self.delivery_points.apply(
            lambda x: self.get_region_for_point(x['lat'], x['lon']), axis=1)
       
        region_counts = self.delivery_points['region'].value_counts()
        colors = [self.depots[r]['color'] for r in region_counts.index]
       
        bars = ax.bar(region_counts.index, region_counts.values, color=colors, alpha=0.7)
       
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
       
        ax.set_title('Delivery Points Distribution by Region', fontsize=16, fontweight='bold')
        ax.set_xlabel('Region')
        ax.set_ylabel('Number of Delivery Points')
        ax.grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.savefig('Regional_Distribution_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    print("Starting Enhanced Divide and Conquer Route Optimization...")
    print("="*60)
   
    try:
        router = DivideConquerRouter(
            graph_file="../chicago_street_network.graphml",
            delivery_points_file="../delivery_points.csv"
        )
       
        # # Run Experiment 1: Scalability
        # print("\n" + "="*50)
        # print("RUNNING EXPERIMENT 1: SCALABILITY TEST")
        # print("="*50)
        # exp1_results = router.run_experiment_1(sizes=[100, 200, 300, 500, 700, 1000])
        # exp1_results.to_csv("experiment_1_results.csv", index=False)
        # print("✓ Experiment 1 completed! Results saved to experiment_1_results.csv")
       
        # Run Experiment 2: Multi-depot
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 2: MULTI-DEPOT ANALYSIS")
        print("="*50)
        exp2_results = router.run_experiment_2()
        exp2_results.to_csv("experiment_2_results.csv", index=False)
        print("✓ Experiment 2 completed! Results saved to experiment_2_results.csv")
       
        # Run Experiment 3: Priority-aware delivery
        print("\n" + "="*50)
        print("RUNNING EXPERIMENT 3: PRIORITY-AWARE DELIVERY")
        print("="*50)
        exp3_results = router.run_experiment_3()
        exp3_results.to_csv("experiment_3_results.csv", index=False)
        print("✓ Experiment 3 completed! Results saved to experiment_3_results.csv")
       
        # Generate comprehensive visualizations
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
       
        # Sample visualizations
        sample_points = router.delivery_points.sample(n=150, random_state=42)
        regions = router.quad_tree_decomposition(sample_points, max_points_per_region=25)
       
        print("Generating Quad-Tree visualization...")
        router.visualize_quad_tree(sample_points, regions, "Enhanced_Quad-Tree_Decomposition")
       
        print("Generating sample route visualization...")
        route, distance, region_data = router.solve_region_routes(regions, 'C')
        router.visualize_route_with_regions(route, sample_points, region_data, "Sample_Delivery_Route")
       
        print("Generating comprehensive analysis plots...")
        router.create_comprehensive_analysis_plots(exp1_results, exp2_results, exp3_results)
       
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGENERATED FILES:")
        print("1. experiment_1_results.csv")
        print("2. experiment_2_results.csv")
        print("3. experiment_3_results.csv")
        print("4. Enhanced_Quad-Tree_Decomposition.png")
        print("5. Sample_Delivery_Route.png")
        print("6. Route_C_to_E.png, Route_S_to_E.png, etc. (for each depot pair)")
        print("7. Priority_Distribution_1.png, Priority_Distribution_2.png, etc.")
        print("8. Comprehensive_Scalability_Analysis.png")
        print("9. Multi_Depot_Performance_Heatmap.png")
        print("10. Priority_Analysis_Radar.png")
        print("11. Regional_Distribution_Analysis.png")
        print("\n" + "="*60)
       
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check if the graph and delivery points files exist in the correct paths.")