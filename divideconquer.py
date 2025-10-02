# divideconquer.py
# Divide & Conquer router (Quad-Tree → regional NN → stitched street paths)
# Key features:
#  - Expands every hop to real road polylines (no jumpy lines)
#  - Feasibility-aware hops that preserve budget to finish at North
#  - Corridor filtering around S→N so it "doesn't go too far"
#  - Fixed Chicago bounds for consistent map frames

import math
from functools import lru_cache

import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree


class DivideConquerRouter:
    def __init__(self, graph_file: str,
                 distance_weight: float = 0.5,
                 delivery_weight: float = 0.5):
        print("Loading graph ...")
        self.G = ox.load_graphml(graph_file)
        self.nodes, self.edges = ox.graph_to_gdfs(self.G)

        # Ensure numeric lon/lat stored in nodes
        self.nodes['x'] = pd.to_numeric(self.nodes['x'], errors='coerce')  # lon
        self.nodes['y'] = pd.to_numeric(self.nodes['y'], errors='coerce')  # lat
        self.nodes = self.nodes.dropna(subset=['x', 'y']).copy()

        # KDTree on (lat, lon)
        print("Building KDTree ...")
        self.kdtree = KDTree(self.nodes[['y', 'x']])

        # Depots
        self.depots = {
            'C': {'lat': 41.85,   'lon': -87.68,   'color': 'red',    'name': 'Central'},
            'N': {'lat': 42.00,   'lon': -87.70,   'color': 'purple', 'name': 'North'},
            'S': {'lat': 41.70,   'lon': -87.63,   'color': 'orange', 'name': 'South'},
            'E': {'lat': 41.8914, 'lon': -87.6099, 'color': 'blue',   'name': 'East'},
            'W': {'lat': 41.8800, 'lon': -87.7500, 'color': 'green',  'name': 'West'},
        }
        self.depot_nodes = {}
        for k, d in self.depots.items():
            _, idx = self.kdtree.query([[d['lat'], d['lon']]])
            self.depot_nodes[k] = self.nodes.iloc[idx[0]].name
            print(f"Depot {k} ({d['name']}) -> node {self.depot_nodes[k]}")

        # Distance oracle cache
        self.dijkstra_cache = lru_cache(maxsize=2000)(self._cached_dijkstra_lengths)

        # Scaling for geometric projections
        self.avg_lat = float(self.nodes['y'].mean())
        self.cos_avg_lat = math.cos(math.radians(self.avg_lat))

        # Optional weights (used in other experiments)
        self.distance_weight = distance_weight
        self.delivery_weight = delivery_weight

        # City boundary cache (for plotting)
        self._city_boundary = None

        print("Router ready.")

    # ---------- helpers: distances / geometry ----------

    def _cached_dijkstra_lengths(self, source_node):
        """Returns dict: node -> shortest-path length (meters) from source_node."""
        return nx.single_source_dijkstra_path_length(self.G, source_node, weight='length')

    def get_scaled_pos(self, lat, lon):
        # Scale lon by cos(phi) to approximate meters in degrees
        return np.array([lat, lon * self.cos_avg_lat], dtype=float)

    def point_to_segment_distance(self, p_lat, p_lon, a_lat, a_lon, b_lat, b_lon):
        """Approx perpendicular distance (km) from P to segment A→B in lat/lon space."""
        p = self.get_scaled_pos(p_lat, p_lon)
        a = self.get_scaled_pos(a_lat, a_lon)
        b = self.get_scaled_pos(b_lat, b_lon)
        ab = b - a
        if float(np.dot(ab, ab)) == 0.0:
            d_scaled = float(np.linalg.norm(p - a))
        else:
            t = float(np.dot(p - a, ab) / np.dot(ab, ab))
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            d_scaled = float(np.linalg.norm(p - closest))
        return d_scaled * 111.0  # deg→km

    def find_nearest_node(self, lat, lon):
        _, idx = self.kdtree.query([[lat, lon]])
        return self.nodes.iloc[idx[0]].name

    # ---------- data ----------

    def load_and_preprocess_points(self, csv_path: str) -> pd.DataFrame:
        """Load delivery points; expects columns: lat, lon. Optional: priority."""
        df = pd.read_csv(csv_path)
        if 'priority' not in df.columns:
            df['priority'] = 2
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df = df.dropna(subset=['lat', 'lon']).copy()
        df['nearest_node'] = df.apply(
            lambda r: self.find_nearest_node(r['lat'], r['lon']), axis=1
        )
        return df

    # ---------- quad-tree divide ----------

    def quad_tree_decomposition(self, points: pd.DataFrame,
                                max_points_per_region: int = 20,
                                depth: int = 0, max_depth: int = 6):
        if len(points) == 0:
            return []
        if len(points) <= max_points_per_region or depth >= max_depth:
            return [{
                'points': points,
                'centroid_lat': float(points['lat'].mean()),
                'centroid_lon': float(points['lon'].mean()),
                'depth': depth,
                'size': int(len(points)),
                'bounds': {
                    'min_lat': float(points['lat'].min()),
                    'max_lat': float(points['lat'].max()),
                    'min_lon': float(points['lon'].min()),
                    'max_lon': float(points['lon'].max()),
                }
            }]

        min_lat, max_lat = float(points['lat'].min()), float(points['lat'].max())
        min_lon, max_lon = float(points['lon'].min()), float(points['lon'].max())
        mid_lat, mid_lon = 0.5*(min_lat+max_lat), 0.5*(min_lon+max_lon)

        quads = [
            points[(points['lat'] <= mid_lat) & (points['lon'] <= mid_lon)],
            points[(points['lat'] <= mid_lat) & (points['lon'] >  mid_lon)],
            points[(points['lat'] >  mid_lat) & (points['lon'] <= mid_lon)],
            points[(points['lat'] >  mid_lat) & (points['lon'] >  mid_lon)],
        ]
        regions = []
        for q in quads:
            if len(q) > 0:
                regions.extend(self.quad_tree_decomposition(
                    q, max_points_per_region, depth+1, max_depth
                ))
        return regions

    # ---------- regional "conquer" ----------

    def _priority_gain(self, region_points: pd.DataFrame, node) -> float:
        pr = region_points.loc[region_points['nearest_node'] == node, 'priority']
        if pr.empty:
            return 1.0
        p = int(pr.iloc[0])
        return {1: 0.5, 2: 0.3, 3: 0.2}.get(p, 1.0)

    def solve_region_tsp(self, region_points: pd.DataFrame,
                         start_node=None,
                         mode: str = "budget",
                         budget_m: float = float("inf"),
                         lam: float = 0.0,
                         global_end_node=None,
                         reserve_factor: float = 1.07):
        """
        Greedy nearest-neighbor within the region with feasibility:
        Only accept next hop if you can STILL afford the end-leg to North.
        Feasibility: total + d(current,next) + reserve_factor * d(next, END) <= budget_m
        Returns (route_order (delivery+start nodes), total_m_used, visited_delivery_nodes).
        """
        if len(region_points) == 0:
            return [], 0.0, set()

        delivery_nodes = set(region_points['nearest_node'].tolist())
        if start_node is None:
            start_node = next(iter(delivery_nodes))

        # Precompute distances from END to everyone (used as oracle)
        end_dists = self.dijkstra_cache(global_end_node) if global_end_node else {}

        unvisited = delivery_nodes.copy()
        current = start_node
        route_order = [current]
        total_m = 0.0
        visited_deliveries = set()
        if current in unvisited:
            unvisited.remove(current)
            visited_deliveries.add(current)

        while unvisited and total_m < budget_m:
            dists = self.dijkstra_cache(current)
            best = None
            best_m = float('inf')

            # Pick nearest FEASIBLE next node
            for n in list(unvisited):
                dm = dists.get(n)
                if dm is None:
                    continue
                end_m = end_dists.get(n, float('inf'))
                feasible = (total_m + dm + reserve_factor * end_m) <= budget_m
                if not feasible:
                    continue
                if dm < best_m:
                    best_m = dm
                    best = n

            if best is None:
                break

            if mode != "budget":
                gain = self._priority_gain(region_points, best)
                if (gain - lam * (best_m/1000.0)) <= 0.0:
                    break

            route_order.append(best)
            total_m += best_m
            unvisited.remove(best)
            visited_deliveries.add(best)
            current = best

        return route_order, total_m, visited_deliveries

    # ---------- combine: stitch regions with street paths ----------

    def _projection_score(self, region, s_lat, s_lon, e_lat, e_lon):
        v = self.get_scaled_pos(region['centroid_lat'], region['centroid_lon']) - self.get_scaled_pos(s_lat, s_lon)
        d = self.get_scaled_pos(e_lat, e_lon) - self.get_scaled_pos(s_lat, s_lon)
        return float(np.dot(v, d))

    def _shortest_path_nodes(self, u, v):
        """Return list of nodes along street path u→v (inclusive)."""
        if u == v:
            return [u]
        try:
            return nx.shortest_path(self.G, u, v, weight='length')
        except Exception:
            return [u, v]

    def _expand_seq_to_path(self, seq):
        """Expand [n0, n1, n2, ...] by stitching shortest paths between consecutive nodes."""
        if not seq:
            return []
        path = [seq[0]]
        for a, b in zip(seq, seq[1:]):
            seg = self._shortest_path_nodes(a, b)
            path.extend(seg[1:])  # avoid dup
        return path

    def _full_region_path_and_cost(self, region_points: pd.DataFrame, start_from_node):
        """
        Build a full NN order that visits ALL deliveries in the region,
        starting from the region node closest (by graph distance) to start_from_node.
        Returns (expanded_path_nodes, expanded_length_m, region_delivery_nodes_set).
        """
        reg_nodes = set(region_points['nearest_node'].tolist())
        if not reg_nodes:
            return [start_from_node], 0.0, set()

        # pick region start = closest region node to current position
        dists_from_start = self.dijkstra_cache(start_from_node)
        start = min(reg_nodes, key=lambda n: dists_from_start.get(n, float('inf')))

        order = [start]
        unvisited = reg_nodes - {start}
        cur = start
        while unvisited:
            dcur = self.dijkstra_cache(cur)
            nxt = min(unvisited, key=lambda n: dcur.get(n, float('inf')))
            order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        # expand to street path and compute exact length
        expanded = self._expand_seq_to_path(order)
        expanded_len = 0.0
        for a, b in zip(expanded, expanded[1:]):
            try:
                expanded_len += nx.shortest_path_length(self.G, a, b, weight='length')
            except Exception:
                pass

        return expanded, expanded_len, reg_nodes

    def solve_clean_single_path(self, delivery_points: pd.DataFrame,
                            start_depot: str, end_depot: str,
                            max_points_per_region: int = 15,
                            corridor_width_km: float = 3.0):
        """
        Creates a clean single path from start→end by visiting regions in geographic order
        """
        print("CLEAN PATH: Building route along shortest path...")
        
        # 1. Get the actual shortest path between depots
        s_node = self.depot_nodes[start_depot]
        e_node = self.depot_nodes[end_depot]
        main_path = nx.shortest_path(self.G, s_node, e_node, weight='length')
        main_path_coords = self._coords_for_nodes(main_path)
        
        # 2. Decompose into regions
        regions = self.quad_tree_decomposition(delivery_points, max_points_per_region)
        
        # 3. Find regions near the main path
        path_regions = []
        for region in regions:
            # Calculate minimum distance from region centroid to any segment of main path
            min_dist = float('inf')
            for i in range(len(main_path_coords)-1):
                lon1, lat1 = main_path_coords[i]
                lon2, lat2 = main_path_coords[i+1]
                
                dist = self.point_to_segment_distance(
                    region['centroid_lat'], region['centroid_lon'],
                    lat1, lon1, lat2, lon2
                )
                min_dist = min(min_dist, dist)
            
            if min_dist <= corridor_width_km:
                # Calculate projection score to order regions along the path
                projection = self._projection_score(region, 
                    main_path_coords[0][1], main_path_coords[0][0],  # start
                    main_path_coords[-1][1], main_path_coords[-1][0]  # end
                )
                path_regions.append((region, projection, min_dist, len(region['points'])))
        
        # 4. Sort regions by position along the path (projection score)
        path_regions.sort(key=lambda x: x[1])
        print(f"  Found {len(path_regions)} regions along the path")
        
        # 5. Build route by visiting regions in order
        complete_route = [s_node]
        current_node = s_node
        total_m = 0.0
        visited_deliveries = set()
        used_regions = []  # Track which regions we actually use
        
        for region, projection, distance, point_count in path_regions:
            region_nodes = set(region['points']['nearest_node'])
            
            # Skip if no new deliveries or already mostly covered
            new_points = region_nodes - visited_deliveries
            if len(new_points) < max(1, point_count * 0.3):  # Skip if <30% new points
                continue
                
            try:
                # Route to this region (use nearest point on main path)
                region_entry_node = min(region_nodes, 
                                    key=lambda n: nx.shortest_path_length(self.G, current_node, n, weight='length'))
                
                # Get path to region
                to_region_path = self._shortest_path_nodes(current_node, region_entry_node)
                to_region_cost = nx.shortest_path_length(self.G, current_node, region_entry_node, weight='length')
                
                # Visit all points in region
                region_path, region_cost, region_visited = self._full_region_path_and_cost(
                    region['points'], region_entry_node
                )
                
                # Add to complete route
                complete_route.extend(to_region_path[1:])
                complete_route.extend(region_path[1:])
                total_m += to_region_cost + region_cost
                visited_deliveries.update(region_visited)
                current_node = region_path[-1]
                
                used_regions.append(region)
                print(f"  Added region with {len(region_visited)} deliveries")
                
            except Exception as e:
                print(f"  Could not route through region: {e}")
                continue
        
        # 6. Route to final destination
        if current_node != e_node:
            try:
                final_path = self._shortest_path_nodes(current_node, e_node)
                final_cost = nx.shortest_path_length(self.G, current_node, e_node, weight='length')
                complete_route.extend(final_path[1:])
                total_m += final_cost
            except Exception as e:
                print(f"  Warning: Could not route to end depot: {e}")
        
        print(f"CLEAN PATH: distance = {total_m/1000:.2f} km | regions used = {len(used_regions)} | deliveries = {len(visited_deliveries)}")
        
        # Return the actual regions used for visualization
        return complete_route, total_m, used_regions, visited_deliveries

    # ---------- metrics ----------

    def calculate_route_distance(self, route_nodes):
        tot = 0.0
        for a, b in zip(route_nodes, route_nodes[1:]):
            try:
                tot += nx.shortest_path_length(self.G, a, b, weight='length')
            except Exception:
                pass
        return tot

    def baseline_sn_distance_km(self, start_depot='S', end_depot='N') -> float:
        s = self.depot_nodes[start_depot]; e = self.depot_nodes[end_depot]
        d_m = nx.shortest_path_length(self.G, s, e, weight='length')
        return d_m / 1000.0

    # ---------- plotting ----------

    def _set_fixed_bounds(self, ax,
                          x_min=-87.90, x_max=-87.50,
                          y_min=41.60,  y_max=42.05):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

    def _style_axes(self, ax, title):
        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True, alpha=0.2)

    def _get_city_boundary(self):
        if self._city_boundary is None:
            self._city_boundary = ox.geocode_to_gdf("Chicago, Illinois, USA")
        return self._city_boundary

    def _coords_for_nodes(self, nodes):
        coords = []
        for n in nodes:
            if n in self.nodes.index:
                r = self.nodes.loc[n]
                coords.append((float(r['x']), float(r['y'])))
        return coords

    def visualize_quad_tree_decomposition(self, regions, delivery_points, title="Quad-Tree Decomposition"):
        fig, ax = plt.subplots(figsize=(14, 16), dpi=180)

        city = self._get_city_boundary()
        city.boundary.plot(ax=ax, linewidth=2.2, edgecolor="#7B7A7A", alpha=0.9, zorder=0)
        self.edges.plot(ax=ax, linewidth=0.35, edgecolor='#B0B0B0', alpha=0.8, zorder=1)

        # All points
        ax.scatter(delivery_points['lon'], delivery_points['lat'],
                   s=18, c="#5B88CC", edgecolors='#5078A0', linewidths=0.5,
                   alpha=1.0, zorder=4, label='All Delivery Points')

        # Regions
        colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(regions))))
        for i, r in enumerate(regions):
            b = r['bounds']
            rect = patches.Rectangle((b['min_lon'], b['min_lat']),
                                     b['max_lon']-b['min_lon'],
                                     b['max_lat']-b['min_lat'],
                                     linewidth=1.2 + 0.5*r['depth'],
                                     edgecolor=colors[i], facecolor='none', alpha=0.95, zorder=2)
            ax.add_patch(rect)
            # centroid
            ax.scatter(r['centroid_lon'], r['centroid_lat'], c='red', s=40, marker='x', zorder=6)

        # Depots
        for name, d in self.depots.items():
            ax.scatter(d['lon'], d['lat'], s=420, marker='s', c=d['color'],
                       edgecolors='black', linewidth=3, zorder=10, label=f'{d["name"]} Depot')

        self._set_fixed_bounds(ax)
        self._style_axes(ax, f"{title}\nRegions: {len(regions)} | Points: {len(delivery_points)}")
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_route(self, route_nodes, delivery_points, visited_deliveries, title,
                        start_depot=None, end_depot=None):
        fig, ax = plt.subplots(figsize=(14, 16), dpi=180)

        self._get_city_boundary().boundary.plot(ax=ax, linewidth=2.2, edgecolor='#7B7A7A', alpha=0.9, zorder=0)
        self.edges.plot(ax=ax, linewidth=0.35, edgecolor='#B0B0B0', alpha=0.85, zorder=1)

        # Unvisited points
        unvisited = delivery_points[~delivery_points['nearest_node'].isin(visited_deliveries)]
        ax.scatter(unvisited['lon'], unvisited['lat'],
                   s=16, c='#E6F0FF', edgecolors="#2367AC", linewidths=0.5,
                   alpha=0.6, zorder=2, label='Unvisited Points')

        # Visited points
        visited = delivery_points[delivery_points['nearest_node'].isin(visited_deliveries)]
        ax.scatter(visited['lon'], visited['lat'],
                   s=32, c='#00FF00', edgecolors="#006400", linewidths=0.8,
                   alpha=1.0, zorder=5, label='Visited Points')

        # Route polyline (already expanded to street nodes)
        coords = self._coords_for_nodes(route_nodes)
        if coords:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, '-', linewidth=3.2, color='#0D0D0D', alpha=0.98, zorder=9, label='Route')

        # Depots
        for name, d in self.depots.items():
            lbl = f'{d["name"]} Depot'
            if start_depot == name: lbl += " (START)"
            if end_depot == name:   lbl += " (END)"
            ax.scatter(d['lon'], d['lat'], s=500, marker='s', c=d['color'],
                       edgecolors='black', linewidth=3, zorder=12, label=lbl)

        self._set_fixed_bounds(ax)
        dist_km = self.calculate_route_distance(route_nodes)/1000.0
        coverage = len(visited_deliveries)
        total = len(delivery_points)
        self._style_axes(ax, f"{title}\nDistance: {dist_km:.1f} km | Coverage: {coverage}/{total} ({100*coverage/total:.1f}%)")
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(f"{title}.png", dpi=320, bbox_inches='tight')
        plt.close()
        
    # alias kept for older calls
    def visualize_quad_tree(self, points, regions, title="Quad-Tree_Decomposition"):
        return self.visualize_quad_tree_decomposition(regions, points, title)
