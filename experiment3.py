# experiment3.py — Priority-only region selection (no corridor), with priority-colored points and region overlays
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from divideconquer import DivideConquerRouter

# ---------- scoring helpers ----------
def calculate_priority_score(counts: dict[int, int]) -> int:
    # High=1, Med=2, Low=3
    return counts.get(1,0)*50 + counts.get(2,0)*30 + counts.get(3,0)*20

def region_priority_counts(df: pd.DataFrame) -> dict[int, int]:
    return {
        1: int((df['priority'] == 1).sum()),
        2: int((df['priority'] == 2).sum()),
        3: int((df['priority'] == 3).sum()),
    }

# ---------- visualization (priority-colored points + region rectangles) ----------
def plot_points_by_priority_with_regions(router: DivideConquerRouter,
                                         points: pd.DataFrame,
                                         selected_regions: list,
                                         title: str = "Priority_Regions_Overlay"):
    fig, ax = plt.subplots(figsize=(14, 16), dpi=180)

    # base layers (city & roads)
    router._get_city_boundary().boundary.plot(ax=ax, linewidth=2.0, edgecolor="#666", alpha=0.9, zorder=0)
    router.edges.plot(ax=ax, linewidth=0.35, edgecolor="#B0B0B0", alpha=0.85, zorder=1)

    # points by priority
    cmap = {1: ("#FF3B30", "#8B0000"),  # red
            2: ("#FFA500", "#8B5A00"),  # orange
            3: ("#00C853", "#006400")}  # green
    for p in [1,2,3]:
        sub = points[points['priority'] == p]
        if not sub.empty:
            face, edge = cmap[p]
            ax.scatter(sub['lon'], sub['lat'], s=22, c=face, edgecolors=edge, linewidths=0.5,
                       alpha=0.95, zorder=4, label=f'Priority {p}')

    # draw selected region rectangles
    for r in selected_regions:
        b = r['bounds']
        rect = plt.Rectangle((b['min_lon'], b['min_lat']),
                             b['max_lon'] - b['min_lon'],
                             b['max_lat'] - b['min_lat'],
                             fill=False, lw=2.0, ec='#1F77B4', alpha=0.95, zorder=6)
        ax.add_patch(rect)
        # centroid mark
        ax.scatter(r['centroid_lon'], r['centroid_lat'], c='#1F77B4', s=36, marker='x', zorder=7)

    # depots
    for name, d in router.depots.items():
        ax.scatter(d['lon'], d['lat'], s=460, marker='s', c=d['color'],
                   edgecolors='black', linewidth=3, zorder=10, label=f'{d["name"]} Depot')

    router._set_fixed_bounds(ax)
    ax.set_title(f"{title}\nSelected regions: {len(selected_regions)} | Points: {len(points)}",
                 fontsize=18, fontweight='bold', pad=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True, alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=320, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {title}.png")

# ---------- priority-first clean path (NO corridor filtering) ----------
def build_priority_clean_path_no_corridor(router: DivideConquerRouter,
                                          points: pd.DataFrame,
                                          start_depot: str = 'S',
                                          end_depot: str = 'N',
                                          max_points_per_region: int = 20,
                                          min_points_per_region: int = 8,
                                          top_k_regions: int | None = 25):
    """
    Select regions purely by priority/size (no corridor/spine distance):
      - Quad-tree partition (<= max_points_per_region each)
      - Filter regions with size >= min_points_per_region
      - Score = 50*#P1 + 30*#P2 + 20*#P3
      - Keep top_k_regions by (score desc, size desc)
      - Order the kept regions south→north by projection along the S→N spine
      - Visit all points in each selected region; stitch via street paths
    """
    print("PRIORITY (NO-CORRIDOR): Selecting regions by priority/size only...")

    # Spine only for ordering, not for selection
    s_node = router.depot_nodes[start_depot]
    e_node = router.depot_nodes[end_depot]
    spine_nodes = nx.shortest_path(router.G, s_node, e_node, weight='length')
    spine_coords = router._coords_for_nodes(spine_nodes)

    # 1) Quad-tree regions
    regions = router.quad_tree_decomposition(points, max_points_per_region=max_points_per_region)

    # 2) Filter & score (no distance check)
    scored = []
    for reg in regions:
        rp = reg['points']
        size = len(rp)
        if size < min_points_per_region:
            continue
        counts = region_priority_counts(rp)
        pr_score = calculate_priority_score(counts)

        proj = router._projection_score(
            reg,
            spine_coords[0][1], spine_coords[0][0],
            spine_coords[-1][1], spine_coords[-1][0]
        )
        # store (region, projection, score, size, counts)
        scored.append((reg, proj, pr_score, size, counts))

    # 3) Keep only the most valuable regions
    if top_k_regions is not None and len(scored) > top_k_regions:
        scored = sorted(scored, key=lambda x: (x[2], x[3]), reverse=True)[:top_k_regions]

    # 4) Order S→N by projection
    scored.sort(key=lambda x: x[1])

    selected_regions = [t[0] for t in scored]

    print(f"  Selected {len(selected_regions)} regions "
          f"(min_points_per_region={min_points_per_region}, top_k={top_k_regions})")

    # 5) Build route by visiting selected regions in order
    route = [s_node]
    current = s_node
    total_m = 0.0
    visited_deliveries = set()

    for (reg, proj, pr_score, size, counts) in scored:
        region_nodes = set(reg['points']['nearest_node'])
        new_points = region_nodes - visited_deliveries
        if not new_points:
            continue

        # Entry via ONE Dijkstra (uses router cache)
        dcur = router.dijkstra_cache(current)
        region_entry = min(new_points, key=lambda n: dcur.get(n, float('inf')))
        to_reg_cost = dcur.get(region_entry, float('inf'))
        to_reg_path = router._shortest_path_nodes(current, region_entry)

        # Visit all points in region (existing NN)
        reg_path, reg_cost, reg_visited = router._full_region_path_and_cost(reg['points'], region_entry)

        # Stitch
        route.extend(to_reg_path[1:])
        route.extend(reg_path[1:])
        total_m += float(to_reg_cost) + float(reg_cost)
        visited_deliveries.update(reg_visited)
        current = reg_path[-1]

        print(f"  + Region depth={reg['depth']}, size={size}, "
              f"score={pr_score} (H/M/L={counts[1]}/{counts[2]}/{counts[3]})")

    # 6) Finish at end depot
    if current != e_node:
        try:
            last_path = router._shortest_path_nodes(current, e_node)
            last_cost = nx.shortest_path_length(router.G, current, e_node, weight='length')
            route.extend(last_path[1:])
            total_m += last_cost
        except Exception as e:
            print(f"  Warning: could not connect to end depot: {e}")

    print(f"PRIORITY (NO-CORRIDOR): distance = {total_m/1000:.2f} km | "
          f"regions used = {len(selected_regions)} | deliveries = {len(visited_deliveries)}")

    return route, total_m, selected_regions, visited_deliveries

# ---------- experiment wrapper ----------
def run_priority_experiment_sn():
    print("Priority-Only Region Selection (S → N)")
    router = DivideConquerRouter("chicago_street_network.graphml")
    points = router.load_and_preprocess_points("delivery_points_1000.csv")

    # Visual 1: all points by priority (no regions yet)
    plot_points_by_priority_with_regions(router, points, [], "Priority_All_Points")

    # Configs: vary how strict we are about region size & how many regions to keep
    configs = [
        {'name': 'TopK20_Size8',  'min_points_per_region': 8,  'top_k_regions': 20},
        {'name': 'TopK30_Size6',  'min_points_per_region': 6,  'top_k_regions': 30},
        {'name': 'TopK15_Size10', 'min_points_per_region': 10, 'top_k_regions': 15},
    ]

    results = []
    for cfg in configs:
        print(f"\nConfig: {cfg['name']}  "
              f"(min_points={cfg['min_points_per_region']}, top_k={cfg['top_k_regions']})")

        t0 = time.time()
        route, dist_m, selected_regions, visited_deliveries = build_priority_clean_path_no_corridor(
            router,
            points,
            start_depot='S',
            end_depot='N',
            max_points_per_region=20,
            min_points_per_region=cfg['min_points_per_region'],
            top_k_regions=cfg['top_k_regions']
        )
        dt = time.time() - t0

        actual_m = router.calculate_route_distance(route)
        covered = len(visited_deliveries)

        results.append({
            'config_name': cfg['name'],
            'computation_time_s': dt,
            'distance_km': actual_m/1000.0,
            'deliveries_covered': covered,
            'efficiency_deliv_per_km': (covered / (actual_m/1000.0)) if actual_m > 0 else 0.0,
            'regions_used': len(selected_regions)
        })

        # Visual 2: show priority points + only the selected regions
        plot_points_by_priority_with_regions(router, points, selected_regions,
                                             f"Priority_Selected_Regions_{cfg['name']}")

        # Visual 3: standard route plot (visited vs unvisited)
        router.visualize_route(
            route_nodes=route,
            delivery_points=points,
            visited_deliveries=visited_deliveries,
            title=f"Priority_NoCorridor_Route_{cfg['name']}",
            start_depot='S',
            end_depot='N'
        )

    df = pd.DataFrame(results)
    df.to_csv("priority_no_corridor_results_sn.csv", index=False)
    print("\nSaved: priority_no_corridor_results_sn.csv")

    # simple summary plot
    fig, ax = plt.subplots(figsize=(12,7))
    ax.bar(df['config_name'], df['deliveries_covered'])
    for r in ax.patches:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+3, f"{int(r.get_height())}",
                ha='center', va='bottom', fontweight='bold')
    ax.set_title("Deliveries Covered (Priority-Only Selection)"); ax.set_ylabel("Points")
    ax.grid(True, alpha=.3); plt.xticks(rotation=25, ha='right'); plt.tight_layout()
    plt.savefig("Priority_NoCorridor_Coverage_Comparison.png", dpi=300)
    plt.close()

    return df

if __name__ == "__main__":
    _ = run_priority_experiment_sn()




"""With Corridor Filtering"""
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx

# from divideconquer import DivideConquerRouter

# # -------------------- scoring helpers --------------------

# def calculate_priority_score(counts):
#     # High=1, Medium=2, Low=3 in your CSV; higher weight for higher priority
#     return counts.get(1,0)*50 + counts.get(2,0)*30 + counts.get(3,0)*20

# def region_priority_counts(region_points: pd.DataFrame):
#     return {
#         1: int((region_points['priority'] == 1).sum()),
#         2: int((region_points['priority'] == 2).sum()),
#         3: int((region_points['priority'] == 3).sum()),
#     }

# # -------------------- priority-aware clean path builder --------------------

# def build_priority_clean_path(router: DivideConquerRouter,
#                               points: pd.DataFrame,
#                               start_depot: str = 'S',
#                               end_depot: str = 'N',
#                               max_points_per_region: int = 20,
#                               corridor_width_km: float = 3.0,
#                               detour_penalty_per_km: float = 1.0,
#                               min_priority_score: int = 50,
#                               top_k_regions: int | None = None):
#     """
#     Priority-first version of your clean S→N path that:
#       - scores regions by (priority_score - detour_penalty_per_km * min_detour_km)
#       - filters by corridor band and min_priority_score
#       - orders regions south→north (projection on the spine)
#       - visits all points in a chosen region (your existing in-region NN)
#     No budget used.
#     """

#     print("PRIORITY CLEAN PATH: Scoring regions and building route...")

#     # 1) Spine S→N (street shortest path)
#     s_node = router.depot_nodes[start_depot]
#     e_node = router.depot_nodes[end_depot]
#     spine_nodes = nx.shortest_path(router.G, s_node, e_node, weight='length')
#     spine_coords = router._coords_for_nodes(spine_nodes)

#     # 2) Quad-tree regions
#     regions = router.quad_tree_decomposition(points, max_points_per_region=max_points_per_region)

#     # 3) Corridor filter + scoring
#     scored = []
#     for reg in regions:
#         rp = reg['points']
#         if rp.empty:
#             continue

#         # min distance (km) from region centroid to any spine segment
#         min_detour_km = float('inf')
#         for i in range(len(spine_coords) - 1):
#             lon1, lat1 = spine_coords[i]
#             lon2, lat2 = spine_coords[i+1]
#             d = router.point_to_segment_distance(
#                 reg['centroid_lat'], reg['centroid_lon'],
#                 lat1, lon1, lat2, lon2
#             )
#             if d < min_detour_km:
#                 min_detour_km = d

#         if min_detour_km > corridor_width_km:
#             continue  # outside corridor band

#         counts = region_priority_counts(rp)
#         pr_score = calculate_priority_score(counts)
#         if pr_score < min_priority_score:
#             continue  # too low value

#         # projection for S→N ordering
#         proj = router._projection_score(
#             reg,
#             spine_coords[0][1], spine_coords[0][0],
#             spine_coords[-1][1], spine_coords[-1][0]
#         )

#         score = pr_score - detour_penalty_per_km * min_detour_km
#         scored.append((reg, proj, min_detour_km, pr_score, counts, score))

#     # optional: keep only top-K by score (still ordered S→N later)
#     if top_k_regions is not None and len(scored) > top_k_regions:
#         scored = sorted(scored, key=lambda x: x[5], reverse=True)[:top_k_regions]

#     # Order south→north (by projection)
#     scored.sort(key=lambda x: x[1])

#     print(f"  Selected {len(scored)} regions within {corridor_width_km} km, "
#           f"min_priority_score={min_priority_score}, detour_penalty={detour_penalty_per_km}")

#     # 4) Build route by visiting selected regions in order
#     route = [s_node]
#     current = s_node
#     total_m = 0.0
#     visited_deliveries = set()
#     used_regions = []

#     for (reg, proj, d_km, pr_score, counts, score) in scored:
#         region_nodes = set(reg['points']['nearest_node'])
#         new_points = region_nodes - visited_deliveries
#         if not new_points:
#             continue

#         # Entry with ONE Dijkstra (faster and uses your cache)
#         dcur = router.dijkstra_cache(current)
#         region_entry = min(new_points, key=lambda n: dcur.get(n, float('inf')))
#         to_reg_cost = dcur.get(region_entry, float('inf'))
#         to_reg_path = router._shortest_path_nodes(current, region_entry)

#         # Visit all points in region (your existing NN inside region)
#         reg_path, reg_cost, reg_visited = router._full_region_path_and_cost(reg['points'], region_entry)

#         # Stitch
#         route.extend(to_reg_path[1:])
#         route.extend(reg_path[1:])
#         total_m += float(to_reg_cost) + float(reg_cost)
#         visited_deliveries.update(reg_visited)
#         current = reg_path[-1]
#         used_regions.append(reg)

#         print(f"  + Region depth={reg['depth']} size={reg['size']} "
#               f"score={score:.1f} (H/M/L={counts[1]}/{counts[2]}/{counts[3]})")

#     # 5) Finish at N
#     if current != e_node:
#         try:
#             last_path = router._shortest_path_nodes(current, e_node)
#             last_cost = nx.shortest_path_length(router.G, current, e_node, weight='length')
#             route.extend(last_path[1:])
#             total_m += last_cost
#         except Exception as e:
#             print(f"  Warning: could not connect to end depot: {e}")

#     print(f"PRIORITY CLEAN PATH: distance = {total_m/1000:.2f} km | "
#           f"regions used = {len(used_regions)} | deliveries = {len(visited_deliveries)}")

#     return route, total_m, used_regions, visited_deliveries, scored

# # -------------------- plotting & experiment wrapper --------------------

# def summarize_priority_coverage(points: pd.DataFrame, visited_nodes: set):
#     out = {}
#     for p in [1,2,3]:
#         sub = points[points['priority'] == p]
#         nodes = set(sub['nearest_node'])
#         cov = len(visited_nodes & nodes)
#         out[f'priority_{p}_covered'] = cov
#         out[f'priority_{p}_total'] = len(sub)
#         out[f'priority_{p}_percent'] = (cov/len(sub)*100.0) if len(sub) else 0.0
#     return out

# def create_priority_analysis_plots(df: pd.DataFrame):
#     fig, axs = plt.subplots(2, 3, figsize=(20, 14))
#     axs = axs.ravel()

#     x = np.arange(len(df))

#     # 1) Weighted score
#     b = axs[0].bar(x, df['priority_score'], alpha=.85)
#     axs[0].set_xticks(x); axs[0].set_xticklabels(df['config_name'], rotation=30, ha='right')
#     axs[0].set_title("Weighted Priority Score")
#     for r in b:
#         axs[0].text(r.get_x()+r.get_width()/2, r.get_height()+5, f"{r.get_height():.0f}",
#                     ha='center', va='bottom')

#     # 2) Coverage by priority
#     width = 0.25
#     for i, p in enumerate([1,2,3]):
#         axs[1].bar(x + i*width, df[f'priority_{p}_percent'], width, label=f'Priority {p}')
#     axs[1].set_xticks(x + width); axs[1].set_xticklabels(df['config_name'], rotation=30, ha='right')
#     axs[1].legend(); axs[1].set_ylim(0, 100); axs[1].set_title("Priority Coverage (%)")

#     # 3) Distance
#     axs[2].bar(df['config_name'], df['distance_km'])
#     axs[2].set_title("Route Distance (km)"); axs[2].tick_params(axis='x', rotation=30)

#     # 4) Delivery composition (stacked)
#     bottom = np.zeros(len(df))
#     labels = ['High','Medium','Low']; cols = ['red','orange','green']
#     for i, p in enumerate([1,2,3]):
#         axs[3].bar(df['config_name'], df[f'priority_{p}_covered'], bottom=bottom,
#                    color=cols[i], label=labels[i])
#         bottom += df[f'priority_{p}_covered'].values
#     axs[3].legend(); axs[3].set_title("Deliveries by Priority")
#     axs[3].tick_params(axis='x', rotation=30)

#     # 5) Efficiency
#     axs[4].bar(df['config_name'], df['efficiency'])
#     axs[4].set_title("Efficiency (deliveries/km)")
#     axs[4].tick_params(axis='x', rotation=30)

#     # 6) Params
#     axs[5].bar(df['config_name'], df['detour_penalty_per_km'], width=0.45, label='Detour penalty', alpha=.7)
#     axs[5].bar(x + 0.45, df['min_priority_score'], width=0.45, label='Min priority score', alpha=.7)
#     axs[5].set_xticks(x + 0.225); axs[5].set_xticklabels(df['config_name'], rotation=30, ha='right')
#     axs[5].legend(); axs[5].set_title("Selection Parameters")

#     plt.tight_layout()
#     plt.savefig("Priority_Analysis_SN_Comprehensive.png", dpi=300, bbox_inches='tight')
#     plt.close()

#     # scatter: distance vs score
#     fig, ax = plt.subplots(figsize=(12, 8))
#     sc = ax.scatter(df['distance_km'], df['priority_score'], s=200,
#                     c=df['detour_penalty_per_km'], cmap='viridis', alpha=.85)
#     for _, row in df.iterrows():
#         ax.annotate(row['config_name'], (row['distance_km'], row['priority_score']),
#                     xytext=(6,6), textcoords='offset points', fontsize=9,
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
#     ax.set_xlabel("Distance (km)"); ax.set_ylabel("Priority Score")
#     ax.set_title("Tradeoff: Distance vs Priority (color = detour penalty)")
#     cbar = plt.colorbar(sc); cbar.set_label("Detour penalty per km")
#     ax.grid(True, alpha=.3)
#     plt.tight_layout()
#     plt.savefig("Priority_Tradeoff_Analysis.png", dpi=300)
#     plt.close()

# def run_priority_experiment_sn():
#     print("Priority-Aware Experiment (S → N)")
#     router = DivideConquerRouter("chicago_street_network.graphml")
#     points = router.load_and_preprocess_points("delivery_points_1000.csv")

#     # Configs: tune how aggressively you trade distance for priority
#     configs = [
#         {'name': 'Distance-Focused',   'detour_penalty_per_km': 3.0, 'min_priority_score': 50,  'top_k_regions': None},
#         {'name': 'Balanced',           'detour_penalty_per_km': 1.5, 'min_priority_score': 50,  'top_k_regions': None},
#         {'name': 'Delivery-Focused',   'detour_penalty_per_km': 0.8, 'min_priority_score': 40,  'top_k_regions': None},
#         {'name': 'Priority-Optimized', 'detour_penalty_per_km': 0.5, 'min_priority_score': 40,  'top_k_regions': None},
#     ]

#     results = []
#     for cfg in configs:
#         print(f"\nConfig: {cfg['name']}  "
#               f"(detour_penalty_km={cfg['detour_penalty_per_km']}, "
#               f"min_priority_score={cfg['min_priority_score']})")

#         t0 = time.time()
#         route, dist_m, used_regions, visited_deliveries, scored = build_priority_clean_path(
#             router,
#             points,
#             start_depot='S',
#             end_depot='N',
#             max_points_per_region=20,
#             corridor_width_km=3.0,
#             detour_penalty_per_km=cfg['detour_penalty_per_km'],
#             min_priority_score=cfg['min_priority_score'],
#             top_k_regions=cfg['top_k_regions']
#         )
#         dt = time.time() - t0

#         actual_m = router.calculate_route_distance(route)
#         visited_nodes = set(visited_deliveries)

#         # coverage by priority
#         cov = summarize_priority_coverage(points, visited_nodes)
#         total_cov = (cov['priority_1_covered'] + cov['priority_2_covered'] + cov['priority_3_covered'])
#         pr_counts = {1: cov['priority_1_covered'], 2: cov['priority_2_covered'], 3: cov['priority_3_covered']}
#         pr_score = calculate_priority_score(pr_counts)

#         max_score = calculate_priority_score({
#             1: cov['priority_1_total'],
#             2: cov['priority_2_total'],
#             3: cov['priority_3_total']
#         })
#         score_pct = (pr_score / max_score * 100.0) if max_score > 0 else 0.0

#         results.append({
#             'config_name': cfg['name'],
#             'detour_penalty_per_km': cfg['detour_penalty_per_km'],
#             'min_priority_score': cfg['min_priority_score'],
#             'computation_time': dt,
#             'distance_km': actual_m/1000.0,
#             'total_deliveries': total_cov,
#             'priority_score': pr_score,
#             'score_percentage': score_pct,
#             **cov,
#             'efficiency': (total_cov/(actual_m/1000.0)) if actual_m > 0 else 0.0
#         })

#         # Use your standard route visual (visited/unvisited shown correctly)
#         router.visualize_route(
#             route_nodes=route,
#             delivery_points=points,
#             visited_deliveries=visited_deliveries,
#             title=f"Priority_SN_{cfg['name'].replace(' ','_')}",
#             start_depot='S',
#             end_depot='N'
#         )

#     df = pd.DataFrame(results)
#     create_priority_analysis_plots(df)
#     df.to_csv("priority_results_sn.csv", index=False)
#     print("\nCompleted priority experiment.")
#     return df

# if __name__ == "__main__":
#     _ = run_priority_experiment_sn()
