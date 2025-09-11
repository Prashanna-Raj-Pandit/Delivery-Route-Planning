# greedy.py
import networkx as nx

def greedy_nearest_neighbor(G, depot, deliveries):
    route = [depot]
    unvisited = set(deliveries)
    current = depot
    total_distance = 0.0

    while unvisited:
        # Compute shortest paths from current node
        dist_dict = nx.single_source_dijkstra_path_length(G, current, weight='length')

        # Find the closest reachable node
        reachable = [d for d in unvisited if d in dist_dict]
        if not reachable:
            break  # or raise an error

        next_node = min(reachable, key=lambda x: dist_dict[x])
        total_distance += dist_dict[next_node]
        route.append(next_node)
        current = next_node
        unvisited.remove(next_node)

    # Return to depot if possible
    if depot in G.nodes and current in G.nodes:
        try:
            total_distance += nx.dijkstra_path_length(G, current, depot, weight='length')
            route.append(depot)
        except nx.NetworkXNoPath:
            print(f"Warning: Depot {depot} unreachable from last node")

    return route, total_distance

