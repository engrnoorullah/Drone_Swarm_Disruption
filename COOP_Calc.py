import numpy as np
import networkx as nx
import math
import itertools

NUM_DRONES = 101

def dist_Rlink(G, i, j):
    try:
        return nx.shortest_path_length(G, source=i, target=j)
    except nx.NetworkXNoPath:
        return float('inf')  # return a large number if no path exists

def update_node_weights(G, target, max_distance, prev_connections, mu_dist, lambda_dist, alpha, beta, gamma):
    for node in G.nodes(data=True):
        pos = np.array(node[1]['pos'])
        distance_to_target = np.linalg.norm(pos - target)
        proximity_factor = math.exp(-mu_dist * (distance_to_target / max_distance))

        # Initialize Theta if it doesn't exist
        if 'Theta' not in node[1]:
            node[1]['Theta'] = node[1]['weight']
        node[1]['Theta']  += (alpha * proximity_factor)

def calculate_coop_score(G, removal_count,  prev_connections, original_data=None, alpha=0.1, beta=0.01, gamma=0.1, lambda_dist=0.5, mu_dist=0.5, target=np.array([9, 9, 4]), speed = 1.0, area_dims=(10, 10, 10)):
    nodes = list(G.nodes())
    node_positions = {node: np.array(pos, dtype=float) for node, pos in nx.get_node_attributes(G, 'pos').items()}
    
    max_distance = np.linalg.norm([area_dims[0], area_dims[1], area_dims[2]])
    

    # Initialize original_data if not already present
    if removal_count == 0 or original_data is None:
        node_weights = nx.get_node_attributes(G, 'weight')
        original_data = {
            'original_node_weights_sum': sum(node_weights.values()),
            'original_possible_connections': (NUM_DRONES * NUM_DRONES-1)/2,  # Initialize this value correctly if needed
            'original_component_factor': None
        }
    update_node_weights(G, target, max_distance, prev_connections, mu_dist, lambda_dist, alpha, beta, gamma)
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    node_thetas = {node: data.get('Theta', 0) for node, data in G.nodes(data=True)}

    #print(node_weights)

    total_comb_sum = 0
    weighted_comb_sum = 0

    # Calculate reliability for each edge and update the component factor
    for i, j in itertools.combinations(nodes, 2):
        distance = dist_Rlink(G,i,j)
        reliability = math.exp(-lambda_dist * distance)
        total_comb_sum += 1
        weighted_comb_sum += reliability
    normalized_component_factor = weighted_comb_sum / original_data['original_possible_connections']
    modified_node_thetas_sum = sum(node_thetas.values())

    # Calculate cooperative score
    coop_score = (modified_node_thetas_sum / original_data['original_node_weights_sum']) * normalized_component_factor
    original_data['original_component_factor'] = weighted_comb_sum

    # Return values including updated original data
    return coop_score, 0, normalized_component_factor, original_data