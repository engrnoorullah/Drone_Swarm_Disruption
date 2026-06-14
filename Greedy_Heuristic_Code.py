import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from dqn_batch import DQNPrioritizedReplay
from uavs_env import UAVSEnv
import os
import scipy.stats as st
from scipy.stats import sem, t
import operator
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager
import csv
import io
import imageio
import time
import statistics
import math
import itertools
import random
import multiprocessing as mp
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import copy
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from SwarmGraph import SwarmGraph
#np.random.seed(1)

# Parameters
NUM_DRONES = 100
AREA_SIZE = 10
communication_range = 1
removed_nodes = []
max_connections = NUM_DRONES-1
#NUM_DRONES = 55
#TARGET = [9, 9]
SPEED = 0.1
SEPARATION = 0.002
stopper = 0

AREA_WIDTH = 10
AREA_HEIGHT = 10
AREA_DEPTH = 10
MIN_SEPARATION = 0.02  # Minimum distance between drones adjusted for your area size
MAX_VELOCITY = 0.1  # Maximum velocity component in any direction
TARGET = np.array([9, 9, 4])

W = 0.5  # Inertia weight
C1 = 0.8  # Cognitive (particle) weight
C2 = 0.9  # Social (swarm) weight

removal_info = {'Weight-Art-Deg': [], 'EdgeBet-State': [], 'Art-Deg': [], 'Betweenness': [], 'DQN': [], 'GA': []}  # Add other methods if necessary

def reset_globals(model, version):
    global G, removal_count, coop_scores, coop_ratios, stopper, final_time_coop_func
    global node_weights_history, edge_weights_history, alg_connectivity_history, speed_history, removal_steps

    removal_count = 0
    coop_scores = []
    coop_ratios = []
    node_weights_history = []
    edge_weights_history = []
    alg_connectivity_history = []
    speed_history = []
    removal_steps = []
    stopper = 0
    final_time_coop_func = 0

def get_largest_components(G, n=2):
    return sorted(nx.connected_components(G), key=len, reverse=True)[:n]

def distance_between_components(G, component1, component2):
    min_distance = float('inf')
    for node1 in component1:
        pos1 = np.array(G.nodes[node1]['pos'])
        for node2 in component2:
            pos2 = np.array(G.nodes[node2]['pos'])
            distance = np.linalg.norm(pos1 - pos2)
            min_distance = min(min_distance, distance)
    return min_distance

def calculate_average_distance_to_target(component, target, G):
    distances = [np.linalg.norm(np.array(G.nodes[node]['pos']) - target) for node in component]
    return np.mean(distances)

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
        
def calculate_coop_score1(G, removal_count,  prev_connections, original_data=None, alpha=0.1, beta=0.01, gamma=0.1, lambda_dist=0.5, mu_dist=0.5, target=np.array([9, 9, 4]), speed = 1.0, area_dims=(10, 10, 10)):
    nodes = list(G.nodes())
    
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

def Rlink2(distance, dopt=0.1, dmin=0.05, dth=1.0):
    if distance <= dopt:
        return 1
    elif dopt < distance <= dth:
        return math.exp(-(distance - dopt) / dmin)
    else:
        return 0

def calculate_coop_score2(G, removal_count, original_data=None, alpha=0.35, beta=0.5, gamma=1, lambda_dist=0.5, mu_dist=0.5, target=np.array([9, 9, 4]), speed=1.0, area_dims=(10, 10, 10)):
    try:

        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        largest_subgraph = G.subgraph(largest_component).copy()
        nodes = list(G.nodes())
        node_positions = {node: np.array(pos, dtype=float) for node, pos in nx.get_node_attributes(G, 'pos').items()}
        node_weights = nx.get_node_attributes(G, 'weight')

        # Cooperative score calculations
        total_le = sum(node_weights.values())  # Sum of all l_{e_i} weights
        total_rlink = 0

        for i, j in itertools.combinations(nodes, 2):
            distance = np.linalg.norm(node_positions[i] - node_positions[j])
            rlink_value = Rlink2(distance)
            if rlink_value > 0:
                total_rlink += (node_weights[i] + node_weights[j]) * (2 ** rlink_value)
        original_data = {
                'total_rlink': total_rlink,
                'original_possible_connections': 0,
                'original_component_factor': None
            }


        coop_score = total_le + total_rlink
        normalized_component_factor = 0
        modified_node_weights_sum = sum(node_weights.values())

        final_time_coop_func = 0  # Placeholder for execution time measure

        return coop_score, final_time_coop_func, normalized_component_factor, original_data
    except Exception as e:
        print("Error occured in COOP")
        return 0,0,0,original_data

def get_node_color(weight):
    # Define ranges and their colors
    color_ranges = [
        #(-np.inf, 0, 'white'),
        (0, 0.1, 'grey'),
        (0.1, 0.2, 'pink'),
        (0.2, 0.3, 'violet'),
        (0.3, 0.4, 'purple'),
        (0.4, 0.5, 'blue'),
        (0.5, 0.6, 'green'),
        (0.5, 0.6, 'yellow'),
        (0.6, 0.7, 'orange'),
        (0.7, 0.8, 'red'),
        (0.8, 0.9, 'brown'),
        (0.9, 1, 'black')
    ]
    
    # Determine color based on the weight
    for (start, end, color) in color_ranges:
        if start <= weight < end:
            return color
    return "black"  # Default color for weights outside the specified ranges

def save_graph_snapshot1(G1, model, version, method, run, coop_score1, removal_count, original_data1, target=np.array([9, 9, 4])):
    output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
    os.makedirs(output_dir, exist_ok=True)
    pos = nx.spring_layout(G1)

    # Set up the figure and grid layout
    plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio for graph to text box
    graph_ax = plt.subplot(gs[0])
    text_ax = plt.subplot(gs[1])

    # Draw the graph in the left subplot
    node_thetas = nx.get_node_attributes(G1, 'Theta')
    node_weights = nx.get_node_attributes(G1, 'weight')
    node_colors = [get_node_color(weight) for weight in node_weights.values()]
    nx.draw(G1, pos, ax=graph_ax, node_size=100, with_labels=False, node_color=node_colors, edge_color='grey')
    graph_ax.axis('off')  # Turn off the axis for the graph

   

    # Metrics extraction
    original_sum_thetas = sum(node_thetas.values())# None if removal_count == 0 else original_data1['original_node_weights_sum']# if removal_count == 0 else None
    current_sum_thetas = sum(node_thetas.values())
    current_sum_weights = sum(node_weights.values())
    weighted_comb_sum = 0 if removal_count == 0 else original_data1['original_component_factor']
    num_edges = G1.number_of_edges()
    num_nodes = G1.number_of_nodes()

    largest_component_distance = float('inf')
    components = list(nx.connected_components(G1))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G1, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)

    # Initialize the weighted distance sum
    weighted_distance_sum = 0
    comp_count = 0
    total_size_of_components = sum(len(component) for component in components if len(component) > 1)

    # Find nodes in components with more than one node
    nodes_in_components = set()
    for component in nx.connected_components(G1):
        if len(component) > 1:
            comp_count += 1
            nodes_in_components.update(component)

    # Calculate the weighted sum of distances to the target for each node in larger components
    for node in nodes_in_components:
        weight = node_weights.get(node, 0)
        node_pos = np.array(G1.nodes[node]['pos'])
        distance = 1 / np.linalg.norm(node_pos - target)  # Calculating the inverse of the norm
        weighted_distance_sum += distance * weight
    weighted_distance_sum = weighted_distance_sum / comp_count
    #print("Power of attack in Cs", weighted_distance_sum)
    # Add text to the right subplot
    text_ax.axis('off')  # Turn off the axis for the text box
    text_info = [
        f"Initial Sum of Thetas: {original_sum_thetas:.1f}",
        f"Current Sum of Thetas: {current_sum_thetas:.1f}",
        f"Current Sum of weights: {current_sum_weights:.1f}",
        f"Power of ATtack: {weighted_distance_sum:.1f}",
        f"Weighted Comb Sum: {weighted_comb_sum:.1f}",
        f"Number of Nodes: {num_nodes}",
        f"Number of Edges: {num_edges}",
        f"Distance from Target: {largest_component_distance:.1f}",
        f"Coop Score: {coop_score1:.1f}"
    ]
    for idx, line in enumerate(text_info):
        text_ax.text(0.05, 1 - idx * 0.1, line, transform=text_ax.transAxes, fontsize=16)

    # Save the figure
    plt.savefig(f"{output_dir}/Cs_snapshot_{removal_count}.png")
    plt.close()

def save_graph_snapshot2(G2, model, version, method, run, coop_score2, removal_count, original_data2, target=np.array([9, 9, 4])):
    output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
    os.makedirs(output_dir, exist_ok=True)
    pos = nx.spring_layout(G2)

    # Set up the figure and grid layout
    plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio for graph to text box
    graph_ax = plt.subplot(gs[0])
    text_ax = plt.subplot(gs[1])

    # Draw the graph in the left subplot
    node_weights = nx.get_node_attributes(G2, 'weight')
    node_colors = [get_node_color(weight) for weight in node_weights.values()]
    nx.draw(G2, pos, ax=graph_ax, node_size=100, with_labels=False, node_color=node_colors, edge_color='grey')
    graph_ax.axis('off')  # Turn off the axis for the graph

    # Metrics extraction
    total_weights = sum(node_weights.values())
    total_rlink = 0 if removal_count == 0 else original_data2['total_rlink']  # Placeholder, replace with the actual sum of Rlinks
    num_edges = G2.number_of_edges()
    num_nodes = G2.number_of_nodes()

    largest_component_distance = float('inf')
    components = list(nx.connected_components(G2))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G2, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
    
    # Initialize the weighted distance sum
    weighted_distance_sum = 0
    comp_count = 0
    total_size_of_components = sum(len(component) for component in components if len(component) > 1)

    # Find nodes in components with more than one node
    nodes_in_components = set()
    for component in nx.connected_components(G2):
        if len(component) > 1:
            comp_count += 1
            nodes_in_components.update(component)

    # Calculate the weighted sum of distances to the target for each node in larger components
    for node in nodes_in_components:
        weight = node_weights.get(node, 0)
        node_pos = np.array(G2.nodes[node]['pos'])
        distance = 1 / np.linalg.norm(node_pos - target)  # Calculating the inverse of the norm
        weighted_distance_sum += distance * weight
    weighted_distance_sum = weighted_distance_sum / comp_count
    #print("Power of attack in COOP", weighted_distance_sum)

    # Add text to the right subplot
    text_ax.axis('off')  # Turn off the axis for the text box
    text_info = [
        f"Total Node Weights: {total_weights:.1f}",
        f"Total Rlink Sum: {total_rlink:.1f}",
        f"Power of Attack: {weighted_distance_sum:.1f}",
        f"Number of Nodes: {num_nodes}",
        f"Number of Edges: {num_edges}",
        f"Distance from Target: {largest_component_distance:.1f}",
        f"Coop Score: {coop_score2:.1f}"
    ]
    for idx, line in enumerate(text_info):
        text_ax.text(0.05, 1 - idx * 0.1, line, transform=text_ax.transAxes, fontsize=16)

    # Save the figure
    plt.savefig(f"{output_dir}/COOP_snapshot_{removal_count}.png")
    plt.close()


def get_node_color(weight):
    # Define a function to map weights to colors
    color_map = {
        0: "grey", 0.1: "pink", 0.2: "violet", 0.3: "purple", 0.4: "blue",
        0.5: "green", 0.6: "yellow", 0.7: "orange", 0.8: "red", 0.9: "brown",
        1: "black"
    }
    for threshold, color in color_map.items():
        if weight <= threshold:
            return color
    return "black"

def get_disconnected_component_sizes(G):
    return [len(c) for c in nx.connected_components(G)]

def within_target_distance(G, target, target_distance):
    # Calculate the distances from each node to the target
    distances = {node: np.linalg.norm(np.array(G.nodes[node]['pos']) - target) for node in G.nodes()}
    # Calculate the total weight
    total_weight = sum(nx.get_node_attributes(G, 'weight').values())
    # Calculate weight within the target distance
    weight_within_distance = sum(G.nodes[node]['weight'] for node in distances if distances[node] <= target_distance)
    
    # Check if 75% of the weight is within the target distance
    return weight_within_distance / total_weight >= 0.75

def geometric_center(nodes, positions):
    if not nodes:
        return np.array([0, 0, 0])  # Return a default center if no nodes are present
    node_positions = np.array([positions[node] for node in nodes])
    return np.mean(node_positions, axis=0)

def calculate_distance_to_target(center, target):
    return np.linalg.norm(center - target)

def greedy_removal_heuristic(G, images, model, version, method, swarm_graph, run, target_distance=1.5):
    coop_scores1, coop_scores2 = [], []
    coop_ratios1, coop_ratios2 = [], []
    frame_paths1, frame_paths2 = [], []
    total_coop_time1, total_coop_time2 = 0, 0
    removal_count1 = 0
    removal_count2 = 0
    removal_count = 0
    frame_paths1, frame_paths2 = [], []
    target = np.array([9, 9, 4])
    original_data1, original_data2 = None, None
    G_copy1 = G
    G_copy2 = G
    save_graph_snapshot1(G_copy1, model, version, method, run, 0, removal_count1, original_data1)
    save_graph_snapshot2(G_copy2, model, version, method, run, 0, removal_count2, original_data2)
    swarm_graph.G = G
    swarm_graph.update_positions()
    G_copy1 = swarm_graph.get_graph()
    G_copy2 = G_copy1.copy()
    original_G1, original_G2 = G_copy1.copy(), G_copy2.copy()

    First_Score2, _, _, original_data2 = calculate_coop_score2(G_copy2, removal_count2, original_data2)
    while removal_count < 25:

        closest_node1 = None
        min_difference1 = float('inf')
        closest_node2 = None
        min_difference2 = float('inf')

        max_decrease1, max_decrease2 = float('-inf'), float('-inf')  # Reset max decrease to handle state correctly
        best_node1, best_node2 = None, None  # Reset best nodes
        original_score1, _, _, original_data1 = calculate_coop_score1(G_copy1, removal_count1, 0, original_data1)
        original_score2, _, _, original_data2 = calculate_coop_score2(G_copy2, removal_count2, original_data2)
        # Testing node removal for Cs
        for node in list(G_copy1.nodes()):
            
            swarm_graph.G = G_copy1.copy()
            swarm_graph.remove_node(node)
            swarm_graph.update_positions()
            temp_graph = swarm_graph.get_graph()
            new_score1, _, _, original_data1 = calculate_coop_score1(temp_graph, removal_count1, 0, original_data1)
            #print("original score is", original_score1)
            #print("new score is", new_score1)
            score_change1 = original_score1 - new_score1

            if score_change1 > max_decrease1:
                max_decrease1 = score_change1
                best_node1 = node
            
            # Track the closest decrease to max_decrease1 or original_score1 if no improvement found
            difference_to_max = abs(max_decrease1 - score_change1)
            difference_to_original = abs(original_score1 - new_score1)
            closest_difference1 = min(difference_to_max, difference_to_original)

            if closest_difference1 < min_difference1:
                min_difference1 = closest_difference1
                closest_node1 = node

        # Decide which node to remove based on the findings
        if best_node1:
            node_to_remove1 = best_node1
        elif closest_node1:
            node_to_remove1 = closest_node1
            print(f"Removing node {closest_node1} with score change closest to max decrease or original score.")
        else:
            print("No beneficial node found to remove, and no nodes left to process.")
            break


        # Apply best removal found for Cs
        #if best_node1:
        swarm_graph.G = original_G1
        swarm_graph.remove_node(node_to_remove1)
        removal_count1 += 1
        swarm_graph.update_positions()
        original_G1 = swarm_graph.get_graph()
        coop_score1, _, _, original_data1 = calculate_coop_score1(original_G1, removal_count1, 0, original_data1)
        coop_scores1.append(coop_score1)
        coop_ratios1.append(coop_score1)# / original_score1 if original_score1 != 0 else 1)
        print(f"Removed node {best_node1} with maximum Cs score decrease.")

        # Testing node removal for Coop
        for node in list(G_copy2.nodes()):
            
            swarm_graph.G = G_copy2.copy()
            swarm_graph.remove_node(node)
            swarm_graph.update_positions()
            temp_graph = swarm_graph.get_graph()
            new_score2, _, _, original_data2 = calculate_coop_score2(temp_graph, removal_count2, original_data2)
            score_change2 = original_score2 - new_score2

            if score_change2 > max_decrease2:
                max_decrease2 = score_change2
                best_node2 = node

            # Track the closest decrease to max_decrease1 or original_score1 if no improvement found
            difference_to_max = abs(max_decrease2 - score_change2)
            difference_to_original = abs(original_score2 - new_score2)
            closest_difference2 = min(difference_to_max, difference_to_original)

            if closest_difference2 < min_difference2:
                min_difference2 = closest_difference2
                closest_node2 = node

        # Decide which node to remove based on the findings
        if best_node2:
            node_to_remove2 = best_node2
        elif closest_node2:
            node_to_remove2 = closest_node2
            print(f"Removing node {closest_node2} with score change closest to max decrease or original score.")
        else:
            print("No beneficial node found to remove, and no nodes left to process.")
            break

        # Apply best removal found for Coop
        #if best_node2:
        swarm_graph.G = original_G2
        swarm_graph.remove_node(node_to_remove2)
        removal_count2 += 1
        swarm_graph.update_positions()
        original_G2 = swarm_graph.get_graph()
        coop_score2, _, _, original_data2 = calculate_coop_score2(original_G2, removal_count2, original_data2)
        coop_scores2.append(coop_score2)
        coop_ratios2.append(coop_score2 / First_Score2 if First_Score2 != 0 else 1)
        print(f"Removed node {best_node2} with maximum COOP score decrease.")

        # Prepare for next iteration
        G_copy1, G_copy2 = original_G1.copy(), original_G2.copy()
        save_graph_snapshot1(G_copy1, model, version, method, run, coop_ratios1[-1], removal_count1, original_data1)
        save_graph_snapshot2(G_copy2, model, version, method, run, coop_ratios2[-1], removal_count2, original_data2)
        removal_count += 1

    return coop_ratios1, coop_ratios2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def create_animation(frame_paths, output_dir, filename):
    frames = []
    for path in frame_paths:
        frames.append(imageio.imread(path))
    animation_path = os.path.join(output_dir, filename)
    imageio.mimsave(animation_path, frames, fps=10)
    print(f"Animation saved as {animation_path}")

def worker(method, run, swarm_graph):
    target = np.array([9, 9, 4])
    model = 'small-world'
    version = 1
    images = []
    results = {
        'method': method,
        'model': model,
        'version': version,
        'run': run,
        'disconnected_components': [],
        'sum_times': 0,
        'weights': 0,
        'sum_weights': 0,
        'coop_ratios1': [],
        'coop_ratios2': [],
        'weight_within_distance': 0,
        'sum_weight_within_distance': 0,
        'reached_percentage': 0,
        'sum_reached_percentage': 0,
        'largest_component_distance': 0,
        'sum_largest_component_distance': 0,
        'largest_component_reached': 0,
        'sum_largest_component_reached': 0

    }

    reset_globals(model, version)
    print(f"Run No. {run+1} in {model} v{version} for {method}")
    G = swarm_graph.get_graph()
    swarm_graph.update_positions()
    G = swarm_graph.G
    G_copy = copy.deepcopy(G)
    start_time = time.perf_counter()
    coop_times = {}
    component_sizes = []  # Initialize component_sizes with a default value
    total_weight = 0      # Initialize total_weight with a default value
    weight_within_distance = 0
    reached_percentage = 0
    largest_component_distance = 0
    largest_component_reached = 0
    sum_weight_within_distance = 0
    removal_count = 0     # Initialize removal_count with a default value

    try:
    
        if method == 'greedy':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios1'], results['coop_ratios2'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths1, frame_paths2, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = greedy_removal_heuristic(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths1 and frame_paths2:# and run == 0:  # Check if there are frames to animate
                animation_filename1 = f'{method}_run_{run+1}_Cs.gif'
                animation_filename2 = f'{method}_run_{run+1}_COOP.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths1, animation_output_dir, animation_filename1)
                create_animation(frame_paths2, animation_output_dir, animation_filename2)
            else:
                print("No frames to animate for method", method)
            #save_individual_run_coop_ratios(method, results, run)
    except Exception as e:
        print(f"Error in method {method}, run {run}: {e}")

    end_time = time.perf_counter()
    results['sum_times'] = end_time - start_time - coop_times.get(method, 0)
    results['disconnected_components'] = [component_sizes]
    results['weights'] = total_weight
    results['weight_within_distance'] = weight_within_distance
    results['reached_percentage'] = reached_percentage
    results['largest_component_distance'] = largest_component_distance
    results['largest_component_reached'] = largest_component_reached

    return results

def execute_worker(args):
    return worker(**args)

def save_disconnected_component_sizes_to_csv(aggregated_results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        max_components = max(len(run_data['disconnected_components']) for model_version, methods_data in aggregated_results.items() for method, runs_data in methods_data.items() for run_data in runs_data)
        headers = ['Method', 'Model_Version', 'Run'] + [f'Disconnected Component Sizes {i + 1}' for i in range(max_components)]
        writer.writerow(headers)
        for model_version, methods_data in aggregated_results.items():
            for method, runs_data in methods_data.items():
                for run_index, run_data in enumerate(runs_data):
                    row = [method, model_version, f'Run {run_index + 1}']
                    if 'disconnected_components' in run_data:
                        row.extend(run_data['disconnected_components'])
                        row.extend(['NA'] * (max_components - len(run_data['disconnected_components'])))
                    else:
                        row.extend(['NA'] * max_components)
                    writer.writerow(row)


def calculate_confidence_interval_errors(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return h

def save_coop_ratios_and_errors_to_csv(model_version, methods_data, str):
    filename = f'Custom_Path/{str}_ratios_{model_version}_with_errors.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Number of Nodes Removed',
                   'Avg greedy Coop Ratio', 'Error greedy'
                   ]
        writer.writerow(headers)
        max_length = max(max(len(ratios) for ratios in method_data) for method_data in methods_data.values())
        for i in range(max_length):
            row = [i + 1]
            for method, data in methods_data.items():
                ratios_at_step = [run[i] for run in data if i < len(run)]
                if ratios_at_step:
                    avg_ratio = np.mean(ratios_at_step)
                    error = calculate_confidence_interval_errors(ratios_at_step)
                    row.extend([avg_ratio, error])
                else:
                    row.extend(['NA', 'NA'])
            writer.writerow(row)

def save_averages_to_csv(averages, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        methods = next(iter(averages.values())).keys()
        header = ['Model_Version'] + list(methods)
        writer.writerow(header)
        for model_version, methods_data in averages.items():
            row = [model_version] + [methods_data[method] for method in methods]
            writer.writerow(row)

def main():
    methods = ['greedy']
    num_runs = 1

    args_list = []
    for run in range(num_runs):
        for method in methods:
            swarm_graph = SwarmGraph(num_nodes=100, area_width=10, area_height=10, min_sep=0.02, max_velocity=0.1, target=(9, 9, 4))
            args = (method, run, swarm_graph)
            args_list.append(args)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(worker, args_list)

    aggregated_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        aggregated_results[model_version][method]['coop_ratios1'].append(result['coop_ratios1'])
        aggregated_results[model_version][method]['coop_ratios2'].append(result['coop_ratios2'])

    aggregated_results = defaultdict(lambda: defaultdict(list))
    
    coop_ratios1_data = defaultdict(lambda: defaultdict(list))
    coop_ratios2_data = defaultdict(lambda: defaultdict(list))

    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        coop_ratios1_data[model_version][method].append(result['coop_ratios1'])
    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        coop_ratios2_data[model_version][method].append(result['coop_ratios2'])

    for model_version, methods_data in coop_ratios1_data.items():
        save_coop_ratios_and_errors_to_csv(model_version, methods_data, 'cs')

    for model_version, methods_data in coop_ratios2_data.items():
        save_coop_ratios_and_errors_to_csv(model_version, methods_data, 'coop')

if __name__ == '__main__':
    main()