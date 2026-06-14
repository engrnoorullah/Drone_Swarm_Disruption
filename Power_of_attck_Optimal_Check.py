
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import os
import scipy.stats as st
from scipy.stats import sem, t
import operator
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager
import csv
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
from functools import partial

from SwarmGraph import SwarmGraph
from COOP_Calc import dist_Rlink, update_node_weights, calculate_coop_score

#np.random.seed(1)

# Parameters
NUM_DRONES = 25
AREA_SIZE = 10
communication_range = 1
removed_nodes = []
max_connections = NUM_DRONES-1
#NUM_DRONES = 55
#TARGET = [9, 9]
SPEED = 0.05
SEPARATION = 0.002
stopper = 0

AREA_WIDTH = 10
AREA_HEIGHT = 10
AREA_DEPTH = 10
MIN_SEPARATION = 0.02  # Minimum distance between drones adjusted for your area size
MAX_VELOCITY = 0.1  # Maximum velocity component in any direction
TARGET = np.array([9, 9, 4])

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

def calculate_distance_to_target(center, target):
    return np.linalg.norm(center - target)

def calculate_power_of_attack(G, target):
    components = list(nx.connected_components(G))
    total_weighted_distance = 0
    num_components = 0  # Number of components with more than one node

    for component in components:
        if len(component) > 1:
            num_components += 1
            component_weighted_distance = sum(G.nodes[node]['weight'] / np.linalg.norm(np.array(G.nodes[node]['pos']) - target) for node in component)
            total_weighted_distance += component_weighted_distance
    
    if num_components > 0:
        return total_weighted_distance / num_components
    return 0

def worker(graph, combination, target, swarm_graph):
    graph_copy = graph.copy()
    original_data = None
    for node in combination:
        swarm_graph.update_positions()  # Adjusting positions
        swarm_graph.remove_node(node)  # Adjusting node removal
        graph_copy = swarm_graph.get_graph()  # Getting updated graph
        coop_score, _, _, original_data = calculate_coop_score(graph_copy, len(combination), {}, original_data)
    
    power_of_attack = calculate_power_of_attack(graph_copy, target)
    return (coop_score, power_of_attack, combination)

def execute_combinations(graph, target, swarm_graph):
    nodes = list(graph.nodes())
    combinations = itertools.combinations(nodes, 4)
    worker_with_graph = partial(worker, swarm_graph=swarm_graph)  # Correctly bind swarm_graph

    # Prepare arguments for each worker task
    tasks = [(graph, combination, target) for combination in combinations]

    with Pool(processes=mp.cpu_count()) as pool:
        # Map each task to the worker
        results = pool.starmap(worker_with_graph, tasks)
    return results

def execute_worker(args):
    return worker(**args)

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Coop Score', 'Power of Attack', 'Nodes Removed'])
        for coop_score, power_of_attack, nodes_removed in results:
            writer.writerow([coop_score, power_of_attack, list(nodes_removed)])

def main():
    num_nodes = 25
    target = np.array([9, 9, 4])
    swarm_graph = SwarmGraph(num_nodes=num_nodes, area_width=10, area_height=10, area_depth=10, min_sep=0.02, max_velocity=0.1, target=target)
    G = swarm_graph.get_graph()

    results = execute_combinations(G, target, swarm_graph)  # Pass swarm_graph
    output_directory = 'Custom_Path'
    os.makedirs(output_directory, exist_ok=True)
    filename = os.path.join(output_directory, 'results.csv')
    save_results_to_csv(results, filename)

if __name__ == '__main__':
    main()