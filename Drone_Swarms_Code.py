import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from dqn_batch import DQNPrioritizedReplay
from uavs_env import UAVSEnv
import os
import itertools
import random
import copy
import time
import multiprocessing as mp
from collections import defaultdict
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Ensure TensorFlow 1.x compatibility

#np.random.seed(1)

# Parameters
NUM_DRONES = 101
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
MIN_SEPARATION = 0.02
MAX_VELOCITY = 2  # Maximum velocity component in any direction
TARGET = np.array([9, 9, 4])

n_actions=50
n_features=2
n_embedding=32
learning_rate=0.0005
reward_decay=0.5
e_greedy=0.9
e_greedy_increment=0.00012
replace_target_iter=1000
memory_size=5000
batch_size=32
#e_greedy_increment=None
#e_greedy_increment=0.00015
prioritized=True
output_graph=True

from utilities import create_animation, save_individual_run_coop_ratios, save_disconnected_component_sizes_to_csv, save_averages_to_csv, save_coop_ratios_and_errors_to_csv
from Removal_Methods import apply_betweenness_heuristic, apply_degree_heuristic, apply_closeness_heuristic, apply_weight_heuristic, remove_node_by_degree_and_weight, dqn_coop_ratios
from SwarmGraph import SwarmGraph

population_size = 100
iterations = 200
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
    all_dqn_coop_ratios = []
    sum_dqn_ratios = {}
    dqn_ratios = []
    critical_nodes = []
    critical_nodes_to_remove = []
    stopper = 0
    final_time_coop_func = 0


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
        'coop_ratios': [],
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
        
        if method == 'betweenness':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = apply_betweenness_heuristic(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths:# and run == 0:  # Check if there are frames to animate
                animation_filename = f'{method}_run_{run+1}.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths, animation_output_dir, animation_filename)
            else:
                print("No frames to animate for method", method)
            save_individual_run_coop_ratios(method, results, run)
        
        elif method == 'degree':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = apply_degree_heuristic(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths:# and run == 0:  # Check if there are frames to animate
                animation_filename = f'{method}_run_{run+1}.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths, animation_output_dir, animation_filename)
            else:
                print("No frames to animate for method", method)
            save_individual_run_coop_ratios(method, results, run)
        
        elif method == 'closeness':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = apply_closeness_heuristic(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths:# and run == 0:  # Check if there are frames to animate
                animation_filename = f'{method}_run_{run+1}.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths, animation_output_dir, animation_filename)
            else:
                print("No frames to animate for method", method)
            save_individual_run_coop_ratios(method, results, run)

        elif method == 'weight':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = apply_weight_heuristic(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths:# and run == 0:  # Check if there are frames to animate
                animation_filename = f'{method}_run_{run+1}.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths, animation_output_dir, animation_filename)
            else:
                print("No frames to animate for method", method)
            save_individual_run_coop_ratios(method, results, run)

        elif method == 'DegWeight':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = remove_node_by_degree_and_weight(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths:# and run == 0:  # Check if there are frames to animate
                animation_filename = f'{method}_run_{run+1}.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths, animation_output_dir, animation_filename)
            else:
                print("No frames to animate for method", method)
            save_individual_run_coop_ratios(method, results, run)
        
        elif method == 'dqn':
            G_copy = copy.deepcopy(G_copy)
            results['coop_ratios'], component_sizes, total_weight, removal_count, coop_times[method], frame_paths, weight_within_distance, reached_percentage, largest_component_distance, largest_component_reached = dqn_coop_ratios(G_copy, images, model, version, method, swarm_graph, run)
            if frame_paths:# and run == 0:  # Check if there are frames to animate
                animation_filename = f'{method}_run_{run+1}.gif'
                animation_output_dir = f'Custom_Path/{model}_v{version}/{method}/{run}'
                create_animation(frame_paths, animation_output_dir, animation_filename)
            else:
                print("No frames to animate for method", method)
            save_individual_run_coop_ratios(method, results, run)
        
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

def main():
    methods = ['betweenness', 'degree', 'closeness', 'weight', 'DegWeight', 'dqn']
    num_runs = 1

    args_list = []
    for run in range(num_runs):
        for method in methods:
            swarm_graph = SwarmGraph(num_nodes=101, area_width=10, area_height=10, min_sep=0.02, max_velocity=2, target=(9, 9, 4))
            args = (method, run, swarm_graph)
            args_list.append(args)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(worker, args_list)

    aggregated_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        aggregated_results[model_version][method]['coop_ratios'].append(result['coop_ratios'])

    #for model_version, methods_data in aggregated_results.items():
    #    plot_results_with_error_bars(model_version, methods_data)

    aggregated_results = defaultdict(lambda: defaultdict(list))

    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        run_number = result['run']
        aggregated_results[model_version][method].append({
            'run': run_number,
            'disconnected_components': result['disconnected_components']
        })

    sizes_filename = f'Custom_Path/disconnected_component_sizes.csv'
    save_disconnected_component_sizes_to_csv(aggregated_results, sizes_filename)

    sum_weights = defaultdict(lambda: defaultdict(float))
    sum_times = defaultdict(lambda: defaultdict(float))
    sum_weight_within_distance = defaultdict(lambda: defaultdict(float))
    sum_reached_percentage = defaultdict(lambda: defaultdict(float))
    sum_largest_component_distance = defaultdict(lambda: defaultdict(float))
    sum_largest_component_reached = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        sum_weights[model_version][method] += result['weights']
        sum_times[model_version][method] += result['sum_times']
        sum_weight_within_distance[model_version][method] += result['weight_within_distance']
        sum_reached_percentage[model_version][method] += result['reached_percentage']
        sum_largest_component_distance[model_version][method] += result['largest_component_distance']
        sum_largest_component_reached[model_version][method] += result['largest_component_reached']
        counts[model_version][method] += 1

    average_weights = {model_version: {method: total / counts[model_version][method] for method, total in methods.items()} for model_version, methods in sum_weights.items()}
    average_times = {model_version: {method: total / counts[model_version][method] for method, total in methods.items()} for model_version, methods in sum_times.items()}
    average_weight_within_distance = {model_version: {method: total / counts[model_version][method] for method, total in methods.items()} for model_version, methods in sum_weight_within_distance.items()}
    average_reached_percentage = {model_version: {method: total / counts[model_version][method] for method, total in methods.items()} for model_version, methods in sum_reached_percentage.items()}
    average_largest_component_distance = {model_version: {method: total / counts[model_version][method] for method, total in methods.items()} for model_version, methods in sum_largest_component_distance.items()}
    average_largest_component_reached = {model_version: {method: total / counts[model_version][method] for method, total in methods.items()} for model_version, methods in sum_largest_component_reached.items()}
    save_averages_to_csv(average_weights, f'Custom_Path/average_weights.csv')
    save_averages_to_csv(average_times, f'Custom_Path/average_times.csv')
    save_averages_to_csv(average_weight_within_distance, f'Custom_Path/average_weight_within_distance.csv')
    save_averages_to_csv(average_reached_percentage, f'Custom_Path/average_reached_percengate.csv')
    save_averages_to_csv(average_largest_component_distance, f'Custom_Path/average_largest_component_distance.csv')
    save_averages_to_csv(average_largest_component_reached, f'Custom_Path/average_largest_component_reached.csv')

    coop_ratios_data = defaultdict(lambda: defaultdict(list))

    for result in results:
        model_version = f"{result['model']}_v{result['version']}"
        method = result['method']
        coop_ratios_data[model_version][method].append(result['coop_ratios'])

    for model_version, methods_data in coop_ratios_data.items():
        save_coop_ratios_and_errors_to_csv(model_version, methods_data)

if __name__ == '__main__':
    main()