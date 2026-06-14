import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import os
import scipy.stats as st
from scipy.stats import sem, t
import operator
import csv
import io
import imageio
import time
import statistics
from scipy.stats import kurtosis
import math
import itertools
import random
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import copy
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

def plot_graph(G, frame_number, model, version, method, run):
    output_dir2 = f'Custom_Path//{model}_v{version}/{method}/{run}'
    os.makedirs(output_dir2, exist_ok=True)
    pos = nx.get_node_attributes(G, 'pos')
    min_size = 50  # Minimum size for visibility
    sizes = [max(min_size, 10 * G.nodes[node]['weight']) for node in G]  # Ensure all nodes are at least min_size
    target = np.array([9,9,4])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    xs = [pos[node][0] for node in G]
    ys = [pos[node][1] for node in G]
    zs = [pos[node][2] for node in G]

    ax.scatter(xs, ys, zs, s=sizes, c=sizes, cmap=plt.cm.viridis, depthshade=True)

    # Plot edges
    edge_width = 0.7  # Adjust edge width as needed
    for edge in G.edges:
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, 'black', linewidth=edge_width)  # Color and width can be adjusted

    # Plot target
    ax.scatter([target[0]], [target[1]], [target[2]], color='red', s=100, label='Target', depthshade=True)

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"Frame {frame_number} - Drones: {len(G.nodes)}")
    plt.legend()
    
    frame_path = os.path.join(output_dir2, f"frame_{frame_number}.png")
    plt.savefig(frame_path)
    plt.close()
    return frame_path


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

def select_component_based_on_proximity(G, components, target, communication_range):
    if len(components) < 2:
        return components[0] if components else None

    # Calculate the number of nodes in the top two components
    size_largest = len(components[0])
    size_second_largest = len(components[1])

    # Check if the largest component is at least twice as large as the second largest
    if size_largest >= 2 * size_second_largest:
        return components[0]  # Select the largest regardless of proximity to the target

    # Calculate average distances of each component to the target if sizes are close
    avg_dists = [calculate_average_distance_to_target(comp, target, G) for comp in components]
    distance_between = distance_between_components(G, components[0], components[1])

    # Select based on proximity if the size condition is not met
    if distance_between > communication_range:
        return components[np.argmin(avg_dists)]
    return components[0]  # Select the largest if they are close enough


def component_size_difference(G):
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) >= 2:
        return len(components[0]) - len(components[1])
    elif components:
        return len(components[0])  # If there's only one component, return its size
    return 0  # No components

def get_node_color(weight):
    # Define ranges and their colors
    color_ranges = [
        (0, 0, 'white'),
        (0.01, 0.2, 'pink'),
        (0.2, 0.4, 'violet'),
        (0.4, 0.6, 'purple'),
        (0.6, 0.8, 'blue'),
        (0.8, 1, 'green'),
        (1, 2, 'yellow'),
        (2, 4, 'orange'),
        (4, 6, 'red'),
        (6, 10, 'brown'),
        (10, np.inf, 'black')
    ]
    
    # Determine color based on the weight
    for (start, end, color) in color_ranges:
        if start <= weight < end:
            return color
    return "black"  # Default color for weights outside the specified ranges

def save_graph_snapshot(G, model, version, method, run, removal_count):
    output_dir = f'Custom_Path//{model}_v{version}/{method}/{run}'
    os.makedirs(output_dir, exist_ok=True)
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    largest_subgraph = G.subgraph(largest_component)
    pos = nx.kamada_kawai_layout(largest_subgraph)  # Get positions for all nodes using spring layout
    plt.figure(figsize=(4, 4))
    
    # Extract weights and corresponding colors
    node_weights = nx.get_node_attributes(largest_subgraph, 'weight')
    node_colors = [get_node_color(weight) for weight in node_weights.values()]
    color_ranges = [
        (0, 0, 'white'),
        (0.01, 0.2, 'pink'),
        (0.2, 0.4, 'violet'),
        (0.4, 0.6, 'purple'),
        (0.6, 0.8, 'blue'),
        (0.8, 1, 'green'),
        (1, 2, 'yellow'),
        (2, 4, 'orange'),
        (4, 6, 'red'),
        (6, 10, 'brown'),
        (10, np.inf, 'black')
    ]

    # Draw the graph
    nx.draw(largest_subgraph, pos, node_size=50, with_labels=False, node_color=node_colors, edge_color='gray')

    # Create a legend for the colors
    legend_handles = [mpatches.Patch(color=color, label=f'{start} - {end}') for (start, end, color) in color_ranges]
    plt.legend(handles=legend_handles, title="Threat Level", bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=14, fontsize=12)

    #plt.title(f"Graph Snapshot at Step {removal_count}")
    plt.savefig(os.path.join(output_dir, f"{method}_step_{removal_count}.png"), bbox_inches='tight')
    plt.close()

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

def min_max_scaling(coop_scores):
    min_score = min(coop_scores)
    max_score = max(coop_scores)
    if max_score - min_score == 0:  # Avoid division by zero
        return [0] * len(coop_scores)  # or handle differently if max_score is always equal to min_score
    return [(x - min_score) / (max_score - min_score) for x in coop_scores]

def create_animation(frame_paths, output_dir, filename):
    frames = []
    for path in frame_paths:
        frames.append(imageio.imread(path))
    animation_path = os.path.join(output_dir, filename)
    imageio.mimsave(animation_path, frames, fps=10)
    print(f"Animation saved as {animation_path}")

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

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    margin = std_err * t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - margin, mean + margin

def save_coop_ratios_and_errors_to_csv(model_version, methods_data):
    filename = f'Custom_Path/coop_ratios_{model_version}_with_errors.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Number of Nodes Removed',
                   'Avg Betweenness Coop Ratio', 'Error Betweenness',
                   'Avg Degree Coop Ratio', 'Error Degree',
                   'Avg Closeness Coop Ratio', 'Error Closeness',
                   'Avg Weight Coop Ratio', 'Error Weight',
                   'Avg DegWeight Coop Ratio', 'Error DegWeight',
                   'Avg DQN Coop Ratio', 'Error DQN'
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

def save_individual_run_coop_ratios(method, run_data, run_index):
    output_dir = f'Custom_Path/{method}'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'coop_ratios_run_{run_index+1}_details.csv')

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Number of Nodes Removed', f'Coop Ratio Run {run_index+1}', f'Error Run {run_index+1}']
        writer.writerow(headers)
        for i, ratio in enumerate(run_data['coop_ratios']):
            error = calculate_confidence_interval_errors([ratio])
            writer.writerow([i + 1, ratio, error])

def save_averages_to_csv(averages, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        methods = next(iter(averages.values())).keys()
        header = ['Model_Version'] + list(methods)
        writer.writerow(header)
        for model_version, methods_data in averages.items():
            row = [model_version] + [methods_data[method] for method in methods]
            writer.writerow(row)