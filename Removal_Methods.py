import networkx as nx
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Ensure TensorFlow 1.x compatibility


from SwarmGraph import SwarmGraph
from DQN_train import SumTree, UAVSEnv, DQNPrioritizedReplay, Memory
from utilities import save_graph_snapshot, plot_graph,  calculate_distance_to_target, get_disconnected_component_sizes, geometric_center
from COOP_Calc import dist_Rlink, update_node_weights, calculate_coop_score

def apply_betweenness_heuristic(G, images, model, version, method, swarm_graph,run,target_distance=1.5):
    coop_scores, coop_ratios = [], []
    total_coop_time = 0
    removal_count = 0
    frame_paths = []
    target = np.array([9, 9, 4])
    original_data = None
    initial_num_nodes = G.number_of_nodes()  # Store the initial number of nodes
    save_graph_snapshot(G, model, version, method, run, f"start_{run}")
    swarm_graph.G = G
    swarm_graph.update_positions()
    G = swarm_graph.G
    save_graph_snapshot(G, model, version, method, run, f"start2_{run}")

    # Initialize prev_connections right before starting removal operations
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    weight_within_distance = 0
    reached_target_percentage = 0
    largest_component_distance = float('inf')
    largest_component_reached = 0

    while removal_count < 101:
        # Fetch updated positions
        if 5 <= removal_count < 56:
            betweenness_centrality = nx.betweenness_centrality(G)
            node_to_remove = max(betweenness_centrality, key=betweenness_centrality.get)
            G.remove_node(node_to_remove)

        frame_path = plot_graph(G, removal_count, model, version, method, run)
        frame_paths.append(frame_path)
            
        # Compute new COOP score
        if removal_count == 0:
            original_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            new_coop_score = original_coop_score
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
            #print(coop_ratios[-1])
        else:
            new_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)

        removal_count += 1

        # Store COOP ratio based on updated score
        swarm_graph.G = G
        swarm_graph.update_positions()
        G = swarm_graph.G
    
    #coop_scores = min_max_scaling(coop_scores)
    # Evaluate the largest connected component
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
        largest_component_reached = 1 if distance <= target_distance else 0
    # Calculate the number of nodes within a small distance to the target
    close_to_target_count = sum(1 for node in G.nodes() if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) < 0.5)
    reached_target_percentage = (close_to_target_count / initial_num_nodes) * 100
    weight_within_distance = sum(G.nodes[node]['weight'] for node in G.nodes if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) <= target_distance)
    save_graph_snapshot(G, model, version, method, run, f"end_{run}")
    total_weight = sum(G.nodes[node]['weight'] for node in G)
    component_sizes = get_disconnected_component_sizes(G)
    return coop_ratios, component_sizes, total_weight, removal_count, total_coop_time, frame_paths, weight_within_distance, reached_target_percentage, largest_component_distance, largest_component_reached

def apply_degree_heuristic(G, images, model, version, method, swarm_graph,run,target_distance=1.5):
    coop_scores, coop_ratios = [], []
    total_coop_time = 0
    removal_count = 0
    frame_paths = []
    target = np.array([9, 9, 4])
    original_data = None
    initial_num_nodes = G.number_of_nodes()  # Store the initial number of nodes
    save_graph_snapshot(G, model, version, method, run, f"start_{run}")
    swarm_graph.G = G
    swarm_graph.update_positions()
    G = swarm_graph.G
    save_graph_snapshot(G, model, version, method, run, f"start2_{run}")

    # Initialize prev_connections right before starting removal operations
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    #weight_target_reached = 0.75 * total_weight
    weight_within_distance = 0
    reached_target_percentage = 0
    largest_component_distance = float('inf')
    largest_component_reached = 0

    while removal_count < 101:
        # Fetch updated positions
        
        if 5 <= removal_count < 56:
            degree_centrality = nx.degree_centrality(G)
            node_to_remove = max(degree_centrality, key=degree_centrality.get)
            G.remove_node(node_to_remove)

        frame_path = plot_graph(G, removal_count, model, version, method, run)
        frame_paths.append(frame_path)
            
        # Compute new COOP score
        if removal_count == 0:
            original_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)

            new_coop_score = original_coop_score
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
        else:
            new_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)

        removal_count += 1
        # Store COOP ratio based on updated score
        swarm_graph.G = G
        swarm_graph.update_positions()
        G = swarm_graph.G
    
    #coop_scores = min_max_scaling(coop_scores)
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
        largest_component_reached = 1 if distance <= target_distance else 0
    # Calculate the number of nodes within a small distance to the target
    close_to_target_count = sum(1 for node in G.nodes() if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) < 0.5)
    reached_target_percentage = (close_to_target_count / initial_num_nodes) * 100
    weight_within_distance = sum(G.nodes[node]['weight'] for node in G.nodes if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) <= target_distance)
    save_graph_snapshot(G, model, version, method, run, f"end_{run}")
    total_weight = sum(G.nodes[node]['weight'] for node in G)
    component_sizes = get_disconnected_component_sizes(G)

    return coop_ratios, component_sizes, total_weight, removal_count, total_coop_time, frame_paths, weight_within_distance, reached_target_percentage, largest_component_distance, largest_component_reached

def apply_closeness_heuristic(G, images, model, version, method, swarm_graph,run,target_distance=1.5):
    coop_scores, coop_ratios = [], []
    total_coop_time = 0
    removal_count = 0
    frame_paths = []
    target = np.array([9, 9, 4])
    original_data = None
    initial_num_nodes = G.number_of_nodes()  # Store the initial number of nodes
    save_graph_snapshot(G, model, version, method, run, f"start_{run}")
    swarm_graph.G = G
    swarm_graph.update_positions()
    G = swarm_graph.G
    save_graph_snapshot(G, model, version, method, run, f"start2_{run}")

    # Initialize prev_connections right before starting removal operations
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    weight_within_distance = 0
    reached_target_percentage = 0
    largest_component_distance = float('inf')
    largest_component_reached = 0

    while removal_count < 101:
        # Fetch updated positions
        
        if 5 <= removal_count < 56:
            closeness_centrality = nx.closeness_centrality(G)
            node_to_remove = max(closeness_centrality, key=closeness_centrality.get)
            G.remove_node(node_to_remove)

        frame_path = plot_graph(G, removal_count, model, version, method, run)
        frame_paths.append(frame_path)
            
        # Compute new COOP score
        if removal_count == 0:
            original_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            #print('original_coop is', original_coop_score)
            new_coop_score = original_coop_score
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
            #print(coop_ratios[-1])
        else:
            new_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)

        removal_count += 1
        # Store COOP ratio based on updated score
        swarm_graph.G = G
        swarm_graph.update_positions()
        G = swarm_graph.G
    
    #coop_scores = min_max_scaling(coop_scores)
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
        largest_component_reached = 1 if distance <= target_distance else 0
    # Calculate the number of nodes within a small distance to the target
    close_to_target_count = sum(1 for node in G.nodes() if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) < 0.5)
    reached_target_percentage = (close_to_target_count / initial_num_nodes) * 100
    weight_within_distance = sum(G.nodes[node]['weight'] for node in G.nodes if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) <= target_distance)
    save_graph_snapshot(G, model, version, method, run, f"end_{run}")
    total_weight = sum(G.nodes[node]['weight'] for node in G)
    component_sizes = get_disconnected_component_sizes(G)

    return coop_ratios, component_sizes, total_weight, removal_count, total_coop_time, frame_paths, weight_within_distance, reached_target_percentage, largest_component_distance, largest_component_reached

def dqn_coop_ratios(G, images, model, version, method, swarm_graph, run,target_distance=1.5):
    coop_scores = []
    coop_ratios = []
    total_coop_time = 0
    removal_count = 0
    frame_paths = []
    original_data = None
    target = np.array([9, 9, 4])
    initial_num_nodes = G.number_of_nodes()  # Store the initial number of nodes
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}

    tf.compat.v1.disable_eager_execution()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # Allows dynamic growth of the memory
    with tf.compat.v1.Session(config=config) as session:
        with tf.compat.v1.variable_scope('DQN_with_prioritized_replay', reuse=tf.compat.v1.AUTO_REUSE):
            dqn_model = DQNPrioritizedReplay(
                n_actions=50,
                n_features=2,
                n_embedding=32,
                learning_rate=0.0005,
                reward_decay=0.9,
                e_greedy=0.9,
                e_greedy_increment=0.00012,
                replace_target_iter=1000,
                memory_size=5000,
                batch_size=32,
                prioritized=True,
                output_graph=True,
                sess=session
            )
            # Initialize variables and restore model
            session.run(tf.compat.v1.global_variables_initializer())
            checkpoint_dir = 'Custom_Path/'
            checkpoint_prefix = 'nrange_50_iter_14294.ckpt'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_prefix)
            saver = tf.compat.v1.train.Saver()
            saver.restore(session, checkpoint_path)
            saver.restore(session, checkpoint_path)
            save_graph_snapshot(G, model, version, method, run, f"start_{run}")
            swarm_graph.G = G
            swarm_graph.update_positions()
            G = swarm_graph.G
            save_graph_snapshot(G, model, version, method, run, f"start2_{run}")
            
            #weight_target_reached = 0.75 * total_weight
            weight_within_distance = 0
            reached_target_percentage = 0
            largest_component_distance = float('inf')
            largest_component_reached = 0

            while removal_count < 101:
                # Fetch updated positions
                
                if 5 <= removal_count < 56:
                    # Disabling eager execution should happen once per session creation
                    
                    env = UAVSEnv()
                    current_state = env.state0(G)
                    action_to_remove = dqn_model.choose_action(current_state, list(G.nodes()))

                    if action_to_remove in G.nodes():
                        if G.nodes[action_to_remove]['weight'] == 1:
                            removal_info['DQN'].append((action_to_remove, removal_count + 1))
                        G.remove_node(action_to_remove)


                frame_path = plot_graph(G, removal_count, model, version, method, run)
                frame_paths.append(frame_path)
                    
                # Compute new COOP score
                if removal_count == 0:
                    original_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
                    new_coop_score = original_coop_score
                    total_coop_time += single_coop_time
                    coop_scores.append(new_coop_score)
                    coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
                else:
                    new_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
                    total_coop_time += single_coop_time
                    coop_scores.append(new_coop_score)
                    coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
                removal_count += 1
                # Store COOP ratio based on updated score
                swarm_graph.G = G
                swarm_graph.update_positions()
                G = swarm_graph.G
    
    #coop_scores =min_max_scaling(coop_scores)
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
        largest_component_reached = 1 if distance <= target_distance else 0
    # Calculate the number of nodes within a small distance to the target
    close_to_target_count = sum(1 for node in G.nodes() if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) < 0.5)
    reached_target_percentage = (close_to_target_count / initial_num_nodes) * 100
    weight_within_distance = sum(G.nodes[node]['weight'] for node in G.nodes if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) <= target_distance)

    save_graph_snapshot(G, model, version, method, run, f"end_{run}")
    total_weight = sum(G.nodes[node]['weight'] for node in G)
    component_sizes = get_disconnected_component_sizes(G)

    return coop_ratios, component_sizes, total_weight, removal_count, total_coop_time, frame_paths, weight_within_distance, reached_target_percentage, largest_component_distance, largest_component_reached

def apply_weight_heuristic(G, images, model, version, method, swarm_graph, run, target_distance=1.5):
    # Setup for logging and output directory (if needed)
    coop_scores, coop_ratios = [], []
    total_coop_time = 0
    removal_count = 0
    frame_paths = []
    target = np.array([9, 9, 4])
    original_data = None
    initial_num_nodes = G.number_of_nodes()  # Store the initial number of nodes

    # Save initial graph state
    save_graph_snapshot(G, model, version, method, run, f"start_{run}")
    swarm_graph.G = G
    swarm_graph.update_positions()
    G = swarm_graph.G
    save_graph_snapshot(G, model, version, method, run, f"start2_{run}")

    # Initialize prev_connections right before starting removal operations
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}

    # Initial cooperative score calculation
    weight_within_distance = 0
    reached_target_percentage = 0
    largest_component_distance = float('inf')
    largest_component_reached = 0
    while removal_count < 101:
        if 5 <= removal_count < 56:
            # Select the node with the highest weight
            node_weights = nx.get_node_attributes(G, 'weight')
            if node_weights:
                node_to_remove = max(node_weights, key=node_weights.get)
                G.remove_node(node_to_remove)

        # Visualization of the graph after each removal
        frame_path = plot_graph(G, removal_count, model, version, method, run)
        frame_paths.append(frame_path)
            
        # Compute new COOP score
        if removal_count == 0:
            original_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            new_coop_score = original_coop_score
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
            #print(coop_ratios[-1])
        else:
            new_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
        removal_count += 1

        # Update the graph's positions reflecting the latest changes
        swarm_graph.G = G.copy()
        swarm_graph.update_positions()
        G = swarm_graph.G

    #coop_scores = min_max_scaling(coop_scores)
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
        largest_component_reached = 1 if distance <= target_distance else 0
    # Calculate the number of nodes within a small distance to the target
    close_to_target_count = sum(1 for node in G.nodes() if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) < 0.5)
    reached_target_percentage = (close_to_target_count / initial_num_nodes) * 100
    # Final state and metrics
    weight_within_distance = sum(G.nodes[node]['weight'] for node in G.nodes if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) <= target_distance)
    save_graph_snapshot(G, model, version, method, run, f"end_{run}")
    total_weight = sum(nx.get_node_attributes(G, 'weight').values())
    component_sizes = get_disconnected_component_sizes(G)

    return coop_ratios, component_sizes, total_weight, removal_count, total_coop_time, frame_paths, weight_within_distance, reached_target_percentage, largest_component_distance, largest_component_reached

def remove_node_by_degree_and_weight(G, images, model, version, method, swarm_graph, run, target_distance=1.5):
    coop_scores, coop_ratios = [], []
    total_coop_time = 0
    removal_count = 0
    frame_paths = []
    original_data = None
    target = np.array([9, 9, 4])
    initial_num_nodes = G.number_of_nodes()  # Store the initial number of nodes

    save_graph_snapshot(G, model, version, method, run, f"start_{run}")
    swarm_graph.G = G
    swarm_graph.update_positions()
    G = swarm_graph.G
    save_graph_snapshot(G, model, version, method, run, f"start2_{run}")

    # Initialize prev_connections right before starting removal operations
    prev_connections = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    weight_within_distance = 0
    reached_target_percentage = 0
    largest_component_distance = float('inf')
    largest_component_reached = 0
    while removal_count < 101:
        if 5 <= removal_count < 56:
            # Compute closeness centrality for all nodes
            degree_centrality = nx.degree_centrality(G)

            # Sort nodes based on closeness centrality and select the top three
            top_degree_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:3]

            # Get node weights
            node_weights = nx.get_node_attributes(G, 'weight')

            # Find the node with the highest weight among the top three closeness nodes
            if top_degree_nodes:
                node_to_remove = max(top_degree_nodes, key=lambda node: node_weights.get(node, 0))

            # Remove the selected node
            if G.has_node(node_to_remove):
                G.remove_node(node_to_remove)

        frame_path = plot_graph(G, removal_count, model, version, method, run)
        frame_paths.append(frame_path)
            
        # Compute new COOP score
        if removal_count == 0:
            original_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            #print('original_coop is', original_coop_score)
            new_coop_score = original_coop_score
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
            #print(coop_ratios[-1])
        else:
            new_coop_score, single_coop_time, component_factor, original_data = calculate_coop_score(G, removal_count, prev_connections, original_data)
            total_coop_time += single_coop_time
            coop_scores.append(new_coop_score)
            coop_ratios.append(new_coop_score / original_coop_score if original_coop_score > 0 else 1)
        removal_count += 1

        swarm_graph.G = G
        swarm_graph.update_positions()
        G = swarm_graph.G

    #coop_scores = min_max_scaling(coop_scores)
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
        center_of_mass = geometric_center(largest_component, nx.get_node_attributes(G, 'pos'))
        distance = calculate_distance_to_target(center_of_mass, target)
        largest_component_distance = min(largest_component_distance, distance)
        largest_component_reached = 1 if distance <= target_distance else 0
    # Calculate the number of nodes within a small distance to the target
    close_to_target_count = sum(1 for node in G.nodes() if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) < 0.5)
    reached_target_percentage = (close_to_target_count / initial_num_nodes) * 100
    weight_within_distance = sum(G.nodes[node]['weight'] for node in G.nodes if np.linalg.norm(np.array(G.nodes[node]['pos']) - target) <= target_distance)
    save_graph_snapshot(G, model, version, method, run, f"end_{run}")
    total_weight = sum(G.nodes[node]['weight'] for node in G)
    component_sizes = get_disconnected_component_sizes(G)

    return coop_ratios, component_sizes, total_weight, removal_count, total_coop_time, frame_paths, weight_within_distance, reached_target_percentage, largest_component_distance, largest_component_reached