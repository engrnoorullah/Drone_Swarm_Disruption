import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class SwarmGraph:
    def __init__(self, num_nodes=100, area_width=10, area_height=10, area_depth=10, min_sep=0.02, max_velocity=0.1, target=(9, 9, 4), communication_range=1):
        self.num_nodes = num_nodes
        self.area_width = area_width
        self.area_height = area_height
        self.area_depth = area_depth
        self.min_sep = min_sep
        self.max_velocity = max_velocity
        self.target = np.array(target)
        self.communication_range = communication_range
        self.G = nx.Graph()
        self.initialize_graph()

    def initialize_graph(self):
        # Calculate grid size along each dimension
        grid_size = int(round(self.num_nodes ** (1/3)))
        # Calculate spacing based on the minimum dimension to fit all nodes
        x_spacing = 0.2
        y_spacing = 0.2
        z_spacing = 0.1

        index = 0
        positions = {}

        # Create grid positions
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if index >= self.num_nodes:
                        break
                    z_coordinate = 4 + z * z_spacing
                    pos = np.array([x * x_spacing, y * y_spacing, z_coordinate])

                    # Check for minimum separation
                    if not any(np.linalg.norm(pos - p) < self.min_sep for p in positions.values()):
                        weight = random.uniform(0.1, 1.0)
                        Theta = weight
                        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, 3)
                        self.G.add_node(index, pos=pos, weight=weight, velocity=velocity, pbest=pos.copy(), Theta=Theta)
                        positions[index] = pos

                    index += 1

        self.add_small_world_edges()

    def add_small_world_edges(self):
        # Adding edges with Small-World connectivity
        nodes = list(self.G.nodes())
        k = 7  # Each node connects to k nearest neighbors initially
        p = 0.1  # Small world rewire probability
        max_connections = 5  # Maximum connections a node can have

        for node in nodes:
            # Ensure the node index exists and does not exceed the node list
            potential_neighbors = [(node + i, node - i) for i in range(1, k//2 + 1)]
            potential_neighbors = [(n1, n2) for n1, n2 in potential_neighbors if n1 in nodes and n2 in nodes]

            for n1, n2 in potential_neighbors:
                # Check and add edge if possible without exceeding the max_connections limit
                if n1 in nodes and self.should_add_edge(node, n1) and self.G.degree(node) < max_connections:
                    self.G.add_edge(node, n1)
                if n2 in nodes and self.should_add_edge(node, n2) and self.G.degree(node) < max_connections:
                    self.G.add_edge(node, n2)

            # Rewiring part using a static list of neighbors
            current_neighbors = list(self.G.neighbors(node))  # Static list of current neighbors
            for neighbor in current_neighbors:
                if random.random() < p and self.G.degree(node) < max_connections:
                    new_neighbor = random.choice(nodes)
                    # Ensure the new neighbor is not already connected and is not the same node
                    if new_neighbor not in current_neighbors and new_neighbor != node:
                        # Check if the node is already at its connection limit before rewiring
                        if self.G.degree(node) < max_connections:
                            self.G.remove_edge(node, neighbor)
                            # Add new edge if it's still under the limit after removing one
                            if self.G.degree(node) < max_connections and self.should_add_edge(node, new_neighbor):
                                self.G.add_edge(node, new_neighbor)

    def should_add_edge(self, node1, node2):
        distance = np.linalg.norm(self.G.nodes[node1]['pos'] - self.G.nodes[node2]['pos'])
        return self.calculate_reliability(distance) > 0

    def calculate_reliability(self, distance):
        d_min = 0.05
        if distance <= d_min:
            return 1
        elif distance > d_min and distance <= self.communication_range:
            return (d_min / distance) ** 2
        return 0

    def remove_node(self, node):
        """Remove a node from the graph and update related structures."""
        if node in self.G:
            self.G.remove_node(node)

    def update_positions(self):
        # Calculate the center of mass based on current positions, not initial positions
        center_of_mass = np.mean([self.G.nodes[node]['pos'] for node in self.G.nodes()], axis=0)
        direction = self.target - center_of_mass
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Calculate movement vector
            movement = (direction / distance) * min(distance, self.max_velocity)
            #print("Calculated movement:", movement)

            # Update positions
            for node in self.G.nodes():
                current_pos = self.G.nodes[node]['pos']
                new_pos = current_pos + movement
                self.G.nodes[node]['pos'] = np.clip(new_pos, [0, 0, 0], [self.area_width, self.area_height, self.area_depth])
        else:
            print("No movement needed, target center of mass reached.")
        #self.add_small_world_edges() # Uncomment for Dynamic Case

    def get_graph(self):
        return self.G