import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class SwarmGraph:
    def __init__(self, num_nodes=100, area_width=10, area_height=10, area_depth=10, min_sep=0.02, max_velocity=0.1, target=(9, 9, 4), communication_range=1, m_links=4):
        self.num_nodes = num_nodes
        self.area_width = area_width
        self.area_height = area_height
        self.area_depth = area_depth
        self.min_sep = min_sep
        self.max_velocity = max_velocity
        self.target = np.array(target)
        self.communication_range = communication_range
        self.m_links = m_links  # Number of edges each new node will attach to existing nodes
        self.G = nx.Graph()  # Start with an empty graph
        self.initial_positions = {}  # Store initial positions to maintain relative distances
        self.initialize_graph()
        self.add_ba_edges()

    def initialize_graph(self):
        grid_size = int(round(self.num_nodes ** (1/3)))
        x_spacing = 0.2
        y_spacing = 0.2
        z_spacing = 0.1

        index = 0
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if index >= self.num_nodes:
                        break
                    pos = np.array([x * x_spacing, y * y_spacing, 4 + z * z_spacing])
                    weight = random.uniform(0.1, 1.0)
                    Theta = weight
                    velocity = np.random.uniform(-self.max_velocity, self.max_velocity, 3)
                    self.G.add_node(index, pos=pos, weight=weight, velocity=velocity, pbest=pos.copy(), Theta=Theta)
                    self.initial_positions[index] = pos
                    index += 1

    def add_ba_edges(self):
        # Start with a small number of interconnected nodes
        initial_nodes = min(self.num_nodes, self.m_links + 1)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                if i in self.G and j in self.G:
                    if self.G.degree(i) < 5 and self.G.degree(j) < 5:
                        if self.should_add_edge(self.initial_positions[i], self.initial_positions[j]):
                            self.G.add_edge(i, j)

        # Preferential attachment for each new node
        for new_node in range(initial_nodes, self.num_nodes):
            if new_node in self.G:  # Ensure the new node still exists in the graph
                targets = self._preferential_attachment_targets(new_node)
                for target in targets:
                    if target in self.G and self.G.degree(new_node) < 5 and self.G.degree(target) < 5:
                        if self.should_add_edge(self.initial_positions[new_node], self.initial_positions[target]):
                            self.G.add_edge(new_node, target)

    def should_add_edge(self, pos1, pos2):
        distance = np.linalg.norm(pos1 - pos2)
        return self.calculate_reliability(distance) > 0

    def calculate_reliability(self, distance):
        d_min = 0.05
        if distance <= d_min:
            return 1
        elif distance > d_min and distance <= self.communication_range:
            return (d_min / distance) ** 2
        return 0

    def _preferential_attachment_targets(self, new_node):
        targets = set()
        total_degree = sum(deg for node, deg in self.G.degree() if node in self.G)  # Calculate degree only for existing nodes
        while len(targets) < self.m_links:
            if total_degree == 0:
                break
            potential_nodes = [node for node in self.G.nodes()]
            rand_node = random.choice(potential_nodes)
            rand_node_degree = self.G.degree(rand_node)
            attachment_prob = rand_node_degree / total_degree
            if random.random() < attachment_prob and rand_node != new_node:
                targets.add(rand_node)
        return targets

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
        #self.add_ba_edges() # Uncomment for Dynamic Case


    def get_graph(self):
        return self.G