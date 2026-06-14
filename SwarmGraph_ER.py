import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class SwarmGraph:
    def __init__(self, num_nodes=100, area_width=10, area_height=10, area_depth=10, min_sep=0.02, max_velocity=0.1, target=(9, 9, 4), communication_range=1, p_link=0.3):
        self.num_nodes = num_nodes
        self.area_width = area_width
        self.area_height = area_height
        self.area_depth = area_depth
        self.min_sep = min_sep
        self.max_velocity = max_velocity
        self.target = np.array(target)
        self.communication_range = communication_range
        #self.G = nx.Graph()
        self.p_link = p_link  # Probability for edge creation
        self.G = nx.empty_graph(self.num_nodes)  # Start with an empty graph with num_nodes nodes
        self.initialize_graph()
        self.add_er_edges()

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

        self.add_er_edges()

    def add_er_edges(self):
        # Apply Erdos-R�nyi model properties to form edges
        nodes = list(self.G.nodes())  # Work with a static list of current nodes
        max_connections = 5  # Maximum connections a node can have

        for i in nodes:
            for j in nodes:
                if i != j:  # Ensure not to connect the node to itself
                    if random.random() < self.p_link:
                        # Check both nodes do not exceed the maximum connections limit
                        if self.G.degree(i) < max_connections and self.G.degree(j) < max_connections:
                            self.G.add_edge(i, j)


    def calculate_reliability(self, distance):
        d_min = 0.05
        if distance <= d_min:
            return 1
        elif distance > d_min and distance <= self.communication_range:
            return (d_min / distance) ** 2
        return 0


    def should_add_edge(self, node1, node2):
        distance = np.linalg.norm(self.G.nodes[node1]['pos'] - self.G.nodes[node2]['pos'])
        return self.calculate_reliability(distance) > 0

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
                #print(f"Node {node} moved from {current_pos} to {new_pos}")
        else:
            print("No movement needed, target center of mass reached.")
        #self.add_ba_edges() # Uncomment for Dynamic Case


    def get_graph(self):
        return self.G