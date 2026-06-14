import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

population_size = 100
iterations = 200
W = 0.5  # Inertia weight
C1 = 0.8  # Cognitive (particle) weight
C2 = 0.9  # Social (swarm) weight

class SwarmGraph:
    def __init__(self, num_nodes=101, area_width=10, area_height=10, area_depth=10, min_sep=0.02, max_velocity=0.35, target=(9, 9, 4), init_area_size=1, communication_range=1):
        self.num_nodes = num_nodes
        self.area_width = area_width
        self.area_height = area_height
        self.area_depth = area_depth
        self.min_sep = min_sep
        self.max_velocity = max_velocity
        self.target = np.array(target)
        self.init_area_size = init_area_size
        self.W = 0.5  # Inertia weight
        self.C1 = 0.8  # Cognitive weight (personal best influence)
        self.C2 = 0.8  # Social weight (global best influence)
        self.C3 = 0.2
        self.G = nx.Graph()
        self.gbest = None
        self.initialize_graph()
        self.communication_range = communication_range

    def initialize_graph(self):
        # Calculate grid size along each dimension
        grid_size = int(round(self.num_nodes ** (1/3)))
        # Calculate spacing based on the minimum dimension to fit all nodes
        x_spacing = 0.05
        y_spacing = 0.05
        z_spacing = 0.05

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

                        # Collect neighbors based on grid adjacency
                        neighbors = []
                        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                            neighbor_index = ((x + dx) * grid_size * grid_size) + ((y + dy) * grid_size) + (z + dz)
                            if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size and 0 <= z + dz < grid_size:
                                if neighbor_index in positions:
                                    neighbors.append((neighbor_index, positions[neighbor_index] - pos))
                        
                        self.G.nodes[index]['neighbors'] = neighbors
                        index += 1

    def calculate_reliability(self, distance):
        d_min = 0.05
        if distance <= d_min:
            return 1
        elif distance > d_min and distance <= self.communication_range:
            # Using inverse square law for reliability
            return (d_min ** 2) / (distance ** 2)
        return 0
    
    def update_edges(self):
        self.G.remove_edges_from(list(self.G.edges()))  # Clear all existing edges before recalculating
        nodes = list(self.G.nodes())
        proximity_threshold = self.communication_range
        potential_edges = []

        # Iterate over pairs of nodes to determine which should be connected
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(self.G.nodes[nodes[i]]['pos'] - self.G.nodes[nodes[j]]['pos'])
                if dist <= proximity_threshold:
                    reliability = self.calculate_reliability(dist)
                    self.G.add_edge(nodes[i], nodes[j], reliability=reliability)

        # Create a dictionary to count connections
        connection_count = {node: 0 for node in nodes}
        
        # Sort potential edges based on reliability, higher first
        potential_edges.sort(key=lambda x: x[2]['reliability'], reverse=True)

        
        # Add edges respecting the maximum connection rule
        for node1, node2, data in potential_edges:
            if connection_count[node1] < 10 and connection_count[node2] < 10:
                self.G.add_edge(node1, node2, **data)
                connection_count[node1] += 1
                connection_count[node2] += 1

        # Remove any excess edges if nodes exceed connection limits
        for node in nodes:
            while self.G.degree[node] > 10:
                # Find the edge with the lowest reliability to remove
                least_reliable_edge = min(self.G.edges(node, data=True), key=lambda x: x[2]['reliability'])
                self.G.remove_edge(*least_reliable_edge[:2])
                    
    def update_gbest(self):
        # Ensuring gbest is recalculated only within the largest connected component
        #if nx.is_empty(self.G):
        #    self.gbest = None
        #else:
        largest_component = max(nx.connected_components(self.G), key=len)
        self.gbest = min(largest_component, key=lambda node: np.linalg.norm(self.G.nodes[node]['pbest'] - self.target))

    def update_positions(self):
        if self.gbest is not None and self.gbest in self.G:
            gbest_pos = self.G.nodes[self.gbest]['pos']
        else:
            self.update_gbest()
            if self.gbest is not None:
                gbest_pos = self.G.nodes[self.gbest]['pos']
            else:
                return  # Exit if there's no valid gbest

        new_positions = {}
        for i in self.G.nodes:
            node_info = self.G.nodes[i]
            pos = node_info['pos']
            vel = node_info['velocity']
            pbest = node_info['pbest']

            # Calculate the distance to the target to modulate the influence
            distance_to_target = np.linalg.norm(pos - self.target)
            target_influence = max(0, 1 - (distance_to_target / 10))  # Reduce target influence with distance

            # Standard PSO update components
            r1, r2, r3 = random.random(), random.random(), random.random()
            social_component = self.C2 * r2 * (gbest_pos - pos)
            cognitive_component = self.C1 * r1 * (pbest - pos)
            vel_update = self.W * vel + cognitive_component + social_component

            # Modulate the influence of the target based on the calculated distance
            target_component = self.C3 * r3 * (self.target - pos) * target_influence
            vel_update += target_component

            # Apply velocity update considering the formation control
            formation_component = self.calculate_formation_control(node_info, i)
            vel_update += formation_component

            # Apply position update and respect boundaries
            proposed_position = np.clip(pos + vel_update, [0, 0, 0], [self.area_width, self.area_height, self.area_depth])

            # Collision checking and update positions
            valid_position = True
            for j in new_positions:
                if np.linalg.norm(proposed_position - new_positions[j]) < self.min_sep:
                    valid_position = False
                    break
            if valid_position:
                new_positions[i] = proposed_position
            else:
                new_positions[i] = pos  # If collision, revert to old position

        # Apply updated positions
        for i in new_positions:
            self.G.nodes[i]['pos'] = new_positions[i]
            if np.linalg.norm(new_positions[i] - self.target) < np.linalg.norm(node_info['pbest'] - self.target):
                self.G.nodes[i]['pbest'] = new_positions[i]

        self.update_edges()

    def calculate_formation_control(self, node_info, node_id):
        # Implement formation control based on the swarm's desired configuration
        formation_component = np.zeros(3)
        if 'neighbors' in node_info:
            for (neighbor_index, relative_pos) in node_info['neighbors']:
                if neighbor_index in self.G:
                    current_relative_pos = self.G.nodes[neighbor_index]['pos'] - node_info['pos']
                    formation_component += (relative_pos - current_relative_pos) * 0.1  # Adjust strength as needed
        return formation_component

    def remove_node(self, node):
        """Remove a node from the graph and update related structures."""
        if node in self.G:
            self.G.remove_node(node)

    def get_graph(self):
        return self.G