import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import psutil  # To track memory consumption
import statistics  # To track consistency for computational stability

# Graph data when there is Heavy Foot Traffic
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
edges = {
    'A': [('B', 5), ('C', 6)],
    'B': [('C', 1), ('E', 2)],
    'C': [('D', 2)],
    'D': [('E', 1), ('F', 1)],
    'E': [('F', 5), ('G', 2), ('B', 2)],  #People tend to cross here
    'F': [('G', 5), ('H', 0.5)], #Major cross-walk here
    'G': [('H', 5), ('I', 5)],   #Many people j-walk here
    'H': [('I', 2)],
    'I': [('J', 0.5)],
    'J': [('K', 0.5)],
    'K': [('L', 2)],
    'L': [('M', 5), ('O', 3)],   #Major cross-walk here
    'M': [('L', 5),('N', 0.5)],   #Major cross-walk here
    'N': [('O', 0.5), ('P', 2)],
    'O': [('P', 1), ('Q', 1)],
    'P': [('Q', 0.5)],
    'Q': [('R', 4)],    #Students come from here frequently
    'R': [('A', 7)],    #Major cross-walk here
}

# Legend mapping for node labels
node_labels = {
    'A': 'Joan C. Edwards Stadium',
    'B': '3rd Ave Garage',
    'C': 'Henderson Center',
    'D': 'Harris Hall',
    'E': 'Biotechnology Science Center',
    'F': 'Science Building',
    'G': 'Engineering Building',
    'H': 'Morrow Library',
    'I': 'Smith Hall',
    'J': 'Old Main',
    'K': 'Corbly Hall',
    'L': 'Memorial Student Center',
    'M': 'Performing Arts',
    'N': 'Harless Dining Hall',
    'O': 'Holderby Hall',
    'P': 'Marshall Police Department',
    'Q': 'Twin Towers',
    'R': 'Recreational Center'
}


# Other constants
EDGE_LENGTH_FEET = 543.8
SPEED_FPS = 44


def calculate_time_seconds(path, graph):
    total_cost = 0
    for i in range(len(path) - 1):
        for neighbor, weight in graph[path[i]]:
            if neighbor == path[i + 1]:
                total_cost += weight
                break
    return (total_cost * EDGE_LENGTH_FEET) / SPEED_FPS


# Metrics tracking lists
bf_times = []
pso_times = []
bf_durations = []  # Runtime duration for Bellman-Ford
pso_durations = []  # Runtime duration for PSO
bf_memories = []  # Memory consumption for Bellman-Ford
pso_memories = []  # Memory consumption for PSO
bf_costs = []  # Solution quality for Bellman-Ford (average travel time)
pso_costs = []  # Solution quality for PSO (average travel time)


# Bellman-Ford Algorithm
def bellman_ford(start, graph):
    distances = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    previous_nodes[neighbor] = node

    return distances, previous_nodes


def reconstruct_path(start, end, previous_nodes):
    path = []
    current_node = end
    while current_node != start:
        path.append(current_node)
        current_node = previous_nodes.get(current_node)
        if current_node is None:
            return []
    path.append(start)
    return path[::-1]

def create_graph_image(graph, path=None, start_node=None, end_node=None, file_name='graph.png'):
    output_dir = 'generatedGraphs2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, file_name)
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)

    node_color = []
    for node in G.nodes():
        if node == start_node:
            node_color.append('blue')
        elif node == end_node:
            node_color.append('purple')
        elif path and node in path:
            node_color.append('red')
        else:
            node_color.append('black')

    edge_color = []
    for u, v in G.edges():
        if path and u in path and v in path and abs(path.index(u) - path.index(v)) == 1:
            edge_color.append('magenta')
        else:
            edge_color.append('black')

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=1000)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='white')

    # Add the legend
    legend_text = '\n'.join([f"{key}: {value}" for key, value in node_labels.items()])
    plt.gcf().text(1.05, 0.5, legend_text, fontsize=10, va='center')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

# Particle Swarm Optimization
class Particle:
    def __init__(self, graph, start, end):
        self.graph = graph
        self.start = start
        self.end = end
        self.position = self.random_path()
        self.velocity = [0] * len(self.position)
        self.best_position = self.position
        self.best_cost = self.calculate_cost(self.position)

    def random_path(self):
        path = [self.start]
        current_node = self.start
        while current_node != self.end:
            neighbors = [neighbor for neighbor, _ in self.graph[current_node]]
            current_node = random.choice(neighbors)
            path.append(current_node)
        return path

    def calculate_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            for neighbor, weight in self.graph[path[i]]:
                if neighbor == path[i + 1]:
                    cost += weight
                    break
        return cost

    def update_position(self):
        for i in range(len(self.position)):
            if random.random() < self.velocity[i]:
                neighbors = [neighbor for neighbor, _ in self.graph[self.position[i]]]
                self.position[i] = random.choice(neighbors)


def run_pso(graph, start, end, num_particles=10, max_iterations=100):
    particles = [Particle(graph, start, end) for _ in range(num_particles)]
    global_best_position = None
    global_best_cost = float('inf')

    for _ in range(max_iterations):
        for particle in particles:
            particle_cost = particle.calculate_cost(particle.position)
            if particle_cost < particle.best_cost:
                particle.best_cost = particle_cost
                particle.best_position = particle.position

            if particle.best_cost < global_best_cost:
                global_best_cost = particle.best_cost
                global_best_position = particle.best_position

    return global_best_position, global_best_cost


# Code for running tests and tracking metrics
def test_bellman_ford():
    start_time = time.time()
    print("Bellman-Ford Results:")
    for start, end in [('F', 'R'), ('B', 'F'), ('I', 'P')]:
        distances, previous_nodes = bellman_ford(start, edges)
        path = reconstruct_path(start, end, previous_nodes)
        print(f"Shortest path from {start} to {end}: {distances[end]}")
        print(f"Path: {path}")
        if path:
            time_taken = time.time() - start_time  # Runtime efficiency
            travel_time = calculate_time_seconds(path, edges)
            bf_times.append(travel_time)
            bf_durations.append(time_taken)
            bf_memories.append(psutil.virtual_memory().percent)  # Memory usage
            bf_costs.append(travel_time)
            create_graph_image(edges, path=path, start_node=start, end_node=end,
                               file_name=f'bellmanFord{start}to{end}.png')


def test_pso():
    start_time = time.time()
    print("PSO Results:")
    for start, end in [('F', 'R'), ('B', 'F'), ('I', 'P')]:
        path, cost = run_pso(edges, start, end)
        print(f"Shortest path from {start} to {end}: {cost}")
        print(f"Path: {path}")
        if path:
            time_taken = time.time() - start_time  # Runtime efficiency
            travel_time = calculate_time_seconds(path, edges)
            pso_times.append(travel_time)
            pso_durations.append(time_taken)
            pso_memories.append(psutil.virtual_memory().percent)  # Memory usage
            pso_costs.append(travel_time)
            create_graph_image(edges, path=path, start_node=start, end_node=end, file_name=f'pso{start}to{end}.png')


# Plotting Metrics (you can extend these functions to plot additional metrics)
def plot_metrics():
    x = np.arange(len(bf_times))

    # Plotting runtime efficiency
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - 0.2, bf_durations, 0.4, label='Bellman-Ford Runtime', color='white', edgecolor='black', hatch='///')
    ax1.bar(x + 0.2, pso_durations, 0.4, label='PSO Runtime', color='lightgrey', edgecolor='black', hatch='')
    ax1.set_title('Runtime Efficiency')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['F→R', 'B→F', 'I→P'])
    ax1.legend()
    plt.savefig('generatedGraphs2/runtime_efficiency.png')

    # Plotting memory consumption
    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - 0.2, bf_memories, 0.4, label='Bellman-Ford Memory', color='white', edgecolor='black', hatch='///')
    ax2.bar(x + 0.2, pso_memories, 0.4, label='PSO Memory', color='lightgrey', edgecolor='black', hatch='')
    ax2.set_title('Memory Consumption')
    ax2.set_ylabel('Memory Usage (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['F→R', 'B→F', 'I→P'])
    ax2.legend()
    plt.savefig('generatedGraphs2/memory_consumption.png')

    # Plotting
# Run everything
create_graph_image(edges, file_name='pedestrainNodeMap.png')
test_bellman_ford()
test_pso()
plot_metrics()