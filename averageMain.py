import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Differnt edge weights produce different averages, leave main unless finding average for another scenario
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
edges = {
    'A': [('B', 5), ('C', 6)],
    'B': [('C', 1), ('E', 2)],
    'C': [('D', 2)],
    'D': [('E', 1), ('F', 1)],
    'E': [('F', 1), ('G', 2)],
    'F': [('G', 1), ('H', 0.5)],
    'G': [('H', 1), ('I', 3)],
    'H': [('I', 2)],
    'I': [('J', 0.5)],
    'J': [('K', 0.5)],
    'K': [('L', 2)],
    'L': [('M', 1), ('O', 3)],
    'M': [('L', 1),('N', 0.5)],
    'N': [('O', 0.5), ('P', 2)],
    'O': [('P', 1), ('Q', 1)],
    'P': [('Q', 0.5)],
    'Q': [('R', 2)],
    'R': [('A', 2)],
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

bf_times = []
pso_times = []
labels = []

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
    output_dir = 'averageGeneratedGraphs'
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

def test_bellman_ford():
    print("Bellman-Ford Results:")
    total_time = 0
    path_count = 0
    for start in nodes:
        for end in nodes:
            if start != end:
                distances, previous_nodes = bellman_ford(start, edges)
                path = reconstruct_path(start, end, previous_nodes)
                if path:
                    time = calculate_time_seconds(path, edges)
                    print(f"{start} → {end} | Cost: {distances[end]:.2f} | Time: {time:.2f} sec | Path: {path}")
                    bf_times.append(time)
                    labels.append(f"{start}→{end}")
                    total_time += time
                    path_count += 1
                    create_graph_image(edges, path=path, start_node=start, end_node=end,
                                       file_name=f'bellmanFord{start}to{end}.png')
    if path_count > 0:
        avg_time = total_time / path_count
        print(f"\nTotal Bellman-Ford travel time: {total_time:.2f} seconds")
        print(f"Average Bellman-Ford travel time: {avg_time:.2f} seconds")

def test_pso():
    print("PSO Results:")
    total_time = 0
    path_count = 0
    for start in nodes:
        for end in nodes:
            if start != end:
                path, cost = run_pso(edges, start, end)
                if path:
                    time = calculate_time_seconds(path, edges)
                    print(f"{start} → {end} | Cost: {cost:.2f} | Time: {time:.2f} sec | Path: {path}")
                    pso_times.append(time)
                    total_time += time
                    path_count += 1
                    create_graph_image(edges, path=path, start_node=start, end_node=end,
                                       file_name=f'pso{start}to{end}.png')
    if path_count > 0:
        avg_time = total_time / path_count
        print(f"\nTotal PSO travel time: {total_time:.2f} seconds")
        print(f"Average PSO travel time: {avg_time:.2f} seconds")

def plot_time_comparison():
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, bf_times, width, label='Bellman-Ford',  color='white', edgecolor='black', hatch='///')
    ax.bar(x + width/2, pso_times, width, label='PSO',  color='lightgrey', edgecolor='black', hatch='')


    ax.set_ylabel('Time (seconds)')
    ax.set_title('Path Travel Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig('averageGeneratedGraphs/time_comparison.png')
    plt.close()

# Run everything
create_graph_image(edges, file_name='basicNodeMap.png')
test_bellman_ford()
test_pso()
plot_time_comparison()
