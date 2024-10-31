import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def visualize_networkx(edges, color_map, output_file='graph.png'):
    """
    Visualize an undirected network with consistent layout.
    
    Parameters:
    - edges: list of tuples (node1, node2) representing undirected edges.
    - color_map: dictionary mapping nodes to colors (e.g., {1: 'blue', 2: 'red'}).
    """
    # Create an undirected graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Assign positions with a fixed layout
    pos = nx.spring_layout(G, seed=42)

    # Generate node colors based on color_map
    node_colors = [color_map[node] for node in G.nodes]

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Remove axis
    plt.axis('off')
    # plt.show()
    plt.savefig(output_file, format=output_file.split('.')[-1], bbox_inches='tight')

def entailment_graph(chains, output_file='entailment_graph.png'):
    # node: chains: keys premises, conclusion, correct(1/0)
    # edge: shared sentences
    sent_to_chains = defaultdict(list)
    for i, chain in enumerate(chains):
        for sent in chain["premises"] + [chain["conclusion"]]:
            sent_to_chains[sent].append(i)
    edges = set()
    color_map = {}
    for i, chain in enumerate(chains):
        color_map[i] = "blue" if chain["correct"] else "gray"
        for sent in chain["premises"] + [chain["conclusion"]]:
            for other_chain in sent_to_chains[sent]:
                if i != other_chain:
                    edges.add((i, other_chain))
    
    visualize_networkx(edges, color_map, output_file=output_file)

# Example usage
if __name__ == '__main__':
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
    color_map = {1: 'blue', 2: 'red', 3: 'blue', 4: 'red'}
    visualize_networkx(edges, color_map)
