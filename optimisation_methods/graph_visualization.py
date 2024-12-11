import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy._typing import NDArray


def display_relation_graph(matrix: NDArray, solution: list):
    G = nx.DiGraph()

    # Add nodes to the graph (1-indexed)
    num_elements = matrix.shape[0]
    G.add_nodes_from(range(1, num_elements + 1))

    # Add directed edges based on the matrix
    for i in range(num_elements):
        for j in range(num_elements):
            if matrix[i, j] == 1:  # If there's a relation (i -> j)
                G.add_edge(i + 1, j + 1)

    # Assign colors to nodes and edges
    node_colors = ['white'] * num_elements
    edge_colors = []

    color_map = plt.cm.get_cmap('tab10', len(solution))  # Unique colors for solution nodes
    # print(color_map.__dict__)

    for node in range(1, num_elements + 1):

        if node not in solution:
            # Color outgoing edges from solution nodes
            for successor in G.successors(node):
                    edge_colors.append('gray')
        elif node in solution:
            idx = solution.index(node)
             # Color outgoing edges from solution nodes
            for successor in G.successors(node):
                edge_colors.append(color_map(idx))
                node_colors[successor - 1] = 'gray'  # Non-solution nodes with incoming edges are gray
            # Set the solution node color
            node_colors[node - 1] = color_map(idx)

    # If there are no outgoing edges for non-solution nodes, leave them gray
    pos = nx.circular_layout(G)

    # Draw nodes and edges with colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100) # , cmap=color_map
    nx.draw_networkx_labels(G, pos, font_size=6)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, connectionstyle="arc3,rad=0.", arrows=True)

    # Show the graph
    plt.show()


if __name__ == "__main__":
    # Example usage
    binary_relation = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ])  # Replace with your binary relation matrix

    solution = [1, 3]  # Example solution nodes, replace with your Neumann-Morgenstern solution

    display_relation_graph(binary_relation, solution)
