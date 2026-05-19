import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_signed_complete_graph(n, p_positive=0.5, seed=None):
    """
    The randomness causes a lot of bad triangles.
    Think of starting with just plus edges and then later on flipping a certain amount of edges randomly.
    Or give give each edge a certain probability of being flipped to -1.
    """
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    matrix = np.zeros((n , n), dtype=int)
    for i in range(n):
        for j in range(i+1,n): # the graph is symmetrical
            sign = 1 if rng.random() < p_positive else -1
            matrix[i, j] = sign
            matrix[j, i] = sign

    return matrix

## This is only used for the graph representation
def matrix_to_graph(matrix):
    graph = nx.Graph()
    n = matrix.shape[0]

    for i in range(n):
        graph.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            sign = matrix[i, j]

            if sign != 0:
                graph.add_edge(i, j, sign=sign)

    return graph



