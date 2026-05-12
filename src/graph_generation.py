import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_signed_complete_graph(n, seed = None, p_positive=0.5):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n , n), dtype=int)
    for i in range(n):
        for j in range(i+1,n):
            if rng.random() < p_positive:
                sign = 1
            else:
                sign = -1

            matrix[i, j] = sign
            matrix[j, i] = sign

    return matrix

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



