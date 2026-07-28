import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_clique_signed_graph(
    cluster_sizes,
    p_pos_inside=0.9,
    p_pos_between=0.1,
    p_delete_inside=0.0,
    p_delete_between=0.0,
    seed=None
):
    """
    Generate a signed graph with planted clique/community structure.

    Nodes inside the same planted cluster are usually connected by positive edges.
    Nodes in different planted clusters are usually connected by negative edges.
    Edges can also be deleted, using different deletion probabilities for
    inside-cluster and between-cluster edges.

    Parameters
    ----------
    cluster_sizes : list[int]
        Sizes of the planted clusters, for example [8, 8, 8].
    p_pos_inside : float
        Probability that an edge inside a planted cluster is positive.
    p_pos_between : float
        Probability that an edge between different planted clusters is positive.
    p_delete_inside : float
        Probability of deleting an edge inside a planted cluster.
    p_delete_between : float
        Probability of deleting an edge between different planted clusters.
    seed : int or None
        Random seed.

    Returns
    -------
    matrix : np.ndarray
        Signed adjacency matrix with 1, -1, and 0 values.
    true_clusters : list[list[int]]
        The planted clusters used to generate the graph.
    """
    rng = np.random.default_rng(int(seed) if seed is not None else None)

    n = sum(cluster_sizes)
    matrix = np.zeros((n, n), dtype=int)

    true_clusters = []
    start = 0

    for size in cluster_sizes:
        cluster = list(range(start, start + size))
        true_clusters.append(cluster)
        start += size

    node_to_cluster = {}

    for cluster_index, cluster in enumerate(true_clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_index

    for i in range(n):
        for j in range(i + 1, n):
            same_cluster = node_to_cluster[i] == node_to_cluster[j]

            if same_cluster:
                p_delete = p_delete_inside
                p_positive = p_pos_inside
            else:
                p_delete = p_delete_between
                p_positive = p_pos_between

            if rng.random() < p_delete:
                sign = 0
            else:
                sign = 1 if rng.random() < p_positive else -1

            matrix[i, j] = sign
            matrix[j, i] = sign

    return matrix, true_clusters




