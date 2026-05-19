import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def run_pivot(S, seed=None):
    rng = np.random.default_rng(seed)

    n = S.shape[0]
    active_nodes = set(range(n))
    clusters = []
    pivots = []

    while active_nodes:
        pivot = int(rng.choice(list(active_nodes)))
        cluster = {pivot}
        for v in list(active_nodes):
            if v == pivot:
                continue
            if S[pivot, v] == 1: #don't need to check if there is an edge between them as it is a complete graph
                cluster.add(v)

        clusters.append(cluster)
        pivots.append(pivot)

        active_nodes -= cluster

    return clusters, pivots
