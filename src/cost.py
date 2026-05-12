import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def in_same_cluster(i, j, clusters):
    for cluster in clusters:
        if i in cluster and j in cluster:
            return True
    return False

def calculate_clustering_cost(S, clusters):
    cost = 0
    n = S.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            sign = S[i, j]

            if sign == 0:
                continue

            same_cluster = in_same_cluster(i, j, clusters)

            if sign == -1 and same_cluster:
                cost += 1

            elif sign == 1 and not same_cluster:
                cost += 1
    return cost