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

def draw_clustered_graph(graph, clusters, pivots, seed=42):
    pos = nx.spring_layout(graph, seed=seed)

    positive_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data["sign"] == 1
    ]
    negative_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data["sign"] == -1
    ]

    node_to_cluster = {}
    for cluster_index, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_index

    node_colors = [node_to_cluster[node] for node in graph.nodes()]

    pivot_labels = {
        pivot: chr(ord("A") + pivot_index)
        for pivot_index, pivot in enumerate(pivots)
    }

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        cmap=plt.cm.Set3,
        node_size=700,
        edgecolors="black",
    )

    node_labels = {node: str(node) for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        font_size=10,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=positive_edges,
        width=2,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=negative_edges,
        style="dashed",
        alpha=0.6,
    )

    pivot_label_pos = {
        pivot: (pos[pivot][0], pos[pivot][1] + 0.2)
        for pivot in pivots
    }

    nx.draw_networkx_labels(
        graph,
        pivot_label_pos,
        labels=pivot_labels,
        font_weight="bold",
        font_size=12,
        font_color="red",
    )

    plt.axis("off")
    plt.show()