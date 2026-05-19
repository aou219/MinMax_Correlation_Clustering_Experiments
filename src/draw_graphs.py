import matplotlib.pyplot as plt
import networkx as nx
import string

def draw_multiple_clustered_graphs(graph, clusterings, pivots_list=None, titles=None, figsize=(15,5)):
    """
    Draw multiple clusterings of the same graph side by side.

    Parameters:
    - graph: networkx graph object
    - clusterings: list of clusterings (each clustering is a list of sets of nodes)
    - pivots_list: list of pivot sets corresponding to each clustering (only first clustering uses pivot labeling)
    - titles: list of titles for each subplot
    - figsize: figure size
    """
    num_graphs = len(clusterings)
    if titles is None:
        titles = [f"Clustering {i+1}" for i in range(num_graphs)]
    if pivots_list is None:
        pivots_list = [set() for _ in range(num_graphs)]

    pos = nx.spring_layout(graph, seed=42)  # consistent layout
    colors = plt.cm.tab20.colors  # color palette

    fig, axes = plt.subplots(1, num_graphs, figsize=figsize)
    if num_graphs == 1:
        axes = [axes]

    letters = string.ascii_uppercase

    for ax, clusters, pivots, title in zip(axes, clusterings, pivots_list, titles):
        ax.set_title(title)
        ax.axis('off')

        # Assign nodes to clusters for coloring
        node_to_cluster = {}
        for cluster_index, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = cluster_index

        node_colors = [node_to_cluster[node] for node in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                               cmap=plt.cm.Set3, node_size=700, edgecolors="black", ax=ax)

        # Node numbers always
        node_labels = {node: str(node) for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, ax=ax)

        # Edges
        positive_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("sign", 1) == 1]
        negative_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("sign", 1) == -1]

        nx.draw_networkx_edges(graph, pos, edgelist=positive_edges, width=2, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=negative_edges, style="dashed", alpha=0.6, ax=ax)

        # Pivot letters only for the first clustering (Pivot)
        if pivots:
            pivot_labels = {pivot: letters[i] for i, pivot in enumerate(pivots)}
            pivot_label_pos = {pivot: (pos[pivot][0], pos[pivot][1] + 0.15) for pivot in pivots}
            nx.draw_networkx_labels(graph, pivot_label_pos, labels=pivot_labels,
                                    font_weight="bold", font_color="red", font_size=12, ax=ax)

    plt.tight_layout()
    plt.show()