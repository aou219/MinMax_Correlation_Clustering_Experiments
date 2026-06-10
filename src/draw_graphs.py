import matplotlib.pyplot as plt
import networkx as nx
import string

def draw_graphs(G_complete, pivot_clusters, ilp_clusters,
                             G_new, pivot_clusters_new, ilp_clusters_new,
                             pivots, pivots_new, figsize=(15,10)):
    """
    Draw 4 graphs in two rows with fixed node positions:
    Row 1: Pivot + ILP (complete graph)
    Row 2: Pivot + ILP (new graph, edges deleted)
    """
    graphs = [G_complete, G_complete, G_new, G_new]
    clusterings = [pivot_clusters, ilp_clusters, pivot_clusters_new, ilp_clusters_new]
    pivots_list = [pivots, set(), pivots_new, set()]
    titles = ["Pivot", "ILP", "Pivot (New)", "ILP (New)"]

    # Fix positions using the complete graph
    pos = nx.spring_layout(G_complete, seed=42)
    letters = string.ascii_uppercase

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, graph, clusters, pivot_set, title in zip(axes, graphs, clusterings, pivots_list, titles):
        ax.set_title(title)
        ax.axis("off")

        # Node colors based on clusters
        node_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = idx
        node_colors = [node_to_cluster[node] for node in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.Set3,
                               node_size=700, edgecolors="black", ax=ax)

        # Node labels
        node_labels = {node: str(node) for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, ax=ax)

        # Edges: only draw the ones actually present in the graph
        positive_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("sign", 1) == 1]
        negative_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("sign", 1) == -1]
        nx.draw_networkx_edges(graph, pos, edgelist=positive_edges, width=2, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=negative_edges, style="dashed", alpha=0.6, ax=ax)

        # Pivot letters
        if pivot_set:
            pivot_labels = {pivot: letters[i] for i, pivot in enumerate(pivot_set)}
            pivot_label_pos = {pivot: (pos[pivot][0], pos[pivot][1] + 0.15) for pivot in pivot_set}
            nx.draw_networkx_labels(graph, pivot_label_pos, labels=pivot_labels,
                                    font_weight="bold", font_color="red", font_size=12, ax=ax)

    plt.tight_layout()
    plt.show()