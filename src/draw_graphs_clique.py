import math
import matplotlib.pyplot as plt
import networkx as nx


def get_node_to_cluster(clusters):
    """
    Convert a list of clusters into a dictionary:
        node -> cluster index
    """
    node_to_cluster = {}

    for cluster_index, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_index

    return node_to_cluster


def get_cluster_colors(clusters):
    """
    Assign a color index to each node based on the clustering.
    """
    node_to_cluster = get_node_to_cluster(clusters)

    colors = []
    for node in sorted(node_to_cluster.keys()):
        colors.append(node_to_cluster[node])

    return node_to_cluster


def create_clique_layout(true_clusters, radius=1.2, spacing=4.0):
    """
    Create fixed positions where every  clique/community is drawn
    as a separate circle.

    This makes the  clique structure visible.
    """
    positions = {}

    for cluster_index, cluster in enumerate(true_clusters):
        center_x = cluster_index * spacing
        center_y = 0

        cluster_size = len(cluster)

        for node_index, node in enumerate(cluster):
            angle = 2 * math.pi * node_index / cluster_size

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            positions[node] = (x, y)

    return positions


def draw_single_clique_graph(
    ax,
    graph,
    positions,
    clusters,
    title,
    pivots=None,
    show_pivot_order=False
):
    """
    Draw one signed graph using the  clique layout.

    Node colors show the clustering found by Pivot or ILP.
    Positive edges are solid black.
    Negative edges are dashed gray.
    """

    node_to_cluster = get_node_to_cluster(clusters)

    node_colors = [
        node_to_cluster.get(node, -1)
        for node in graph.nodes()
    ]

    positive_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("sign") == 1
    ]

    negative_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("sign") == -1
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive_edges,
        ax=ax,
        edge_color="black",
        width=1.2,
        style="solid",
        alpha=0.8
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative_edges,
        ax=ax,
        edge_color="gray",
        width=1.0,
        style="dashed",
        alpha=0.6
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        cmap=plt.cm.Set3,
        node_size=700,
        edgecolors="black",
        linewidths=1.0
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        ax=ax,
        font_size=9,
        font_color="black"
    )

    if show_pivot_order and pivots is not None:
        for order, pivot in enumerate(pivots):
            if pivot not in positions:
                continue

            x, y = positions[pivot]
            ax.text(
                x,
                y + 0.35,
                chr(65 + order),
                fontsize=11,
                color="red",
                fontweight="bold",
                ha="center"
            )

    ax.set_title(title)
    ax.axis("off")


def draw_clique_graphs(
    G_complete,
    true_clusters,
    pivot_clusters,
    ilp_clusters,
    G_new,
    pivot_clusters_new,
    ilp_clusters_new,
    pivots=None,
    pivots_new=None
):
    """
    Draw complete and edge-deleted  clique graphs.

    The layout is based on the  true clusters, not on a spring layout.
    This makes the clique/community structure much easier to see.
    """

    positions = create_clique_layout(true_clusters)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    draw_single_clique_graph(
        ax=axes[0, 0],
        graph=G_complete,
        positions=positions,
        clusters=pivot_clusters,
        title="Pivot on clique graph",
        pivots=pivots,
        show_pivot_order=True
    )

    draw_single_clique_graph(
        ax=axes[0, 1],
        graph=G_complete,
        positions=positions,
        clusters=ilp_clusters,
        title="ILP on clique graph"
    )

    draw_single_clique_graph(
        ax=axes[1, 0],
        graph=G_new,
        positions=positions,
        clusters=pivot_clusters_new,
        title="Pivot on new clique graph",
        pivots=pivots_new,
        show_pivot_order=True
    )

    draw_single_clique_graph(
        ax=axes[1, 1],
        graph=G_new,
        positions=positions,
        clusters=ilp_clusters_new,
        title="ILP on new clique graph"
    )

    plt.tight_layout()
    plt.show()
