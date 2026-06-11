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


def create_clique_layout(true_clusters, radius=1.2, spacing=4.0):
    """
    Create fixed positions where every clique/community is drawn
    as a separate circle.

    This makes the clique/community structure visible.
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
    show_pivot_order=False,
    show_negative_edges=True
):
    """
    Draw one signed graph using the clique layout.

    Node colors show the clustering found by Pivot or ILP.
    Positive edges are solid black.
    Negative edges are dashed gray, but can be hidden.
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

    # Draw real positive connections
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

    # Draw negative edges only when wanted
    if show_negative_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=negative_edges,
            ax=ax,
            edge_color="gray",
            width=0.6,
            style="dashed",
            alpha=0.25
        )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        cmap=plt.cm.Set3,
        node_size=450,
        edgecolors="black",
        linewidths=0.8
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        ax=ax,
        font_size=7,
        font_color="black"
    )

    if show_pivot_order and pivots is not None:
        for order, pivot in enumerate(pivots):
            if pivot not in positions:
                continue

            x, y = positions[pivot]
            ax.text(
                x,
                y + 0.28,
                chr(65 + order),
                fontsize=10,
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
    Draw complete and edge-deleted clique/Facebook-circle graphs.

    For the complete graph, negative edges are hidden so that only
    real positive Facebook connections are visible.

    For the edge-deleted graph, negative edges are shown because the graph
    is no longer complete and the missing/deleted structure matters more.
    """

    positions = create_clique_layout(true_clusters)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    draw_single_clique_graph(
        ax=axes[0, 0],
        graph=G_complete,
        positions=positions,
        clusters=pivot_clusters,
        title="Pivot on complete graph",
        pivots=pivots,
        show_pivot_order=True,
        show_negative_edges=False
    )

    draw_single_clique_graph(
        ax=axes[0, 1],
        graph=G_complete,
        positions=positions,
        clusters=ilp_clusters,
        title="ILP on complete graph",
        show_negative_edges=False
    )

    draw_single_clique_graph(
        ax=axes[1, 0],
        graph=G_new,
        positions=positions,
        clusters=pivot_clusters_new,
        title="Pivot on edge-deleted graph",
        pivots=pivots_new,
        show_pivot_order=True,
        show_negative_edges=True
    )

    draw_single_clique_graph(
        ax=axes[1, 1],
        graph=G_new,
        positions=positions,
        clusters=ilp_clusters_new,
        title="ILP on edge-deleted graph",
        show_negative_edges=True
    )

    plt.tight_layout()
    plt.show()
