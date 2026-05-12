from graph_generation import generate_signed_complete_graph, matrix_to_graph
from pivot import run_pivot, draw_clustered_graph
from cost import calculate_clustering_cost

if __name__ == "__main__":
    S = generate_signed_complete_graph(n=8, p_positive=0.5, seed=42)
    G = matrix_to_graph(S)

    clusters, pivots = run_pivot(S, seed=42)

    cost = calculate_clustering_cost(S, clusters)

    print("Clusters:", clusters)
    print("Pivots:", pivots)
    print("Cost:", cost)

    draw_clustered_graph(G, clusters, pivots)