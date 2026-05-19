from graph_generation import generate_signed_complete_graph, matrix_to_graph
from pivot import run_pivot, draw_clustered_graph
from cost import calculate_clustering_cost
from bad_triangles import find_bad_triangles, count_bad_triangles, bad_triangles_containing_edge
from ilp_solver import solve_correlation_clustering_ilp

if __name__ == "__main__":
    S = generate_signed_complete_graph(n=30, p_positive=0.5, seed=42)                # Change these parameters if you want
    G = matrix_to_graph(S)

    clusters, pivots = run_pivot(S, seed=42)                                        # Change these parameters if you want

    cost = calculate_clustering_cost(S, clusters)

    bad_triangles = find_bad_triangles(S)
    num_bad_triangles = count_bad_triangles(bad_triangles)
    bad_triangles_with_edge_01 = bad_triangles_containing_edge(bad_triangles, n1 =3, n2= 1) # Change these parameters if you want
    ilp_cost, x_values = solve_correlation_clustering_ilp(S, verbose=False)

    print("Clusters:", clusters)
    print("Pivots:", pivots)
    print("Pivot cost:", cost)
    draw_clustered_graph(G, clusters, pivots)

    print("ILP optimal cost:", ilp_cost)
    print("Amount of bad_triangles", num_bad_triangles)
    print("Bad triangles with edge 01:", bad_triangles_with_edge_01)

