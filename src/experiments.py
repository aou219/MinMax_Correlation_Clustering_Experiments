import os
import json
from graph_generation import generate_signed_complete_graph, matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from ilp_solver import find_ilp_clusters
from draw_graphs import draw_graphs
from bad_triangles import find_bad_triangles, count_bad_triangles, find_edge_disjoint_bad_triangles_min, make_edge_to_triangle_map, find_edge_disjoint_bad_triangles_max
from ilp_solver import solve_correlation_clustering_ilp
from lp_formulations import solve_correlation_clustering_primal, solve_correlation_clustering_dual
from edge_deletion import delete_edges

RESULTS_FILE = "results/experiments_results.json"

def save_results_append(filename, new_results):
    """Append new experiment results to a JSON file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(new_results)

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":

    # Parameters
    n = 10
    p_positive = 0.5
    p_delete = 0.2
    seed = 2
    # Generate signed complete graph
    S = generate_signed_complete_graph(n, p_positive, seed)
    G = matrix_to_graph(S)

    # Run Pivot algorithm on complete graph
    pivot_clusters, pivots = run_pivot(S, seed)
    pivot_cluster_count = len(pivot_clusters)

    # Calculate Pivot cost of complete graph
    pivot_cost = calculate_clustering_cost(S, pivot_clusters)

    # Find  bad triangles of complete graph
    all_bad_triangles = find_bad_triangles(S)
    edge_to_triangles = make_edge_to_triangle_map(all_bad_triangles)
    min_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_min( edge_to_triangles)
    min_num_bad_triangles = count_bad_triangles(min_disjoint_bad_triangles)
    max_disjoint_bad_triangles =  find_edge_disjoint_bad_triangles_max( edge_to_triangles)
    max_num_bad_triangles = count_bad_triangles(max_disjoint_bad_triangles)

    # Solve LP for exact cost of complete graph
    primal_cost, primal_x_values = solve_correlation_clustering_primal(S,all_bad_triangles, verbose=False)
    dual_cost, dual_x_values = solve_correlation_clustering_dual(S,all_bad_triangles, verbose=False)
    # lp_clusters = find_ilp_clusters(lp_x_values, n) IF YOU DO LP ROUNDING YOU MAY WANT TO DO THIS
    # lp_cluster_count = len(lp_clusters)

    # Solve ILP for exact cost of complete graph
    ilp_cost, ilp_x_values = solve_correlation_clustering_ilp(S, verbose=False)
    ilp_clusters = find_ilp_clusters(ilp_x_values, n)
    ilp_cluster_count = len(ilp_clusters)


    # Generate signed incomplete graph/ new graph
    S_new,num_edges_deleted = delete_edges(S, p_delete, seed)
    G_new  = matrix_to_graph(S_new)

    # Run pivot algorithm on new graph
    pivot_clusters_new, pivots_new = run_pivot(S_new, seed)
    pivot_cluster_count_new = len(pivot_clusters_new)

    # Calculate Pivot cost of new graph
    pivot_cost_new = calculate_clustering_cost(S_new, pivot_clusters_new)

    # Find  bad triangles of new graph
    all_bad_triangles_new = find_bad_triangles(S_new)
    edge_to_triangles_new = make_edge_to_triangle_map(all_bad_triangles_new)
    min_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_min( edge_to_triangles_new)
    min_num_bad_triangles_new = count_bad_triangles(min_disjoint_bad_triangles_new)
    max_disjoint_bad_triangles_new =  find_edge_disjoint_bad_triangles_max( edge_to_triangles_new)
    max_num_bad_triangles_new = count_bad_triangles(max_disjoint_bad_triangles_new)

    # Solve LP for exact cost of new graph
    primal_cost_new, primal_x_values_new = solve_correlation_clustering_primal(S_new,all_bad_triangles_new, verbose=False)
    dual_cost_new, dual_x_values_new = solve_correlation_clustering_dual(S_new,all_bad_triangles_new, verbose=False)
    # lp_clusters_new = find_ilp_clusters(lp_x_values_new, n) IF YOU DO LP ROUNDING YOU MAY WANT TO DO THIS
    # lp_cluster_count_new = len(lp_clusters_new)

    # Solve ILP for exact cost of new graph
    ilp_cost_new, ilp_x_values_new = solve_correlation_clustering_ilp(S_new, verbose=False)
    ilp_clusters_new = find_ilp_clusters(ilp_x_values_new, n)
    ilp_cluster_count_new = len(ilp_clusters_new)



    # Print statements of complete graph
    print("##################### COMPLETE GRAPH #########################")
    # print("Amount of pivot clusters of complete graph:", pivot_cluster_count)
    # print("Pivots of complete graph:", pivots)
    print("Pivot cost of complete graph:", pivot_cost)
    print("Minimum amount of disjoint bad triangles of complete graph:", min_num_bad_triangles)
    print("Maximum amount of disjoint bad triangles of complete graph:", max_num_bad_triangles)
    # print("ILP clusters of complete graph:", ilp_clusters)
    # print("Amount of ILP clusters of complete graph:", ilp_cluster_count)
    print("LP-primal optimal cost of complete graph:", primal_cost)
    print("LP-dual optimal cost of complete graph:", dual_cost)
    print("ILP optimal cost of complete graph:", ilp_cost)

    # Print statements of new graph
    print("##################### NEW GRAPH #########################")
    # print("Amount of pivot clusters of new graph", pivot_cluster_count_new)
    # print("Pivots of new graph:", pivots_new)
    print("Number of edges deleted:", num_edges_deleted)
    print("Pivot cost of new graph:", pivot_cost_new)
    print("Minimum amount of disjoint bad triangles of new graph:", min_num_bad_triangles_new)
    print("Maximum amount of disjoint bad triangles of new graph:", max_num_bad_triangles_new)
    # print("ILP clusters of new graph:", ilp_clusters_new)
    # print("Amount of ILP clusters of new graph:", ilp_cluster_count_new)
    print("LP-primal optimal cost of new graph:", primal_cost_new)
    print("LP-dual optimal cost of new graph:", dual_cost_new)
    print("ILP optimal cost of new graph:", ilp_cost_new)

    # Draw clustered graph
    draw_graphs(
        G_complete=G,
        pivot_clusters=pivot_clusters,
        ilp_clusters=ilp_clusters,
        G_new=G_new,
        pivot_clusters_new=pivot_clusters_new,
        ilp_clusters_new=ilp_clusters_new,
        pivots=pivots,
        pivots_new=pivots_new
    )
    # Store all results
    results = {
    "graph_params": {
        "num_nodes": n,        # shared
        "p_positive": p_positive,
        "seed": seed,
        "p_delete":p_delete
    },
    "complete_graph": {
        "Pivot": {
            "cluster_count": pivot_cluster_count,
            "pivots": pivots,
            "cost": pivot_cost
        },
        "min_bad_triangles_count": min_num_bad_triangles,
        "max_bad_triangles_count": max_num_bad_triangles,
        "LP_primal_cost": primal_cost,
        "LP_dual_cost": dual_cost,
        "ILP_cost": ilp_cost
    },
    "new_graph": {
        "Pivot": {
            "cluster_count": pivot_cluster_count_new,
            "pivots": pivots_new,
            "cost": pivot_cost_new
        },
        "min_bad_triangles_count": min_num_bad_triangles_new,
        "max_bad_triangles_count": max_num_bad_triangles_new,
        "LP_primal_cost": primal_cost_new,
        "LP_dual_cost": dual_cost_new,
        "ILP_cost": ilp_cost_new
    }
}
    save_results_append(RESULTS_FILE, results)