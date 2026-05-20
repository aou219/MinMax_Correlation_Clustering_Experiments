import os
import json
from graph_generation import generate_signed_complete_graph, matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from ilp_solver import find_ilp_clusters
from draw_graphs import draw_multiple_clustered_graphs
from bad_triangles import find_bad_triangles, count_bad_triangles, find_edge_disjoint_bad_triangles_min, make_edge_to_triangle_map, find_edge_disjoint_bad_triangles_max
from ilp_solver import solve_correlation_clustering_ilp
from lp_formulations import solve_correlation_clustering_primal, solve_correlation_clustering_dual

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
    n = 30
    p_positive = 0.5
    seed = 2

    # Generate signed complete graph
    S = generate_signed_complete_graph(n, p_positive, seed)
    G = matrix_to_graph(S)

    # Run Pivot algorithm
    pivot_clusters, pivots = run_pivot(S, seed)
    pivot_cluster_count = len(pivot_clusters)

    # Calculate Pivot cost
    pivot_cost = calculate_clustering_cost(S, pivot_clusters)

    # Find  bad triangles
    all_bad_triangles = find_bad_triangles(S)
    edge_to_triangles = make_edge_to_triangle_map(all_bad_triangles)
    min_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_min( edge_to_triangles)
    min_num_bad_triangles = count_bad_triangles(min_disjoint_bad_triangles)

    max_disjoint_bad_triangles =  find_edge_disjoint_bad_triangles_max( edge_to_triangles)
    max_num_bad_triangles = count_bad_triangles(max_disjoint_bad_triangles)

    # Solve LP for exact cost
    primal_cost, primal_x_values = solve_correlation_clustering_primal(S,all_bad_triangles, verbose=False)
    dual_cost, dual_x_values = solve_correlation_clustering_dual(S,all_bad_triangles, verbose=False)
    # lp_clusters = find_ilp_clusters(lp_x_values, n) IF YOU DO LP ROUNDING YOU MAY WANT TO DO THIS
    # lp_cluster_count = len(lp_clusters)

    # Solve ILP for exact cost
    ilp_cost, ilp_x_values = solve_correlation_clustering_ilp(S, verbose=False)
    ilp_clusters = find_ilp_clusters(ilp_x_values, n)
    ilp_cluster_count = len(ilp_clusters)


    # print("Amount of pivot clusters:", pivot_cluster_count)
    # print("Pivots:", pivots)
    print("Pivot cost:", pivot_cost)
    print("Minimum amount of disjoint bad triangles", min_num_bad_triangles)
    print("Maximum amount of disjoint bad triangles", max_num_bad_triangles)
    # print("ILP clusters:", ilp_clusters)
    # print("Amount of ILP clusters:", ilp_cluster_count)
    print("LP-primal optimal cost:", primal_cost)
    print("LP-dual optimal cost", dual_cost)
    print("ILP optimal cost:", ilp_cost)


    # Draw clustered graph
    # draw_multiple_clustered_graphs(
    # G,
    # clusterings=[pivot_clusters, ilp_clusters],
    # pivots_list=[pivots, set(), set()],
    # titles=["Pivot", "ILP", "LP-rounding"]
    # )

    # Store all results
    results = {
        "num_nodes": n,
        "p_positive": p_positive,
        "seed": seed,
        # "clusters": [list(c) for c in clusters],
        "Pivot cluster_count": pivot_cluster_count,
        "pivots": pivots,
        "pivot_cost": pivot_cost,
        "min_bad_triangles_count": min_num_bad_triangles,
        "max_bad_triangles_count": max_num_bad_triangles,
        "primal_cost":primal_cost,
        "dual_cost":dual_cost,
        "ilp_cost": ilp_cost,
        # "x_values": {f"{k}": v for k, v in x_values.items()},
        # "bad_triangles_with_edge": [list(bt) for bt in bad_triangles_with_edge_01],
        # "num_bad_triangles_edge_01": num_bad_triangles_with_edge_01
    }
    save_results_append(RESULTS_FILE, results)