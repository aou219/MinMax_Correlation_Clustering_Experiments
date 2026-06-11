import os
import json
import numpy as np
import time
from graph_generation import generate_signed_complete_graph, matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from draw_graphs import draw_graphs
from bad_triangles import (
    find_bad_triangles,
    count_bad_triangles,
    find_edge_disjoint_bad_triangles_min,
    make_edge_to_triangle_map,
    find_edge_disjoint_bad_triangles_max,
)
from lp_formulations import solve_primal, solve_dual
from edge_deletion import delete_edges
from ilp_solver import solve_ilp, find_ilp_clusters


RESULTS_FILE = "results/experiments_results.json"


def json_converter(obj):
    """Convert NumPy objects to normal Python types for JSON saving."""
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, tuple):
        return list(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results_append(filename, new_results):
    """Append new experiment results to a JSON file."""
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(new_results)

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4, default=json_converter)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    print("\n--- " + title + " ---")


def get_edge_value(x_values, edge):
    """
    Get the value of an edge variable.

    This is useful because edges may be stored as (i, j) or (j, i).
    """
    u, v = edge

    if (u, v) in x_values:
        return x_values[(u, v)]

    if (v, u) in x_values:
        return x_values[(v, u)]

    return 0


def check_violated_bad_cycles(x_values, bad_cycles, tolerance=1e-6):
    """
    Check how many bad 4-cycle constraints are violated.

    The constraint is:
        x_negative <= sum of the other three cycle edges
    """
    violated_cycles = []

    for cycle, cycle_edges, signs, diagonal_1, diagonal_2 in bad_cycles:
        negative_edges = [
            edge for edge, sign in zip(cycle_edges, signs)
            if sign == -1
        ]

        if len(negative_edges) != 1:
            continue

        negative_edge = negative_edges[0]
        other_edges = [edge for edge in cycle_edges if edge != negative_edge]

        lhs = get_edge_value(x_values, negative_edge)
        rhs = sum(get_edge_value(x_values, edge) for edge in other_edges)

        if lhs > rhs + tolerance:
            violated_cycles.append((cycle, cycle_edges, signs, lhs, rhs))

    return violated_cycles


if __name__ == "__main__":
    start_time = time.time()
    # ============================================================
    # Parameters for random signed graph
    # ============================================================

    n = 150
    p_positive = 0.5
    p_delete = 0.15
    seed = 1

    # ============================================================
    # Generate complete random signed graph
    # ============================================================

    S = generate_signed_complete_graph(n, p_positive, seed)
    G = matrix_to_graph(S)

    # ============================================================
    # Complete graph: Pivot
    # ============================================================

    pivot_clusters, pivots = run_pivot(S, seed)
    pivot_cluster_count = len(pivot_clusters)
    pivot_cost = calculate_clustering_cost(S, pivot_clusters)

    # ============================================================
    # Complete graph: bad triangles
    # ============================================================

    # all_bad_triangles = find_bad_triangles(S)
    # edge_to_triangles = make_edge_to_triangle_map(all_bad_triangles)

    # min_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_min(edge_to_triangles)
    # min_num_bad_triangles = count_bad_triangles(min_disjoint_bad_triangles)

    # max_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_max(edge_to_triangles)
    # max_num_bad_triangles = count_bad_triangles(max_disjoint_bad_triangles)

    # ============================================================
    # Complete graph: ILP
    # ============================================================

    # ilp_cost, ilp_x_values, bad_cycles_ilp = solve_ilp(
    #     S,
    #     verbose=False,
    #     relax=False,
    #     add_four_cycles=False
    # )

    # ilp_clusters = find_ilp_clusters(ilp_x_values, n)
    # ilp_cluster_count = len(ilp_clusters)

    # ============================================================
    # Complete graph: ILP relaxation
    # ============================================================

    lp_cost, lp_x_values, bad_cycles_lp = solve_ilp(
        S,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    # ============================================================
    # Complete graph: bad-triangle LP bounds
    # ============================================================

    # primal_cost, primal_x_values = solve_primal(
    #     S,
    #     all_bad_triangles,
    #     verbose=False
    # )

    # dual_cost, dual_x_values = solve_dual(
    #     S,
    #     all_bad_triangles,
    #     verbose=False
    # )

    # ============================================================
    # Generate incomplete graph by deleting edges
    # ============================================================

    S_new, num_edges_deleted = delete_edges(S, p_delete, seed)
    G_new = matrix_to_graph(S_new)

    # ============================================================
    # Incomplete graph: Pivot
    # ============================================================

    pivot_clusters_new, pivots_new = run_pivot(S_new, seed)
    pivot_cluster_count_new = len(pivot_clusters_new)
    pivot_cost_new = calculate_clustering_cost(S_new, pivot_clusters_new)

    # ============================================================
    # Incomplete graph: bad triangles
    # ============================================================

    # all_bad_triangles_new = find_bad_triangles(S_new)
    # edge_to_triangles_new = make_edge_to_triangle_map(all_bad_triangles_new)

    # min_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_min(edge_to_triangles_new)
    # min_num_bad_triangles_new = count_bad_triangles(min_disjoint_bad_triangles_new)

    # max_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_max(edge_to_triangles_new)
    # max_num_bad_triangles_new = count_bad_triangles(max_disjoint_bad_triangles_new)

    # ============================================================
    # Incomplete graph: ILP without bad 4-cycle constraints
    # ============================================================

    # ilp_cost_new_no4, ilp_x_values_new_no4, bad_cycles_ilp_new_no4 = solve_ilp(
    #     S_new,
    #     verbose=False,
    #     relax=False,
    #     add_four_cycles=False
    # )

    # ilp_clusters_new_no4 = find_ilp_clusters(ilp_x_values_new_no4, n)
    # ilp_cluster_count_new_no4 = len(ilp_clusters_new_no4)

    # ============================================================
    # Incomplete graph: ILP with bad 4-cycle constraints
    # ============================================================

    # ilp_cost_new_with4, ilp_x_values_new_with4, bad_cycles_ilp_new_with4 = solve_ilp(
    #     S_new,
    #     verbose=False,
    #     relax=False,
    #     add_four_cycles=True
    # )

    # ilp_clusters_new_with4 = find_ilp_clusters(ilp_x_values_new_with4, n)
    # ilp_cluster_count_new_with4 = len(ilp_clusters_new_with4)

    # ============================================================
    # Incomplete graph: ILP relaxation without bad 4-cycle constraints
    # ============================================================

    # lp_cost_new_no4, lp_x_values_new_no4, bad_cycles_lp_new_no4 = solve_ilp(
    #     S_new,
    #     verbose=False,
    #     relax=True,
    #     add_four_cycles=False
    # )

    # ============================================================
    # Incomplete graph: ILP relaxation with bad 4-cycle constraints
    # ============================================================

    lp_cost_new_with4, lp_x_values_new_with4, bad_cycles_lp_new_with4 = solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=True
    )

    # ============================================================
    # Check violated bad 4-cycle constraints
    # ============================================================

    # violated_cycles_ilp_new = check_violated_bad_cycles(
    #     ilp_x_values_new_no4,
    #     bad_cycles_ilp_new_with4
    # )

    # violated_cycles_lp_new = check_violated_bad_cycles(
    #     lp_x_values_new_no4,
    #     bad_cycles_lp_new_with4
    # )

    # ============================================================
    # Incomplete graph: bad-triangle LP bounds
    # ============================================================

    # primal_cost_new, primal_x_values_new = solve_primal(
    #     S_new,
    #     all_bad_triangles_new,
    #     verbose=False
    # )

    # dual_cost_new, dual_x_values_new = solve_dual(
    #     S_new,
    #     all_bad_triangles_new,
    #     verbose=False
    # )
    runtime = time.time() - start_time
    # ============================================================
    # Print results
    # ============================================================

    print_section("Random Signed Graph")
    print("Number of nodes:", n)
    print("p_positive:", p_positive)
    print("p_delete:", p_delete)
    print("seed:", seed)
    print("runtime:", runtime)

    print_section("Complete Graph")
    print("Pivot cost of complete graph:", pivot_cost)
    print("Pivot cluster count of complete graph:", pivot_cluster_count)
    # print("Total number of bad triangles of complete graph:", len(all_bad_triangles))
    # print("Minimum amount of disjoint bad triangles of complete graph:", min_num_bad_triangles)
    # print("Maximum amount of disjoint bad triangles of complete graph:", max_num_bad_triangles)

    # print_subsection("Bad-triangle LP bounds")
    # print("LP-primal optimal cost of complete graph:", primal_cost)
    # print("LP-dual optimal cost of complete graph:", dual_cost)

    # print_subsection("ILP")
    # print("ILP optimal cost of complete graph:", ilp_cost)
    # print("ILP cluster count of complete graph:", ilp_cluster_count)

    print_subsection("ILP Relaxation")
    print("ILP relaxation cost of complete graph:", lp_cost)

    print_section("Edge-Deleted Graph")
    print("Number of edges deleted:", num_edges_deleted)
    print("Pivot cost of new graph:", pivot_cost_new)
    print("Pivot cluster count of new graph:", pivot_cluster_count_new)
    # print("Total number of bad triangles of new graph:", len(all_bad_triangles_new))
    # print("Minimum amount of disjoint bad triangles of new graph:", min_num_bad_triangles_new)
    # print("Maximum amount of disjoint bad triangles of new graph:", max_num_bad_triangles_new)

    # print_subsection("Bad-triangle LP bounds")
    # print("LP-primal optimal cost of new graph:", primal_cost_new)
    # print("LP-dual optimal cost of new graph:", dual_cost_new)

    # print_subsection("ILP")
    # print("ILP cost without 4-cycles:", ilp_cost_new_no4)
    # print("ILP cost with 4-cycles:", ilp_cost_new_with4)
    # print("ILP cluster count without 4-cycles:", ilp_cluster_count_new_no4)
    # print("ILP cluster count with 4-cycles:", ilp_cluster_count_new_with4)
    # print("Bad 4-cycles detected ILP:", len(bad_cycles_ilp_new_with4))
    # print("Violated bad 4-cycles in ILP no-4 solution:", len(violated_cycles_ilp_new))

    print_subsection("ILP Relaxation")
    # print("ILP relaxation cost without 4-cycles:", lp_cost_new_no4)
    print("ILP relaxation cost with 4-cycles:", lp_cost_new_with4)
    # print("Bad 4-cycles detected LP:", len(bad_cycles_lp_new_with4))
    # print("Violated bad 4-cycles in LP no-4 solution:", len(violated_cycles_lp_new))

    # ============================================================
    # Draw clustered graphs
    # ============================================================

    # draw_graphs(
    #     G_complete=G,
    #     pivot_clusters=pivot_clusters,
    #     # ilp_clusters=ilp_clusters,
    #     G_new=G_new,
    #     pivot_clusters_new=pivot_clusters_new,
    #     # ilp_clusters_new=ilp_clusters_new_with4,
    #     pivots=pivots,
    #     pivots_new=pivots_new
    # )

    # ============================================================
    # Save results
    # ============================================================

    # results = {
    #     "graph_params": {
    #         "graph_type": "random_signed",
    #         "num_nodes": n,
    #         "p_positive": p_positive,
    #         "seed": seed,
    #         "p_delete": p_delete,
    #         "num_edges_deleted": num_edges_deleted
    #     },

    #     "complete_graph": {
    #         "pivot": {
    #             "cost": pivot_cost,
    #             "cluster_count": pivot_cluster_count,
    #             "pivots": pivots
    #         },

    #         "bad_triangles": {
    #             "total_count": len(all_bad_triangles),
    #             "min_edge_disjoint_count": min_num_bad_triangles,
    #             "max_edge_disjoint_count": max_num_bad_triangles
    #         },

    #         "ilp": {
    #             "cost": ilp_cost,
    #             "cluster_count": ilp_cluster_count,
    #             "bad_4_cycles_count": len(bad_cycles_ilp)
    #         },

    #         "ilp_relaxation": {
    #             "cost": lp_cost,
    #             "bad_4_cycles_count": len(bad_cycles_lp)
    #         },

    #         "bad_triangle_lp_bounds": {
    #             "primal_cost": primal_cost,
    #             "dual_cost": dual_cost
    #         }
    #     },

    #     "new_graph": {
    #         "pivot": {
    #             "cost": pivot_cost_new,
    #             "cluster_count": pivot_cluster_count_new,
    #             "pivots": pivots_new
    #         },

    #         "bad_triangles": {
    #             "total_count": len(all_bad_triangles_new),
    #             "min_edge_disjoint_count": min_num_bad_triangles_new,
    #             "max_edge_disjoint_count": max_num_bad_triangles_new
    #         },

    #         "ilp": {
    #             "without_4_cycles": {
    #                 "cost": ilp_cost_new_no4,
    #                 "cluster_count": ilp_cluster_count_new_no4,
    #                 "violated_bad_4_cycles_count": len(violated_cycles_ilp_new)
    #             },
    #             "with_4_cycles": {
    #                 "cost": ilp_cost_new_with4,
    #                 "cluster_count": ilp_cluster_count_new_with4,
    #                 "bad_4_cycles_count": len(bad_cycles_ilp_new_with4)
    #             }
    #         },

    #         "ilp_relaxation": {
    #             "without_4_cycles": {
    #                 "cost": lp_cost_new_no4,
    #                 "violated_bad_4_cycles_count": len(violated_cycles_lp_new)
    #             },
    #             "with_4_cycles": {
    #                 "cost": lp_cost_new_with4,
    #                 "bad_4_cycles_count": len(bad_cycles_lp_new_with4)
    #             }
    #         },

    #         "bad_triangle_lp_bounds": {
    #             "primal_cost": primal_cost_new,
    #             "dual_cost": dual_cost_new
    #         }
    #     }
    # }

    # save_results_append(RESULTS_FILE, results)
