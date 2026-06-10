import os
import json
import numpy as np

from graph_generation import generate_clique_signed_graph, matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from draw_graphs_clique import draw_clique_graphs
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
from experiments import check_violated_bad_cycles, save_results_append, print_section

RESULTS_FILE = "results/experiments_results_clique.json"


if __name__ == "__main__":

    # ============================================================
    # Parameters for clique graph
    # ============================================================

    cluster_sizes = [10, 40, 20]
    n = sum(cluster_sizes)

    # Inside planted clusters, edges are usually positive.
    p_pos_inside = 0.8

    # Between planted clusters, edges are usually negative.
    p_pos_between = 0.2

    # Edge deletion probability after generating the complete planted graph.
    p_delete = 0.25

    seed = None

    # ============================================================
    # Generate complete clique graph
    # ============================================================

    S, true_clusters = generate_clique_signed_graph(
        cluster_sizes=cluster_sizes,
        p_pos_inside=p_pos_inside,
        p_pos_between=p_pos_between,
        seed=seed
    )

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

    all_bad_triangles = find_bad_triangles(S)
    edge_to_triangles = make_edge_to_triangle_map(all_bad_triangles)

    min_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_min(edge_to_triangles)
    min_num_bad_triangles = count_bad_triangles(min_disjoint_bad_triangles)

    max_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_max(edge_to_triangles)
    max_num_bad_triangles = count_bad_triangles(max_disjoint_bad_triangles)

    # ============================================================
    # Complete graph: ILP
    # ============================================================

    ilp_cost, ilp_x_values, bad_cycles_ilp = solve_ilp(
        S,
        verbose=False,
        relax=False,
        add_four_cycles=False
    )

    ilp_clusters = find_ilp_clusters(ilp_x_values, n)
    ilp_cluster_count = len(ilp_clusters)

    # ============================================================
    # Complete graph: LP relaxation
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

    primal_cost, primal_x_values = solve_primal(
        S,
        all_bad_triangles,
        verbose=False
    )

    dual_cost, dual_x_values = solve_dual(
        S,
        all_bad_triangles,
        verbose=False
    )

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

    all_bad_triangles_new = find_bad_triangles(S_new)
    edge_to_triangles_new = make_edge_to_triangle_map(all_bad_triangles_new)

    min_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_min(edge_to_triangles_new)
    min_num_bad_triangles_new = count_bad_triangles(min_disjoint_bad_triangles_new)

    max_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_max(edge_to_triangles_new)
    max_num_bad_triangles_new = count_bad_triangles(max_disjoint_bad_triangles_new)

    # ============================================================
    # Incomplete graph: ILP without bad 4-cycle constraints
    # ============================================================

    ilp_cost_new_no4, ilp_x_values_new_no4, bad_cycles_ilp_new_no4 = solve_ilp(
        S_new,
        verbose=False,
        relax=False,
        add_four_cycles=False
    )

    ilp_clusters_new_no4 = find_ilp_clusters(ilp_x_values_new_no4, n)
    ilp_cluster_count_new_no4 = len(ilp_clusters_new_no4)

    # ============================================================
    # Incomplete graph: ILP with bad 4-cycle constraints
    # ============================================================

    ilp_cost_new_with4, ilp_x_values_new_with4, bad_cycles_ilp_new_with4 = solve_ilp(
        S_new,
        verbose=False,
        relax=False,
        add_four_cycles=True
    )

    ilp_clusters_new_with4 = find_ilp_clusters(ilp_x_values_new_with4, n)
    ilp_cluster_count_new_with4 = len(ilp_clusters_new_with4)

    # ============================================================
    # Incomplete graph: LP relaxation without bad 4-cycle constraints
    # ============================================================

    lp_cost_new_no4, lp_x_values_new_no4, bad_cycles_lp_new_no4 = solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    # ============================================================
    # Incomplete graph: LP relaxation with bad 4-cycle constraints
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

    violated_cycles_ilp_new = check_violated_bad_cycles(
        ilp_x_values_new_no4,
        bad_cycles_ilp_new_with4
    )

    violated_cycles_lp_new = check_violated_bad_cycles(
        lp_x_values_new_no4,
        bad_cycles_lp_new_with4
    )

    # ============================================================
    # Incomplete graph: bad-triangle LP bounds
    # ============================================================

    primal_cost_new, primal_x_values_new = solve_primal(
        S_new,
        all_bad_triangles_new,
        verbose=False
    )

    dual_cost_new, dual_x_values_new = solve_dual(
        S_new,
        all_bad_triangles_new,
        verbose=False
    )

    # ============================================================
    # Print results
    # ============================================================

    print_section(" Clique Graph Parameters")
    print("Cluster sizes:", cluster_sizes)
    print("p_pos_inside:", p_pos_inside)
    print("p_pos_between:", p_pos_between)

    print_section("Complete Planted Graph")
    print("Pivot cost of complete graph:", pivot_cost)
    print("Pivot cluster count of complete graph:", pivot_cluster_count)
    print("Minimum amount of disjoint bad triangles of complete graph:", min_num_bad_triangles)
    print("Maximum amount of disjoint bad triangles of complete graph:", max_num_bad_triangles)

    print("\n--- Bad-triangle LP bounds ---")
    print("LP-primal optimal cost of complete graph:", primal_cost)
    print("LP-dual optimal cost of complete graph:", dual_cost)

    print("\n--- ILP ---")
    print("ILP optimal cost of complete graph:", ilp_cost)
    print("ILP cluster count of complete graph:", ilp_cluster_count)

    print("\n--- LP Relaxation ---")
    print("LP relaxation cost of complete graph:", lp_cost)

    print_section("Edge-Deleted Planted Graph")
    print("Number of edges deleted:", num_edges_deleted)
    print("Pivot cost of new graph:", pivot_cost_new)
    print("Pivot cluster count of new graph:", pivot_cluster_count_new)
    print("Minimum amount of disjoint bad triangles of new graph:", min_num_bad_triangles_new)
    print("Maximum amount of disjoint bad triangles of new graph:", max_num_bad_triangles_new)

    print("\n--- Bad-triangle LP bounds ---")
    print("LP-primal optimal cost of new graph:", primal_cost_new)
    print("LP-dual optimal cost of new graph:", dual_cost_new)

    print("\n--- ILP ---")
    print("ILP cost without 4-cycles:", ilp_cost_new_no4)
    print("ILP cost with 4-cycles:", ilp_cost_new_with4)
    print("ILP cluster count without 4-cycles:", ilp_cluster_count_new_no4)
    print("ILP cluster count with 4-cycles:", ilp_cluster_count_new_with4)
    print("Bad 4-cycles detected ILP:", len(bad_cycles_ilp_new_with4))
    print("Violated bad 4-cycles in ILP no-4 solution:", len(violated_cycles_ilp_new))

    print("\n--- LP relaxation ---")
    print("LP relaxation cost without 4-cycles:", lp_cost_new_no4)
    print("LP relaxation cost with 4-cycles:", lp_cost_new_with4)
    print("Bad 4-cycles detected LP:", len(bad_cycles_lp_new_with4))
    print("Violated bad 4-cycles in LP no-4 solution:", len(violated_cycles_lp_new))

    # ============================================================
    # Draw clustered graphs
    # ============================================================

    draw_clique_graphs(
        G_complete=G,
        true_clusters=true_clusters,
        pivot_clusters=pivot_clusters,
        ilp_clusters=ilp_clusters,
        G_new=G_new,
        pivot_clusters_new=pivot_clusters_new,
        ilp_clusters_new=ilp_clusters_new_with4,
        pivots=pivots,
        pivots_new=pivots_new
    )

    # ============================================================
    # Save results
    # ============================================================

    results = {
        "graph_params": {
            "graph_type": "clique",
            "num_nodes": n,
            "cluster_sizes": cluster_sizes,
            "p_pos_inside": p_pos_inside,
            "p_pos_between": p_pos_between,
            "seed": seed,
            "p_delete": p_delete,
            "num_edges_deleted": num_edges_deleted,
            "true_clusters": true_clusters
        },

        "complete_graph": {
            "pivot": {
                "cost": pivot_cost,
                "cluster_count": pivot_cluster_count,
                "pivots": pivots
            },

            "bad_triangles": {
                "total_count": len(all_bad_triangles),
                "min_edge_disjoint_count": min_num_bad_triangles,
                "max_edge_disjoint_count": max_num_bad_triangles
            },

            "ilp": {
                "cost": ilp_cost,
                "cluster_count": ilp_cluster_count,
                "bad_4_cycles_count": len(bad_cycles_ilp)
            },

            "lp_relaxation": {
                "cost": lp_cost,
                "bad_4_cycles_count": len(bad_cycles_lp)
            },

            "bad_triangle_lp_bounds": {
                "primal_cost": primal_cost,
                "dual_cost": dual_cost
            }
        },

        "new_graph": {
            "pivot": {
                "cost": pivot_cost_new,
                "cluster_count": pivot_cluster_count_new,
                "pivots": pivots_new
            },

            "bad_triangles": {
                "total_count": len(all_bad_triangles_new),
                "min_edge_disjoint_count": min_num_bad_triangles_new,
                "max_edge_disjoint_count": max_num_bad_triangles_new
            },

            "ilp": {
                "without_4_cycles": {
                    "cost": ilp_cost_new_no4,
                    "cluster_count": ilp_cluster_count_new_no4,
                    "violated_bad_4_cycles_count": len(violated_cycles_ilp_new)
                },
                "with_4_cycles": {
                    "cost": ilp_cost_new_with4,
                    "cluster_count": ilp_cluster_count_new_with4,
                    "bad_4_cycles_count": len(bad_cycles_ilp_new_with4)
                }
            },

            "lp_relaxation": {
                "without_4_cycles": {
                    "cost": lp_cost_new_no4,
                    "violated_bad_4_cycles_count": len(violated_cycles_lp_new)
                },
                "with_4_cycles": {
                    "cost": lp_cost_new_with4,
                    "bad_4_cycles_count": len(bad_cycles_lp_new_with4)
                }
            },

            "bad_triangle_lp_bounds": {
                "primal_cost": primal_cost_new,
                "dual_cost": dual_cost_new
            }
        }
    }

    # save_results_append(RESULTS_FILE, results)
