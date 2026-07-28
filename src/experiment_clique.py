from graph_generation import generate_clique_signed_graph, matrix_to_graph

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results,
)


# ============================================================
# Manual parameters for clique graph
# Change these by hand.
# ============================================================

CLUSTER_SIZE_CASES = [
    [10, 10,10,10,10,10,10,10,10,10],
    # [5, 5, 5, 5],
    # [20, 20],
    # [10, 10, 10, 10],
]

P_POS_INSIDE_VALUES = [0.9]
P_POS_BETWEEN_VALUES = [0.1]
P_DELETE_VALUES = [0.05]
SEEDS = [1]
PIVOT_SEEDS = list(range(1, 11))

# ============================================================
# Manual True/False switches
# ============================================================

SAVE_RESULTS = False
DRAW_GRAPH = False

COMPUTE_PIVOT = False
COMPUTE_BAD_TRIANGLES = True
COMPUTE_DISJOINT_BAD_TRIANGLES = False
COMPUTE_BAD_TRIANGLE_PRIMAL_BOUND = False
COMPUTE_BAD_TRIANGLE_DUAL_BOUND = False
COMPUTE_MIN_MAX = True
# Main formulation you want: all_pairs_solver.py
COMPUTE_ACTUAL_LP = True
COMPUTE_ACTUAL_ILP = False

# Old observed-edge formulation from ilp_solver.py
COMPUTE_OBSERVED_EDGE_LP = False
COMPUTE_OBSERVED_EDGE_ILP = False
COMPUTE_OBSERVED_EDGE_FOUR_CYCLE_LP = False
COMPUTE_OBSERVED_EDGE_FOUR_CYCLE_ILP = False

RESULTS_FILE = "results/experiments_results_clique/clique_manual.json"


def cluster_sizes_to_tag(cluster_sizes):
    return "_".join(str(size) for size in cluster_sizes)


if __name__ == "__main__":

    # ============================================================
    # Run experiments
    # ============================================================

    for cluster_sizes in CLUSTER_SIZE_CASES:
        n = sum(cluster_sizes)
        cluster_tag = cluster_sizes_to_tag(cluster_sizes)

        for p_pos_inside in P_POS_INSIDE_VALUES:
            for p_pos_between in P_POS_BETWEEN_VALUES:
                for p_delete in P_DELETE_VALUES:
                    for seed in SEEDS:

                        print("\n" + "=" * 70)
                        print("RUNNING CLIQUE EXPERIMENT")
                        print("cluster_sizes:", cluster_sizes)
                        print("n:", n)
                        print("p_pos_inside:", p_pos_inside)
                        print("p_pos_between:", p_pos_between)
                        print("p_delete:", p_delete)
                        print("seed:", seed)
                        print("SAVE_RESULTS =", SAVE_RESULTS)
                        print("=" * 70)

                        # ============================================================
                        # Generate complete clique graph
                        # ============================================================

                        S, true_clusters = generate_clique_signed_graph(
                            cluster_sizes=cluster_sizes,
                            p_pos_inside=p_pos_inside,
                            p_pos_between=p_pos_between,
                            seed=seed,
                        )

                        G = matrix_to_graph(S)

                        # ============================================================
                        # Run full experiment
                        # ============================================================

                        experiment_data = run_full_experiment(
                            S=S,
                            p_delete=p_delete,
                            seed=seed,
                            pivot_seeds=PIVOT_SEEDS,
                            compute_pivot=COMPUTE_PIVOT,
                            compute_bad_triangles=COMPUTE_BAD_TRIANGLES,
                            compute_disjoint_bad_triangles=COMPUTE_DISJOINT_BAD_TRIANGLES,
                            compute_bad_triangle_primal_bound=COMPUTE_BAD_TRIANGLE_PRIMAL_BOUND,
                            compute_bad_triangle_dual_bound=COMPUTE_BAD_TRIANGLE_DUAL_BOUND,
                            compute_actual_lp=COMPUTE_ACTUAL_LP,
                            compute_actual_ilp=COMPUTE_ACTUAL_ILP,
                            compute_observed_edge_lp=COMPUTE_OBSERVED_EDGE_LP,
                            compute_observed_edge_ilp=COMPUTE_OBSERVED_EDGE_ILP,
                            compute_observed_edge_four_cycle_lp=COMPUTE_OBSERVED_EDGE_FOUR_CYCLE_LP,
                            compute_observed_edge_four_cycle_ilp=COMPUTE_OBSERVED_EDGE_FOUR_CYCLE_ILP,
                            compute_min_max = COMPUTE_MIN_MAX
                        )

                        # ============================================================
                        # Graph-specific parameters
                        # ============================================================

                        graph_params = {
                            "graph_type": "clique",
                            "num_nodes": n,
                            "cluster_sizes": cluster_sizes,
                            "cluster_tag": cluster_tag,
                            "p_pos_inside": p_pos_inside,
                            "p_pos_between": p_pos_between,
                            "seed": seed,
                            "pivot_seeds": PIVOT_SEEDS,
                            "p_delete": p_delete,
                            "num_edges_deleted": experiment_data["num_edges_deleted"],
                            "num_true_clusters": len(true_clusters),
                            "true_cluster_sizes": [len(cluster) for cluster in true_clusters],
                        }

                        # ============================================================
                        # Print results
                        # ============================================================

                        print_standard_results(
                            graph_type="clique",
                            graph_params=graph_params,
                            experiment_data=experiment_data,
                        )

                        # ============================================================
                        # Save results, optional
                        # ============================================================

                        if SAVE_RESULTS:
                            results = build_saveable_results(
                                graph_params=graph_params,
                                experiment_data=experiment_data,
                            )
                            save_results_append(RESULTS_FILE, results)

                        # ============================================================
                        # Draw clustered graphs, optional
                        # ============================================================

                        if DRAW_GRAPH:
                            draw_clique_graphs(
                                G_complete=G,
                                true_clusters=true_clusters,
                                pivot_clusters=experiment_data["pivot_clusters"],
                                ilp_clusters=experiment_data["actual_ilp_clusters"],
                                G_new=experiment_data["G_new"],
                                pivot_clusters_new=experiment_data["pivot_clusters_new"],
                                ilp_clusters_new=experiment_data["actual_ilp_clusters_new"],
                                pivots=experiment_data["pivots"],
                                pivots_new=experiment_data["pivots_new"],
                            )
