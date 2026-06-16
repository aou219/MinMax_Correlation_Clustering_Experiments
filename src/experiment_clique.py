from graph_generation import generate_clique_signed_graph, matrix_to_graph
from draw_graphs_clique import draw_clique_graphs

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results
)

RESULTS_FILE = "results/experiments_results_clique/clq_n10.json"


if __name__ == "__main__":

    # ============================================================
    # Parameters for clique graph
    # ============================================================

    cluster_size_cases = [
    [5,5],[3,3,4],[3,2,5]
]
    for cluster_sizes in cluster_size_cases:
        n = sum(cluster_sizes)

        p_pos_inside = 0.9
        p_pos_between = 0.1
        p_delete = 0.15

        seeds = [41,42,43,44,45,46,47,48,49,50]
        pivot_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        draw_graph = False

        # ============================================================
        # Generate complete clique graph
        # ============================================================
        for seed in seeds:
            print("\n\nRUNNING CLIQUE EXPERIMENT")
            print("cluster_sizes:", cluster_sizes)
            print("seed:", seed)
            S, true_clusters = generate_clique_signed_graph(
                cluster_sizes=cluster_sizes,
                p_pos_inside=p_pos_inside,
                p_pos_between=p_pos_between,
                seed=seed
            )

            G = matrix_to_graph(S)

            # ============================================================
            # Run full experiment
            # ============================================================

            experiment_data = run_full_experiment(
                S=S,
                p_delete=p_delete,
                seed=seed,
                pivot_seeds=pivot_seeds
            )

            # ============================================================
            # Graph-specific parameters
            # ============================================================

            graph_params = {
                "graph_type": "clique",
                "num_nodes": n,
                "cluster_sizes": cluster_sizes,
                "p_pos_inside": p_pos_inside,
                "p_pos_between": p_pos_between,
                "seed": seed,
                "pivot_seeds": pivot_seeds,
                "p_delete": p_delete,
                "num_edges_deleted": experiment_data["num_edges_deleted"],
                "num_true_clusters": len(true_clusters),
                "true_cluster_sizes": [len(cluster) for cluster in true_clusters]
            }

            # ============================================================
            # Print results
            # ============================================================

            print_standard_results(
                graph_type="clique",
                graph_params=graph_params,
                experiment_data=experiment_data
            )

            # ============================================================
            # Save results
            # ============================================================

            results = build_saveable_results(
                graph_params=graph_params,
                experiment_data=experiment_data
            )

            save_results_append(RESULTS_FILE, results)

            # ============================================================
            # Draw clustered graphs
            # ============================================================

            if draw_graph:
                draw_clique_graphs(
                    G_complete=G,
                    true_clusters=true_clusters,
                    pivot_clusters=experiment_data["pivot_clusters"],
                    ilp_clusters=experiment_data["ilp_clusters"],
                    G_new=experiment_data["G_new"],
                    pivot_clusters_new=experiment_data["pivot_clusters_new"],
                    ilp_clusters_new=experiment_data["ilp_clusters_new_with4"],
                    pivots=experiment_data["pivots"],
                    pivots_new=experiment_data["pivots_new"]
                )
