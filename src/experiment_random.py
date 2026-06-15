from graph_generation import generate_signed_complete_graph
from draw_graphs import draw_graphs

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results
)


RESULTS_FILE = "results/experiments_results_random.json"


if __name__ == "__main__":

    # ============================================================
    # Parameters for random signed graph
    # ============================================================

    n = 30
    p_positive = 0.5
    p_delete = 0.15

    seeds = [1, 2]
    pivot_seeds = [1,2,3,4,5,6,7,8,9,10]
    draw_graph = False

    # ============================================================
    # Generate complete random signed graph
    # ============================================================
    for seed in seeds:
        S = generate_signed_complete_graph(
            n=n,
            p_positive=p_positive,
            seed=seed
        )

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
            "graph_type": "random",
            "num_nodes": n,
            "p_positive": p_positive,
            "seed": seed,
            "pivot_seeds": pivot_seeds,
            "p_delete": p_delete,
            "num_edges_deleted": experiment_data["num_edges_deleted"]
        }

        # ============================================================
        # Print results
        # ============================================================

        print_standard_results(
            graph_type="random",
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
            draw_graphs(
                G_complete=experiment_data["G"],
                pivot_clusters=experiment_data["pivot_clusters"],
                ilp_clusters=experiment_data["ilp_clusters"],
                G_new=experiment_data["G_new"],
                pivot_clusters_new=experiment_data["pivot_clusters_new"],
                ilp_clusters_new=experiment_data["ilp_clusters_new_with4"],
                pivots=experiment_data["pivots"],
                pivots_new=experiment_data["pivots_new"]
            )

