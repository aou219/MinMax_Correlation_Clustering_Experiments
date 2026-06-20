from pathlib import Path

from graph_generation import generate_signed_complete_graph
from draw_graphs import draw_graphs

from experiment_helpers import (
save_results_append,
run_full_experiment,
print_standard_results,
build_saveable_results,
)

def p_delete_to_folder(p_delete):
    """
    0.05 -> p_delete_005
    0.15 -> p_delete_015
    0.25 -> p_delete_025
    0.40 -> p_delete_040
    """
    return f"p_delete_{int(round(p_delete * 100)):03d}"

def p_positive_to_tag(p_positive):
    """
    0.2 -> p02
    0.5 -> p05
    0.8 -> p08
    """
    return f"p{int(round(p_positive * 10)):02d}"

if __name__ == "__main__":


    # ============================================================
    # Parameters for one random signed graph instance
    # ============================================================

    n = 20
    p_positive = 0.5
    p_delete = 0.25
    seed = 1

    pivot_seeds = list(range(1, 11))
    draw_graph = True

    # p_delete_folder = p_delete_to_folder(p_delete)
    # p_tag = p_positive_to_tag(p_positive)

    # ============================================================
    # Output path
    # ============================================================

    # results_dir = Path(
    #     f"results/{p_delete_folder}/experiments_results_random/n{n}"
    # )
    # results_dir.mkdir(parents=True, exist_ok=True)

    # results_file = results_dir / f"random_n{n}_{p_tag}.json"

    print("\n" + "=" * 70, flush=True)
    print("RUNNING ONE RANDOM GRAPH EXPERIMENT", flush=True)
    print(f"n = {n}", flush=True)
    print(f"p_positive = {p_positive}", flush=True)
    print(f"p_delete = {p_delete}", flush=True)
    print(f"seed = {seed}", flush=True)
    # print(f"Saving to: {results_file}", flush=True)
    print("=" * 70, flush=True)

    # ============================================================
    # Generate complete random signed graph
    # ============================================================

    S = generate_signed_complete_graph(
        n=n,
        p_positive=p_positive,
        seed=seed,
    )

    # ============================================================
    # Run full experiment
    # ============================================================

    experiment_data = run_full_experiment(
        S=S,
        p_delete=p_delete,
        seed=seed,
        pivot_seeds=pivot_seeds,
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
        "num_edges_deleted": experiment_data["num_edges_deleted"],
    }

    # ============================================================
    # Print results
    # ============================================================

    print_standard_results(
        graph_type="random",
        graph_params=graph_params,
        experiment_data=experiment_data,
    )

    # ============================================================
    # Save results
    # ============================================================

    results = build_saveable_results(
        graph_params=graph_params,
        experiment_data=experiment_data,
    )

    # save_results_append(str(results_file), results)

    # print(f"Saved result to {results_file}", flush=True)

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
            pivots_new=experiment_data["pivots_new"],
        )

