from pathlib import Path
from graph_generation import generate_clique_signed_graph, matrix_to_graph
from draw_graphs_clique import draw_clique_graphs

from experiment_helpers import (
save_results_append,
run_full_experiment,
print_standard_results,
build_saveable_results,
)



if __name__ == "__main__":

    # ============================================================
    # Parameters for one clique/community graph instance
    # ============================================================

    p_delete = 0.15
    # p_delete_folder = p_delete_to_folder(p_delete)

    p_pos_inside = 0.9
    p_pos_between = 0.1

    seed = 1
    pivot_seeds = list(range(1, 11))

    draw_graph = True
    cluster_sizes = [5, 5, 5]

    n = sum(cluster_sizes)
    # cluster_tag = cluster_sizes_to_tag(cluster_sizes)

    # ============================================================
    # Output path
    # ============================================================

    # results_dir = Path(
    #     f"results/{p_delete_folder}/experiments_results_clique/n{n}"
    # )
    # results_dir.mkdir(parents=True, exist_ok=True)

    # results_file = results_dir / f"clq_n{n}_{cluster_tag}.json"

    print("\n" + "=" * 70, flush=True)
    print("RUNNING ONE CLIQUE EXPERIMENT", flush=True)
    print(f"n = {n}", flush=True)
    print(f"cluster_sizes = {cluster_sizes}", flush=True)
    print(f"p_pos_inside = {p_pos_inside}", flush=True)
    print(f"p_pos_between = {p_pos_between}", flush=True)
    print(f"p_delete = {p_delete}", flush=True)
    print(f"seed = {seed}", flush=True)
    # print(f"Saving to: {results_file}", flush=True)
    print("=" * 70, flush=True)

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
        pivot_seeds=pivot_seeds,
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
        draw_clique_graphs(
            G_complete=G,
            true_clusters=true_clusters,
            pivot_clusters=experiment_data["pivot_clusters"],
            ilp_clusters=experiment_data["ilp_clusters"],
            G_new=experiment_data["G_new"],
            pivot_clusters_new=experiment_data["pivot_clusters_new"],
            ilp_clusters_new=experiment_data["ilp_clusters_new_with4"],
            pivots=experiment_data["pivots"],
            pivots_new=experiment_data["pivots_new"],
        )
