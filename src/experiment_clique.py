from pathlib import Path
import json

from graph_generation import generate_clique_signed_graph, matrix_to_graph
from draw_graphs_clique import draw_clique_graphs

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results
)


def p_delete_to_folder(p_delete):
    """
    0.05 -> p_delete_005
    0.15 -> p_delete_015
    0.25 -> p_delete_025
    0.40 -> p_delete_040
    """
    return f"p_delete_{int(round(p_delete * 100)):03d}"


def cluster_sizes_to_tag(cluster_sizes):
    """
    [5, 5] -> 2x5
    [3, 3, 4] -> 4_3_3
    [3, 2, 5] -> 5_3_2
    [25, 25, 25, 25] -> 4x25
    [60, 25, 10, 5] -> 60_25_10_5
    """
    sorted_sizes = sorted(cluster_sizes, reverse=True)

    if len(set(sorted_sizes)) == 1:
        return f"{len(sorted_sizes)}x{sorted_sizes[0]}"

    return "_".join(str(x) for x in sorted_sizes)


def get_completed_seeds(results_file):
    """
    Reads an existing json file and returns the seeds that are already saved.

    Expected format:
    {
        "shared_graph_params": {...},
        "experiments": [
            {
                "graph_params": {
                    "seed": 41
                },
                ...
            }
        ]
    }
    """
    results_file = Path(results_file)

    if not results_file.exists():
        return set()

    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"WARNING: Could not read {results_file}. Treating as empty.")
        return set()

    completed = set()

    # Normal format from build_saveable_results/save_results_append
    if isinstance(data, dict) and "experiments" in data:
        for experiment in data.get("experiments", []):
            graph_params = experiment.get("graph_params", {})
            seed = graph_params.get("seed")

            if seed is not None:
                completed.add(seed)

    # Fallback if a file was ever saved as list
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            graph_params = item.get("graph_params", {})
            seed = graph_params.get("seed")

            if seed is not None:
                completed.add(seed)

    return completed


if __name__ == "__main__":

    # ============================================================
    # Parameters for clique/community graphs
    # ============================================================

    cluster_size_cases = [
        # n = 10
        [5, 5],
        [3, 3, 4],
        [3, 2, 5],

        # n = 15
        [5, 5, 5],
        [5, 5, 3, 2],
        [8, 7],

        # n = 20
        [10, 10],
        [5, 5, 5, 5],
        [7, 7, 6],

        # n = 25
        [10, 10, 5],
        [12, 7, 6],
        [13, 12],
        [9, 8, 8],

        # n = 30
        [15, 10, 5],
        [20, 5, 5],
        [15, 15],
        [10, 10, 10],

        # n = 100
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [25, 25, 25, 25],
        [60, 25, 10, 5],
    ]

    # Nieuwe p_delete values.
    # p_delete_015 heb je al, dus die laten we eruit.
    p_delete_values = [0.05, 0.25, 0.40]

    p_pos_inside = 0.9
    p_pos_between = 0.1

    seeds = list(range(41, 51))
    pivot_seeds = list(range(1, 11))

    draw_graph = False

    # ============================================================
    # Run experiments
    # ============================================================

    for p_delete in p_delete_values:
        p_delete_folder = p_delete_to_folder(p_delete)

        for cluster_sizes in cluster_size_cases:
            n = sum(cluster_sizes)
            cluster_tag = cluster_sizes_to_tag(cluster_sizes)

            results_dir = Path(
                f"results/{p_delete_folder}/experiments_results_clique/n{n}"
            )
            results_dir.mkdir(parents=True, exist_ok=True)

            results_file = results_dir / f"clq_n{n}_{cluster_tag}.json"

            completed_seeds = get_completed_seeds(results_file)
            missing_seeds = [seed for seed in seeds if seed not in completed_seeds]

            if not missing_seeds:
                print(f"SKIP: already complete -> {results_file}", flush=True)
                continue

            print("\n" + "=" * 70, flush=True)
            print("RUNNING MISSING CLIQUE EXPERIMENTS", flush=True)
            print(f"n = {n}", flush=True)
            print(f"cluster_sizes = {cluster_sizes}", flush=True)
            print(f"p_delete = {p_delete}", flush=True)
            print(f"Already completed seeds: {sorted(completed_seeds)}", flush=True)
            print(f"Missing seeds: {missing_seeds}", flush=True)
            print(f"Saving to: {results_file}", flush=True)
            print("=" * 70, flush=True)

            for seed in missing_seeds:
                print(
                    f"Running seed {seed} | "
                    f"n={n} | clusters={cluster_sizes} | p_delete={p_delete}",
                    flush=True
                )

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

                save_results_append(str(results_file), results)

                print(
                    f"Saved seed {seed} to {results_file}",
                    flush=True
                )

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