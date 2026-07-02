from pathlib import Path

from graph_generation import generate_signed_complete_graph
from draw_graphs import draw_graphs

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results,
)


# ============================================================
# Manual parameters for random signed graphs
# Change these by hand.
# ============================================================

N_VALUES = [100]
P_DELETE_VALUES = [0.15]
P_POSITIVE_VALUES = [0.5]
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

# Main formulation you want: all_pairs_solver.py
COMPUTE_ACTUAL_LP = True
COMPUTE_ACTUAL_ILP = False

# Old observed-edge formulation from ilp_solver.py
COMPUTE_OBSERVED_EDGE_LP = False
COMPUTE_OBSERVED_EDGE_ILP = False
COMPUTE_OBSERVED_EDGE_FOUR_CYCLE_LP = False
COMPUTE_OBSERVED_EDGE_FOUR_CYCLE_ILP = False


def p_delete_to_folder(p_delete):
    return f"p_delete_{int(round(p_delete * 100)):03d}"


def p_positive_to_tag(p_positive):
    return f"p{int(round(p_positive * 10)):02d}"


if __name__ == "__main__":

    # ============================================================
    # Run experiments
    # ============================================================

    for n in N_VALUES:
        for p_delete in P_DELETE_VALUES:
            p_delete_folder = p_delete_to_folder(p_delete)

            for p_positive in P_POSITIVE_VALUES:
                p_tag = p_positive_to_tag(p_positive)

                results_dir = Path(
                    f"results/{p_delete_folder}/experiments_results_random/n{n}"
                )
                results_file = results_dir / f"random_n{n}_{p_tag}.json"

                print("\n" + "=" * 70)
                print("Running random graph experiments")
                print(f"n = {n}, p_positive = {p_positive}, p_delete = {p_delete}")
                print("SAVE_RESULTS =", SAVE_RESULTS)
                if SAVE_RESULTS:
                    print("Saving to:", results_file)
                print("=" * 70)

                for seed in SEEDS:

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
                    )

                    # ============================================================
                    # Graph-specific parameters
                    # ============================================================

                    graph_params = {
                        "graph_type": "random",
                        "num_nodes": n,
                        "p_positive": p_positive,
                        "seed": seed,
                        "pivot_seeds": PIVOT_SEEDS,
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
                    # Save results, optional
                    # ============================================================

                    if SAVE_RESULTS:
                        results = build_saveable_results(
                            graph_params=graph_params,
                            experiment_data=experiment_data,
                        )
                        save_results_append(str(results_file), results)

                    # ============================================================
                    # Draw clustered graphs, optional
                    # ============================================================

                    if DRAW_GRAPH:
                        draw_graphs(
                            G_complete=experiment_data["G"],
                            pivot_clusters=experiment_data["pivot_clusters"],
                            ilp_clusters=experiment_data["actual_ilp_clusters"],
                            G_new=experiment_data["G_new"],
                            pivot_clusters_new=experiment_data["pivot_clusters_new"],
                            ilp_clusters_new=experiment_data["actual_ilp_clusters_new"],
                            pivots=experiment_data["pivots"],
                            pivots_new=experiment_data["pivots_new"],
                        )
