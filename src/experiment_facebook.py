import os

from facebook_sampling import (
    load_facebook_ego_edges,
    load_facebook_circles,
    build_complete_signed_matrix_from_facebook_sample,
)

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results,
)


# ============================================================
# Manual parameters for Facebook ego graphs
# Change these by hand.
# ============================================================

EGO_IDS = ["348", "0"]
P_DELETE_VALUES = [0.05]
SEEDS = [1]
PIVOT_SEEDS = list(range(1, 11))

# ============================================================
# Manual True/False switches
# ============================================================

SAVE_RESULTS = False

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


def get_all_nodes_from_edges_and_circles(edge_nodes, circles):
    circle_nodes = set()

    for circle in circles:
        circle_nodes.update(circle["nodes"])

    all_nodes = edge_nodes | circle_nodes
    return sorted(all_nodes)


if __name__ == "__main__":

    # ============================================================
    # Full Facebook ego-network experiments
    # ============================================================

    for ego_id in EGO_IDS:
        for p_delete in P_DELETE_VALUES:
            for seed in SEEDS:

                edges_file = f"data/facebook/{ego_id}.edges"
                circles_file = f"data/facebook/{ego_id}.circles"

                results_file = f"results/experiments_results_facebook/full/fb_ego{ego_id}_manual.json"
                if SAVE_RESULTS:
                    os.makedirs(os.path.dirname(results_file), exist_ok=True)

                print("\n" + "=" * 70)
                print("RUNNING FACEBOOK EXPERIMENT")
                print("ego_id:", ego_id)
                print("p_delete:", p_delete)
                print("seed:", seed)
                print("SAVE_RESULTS =", SAVE_RESULTS)
                print("=" * 70)

                # ============================================================
                # Load full ego-network
                # ============================================================

                edge_nodes, facebook_edges = load_facebook_ego_edges(edges_file)
                circles = load_facebook_circles(circles_file)

                all_nodes = get_all_nodes_from_edges_and_circles(
                    edge_nodes=edge_nodes,
                    circles=circles,
                )

                n = len(all_nodes)
                circle_sizes = sorted(
                    [len(circle["nodes"]) for circle in circles],
                    reverse=True,
                )

                # ============================================================
                # Build complete signed graph
                # ============================================================

                S, node_to_index, positive_count, negative_count = build_complete_signed_matrix_from_facebook_sample(
                    all_nodes,
                    facebook_edges,
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
                    "graph_type": "facebook_full_ego",
                    "ego_id": ego_id,
                    "num_nodes": n,
                    "num_facebook_edges": len(facebook_edges),
                    "num_circles": len(circles),
                    "circle_sizes": circle_sizes,
                    "seed": seed,
                    "pivot_seeds": PIVOT_SEEDS,
                    "p_delete": p_delete,
                    "num_edges_deleted": experiment_data["num_edges_deleted"],
                    "positive_edges": positive_count,
                    "negative_edges": negative_count,
                    "sample_type": "full_ego_network",
                    "signing_rule": {
                        "existing_facebook_friendship": "+1",
                        "missing_friendship_inside_full_ego_network": "-1",
                        "deleted_edge": "0",
                    },
                }

                # ============================================================
                # Print results
                # ============================================================

                print_standard_results(
                    graph_type="facebook_full_ego",
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
                    save_results_append(results_file, results)
                    print()
                    print("Saved:", results_file)
