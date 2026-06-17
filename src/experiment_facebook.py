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
    build_saveable_results
)


def get_all_nodes_from_edges_and_circles(edge_nodes, circles):
    """
    Use the full ego-network:
    - all nodes that appear in the .edges file
    - plus all nodes that appear in the .circles file
    """
    circle_nodes = set()

    for circle in circles:
        circle_nodes.update(circle["nodes"])

    all_nodes = edge_nodes | circle_nodes

    return sorted(all_nodes)


if __name__ == "__main__":

    # ============================================================
    # Full Facebook ego-network experiments
    # ============================================================

    ego_ids = ["414", "686", "348", "0"]

    p_delete = 0.15
    seed = 1
    pivot_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for ego_id in ego_ids:

        edges_file = f"data/facebook/{ego_id}.edges"
        circles_file = f"data/facebook/{ego_id}.circles"

        results_file = f"results/experiments_results_facebook/full/fb_ego{ego_id}_full_without_ilp.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        # Remove old result file for this ego, so rerunning does not append duplicates.
        if os.path.exists(results_file):
            os.remove(results_file)

        # ============================================================
        # Load full ego-network
        # ============================================================

        edge_nodes, facebook_edges = load_facebook_ego_edges(edges_file)
        circles = load_facebook_circles(circles_file)

        all_nodes = get_all_nodes_from_edges_and_circles(
            edge_nodes=edge_nodes,
            circles=circles
        )

        n = len(all_nodes)

        circle_sizes = sorted(
            [len(circle["nodes"]) for circle in circles],
            reverse=True
        )

        # ============================================================
        # Build complete signed graph
        # ============================================================

        S, node_to_index, positive_count, negative_count = build_complete_signed_matrix_from_facebook_sample(
            all_nodes,
            facebook_edges
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
            "graph_type": "facebook_full_ego",
            "ego_id": ego_id,
            "num_nodes": n,
            "num_facebook_edges": len(facebook_edges),
            "num_circles": len(circles),
            "circle_sizes": circle_sizes,
            "seed": seed,
            "pivot_seeds": pivot_seeds,
            "p_delete": p_delete,
            "num_edges_deleted": experiment_data["num_edges_deleted"],
            "positive_edges": positive_count,
            "negative_edges": negative_count,
            "sample_type": "full_ego_network",
            "signing_rule": {
                "existing_facebook_friendship": "+1",
                "missing_friendship_inside_full_ego_network": "-1",
                "deleted_edge": "0"
            }
        }

        # ============================================================
        # Print results
        # ============================================================

        print_standard_results(
            graph_type="facebook_full_ego",
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

        save_results_append(results_file, results)

        print()
        print("Saved:", results_file)
