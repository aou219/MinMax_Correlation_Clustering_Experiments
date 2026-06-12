from facebook_sampling import (
    load_facebook_ego_edges,
    load_facebook_circles,
    sample_nodes_from_circles,
    build_complete_signed_matrix_from_facebook_sample,
)

from draw_graphs_clique import draw_clique_graphs

from experiment_helpers import (
    save_results_append,
    run_full_experiment,
    print_standard_results,
    build_saveable_results
)


RESULTS_FILE = "results/experiments_results_facebook.json"


def convert_selected_circles_to_index_clusters(selected_circles, node_to_index):
    true_clusters = []

    for circle in selected_circles:
        cluster = [
            node_to_index[node]
            for node in circle["nodes"]
            if node in node_to_index
        ]
        true_clusters.append(cluster)

    return true_clusters


if __name__ == "__main__":

    # ============================================================
    # Parameters for Facebook circle sample
    # ============================================================

    ego_id = "1912"

    edges_file = f"data/facebook/{ego_id}.edges"
    circles_file = f"data/facebook/{ego_id}.circles"

    num_circles = 4
    nodes_per_circle = 25
    n = num_circles * nodes_per_circle

    p_delete = 0.15
    seed = 1
    pivot_seeds = [1, 2, 3, 4, 5]
    draw_graph = False

    # ============================================================
    # Load Facebook ego-network
    # ============================================================

    nodes, facebook_edges = load_facebook_ego_edges(edges_file)
    circles = load_facebook_circles(circles_file)

    # ============================================================
    # Sample nodes from Facebook circles
    # ============================================================

    sampled_nodes, selected_circles = sample_nodes_from_circles(
        circles=circles,
        num_circles=num_circles,
        nodes_per_circle=nodes_per_circle,
        seed=seed
    )

    # ============================================================
    # Build complete signed graph
    # ============================================================

    S, node_to_index, positive_count, negative_count = build_complete_signed_matrix_from_facebook_sample(
        sampled_nodes,
        facebook_edges
    )

    true_clusters = convert_selected_circles_to_index_clusters(
        selected_circles,
        node_to_index
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
        "graph_type": "facebook",
        "ego_id": ego_id,
        "num_nodes": n,
        "num_circles": num_circles,
        "nodes_per_circle": nodes_per_circle,
        "seed": seed,
        "pivot_seeds": pivot_seeds,
        "p_delete": p_delete,
        "num_edges_deleted": experiment_data["num_edges_deleted"],
        "selected_circles": selected_circles,
        "true_clusters": true_clusters,
        "positive_edges": positive_count,
        "negative_edges": negative_count,
        "signing_rule": {
            "existing_facebook_friendship": "+1",
            "missing_friendship_inside_sample": "-1",
            "deleted_edge": "0"
        }
    }

    # ============================================================
    # Print results
    # ============================================================

    print_standard_results(
        graph_type="facebook",
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
            G_complete=experiment_data["G"],
            true_clusters=true_clusters,
            pivot_clusters=experiment_data["pivot_clusters"],
            ilp_clusters=experiment_data["ilp_clusters"],
            G_new=experiment_data["G_new"],
            pivot_clusters_new=experiment_data["pivot_clusters_new"],
            ilp_clusters_new=experiment_data["ilp_clusters_new_with4"],
            pivots=experiment_data["pivots"],
            pivots_new=experiment_data["pivots_new"]
        )