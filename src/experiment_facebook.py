from pathlib import Path

from facebook_sampling import (
load_facebook_ego_edges,
load_facebook_circles,
build_complete_signed_matrix_from_facebook_sample,
)

from draw_graphs import draw_graphs

from experiment_helpers import (
save_results_append,
run_full_experiment,
print_standard_results,
build_saveable_results,
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
    # Parameters for one Facebook ego-network experiment
    # ============================================================

    ego_id = "3980"

    p_delete = 0.15
    seed = 1
    pivot_seeds = list(range(1, 11))

    draw_graph = True

    edges_file = Path(f"data/facebook/{ego_id}.edges")
    circles_file = Path(f"data/facebook/{ego_id}.circles")

    results_dir = Path("results/experiments_results_facebook/full")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"fb_ego{ego_id}_full.json"

    if results_file.exists():
        results_file.unlink()

    print("\n" + "=" * 70, flush=True)
    print("RUNNING ONE FACEBOOK EGO EXPERIMENT WITH ILP", flush=True)
    print(f"ego_id = {ego_id}", flush=True)
    print(f"p_delete = {p_delete}", flush=True)
    print(f"seed = {seed}", flush=True)
    print(f"edges_file = {edges_file}", flush=True)
    print(f"circles_file = {circles_file}", flush=True)
    print(f"Saving to: {results_file}", flush=True)
    print("=" * 70, flush=True)

    # ============================================================
    # Load full ego-network
    # ============================================================

    edge_nodes, facebook_edges = load_facebook_ego_edges(str(edges_file))
    circles = load_facebook_circles(str(circles_file))

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

    S, node_to_index, positive_count, negative_count = (
        build_complete_signed_matrix_from_facebook_sample(
            all_nodes,
            facebook_edges,
        )
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
    # Save results
    # ============================================================

    results = build_saveable_results(
        graph_params=graph_params,
        experiment_data=experiment_data,
    )

    save_results_append(str(results_file), results)

    print(f"Saved result to {results_file}", flush=True)

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

