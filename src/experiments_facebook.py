import time

from facebook_sampling import (
    load_facebook_ego_edges,
    load_facebook_circles,
    sample_nodes_from_circles,
    build_complete_signed_matrix_from_facebook_sample
)
from graph_generation import matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from edge_deletion import delete_edges
from draw_graphs_clique import draw_clique_graphs
from ilp_solver import solve_ilp, find_ilp_clusters
from experiments import (
    save_results_append,
    print_section,
    check_violated_bad_cycles
)


RESULTS_FILE = "results/experiments_facebook_results.json"

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

    start_time = time.time()

    # ============================================================
    # Parameters for Facebook circle sample
    # ============================================================

    ego_id = "1912"

    edges_file = f"data/facebook/{ego_id}.edges"
    circles_file = f"data/facebook/{ego_id}.circles"

    num_circles = 10
    nodes_per_circle = 10
    n = num_circles * nodes_per_circle

    p_delete = 0.25
    seed = 1

    # Zet deze op False als n=100 te lang duurt
    run_ilp = True

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

    G = matrix_to_graph(S)

    # ============================================================
    # Complete Facebook-based graph: Pivot
    # ============================================================

    pivot_start = time.time()

    pivot_clusters, pivots = run_pivot(S, seed)
    pivot_cluster_count = len(pivot_clusters)
    pivot_cost = calculate_clustering_cost(S, pivot_clusters)

    pivot_runtime = time.time() - pivot_start

    # ============================================================
    # Complete Facebook-based graph: LP relaxation
    # ============================================================

    lp_start = time.time()

    lp_cost, lp_x_values, bad_cycles_lp = solve_ilp(
        S,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    lp_runtime = time.time() - lp_start

    # ============================================================
    # Complete Facebook-based graph: ILP
    # ============================================================

    ilp_cost = None
    ilp_clusters = None
    ilp_runtime = None

    if run_ilp:
        ilp_start = time.time()

        ilp_cost, ilp_x_values, bad_cycles_ilp = solve_ilp(
            S,
            verbose=False,
            relax=False,
            add_four_cycles=False
        )
        ilp_clusters = find_ilp_clusters(ilp_x_values, n)
        ilp_runtime = time.time() - ilp_start

    # ============================================================
    # Generate incomplete graph by deleting edges
    # ============================================================

    S_new, num_edges_deleted = delete_edges(S, p_delete, seed)
    G_new = matrix_to_graph(S_new)

    # ============================================================
    # Edge-deleted Facebook-based graph: Pivot
    # ============================================================

    pivot_new_start = time.time()

    pivot_clusters_new, pivots_new = run_pivot(S_new, seed)
    pivot_cluster_count_new = len(pivot_clusters_new)
    pivot_cost_new = calculate_clustering_cost(S_new, pivot_clusters_new)

    pivot_new_runtime = time.time() - pivot_new_start

    # ============================================================
    # Edge-deleted Facebook-based graph: LP relaxation without 4-cycles
    # ============================================================

    lp_new_no4_start = time.time()

    lp_cost_new_no4, lp_x_values_new_no4, bad_cycles_lp_new_no4 = solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    lp_new_no4_runtime = time.time() - lp_new_no4_start

    # ============================================================
    # Edge-deleted Facebook-based graph: LP relaxation with 4-cycles
    # ============================================================

    lp_new_with4_start = time.time()

    lp_cost_new_with4, lp_x_values_new_with4, bad_cycles_lp_new_with4 = solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=True
    )

    lp_new_with4_runtime = time.time() - lp_new_with4_start

    violated_cycles_lp_new = check_violated_bad_cycles(
        lp_x_values_new_no4,
        bad_cycles_lp_new_with4
    )

    # ============================================================
    # Edge-deleted Facebook-based graph: ILP without and with 4-cycles
    # ============================================================

    ilp_cost_new_no4 = None
    ilp_runtime_new_no4 = None
    ilp_cost_new_with4 = None
    ilp_runtime_new_with4 = None
    ilp_clusters_new_with4 = None

    if run_ilp:
        ilp_new_no4_start = time.time()

        ilp_cost_new_no4, ilp_x_values_new_no4, bad_cycles_ilp_new_no4 = solve_ilp(
            S_new,
            verbose=False,
            relax=False,
            add_four_cycles=False
        )

        ilp_runtime_new_no4 = time.time() - ilp_new_no4_start

        ilp_new_with4_start = time.time()

        ilp_cost_new_with4, ilp_x_values_new_with4, bad_cycles_ilp_new_with4 = solve_ilp(
            S_new,
            verbose=False,
            relax=False,
            add_four_cycles=True
        )
        ilp_clusters_new_with4 = find_ilp_clusters(ilp_x_values_new_with4, n)
        ilp_runtime_new_with4 = time.time() - ilp_new_with4_start

    total_runtime = time.time() - start_time

    # ============================================================
    # Print results
    # ============================================================

    print_section("Facebook Circle-Based Signed Graph")
    print("ego_id:", ego_id)
    print("Number of sampled nodes:", n)
    print("Number of circles:", num_circles)
    print("Nodes per circle:", nodes_per_circle)
    print("p_delete:", p_delete)
    print("seed:", seed)
    print("run_ilp:", run_ilp)
    print("total runtime:", round(total_runtime, 2), "seconds")

    print_section("Sampled Facebook Circles")
    for circle in selected_circles:
        print(circle["name"], "size:", len(circle["nodes"]))

    print_section("Complete Facebook-Based Signed Graph")
    print("Positive edges:", positive_count)
    print("Negative edges:", negative_count)
    print("Pivot cost:", pivot_cost)
    print("Pivot cluster count:", pivot_cluster_count)
    print("Pivot runtime:", round(pivot_runtime, 2), "seconds")
    print("LP relaxation cost:", lp_cost)
    print("LP runtime:", round(lp_runtime, 2), "seconds")

    if run_ilp:
        print("ILP cost:", ilp_cost)
        print("ILP runtime:", round(ilp_runtime, 2), "seconds")

    print_section("Edge-Deleted Facebook-Based Signed Graph")
    print("Number of edges deleted:", num_edges_deleted)
    print("Pivot cost:", pivot_cost_new)
    print("Pivot cluster count:", pivot_cluster_count_new)
    print("Pivot runtime:", round(pivot_new_runtime, 2), "seconds")
    print("LP relaxation cost without 4-cycles:", lp_cost_new_no4)
    print("LP no-4 runtime:", round(lp_new_no4_runtime, 2), "seconds")
    print("LP relaxation cost with 4-cycles:", lp_cost_new_with4)
    print("LP with-4 runtime:", round(lp_new_with4_runtime, 2), "seconds")
    print("Bad 4-cycles detected LP:", len(bad_cycles_lp_new_with4))
    print("Violated bad 4-cycles in LP no-4 solution:", len(violated_cycles_lp_new))

    if run_ilp:
        print("ILP cost without 4-cycles:", ilp_cost_new_no4)
        print("ILP no-4 runtime:", round(ilp_runtime_new_no4, 2), "seconds")
        print("ILP cost with 4-cycles:", ilp_cost_new_with4)
        print("ILP with-4 runtime:", round(ilp_runtime_new_with4, 2), "seconds")

    # ============================================================
    # Save results
    # ============================================================

    results = {
        "graph_params": {
            "graph_type": "facebook_circle_based_complete_signed",
            "ego_id": ego_id,
            "num_nodes": n,
            "num_circles": num_circles,
            "nodes_per_circle": nodes_per_circle,
            "p_delete": p_delete,
            "seed": seed,
            "num_edges_deleted": num_edges_deleted,
            "run_ilp": run_ilp,
            "runtime_seconds": round(total_runtime, 4),
            "signing_rule": {
                "existing_facebook_friendship": "+1",
                "missing_friendship_inside_sample": "-1",
                "deleted_edge": "0"
            }
        },

        "facebook_sample": {
            "selected_circles": selected_circles,
            "positive_edges": positive_count,
            "negative_edges": negative_count
        },

        "complete_graph": {
            "pivot": {
                "cost": pivot_cost,
                "cluster_count": pivot_cluster_count,
                "runtime_seconds": round(pivot_runtime, 4),
                "pivots": pivots
            },
            "lp_relaxation": {
                "cost": lp_cost,
                "num_variables": len(lp_x_values),
                "bad_4_cycles_count": len(bad_cycles_lp),
                "runtime_seconds": round(lp_runtime, 4)
            },
            "ilp": {
                "cost": ilp_cost,
                "runtime_seconds": None if ilp_runtime is None else round(ilp_runtime, 4)
            }
        },

        "new_graph": {
            "pivot": {
                "cost": pivot_cost_new,
                "cluster_count": pivot_cluster_count_new,
                "runtime_seconds": round(pivot_new_runtime, 4),
                "pivots": pivots_new
            },
            "lp_relaxation": {
                "without_4_cycles": {
                    "cost": lp_cost_new_no4,
                    "num_variables": len(lp_x_values_new_no4),
                    "bad_4_cycles_count": len(bad_cycles_lp_new_no4),
                    "runtime_seconds": round(lp_new_no4_runtime, 4),
                    "violated_bad_4_cycles_count": len(violated_cycles_lp_new)
                },
                "with_4_cycles": {
                    "cost": lp_cost_new_with4,
                    "num_variables": len(lp_x_values_new_with4),
                    "bad_4_cycles_count": len(bad_cycles_lp_new_with4),
                    "runtime_seconds": round(lp_new_with4_runtime, 4)
                }
            },
            "ilp": {
                "without_4_cycles": {
                    "cost": ilp_cost_new_no4,
                    "runtime_seconds": None if ilp_runtime_new_no4 is None else round(ilp_runtime_new_no4, 4)
                },
                "with_4_cycles": {
                    "cost": ilp_cost_new_with4,
                    "runtime_seconds": None if ilp_runtime_new_with4 is None else round(ilp_runtime_new_with4, 4)
                }
            }
        }
    }

    save_results_append(RESULTS_FILE, results)

    true_clusters = convert_selected_circles_to_index_clusters(
        selected_circles,
        node_to_index
    )

    if run_ilp:
        draw_clique_graphs(
            G,
            true_clusters,
            pivot_clusters,
            ilp_clusters,
            G_new,
            pivot_clusters_new,
            ilp_clusters_new_with4,
            pivots=pivots,
            pivots_new=pivots_new
        )