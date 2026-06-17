import os
import json
import time
import copy
import numpy as np

from graph_generation import matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from edge_deletion import delete_edges
from ilp_solver import solve_ilp, find_ilp_clusters, same_clustering
from lp_formulations import solve_primal, solve_dual

from bad_triangles import (
    find_bad_triangles,
    count_bad_triangles,
    find_edge_disjoint_bad_triangles_min,
    make_edge_to_triangle_map,
    find_edge_disjoint_bad_triangles_max,
)


def json_converter(obj):
    """Convert NumPy objects to normal Python types for JSON saving."""
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, tuple):
        return list(obj)

    if isinstance(obj, set):
        return list(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def safe_ratio(numerator, denominator):
    """Return numerator / denominator, unless denominator is zero or missing."""
    if numerator is None or denominator is None:
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def remove_key(d, key):
    """Remove a key from a dictionary if it exists."""
    if isinstance(d, dict):
        d.pop(key, None)


def make_json_safe(obj):
    """Convert NumPy objects to normal Python objects."""
    return json.loads(json.dumps(obj, default=json_converter))


def expand_shared_results(data):
    """
    Convert saved JSON data to a normal list of experiments.

    Supports:
    - old format: [experiment, experiment, ...]
    - new format: {"shared_graph_params": ..., "experiments": [...]}
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "experiments" in data:
        shared_graph_params = data.get("shared_graph_params", {})
        experiments = []

        for exp in data["experiments"]:
            full_exp = make_json_safe(exp)

            full_graph_params = {}
            full_graph_params.update(shared_graph_params)
            full_graph_params.update(full_exp.get("graph_params", {}))

            full_exp["graph_params"] = full_graph_params
            experiments.append(full_exp)

        return experiments

    return []


def compress_shared_results(experiments):
    """
    Move graph parameters that are the same for every experiment to shared_graph_params.
    Only varying parameters, such as seed, stay inside each experiment.
    """
    experiments = make_json_safe(experiments)

    full_params_per_exp = []

    for exp in experiments:
        graph_params = exp.get("graph_params", {})

        remove_key(graph_params, "num_edges_deleted")
        remove_key(graph_params, "num_true_clusters")
        remove_key(graph_params, "true_cluster_sizes")

        full_params_per_exp.append(graph_params)

    all_keys = set()

    for graph_params in full_params_per_exp:
        all_keys.update(graph_params.keys())

    shared_graph_params = {}
    varying_keys = set()

    for key in all_keys:
        values = [graph_params.get(key) for graph_params in full_params_per_exp]

        if key == "seed":
            varying_keys.add(key)
        elif all(value == values[0] for value in values):
            shared_graph_params[key] = values[0]
        else:
            varying_keys.add(key)

    final_experiments = []

    for exp, full_graph_params in zip(experiments, full_params_per_exp):
        exp_without_full_params = make_json_safe(exp)
        exp_without_full_params.pop("graph_params", None)

        per_experiment_params = {}

        for key in sorted(varying_keys):
            if key in full_graph_params:
                per_experiment_params[key] = full_graph_params[key]

        final_exp = {
            "graph_params": per_experiment_params
        }

        for key, value in exp_without_full_params.items():
            final_exp[key] = value

        final_experiments.append(final_exp)

    return {
        "shared_graph_params": shared_graph_params,
        "experiments": final_experiments
    }


def save_results_append(filename, new_results):
    """
    Append new experiment results to a JSON file.

    The file is saved in this structure:
    {
        "shared_graph_params": {...},
        "experiments": [...]
    }
    """
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)

        all_results = expand_shared_results(data)
    else:
        all_results = []

    all_results.append(new_results)

    compact_results = compress_shared_results(all_results)

    with open(filename, "w") as f:
        json.dump(compact_results, f, indent=4, default=json_converter)



def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    print("\n--- " + title + " ---")


def get_edge_value(x_values, edge):
    """
    Get the value of an edge variable.

    This is useful because edges may be stored as (i, j) or (j, i).
    """
    u, v = edge

    if (u, v) in x_values:
        return x_values[(u, v)]

    if (v, u) in x_values:
        return x_values[(v, u)]

    return 0


def check_violated_bad_cycles(x_values, bad_cycles, tolerance=1e-6):
    """
    Check how many bad 4-cycle constraints are violated.

    The constraint is:
        x_negative <= sum of the other three cycle edges
    """
    violated_cycles = []

    for cycle, cycle_edges, signs, diagonal_1, diagonal_2 in bad_cycles:
        negative_edges = [
            edge for edge, sign in zip(cycle_edges, signs)
            if sign == -1
        ]

        if len(negative_edges) != 1:
            continue

        negative_edge = negative_edges[0]
        other_edges = [edge for edge in cycle_edges if edge != negative_edge]

        lhs = get_edge_value(x_values, negative_edge)
        rhs = sum(get_edge_value(x_values, edge) for edge in other_edges)

        if lhs > rhs + tolerance:
            violated_cycles.append((cycle, cycle_edges, signs, lhs, rhs))

    return violated_cycles


def run_pivot_multiple(S, pivot_seeds):
    """
    Run Pivot multiple times on the same graph.

    Keeps the best clusters/pivots in memory for drawing,
    but does not store all clusters in the JSON runs.
    """
    pivot_runs = []
    total_start = time.time()

    best_clusters = None
    best_pivots = None
    best_cost = None
    best_cluster_count = None

    for pivot_seed in pivot_seeds:
        run_start = time.time()

        clusters, pivots = run_pivot(S, pivot_seed)

        runtime = time.time() - run_start
        cost = calculate_clustering_cost(S, clusters)
        cluster_count = len(clusters)

        pivot_runs.append({
            "pivot_seed": pivot_seed,
            "cost": cost,
            "cluster_count": cluster_count,
            "runtime_seconds": round(runtime, 6)
        })

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cluster_count = cluster_count
            best_clusters = clusters
            best_pivots = pivots

    total_runtime = time.time() - total_start

    average_cost = sum(run["cost"] for run in pivot_runs) / len(pivot_runs)
    average_cluster_count = sum(run["cluster_count"] for run in pivot_runs) / len(pivot_runs)
    average_runtime = sum(run["runtime_seconds"] for run in pivot_runs) / len(pivot_runs)

    return {
        "runs": pivot_runs,
        "best_cost": best_cost,
        "best_cluster_count": best_cluster_count,
        "best_clusters": best_clusters,
        "best_pivots": best_pivots,
        "average_cost": average_cost,
        "average_cluster_count": average_cluster_count,
        "average_runtime_seconds": average_runtime,
        "total_runtime_seconds": round(total_runtime, 6)
    }

def run_full_experiment(S, p_delete, seed, pivot_seeds):
    """
    Run the full experiment pipeline on one signed graph.

    This includes:
    - edge deletion
    - Pivot multiple times
    - bad triangles
    - min/max edge-disjoint bad triangles
    - ILP
    - LP relaxation
    - bad-triangle LP bounds
    - LP/ILP with and without 4-cycle constraints after edge deletion
    """
    start_time = time.time()
    n = S.shape[0]

    G = matrix_to_graph(S)

    # ============================================================
    # Generate incomplete graph by deleting edges
    # ============================================================

    S_new, num_edges_deleted = delete_edges(S, p_delete, seed)
    G_new = matrix_to_graph(S_new)

    # ============================================================
    # Complete graph: Pivot
    # ============================================================

    pivot_results = run_pivot_multiple(S, pivot_seeds)
    pivot_clusters = pivot_results["best_clusters"]
    pivots = pivot_results["best_pivots"]

    # ============================================================
    # Complete graph: bad triangles
    # ============================================================

    all_bad_triangles = find_bad_triangles(S)
    edge_to_triangles = make_edge_to_triangle_map(all_bad_triangles)

    min_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_min(
        copy.deepcopy(edge_to_triangles)
    )
    min_num_bad_triangles = count_bad_triangles(min_disjoint_bad_triangles)

    max_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_max(
        copy.deepcopy(edge_to_triangles)
    )
    max_num_bad_triangles = count_bad_triangles(max_disjoint_bad_triangles)

    # ============================================================
    # Complete graph: ILP
    # ============================================================

    # ilp_cost, ilp_x_values, bad_cycles_ilp = solve_ilp(
    #     S,
    #     verbose=False,
    #     relax=False,
    #     add_four_cycles=False
    # )

    # ilp_clusters = find_ilp_clusters(ilp_x_values, n)
    # ilp_cluster_count = len(ilp_clusters)

    # ============================================================
    # Complete graph: LP relaxation
    # ============================================================

    lp_cost, lp_x_values, bad_cycles_lp = solve_ilp(
        S,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    # ============================================================
    # Complete graph: bad-triangle LP bounds
    # ============================================================

    primal_cost, primal_x_values = solve_primal(
        S,
        all_bad_triangles,
        verbose=False
    )

    dual_cost, dual_x_values = solve_dual(
        S,
        all_bad_triangles,
        verbose=False
    )

    # ============================================================
    # Incomplete graph: Pivot
    # ============================================================

    pivot_results_new = run_pivot_multiple(S_new, pivot_seeds)
    pivot_clusters_new = pivot_results_new["best_clusters"]
    pivots_new = pivot_results_new["best_pivots"]

    # ============================================================
    # Incomplete graph: bad triangles
    # ============================================================

    all_bad_triangles_new = find_bad_triangles(S_new)
    edge_to_triangles_new = make_edge_to_triangle_map(all_bad_triangles_new)

    min_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_min(
        copy.deepcopy(edge_to_triangles_new)
    )
    min_num_bad_triangles_new = count_bad_triangles(min_disjoint_bad_triangles_new)

    max_disjoint_bad_triangles_new = find_edge_disjoint_bad_triangles_max(
        copy.deepcopy(edge_to_triangles_new)
    )
    max_num_bad_triangles_new = count_bad_triangles(max_disjoint_bad_triangles_new)

    # ============================================================
    # Incomplete graph: ILP without bad 4-cycle constraints
    # ============================================================

    # ilp_cost_new_no4, ilp_x_values_new_no4, bad_cycles_ilp_new_no4 = solve_ilp(
    #     S_new,
    #     verbose=False,
    #     relax=False,
    #     add_four_cycles=False
    # )

    # ilp_clusters_new_no4 = find_ilp_clusters(ilp_x_values_new_no4, n)
    # ilp_cluster_count_new_no4 = len(ilp_clusters_new_no4)

    # ============================================================
    # Incomplete graph: ILP with bad 4-cycle constraints
    # ============================================================

    # ilp_cost_new_with4, ilp_x_values_new_with4, bad_cycles_ilp_new_with4 = solve_ilp(
    #     S_new,
    #     verbose=False,
    #     relax=False,
    #     add_four_cycles=True
    # )

    # ilp_clusters_new_with4 = find_ilp_clusters(ilp_x_values_new_with4, n)
    # ilp_cluster_count_new_with4 = len(ilp_clusters_new_with4)

    # ============================================================
    # Incomplete graph: LP relaxation without bad 4-cycle constraints
    # ============================================================

    lp_cost_new_no4, lp_x_values_new_no4, bad_cycles_lp_new_no4 = solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    # ============================================================
    # Incomplete graph: LP relaxation with bad 4-cycle constraints
    # ============================================================

    lp_cost_new_with4, lp_x_values_new_with4, bad_cycles_lp_new_with4 = solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=True
    )

    # ============================================================
    # Check violated bad 4-cycle constraints
    # ============================================================

    # violated_cycles_ilp_new = check_violated_bad_cycles(
    #     ilp_x_values_new_no4,
    #     bad_cycles_ilp_new_with4
    # )

    violated_cycles_lp_new = check_violated_bad_cycles(
        lp_x_values_new_no4,
        bad_cycles_lp_new_with4
    )
    # =========================================================================================
    # Incomplete graph: Check whether with and without 4 cycle clustering is the same
    # =========================================================================================
    # same_clustering_4_cycle = same_clustering(ilp_clusters_new_no4, ilp_clusters_new_with4)
    # ============================================================
    # Incomplete graph: bad-triangle LP bounds
    # ============================================================

    primal_cost_new, primal_x_values_new = solve_primal(
        S_new,
        all_bad_triangles_new,
        verbose=False
    )

    dual_cost_new, dual_x_values_new = solve_dual(
        S_new,
        all_bad_triangles_new,
        verbose=False
    )

    total_runtime = time.time() - start_time

    experiment_data = {
        # Useful for drawing
        "S": S,
        "G": G,
        "S_new": S_new,
        "G_new": G_new,

        "pivot_clusters": pivot_clusters,
        "pivots": pivots,
        # "ilp_clusters": ilp_clusters,

        "pivot_clusters_new": pivot_clusters_new,
        "pivots_new": pivots_new,
        # "ilp_clusters_new_with4": ilp_clusters_new_with4,

        # General
        "num_edges_deleted": num_edges_deleted,
        "total_runtime": total_runtime,

        # Complete graph
        "pivot_results": pivot_results,
        "all_bad_triangles": all_bad_triangles,
        "min_num_bad_triangles": min_num_bad_triangles,
        "max_num_bad_triangles": max_num_bad_triangles,

        # "ilp_cost": ilp_cost,
        # "ilp_cluster_count": ilp_cluster_count,
        # "bad_cycles_ilp": bad_cycles_ilp,

        "lp_cost": lp_cost,
        "bad_cycles_lp": bad_cycles_lp,

        "primal_cost": primal_cost,
        "dual_cost": dual_cost,

        # Edge-deleted graph
        "pivot_results_new": pivot_results_new,
        "all_bad_triangles_new": all_bad_triangles_new,
        "min_num_bad_triangles_new": min_num_bad_triangles_new,
        "max_num_bad_triangles_new": max_num_bad_triangles_new,

        # "ilp_cost_new_no4": ilp_cost_new_no4,
        # "ilp_cluster_count_new_no4": ilp_cluster_count_new_no4,
        # "bad_cycles_ilp_new_no4": bad_cycles_ilp_new_no4,

        # "ilp_cost_new_with4": ilp_cost_new_with4,
        # "ilp_cluster_count_new_with4": ilp_cluster_count_new_with4,
        # "bad_cycles_ilp_new_with4": bad_cycles_ilp_new_with4,

        "lp_cost_new_no4": lp_cost_new_no4,
        "bad_cycles_lp_new_no4": bad_cycles_lp_new_no4,

        "lp_cost_new_with4": lp_cost_new_with4,
        "bad_cycles_lp_new_with4": bad_cycles_lp_new_with4,

        # "violated_cycles_ilp_new": violated_cycles_ilp_new,
        "violated_cycles_lp_new": violated_cycles_lp_new,
        # "same_clustering_4_cycle": same_clustering_4_cycle,

        "primal_cost_new": primal_cost_new,
        "dual_cost_new": dual_cost_new,
    }

    return experiment_data


def build_saveable_results(graph_params, experiment_data):
    """
    Build the dictionary that will be saved to JSON.

    This matches the compact JSON structure:
    - complete_graph
    - edge_deleted_graph
    - approximations
    - runtime_seconds
    """

    cleaned_graph_params = dict(graph_params)

    remove_key(cleaned_graph_params, "num_edges_deleted")
    remove_key(cleaned_graph_params, "num_true_clusters")
    remove_key(cleaned_graph_params, "true_cluster_sizes")

    results = {
        "graph_params": cleaned_graph_params,

        "complete_graph": {
            "pivot": {
                "best_cost": experiment_data["pivot_results"]["best_cost"],
                "average_cost": experiment_data["pivot_results"]["average_cost"]
            },

            "bad_triangles": {
                "total_count": len(experiment_data["all_bad_triangles"]),
                "min_edge_disjoint_count": experiment_data["min_num_bad_triangles"],
                "max_edge_disjoint_count": experiment_data["max_num_bad_triangles"]
            },

            # "ilp": {
            #     "cost": experiment_data["ilp_cost"]
            # },

            "lp_relaxation": {
                "cost": experiment_data["lp_cost"]
            },

            "bad_triangle_lp_bounds": {
                "primal_cost": experiment_data["primal_cost"],
                "dual_cost": experiment_data["dual_cost"]
            }
        },

        "edge_deleted_graph": {
            "num_edges_deleted": experiment_data["num_edges_deleted"],

            "pivot": {
                "best_cost": experiment_data["pivot_results_new"]["best_cost"],
                "average_cost": experiment_data["pivot_results_new"]["average_cost"]
            },

            "bad_triangles": {
                "total_count": len(experiment_data["all_bad_triangles_new"]),
                "min_edge_disjoint_count": experiment_data["min_num_bad_triangles_new"],
                "max_edge_disjoint_count": experiment_data["max_num_bad_triangles_new"]
            },

            # "ilp": {
            #     "without_4_cycles": {
            #         "cost": experiment_data["ilp_cost_new_no4"]
            #     },
            #     "with_4_cycles": {
            #         "cost": experiment_data["ilp_cost_new_with4"],
            #         "bad_4_cycles_count": len(experiment_data["bad_cycles_ilp_new_with4"])
            #     },
            #     "same_clustering_4_cycle": experiment_data["same_clustering_4_cycle"]
            # },

            "lp_relaxation": {
                "without_4_cycles": {
                    "cost": experiment_data["lp_cost_new_no4"]
                },
                "with_4_cycles": {
                    "cost": experiment_data["lp_cost_new_with4"]
                }
            },

            "bad_triangle_lp_bounds": {
                "primal_cost": experiment_data["primal_cost_new"],
                "dual_cost": experiment_data["dual_cost_new"]
            }
        },

        # "approximations": {
        #     "complete_graph": {
        #         "best_pivot_approximation": safe_ratio(
        #             experiment_data["pivot_results"]["best_cost"],
        #             experiment_data["ilp_cost"]
        #         ),
        #         "average_pivot_approximation": safe_ratio(
        #             experiment_data["pivot_results"]["average_cost"],
        #             experiment_data["ilp_cost"]
        #         ),
        #         "lp_relaxation_ratio": safe_ratio(
        #             experiment_data["lp_cost"],
        #             experiment_data["ilp_cost"]
        #         ),
        #         "bad_triangle_primal_ratio": safe_ratio(
        #             experiment_data["primal_cost"],
        #             experiment_data["ilp_cost"]
        #         ),
        #         "bad_triangle_dual_ratio": safe_ratio(
        #             experiment_data["dual_cost"],
        #             experiment_data["ilp_cost"]
        #         ),
        #         "min_disjoint_bad_triangle_ratio": safe_ratio(
        #             experiment_data["min_num_bad_triangles"],
        #             experiment_data["ilp_cost"]
        #         ),
        #         "max_disjoint_bad_triangle_ratio": safe_ratio(
        #             experiment_data["max_num_bad_triangles"],
        #             experiment_data["ilp_cost"]
        #         )
        #     },

        #     "edge_deleted_graph": {
        #         "best_pivot_approximation_with_4_cycles": safe_ratio(
        #             experiment_data["pivot_results_new"]["best_cost"],
        #             experiment_data["ilp_cost_new_with4"]
        #         ),
        #         "average_pivot_approximation_with_4_cycles": safe_ratio(
        #             experiment_data["pivot_results_new"]["average_cost"],
        #             experiment_data["ilp_cost_new_with4"]
        #         ),
        #         "lp_relaxation_ratio_with_4_cycles": safe_ratio(
        #             experiment_data["lp_cost_new_with4"],
        #             experiment_data["ilp_cost_new_with4"]
        #         ),
        #         "bad_triangle_primal_ratio": safe_ratio(
        #             experiment_data["primal_cost_new"],
        #             experiment_data["ilp_cost_new_with4"]
        #         ),
        #         "bad_triangle_dual_ratio": safe_ratio(
        #             experiment_data["dual_cost_new"],
        #             experiment_data["ilp_cost_new_with4"]
        #         ),
        #         "min_disjoint_bad_triangle_ratio": safe_ratio(
        #             experiment_data["min_num_bad_triangles_new"],
        #             experiment_data["ilp_cost_new_with4"]
        #         ),
        #         "max_disjoint_bad_triangle_ratio": safe_ratio(
        #             experiment_data["max_num_bad_triangles_new"],
        #             experiment_data["ilp_cost_new_with4"]
        #         )
        #     }
        # },

        "runtime_seconds": experiment_data["total_runtime"]
    }

    return results



def print_ratio(label, value):
    """Print a ratio cleanly."""
    if value is None:
        print(f"{label}: None")
    else:
        print(f"{label}: {value:.4f}")


def print_standard_results(graph_type, graph_params, experiment_data):
    """
    Print standard experiment results.

    The print structure matches the saved JSON:
    - complete_graph
    - edge_deleted_graph
    - approximations
    - runtime_seconds
    """

    print_section(f"{graph_type.upper()} Graph Parameters")

    for key, value in graph_params.items():
        if key == "selected_circles":
            print("selected_circles:", [circle["name"] for circle in value])
        elif key == "true_clusters":
            print("true_clusters:", f"{len(value)} clusters")
        else:
            print(f"{key}:", value)

    print_section(f"Complete {graph_type} Graph")

    print_subsection("Pivot")
    print("Best Pivot cost:", experiment_data["pivot_results"]["best_cost"])
    print("Average Pivot cost:", experiment_data["pivot_results"]["average_cost"])

    print_subsection("Bad triangles")
    print("Total number of bad triangles:", len(experiment_data["all_bad_triangles"]))
    print("Minimum amount of disjoint bad triangles:", experiment_data["min_num_bad_triangles"])
    print("Maximum amount of disjoint bad triangles:", experiment_data["max_num_bad_triangles"])

    print_subsection("ILP")
    # print("ILP optimal cost:", experiment_data["ilp_cost"])

    print_subsection("LP relaxation")
    print("LP relaxation cost:", experiment_data["lp_cost"])

    print_subsection("Bad-triangle LP bounds")
    print("LP-primal optimal cost:", experiment_data["primal_cost"])
    print("LP-dual optimal cost:", experiment_data["dual_cost"])

    print_section(f"Edge-Deleted {graph_type} Graph")
    print("Number of edges deleted:", experiment_data["num_edges_deleted"])

    print_subsection("Pivot")
    print("Best Pivot cost:", experiment_data["pivot_results_new"]["best_cost"])
    print("Average Pivot cost:", experiment_data["pivot_results_new"]["average_cost"])

    print_subsection("Bad triangles")
    print("Total number of bad triangles:", len(experiment_data["all_bad_triangles_new"]))
    print("Minimum amount of disjoint bad triangles:", experiment_data["min_num_bad_triangles_new"])
    print("Maximum amount of disjoint bad triangles:", experiment_data["max_num_bad_triangles_new"])

    # print_subsection("ILP")
    # print("ILP cost without 4-cycles:", experiment_data["ilp_cost_new_no4"])
    # print("ILP cost with 4-cycles:", experiment_data["ilp_cost_new_with4"])
    # print("Bad 4-cycles detected ILP:", len(experiment_data["bad_cycles_ilp_new_with4"]))
    # print(
    #     "With and without 4-cycle same clusters:",
    #     experiment_data["same_clustering_4_cycle"]
    # )

    print_subsection("LP relaxation")
    print("LP relaxation cost without 4-cycles:", experiment_data["lp_cost_new_no4"])
    print("LP relaxation cost with 4-cycles:", experiment_data["lp_cost_new_with4"])

    print_subsection("Bad-triangle LP bounds")
    print("LP-primal optimal cost:", experiment_data["primal_cost_new"])
    print("LP-dual optimal cost:", experiment_data["dual_cost_new"])

    print_section("Approximations")

    print_subsection("Complete graph")
    # print_ratio(
    #     "Best Pivot approximation",
    #     safe_ratio(experiment_data["pivot_results"]["best_cost"], experiment_data["ilp_cost"])
    # )
    # print_ratio(
    #     "Average Pivot approximation",
    #     safe_ratio(experiment_data["pivot_results"]["average_cost"], experiment_data["ilp_cost"])
    # )
    # print_ratio(
    #     "LP relaxation ratio",
    #     safe_ratio(experiment_data["lp_cost"], experiment_data["ilp_cost"])
    # )
    # print_ratio(
    #     "Bad-triangle primal ratio",
    #     safe_ratio(experiment_data["primal_cost"], experiment_data["ilp_cost"])
    # )
    # print_ratio(
    #     "Bad-triangle dual ratio",
    #     safe_ratio(experiment_data["dual_cost"], experiment_data["ilp_cost"])
    # )
    # print_ratio(
    #     "Min disjoint bad triangle ratio",
    #     safe_ratio(experiment_data["min_num_bad_triangles"], experiment_data["ilp_cost"])
    # )
    # print_ratio(
    #     "Max disjoint bad triangle ratio",
    #     safe_ratio(experiment_data["max_num_bad_triangles"], experiment_data["ilp_cost"])
    # )

    # print_subsection("Edge-deleted graph")
    # print_ratio(
    #     "Best Pivot approximation with 4-cycles",
    #     safe_ratio(experiment_data["pivot_results_new"]["best_cost"], experiment_data["ilp_cost_new_with4"])
    # )
    # print_ratio(
    #     "Average Pivot approximation with 4-cycles",
    #     safe_ratio(experiment_data["pivot_results_new"]["average_cost"], experiment_data["ilp_cost_new_with4"])
    # )
    # print_ratio(
    #     "LP relaxation ratio with 4-cycles",
    #     safe_ratio(experiment_data["lp_cost_new_with4"], experiment_data["ilp_cost_new_with4"])
    # )
    # print_ratio(
    #     "Bad-triangle primal ratio",
    #     safe_ratio(experiment_data["primal_cost_new"], experiment_data["ilp_cost_new_with4"])
    # )
    # print_ratio(
    #     "Bad-triangle dual ratio",
    #     safe_ratio(experiment_data["dual_cost_new"], experiment_data["ilp_cost_new_with4"])
    # )
    # print_ratio(
    #     "Min disjoint bad triangle ratio",
    #     safe_ratio(experiment_data["min_num_bad_triangles_new"], experiment_data["ilp_cost_new_with4"])
    # )
    # print_ratio(
    #     "Max disjoint bad triangle ratio",
    #     safe_ratio(experiment_data["max_num_bad_triangles_new"], experiment_data["ilp_cost_new_with4"])
    # )

    print_section("Runtime")
    print("Total runtime:", round(experiment_data["total_runtime"], 2), "seconds")

