import os
import json
import time
import copy
import numpy as np

from graph_generation import matrix_to_graph
from pivot import run_pivot
from cost import calculate_clustering_cost
from edge_deletion import delete_edges
from min_max import min_max_cc, vertex_disagreement, max_disagreement
from min_max_lp import (
    MinMaxLP,
    cluster as min_max_lp_cluster,
    DegreeDist,
    LocalObj,
)

# ============================================================
# Solver imports
# ============================================================

# ACTUAL FORMULATION YOU WANT TO USE:
# all-pairs formulation from all_pairs_solver.py
# relax=False -> actual all-pairs ILP
# relax=True  -> LP relaxation of the actual all-pairs ILP
from all_pairs_solver import solve_all_pairs

# OLD FORMULATION:
# observed-edge formulation from ilp_solver.py
# Only used if COMPUTE_OBSERVED_EDGE_LP or COMPUTE_OBSERVED_EDGE_ILP is True.
from ilp_solver import solve_ilp, find_ilp_clusters

# Bad-triangle LP bounds, optional.
from lp_formulations import solve_primal, solve_dual

from bad_triangles import (
    find_bad_triangles,
    count_bad_triangles,
    find_edge_disjoint_bad_triangles_min,
    make_edge_to_triangle_map,
    find_edge_disjoint_bad_triangles_max,
)


# ============================================================
# Solver wrappers
# ============================================================

def solve_actual_all_pairs_formulation(S, relax=False):
    """
    Main solver for the formulation you actually want.

    relax=False -> actual all-pairs ILP
    relax=True  -> LP relaxation of the actual all-pairs ILP
    """
    cost, x_values, solve_info = solve_all_pairs(
        S,
        verbose=False,
        relax=relax,
        return_x_values=True,
    )

    return cost, x_values, solve_info


def solve_observed_edge_formulation(S, relax=False, add_four_cycles=False):
    """
    Old observed-edge solver.

    relax=False -> observed-edge ILP
    relax=True  -> observed-edge LP relaxation
    """
    cost, x_values, bad_cycles = solve_ilp(
        S,
        verbose=False,
        relax=relax,
        add_four_cycles=add_four_cycles,
    )

    return cost, x_values, bad_cycles


# ============================================================
# JSON helpers
# ============================================================

def json_converter(obj):
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
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def save_results_append(filename, new_results):
    """Append new experiment results to a JSON file."""
    directory = os.path.dirname(filename)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    if isinstance(all_results, dict) and "experiments" in all_results:
        all_results = all_results["experiments"]

    all_results.append(new_results)

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4, default=json_converter)


# ============================================================
# Print helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    print("\n--- " + title + " ---")


def print_if_not_none(label, value):
    if value is not None:
        print(label + ":", value)


# ============================================================
# Pivot helpers
# ============================================================

def run_pivot_multiple(S, pivot_seeds):
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
            "runtime_seconds": round(runtime, 6),
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
        "total_runtime_seconds": round(total_runtime, 6),
    }


def empty_pivot_results():
    return {
        "runs": [],
        "best_cost": None,
        "best_cluster_count": None,
        "best_clusters": None,
        "best_pivots": None,
        "average_cost": None,
        "average_cluster_count": None,
        "average_runtime_seconds": None,
        "total_runtime_seconds": 0,
    }


# ============================================================
# Bad triangle helpers
# ============================================================

def compute_bad_triangle_data(S, compute_bad_triangles, compute_disjoint_bad_triangles):
    """
    Returns bad triangles and optionally min/max edge-disjoint counts.
    If compute_bad_triangles=False, this does not call find_bad_triangles.
    """
    if not compute_bad_triangles:
        return {
            "all_bad_triangles": [],
            "min_num_bad_triangles": None,
            "max_num_bad_triangles": None,
        }

    all_bad_triangles = find_bad_triangles(S)

    if compute_disjoint_bad_triangles:
        edge_to_triangles = make_edge_to_triangle_map(all_bad_triangles)

        min_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_min(
            copy.deepcopy(edge_to_triangles)
        )
        min_num_bad_triangles = count_bad_triangles(min_disjoint_bad_triangles)

        max_disjoint_bad_triangles = find_edge_disjoint_bad_triangles_max(
            copy.deepcopy(edge_to_triangles)
        )
        max_num_bad_triangles = count_bad_triangles(max_disjoint_bad_triangles)
    else:
        min_num_bad_triangles = None
        max_num_bad_triangles = None

    return {
        "all_bad_triangles": all_bad_triangles,
        "min_num_bad_triangles": min_num_bad_triangles,
        "max_num_bad_triangles": max_num_bad_triangles,
    }


def compute_bad_triangle_lp_bounds(
    S,
    all_bad_triangles,
    compute_primal_bound,
    compute_dual_bound,
):
    if compute_primal_bound:
        primal_cost, primal_x_values = solve_primal(
            S,
            all_bad_triangles,
            verbose=False,
        )
    else:
        primal_cost = None
        primal_x_values = None

    if compute_dual_bound:
        dual_cost, dual_x_values = solve_dual(
            S,
            all_bad_triangles,
            verbose=False,
        )
    else:
        dual_cost = None
        dual_x_values = None

    return primal_cost, primal_x_values, dual_cost, dual_x_values


# ============================================================
# Min-max helpers
# ============================================================

def compute_min_max_cc_data(S, compute_min_max, param_1=2, param_2=2):
    """
    Compute the min_max.py heuristic/result that you were already printing.
    Now it is returned in a clean dictionary so it can be printed and saved.
    """
    if not compute_min_max:
        return {
            "clustering": None,
            "cluster_count": None,
            "max_disagreement": None,
            "runtime_seconds": None,
        }

    start = time.time()
    clustering = min_max_cc(S, param_1, param_2)
    runtime = time.time() - start

    return {
        "clustering": clustering,
        "cluster_count": len(clustering) if clustering is not None else None,
        "max_disagreement": max_disagreement(clustering, S),
        "runtime_seconds": round(runtime, 6),
    }


def compute_min_max_lp_data(
    S,
    compute_min_max_lp,
    r=0.1,
    r2=0.5,
    method=0,
    norm=np.inf,
):
    """
    Run the adapted min_max_lp.py LP + rounding code directly on your
    signed matrix S.

    Matrix convention used by min_max_lp.py now matches the rest of
    your project:
        1  = positive edge
       -1  = negative edge
        0  = deleted / unobserved edge

    The LP/local objective ignores deleted/unobserved edges instead of
    treating them as negative edges.
    """
    if not compute_min_max_lp:
        return {
            "lp_cost": None,
            "rounding_cost": None,
            "max_disagreement_vertex": None,
            "clustering": None,
            "cluster_count": None,
            "lp_runtime_seconds": None,
            "rounding_runtime_seconds": None,
            "total_runtime_seconds": None,
        }

    total_start = time.time()

    lp_cost, distances, L_t_vals, neighborsR, neighborsR2, lp_runtime = MinMaxLP(
        S,
        r,
        r2,
        method,
    )

    clustering, rounding_runtime = min_max_lp_cluster(
        distances,
        L_t_vals,
        neighborsR,
        neighborsR2,
        r,
        r2,
    )

    pos_degrees = DegreeDist(S)

    disagreement_vector, rounding_cost, max_disagreement_vertex = LocalObj(
        S,
        clustering,
        pos_degrees,
        norm,
    )

    total_runtime = time.time() - total_start

    return {
        "lp_cost": lp_cost,
        "rounding_cost": rounding_cost,
        "max_disagreement_vertex": max_disagreement_vertex,
        "disagreement_vector": disagreement_vector,
        "clustering": clustering,
        "cluster_count": len(clustering),
        "r": r,
        "r2": r2,
        "method": method,
        "norm": "inf" if norm == np.inf else norm,
        "lp_runtime_seconds": round(lp_runtime, 6),
        "rounding_runtime_seconds": round(rounding_runtime, 6),
        "total_runtime_seconds": round(total_runtime, 6),
    }


# ============================================================
# Main experiment function
# ============================================================

def run_full_experiment(
    S,
    p_delete,
    seed,
    pivot_seeds,
    compute_pivot=False,
    compute_bad_triangles=True,
    compute_disjoint_bad_triangles=False,
    compute_bad_triangle_primal_bound=False,
    compute_bad_triangle_dual_bound=False,
    compute_actual_lp=True,
    compute_actual_ilp=False,
    compute_observed_edge_lp=False,
    compute_observed_edge_ilp=False,
    compute_observed_edge_four_cycle_lp=False,
    compute_observed_edge_four_cycle_ilp=False,
    compute_min_max=True,
    compute_min_max_lp=True,
    min_max_cc_param_1=2,
    min_max_cc_param_2=2,
    min_max_lp_r=0.1,
    min_max_lp_r2=0.5,
    min_max_lp_method=0,
    min_max_lp_norm=np.inf,
):
    """
    Run one experiment on one signed graph.

    Main formulation:
        solve_all_pairs(..., relax=True)  -> actual LP relaxation
        solve_all_pairs(..., relax=False) -> actual ILP

    Old observed-edge formulation:
        solve_ilp(..., relax=True/False)
        Only runs if the observed-edge flags are True.
    """
    start_time = time.time()
    n = S.shape[0]

    G = matrix_to_graph(S)

    # ============================================================
    # Complete graph: min-max methods
    # ============================================================

    min_max_cc_results = compute_min_max_cc_data(
        S,
        compute_min_max=compute_min_max,
        param_1=min_max_cc_param_1,
        param_2=min_max_cc_param_2,
    )

    min_max_lp_results = compute_min_max_lp_data(
        S,
        compute_min_max_lp=compute_min_max_lp,
        r=min_max_lp_r,
        r2=min_max_lp_r2,
        method=min_max_lp_method,
        norm=min_max_lp_norm,
    )

    # ============================================================
    # Generate incomplete graph by deleting edges
    # ============================================================

    S_new, num_edges_deleted = delete_edges(S, p_delete, seed)
    G_new = matrix_to_graph(S_new)

    # ============================================================
    # Edge-deleted graph: min-max methods
    # ============================================================

    min_max_cc_results_new = compute_min_max_cc_data(
        S_new,
        compute_min_max=compute_min_max,
        param_1=min_max_cc_param_1,
        param_2=min_max_cc_param_2,
    )

    min_max_lp_results_new = compute_min_max_lp_data(
        S_new,
        compute_min_max_lp=compute_min_max_lp,
        r=min_max_lp_r,
        r2=min_max_lp_r2,
        method=min_max_lp_method,
        norm=min_max_lp_norm,
    )

    # ============================================================
    # Complete graph: Pivot
    # ============================================================

    if compute_pivot:
        pivot_results = run_pivot_multiple(S, pivot_seeds)
        pivot_clusters = pivot_results["best_clusters"]
        pivots = pivot_results["best_pivots"]
    else:
        pivot_results = empty_pivot_results()
        pivot_clusters = None
        pivots = None

    # ============================================================
    # Complete graph: bad triangles and disjoint bad triangles
    # ============================================================

    bad_triangle_data = compute_bad_triangle_data(
        S,
        compute_bad_triangles=compute_bad_triangles,
        compute_disjoint_bad_triangles=compute_disjoint_bad_triangles,
    )

    all_bad_triangles = bad_triangle_data["all_bad_triangles"]
    min_num_bad_triangles = bad_triangle_data["min_num_bad_triangles"]
    max_num_bad_triangles = bad_triangle_data["max_num_bad_triangles"]

    # ============================================================
    # Complete graph: actual all-pairs LP relaxation
    # ============================================================

    if compute_actual_lp:
        actual_lp_cost, actual_lp_x_values, actual_lp_info = solve_actual_all_pairs_formulation(
            S,
            relax=True,
        )
    else:
        actual_lp_cost = None
        actual_lp_x_values = None
        actual_lp_info = None

    # ============================================================
    # Complete graph: actual all-pairs ILP
    # ============================================================

    if compute_actual_ilp:
        actual_ilp_cost, actual_ilp_x_values, actual_ilp_info = solve_actual_all_pairs_formulation(
            S,
            relax=False,
        )
        actual_ilp_clusters = find_ilp_clusters(actual_ilp_x_values, n)
        actual_ilp_cluster_count = len(actual_ilp_clusters)
    else:
        actual_ilp_cost = None
        actual_ilp_x_values = None
        actual_ilp_info = None
        actual_ilp_clusters = None
        actual_ilp_cluster_count = None

    # ============================================================
    # Complete graph: observed-edge LP/ILP, optional old formulation
    # ============================================================

    if compute_observed_edge_lp:
        observed_edge_lp_cost, observed_edge_lp_x_values, observed_edge_lp_bad_cycles = (
            solve_observed_edge_formulation(
                S,
                relax=True,
                add_four_cycles=False,
            )
        )
    else:
        observed_edge_lp_cost = None
        observed_edge_lp_x_values = None
        observed_edge_lp_bad_cycles = []

    if compute_observed_edge_ilp:
        observed_edge_ilp_cost, observed_edge_ilp_x_values, observed_edge_ilp_bad_cycles = (
            solve_observed_edge_formulation(
                S,
                relax=False,
                add_four_cycles=False,
            )
        )
        observed_edge_ilp_clusters = find_ilp_clusters(observed_edge_ilp_x_values, n)
        observed_edge_ilp_cluster_count = len(observed_edge_ilp_clusters)
    else:
        observed_edge_ilp_cost = None
        observed_edge_ilp_x_values = None
        observed_edge_ilp_bad_cycles = []
        observed_edge_ilp_clusters = None
        observed_edge_ilp_cluster_count = None

    # ============================================================
    # Complete graph: bad-triangle primal/dual LP bounds
    # ============================================================

    primal_cost, primal_x_values, dual_cost, dual_x_values = compute_bad_triangle_lp_bounds(
        S,
        all_bad_triangles,
        compute_primal_bound=compute_bad_triangle_primal_bound,
        compute_dual_bound=compute_bad_triangle_dual_bound,
    )

    # ============================================================
    # Incomplete graph: Pivot
    # ============================================================

    if compute_pivot:
        pivot_results_new = run_pivot_multiple(S_new, pivot_seeds)
        pivot_clusters_new = pivot_results_new["best_clusters"]
        pivots_new = pivot_results_new["best_pivots"]
    else:
        pivot_results_new = empty_pivot_results()
        pivot_clusters_new = None
        pivots_new = None

    # ============================================================
    # Incomplete graph: bad triangles and disjoint bad triangles
    # ============================================================

    bad_triangle_data_new = compute_bad_triangle_data(
        S_new,
        compute_bad_triangles=compute_bad_triangles,
        compute_disjoint_bad_triangles=compute_disjoint_bad_triangles,
    )

    all_bad_triangles_new = bad_triangle_data_new["all_bad_triangles"]
    min_num_bad_triangles_new = bad_triangle_data_new["min_num_bad_triangles"]
    max_num_bad_triangles_new = bad_triangle_data_new["max_num_bad_triangles"]

    # ============================================================
    # Incomplete graph: actual all-pairs LP relaxation
    # ============================================================

    if compute_actual_lp:
        actual_lp_cost_new, actual_lp_x_values_new, actual_lp_info_new = solve_actual_all_pairs_formulation(
            S_new,
            relax=True,
        )
    else:
        actual_lp_cost_new = None
        actual_lp_x_values_new = None
        actual_lp_info_new = None

    # ============================================================
    # Incomplete graph: actual all-pairs ILP
    # ============================================================

    if compute_actual_ilp:
        actual_ilp_cost_new, actual_ilp_x_values_new, actual_ilp_info_new = solve_actual_all_pairs_formulation(
            S_new,
            relax=False,
        )
        actual_ilp_clusters_new = find_ilp_clusters(actual_ilp_x_values_new, n)
        actual_ilp_cluster_count_new = len(actual_ilp_clusters_new)
    else:
        actual_ilp_cost_new = None
        actual_ilp_x_values_new = None
        actual_ilp_info_new = None
        actual_ilp_clusters_new = None
        actual_ilp_cluster_count_new = None

    # ============================================================
    # Incomplete graph: observed-edge LP without bad 4-cycles
    # ============================================================

    if compute_observed_edge_lp:
        observed_edge_lp_cost_new_no4, observed_edge_lp_x_values_new_no4, observed_edge_lp_bad_cycles_new_no4 = (
            solve_observed_edge_formulation(
                S_new,
                relax=True,
                add_four_cycles=False,
            )
        )
    else:
        observed_edge_lp_cost_new_no4 = None
        observed_edge_lp_x_values_new_no4 = None
        observed_edge_lp_bad_cycles_new_no4 = []

    # ============================================================
    # Incomplete graph: observed-edge LP with bad 4-cycles
    # ============================================================

    if compute_observed_edge_lp and compute_observed_edge_four_cycle_lp:
        observed_edge_lp_cost_new_with4, observed_edge_lp_x_values_new_with4, observed_edge_lp_bad_cycles_new_with4 = (
            solve_observed_edge_formulation(
                S_new,
                relax=True,
                add_four_cycles=True,
            )
        )
    else:
        observed_edge_lp_cost_new_with4 = None
        observed_edge_lp_x_values_new_with4 = None
        observed_edge_lp_bad_cycles_new_with4 = []

    # ============================================================
    # Incomplete graph: observed-edge ILP without bad 4-cycles
    # ============================================================

    if compute_observed_edge_ilp:
        observed_edge_ilp_cost_new_no4, observed_edge_ilp_x_values_new_no4, observed_edge_ilp_bad_cycles_new_no4 = (
            solve_observed_edge_formulation(
                S_new,
                relax=False,
                add_four_cycles=False,
            )
        )
        observed_edge_ilp_clusters_new_no4 = find_ilp_clusters(
            observed_edge_ilp_x_values_new_no4,
            n,
        )
        observed_edge_ilp_cluster_count_new_no4 = len(observed_edge_ilp_clusters_new_no4)
    else:
        observed_edge_ilp_cost_new_no4 = None
        observed_edge_ilp_x_values_new_no4 = None
        observed_edge_ilp_bad_cycles_new_no4 = []
        observed_edge_ilp_clusters_new_no4 = None
        observed_edge_ilp_cluster_count_new_no4 = None

    # ============================================================
    # Incomplete graph: observed-edge ILP with bad 4-cycles
    # ============================================================

    if compute_observed_edge_ilp and compute_observed_edge_four_cycle_ilp:
        observed_edge_ilp_cost_new_with4, observed_edge_ilp_x_values_new_with4, observed_edge_ilp_bad_cycles_new_with4 = (
            solve_observed_edge_formulation(
                S_new,
                relax=False,
                add_four_cycles=True,
            )
        )
        observed_edge_ilp_clusters_new_with4 = find_ilp_clusters(
            observed_edge_ilp_x_values_new_with4,
            n,
        )
        observed_edge_ilp_cluster_count_new_with4 = len(observed_edge_ilp_clusters_new_with4)
    else:
        observed_edge_ilp_cost_new_with4 = None
        observed_edge_ilp_x_values_new_with4 = None
        observed_edge_ilp_bad_cycles_new_with4 = []
        observed_edge_ilp_clusters_new_with4 = None
        observed_edge_ilp_cluster_count_new_with4 = None

    # ============================================================
    # Incomplete graph: bad-triangle primal/dual LP bounds
    # ============================================================

    primal_cost_new, primal_x_values_new, dual_cost_new, dual_x_values_new = compute_bad_triangle_lp_bounds(
        S_new,
        all_bad_triangles_new,
        compute_primal_bound=compute_bad_triangle_primal_bound,
        compute_dual_bound=compute_bad_triangle_dual_bound,
    )

    total_runtime = time.time() - start_time

    experiment_data = {
        # Useful for drawing/debugging
        "S": S,
        "G": G,
        "S_new": S_new,
        "G_new": G_new,

        "pivot_clusters": pivot_clusters,
        "pivots": pivots,
        "actual_ilp_clusters": actual_ilp_clusters,
        "observed_edge_ilp_clusters": observed_edge_ilp_clusters,

        "pivot_clusters_new": pivot_clusters_new,
        "pivots_new": pivots_new,
        "actual_ilp_clusters_new": actual_ilp_clusters_new,
        "observed_edge_ilp_clusters_new_no4": observed_edge_ilp_clusters_new_no4,
        "observed_edge_ilp_clusters_new_with4": observed_edge_ilp_clusters_new_with4,

        # General
        "num_edges_deleted": num_edges_deleted,
        "total_runtime": total_runtime,

        # Complete graph
        "pivot_results": pivot_results,
        "all_bad_triangles": all_bad_triangles,
        "min_num_bad_triangles": min_num_bad_triangles,
        "max_num_bad_triangles": max_num_bad_triangles,


        # ============================================================
        # START MIN MAX THINGS
        # ============================================================

        "min_max_cc_results": min_max_cc_results,
        "min_max_lp_results": min_max_lp_results,

        # ============================================================
        # END MIN MAX THINGS
        # ============================================================

        "actual_lp_cost": actual_lp_cost,
        "actual_lp_info": actual_lp_info,
        "actual_ilp_cost": actual_ilp_cost,
        "actual_ilp_info": actual_ilp_info,
        "actual_ilp_cluster_count": actual_ilp_cluster_count,

        "observed_edge_lp_cost": observed_edge_lp_cost,
        "observed_edge_ilp_cost": observed_edge_ilp_cost,
        "observed_edge_ilp_cluster_count": observed_edge_ilp_cluster_count,
        "observed_edge_lp_bad_cycles": observed_edge_lp_bad_cycles,
        "observed_edge_ilp_bad_cycles": observed_edge_ilp_bad_cycles,

        "primal_cost": primal_cost,
        "dual_cost": dual_cost,

        # Edge-deleted graph
        "pivot_results_new": pivot_results_new,
        "all_bad_triangles_new": all_bad_triangles_new,
        "min_num_bad_triangles_new": min_num_bad_triangles_new,
        "max_num_bad_triangles_new": max_num_bad_triangles_new,
        # ============================================================
        # START MIN MAX THINGS
        # ============================================================

        "min_max_cc_results_new": min_max_cc_results_new,
        "min_max_lp_results_new": min_max_lp_results_new,

        # ============================================================
        # END MIN MAX THINGS
        # ============================================================

        "actual_lp_cost_new": actual_lp_cost_new,
        "actual_lp_info_new": actual_lp_info_new,
        "actual_ilp_cost_new": actual_ilp_cost_new,
        "actual_ilp_info_new": actual_ilp_info_new,
        "actual_ilp_cluster_count_new": actual_ilp_cluster_count_new,

        "observed_edge_lp_cost_new_no4": observed_edge_lp_cost_new_no4,
        "observed_edge_lp_cost_new_with4": observed_edge_lp_cost_new_with4,
        "observed_edge_lp_bad_cycles_new_no4": observed_edge_lp_bad_cycles_new_no4,
        "observed_edge_lp_bad_cycles_new_with4": observed_edge_lp_bad_cycles_new_with4,

        "observed_edge_ilp_cost_new_no4": observed_edge_ilp_cost_new_no4,
        "observed_edge_ilp_cost_new_with4": observed_edge_ilp_cost_new_with4,
        "observed_edge_ilp_cluster_count_new_no4": observed_edge_ilp_cluster_count_new_no4,
        "observed_edge_ilp_cluster_count_new_with4": observed_edge_ilp_cluster_count_new_with4,
        "observed_edge_ilp_bad_cycles_new_no4": observed_edge_ilp_bad_cycles_new_no4,
        "observed_edge_ilp_bad_cycles_new_with4": observed_edge_ilp_bad_cycles_new_with4,

        "primal_cost_new": primal_cost_new,
        "dual_cost_new": dual_cost_new,
    }

    return experiment_data


# ============================================================
# JSON result builder
# ============================================================

def build_saveable_results(graph_params, experiment_data):
    return {
        "graph_params": graph_params,

        "complete_graph": {
            "pivot": {
                "best_cost": experiment_data["pivot_results"]["best_cost"],
                "average_cost": experiment_data["pivot_results"]["average_cost"],
            },
            "bad_triangles": {
                "total_count": len(experiment_data["all_bad_triangles"]),
                "min_edge_disjoint_count": experiment_data["min_num_bad_triangles"],
                "max_edge_disjoint_count": experiment_data["max_num_bad_triangles"],
            },
                # ============================================================
                # START MIN MAX THINGS
                # ============================================================

            "min_max_cc": {
                "max_disagreement": experiment_data["min_max_cc_results"]["max_disagreement"],
                "cluster_count": experiment_data["min_max_cc_results"]["cluster_count"],
                "runtime_seconds": experiment_data["min_max_cc_results"]["runtime_seconds"],
            },
            "min_max_lp_rounding": {
                "lp_cost": experiment_data["min_max_lp_results"]["lp_cost"],
                "rounding_cost": experiment_data["min_max_lp_results"]["rounding_cost"],
                "cluster_count": experiment_data["min_max_lp_results"]["cluster_count"],
                "r": experiment_data["min_max_lp_results"]["r"],
                "r2": experiment_data["min_max_lp_results"]["r2"],
                "method": experiment_data["min_max_lp_results"]["method"],
                "lp_runtime_seconds": experiment_data["min_max_lp_results"]["lp_runtime_seconds"],
                "rounding_runtime_seconds": experiment_data["min_max_lp_results"]["rounding_runtime_seconds"],
            },
                # ============================================================
                # END MIN MAX THINGS
                # ============================================================

            "actual_all_pairs_lp_relaxation": {
                "cost": experiment_data["actual_lp_cost"],
                "solve_info": experiment_data["actual_lp_info"],
            },
            "actual_all_pairs_ilp": {
                "cost": experiment_data["actual_ilp_cost"],
                "solve_info": experiment_data["actual_ilp_info"],
                "cluster_count": experiment_data["actual_ilp_cluster_count"],
            },
            "observed_edge_lp_relaxation": {
                "cost": experiment_data["observed_edge_lp_cost"],
            },
            "observed_edge_ilp": {
                "cost": experiment_data["observed_edge_ilp_cost"],
                "cluster_count": experiment_data["observed_edge_ilp_cluster_count"],
            },
            "bad_triangle_lp_bounds": {
                "primal_cost": experiment_data["primal_cost"],
                "dual_cost": experiment_data["dual_cost"],
            },
        },

        "edge_deleted_graph": {
            "num_edges_deleted": experiment_data["num_edges_deleted"],
            "pivot": {
                "best_cost": experiment_data["pivot_results_new"]["best_cost"],
                "average_cost": experiment_data["pivot_results_new"]["average_cost"],
            },
            "bad_triangles": {
                "total_count": len(experiment_data["all_bad_triangles_new"]),
                "min_edge_disjoint_count": experiment_data["min_num_bad_triangles_new"],
                "max_edge_disjoint_count": experiment_data["max_num_bad_triangles_new"],
            },
                # ============================================================
                # START MIN MAX THINGS
                # ============================================================

            "min_max_cc": {
                "max_disagreement": experiment_data["min_max_cc_results_new"]["max_disagreement"],
                "cluster_count": experiment_data["min_max_cc_results_new"]["cluster_count"],
                "runtime_seconds": experiment_data["min_max_cc_results_new"]["runtime_seconds"],
            },
            "min_max_lp_rounding": {
                "lp_cost": experiment_data["min_max_lp_results_new"]["lp_cost"],
                "rounding_cost": experiment_data["min_max_lp_results_new"]["rounding_cost"],
                "cluster_count": experiment_data["min_max_lp_results_new"]["cluster_count"],
                "r": experiment_data["min_max_lp_results_new"]["r"],
                "r2": experiment_data["min_max_lp_results_new"]["r2"],
                "method": experiment_data["min_max_lp_results_new"]["method"],
                "lp_runtime_seconds": experiment_data["min_max_lp_results_new"]["lp_runtime_seconds"],
                "rounding_runtime_seconds": experiment_data["min_max_lp_results_new"]["rounding_runtime_seconds"],
            },
                # ============================================================
                # END MIN MAX THINGS
                # ============================================================

            "actual_all_pairs_lp_relaxation": {
                "cost": experiment_data["actual_lp_cost_new"],
                "solve_info": experiment_data["actual_lp_info_new"],
            },
            "actual_all_pairs_ilp": {
                "cost": experiment_data["actual_ilp_cost_new"],
                "solve_info": experiment_data["actual_ilp_info_new"],
                "cluster_count": experiment_data["actual_ilp_cluster_count_new"],
            },
            "observed_edge_lp_relaxation": {
                "without_4_cycles": {
                    "cost": experiment_data["observed_edge_lp_cost_new_no4"],
                },
                "with_4_cycles": {
                    "cost": experiment_data["observed_edge_lp_cost_new_with4"],
                    "bad_4_cycles_count": len(experiment_data["observed_edge_lp_bad_cycles_new_with4"]),
                },
            },
            "observed_edge_ilp": {
                "without_4_cycles": {
                    "cost": experiment_data["observed_edge_ilp_cost_new_no4"],
                    "cluster_count": experiment_data["observed_edge_ilp_cluster_count_new_no4"],
                },
                "with_4_cycles": {
                    "cost": experiment_data["observed_edge_ilp_cost_new_with4"],
                    "cluster_count": experiment_data["observed_edge_ilp_cluster_count_new_with4"],
                    "bad_4_cycles_count": len(experiment_data["observed_edge_ilp_bad_cycles_new_with4"]),
                },
            },
            "bad_triangle_lp_bounds": {
                "primal_cost": experiment_data["primal_cost_new"],
                "dual_cost": experiment_data["dual_cost_new"],
            },
        },

        "runtime_seconds": experiment_data["total_runtime"],
    }


# ============================================================
# Standard printing
# ============================================================

def print_standard_results(graph_type, graph_params, experiment_data):
    print_section(f"{graph_type.upper()} Graph Parameters")

    for key, value in graph_params.items():
        print(f"{key}:", value)

    print_section(f"Complete {graph_type} Graph")

    if experiment_data["pivot_results"]["best_cost"] is not None:
        print_subsection("Pivot")
        print("Best Pivot cost:", experiment_data["pivot_results"]["best_cost"])
        print("Average Pivot cost:", experiment_data["pivot_results"]["average_cost"])

    if experiment_data["all_bad_triangles"]:
        print_subsection("Bad triangles")
        print("Total number of bad triangles:", len(experiment_data["all_bad_triangles"]))
        print_if_not_none("Minimum amount of disjoint bad triangles", experiment_data["min_num_bad_triangles"])
        print_if_not_none("Maximum amount of disjoint bad triangles", experiment_data["max_num_bad_triangles"])

    # ============================================================
    # START MIN MAX THINGS
    # ============================================================

    if experiment_data["min_max_cc_results"]["max_disagreement"] is not None:
        print_subsection("Min-max from min_max.py")
        print("Min-max max disagreement:", experiment_data["min_max_cc_results"]["max_disagreement"])
        print("Min-max cluster count:", experiment_data["min_max_cc_results"]["cluster_count"])

    if experiment_data["min_max_lp_results"]["lp_cost"] is not None:
        print_subsection("Min-max LP + rounding from min_max_lp.py")
        print("Min-max LP cost:", experiment_data["min_max_lp_results"]["lp_cost"])
        print("Min-max LP rounding cost:", experiment_data["min_max_lp_results"]["rounding_cost"])
        print("Min-max LP cluster count:", experiment_data["min_max_lp_results"]["cluster_count"])

    # ============================================================
    # END MIN MAX THINGS
    # ============================================================

    if experiment_data["actual_lp_cost"] is not None:
        print_subsection("Actual all-pairs LP relaxation")
        print("Actual LP cost:", experiment_data["actual_lp_cost"])

    if experiment_data["actual_ilp_cost"] is not None:
        print_subsection("Actual all-pairs ILP")
        print("Actual ILP cost:", experiment_data["actual_ilp_cost"])
        print("Actual ILP cluster count:", experiment_data["actual_ilp_cluster_count"])

    if experiment_data["observed_edge_lp_cost"] is not None:
        print_subsection("Observed-edge LP relaxation")
        print("Observed-edge LP cost:", experiment_data["observed_edge_lp_cost"])

    if experiment_data["observed_edge_ilp_cost"] is not None:
        print_subsection("Observed-edge ILP")
        print("Observed-edge ILP cost:", experiment_data["observed_edge_ilp_cost"])
        print("Observed-edge ILP cluster count:", experiment_data["observed_edge_ilp_cluster_count"])

    if experiment_data["primal_cost"] is not None or experiment_data["dual_cost"] is not None:
        print_subsection("Bad-triangle LP bounds")
        print_if_not_none("LP-primal optimal cost", experiment_data["primal_cost"])
        print_if_not_none("LP-dual optimal cost", experiment_data["dual_cost"])

    print_section(f"Edge-Deleted {graph_type} Graph")
    print("Number of edges deleted:", experiment_data["num_edges_deleted"])

    if experiment_data["pivot_results_new"]["best_cost"] is not None:
        print_subsection("Pivot")
        print("Best Pivot cost:", experiment_data["pivot_results_new"]["best_cost"])
        print("Average Pivot cost:", experiment_data["pivot_results_new"]["average_cost"])

    if experiment_data["all_bad_triangles_new"]:
        print_subsection("Bad triangles")
        print("Total number of bad triangles:", len(experiment_data["all_bad_triangles_new"]))
        print_if_not_none("Minimum amount of disjoint bad triangles", experiment_data["min_num_bad_triangles_new"])
        print_if_not_none("Maximum amount of disjoint bad triangles", experiment_data["max_num_bad_triangles_new"])


    # ============================================================
    # START MIN MAX THINGS
    # ============================================================


    if experiment_data["min_max_cc_results_new"]["max_disagreement"] is not None:
        print_subsection("Min-max from min_max.py")
        print("Min-max max disagreement on edge-deleted graph:", experiment_data["min_max_cc_results_new"]["max_disagreement"])
        print("Min-max cluster count:", experiment_data["min_max_cc_results_new"]["cluster_count"])

    if experiment_data["min_max_lp_results_new"]["lp_cost"] is not None:
        print_subsection("Min-max LP + rounding from min_max_lp.py")
        print("Min-max LP cost on edge-deleted graph:", experiment_data["min_max_lp_results_new"]["lp_cost"])
        print("Min-max LP rounding cost on edge-deleted graph:", experiment_data["min_max_lp_results_new"]["rounding_cost"])
        print("Min-max LP cluster count:", experiment_data["min_max_lp_results_new"]["cluster_count"])


    # ============================================================
    # END MIN MAX THINGS
    # ============================================================

    if experiment_data["actual_lp_cost_new"] is not None:
        print_subsection("Actual all-pairs LP relaxation")
        print("Actual LP cost on edge-deleted graph:", experiment_data["actual_lp_cost_new"])

    if experiment_data["actual_ilp_cost_new"] is not None:
        print_subsection("Actual all-pairs ILP")
        print("Actual ILP cost on edge-deleted graph:", experiment_data["actual_ilp_cost_new"])
        print("Actual ILP cluster count:", experiment_data["actual_ilp_cluster_count_new"])

    if experiment_data["observed_edge_lp_cost_new_no4"] is not None:
        print_subsection("Observed-edge LP relaxation")
        print("Observed-edge LP cost without 4-cycles:", experiment_data["observed_edge_lp_cost_new_no4"])
        print_if_not_none("Observed-edge LP cost with 4-cycles", experiment_data["observed_edge_lp_cost_new_with4"])
        print("Bad 4-cycles detected:", len(experiment_data["observed_edge_lp_bad_cycles_new_with4"]))

    if experiment_data["observed_edge_ilp_cost_new_no4"] is not None:
        print_subsection("Observed-edge ILP")
        print("Observed-edge ILP cost without 4-cycles:", experiment_data["observed_edge_ilp_cost_new_no4"])
        print_if_not_none("Observed-edge ILP cost with 4-cycles", experiment_data["observed_edge_ilp_cost_new_with4"])
        print("Bad 4-cycles detected:", len(experiment_data["observed_edge_ilp_bad_cycles_new_with4"]))

    if experiment_data["primal_cost_new"] is not None or experiment_data["dual_cost_new"] is not None:
        print_subsection("Bad-triangle LP bounds")
        print_if_not_none("LP-primal optimal cost", experiment_data["primal_cost_new"])
        print_if_not_none("LP-dual optimal cost", experiment_data["dual_cost_new"])

    print_section("Runtime")
    print("Total runtime:", round(experiment_data["total_runtime"], 2), "seconds")
