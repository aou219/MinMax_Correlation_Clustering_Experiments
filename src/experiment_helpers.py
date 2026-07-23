#!/usr/bin/env python3
"""Shared helpers for the current Facebook correlation-clustering experiments.

Kept in this project:
- Pivot
- the all-pairs correlation-clustering LP/ILP
- MinMaxCC
- MinMaxLP as an LP lower bound
- edge deletion
- JSON saving and console output
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    # Works when running modules with: python -m src.experiment_facebook
    from .all_pairs_solver import solve_all_pairs
    from .cost import calculate_clustering_cost
    from .edge_deletion import delete_edges
    from .min_max import max_disagreement, min_max_cc
    from .min_max_lp import DegreeDist, LocalObj, MinMaxLP, cluster
    from .pivot import run_pivot
except ImportError:
    # Works when running directly with: python src/experiment_facebook.py
    from all_pairs_solver import solve_all_pairs
    from cost import calculate_clustering_cost
    from edge_deletion import delete_edges
    from min_max import max_disagreement, min_max_cc
    from min_max_lp import DegreeDist, LocalObj, MinMaxLP, cluster
    from pivot import run_pivot


DEFAULT_D_HAT = 8
DEFAULT_LAMBDA = 5
DEFAULT_MIN_MAX_LP_R = 0.4
DEFAULT_MIN_MAX_LP_R2 = 0.4
DEFAULT_MIN_MAX_LP_METHOD = 2

# The current experiment_facebook.py may still pass these old switches.
# False values are accepted temporarily so that script does not break.
REMOVED_FALSE_ONLY_OPTIONS = {
    "compute_bad_triangles",
    "compute_disjoint_bad_triangles",
    "compute_bad_triangle_primal_bound",
    "compute_bad_triangle_dual_bound",
    "compute_observed_edge_lp",
    "compute_observed_edge_ilp",
    "compute_observed_edge_four_cycle_lp",
    "compute_observed_edge_four_cycle_ilp",
}


# ============================================================
# Validation and general helpers
# ============================================================

def validate_signed_matrix(S: np.ndarray) -> np.ndarray:
    """Validate the signed adjacency-matrix convention.

    Values:
        1  positive edge
       -1  negative edge
        0  deleted/unobserved edge or diagonal
    """
    matrix = np.asarray(S)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"S must be square, received shape {matrix.shape}."
        )

    if not np.array_equal(matrix, matrix.T):
        raise ValueError("S must be symmetric.")

    if not np.all(np.diag(matrix) == 0):
        raise ValueError("The diagonal of S must contain only zeros.")

    valid = np.isin(matrix, (-1, 0, 1))
    if not np.all(valid):
        invalid_values = sorted(
            set(np.unique(matrix[~valid]).tolist())
        )
        raise ValueError(
            "S may contain only -1, 0, and 1. "
            f"Invalid values: {invalid_values}"
        )

    return matrix


def safe_ratio(
    numerator: Any,
    denominator: Any,
) -> float | None:
    if numerator is None or denominator is None:
        return None

    denominator_value = float(denominator)
    if denominator_value == 0:
        return None

    return float(numerator) / denominator_value


def json_converter(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, set):
        return sorted(obj)

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable."
    )


def save_results_append(
    filename: str | Path,
    new_results: Mapping[str, Any],
) -> None:
    """Append one result dictionary to a JSON list."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            all_results = json.load(handle)
    else:
        all_results = []

    # Compatibility with older result files.
    if isinstance(all_results, dict) and "experiments" in all_results:
        all_results = all_results["experiments"]

    if not isinstance(all_results, list):
        raise ValueError(
            f"Expected a JSON list in {path}, found "
            f"{type(all_results).__name__}."
        )

    all_results.append(dict(new_results))

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            all_results,
            handle,
            indent=4,
            default=json_converter,
        )


def _check_removed_options(options: Mapping[str, Any]) -> None:
    unknown = sorted(
        set(options).difference(REMOVED_FALSE_ONLY_OPTIONS)
    )
    if unknown:
        raise TypeError(
            "Unexpected run_full_experiment option(s): "
            + ", ".join(unknown)
        )

    enabled = sorted(
        name
        for name, value in options.items()
        if bool(value)
    )
    if enabled:
        raise ValueError(
            "These removed thesis-only options cannot be enabled: "
            + ", ".join(enabled)
        )


# ============================================================
# Pivot
# ============================================================

def empty_pivot_results() -> dict[str, Any]:
    return {
        "computed": False,
        "runs": [],
        "best_cost": None,
        "best_cluster_count": None,
        "best_clusters": None,
        "best_pivots": None,
        "average_cost": None,
        "average_cluster_count": None,
        "average_runtime_seconds": None,
        "total_runtime_seconds": 0.0,
    }


def run_pivot_multiple(
    S: np.ndarray,
    pivot_seeds: Iterable[int],
) -> dict[str, Any]:
    """Run Pivot once for every supplied seed."""
    matrix = validate_signed_matrix(S)
    seeds = list(pivot_seeds)

    if not seeds:
        return empty_pivot_results()

    runs: list[dict[str, Any]] = []
    best_cost: float | int | None = None
    best_cluster_count: int | None = None
    best_clusters = None
    best_pivots = None

    total_start = time.perf_counter()

    for pivot_seed in seeds:
        run_start = time.perf_counter()
        clusters, pivots = run_pivot(matrix, pivot_seed)
        runtime = time.perf_counter() - run_start

        cost = calculate_clustering_cost(matrix, clusters)
        cluster_count = len(clusters)

        runs.append({
            "pivot_seed": int(pivot_seed),
            "cost": cost,
            "cluster_count": cluster_count,
            "runtime_seconds": round(runtime, 6),
        })

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cluster_count = cluster_count
            best_clusters = clusters
            best_pivots = pivots

    total_runtime = time.perf_counter() - total_start

    return {
        "computed": True,
        "runs": runs,
        "best_cost": best_cost,
        "best_cluster_count": best_cluster_count,
        "best_clusters": best_clusters,
        "best_pivots": best_pivots,
        "average_cost": (
            sum(run["cost"] for run in runs) / len(runs)
        ),
        "average_cluster_count": (
            sum(run["cluster_count"] for run in runs) / len(runs)
        ),
        "average_runtime_seconds": (
            sum(run["runtime_seconds"] for run in runs) / len(runs)
        ),
        "total_runtime_seconds": round(total_runtime, 6),
    }


# ============================================================
# All-pairs standard correlation clustering
# ============================================================

def solve_actual_all_pairs_formulation(
    S: np.ndarray,
    relax: bool = False,
    time_limit: float | None = None,
) -> tuple[float, dict[tuple[int, int], Any] | None, dict[str, Any]]:
    """Run the all-pairs formulation from all_pairs_solver.py."""
    matrix = validate_signed_matrix(S)

    return solve_all_pairs(
        matrix,
        time_limit=time_limit,
        verbose=False,
        relax=relax,
        return_x_values=True,
    )


def clusters_from_all_pairs_ilp(
    x_values: Mapping[tuple[int, int], Any],
    n: int,
) -> list[set[int]]:
    """Recover clusters from an integral all-pairs ILP solution.

    In the all-pairs formulation x_ij = 0 means that i and j are in
    the same cluster, while x_ij = 1 means that they are separated.
    """
    parent = list(range(n))
    rank = [0] * n

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)

        if root_left == root_right:
            return

        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    for (i, j), value in x_values.items():
        if float(value) <= 0.5:
            union(int(i), int(j))

    groups: dict[int, set[int]] = {}
    for node in range(n):
        groups.setdefault(find(node), set()).add(node)

    return sorted(
        groups.values(),
        key=lambda cluster: min(cluster),
    )


def compute_all_pairs_data(
    S: np.ndarray,
    compute_lp: bool = True,
    compute_ilp: bool = False,
    time_limit: float | None = None,
) -> dict[str, Any]:
    """Compute the all-pairs LP relaxation and/or ILP."""
    matrix = validate_signed_matrix(S)
    n = matrix.shape[0]

    if compute_lp:
        lp_cost, lp_x_values, lp_info = (
            solve_actual_all_pairs_formulation(
                matrix,
                relax=True,
                time_limit=time_limit,
            )
        )
    else:
        lp_cost = None
        lp_x_values = None
        lp_info = None

    if compute_ilp:
        ilp_cost, ilp_x_values, ilp_info = (
            solve_actual_all_pairs_formulation(
                matrix,
                relax=False,
                time_limit=time_limit,
            )
        )
        if ilp_x_values is None:
            raise RuntimeError(
                "The all-pairs ILP returned no x-values."
            )
        ilp_clusters = clusters_from_all_pairs_ilp(
            ilp_x_values,
            n,
        )
        ilp_cluster_count = len(ilp_clusters)
    else:
        ilp_cost = None
        ilp_x_values = None
        ilp_info = None
        ilp_clusters = None
        ilp_cluster_count = None

    return {
        "lp_computed": compute_lp,
        "lp_cost": lp_cost,
        "lp_x_values": lp_x_values,
        "lp_info": lp_info,
        "ilp_computed": compute_ilp,
        "ilp_cost": ilp_cost,
        "ilp_x_values": ilp_x_values,
        "ilp_info": ilp_info,
        "ilp_clusters": ilp_clusters,
        "ilp_cluster_count": ilp_cluster_count,
    }


# ============================================================
# MinMaxCC
# ============================================================

def compute_min_max_cc_data(
    S: np.ndarray,
    compute_min_max: bool = True,
    param_1: int = DEFAULT_D_HAT,
    param_2: int = DEFAULT_LAMBDA,
) -> dict[str, Any]:
    """Run MinMaxCC and record its runtime."""
    matrix = validate_signed_matrix(S)

    if not compute_min_max:
        return {
            "computed": False,
            "clustering": None,
            "cluster_count": None,
            "max_disagreement": None,
            "d_hat": param_1,
            "lambda": param_2,
            "runtime_seconds": None,
        }

    if param_1 < 0:
        raise ValueError("d_hat must be non-negative.")
    if param_2 <= 4:
        raise ValueError("lambda must be greater than 4.")

    start = time.perf_counter()
    clustering = min_max_cc(matrix, param_1, param_2)
    runtime = time.perf_counter() - start

    if clustering is None:
        raise RuntimeError("min_max_cc returned None.")

    return {
        "computed": True,
        "clustering": clustering,
        "cluster_count": len(clustering),
        "max_disagreement": max_disagreement(
            clustering,
            matrix,
        ),
        "d_hat": param_1,
        "lambda": param_2,
        "runtime_seconds": round(runtime, 6),
    }


# ============================================================
# MinMaxLP
# ============================================================

def compute_min_max_lp_data(
    S: np.ndarray,
    compute_min_max_lp: bool = True,
    r: float = DEFAULT_MIN_MAX_LP_R,
    r2: float = DEFAULT_MIN_MAX_LP_R2,
    method: int = DEFAULT_MIN_MAX_LP_METHOD,
    norm: Any = np.inf,
) -> dict[str, Any]:
    """Solve MinMaxLP and run its rounding algorithm."""
    matrix = validate_signed_matrix(S)
    norm_for_json = "inf" if norm == np.inf else norm

    if not compute_min_max_lp:
        return {
            "computed": False,
            "lp_cost": None,
            "clustering": None,
            "cluster_count": None,
            "disagreement_vector": None,
            "rounding_cost": None,
            "max_disagreement_vertex": None,
            "r": r,
            "r2": r2,
            "method": method,
            "norm": norm_for_json,
            "lp_runtime_seconds": None,
            "rounding_runtime_seconds": None,
            "total_runtime_seconds": None,
        }

    (
        lp_cost,
        distances,
        l_t_values,
        neighbors_r,
        neighbors_r2,
        lp_runtime,
    ) = MinMaxLP(matrix, r, r2, method)

    clustering, rounding_runtime = cluster(
        distances,
        l_t_values,
        neighbors_r,
        neighbors_r2,
        r,
        r2,
    )

    positive_degrees = DegreeDist(matrix)
    (
        disagreement_vector,
        rounding_cost,
        max_disagreement_vertex,
    ) = LocalObj(
        matrix,
        clustering,
        positive_degrees,
        norm,
    )

    lp_runtime_value = float(lp_runtime)
    rounding_runtime_value = float(rounding_runtime)

    return {
        "computed": True,
        "lp_cost": float(lp_cost),
        "clustering": clustering,
        "cluster_count": len(clustering),
        "disagreement_vector": disagreement_vector,
        "rounding_cost": float(rounding_cost),
        "max_disagreement_vertex": int(max_disagreement_vertex),
        "r": r,
        "r2": r2,
        "method": method,
        "norm": norm_for_json,
        "lp_runtime_seconds": round(lp_runtime_value, 6),
        "rounding_runtime_seconds": round(
            rounding_runtime_value,
            6,
        ),
        "total_runtime_seconds": round(
            lp_runtime_value + rounding_runtime_value,
            6,
        ),
    }


# ============================================================
# Full complete + edge-deleted experiment
# ============================================================

def run_full_experiment(
    S: np.ndarray,
    p_delete: float,
    seed: int,
    pivot_seeds: Iterable[int],
    compute_pivot: bool = False,
    compute_actual_lp: bool = True,
    compute_actual_ilp: bool = False,
    compute_min_max: bool = True,
    compute_min_max_lp: bool = True,
    min_max_cc_param_1: int = DEFAULT_D_HAT,
    min_max_cc_param_2: int = DEFAULT_LAMBDA,
    min_max_lp_r: float = DEFAULT_MIN_MAX_LP_R,
    min_max_lp_r2: float = DEFAULT_MIN_MAX_LP_R2,
    min_max_lp_method: int = DEFAULT_MIN_MAX_LP_METHOD,
    min_max_lp_norm: Any = np.inf,
    all_pairs_time_limit: float | None = None,
    **removed_options: Any,
) -> dict[str, Any]:
    """Run the retained algorithms on complete and edge-deleted graphs."""
    _check_removed_options(removed_options)

    matrix = validate_signed_matrix(S)

    if not 0 <= p_delete <= 1:
        raise ValueError("p_delete must be between 0 and 1.")

    total_start = time.perf_counter()

    complete_pivot = (
        run_pivot_multiple(matrix, pivot_seeds)
        if compute_pivot
        else empty_pivot_results()
    )
    complete_all_pairs = compute_all_pairs_data(
        matrix,
        compute_lp=compute_actual_lp,
        compute_ilp=compute_actual_ilp,
        time_limit=all_pairs_time_limit,
    )
    complete_min_max_cc = compute_min_max_cc_data(
        matrix,
        compute_min_max=compute_min_max,
        param_1=min_max_cc_param_1,
        param_2=min_max_cc_param_2,
    )
    complete_min_max_lp = compute_min_max_lp_data(
        matrix,
        compute_min_max_lp=compute_min_max_lp,
        r=min_max_lp_r,
        r2=min_max_lp_r2,
        method=min_max_lp_method,
        norm=min_max_lp_norm,
    )

    edge_matrix, num_edges_deleted = delete_edges(
        matrix,
        p_delete,
        seed,
    )
    edge_matrix = validate_signed_matrix(edge_matrix)

    edge_pivot = (
        run_pivot_multiple(edge_matrix, pivot_seeds)
        if compute_pivot
        else empty_pivot_results()
    )
    edge_all_pairs = compute_all_pairs_data(
        edge_matrix,
        compute_lp=compute_actual_lp,
        compute_ilp=compute_actual_ilp,
        time_limit=all_pairs_time_limit,
    )
    edge_min_max_cc = compute_min_max_cc_data(
        edge_matrix,
        compute_min_max=compute_min_max,
        param_1=min_max_cc_param_1,
        param_2=min_max_cc_param_2,
    )
    edge_min_max_lp = compute_min_max_lp_data(
        edge_matrix,
        compute_min_max_lp=compute_min_max_lp,
        r=min_max_lp_r,
        r2=min_max_lp_r2,
        method=min_max_lp_method,
        norm=min_max_lp_norm,
    )

    total_runtime = time.perf_counter() - total_start

    # The old key names are preserved where they are still meaningful,
    # so experiment_facebook.py and existing result processing keep working.
    return {
        "S": matrix,
        "S_new": edge_matrix,
        "p_delete": float(p_delete),
        "seed": int(seed),
        "num_edges_deleted": int(num_edges_deleted),

        "pivot_results": complete_pivot,
        "pivot_clusters": complete_pivot["best_clusters"],
        "pivots": complete_pivot["best_pivots"],

        "actual_lp_cost": complete_all_pairs["lp_cost"],
        "actual_lp_x_values": complete_all_pairs["lp_x_values"],
        "actual_lp_info": complete_all_pairs["lp_info"],
        "actual_ilp_cost": complete_all_pairs["ilp_cost"],
        "actual_ilp_x_values": complete_all_pairs["ilp_x_values"],
        "actual_ilp_info": complete_all_pairs["ilp_info"],
        "actual_ilp_clusters": complete_all_pairs["ilp_clusters"],
        "actual_ilp_cluster_count": complete_all_pairs[
            "ilp_cluster_count"
        ],

        "min_max_cc_results": complete_min_max_cc,
        "min_max_lp_results": complete_min_max_lp,
        "complete_min_max_approximation_ratio": safe_ratio(
            complete_min_max_cc["max_disagreement"],
            complete_min_max_lp["lp_cost"],
        ),

        "pivot_results_new": edge_pivot,
        "pivot_clusters_new": edge_pivot["best_clusters"],
        "pivots_new": edge_pivot["best_pivots"],

        "actual_lp_cost_new": edge_all_pairs["lp_cost"],
        "actual_lp_x_values_new": edge_all_pairs["lp_x_values"],
        "actual_lp_info_new": edge_all_pairs["lp_info"],
        "actual_ilp_cost_new": edge_all_pairs["ilp_cost"],
        "actual_ilp_x_values_new": edge_all_pairs["ilp_x_values"],
        "actual_ilp_info_new": edge_all_pairs["ilp_info"],
        "actual_ilp_clusters_new": edge_all_pairs["ilp_clusters"],
        "actual_ilp_cluster_count_new": edge_all_pairs[
            "ilp_cluster_count"
        ],

        "min_max_cc_results_new": edge_min_max_cc,
        "min_max_lp_results_new": edge_min_max_lp,
        "edge_min_max_approximation_ratio": safe_ratio(
            edge_min_max_cc["max_disagreement"],
            edge_min_max_lp["lp_cost"],
        ),

        "total_runtime": round(total_runtime, 6),
    }


# ============================================================
# Saveable result structure
# ============================================================

def _saveable_pivot(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "computed": result["computed"],
        "runs": result["runs"],
        "best_cost": result["best_cost"],
        "best_cluster_count": result["best_cluster_count"],
        "average_cost": result["average_cost"],
        "average_cluster_count": result[
            "average_cluster_count"
        ],
        "average_runtime_seconds": result[
            "average_runtime_seconds"
        ],
        "total_runtime_seconds": result[
            "total_runtime_seconds"
        ],
    }


def _saveable_min_max_cc(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "computed": result["computed"],
        "clustering": result["clustering"],
        "cluster_count": result["cluster_count"],
        "max_disagreement": result["max_disagreement"],
        "d_hat": result["d_hat"],
        "lambda": result["lambda"],
        "runtime_seconds": result["runtime_seconds"],
    }


def _saveable_min_max_lp(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "computed": result["computed"],
        "lp_cost": result["lp_cost"],
        "clustering": result.get("clustering"),
        "cluster_count": result.get("cluster_count"),
        "disagreement_vector": result.get(
            "disagreement_vector"
        ),
        "rounding_cost": result.get("rounding_cost"),
        "max_disagreement_vertex": result.get(
            "max_disagreement_vertex"
        ),
        "r": result["r"],
        "r2": result["r2"],
        "method": result["method"],
        "norm": result["norm"],
        "lp_runtime_seconds": result.get(
            "lp_runtime_seconds"
        ),
        "rounding_runtime_seconds": result.get(
            "rounding_runtime_seconds"
        ),
        "total_runtime_seconds": result.get(
            "total_runtime_seconds"
        ),
    }


def build_saveable_results(
    graph_params: Mapping[str, Any],
    experiment_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact JSON result using only retained algorithms."""
    return {
        "graph_params": dict(graph_params),

        "complete_graph": {
            "pivot": _saveable_pivot(
                experiment_data["pivot_results"]
            ),
            "all_pairs_lp_relaxation": {
                "computed": (
                    experiment_data["actual_lp_cost"]
                    is not None
                ),
                "cost": experiment_data["actual_lp_cost"],
                "solve_info": experiment_data[
                    "actual_lp_info"
                ],
            },
            "all_pairs_ilp": {
                "computed": (
                    experiment_data["actual_ilp_cost"]
                    is not None
                ),
                "cost": experiment_data["actual_ilp_cost"],
                "solve_info": experiment_data[
                    "actual_ilp_info"
                ],
                "clusters": experiment_data[
                    "actual_ilp_clusters"
                ],
                "cluster_count": experiment_data[
                    "actual_ilp_cluster_count"
                ],
            },
            "min_max_cc": _saveable_min_max_cc(
                experiment_data["min_max_cc_results"]
            ),
            "min_max_lp": _saveable_min_max_lp(
                experiment_data["min_max_lp_results"]
            ),
            "min_max_cc_to_lp_ratio": experiment_data[
                "complete_min_max_approximation_ratio"
            ],
        },

        "edge_deleted_graph": {
            "num_edges_deleted": experiment_data[
                "num_edges_deleted"
            ],
            "pivot": _saveable_pivot(
                experiment_data["pivot_results_new"]
            ),
            "all_pairs_lp_relaxation": {
                "computed": (
                    experiment_data["actual_lp_cost_new"]
                    is not None
                ),
                "cost": experiment_data[
                    "actual_lp_cost_new"
                ],
                "solve_info": experiment_data[
                    "actual_lp_info_new"
                ],
            },
            "all_pairs_ilp": {
                "computed": (
                    experiment_data["actual_ilp_cost_new"]
                    is not None
                ),
                "cost": experiment_data[
                    "actual_ilp_cost_new"
                ],
                "solve_info": experiment_data[
                    "actual_ilp_info_new"
                ],
                "clusters": experiment_data[
                    "actual_ilp_clusters_new"
                ],
                "cluster_count": experiment_data[
                    "actual_ilp_cluster_count_new"
                ],
            },
            "min_max_cc": _saveable_min_max_cc(
                experiment_data["min_max_cc_results_new"]
            ),
            "min_max_lp": _saveable_min_max_lp(
                experiment_data["min_max_lp_results_new"]
            ),
            "min_max_cc_to_lp_ratio": experiment_data[
                "edge_min_max_approximation_ratio"
            ],
        },

        "runtime_seconds": experiment_data["total_runtime"],
    }


# ============================================================
# Console output
# ============================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str) -> None:
    print("\n--- " + title + " ---")


def _print_graph_results(
    experiment_data: Mapping[str, Any],
    edge_deleted: bool,
) -> None:
    suffix = "_new" if edge_deleted else ""
    pivot_key = (
        "pivot_results_new"
        if edge_deleted
        else "pivot_results"
    )
    cc_key = (
        "min_max_cc_results_new"
        if edge_deleted
        else "min_max_cc_results"
    )
    lp_key = (
        "min_max_lp_results_new"
        if edge_deleted
        else "min_max_lp_results"
    )
    ratio_key = (
        "edge_min_max_approximation_ratio"
        if edge_deleted
        else "complete_min_max_approximation_ratio"
    )

    pivot_result = experiment_data[pivot_key]
    cc_result = experiment_data[cc_key]
    lp_result = experiment_data[lp_key]

    if pivot_result["computed"]:
        print_subsection("Pivot")
        print("Best cost:", pivot_result["best_cost"])
        print("Average cost:", pivot_result["average_cost"])
        print(
            "Average runtime:",
            pivot_result["average_runtime_seconds"],
            "seconds",
        )

    actual_lp_cost = experiment_data[
        f"actual_lp_cost{suffix}"
    ]
    actual_lp_info = experiment_data[
        f"actual_lp_info{suffix}"
    ]
    if actual_lp_cost is not None:
        print_subsection(
            "All-pairs LP relaxation (standard/L1 CC)"
        )
        print("LP cost:", actual_lp_cost)
        if actual_lp_info is not None:
            print(
                "Runtime:",
                actual_lp_info.get("runtime_seconds"),
                "seconds",
            )

    actual_ilp_cost = experiment_data[
        f"actual_ilp_cost{suffix}"
    ]
    actual_ilp_info = experiment_data[
        f"actual_ilp_info{suffix}"
    ]
    actual_ilp_cluster_count = experiment_data[
        f"actual_ilp_cluster_count{suffix}"
    ]
    if actual_ilp_cost is not None:
        print_subsection(
            "All-pairs ILP (standard/L1 CC)"
        )
        print("ILP cost:", actual_ilp_cost)
        print(
            "Cluster count:",
            actual_ilp_cluster_count,
        )
        if actual_ilp_info is not None:
            print(
                "Runtime:",
                actual_ilp_info.get("runtime_seconds"),
                "seconds",
            )

    if cc_result["computed"]:
        print_subsection("MinMaxCC")
        print(
            "Max disagreement:",
            cc_result["max_disagreement"],
        )
        print("Cluster count:", cc_result["cluster_count"])
        print("d_hat:", cc_result["d_hat"])
        print("lambda:", cc_result["lambda"])
        print(
            "Runtime:",
            cc_result["runtime_seconds"],
            "seconds",
        )

    if lp_result["computed"]:
        print_subsection("MinMaxLP")
        print("LP cost:", lp_result["lp_cost"])
        print(
            "Runtime:",
            lp_result["lp_runtime_seconds"],
            "seconds",
        )

    ratio_value = experiment_data[ratio_key]
    if ratio_value is not None:
        print(
            "\nMinMaxCC / MinMaxLP ratio:",
            ratio_value,
        )


def print_standard_results(
    graph_type: str,
    graph_params: Mapping[str, Any],
    experiment_data: Mapping[str, Any],
) -> None:
    print_section(f"{graph_type.upper()} PARAMETERS")
    for key, value in graph_params.items():
        print(f"{key}: {value}")

    print_section(f"COMPLETE {graph_type.upper()} GRAPH")
    _print_graph_results(
        experiment_data,
        edge_deleted=False,
    )

    print_section(
        f"EDGE-DELETED {graph_type.upper()} GRAPH"
    )
    print(
        "Number of deleted edges:",
        experiment_data["num_edges_deleted"],
    )
    _print_graph_results(
        experiment_data,
        edge_deleted=True,
    )

    print_section("TOTAL RUNTIME")
    print(experiment_data["total_runtime"], "seconds")


__all__ = [
    "DEFAULT_D_HAT",
    "DEFAULT_LAMBDA",
    "build_saveable_results",
    "clusters_from_all_pairs_ilp",
    "compute_all_pairs_data",
    "compute_min_max_cc_data",
    "compute_min_max_lp_data",
    "empty_pivot_results",
    "json_converter",
    "print_standard_results",
    "run_full_experiment",
    "run_pivot_multiple",
    "safe_ratio",
    "save_results_append",
    "solve_actual_all_pairs_formulation",
    "validate_signed_matrix",
]
