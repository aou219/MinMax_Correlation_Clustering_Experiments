#!/usr/bin/env python3
"""Shared helpers for the current correlation-clustering experiments."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

try:
    from .normal_lp import solve_normal_lp
    from .cost import calculate_clustering_cost
    from .edge_deletion import delete_edges
    from .min_max import max_disagreement, min_max_cc
    from .min_max_lp import DegreeDist, LocalObj, MinMaxLP, cluster
    from .pivot import run_pivot
except ImportError:
    from normal_lp import solve_normal_lp
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


def validate_signed_matrix(S: np.ndarray) -> np.ndarray:
    """Validate a signed adjacency matrix."""
    matrix = np.asarray(S)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("S must be square.")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("S must be symmetric.")
    if not np.all(np.diag(matrix) == 0):
        raise ValueError("The diagonal of S must be zero.")
    if not np.all(np.isin(matrix, (-1, 0, 1))):
        raise ValueError("S may contain only -1, 0, and 1.")

    return matrix


def run_pivot_multiple(
    S: np.ndarray,
    pivot_seeds: Iterable[int],
) -> dict[str, Any]:
    """Run Pivot for every supplied seed."""
    matrix = validate_signed_matrix(S)
    runs = []

    for pivot_seed in pivot_seeds:
        start = time.perf_counter()
        clusters, _ = run_pivot(matrix, pivot_seed)

        runs.append({
            "cost": calculate_clustering_cost(
                matrix,
                clusters,
            ),
            "runtime_seconds": (
                time.perf_counter() - start
            ),
        })

    if not runs:
        return {
            "best_cost": None,
            "average_cost": None,
            "average_runtime_seconds": None,
        }

    return {
        "best_cost": min(run["cost"] for run in runs),
        "average_cost": (
            sum(run["cost"] for run in runs) / len(runs)
        ),
        "average_runtime_seconds": (
            sum(run["runtime_seconds"] for run in runs)
            / len(runs)
        ),
    }


def compute_normal_lp_data(
    S: np.ndarray,
    compute_normal_lp: bool = True,
    time_limit: float | None = None,
) -> dict[str, Any] | None:
    """Run the normal correlation-clustering LP."""
    if not compute_normal_lp:
        return None

    cost, info = solve_normal_lp(
        validate_signed_matrix(S),
        time_limit=time_limit,
        verbose=False,
    )

    return {
        "cost": cost,
        "info": info,
    }


def compute_min_max_cc_data(
    S: np.ndarray,
    compute_min_max: bool = True,
    d_hat: int = DEFAULT_D_HAT,
    lambda_value: int = DEFAULT_LAMBDA,
) -> dict[str, Any] | None:
    """Run MinMaxCC."""
    if not compute_min_max:
        return None
    if d_hat < 0:
        raise ValueError("d_hat must be non-negative.")
    if lambda_value <= 4:
        raise ValueError("lambda must be greater than 4.")

    matrix = validate_signed_matrix(S)
    start = time.perf_counter()
    clustering = min_max_cc(
        matrix,
        d_hat,
        lambda_value,
    )

    if clustering is None:
        raise RuntimeError("min_max_cc returned None.")

    return {
        "cluster_count": len(clustering),
        "max_disagreement": max_disagreement(
            clustering,
            matrix,
        ),
        "d_hat": d_hat,
        "lambda": lambda_value,
        "runtime_seconds": round(
            time.perf_counter() - start,
            6,
        ),
    }


def compute_min_max_lp_data(
    S: np.ndarray,
    compute_min_max_lp: bool = True,
    r: float = DEFAULT_MIN_MAX_LP_R,
    r2: float = DEFAULT_MIN_MAX_LP_R2,
    method: int = DEFAULT_MIN_MAX_LP_METHOD,
    norm: Any = np.inf,
) -> dict[str, Any] | None:
    """Solve MinMaxLP and run its rounding algorithm."""
    if not compute_min_max_lp:
        return None

    matrix = validate_signed_matrix(S)

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

    (
        _,
        rounding_cost,
        max_disagreement_vertex,
    ) = LocalObj(
        matrix,
        clustering,
        DegreeDist(matrix),
        norm,
    )

    lp_runtime = float(lp_runtime)
    rounding_runtime = float(rounding_runtime)

    return {
        "lp_cost": float(lp_cost),
        "rounding_cost": float(rounding_cost),
        "cluster_count": len(clustering),
        "max_disagreement_vertex": int(
            max_disagreement_vertex
        ),
        "r": r,
        "r2": r2,
        "method": method,
        "norm": "inf" if norm == np.inf else norm,
        "lp_runtime_seconds": round(lp_runtime, 6),
        "rounding_runtime_seconds": round(
            rounding_runtime,
            6,
        ),
        "total_runtime_seconds": round(
            lp_runtime + rounding_runtime,
            6,
        ),
    }


def run_full_experiment(
    S: np.ndarray,
    p_delete: float,
    seed: int,
    pivot_seeds: Iterable[int],
    compute_pivot: bool = False,
    compute_normal_lp: bool = True,
    compute_min_max: bool = True,
    compute_min_max_lp: bool = True,
    min_max_cc_param_1: int = DEFAULT_D_HAT,
    min_max_cc_param_2: int = DEFAULT_LAMBDA,
    min_max_lp_r: float = DEFAULT_MIN_MAX_LP_R,
    min_max_lp_r2: float = DEFAULT_MIN_MAX_LP_R2,
    min_max_lp_method: int = DEFAULT_MIN_MAX_LP_METHOD,
    min_max_lp_norm: Any = np.inf,
    normal_lp_time_limit: float | None = None,
) -> dict[str, Any]:
    """Run all selected algorithms on both graph versions."""
    matrix = validate_signed_matrix(S)

    if not 0 <= p_delete <= 1:
        raise ValueError(
            "p_delete must be between 0 and 1."
        )

    total_start = time.perf_counter()

    complete_pivot = (
        run_pivot_multiple(matrix, pivot_seeds)
        if compute_pivot
        else None
    )
    complete_normal_lp = compute_normal_lp_data(
        matrix,
        compute_normal_lp=compute_normal_lp,
        time_limit=normal_lp_time_limit,
    )
    complete_min_max_cc = compute_min_max_cc_data(
        matrix,
        compute_min_max=compute_min_max,
        d_hat=min_max_cc_param_1,
        lambda_value=min_max_cc_param_2,
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

    edge_pivot = (
        run_pivot_multiple(edge_matrix, pivot_seeds)
        if compute_pivot
        else None
    )
    edge_normal_lp = compute_normal_lp_data(
        edge_matrix,
        compute_normal_lp=compute_normal_lp,
        time_limit=normal_lp_time_limit,
    )
    edge_min_max_cc = compute_min_max_cc_data(
        edge_matrix,
        compute_min_max=compute_min_max,
        d_hat=min_max_cc_param_1,
        lambda_value=min_max_cc_param_2,
    )
    edge_min_max_lp = compute_min_max_lp_data(
        edge_matrix,
        compute_min_max_lp=compute_min_max_lp,
        r=min_max_lp_r,
        r2=min_max_lp_r2,
        method=min_max_lp_method,
        norm=min_max_lp_norm,
    )

    complete_ratio = None
    if complete_min_max_cc and complete_min_max_lp:
        denominator = complete_min_max_lp["lp_cost"]
        if denominator != 0:
            complete_ratio = (
                complete_min_max_cc["max_disagreement"]
                / denominator
            )

    edge_ratio = None
    if edge_min_max_cc and edge_min_max_lp:
        denominator = edge_min_max_lp["lp_cost"]
        if denominator != 0:
            edge_ratio = (
                edge_min_max_cc["max_disagreement"]
                / denominator
            )

    return {
        "num_edges_deleted": int(num_edges_deleted),

        "pivot_results": complete_pivot,
        "normal_lp_cost": (
            complete_normal_lp["cost"]
            if complete_normal_lp
            else None
        ),
        "normal_lp_info": (
            complete_normal_lp["info"]
            if complete_normal_lp
            else None
        ),
        "min_max_cc_results": complete_min_max_cc,
        "min_max_lp_results": complete_min_max_lp,
        "complete_min_max_approximation_ratio": (
            complete_ratio
        ),

        "pivot_results_new": edge_pivot,
        "normal_lp_cost_new": (
            edge_normal_lp["cost"]
            if edge_normal_lp
            else None
        ),
        "normal_lp_info_new": (
            edge_normal_lp["info"]
            if edge_normal_lp
            else None
        ),
        "min_max_cc_results_new": edge_min_max_cc,
        "min_max_lp_results_new": edge_min_max_lp,
        "edge_min_max_approximation_ratio": edge_ratio,

        "total_runtime": round(
            time.perf_counter() - total_start,
            6,
        ),
    }
