"""Shared, resumable experiment pipeline for all graph families."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from all_pairs_solver import solve_all_pairs
from bad_triangles import (
    count_bad_triangles,
    find_bad_triangles,
    find_edge_disjoint_bad_triangles_max,
    make_edge_to_triangle_map,
)
from edge_deletion import delete_edges
from ilp_solver import solve_ilp
from lp_formulations import solve_primal
from pivot import run_pivot
from cost import calculate_clustering_cost


P_DELETE_VALUES = (0.05, 0.15, 0.25, 0.40)
PIVOT_SEEDS = tuple(range(1, 101))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def atomic_json_dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=4, default=json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def format_runtime(seconds):
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_pivot_multiple(S, pivot_seeds=PIVOT_SEEDS):
    best_cost = None
    costs = []

    for pivot_seed in pivot_seeds:
        clusters, _ = run_pivot(S, pivot_seed)
        cost = calculate_clustering_cost(S, clusters)
        costs.append(cost)
        if best_cost is None or cost < best_cost:
            best_cost = cost

    return {
        "best_cost": best_cost,
        "average_cost": sum(costs) / len(costs),
    }


def bad_triangle_metrics(S):
    triangles = find_bad_triangles(S)
    edge_map = make_edge_to_triangle_map(triangles)
    packing = find_edge_disjoint_bad_triangles_max(copy.deepcopy(edge_map))
    return triangles, count_bad_triangles(packing)


def run_complete_graph(S, include_optimization=True):
    pivot = run_pivot_multiple(S)
    triangles, packing_count = bad_triangle_metrics(S)

    result = {
        "pivot": pivot,
        "bad_triangles": {
            "total_count": len(triangles),
            "max_edge_disjoint_count": packing_count,
        },
    }
    approximations = {
        "best_pivot_approximation": None,
        "average_pivot_approximation": None,
    }
    sparse_approximations = {}

    if include_optimization:
        ilp_cost, _, _ = solve_ilp(
            S, verbose=False, relax=False, add_four_cycles=False
        )
        lp_cost, _, _ = solve_ilp(
            S, verbose=False, relax=True, add_four_cycles=False
        )
        primal_cost, _ = solve_primal(S, triangles, verbose=False)

        result.update(
            {
                "bad_triangle_lp_bounds": {"primal_cost": primal_cost},
                "sparse_ilp": {"cost": ilp_cost},
                "sparse_lp_relaxation": {"cost": lp_cost},
            }
        )
        sparse_approximations["sparse_lp_to_ilp_ratio"] = safe_ratio(
            lp_cost, ilp_cost
        )

    return result, approximations, sparse_approximations


def run_edge_deleted_graph(
    S,
    p_delete,
    seed,
    include_optimization=True,
):
    started = time.time()
    S_new, num_edges_deleted = delete_edges(S, p_delete, seed)
    pivot = run_pivot_multiple(S_new)
    triangles, packing_count = bad_triangle_metrics(S_new)

    edge_result = {
        "num_edges_deleted": int(num_edges_deleted),
        "pivot": pivot,
        "bad_triangles": {
            "total_count": len(triangles),
            "max_edge_disjoint_count": packing_count,
        },
    }
    approximations = {}

    if include_optimization:
        sparse_ilp_no4, _, _ = solve_ilp(
            S_new, verbose=False, relax=False, add_four_cycles=False
        )
        sparse_ilp_with4, _, bad_4_cycles = solve_ilp(
            S_new, verbose=False, relax=False, add_four_cycles=True
        )
        sparse_lp_no4, _, _ = solve_ilp(
            S_new, verbose=False, relax=True, add_four_cycles=False
        )
        sparse_lp_with4, _, _ = solve_ilp(
            S_new, verbose=False, relax=True, add_four_cycles=True
        )
        primal_cost, _ = solve_primal(S_new, triangles, verbose=False)

        all_pairs_ilp, _, ilp_info = solve_all_pairs(
            S_new,
            verbose=False,
            relax=False,
            return_x_values=False,
        )
        if not ilp_info["is_optimal"]:
            raise RuntimeError(
                f"All-pairs ILP was not optimal: seed={seed}, "
                f"p_delete={p_delete}"
            )

        all_pairs_lp, _, lp_info = solve_all_pairs(
            S_new,
            verbose=False,
            relax=True,
            return_x_values=False,
        )
        if not lp_info["is_optimal"]:
            raise RuntimeError(
                f"All-pairs LP was not optimal: seed={seed}, "
                f"p_delete={p_delete}"
            )

        edge_result.update(
            {
                "bad_triangle_lp_bounds": {"primal_cost": primal_cost},
                "sparse_ilp": {
                    "without_4_cycles": {"cost": sparse_ilp_no4},
                    "with_4_cycles": {
                        "cost": sparse_ilp_with4,
                        "bad_4_cycles_count": len(bad_4_cycles),
                    },
                },
                "sparse_lp_relaxation": {
                    "without_4_cycles": {"cost": sparse_lp_no4},
                    "with_4_cycles": {"cost": sparse_lp_with4},
                },
                "all_pairs_ilp": {
                    "cost": all_pairs_ilp,
                    "optimal": ilp_info["is_optimal"],
                    "runtime_seconds": ilp_info["runtime_seconds"],
                    "mip_gap": ilp_info["mip_gap"],
                },
                "all_pairs_lp_relaxation": {
                    "cost": all_pairs_lp,
                    "optimal": lp_info["is_optimal"],
                    "runtime_seconds": lp_info["runtime_seconds"],
                },
            }
        )
        approximations.update(
            {
                "sparse_lp_to_ilp_ratio_with_4_cycles": safe_ratio(
                    sparse_lp_with4, sparse_ilp_with4
                ),
                "all_pairs_lp_to_ilp_ratio": safe_ratio(
                    all_pairs_lp, all_pairs_ilp
                ),
                "average_pivot_approximation_with_4_cycles": safe_ratio(
                    pivot["average_cost"], sparse_ilp_with4
                ),
                "average_pivot_approximation_without_4_cycles": safe_ratio(
                    pivot["average_cost"], sparse_ilp_no4
                ),
            }
        )

    return {
        "p_delete": p_delete,
        "edge_deleted_graph": edge_result,
        "approximations": approximations,
        "runtime_seconds": time.time() - started,
    }


def initialize_file(path, shared_params, seeds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open() as handle:
            existing = json.load(handle)
        existing_shared = existing.get("shared_graph_params", {})
        for key in ("graph_type", "num_nodes"):
            if (
                key in existing_shared
                and existing_shared[key] != shared_params.get(key)
            ):
                raise ValueError(
                    f"{path}: existing {key}={existing_shared[key]!r}, "
                    f"expected {shared_params.get(key)!r}"
                )
        if (
            "cluster_sizes" in shared_params
            and existing_shared.get("cluster_sizes")
            != shared_params["cluster_sizes"]
        ):
            raise ValueError(
                f"{path}: existing cluster_sizes do not match configuration"
            )
        return

    data = {
        "shared_graph_params": shared_params,
        "experiments": [
            {
                "graph_params": {"seed": seed},
                "p_delete_results": {},
            }
            for seed in seeds
        ],
    }
    atomic_json_dump(path, data)


def find_experiment(data, seed):
    for experiment in data["experiments"]:
        if int(experiment["graph_params"]["seed"]) == seed:
            return experiment
    experiment = {
        "graph_params": {"seed": seed},
        "p_delete_results": {},
    }
    data["experiments"].append(experiment)
    return experiment


def read_experiment(path, seed):
    with Path(path).open() as handle:
        data = json.load(handle)
    return data, find_experiment(data, seed)


def save_complete_result(
    path,
    seed,
    complete_graph,
    approximations,
    sparse_approximations,
):
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open() as handle:
                data = json.load(handle)
            experiment = find_experiment(data, seed)
            if "complete_graph" in experiment:
                return False
            experiment["complete_graph"] = complete_graph
            experiment["complete_graph_sparse_approximations"] = (
                sparse_approximations
            )
            experiment["approximations"] = {
                "complete_graph": approximations
            }
            atomic_json_dump(path, data)
            return True
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def save_p_delete_result(path, seed, p_delete, result):
    path = Path(path)
    key = f"{p_delete:.2f}"
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open() as handle:
                data = json.load(handle)
            experiment = find_experiment(data, seed)
            saved = experiment.setdefault("p_delete_results", {})
            if key in saved:
                return False
            saved[key] = result
            atomic_json_dump(path, data)
            return True
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def has_complete_result(path, seed):
    _, experiment = read_experiment(path, seed)
    return "complete_graph" in experiment


def has_p_delete_result(path, seed, p_delete):
    _, experiment = read_experiment(path, seed)
    return f"{p_delete:.2f}" in experiment.get("p_delete_results", {})


def missing_summary(path, seeds, p_delete_values):
    """Count missing work without creating or changing any files."""
    path = Path(path)
    if not path.exists():
        return {
            "complete_graphs": len(seeds),
            "p_delete_results": len(seeds) * len(p_delete_values),
        }

    with path.open() as handle:
        data = json.load(handle)
    experiments = {
        int(exp["graph_params"]["seed"]): exp
        for exp in data.get("experiments", [])
    }

    complete_missing = 0
    p_delete_missing = 0
    for seed in seeds:
        experiment = experiments.get(seed)
        if experiment is None or "complete_graph" not in experiment:
            complete_missing += 1
        saved = (
            experiment.get("p_delete_results", {})
            if experiment is not None
            else {}
        )
        for p_delete in p_delete_values:
            if f"{p_delete:.2f}" not in saved:
                p_delete_missing += 1

    return {
        "complete_graphs": complete_missing,
        "p_delete_results": p_delete_missing,
    }
