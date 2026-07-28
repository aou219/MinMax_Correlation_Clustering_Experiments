#!/usr/bin/env python3
"""Run the final Facebook experiments and update the flat results table.

Algorithms:
- Pivot
- standard all-pairs LP relaxation
- MinMaxCC
- MinMaxLP with rounding

The script updates the CSV atomically after every completed algorithm, so an
interrupted run can be restarted safely. Existing completed values are skipped
unless ``OVERWRITE_FINISHED`` is set to True.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src import experiment_helpers as h
from src.edge_deletion import delete_edges
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)


# =============================================================================
# Experiment settings
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"

EGO_IDS = ["3980", "698", "414", "686"]
P_DELETE_VALUES = [0.05, 0.15, 0.25, 0.40]
SEEDS = range(1, 31)
PIVOT_SEEDS = range(1, 101)

# Final paper algorithms.
RUN_PIVOT = True
RUN_NORMAL_LP = True
RUN_MINMAX_CC = True
RUN_MINMAX_LP = True  # Includes the MinMaxLP rounding algorithm.

RUN_COMPLETE_GRAPHS = True
RUN_EDGE_DELETED_GRAPHS = True
OVERWRITE_FINISHED = False

D_HAT = 8
LAMBDA = 5
MINMAX_LP_R = 0.4
MINMAX_LP_R2 = 0.4
MINMAX_LP_METHOD = 2


# =============================================================================
# CSV helpers
# =============================================================================

CLUSTER_COLUMNS = [
    "complete_min_max_lp_clustering_json",
    "edge_min_max_lp_clustering_json",
]


def read_table() -> tuple[list[str], list[dict[str, str]]]:
    """Read the existing Facebook grid table."""
    if not TABLE.exists():
        raise FileNotFoundError(
            f"Results table not found: {TABLE}\n"
            "Create the flat Facebook grid table before running this script."
        )

    with TABLE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])

        if not fields:
            raise ValueError(f"CSV has no header: {TABLE}")

        for column in CLUSTER_COLUMNS:
            if column not in fields:
                fields.append(column)

        rows = [
            {field: source_row.get(field, "") for field in fields}
            for source_row in reader
        ]

    return fields, rows


def write_table(fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write the table atomically."""
    TABLE.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=TABLE.name + ".",
        suffix=".tmp",
        dir=TABLE.parent,
        text=True,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary_name, TABLE)

    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def fields_are_filled(
    rows: list[dict[str, str]],
    columns: list[str],
) -> bool:
    """Return True when every requested cell contains a value."""
    return bool(rows) and all(
        str(row.get(column, "")).strip()
        for row in rows
        for column in columns
    )


def set_values(
    rows: list[dict[str, str]],
    values: dict[str, Any],
) -> None:
    """Assign values to one or more result rows."""
    for row in rows:
        for column, output in values.items():
            if isinstance(output, np.generic):
                output = output.item()
            row[column] = "" if output is None else str(output)


# =============================================================================
# Facebook graph construction
# =============================================================================


def locate_edges_file(ego_id: str) -> Path:
    """Locate the .edges file for one Facebook ego graph."""
    candidates = [
        ROOT / f"data/facebook/{ego_id}.edges",
        ROOT / f"data/facebook/facebook_3/{ego_id}.edges",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No .edges file found for Facebook ego {ego_id}."
    )


def build_complete_graph(
    ego_id: str,
) -> tuple[np.ndarray, dict[int, Any]]:
    """Build the complete signed matrix using only .edges endpoints."""
    nodes, edges = load_facebook_ego_edges(
        str(locate_edges_file(ego_id))
    )

    matrix, node_to_index, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            sorted(nodes),
            edges,
        )
    )

    index_to_node = {
        int(index): node
        for node, index in node_to_index.items()
    }

    return matrix, index_to_node


def clustering_to_json(
    clustering: list[Any],
    index_to_node: dict[int, Any],
) -> str:
    """Translate internal vertex indices back to Facebook node IDs."""
    clusters = [
        [index_to_node[int(vertex)] for vertex in cluster]
        for cluster in clustering
    ]
    return json.dumps(
        clusters,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# =============================================================================
# Algorithms
# =============================================================================


def run_pivot(matrix: np.ndarray) -> dict[str, Any]:
    return h.run_pivot_multiple(matrix, PIVOT_SEEDS)


def run_normal_lp(matrix: np.ndarray) -> dict[str, Any]:
    return h.compute_all_pairs_data(
        matrix,
        compute_lp=True,
        compute_ilp=False,
    )


def run_minmax_cc(matrix: np.ndarray) -> dict[str, Any]:
    return h.compute_min_max_cc_data(
        matrix,
        compute_min_max=True,
        param_1=D_HAT,
        param_2=LAMBDA,
    )


def run_minmax_lp(matrix: np.ndarray) -> dict[str, Any]:
    result = h.compute_min_max_lp_data(
        matrix,
        compute_min_max_lp=True,
        r=MINMAX_LP_R,
        r2=MINMAX_LP_R2,
        method=MINMAX_LP_METHOD,
        norm=np.inf,
    )

    required = [
        "lp_cost",
        "clustering",
        "cluster_count",
        "rounding_cost",
        "max_disagreement_vertex",
        "lp_runtime_seconds",
        "rounding_runtime_seconds",
        "total_runtime_seconds",
    ]
    missing = [key for key in required if result.get(key) is None]

    if missing:
        raise RuntimeError(
            "Missing MinMaxLP output: " + ", ".join(missing)
        )

    return result


AlgorithmFunction = Callable[[np.ndarray], dict[str, Any]]

ALGORITHMS: list[tuple[str, bool, AlgorithmFunction]] = [
    ("pivot", RUN_PIVOT, run_pivot),
    ("normal_lp", RUN_NORMAL_LP, run_normal_lp),
    ("minmax_cc", RUN_MINMAX_CC, run_minmax_cc),
    ("minmax_lp", RUN_MINMAX_LP, run_minmax_lp),
]


def output_columns(name: str, scope: str) -> list[str]:
    """Return the CSV columns produced by one algorithm."""
    if name == "pivot":
        return [
            f"{scope}_pivot_best_cost",
            f"{scope}_pivot_average_cost",
        ]

    if name == "normal_lp":
        return [
            "complete_lp_cost"
            if scope == "complete"
            else "edge_all_pairs_lp_cost"
        ]

    if name == "minmax_cc":
        prefix = f"{scope}_min_max_cc"
        return [
            f"{prefix}_computed",
            f"{prefix}_cluster_count",
            f"{prefix}_max_disagreement",
            f"{prefix}_d_hat",
            f"{prefix}_lambda",
            f"{prefix}_runtime_seconds",
        ]

    prefix = f"{scope}_min_max_lp"
    return [
        f"{prefix}_computed",
        f"{prefix}_cost",
        f"{prefix}_rounding_cost",
        f"{prefix}_max_disagreement_vertex",
        f"{prefix}_cluster_count",
        f"{prefix}_clustering_json",
        f"{prefix}_r",
        f"{prefix}_r2",
        f"{prefix}_method",
        f"{prefix}_norm",
        f"{prefix}_runtime_seconds",
        f"{prefix}_rounding_runtime_seconds",
        f"{prefix}_total_runtime_seconds",
    ]


def save_result(
    name: str,
    scope: str,
    rows: list[dict[str, str]],
    result: dict[str, Any],
    index_to_node: dict[int, Any],
) -> None:
    """Map one algorithm result to the Facebook flat-table columns."""
    if name == "pivot":
        values = {
            f"{scope}_pivot_best_cost": result["best_cost"],
            f"{scope}_pivot_average_cost": result["average_cost"],
        }

    elif name == "normal_lp":
        column = (
            "complete_lp_cost"
            if scope == "complete"
            else "edge_all_pairs_lp_cost"
        )
        values = {column: result["lp_cost"]}

    elif name == "minmax_cc":
        prefix = f"{scope}_min_max_cc"
        values = {
            f"{prefix}_computed": result["computed"],
            f"{prefix}_cluster_count": result["cluster_count"],
            f"{prefix}_max_disagreement": result["max_disagreement"],
            f"{prefix}_d_hat": result["d_hat"],
            f"{prefix}_lambda": result["lambda"],
            f"{prefix}_runtime_seconds": result["runtime_seconds"],
        }

    else:
        prefix = f"{scope}_min_max_lp"
        values = {
            f"{prefix}_computed": result["computed"],
            f"{prefix}_cost": result["lp_cost"],
            f"{prefix}_rounding_cost": result["rounding_cost"],
            f"{prefix}_max_disagreement_vertex": (
                result["max_disagreement_vertex"]
            ),
            f"{prefix}_cluster_count": result["cluster_count"],
            f"{prefix}_clustering_json": clustering_to_json(
                result["clustering"],
                index_to_node,
            ),
            f"{prefix}_r": result["r"],
            f"{prefix}_r2": result["r2"],
            f"{prefix}_method": result["method"],
            f"{prefix}_norm": result["norm"],
            f"{prefix}_runtime_seconds": result["lp_runtime_seconds"],
            f"{prefix}_rounding_runtime_seconds": (
                result["rounding_runtime_seconds"]
            ),
            f"{prefix}_total_runtime_seconds": (
                result["total_runtime_seconds"]
            ),
        }

    set_values(rows, values)


def cleanup_solver_memory() -> None:
    """Release Python and Gurobi memory between solver calls."""
    gc.collect()

    try:
        import gurobipy as gp

        gp.disposeDefaultEnv()
    except Exception:
        pass


# =============================================================================
# Main experiment loop
# =============================================================================


def main() -> None:
    fields, rows = read_table()

    enabled = [
        (name, function)
        for name, is_enabled, function in ALGORITHMS
        if is_enabled
    ]

    print("Table:", TABLE)
    print("Enabled algorithms:", ", ".join(name for name, _ in enabled))
    print("Facebook egos:", ", ".join(EGO_IDS))

    for ego_id in EGO_IDS:
        print(f"\n=== Facebook ego {ego_id} ===")

        complete_matrix, index_to_node = build_complete_graph(ego_id)
        ego_rows = [
            row
            for row in rows
            if str(row.get("ego_id", "")).strip() == ego_id
        ]

        if not ego_rows:
            raise ValueError(f"No table rows found for ego {ego_id}.")

        for row in ego_rows:
            row["n"] = str(len(complete_matrix))

        if RUN_COMPLETE_GRAPHS:
            for name, function in enabled:
                columns = output_columns(name, "complete")

                if (
                    fields_are_filled(ego_rows, columns)
                    and not OVERWRITE_FINISHED
                ):
                    print("skip complete", name)
                    continue

                print("run complete", name)
                result = function(complete_matrix)
                save_result(
                    name,
                    "complete",
                    ego_rows,
                    result,
                    index_to_node,
                )
                write_table(fields, rows)
                cleanup_solver_memory()

        if RUN_EDGE_DELETED_GRAPHS:
            for p_delete in P_DELETE_VALUES:
                for seed in SEEDS:
                    matches = [
                        row
                        for row in ego_rows
                        if abs(
                            float(row["p_delete"]) - p_delete
                        ) < 1e-12
                        and int(float(row["seed"])) == seed
                    ]

                    if len(matches) != 1:
                        raise ValueError(
                            "Missing or duplicate table row for "
                            f"ego={ego_id}, p_delete={p_delete}, "
                            f"seed={seed}."
                        )

                    target = matches[0]
                    deleted_matrix: np.ndarray | None = None

                    for name, function in enabled:
                        columns = output_columns(name, "edge")

                        if (
                            fields_are_filled([target], columns)
                            and not OVERWRITE_FINISHED
                        ):
                            continue

                        if deleted_matrix is None:
                            deleted_matrix, _ = delete_edges(
                                complete_matrix.copy(),
                                p_delete,
                                seed,
                            )

                        print(
                            f"run {name}: ego={ego_id}, "
                            f"p_delete={p_delete}, seed={seed}"
                        )
                        result = function(deleted_matrix)
                        save_result(
                            name,
                            "edge",
                            [target],
                            result,
                            index_to_node,
                        )
                        write_table(fields, rows)
                        cleanup_solver_memory()

        del complete_matrix
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
