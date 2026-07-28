#!/usr/bin/env python3
"""Run the final clique-graph experiments and create a flat CSV.

Only the algorithms used for the clique results are executed:
- Pivot
- the standard all-pairs LP relaxation

The output is written to:

    results/research_tables/clique_runs_flat.csv

The file is updated atomically after every completed result. Existing complete
or edge-deleted results are skipped unless ``OVERWRITE_FINISHED`` is True.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src import experiment_helpers as h
from src.edge_deletion import delete_edges
from src.graph_generation import generate_clique_signed_graph


# =============================================================================
# Experiment settings
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/research_tables/clique_runs_flat.csv"

# Edit this list to select the clique configurations to run.
CLUSTER_SIZE_CASES = [
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
]

P_POS_INSIDE_VALUES = [0.9]
P_POS_BETWEEN_VALUES = [0.1]
P_DELETE_VALUES = [0.05, 0.15, 0.25, 0.40]

# The final data used 50 graph seeds for n <= 30 and 20 for n = 100.
DEFAULT_GRAPH_SEEDS = range(1, 51)
GRAPH_SEEDS_BY_N = {
    100: range(1, 21),
}

# Pivot is averaged over these internal randomized runs.
PIVOT_SEEDS = range(1, 11)

RUN_COMPLETE_GRAPHS = True
RUN_EDGE_DELETED_GRAPHS = True
OVERWRITE_FINISHED = False


# =============================================================================
# Flat-table columns and CSV helpers
# =============================================================================

REQUIRED_FIELDS = [
    "file_name",
    "file_path",
    "graph_family",
    "graph_type",
    "n",
    "seed",
    "p_delete",
    "p_pos_inside",
    "p_pos_between",
    "cluster_sizes",
    "complete_pivot_best_cost",
    "complete_pivot_average_cost",
    "complete_lp_cost",
    "complete_all_pairs_lp_runtime_seconds",
    "edge_num_edges_deleted",
    "edge_pivot_best_cost",
    "edge_pivot_average_cost",
    "edge_all_pairs_lp_cost",
    "edge_all_pairs_lp_runtime_seconds",
]

COMPLETE_RESULT_COLUMNS = [
    "complete_pivot_best_cost",
    "complete_pivot_average_cost",
    "complete_lp_cost",
    "complete_all_pairs_lp_runtime_seconds",
]

EDGE_RESULT_COLUMNS = [
    "edge_num_edges_deleted",
    "edge_pivot_best_cost",
    "edge_pivot_average_cost",
    "edge_all_pairs_lp_cost",
    "edge_all_pairs_lp_runtime_seconds",
]


def read_table() -> tuple[list[str], list[dict[str, str]]]:
    """Read an existing table, or initialize an empty table."""
    if not TABLE.exists():
        return list(REQUIRED_FIELDS), []

    with TABLE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])

        if not fields:
            raise ValueError(f"CSV has no header: {TABLE}")

        for field in REQUIRED_FIELDS:
            if field not in fields:
                fields.append(field)

        rows = [
            {field: source_row.get(field, "") for field in fields}
            for source_row in reader
        ]

    return fields, rows


def write_table(fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write the flat table atomically."""
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


def set_values(row: dict[str, str], values: dict[str, Any]) -> None:
    """Assign scalar values to one result row."""
    for column, output in values.items():
        if isinstance(output, np.generic):
            output = output.item()
        row[column] = "" if output is None else str(output)


def fields_are_filled(
    row: dict[str, str],
    columns: list[str],
) -> bool:
    return all(str(row.get(column, "")).strip() for column in columns)


def probability_key(value: Any) -> str:
    return f"{float(value):.12g}"


def row_key(
    file_name: str,
    seed: int,
    p_delete: float,
) -> tuple[str, int, str]:
    return file_name, int(seed), probability_key(p_delete)


# =============================================================================
# Clique graph metadata
# =============================================================================


def cluster_sizes_to_tag(cluster_sizes: list[int]) -> str:
    """Create a concise, stable identifier for one clique configuration."""
    if cluster_sizes and len(set(cluster_sizes)) == 1:
        return f"{len(cluster_sizes)}x{cluster_sizes[0]}"

    return "_".join(str(size) for size in cluster_sizes)


def graph_file_name(cluster_sizes: list[int]) -> str:
    n = sum(cluster_sizes)
    tag = cluster_sizes_to_tag(cluster_sizes)
    return f"clq_n{n}_{tag}.json"


def graph_seeds(n: int) -> list[int]:
    selected = GRAPH_SEEDS_BY_N.get(n, DEFAULT_GRAPH_SEEDS)
    return list(selected)


def new_result_row(
    fields: list[str],
    cluster_sizes: list[int],
    p_pos_inside: float,
    p_pos_between: float,
    seed: int,
    p_delete: float,
) -> dict[str, str]:
    """Create one empty flat-table row for an edge-deleted instance."""
    n = sum(cluster_sizes)
    file_name = graph_file_name(cluster_sizes)

    row = {field: "" for field in fields}
    set_values(
        row,
        {
            "file_name": file_name,
            "file_path": f"generated/clique/{file_name}",
            "graph_family": "clique",
            "graph_type": "clique",
            "n": n,
            "seed": seed,
            "p_delete": p_delete,
            "p_pos_inside": p_pos_inside,
            "p_pos_between": p_pos_between,
            "cluster_sizes": json.dumps(
                cluster_sizes,
                separators=(",", ":"),
            ),
        },
    )
    return row


def lp_runtime(result: dict[str, Any]) -> Any:
    info = result.get("lp_info")
    if isinstance(info, dict):
        return info.get("runtime_seconds")
    return None


def cleanup_solver_memory() -> None:
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

    index = {
        row_key(
            str(row.get("file_name", "")).strip(),
            int(float(row["seed"])),
            float(row["p_delete"]),
        ): row
        for row in rows
        if str(row.get("file_name", "")).strip()
        and str(row.get("seed", "")).strip()
        and str(row.get("p_delete", "")).strip()
    }

    print("Output:", TABLE)
    print("Algorithms: Pivot, all-pairs LP")
    print("Pivot seeds:", list(PIVOT_SEEDS))

    for cluster_sizes in CLUSTER_SIZE_CASES:
        n = sum(cluster_sizes)
        file_name = graph_file_name(cluster_sizes)

        for p_pos_inside in P_POS_INSIDE_VALUES:
            for p_pos_between in P_POS_BETWEEN_VALUES:
                for seed in graph_seeds(n):
                    print(
                        f"\n=== {file_name}: graph seed {seed} ==="
                    )

                    complete_matrix, _ = generate_clique_signed_graph(
                        cluster_sizes=cluster_sizes,
                        p_pos_inside=p_pos_inside,
                        p_pos_between=p_pos_between,
                        seed=seed,
                    )

                    graph_rows: list[dict[str, str]] = []

                    for p_delete in P_DELETE_VALUES:
                        key = row_key(file_name, seed, p_delete)
                        row = index.get(key)

                        if row is None:
                            row = new_result_row(
                                fields=fields,
                                cluster_sizes=cluster_sizes,
                                p_pos_inside=p_pos_inside,
                                p_pos_between=p_pos_between,
                                seed=seed,
                                p_delete=p_delete,
                            )
                            rows.append(row)
                            index[key] = row

                        graph_rows.append(row)

                    complete_finished = all(
                        fields_are_filled(row, COMPLETE_RESULT_COLUMNS)
                        for row in graph_rows
                    )

                    if RUN_COMPLETE_GRAPHS and (
                        OVERWRITE_FINISHED or not complete_finished
                    ):
                        print("run complete Pivot")
                        pivot_result = h.run_pivot_multiple(
                            complete_matrix,
                            PIVOT_SEEDS,
                        )

                        print("run complete all-pairs LP")
                        lp_result = h.compute_all_pairs_data(
                            complete_matrix,
                            compute_lp=True,
                            compute_ilp=False,
                        )

                        complete_values = {
                            "complete_pivot_best_cost": (
                                pivot_result["best_cost"]
                            ),
                            "complete_pivot_average_cost": (
                                pivot_result["average_cost"]
                            ),
                            "complete_lp_cost": lp_result["lp_cost"],
                            "complete_all_pairs_lp_runtime_seconds": (
                                lp_runtime(lp_result)
                            ),
                        }

                        for row in graph_rows:
                            set_values(row, complete_values)

                        write_table(fields, rows)
                        cleanup_solver_memory()
                    else:
                        print("skip complete Pivot and LP")

                    if RUN_EDGE_DELETED_GRAPHS:
                        for p_delete, row in zip(
                            P_DELETE_VALUES,
                            graph_rows,
                        ):
                            if (
                                fields_are_filled(row, EDGE_RESULT_COLUMNS)
                                and not OVERWRITE_FINISHED
                            ):
                                print(
                                    f"skip deleted p={p_delete}, "
                                    f"seed={seed}"
                                )
                                continue

                            deleted_matrix, num_edges_deleted = delete_edges(
                                complete_matrix.copy(),
                                p_delete,
                                seed,
                            )

                            print(
                                f"run deleted Pivot: p={p_delete}, "
                                f"seed={seed}"
                            )
                            pivot_result = h.run_pivot_multiple(
                                deleted_matrix,
                                PIVOT_SEEDS,
                            )

                            print(
                                f"run deleted all-pairs LP: "
                                f"p={p_delete}, seed={seed}"
                            )
                            lp_result = h.compute_all_pairs_data(
                                deleted_matrix,
                                compute_lp=True,
                                compute_ilp=False,
                            )

                            set_values(
                                row,
                                {
                                    "edge_num_edges_deleted": (
                                        num_edges_deleted
                                    ),
                                    "edge_pivot_best_cost": (
                                        pivot_result["best_cost"]
                                    ),
                                    "edge_pivot_average_cost": (
                                        pivot_result["average_cost"]
                                    ),
                                    "edge_all_pairs_lp_cost": (
                                        lp_result["lp_cost"]
                                    ),
                                    "edge_all_pairs_lp_runtime_seconds": (
                                        lp_runtime(lp_result)
                                    ),
                                },
                            )

                            write_table(fields, rows)
                            cleanup_solver_memory()

                    del complete_matrix
                    gc.collect()

    print("\nDone.")
    print("Rows in table:", len(rows))


if __name__ == "__main__":
    main()
