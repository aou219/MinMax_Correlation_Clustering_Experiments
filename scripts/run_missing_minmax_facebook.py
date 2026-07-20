#!/usr/bin/env python3
"""
Recompute only MinMaxCC for every Facebook row already present in the grid.

- Uses only nodes occurring in each .edges file.
- Does not load or add circle-only metadata nodes.
- Runs only MinMaxCC.
- Overwrites existing rows in place; it does not append rows.
- Preserves LP, ILP, Pivot, and all unrelated columns.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.edge_deletion import delete_edges
from src.experiment_helpers import compute_min_max_cc_data
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)

DEFAULT_GRID = (
    REPO_ROOT
    / "results/processed/research_tables/minmax_facebook_grid_runs_flat.csv"
)


def parse_int_spec(raw: str) -> list[int]:
    values: list[int] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start, end = map(int, part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    return sorted(set(values))


def parse_float_list(raw: str) -> list[float]:
    return sorted(
        set(
            float(value.strip())
            for value in raw.split(",")
            if value.strip()
        )
    )


def f8(value: Any) -> str:
    return f"{float(value):.8f}"


def locate_edges_file(ego_id: str) -> Path:
    candidates = (
        REPO_ROOT / f"data/facebook/{ego_id}.edges",
        REPO_ROOT / f"data/facebook/facebook_3/{ego_id}.edges",
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing Facebook .edges file for ego {ego_id}"
    )


def reconstruct_complete_graph(ego_id: str) -> np.ndarray:
    """Use only unique endpoints from the Facebook .edges file."""
    edge_nodes, facebook_edges = load_facebook_ego_edges(
        str(locate_edges_file(ego_id))
    )

    nodes = sorted(edge_nodes)

    matrix, _, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            nodes,
            facebook_edges,
        )
    )

    return matrix


def atomic_write(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    fd, temporary_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary_path, path)

    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def put_cc(
    row: dict[str, str],
    prefix: str,
    result: dict[str, Any],
    d_hat: int,
    lambda_value: int,
) -> None:
    row[f"{prefix}_min_max_cc_computed"] = str(
        result["max_disagreement"] is not None
    )
    row[f"{prefix}_min_max_cc_cluster_count"] = str(
        result["cluster_count"]
    )
    row[f"{prefix}_min_max_cc_max_disagreement"] = str(
        result["max_disagreement"]
    )
    row[f"{prefix}_min_max_cc_d_hat"] = str(d_hat)
    row[f"{prefix}_min_max_cc_lambda"] = str(lambda_value)
    row[f"{prefix}_min_max_cc_runtime_seconds"] = str(
        result["runtime_seconds"]
    )


def ensure_cc_columns(fieldnames: list[str]) -> None:
    required = []

    for prefix in ("complete", "edge"):
        required.extend([
            f"{prefix}_min_max_cc_computed",
            f"{prefix}_min_max_cc_cluster_count",
            f"{prefix}_min_max_cc_max_disagreement",
            f"{prefix}_min_max_cc_d_hat",
            f"{prefix}_min_max_cc_lambda",
            f"{prefix}_min_max_cc_runtime_seconds",
        ])

    missing = [name for name in required if name not in fieldnames]

    if missing:
        raise ValueError(
            "The output table is missing MinMaxCC columns: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--d-hat", type=int, default=8)
    parser.add_argument("--lambda-value", type=int, default=5)
    parser.add_argument("--seeds", default="1-30")
    parser.add_argument(
        "--p-delete-values",
        default="0.05,0.15,0.25,0.4",
    )
    parser.add_argument(
        "--ego-order",
        default="",
        help=(
            "Comma-separated ego IDs. When omitted, all ego IDs "
            "already present in the grid are used."
        ),
    )
    parser.add_argument("--no-backup", action="store_true")

    args = parser.parse_args()

    if args.lambda_value <= 4:
        raise ValueError("For q=0, lambda must be > 4.")

    grid = (
        args.grid
        if args.grid.is_absolute()
        else REPO_ROOT / args.grid
    )

    seeds = parse_int_spec(args.seeds)
    p_delete_values = parse_float_list(args.p_delete_values)

    with grid.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"Grid contains no rows: {grid}")

    ensure_cc_columns(fieldnames)

    grid_ego_ids = sorted(
        {
            str(row.get("ego_id", "")).strip()
            for row in rows
            if str(row.get("ego_id", "")).strip()
        },
        key=lambda value: int(float(value)),
    )

    if args.ego_order.strip():
        ego_ids = [
            value.strip()
            for value in args.ego_order.split(",")
            if value.strip()
        ]
    else:
        ego_ids = grid_ego_ids

    row_indices: dict[tuple[str, str, int], list[int]] = defaultdict(list)

    for index, row in enumerate(rows):
        ego_id = str(row.get("ego_id", "")).strip()
        p_delete = str(row.get("p_delete", "")).strip()
        seed_raw = str(row.get("seed", "1")).strip() or "1"

        if not ego_id or not p_delete:
            continue

        row_indices[(ego_id, f8(p_delete), int(float(seed_raw)))].append(index)

    missing_combinations = []

    for ego_id in ego_ids:
        for p_delete in p_delete_values:
            for seed in seeds:
                combination = (ego_id, f8(p_delete), seed)
                if combination not in row_indices:
                    missing_combinations.append(combination)

    if missing_combinations:
        preview = ", ".join(
            f"(ego={ego}, p={p_value}, seed={seed})"
            for ego, p_value, seed in missing_combinations[:10]
        )
        raise ValueError(
            f"{len(missing_combinations)} requested combinations are "
            f"missing from the grid. First missing: {preview}. "
            "This script only overwrites existing rows and does not append."
        )

    if not args.no_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = grid.with_name(
            f"{grid.stem}_before_corrected_minmaxcc_"
            f"{timestamp}{grid.suffix}"
        )
        shutil.copy2(grid, backup)
        print("Backup:", backup)

    total_combinations = len(ego_ids) * len(p_delete_values) * len(seeds)
    completed = 0

    for ego_id in ego_ids:
        print(f"\nReconstructing FB {ego_id} from .edges nodes only")
        complete_graph = reconstruct_complete_graph(ego_id)

        print(
            f"FB {ego_id}: n={complete_graph.shape[0]}, "
            "complete MinMaxCC"
        )
        complete_cc = compute_min_max_cc_data(
            complete_graph,
            True,
            args.d_hat,
            args.lambda_value,
        )

        for p_delete in p_delete_values:
            for seed in seeds:
                edge_deleted_graph, deleted_count = delete_edges(
                    complete_graph,
                    p_delete,
                    seed,
                )

                edge_cc = compute_min_max_cc_data(
                    edge_deleted_graph,
                    True,
                    args.d_hat,
                    args.lambda_value,
                )

                combination = (ego_id, f8(p_delete), seed)

                for index in row_indices[combination]:
                    put_cc(
                        rows[index],
                        "complete",
                        complete_cc,
                        args.d_hat,
                        args.lambda_value,
                    )
                    put_cc(
                        rows[index],
                        "edge",
                        edge_cc,
                        args.d_hat,
                        args.lambda_value,
                    )
                    rows[index]["n"] = str(complete_graph.shape[0])

                completed += 1

                print(
                    f"ego={ego_id} p_delete={p_delete} seed={seed} "
                    f"deleted={deleted_count} "
                    f"edge_CC={edge_cc['max_disagreement']} "
                    f"[{completed}/{total_combinations}]"
                )

                atomic_write(grid, fieldnames, rows)

    print("\nFinished.")
    print("Grid rows preserved:", len(rows))
    print("Recomputed combinations:", completed)
    print("Output:", grid)
    print(
        "Changed only n and complete_/edge_ MinMaxCC columns. "
        "LP, ILP, Pivot, and all other values were preserved."
    )


if __name__ == "__main__":
    main()
