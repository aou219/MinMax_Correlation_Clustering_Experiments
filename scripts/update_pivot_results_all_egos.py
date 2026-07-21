#!/usr/bin/env python3
"""
Rerun only Pivot and update the existing flat Facebook experiment table.

This script does NOT run:
- the all-pairs LP,
- the all-pairs ILP,
- MinMaxLP,
- MinMaxCC.

It updates only these four columns:

    complete_pivot_best_cost
    complete_pivot_average_cost
    edge_pivot_best_cost
    edge_pivot_average_cost

The remaining columns are preserved exactly as strings.

Default table:
    results/research_tables/minmax_facebook_grid_runs_flat.csv

Reproducibility choices:
- Facebook vertices are only endpoints that occur in each .edges file.
- Node IDs are sorted before matrix construction.
- Each table row uses its own p_delete and deletion seed.
- Pivot uses seeds 1 through 100 by default.
- Pivot chooses from sorted active nodes, so set iteration order cannot affect
  the result.
- The script creates an automatic backup before the first run.
- The table is saved atomically after every completed edge-deleted instance.
- A progress JSON file makes interrupted runs resumable.

By default, the script runs Pivot for every ego ID occurring in the table.
An existing progress file may contain only a subset of those ego IDs; the
script expands that progress safely and preserves already completed runs.

Typical use:
    python scripts/update_pivot_results.py

Restart the Pivot rerun from the beginning:
    python scripts/update_pivot_results.py --restart
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Establish deterministic environment before importing NumPy.
# ---------------------------------------------------------------------------

import os
import sys


DETERMINISTIC_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

if any(
    os.environ.get(name) != value
    for name, value in DETERMINISTIC_ENV.items()
):
    environment = os.environ.copy()
    environment.update(DETERMINISTIC_ENV)
    os.execvpe(
        sys.executable,
        [sys.executable, *sys.argv],
        environment,
    )


import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "2.0.0"
RESULT_DECIMALS = 8

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


import src.experiment_helpers as experiment_helpers
from src.edge_deletion import delete_edges
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)


DEFAULT_TABLE = (
    REPO_ROOT
    / "results/research_tables/"
    "minmax_facebook_grid_runs_flat.csv"
)

FALLBACK_PIVOT_EGOS = [
    "0", "107", "348", "414", "686",
    "698", "1684", "1912", "3437", "3980",
]

PIVOT_COLUMNS = [
    "complete_pivot_best_cost",
    "complete_pivot_average_cost",
    "edge_pivot_best_cost",
    "edge_pivot_average_cost",
]

REQUIRED_COLUMNS = {
    "ego_id",
    "n",
    "seed",
    "p_delete",
    *PIVOT_COLUMNS,
}


# ---------------------------------------------------------------------------
# Deterministic Pivot
# ---------------------------------------------------------------------------

def deterministic_run_pivot(
    signed_matrix: np.ndarray,
    seed: int | None = None,
):
    """
    Run Pivot with stable active-node ordering.

    The clustering rule is unchanged. Only the ordering supplied to the random
    generator is stabilized.
    """
    rng = np.random.default_rng(seed)
    active_nodes = set(range(signed_matrix.shape[0]))

    clusters = []
    pivots = []

    while active_nodes:
        ordered_active = sorted(active_nodes)

        pivot = int(
            rng.choice(
                np.asarray(ordered_active, dtype=int)
            )
        )

        cluster = {pivot}

        for vertex in ordered_active:
            if (
                vertex != pivot
                and signed_matrix[pivot, vertex] == 1
            ):
                cluster.add(vertex)

        clusters.append(cluster)
        pivots.append(pivot)
        active_nodes.difference_update(cluster)

    return clusters, pivots


def install_deterministic_pivot() -> None:
    """
    Make experiment_helpers.run_pivot_multiple use the deterministic Pivot.
    """
    experiment_helpers.run_pivot = deterministic_run_pivot


# ---------------------------------------------------------------------------
# CLI and parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only Pivot and update the existing Facebook result table."
        )
    )

    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
    )
    parser.add_argument(
        "--ego-ids",
        default=None,
        help=(
            "Optional comma-separated ego IDs. When omitted, ego IDs with "
            "existing Pivot values are detected automatically."
        ),
    )
    parser.add_argument(
        "--pivot-seeds",
        default="1-100",
        help=(
            "Pivot seed specification, for example 1-100 or 1,2,3."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Discard Pivot progress and rerun every selected row."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most this many new edge-deleted instances."
        ),
    )

    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_csv_list(raw: str) -> list[str]:
    return [
        value.strip()
        for value in raw.split(",")
        if value.strip()
    ]


def parse_integer_spec(raw: str) -> list[int]:
    values: list[int] = []

    for part in raw.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)

            if end < start:
                raise ValueError(
                    f"Invalid integer range: {part}"
                )

            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    result = sorted(set(values))

    if not result:
        raise ValueError(
            "At least one Pivot seed is required."
        )

    return result


def normalized_probability(value: Any) -> str:
    return f"{float(value):.8f}"


def instance_key(
    ego_id: Any,
    p_delete: Any,
    seed: Any,
) -> str:
    return "|".join(
        [
            str(ego_id).strip(),
            normalized_probability(p_delete),
            str(int(float(seed))),
        ]
    )


def canonical_number(
    value: Any,
    decimals: int = RESULT_DECIMALS,
) -> str:
    if value is None:
        return ""

    number = float(value)

    if math.isnan(number):
        return ""

    if math.isinf(number):
        return "inf" if number > 0 else "-inf"

    rounded = round(number, decimals)

    if (
        abs(rounded - round(rounded))
        < 10 ** (-decimals)
    ):
        return str(int(round(rounded)))

    return (
        f"{rounded:.{decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )


# ---------------------------------------------------------------------------
# Files and graph construction
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_matrix(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(matrix)

    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(repr(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))

    return digest.hexdigest()


def locate_edges_file(ego_id: str) -> Path:
    candidates = [
        REPO_ROOT / f"data/facebook/{ego_id}.edges",
        REPO_ROOT / f"data/facebook/facebook_3/{ego_id}.edges",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No .edges file found for ego {ego_id}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def construct_complete_matrix(
    ego_id: str,
) -> tuple[np.ndarray, Path]:
    edge_file = locate_edges_file(ego_id)

    edge_nodes, facebook_edges = load_facebook_ego_edges(
        str(edge_file)
    )

    # Corrected preprocessing: only nodes occurring in the edge file.
    ordered_nodes = sorted(edge_nodes)

    matrix, _, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            ordered_nodes,
            facebook_edges,
        )
    )

    return matrix, edge_file


# ---------------------------------------------------------------------------
# Table I/O
# ---------------------------------------------------------------------------

def read_table(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Table not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"Table has no header: {path}"
            )

        fieldnames = list(reader.fieldnames)
        missing = sorted(
            REQUIRED_COLUMNS.difference(fieldnames)
        )

        if missing:
            raise ValueError(
                "Table is missing required columns: "
                + ", ".join(missing)
            )

        rows = [
            {
                field: source_row.get(field, "")
                for field in fieldnames
            }
            for source_row in reader
        ]

    if not rows:
        raise ValueError(
            f"Table contains no rows: {path}"
        )

    return fieldnames, rows


def atomic_write_table(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def backup_table(path: Path) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = path.with_name(
        f"{path.stem}_before_pivot_rerun_"
        f"{timestamp}{path.suffix}"
    )

    shutil.copy2(path, backup)
    return backup


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def progress_path_for(table_path: Path) -> Path:
    return table_path.with_name(
        f"{table_path.stem}_pivot_rerun_progress.json"
    )


def manifest_path_for(table_path: Path) -> Path:
    return table_path.with_name(
        f"{table_path.stem}_pivot_rerun_manifest.json"
    )


def empty_progress(
    *,
    table_path: Path,
    ego_ids: Sequence[str],
    pivot_seeds: Sequence[int],
    backup_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "table": str(table_path),
        "backup": str(backup_path),
        "ego_ids": list(ego_ids),
        "pivot_seeds": list(pivot_seeds),
        "completed_edge_instances": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_or_create_progress(
    *,
    progress_path: Path,
    table_path: Path,
    ego_ids: Sequence[str],
    pivot_seeds: Sequence[int],
    restart: bool,
) -> dict[str, Any]:
    if restart and progress_path.exists():
        progress_path.unlink()

    if progress_path.exists():
        with progress_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            progress = json.load(handle)

        expected_egos = list(ego_ids)
        expected_pivot_seeds = list(pivot_seeds)

        if progress.get("pivot_seeds") != expected_pivot_seeds:
            raise ValueError(
                "The existing Pivot progress file uses different Pivot seeds. "
                "Run with --restart only when you intentionally want to "
                "recompute every Pivot result."
            )

        existing_egos = set(progress.get("ego_ids", []))
        requested_egos = set(expected_egos)

        if not existing_egos.issubset(requested_egos):
            raise ValueError(
                "The requested ego-ID set excludes ego IDs already tracked "
                "in the progress file. Use all ego IDs or run with --restart."
            )

        # Allow a previous four-ego run to expand to all ego graphs. The
        # already completed instance keys are retained and will be skipped.
        if progress.get("ego_ids") != expected_egos:
            progress["ego_ids"] = expected_egos
            progress["updated_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )
            progress["status"] = "running"
            atomic_write_json(progress_path, progress)

            added = sorted(
                requested_egos - existing_egos,
                key=lambda value: int(value),
            )
            print(
                "Expanded existing Pivot progress with ego IDs:",
                added,
            )

        return progress

    backup = backup_table(table_path)
    print("Backup created:", backup)

    progress = empty_progress(
        table_path=table_path,
        ego_ids=ego_ids,
        pivot_seeds=pivot_seeds,
        backup_path=backup,
    )
    atomic_write_json(progress_path, progress)

    return progress


# ---------------------------------------------------------------------------
# Selection and Pivot execution
# ---------------------------------------------------------------------------

def has_numeric_text(value: Any) -> bool:
    text = str(value).strip()

    if not text:
        return False

    try:
        float(text)
        return True
    except ValueError:
        return False


def detect_pivot_egos(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """
    Select every ego ID present in the flat experiment table.

    This intentionally includes ego graphs without any LP result. Their Pivot
    costs are still computed; a later paper-table script leaves the LP ratio
    blank when no matching LP is available.
    """
    detected = {
        str(row.get("ego_id", "")).strip()
        for row in rows
        if str(row.get("ego_id", "")).strip()
    }

    if detected:
        return sorted(
            detected,
            key=lambda value: int(value),
        )

    return list(FALLBACK_PIVOT_EGOS)


def run_pivot_summary(
    matrix: np.ndarray,
    pivot_seeds: Sequence[int],
) -> tuple[str, str]:
    result = experiment_helpers.run_pivot_multiple(
        matrix,
        pivot_seeds,
    )

    best_cost = canonical_number(
        result.get("best_cost")
    )
    average_cost = canonical_number(
        result.get("average_cost")
    )

    if not best_cost or not average_cost:
        raise RuntimeError(
            "Pivot did not return both best_cost and average_cost."
        )

    return best_cost, average_cost


def selected_row_indices(
    rows: Sequence[Mapping[str, Any]],
    ego_id: str,
) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if str(row.get("ego_id", "")).strip() == ego_id
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    install_deterministic_pivot()

    table_path = resolve(args.table)
    progress_path = progress_path_for(table_path)
    manifest_path = manifest_path_for(table_path)

    fieldnames, rows = read_table(table_path)

    if args.ego_ids is None:
        ego_ids = detect_pivot_egos(rows)
    else:
        ego_ids = sorted(
            set(parse_csv_list(args.ego_ids)),
            key=lambda value: int(value),
        )

    pivot_seeds = parse_integer_spec(
        args.pivot_seeds
    )

    if not ego_ids:
        raise ValueError(
            "No Pivot ego IDs were selected."
        )

    table_ego_ids = {
        str(row.get("ego_id", "")).strip()
        for row in rows
    }

    missing_egos = [
        ego_id
        for ego_id in ego_ids
        if ego_id not in table_ego_ids
    ]

    if missing_egos:
        raise ValueError(
            "Selected ego IDs are absent from the table: "
            + ", ".join(missing_egos)
        )

    progress = load_or_create_progress(
        progress_path=progress_path,
        table_path=table_path,
        ego_ids=ego_ids,
        pivot_seeds=pivot_seeds,
        restart=args.restart,
    )

    completed_keys = set(
        progress.get(
            "completed_edge_instances",
            [],
        )
    )

    selected_instances = {
        instance_key(
            row["ego_id"],
            row["p_delete"],
            row["seed"],
        )
        for row in rows
        if str(row["ego_id"]).strip() in set(ego_ids)
    }

    print("Table:", table_path)
    print("Selected Pivot ego IDs:", ego_ids)
    print("Pivot seeds:", pivot_seeds)
    print(
        "Unique edge-deleted instances:",
        len(selected_instances),
    )
    print(
        "Already completed:",
        len(completed_keys & selected_instances),
    )
    print("LP/ILP/MinMax algorithms: NOT RUN")

    processed_now = 0
    failures = 0
    edge_files: dict[str, dict[str, str]] = {}
    graph_hashes: dict[str, str] = {}

    for ego_position, ego_id in enumerate(
        ego_ids,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(
            f"FACEBOOK EGO {ego_id} "
            f"({ego_position}/{len(ego_ids)})"
        )
        print("=" * 80)

        try:
            complete_matrix, edge_file = (
                construct_complete_matrix(ego_id)
            )

            n = int(complete_matrix.shape[0])
            complete_hash = sha256_matrix(
                complete_matrix
            )

            edge_files[ego_id] = {
                "path": str(
                    edge_file.relative_to(REPO_ROOT)
                ),
                "sha256": sha256_file(edge_file),
            }
            graph_hashes[ego_id] = complete_hash

            print("n:", n)
            print(
                "Complete matrix SHA-256:",
                complete_hash,
            )

            complete_best, complete_average = (
                run_pivot_summary(
                    complete_matrix,
                    pivot_seeds,
                )
            )

            print(
                "Complete Pivot best:",
                complete_best,
            )
            print(
                "Complete Pivot average:",
                complete_average,
            )

            indices = selected_row_indices(
                rows,
                ego_id,
            )

            # Complete Pivot is one computation per ego and is repeated in all
            # rows belonging to that ego graph.
            for index in indices:
                rows[index][
                    "complete_pivot_best_cost"
                ] = complete_best
                rows[index][
                    "complete_pivot_average_cost"
                ] = complete_average

            atomic_write_table(
                table_path,
                fieldnames,
                rows,
            )

        except Exception as error:
            failures += 1
            print(
                f"FAILED to prepare ego {ego_id}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

            if args.continue_on_error:
                continue

            raise

        for index in indices:
            row = rows[index]

            p_delete = float(row["p_delete"])
            deletion_seed = int(
                float(row["seed"])
            )

            key = instance_key(
                ego_id,
                p_delete,
                deletion_seed,
            )

            if key in completed_keys:
                print(
                    f"SKIP ego={ego_id} "
                    f"p_delete={p_delete:g} "
                    f"seed={deletion_seed}"
                )
                continue

            if (
                args.limit is not None
                and processed_now >= args.limit
            ):
                print(
                    "\nLimit reached. Progress is saved."
                )
                break

            print(
                f"\nRUN ego={ego_id} "
                f"p_delete={p_delete:g} "
                f"seed={deletion_seed}",
                flush=True,
            )

            try:
                # Use a copy defensively in case delete_edges mutates input.
                edge_matrix, num_deleted = delete_edges(
                    complete_matrix.copy(),
                    p_delete,
                    deletion_seed,
                )

                edge_best, edge_average = (
                    run_pivot_summary(
                        edge_matrix,
                        pivot_seeds,
                    )
                )

                row[
                    "edge_pivot_best_cost"
                ] = edge_best
                row[
                    "edge_pivot_average_cost"
                ] = edge_average

                completed_keys.add(key)
                processed_now += 1

                progress[
                    "completed_edge_instances"
                ] = sorted(completed_keys)
                progress["updated_utc"] = (
                    datetime.now(timezone.utc)
                    .isoformat()
                )
                progress["status"] = "running"

                # Save the updated table and progress after every instance.
                atomic_write_table(
                    table_path,
                    fieldnames,
                    rows,
                )
                atomic_write_json(
                    progress_path,
                    progress,
                )

                print(
                    "Deleted edges:",
                    num_deleted,
                )
                print(
                    "Edge Pivot best:",
                    edge_best,
                )
                print(
                    "Edge Pivot average:",
                    edge_average,
                )
                print(
                    "Edge matrix SHA-256:",
                    sha256_matrix(edge_matrix),
                )
                print("Checkpoint saved.")

            except Exception as error:
                failures += 1

                print(
                    f"FAILED ego={ego_id} "
                    f"p_delete={p_delete:g} "
                    f"seed={deletion_seed}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

                if not args.continue_on_error:
                    raise

        if (
            args.limit is not None
            and processed_now >= args.limit
        ):
            break

    completed_selected = (
        completed_keys & selected_instances
    )

    status = (
        "complete"
        if (
            len(completed_selected)
            == len(selected_instances)
            and failures == 0
        )
        else "incomplete"
    )

    progress[
        "completed_edge_instances"
    ] = sorted(completed_keys)
    progress["updated_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    progress["status"] = status

    atomic_write_table(
        table_path,
        fieldnames,
        rows,
    )
    atomic_write_json(
        progress_path,
        progress,
    )

    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "status": status,
        "table": str(table_path),
        "table_sha256": sha256_file(
            table_path
        ),
        "backup": progress.get("backup"),
        "selected_ego_ids": ego_ids,
        "pivot_seeds": pivot_seeds,
        "selected_edge_instances": len(
            selected_instances
        ),
        "completed_edge_instances": len(
            completed_selected
        ),
        "failures": failures,
        "preprocessing": (
            "Vertices are sorted unique endpoints in the .edges file only."
        ),
        "pivot": (
            "Pivot chooses from sorted active nodes using NumPy default_rng."
        ),
        "edge_files": edge_files,
        "complete_matrix_hashes": graph_hashes,
        "untouched_columns_note": (
            "Only the four Pivot columns were modified. LP, ILP, MinMaxLP, "
            "and MinMaxCC columns were preserved."
        ),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print("\n" + "=" * 80)
    print("PIVOT UPDATE FINISHED")
    print("=" * 80)
    print("Status:", status)
    print(
        "Completed edge instances:",
        len(completed_selected),
        "/",
        len(selected_instances),
    )
    print(
        "Processed during this invocation:",
        processed_now,
    )
    print("Failures:", failures)
    print("Updated table:", table_path)
    print("Progress:", progress_path)
    print("Manifest:", manifest_path)
    print("LP/ILP/MinMax values were not changed.")

    if status != "complete" and args.limit is None:
        raise RuntimeError(
            "The Pivot update did not finish every selected instance."
        )


if __name__ == "__main__":
    main()
