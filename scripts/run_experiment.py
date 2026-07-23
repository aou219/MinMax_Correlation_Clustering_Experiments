#!/usr/bin/env python3
"""
Recompute corrected Facebook LP and MinMax results in the existing flat grid.

This script updates, in place:

    results/research_tables/minmax_facebook_grid_runs_flat.csv

It never recomputes or changes Pivot or ILP columns.

Corrected Facebook preprocessing
--------------------------------
Only vertices that occur as endpoints in the ego graph's ``.edges`` file are
used. Circle-only vertices are not added.

Modes
-----
``--mode normal``
    Recompute the ordinary all-pairs correlation-clustering LP.

``--mode minmax``
    Recompute MinMaxCC and/or MinMaxLP. Choose components with
    ``--minmax-components cc,lp``, ``cc``, or ``lp``.

``--mode all``
    Recompute the normal LP plus the selected MinMax components.

Long-running safety
-------------------
- The table is written atomically after every completed algorithm result.
- A progress JSON file makes interrupted runs resumable.
- A timestamped CSV backup is created before the first actual update.
- Existing Pivot and ILP columns are preserved exactly.
- ``--dry-run`` performs no solver calls and writes no files.
- ``--restart`` ignores/removes progress for the selected run configuration.

Typical commands
----------------
Preview ordinary LP work:

    python scripts/run_experiment.py --mode normal --dry-run

Run ordinary LP work:

    python scripts/run_experiment.py --mode normal

Preview only MinMaxLP:

    python scripts/run_experiment.py \
        --mode minmax \
        --minmax-components lp \
        --dry-run

Run only MinMaxLP:

    python scripts/run_experiment.py \
        --mode minmax \
        --minmax-components lp

Run MinMaxCC and MinMaxLP:

    python scripts/run_experiment.py \
        --mode minmax \
        --minmax-components cc,lp

Small real test:

    python scripts/run_experiment.py \
        --mode normal \
        --ego-ids 3980 \
        --p-delete-values 0.05 \
        --seeds 1 \
        --limit 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_VERSION = "2026-07-23-minmax-lp-rounding-clustering-v3"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = (
    REPO_ROOT
    / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
)

DEFAULT_NORMAL_LP_EGOS = "414,686,698,3980"
DEFAULT_MINMAX_LP_EGOS = "414,686,698,3980"
DEFAULT_MINMAX_CC_EGOS = "all"

DEFAULT_D_HAT = 8
DEFAULT_LAMBDA = 5
DEFAULT_MINMAX_LP_R = 0.4
DEFAULT_MINMAX_LP_R2 = 0.4
DEFAULT_MINMAX_LP_METHOD = 2

KEY_COLUMNS = ["ego_id", "p_delete", "seed"]

PIVOT_COLUMNS = {
    "complete_pivot_best_cost",
    "complete_pivot_average_cost",
    "edge_pivot_best_cost",
    "edge_pivot_average_cost",
}

ILP_COLUMNS = {
    "complete_ilp_cost",
    "edge_all_pairs_ilp_cost",
}

NORMAL_COMPLETE_COLUMNS = ["complete_lp_cost"]
NORMAL_EDGE_COLUMNS = ["edge_all_pairs_lp_cost"]

MINMAX_CC_COMPLETE_COLUMNS = [
    "complete_min_max_cc_computed",
    "complete_min_max_cc_cluster_count",
    "complete_min_max_cc_max_disagreement",
    "complete_min_max_cc_d_hat",
    "complete_min_max_cc_lambda",
    "complete_min_max_cc_runtime_seconds",
]

MINMAX_CC_EDGE_COLUMNS = [
    "edge_min_max_cc_computed",
    "edge_min_max_cc_cluster_count",
    "edge_min_max_cc_max_disagreement",
    "edge_min_max_cc_d_hat",
    "edge_min_max_cc_lambda",
    "edge_min_max_cc_runtime_seconds",
]

MINMAX_LP_COMPLETE_COLUMNS = [
    "complete_min_max_lp_computed",
    "complete_min_max_lp_cost",
    "complete_min_max_lp_rounding_cost",
    "complete_min_max_lp_max_disagreement_vertex",
    "complete_min_max_lp_cluster_count",
    "complete_min_max_lp_clustering_json",
    "complete_min_max_lp_r",
    "complete_min_max_lp_r2",
    "complete_min_max_lp_method",
    "complete_min_max_lp_norm",
    "complete_min_max_lp_runtime_seconds",
    "complete_min_max_lp_rounding_runtime_seconds",
    "complete_min_max_lp_total_runtime_seconds",
]

MINMAX_LP_EDGE_COLUMNS = [
    "edge_min_max_lp_computed",
    "edge_min_max_lp_cost",
    "edge_min_max_lp_rounding_cost",
    "edge_min_max_lp_max_disagreement_vertex",
    "edge_min_max_lp_cluster_count",
    "edge_min_max_lp_clustering_json",
    "edge_min_max_lp_r",
    "edge_min_max_lp_r2",
    "edge_min_max_lp_method",
    "edge_min_max_lp_norm",
    "edge_min_max_lp_runtime_seconds",
    "edge_min_max_lp_rounding_runtime_seconds",
    "edge_min_max_lp_total_runtime_seconds",
]

OPTIONAL_OUTPUT_COLUMNS = {
    "complete_min_max_lp_clustering_json",
    "edge_min_max_lp_clustering_json",
}

REQUIRED_COLUMNS = set(
    KEY_COLUMNS
    + ["n"]
    + list(PIVOT_COLUMNS)
    + list(ILP_COLUMNS)
    + NORMAL_COMPLETE_COLUMNS
    + NORMAL_EDGE_COLUMNS
    + MINMAX_CC_COMPLETE_COLUMNS
    + MINMAX_CC_EDGE_COLUMNS
    + [
        column
        for column in MINMAX_LP_COMPLETE_COLUMNS
        if column not in OPTIONAL_OUTPUT_COLUMNS
    ]
    + [
        column
        for column in MINMAX_LP_EDGE_COLUMNS
        if column not in OPTIONAL_OUTPUT_COLUMNS
    ]
)


# =============================================================================
# CLI and parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute corrected ordinary LP and/or MinMax results and "
            "update the existing Facebook flat grid in place."
        )
    )

    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Existing minmax_facebook_grid_runs_flat.csv to update.",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "minmax", "all"],
        default="all",
        help=(
            "normal: ordinary all-pairs LP only; "
            "minmax: selected MinMax components; "
            "all: both."
        ),
    )
    parser.add_argument(
        "--minmax-components",
        default="cc,lp",
        help="Comma-separated MinMax components: cc, lp, or cc,lp.",
    )

    parser.add_argument(
        "--ego-ids",
        default="all",
        help="Global ego selection: all, comma list, or ranges.",
    )
    parser.add_argument(
        "--p-delete-values",
        default="all",
        help="Deletion probabilities, e.g. all or 0.05,0.15.",
    )
    parser.add_argument(
        "--seeds",
        default="all",
        help="Deletion seeds, e.g. all, 1, 1-5, or 1,3,7-10.",
    )

    parser.add_argument(
        "--normal-lp-egos",
        default=DEFAULT_NORMAL_LP_EGOS,
        help=(
            "Egos for the ordinary all-pairs LP. Default: "
            f"{DEFAULT_NORMAL_LP_EGOS}."
        ),
    )
    parser.add_argument(
        "--minmax-cc-egos",
        default=DEFAULT_MINMAX_CC_EGOS,
        help="Egos for MinMaxCC. Default: all selected ego graphs.",
    )
    parser.add_argument(
        "--minmax-lp-egos",
        default=DEFAULT_MINMAX_LP_EGOS,
        help=(
            "Egos for MinMaxLP. Default: "
            f"{DEFAULT_MINMAX_LP_EGOS}."
        ),
    )

    parser.add_argument("--d-hat", type=int, default=DEFAULT_D_HAT)
    parser.add_argument(
        "--lambda-value",
        type=int,
        default=DEFAULT_LAMBDA,
    )
    parser.add_argument(
        "--min-max-lp-r",
        type=float,
        default=DEFAULT_MINMAX_LP_R,
    )
    parser.add_argument(
        "--min-max-lp-r2",
        type=float,
        default=DEFAULT_MINMAX_LP_R2,
    )
    parser.add_argument(
        "--min-max-lp-method",
        type=int,
        default=DEFAULT_MINMAX_LP_METHOD,
    )
    parser.add_argument(
        "--all-pairs-time-limit",
        type=float,
        default=None,
        help="Optional time limit in seconds for each ordinary LP solve.",
    )
    parser.add_argument(
        "--memory-cleanup",
        choices=["off", "gc", "gurobi"],
        default="gurobi",
        help=(
            "Memory cleanup after each solver call. "
            "'off' performs no forced cleanup; "
            "'gc' clears result objects and runs Python garbage collection; "
            "'gurobi' additionally disposes Gurobi's default environment. "
            "Default: gurobi."
        ),
    )

    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Optional progress JSON path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional reproducibility manifest JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without solver calls or file writes.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard progress for this run configuration and recompute.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue with later instances.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most this many new edge-deleted graph instances. "
            "Complete-graph computations are not counted."
        ),
    )

    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def split_csv(raw: str) -> list[str]:
    return [
        part.strip()
        for part in str(raw).split(",")
        if part.strip()
    ]


def parse_int_spec(raw: str, available: Sequence[int]) -> list[int]:
    text = str(raw).strip().lower()

    if text == "all":
        return sorted(set(int(value) for value in available))

    values: set[int] = set()

    for part in split_csv(text):
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)

            if stop < start:
                raise ValueError(f"Invalid descending range: {part}")

            values.update(range(start, stop + 1))
        else:
            values.add(int(part))

    missing = sorted(values.difference(set(available)))
    if missing:
        raise ValueError(
            "Requested values not present in the table: "
            + ", ".join(str(value) for value in missing)
        )

    return sorted(values)


def parse_probability_spec(
    raw: str,
    available: Sequence[float],
) -> list[float]:
    text = str(raw).strip().lower()

    if text == "all":
        return sorted(set(float(value) for value in available))

    requested = [float(part) for part in split_csv(text)]
    available_by_key = {
        probability_key(value): float(value)
        for value in available
    }

    missing = [
        value
        for value in requested
        if probability_key(value) not in available_by_key
    ]
    if missing:
        raise ValueError(
            "Requested p_delete values not present in the table: "
            + ", ".join(f"{value:g}" for value in missing)
        )

    return sorted(
        {
            available_by_key[probability_key(value)]
            for value in requested
        }
    )


def parse_ego_spec(
    raw: str,
    available: Sequence[str],
) -> list[str]:
    text = str(raw).strip().lower()
    available_set = set(str(value) for value in available)

    if text == "all":
        return sorted(available_set, key=int)

    values: list[str] = []
    seen: set[str] = set()

    for part in split_csv(text):
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)

            if stop < start:
                raise ValueError(f"Invalid descending range: {part}")

            expanded = [str(value) for value in range(start, stop + 1)]
        else:
            expanded = [str(int(part))]

        for value in expanded:
            if value not in seen:
                values.append(value)
                seen.add(value)

    missing = sorted(set(values).difference(available_set), key=int)
    if missing:
        raise ValueError(
            "Requested ego IDs not present in the table: "
            + ", ".join(missing)
        )

    return values


def parse_minmax_components(raw: str) -> set[str]:
    components = {
        component.lower()
        for component in split_csv(raw)
    }

    invalid = sorted(components.difference({"cc", "lp"}))
    if invalid:
        raise ValueError(
            "Invalid MinMax component(s): " + ", ".join(invalid)
        )

    if not components:
        raise ValueError(
            "--minmax-components must contain cc, lp, or both."
        )

    return components


# =============================================================================
# Stable formatting, keys, hashes
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def probability_key(value: Any) -> str:
    return f"{float(value):.12g}"


def row_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["ego_id"]).strip(),
        probability_key(row["p_delete"]),
        int(float(row["seed"])),
    )


def task_key(
    scope: str,
    component: str,
    ego_id: str,
    p_delete: float | None = None,
    seed: int | None = None,
) -> str:
    if scope == "complete":
        return f"complete|{component}|{ego_id}"

    if p_delete is None or seed is None:
        raise ValueError("Edge task requires p_delete and seed.")

    return (
        f"edge|{component}|{ego_id}|"
        f"{probability_key(p_delete)}|{int(seed)}"
    )


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def csv_value(value: Any) -> str:
    value = json_safe(value)

    if value is None:
        return ""

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return repr(value)

    return str(value)


def sha256_matrix(matrix: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


# =============================================================================
# CSV and JSON I/O
# =============================================================================

def read_table(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))

        if missing:
            raise ValueError(
                "The table is missing required columns: "
                + ", ".join(missing)
            )

        for column in sorted(OPTIONAL_OUTPUT_COLUMNS):
            if column not in fieldnames:
                fieldnames.append(column)

        rows = [
            {
                field: source_row.get(field, "")
                for field in fieldnames
            }
            for source_row in reader
        ]

    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "The table contains duplicate ego_id/p_delete/seed rows."
        )

    return fieldnames, rows


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return value


def make_backup(table_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = table_path.with_name(
        f"{table_path.stem}_before_corrected_lp_minmax_"
        f"{timestamp}{table_path.suffix}"
    )
    shutil.copy2(table_path, backup)
    return backup


# =============================================================================
# Planning and progress
# =============================================================================

def active_components(
    mode: str,
    minmax_components: set[str],
) -> list[str]:
    result: list[str] = []

    if mode in {"normal", "all"}:
        result.append("normal_lp")

    if mode in {"minmax", "all"}:
        if "cc" in minmax_components:
            result.append("minmax_cc")
        if "lp" in minmax_components:
            result.append("minmax_lp")

    return result


def run_tag(components: Sequence[str]) -> str:
    return "_".join(components)


def derive_sidecar_paths(
    table_path: Path,
    args: argparse.Namespace,
    components: Sequence[str],
) -> tuple[Path, Path]:
    tag = run_tag(components)

    progress = (
        resolve(args.progress)
        if args.progress is not None
        else table_path.with_name(
            f"{table_path.stem}_{tag}_progress.json"
        )
    )
    manifest = (
        resolve(args.manifest)
        if args.manifest is not None
        else table_path.with_name(
            f"{table_path.stem}_{tag}_manifest.json"
        )
    )

    return progress, manifest


def algorithm_signature(
    args: argparse.Namespace,
    components: Sequence[str],
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "components": list(components),
        "graph_preprocessing": (
            "Only endpoints occurring in each Facebook .edges file; "
            "circle-only nodes excluded."
        ),
        "edge_deletion": (
            "src.edge_deletion.delete_edges(matrix, p_delete, seed)"
        ),
        "normal_lp": {
            "enabled": "normal_lp" in components,
            "compute_lp": True,
            "compute_ilp": False,
            "time_limit": args.all_pairs_time_limit,
        },
        "minmax_cc": {
            "enabled": "minmax_cc" in components,
            "d_hat": args.d_hat,
            "lambda": args.lambda_value,
        },
        "minmax_lp": {
            "enabled": "minmax_lp" in components,
            "r": args.min_max_lp_r,
            "r2": args.min_max_lp_r2,
            "method": args.min_max_lp_method,
            "norm": "inf",
            "rounding_required": True,
        },
    }


def new_progress(signature: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "signature": dict(signature),
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "backup": None,
        "completed": [],
        "complete_matrix_hashes": {},
        "failures": [],
    }


def load_progress(
    path: Path,
    signature: Mapping[str, Any],
    restart: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if restart:
        if path.exists() and not dry_run:
            path.unlink()
        return new_progress(signature)

    if not path.exists():
        return new_progress(signature)

    progress = read_json(path)

    if progress.get("signature") != dict(signature):
        raise ValueError(
            "The existing progress file belongs to a different algorithm "
            "configuration. Use --restart or specify another --progress path."
        )

    progress.setdefault("completed", [])
    progress.setdefault("complete_matrix_hashes", {})
    progress.setdefault("failures", [])
    return progress


def component_ego_sets(
    args: argparse.Namespace,
    selected_egos: Sequence[str],
    available_egos: Sequence[str],
    components: Sequence[str],
) -> dict[str, set[str]]:
    selected_set = set(selected_egos)
    output: dict[str, set[str]] = {
        "normal_lp": set(),
        "minmax_cc": set(),
        "minmax_lp": set(),
    }

    specs = {
        "normal_lp": args.normal_lp_egos,
        "minmax_cc": args.minmax_cc_egos,
        "minmax_lp": args.minmax_lp_egos,
    }

    for component in components:
        parsed = set(parse_ego_spec(specs[component], available_egos))
        output[component] = parsed.intersection(selected_set)

    return output


def selected_rows(
    rows: Sequence[dict[str, str]],
    ego_ids: set[str],
    p_delete_values: set[str],
    seeds: set[int],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row["ego_id"]).strip() in ego_ids
        and probability_key(row["p_delete"]) in p_delete_values
        and int(float(row["seed"])) in seeds
    ]


def print_plan(
    table_path: Path,
    progress_path: Path,
    manifest_path: Path,
    components: Sequence[str],
    component_egos: Mapping[str, set[str]],
    rows_to_process: Sequence[dict[str, str]],
    completed: set[str],
    args: argparse.Namespace,
) -> None:
    print("=" * 78)
    print("CORRECTED FACEBOOK RECOMPUTATION PLAN")
    print("=" * 78)
    print("Table:", table_path)
    print("Mode:", args.mode)
    print("Components:", ", ".join(components))
    print("Progress:", progress_path)
    print("Manifest:", manifest_path)
    print("Dry run:", args.dry_run)
    print("Restart:", args.restart)
    print("Memory cleanup:", args.memory_cleanup)
    print()

    print("Corrected graph definition:")
    print("  vertices = sorted endpoints occurring in the .edges file")
    print("  circle-only vertices are excluded")
    print()

    for component in components:
        print(
            f"{component} egos:",
            ",".join(sorted(component_egos[component], key=int))
            or "(none)",
        )

    complete_remaining: dict[str, int] = {}
    edge_remaining: dict[str, int] = {}

    for component in components:
        complete_remaining[component] = sum(
            task_key("complete", component, ego_id) not in completed
            for ego_id in component_egos[component]
        )

        edge_remaining[component] = sum(
            (
                str(row["ego_id"]).strip()
                in component_egos[component]
                and task_key(
                    "edge",
                    component,
                    str(row["ego_id"]).strip(),
                    float(row["p_delete"]),
                    int(float(row["seed"])),
                )
                not in completed
            )
            for row in rows_to_process
        )

    edge_graphs_remaining = sum(
        any(
            str(row["ego_id"]).strip() in component_egos[component]
            and task_key(
                "edge",
                component,
                str(row["ego_id"]).strip(),
                float(row["p_delete"]),
                int(float(row["seed"])),
            )
            not in completed
            for component in components
        )
        for row in rows_to_process
    )

    print()
    print("Remaining solver calls:")
    for component in components:
        print(
            f"  {component}: "
            f"{complete_remaining[component]} complete + "
            f"{edge_remaining[component]} edge"
        )
    print("  unique edge matrices to construct:", edge_graphs_remaining)
    print("  already completed tasks:", len(completed))

    if args.limit is not None:
        print("  edge-instance limit:", args.limit)

    if args.dry_run:
        print()
        print("DRY RUN: no graph, solver, CSV, JSON, or backup changes made.")


# =============================================================================
# Project imports and corrected graph construction
# =============================================================================

def load_project_modules() -> tuple[Any, Any, Any, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import numpy as np
    from src import experiment_helpers
    from src.edge_deletion import delete_edges
    from src.facebook_sampling import (
        build_complete_signed_matrix_from_facebook_sample,
        load_facebook_ego_edges,
    )

    return (
        np,
        experiment_helpers,
        delete_edges,
        (
            load_facebook_ego_edges,
            build_complete_signed_matrix_from_facebook_sample,
        ),
    )


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
    facebook_modules: tuple[Any, Any],
) -> tuple[Any, Path, dict[int, Any]]:
    (
        load_facebook_ego_edges,
        build_complete_signed_matrix_from_facebook_sample,
    ) = facebook_modules

    edge_file = locate_edges_file(ego_id)
    edge_nodes, facebook_edges = load_facebook_ego_edges(
        str(edge_file)
    )

    # Corrected preprocessing: do not union with circle-only nodes.
    ordered_nodes = sorted(edge_nodes)

    matrix, node_to_index, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            ordered_nodes,
            facebook_edges,
        )
    )
    index_to_node = {
        int(index): node
        for node, index in node_to_index.items()
    }

    return matrix, edge_file, index_to_node


# =============================================================================
# Algorithm execution and table mapping
# =============================================================================

def run_normal_lp(
    matrix: Any,
    helpers: Any,
    time_limit: float | None,
) -> dict[str, Any]:
    result = helpers.compute_all_pairs_data(
        matrix,
        compute_lp=True,
        compute_ilp=False,
        time_limit=time_limit,
    )

    if result.get("lp_cost") is None:
        raise RuntimeError("The ordinary all-pairs LP returned no cost.")

    return result


def run_minmax_cc(
    matrix: Any,
    helpers: Any,
    d_hat: int,
    lambda_value: int,
) -> dict[str, Any]:
    result = helpers.compute_min_max_cc_data(
        matrix,
        compute_min_max=True,
        param_1=d_hat,
        param_2=lambda_value,
    )

    if not result.get("computed", False):
        raise RuntimeError("MinMaxCC was not computed.")

    return result


def run_minmax_lp(
    matrix: Any,
    helpers: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np

    result = helpers.compute_min_max_lp_data(
        matrix,
        compute_min_max_lp=True,
        r=args.min_max_lp_r,
        r2=args.min_max_lp_r2,
        method=args.min_max_lp_method,
        norm=np.inf,
    )

    if not result.get("computed", False):
        raise RuntimeError("MinMaxLP was not computed.")

    required_values = {
        "lp_cost": result.get("lp_cost"),
        "rounding_cost": result.get("rounding_cost"),
        "max_disagreement_vertex": result.get(
            "max_disagreement_vertex"
        ),
        "cluster_count": result.get("cluster_count"),
        "clustering": result.get("clustering"),
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
    missing = [
        key
        for key, value in required_values.items()
        if value is None
    ]
    if missing:
        raise RuntimeError(
            "MinMaxLP/rounding returned missing values: "
            + ", ".join(missing)
        )

    return result


def rows_for_ego(
    rows: Sequence[dict[str, str]],
    ego_id: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row["ego_id"]).strip() == ego_id
    ]


def update_normal_result(
    target_rows: Sequence[dict[str, str]],
    scope: str,
    result: Mapping[str, Any],
) -> None:
    field = (
        "complete_lp_cost"
        if scope == "complete"
        else "edge_all_pairs_lp_cost"
    )
    value = csv_value(result["lp_cost"])

    for row in target_rows:
        row[field] = value


def update_minmax_cc_result(
    target_rows: Sequence[dict[str, str]],
    scope: str,
    result: Mapping[str, Any],
) -> None:
    prefix = f"{scope}_min_max_cc"

    values = {
        f"{prefix}_computed": result.get("computed"),
        f"{prefix}_cluster_count": result.get("cluster_count"),
        f"{prefix}_max_disagreement": result.get("max_disagreement"),
        f"{prefix}_d_hat": result.get("d_hat"),
        f"{prefix}_lambda": result.get("lambda"),
        f"{prefix}_runtime_seconds": result.get("runtime_seconds"),
    }

    for row in target_rows:
        for field, value in values.items():
            row[field] = csv_value(value)


def clustering_as_node_ids_json(
    clustering: Any,
    index_to_node: Mapping[int, Any],
) -> str:
    """Serialize clusters using original Facebook node IDs."""
    if clustering is None:
        return ""

    converted: list[list[Any]] = []
    for cluster_values in clustering:
        converted_cluster: list[Any] = []
        for raw_index in cluster_values:
            index = int(raw_index)
            if index not in index_to_node:
                raise ValueError(
                    f"Clustering contains unknown matrix index {index}."
                )
            converted_cluster.append(index_to_node[index])
        converted.append(converted_cluster)

    return json.dumps(
        converted,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def first_present(
    result: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in result and result.get(key) is not None:
            return result.get(key)
    return None


def update_minmax_lp_result(
    target_rows: Sequence[dict[str, str]],
    scope: str,
    result: Mapping[str, Any],
    args: argparse.Namespace,
    index_to_node: Mapping[int, Any],
) -> None:
    prefix = f"{scope}_min_max_lp"

    lp_runtime = first_present(
        result,
        "lp_runtime_seconds",
        "runtime_seconds",
    )
    rounding_runtime = first_present(
        result,
        "rounding_runtime_seconds",
    )
    total_runtime = first_present(
        result,
        "total_runtime_seconds",
    )

    if total_runtime is None:
        numeric_parts = [
            float(value)
            for value in (lp_runtime, rounding_runtime)
            if value is not None
        ]
        total_runtime = sum(numeric_parts) if numeric_parts else None

    norm = first_present(result, "norm")
    if norm is None:
        norm = "inf"

    values = {
        f"{prefix}_computed": result.get("computed", True),
        f"{prefix}_cost": result.get("lp_cost"),
        f"{prefix}_rounding_cost": first_present(
            result,
            "rounding_cost",
        ),
        f"{prefix}_max_disagreement_vertex": first_present(
            result,
            "max_disagreement_vertex",
        ),
        f"{prefix}_cluster_count": first_present(
            result,
            "cluster_count",
        ),
        f"{prefix}_clustering_json": clustering_as_node_ids_json(
            first_present(result, "clustering"),
            index_to_node,
        ),
        f"{prefix}_r": result.get("r", args.min_max_lp_r),
        f"{prefix}_r2": result.get("r2", args.min_max_lp_r2),
        f"{prefix}_method": result.get(
            "method",
            args.min_max_lp_method,
        ),
        f"{prefix}_norm": norm,
        f"{prefix}_runtime_seconds": lp_runtime,
        f"{prefix}_rounding_runtime_seconds": rounding_runtime,
        f"{prefix}_total_runtime_seconds": total_runtime,
    }

    for row in target_rows:
        for field, value in values.items():
            row[field] = csv_value(value)


def run_component(
    component: str,
    matrix: Any,
    helpers: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if component == "normal_lp":
        return run_normal_lp(
            matrix,
            helpers,
            args.all_pairs_time_limit,
        )

    if component == "minmax_cc":
        return run_minmax_cc(
            matrix,
            helpers,
            args.d_hat,
            args.lambda_value,
        )

    if component == "minmax_lp":
        return run_minmax_lp(
            matrix,
            helpers,
            args,
        )

    raise ValueError(f"Unknown component: {component}")


def apply_component_result(
    component: str,
    scope: str,
    target_rows: Sequence[dict[str, str]],
    result: Mapping[str, Any],
    args: argparse.Namespace,
    index_to_node: Mapping[int, Any],
) -> None:
    if component == "normal_lp":
        update_normal_result(target_rows, scope, result)
    elif component == "minmax_cc":
        update_minmax_cc_result(target_rows, scope, result)
    elif component == "minmax_lp":
        update_minmax_lp_result(
            target_rows,
            scope,
            result,
            args,
            index_to_node,
        )
    else:
        raise ValueError(f"Unknown component: {component}")


# =============================================================================
# Memory cleanup
# =============================================================================

def release_solver_memory(
    result: dict[str, Any] | None,
    mode: str,
) -> None:
    """Release Python and optional Gurobi resources after a solver call."""
    if isinstance(result, dict):
        result.clear()

    result = None
    gc.collect()

    if mode == "gurobi":
        try:
            import gurobipy as gp
            gp.disposeDefaultEnv()
        except Exception as exc:
            print(
                "Memory-cleanup warning: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        gc.collect()


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    if args.d_hat < 0:
        raise ValueError("--d-hat must be non-negative.")

    if args.lambda_value <= 4:
        raise ValueError("--lambda-value must be greater than 4.")

    table_path = resolve(args.table)
    fieldnames, rows = read_table(table_path)

    available_egos = sorted(
        {
            str(row["ego_id"]).strip()
            for row in rows
        },
        key=int,
    )
    available_probabilities = sorted(
        {
            float(row["p_delete"])
            for row in rows
        }
    )
    available_seeds = sorted(
        {
            int(float(row["seed"]))
            for row in rows
        }
    )

    selected_egos = parse_ego_spec(
        args.ego_ids,
        available_egos,
    )
    selected_probabilities = parse_probability_spec(
        args.p_delete_values,
        available_probabilities,
    )
    selected_seeds = parse_int_spec(
        args.seeds,
        available_seeds,
    )

    minmax_components = parse_minmax_components(
        args.minmax_components
    )
    components = active_components(
        args.mode,
        minmax_components,
    )

    component_egos = component_ego_sets(
        args,
        selected_egos,
        available_egos,
        components,
    )

    probability_keys = {
        probability_key(value)
        for value in selected_probabilities
    }
    seed_set = set(selected_seeds)
    global_ego_set = set(selected_egos)

    rows_to_process = selected_rows(
        rows,
        global_ego_set,
        probability_keys,
        seed_set,
    )

    progress_path, manifest_path = derive_sidecar_paths(
        table_path,
        args,
        components,
    )
    signature = algorithm_signature(
        args,
        components,
    )
    progress = load_progress(
        progress_path,
        signature,
        args.restart,
        args.dry_run,
    )
    completed = set(
        str(value)
        for value in progress.get("completed", [])
    )

    print_plan(
        table_path,
        progress_path,
        manifest_path,
        components,
        component_egos,
        rows_to_process,
        completed,
        args,
    )

    if args.dry_run:
        return

    (
        np,
        helpers,
        delete_edges,
        facebook_modules,
    ) = load_project_modules()

    manifest = {
        "script_version": SCRIPT_VERSION,
        "started_utc": utc_now(),
        "table": str(table_path),
        "progress": str(progress_path),
        "mode": args.mode,
        "components": components,
        "selected_egos": selected_egos,
        "selected_p_delete_values": selected_probabilities,
        "selected_seeds": selected_seeds,
        "component_egos": {
            component: sorted(values, key=int)
            for component, values in component_egos.items()
            if component in components
        },
        "signature": signature,
        "preserved_columns": sorted(
            PIVOT_COLUMNS | ILP_COLUMNS
        ),
        "status": "running",
    }
    atomic_write_json(manifest_path, manifest)

    backup_created = False

    def ensure_backup() -> None:
        nonlocal backup_created

        if backup_created:
            return

        existing_backup = progress.get("backup")
        if existing_backup:
            backup_created = True
            return

        backup = make_backup(table_path)
        progress["backup"] = str(backup)
        progress["updated_utc"] = utc_now()
        atomic_write_json(progress_path, progress)
        backup_created = True
        print("Backup:", backup, flush=True)

    def save_checkpoint() -> None:
        progress["completed"] = sorted(completed)
        progress["updated_utc"] = utc_now()
        atomic_write_csv(table_path, fieldnames, rows)
        atomic_write_json(progress_path, progress)
        print("Checkpoint saved.", flush=True)

    failures: list[dict[str, Any]] = progress.setdefault(
        "failures",
        []
    )

    edge_instances_processed = 0
    stop_due_to_limit = False

    try:
        for ego_index, ego_id in enumerate(
            selected_egos,
            start=1,
        ):
            selected_for_any_component = any(
                ego_id in component_egos[component]
                for component in components
            )
            if not selected_for_any_component:
                continue

            print()
            print("=" * 78)
            print(
                f"FACEBOOK EGO {ego_id} "
                f"({ego_index}/{len(selected_egos)})"
            )
            print("=" * 78)

            (
                complete_matrix,
                edge_file,
                index_to_node,
            ) = construct_complete_matrix(
                ego_id,
                facebook_modules,
            )
            n = int(complete_matrix.shape[0])
            complete_hash = sha256_matrix(complete_matrix)

            prior_hash = progress[
                "complete_matrix_hashes"
            ].get(ego_id)
            if prior_hash is not None and prior_hash != complete_hash:
                raise ValueError(
                    f"Corrected complete matrix for ego {ego_id} changed "
                    "since the progress file was created. Use --restart "
                    "after verifying the input data."
                )

            progress["complete_matrix_hashes"][ego_id] = complete_hash

            ego_rows = rows_for_ego(rows, ego_id)
            n_changed = any(row.get("n") != str(n) for row in ego_rows)
            for row in ego_rows:
                row["n"] = str(n)

            print("Edges file:", edge_file)
            print("Corrected n:", n)
            print("Complete matrix SHA-256:", complete_hash)

            if n_changed:
                ensure_backup()
                save_checkpoint()

            # Complete graph: each component is computed once per ego and
            # propagated to every row of that ego graph.
            for component in components:
                if ego_id not in component_egos[component]:
                    continue

                complete_task = task_key(
                    "complete",
                    component,
                    ego_id,
                )

                if complete_task in completed:
                    print(
                        f"SKIP complete {component}: already checkpointed."
                    )
                    continue

                print(
                    f"RUN complete {component} for ego={ego_id}",
                    flush=True,
                )
                start = time.perf_counter()
                result: dict[str, Any] | None = None

                try:
                    result = run_component(
                        component,
                        complete_matrix,
                        helpers,
                        args,
                    )
                    apply_component_result(
                        component,
                        "complete",
                        ego_rows,
                        result,
                        args,
                        index_to_node,
                    )

                    ensure_backup()
                    completed.add(complete_task)
                    save_checkpoint()

                    print(
                        f"Complete {component} runtime: "
                        f"{time.perf_counter() - start:.2f}s",
                        flush=True,
                    )

                    if args.memory_cleanup != "off":
                        release_solver_memory(result, args.memory_cleanup)
                        result = None
                        print("Solver memory cleanup complete.", flush=True)

                except Exception as exc:
                    if args.memory_cleanup != "off":
                        release_solver_memory(result, args.memory_cleanup)
                        result = None
                    failure = {
                        "utc": utc_now(),
                        "task": complete_task,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    progress["updated_utc"] = utc_now()
                    atomic_write_json(progress_path, progress)

                    print(
                        f"FAILED {complete_task}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                    if not args.continue_on_error:
                        raise

            ego_selected_rows = [
                row
                for row in rows_to_process
                if str(row["ego_id"]).strip() == ego_id
            ]
            ego_selected_rows.sort(
                key=lambda row: (
                    float(row["p_delete"]),
                    int(float(row["seed"])),
                )
            )

            for row in ego_selected_rows:
                p_delete = float(row["p_delete"])
                seed = int(float(row["seed"]))

                needed_components = [
                    component
                    for component in components
                    if ego_id in component_egos[component]
                    and task_key(
                        "edge",
                        component,
                        ego_id,
                        p_delete,
                        seed,
                    )
                    not in completed
                ]

                if not needed_components:
                    continue

                if (
                    args.limit is not None
                    and edge_instances_processed >= args.limit
                ):
                    stop_due_to_limit = True
                    break

                print()
                print(
                    f"RUN ego={ego_id} "
                    f"p_delete={p_delete:g} seed={seed}"
                )

                edge_matrix, num_deleted = delete_edges(
                    complete_matrix.copy(),
                    p_delete,
                    seed,
                )
                edge_hash = sha256_matrix(edge_matrix)

                print("Deleted edges:", num_deleted)
                print("Edge matrix SHA-256:", edge_hash)

                for component in needed_components:
                    edge_task = task_key(
                        "edge",
                        component,
                        ego_id,
                        p_delete,
                        seed,
                    )
                    start = time.perf_counter()
                    result: dict[str, Any] | None = None

                    try:
                        result = run_component(
                            component,
                            edge_matrix,
                            helpers,
                            args,
                        )
                        apply_component_result(
                            component,
                            "edge",
                            [row],
                            result,
                            args,
                            index_to_node,
                        )

                        ensure_backup()
                        completed.add(edge_task)
                        save_checkpoint()

                        print(
                            f"{component} runtime: "
                            f"{time.perf_counter() - start:.2f}s",
                            flush=True,
                        )

                        if args.memory_cleanup != "off":
                            release_solver_memory(result, args.memory_cleanup)
                            result = None
                            print("Solver memory cleanup complete.", flush=True)

                    except Exception as exc:
                        if args.memory_cleanup != "off":
                            release_solver_memory(result, args.memory_cleanup)
                            result = None
                        failure = {
                            "utc": utc_now(),
                            "task": edge_task,
                            "edge_matrix_sha256": edge_hash,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failures.append(failure)
                        progress["updated_utc"] = utc_now()
                        atomic_write_json(
                            progress_path,
                            progress,
                        )

                        print(
                            f"FAILED {edge_task}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )

                        if not args.continue_on_error:
                            raise

                edge_instances_processed += 1

                if args.memory_cleanup != "off":
                    del edge_matrix
                    gc.collect()

            if args.memory_cleanup != "off":
                del complete_matrix
                gc.collect()

            if stop_due_to_limit:
                break

    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["finished_utc"] = utc_now()
        manifest["completed_tasks"] = len(completed)
        manifest["failures"] = len(failures)
        atomic_write_json(manifest_path, manifest)

        print()
        print("Interrupted with Ctrl+C.")
        print("All previous checkpoints are preserved.")
        print("Resume with the same command.")
        raise SystemExit(130)

    remaining = 0

    for component in components:
        remaining += sum(
            task_key("complete", component, ego_id) not in completed
            for ego_id in component_egos[component]
        )

        remaining += sum(
            (
                str(row["ego_id"]).strip()
                in component_egos[component]
                and task_key(
                    "edge",
                    component,
                    str(row["ego_id"]).strip(),
                    float(row["p_delete"]),
                    int(float(row["seed"])),
                )
                not in completed
            )
            for row in rows_to_process
        )

    manifest["status"] = (
        "limited"
        if stop_due_to_limit
        else "complete"
        if remaining == 0
        else "incomplete"
    )
    manifest["finished_utc"] = utc_now()
    manifest["completed_tasks"] = len(completed)
    manifest["remaining_requested_tasks"] = remaining
    manifest["failures"] = len(failures)
    manifest["backup"] = progress.get("backup")
    atomic_write_json(manifest_path, manifest)

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("Status:", manifest["status"])
    print("Table:", table_path)
    print("Progress:", progress_path)
    print("Manifest:", manifest_path)
    print("Completed tasks:", len(completed))
    print("Remaining requested tasks:", remaining)
    print("Failures:", len(failures))
    print(
        "Pivot and ILP columns were not changed by this script."
    )


if __name__ == "__main__":
    main()
