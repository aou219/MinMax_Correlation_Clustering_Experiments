#!/usr/bin/env python3
"""

Default experiment profile
--------------------------
All 10 Facebook ego graphs:
    MinMaxCC, d_hat=8, lambda=5

Ego graphs 414, 686, 698, and 3980:
    Pivot with seeds 1..100
    all-pairs LP relaxation
    MinMaxLP with r=0.4, r2=0.4, method=2

Ego graphs 414, 698, and 3980:
    all-pairs ILP

Deletion probabilities:
    0.05, 0.15, 0.25, 0.40

Deletion seeds:
    1..30

The output schema matches the existing
``minmax_facebook_grid_runs_flat.csv`` table. Legacy MinMaxLP-rounding columns
remain in the CSV for compatibility, but are intentionally left blank because
the current project uses MinMaxLP only as a lower bound and no longer runs its
rounding algorithm.

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from src.edge_deletion import delete_edges
from src.experiment_helpers import (
    compute_all_pairs_data,
    compute_min_max_cc_data,
    compute_min_max_lp_data,
    empty_pivot_results,
    run_pivot_multiple,
)
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
)

DEFAULT_EGO_IDS = [
    "0",
    "107",
    "348",
    "414",
    "686",
    "698",
    "1684",
    "1912",
    "3437",
    "3980",
]

DEFAULT_P_DELETE_VALUES = [0.05, 0.15, 0.25, 0.40]
DEFAULT_DELETION_SEEDS = list(range(1, 31))
DEFAULT_PIVOT_SEEDS = list(range(1, 101))

# These sets reproduce which algorithms are populated in the supplied table.
DEFAULT_PIVOT_EGOS = {"414", "686", "698", "3980"}
DEFAULT_ALL_PAIRS_LP_EGOS = {"414", "686", "698", "3980"}
DEFAULT_ALL_PAIRS_ILP_EGOS = {"414", "698", "3980"}
DEFAULT_MIN_MAX_LP_EGOS = {"414", "686", "698", "3980"}

DEFAULT_D_HAT = 8
DEFAULT_LAMBDA = 5
DEFAULT_MIN_MAX_LP_R = 0.4
DEFAULT_MIN_MAX_LP_R2 = 0.4
DEFAULT_MIN_MAX_LP_METHOD = 2


FIELDNAMES = [
    "ego_id",
    "n",
    "seed",
    "complete_pivot_best_cost",
    "complete_pivot_average_cost",
    "complete_ilp_cost",
    "complete_lp_cost",
    "p_delete",
    "edge_pivot_best_cost",
    "edge_pivot_average_cost",
    "edge_all_pairs_ilp_cost",
    "edge_all_pairs_lp_cost",

    "complete_min_max_cc_computed",
    "complete_min_max_cc_cluster_count",
    "complete_min_max_cc_max_disagreement",
    "complete_min_max_cc_d_hat",
    "complete_min_max_cc_lambda",
    "complete_min_max_cc_runtime_seconds",

    "complete_min_max_lp_computed",
    "complete_min_max_lp_cost",
    "complete_min_max_lp_rounding_cost",
    "complete_min_max_lp_max_disagreement_vertex",
    "complete_min_max_lp_cluster_count",
    "complete_min_max_lp_r",
    "complete_min_max_lp_r2",
    "complete_min_max_lp_method",
    "complete_min_max_lp_norm",
    "complete_min_max_lp_runtime_seconds",
    "complete_min_max_lp_rounding_runtime_seconds",
    "complete_min_max_lp_total_runtime_seconds",

    "edge_min_max_cc_computed",
    "edge_min_max_cc_cluster_count",
    "edge_min_max_cc_max_disagreement",
    "edge_min_max_cc_d_hat",
    "edge_min_max_cc_lambda",
    "edge_min_max_cc_runtime_seconds",

    "edge_min_max_lp_computed",
    "edge_min_max_lp_cost",
    "edge_min_max_lp_rounding_cost",
    "edge_min_max_lp_max_disagreement_vertex",
    "edge_min_max_lp_cluster_count",
    "edge_min_max_lp_r",
    "edge_min_max_lp_r2",
    "edge_min_max_lp_method",
    "edge_min_max_lp_norm",
    "edge_min_max_lp_runtime_seconds",
    "edge_min_max_lp_rounding_runtime_seconds",
    "edge_min_max_lp_total_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce results/research_tables/"
            "minmax_facebook_grid_runs_flat.csv."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--ego-ids",
        default=",".join(DEFAULT_EGO_IDS),
        help="Comma-separated Facebook ego IDs.",
    )
    parser.add_argument(
        "--p-delete-values",
        default="0.05,0.15,0.25,0.40",
        help="Comma-separated edge-deletion probabilities.",
    )
    parser.add_argument(
        "--seeds",
        default="1-30",
        help="Deletion seeds, for example 1-30 or 1,5,10.",
    )
    parser.add_argument(
        "--pivot-seeds",
        default="1-100",
        help="Pivot seeds, for example 1-100.",
    )

    parser.add_argument(
        "--pivot-egos",
        default="414,686,698,3980",
        help="Ego IDs on which Pivot is run. Use an empty string for none.",
    )
    parser.add_argument(
        "--all-pairs-lp-egos",
        default="414,686,698,3980",
        help=(
            "Ego IDs on which the standard all-pairs LP relaxation is run."
        ),
    )
    parser.add_argument(
        "--all-pairs-ilp-egos",
        default="414,698,3980",
        help="Ego IDs on which the standard all-pairs ILP is run.",
    )
    parser.add_argument(
        "--min-max-lp-egos",
        default="414,686,698,3980",
        help="Ego IDs on which MinMaxLP is run.",
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
        default=DEFAULT_MIN_MAX_LP_R,
    )
    parser.add_argument(
        "--min-max-lp-r2",
        type=float,
        default=DEFAULT_MIN_MAX_LP_R2,
    )
    parser.add_argument(
        "--min-max-lp-method",
        type=int,
        default=DEFAULT_MIN_MAX_LP_METHOD,
    )
    parser.add_argument(
        "--all-pairs-time-limit",
        type=float,
        default=None,
        help=(
            "Optional Gurobi time limit in seconds for each all-pairs "
            "LP/ILP solve."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Start from scratch instead of resuming existing rows. "
            "The existing CSV is backed up first."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next instance after an error.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of new edge-deleted rows to compute.",
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


def parse_csv_set(raw: str) -> set[str]:
    return set(parse_csv_list(raw))


def parse_float_list(raw: str) -> list[float]:
    values = sorted(
        {
            float(value.strip())
            for value in raw.split(",")
            if value.strip()
        }
    )
    if not values:
        raise ValueError("At least one p_delete value is required.")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Every p_delete value must be between 0 and 1.")
    return values


def parse_integer_spec(raw: str) -> list[int]:
    values: list[int] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)

            if end < start:
                raise ValueError(f"Invalid integer range: {part}")

            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    values = sorted(set(values))

    if not values:
        raise ValueError("At least one integer value is required.")

    return values


def scalar_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return str(value)
    return str(value)


def normalized_p_delete(value: Any) -> str:
    return f"{float(value):.8f}"


def row_key(
    ego_id: Any,
    p_delete: Any,
    seed: Any,
) -> tuple[str, str, int]:
    return (
        str(ego_id).strip(),
        normalized_p_delete(p_delete),
        int(float(seed)),
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
        f"Could not find a .edges file for Facebook ego {ego_id}. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def reconstruct_facebook_matrix(ego_id: str) -> np.ndarray:
    """Build the signed matrix from edge-file nodes only."""
    edge_nodes, facebook_edges = load_facebook_ego_edges(
        str(locate_edges_file(ego_id))
    )

    # Important: circle-only metadata nodes are not graph vertices.
    nodes = sorted(edge_nodes)

    matrix, _, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            nodes,
            facebook_edges,
        )
    )

    return matrix


def empty_min_max_lp_result(
    r: float,
    r2: float,
    method: int,
) -> dict[str, Any]:
    return {
        "computed": False,
        "lp_cost": None,
        "r": r,
        "r2": r2,
        "method": method,
        "norm": "inf",
        "lp_runtime_seconds": None,
    }


def empty_all_pairs_result() -> dict[str, Any]:
    return {
        "lp_cost": None,
        "ilp_cost": None,
    }


def flatten_min_max_cc(
    prefix: str,
    result: Mapping[str, Any],
) -> dict[str, str]:
    return {
        f"{prefix}_min_max_cc_computed": scalar_cell(
            result.get("computed")
        ),
        f"{prefix}_min_max_cc_cluster_count": scalar_cell(
            result.get("cluster_count")
        ),
        f"{prefix}_min_max_cc_max_disagreement": scalar_cell(
            result.get("max_disagreement")
        ),
        f"{prefix}_min_max_cc_d_hat": scalar_cell(
            result.get("d_hat")
        ),
        f"{prefix}_min_max_cc_lambda": scalar_cell(
            result.get("lambda")
        ),
        f"{prefix}_min_max_cc_runtime_seconds": scalar_cell(
            result.get("runtime_seconds")
        ),
    }


def flatten_min_max_lp(
    prefix: str,
    result: Mapping[str, Any] | None,
    selected: bool,
) -> dict[str, str]:
    if not selected or result is None:
        return {
            f"{prefix}_min_max_lp_computed": "",
            f"{prefix}_min_max_lp_cost": "",
            f"{prefix}_min_max_lp_rounding_cost": "",
            f"{prefix}_min_max_lp_max_disagreement_vertex": "",
            f"{prefix}_min_max_lp_cluster_count": "",
            f"{prefix}_min_max_lp_r": "",
            f"{prefix}_min_max_lp_r2": "",
            f"{prefix}_min_max_lp_method": "",
            f"{prefix}_min_max_lp_norm": "",
            f"{prefix}_min_max_lp_runtime_seconds": "",
            f"{prefix}_min_max_lp_rounding_runtime_seconds": "",
            f"{prefix}_min_max_lp_total_runtime_seconds": "",
        }

    # Rounding-related fields stay blank by design.
    return {
        f"{prefix}_min_max_lp_computed": scalar_cell(
            result.get("computed")
        ),
        f"{prefix}_min_max_lp_cost": scalar_cell(
            result.get("lp_cost")
        ),
        f"{prefix}_min_max_lp_rounding_cost": "",
        f"{prefix}_min_max_lp_max_disagreement_vertex": "",
        f"{prefix}_min_max_lp_cluster_count": "",
        f"{prefix}_min_max_lp_r": scalar_cell(
            result.get("r")
        ),
        f"{prefix}_min_max_lp_r2": scalar_cell(
            result.get("r2")
        ),
        f"{prefix}_min_max_lp_method": scalar_cell(
            result.get("method")
        ),
        f"{prefix}_min_max_lp_norm": scalar_cell(
            result.get("norm")
        ),
        f"{prefix}_min_max_lp_runtime_seconds": scalar_cell(
            result.get("lp_runtime_seconds")
        ),
        f"{prefix}_min_max_lp_rounding_runtime_seconds": "",
        f"{prefix}_min_max_lp_total_runtime_seconds": scalar_cell(
            result.get("lp_runtime_seconds")
        ),
    }


def read_existing_rows(
    path: Path,
) -> dict[tuple[str, str, int], dict[str, str]]:
    if not path.exists():
        return {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        rows: dict[
            tuple[str, str, int],
            dict[str, str],
        ] = {}

        for original_row in reader:
            ego_id = str(
                original_row.get("ego_id", "")
            ).strip()
            p_delete = str(
                original_row.get("p_delete", "")
            ).strip()
            seed = str(
                original_row.get("seed", "")
            ).strip()

            if not ego_id or not p_delete or not seed:
                continue

            normalized_row = {
                field: original_row.get(field, "")
                for field in FIELDNAMES
            }

            rows[row_key(ego_id, p_delete, seed)] = (
                normalized_row
            )

    return rows


def atomic_write(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
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


def ordered_rows(
    rows_by_key: Mapping[
        tuple[str, str, int],
        Mapping[str, Any],
    ],
    ego_ids: list[str],
    p_delete_values: list[float],
    seeds: list[int],
) -> list[Mapping[str, Any]]:
    output = []

    for ego_id in ego_ids:
        for p_delete in p_delete_values:
            for seed in seeds:
                key = row_key(ego_id, p_delete, seed)
                if key in rows_by_key:
                    output.append(rows_by_key[key])

    # Preserve any unrelated pre-existing rows after the requested grid.
    requested = {
        row_key(ego_id, p_delete, seed)
        for ego_id in ego_ids
        for p_delete in p_delete_values
        for seed in seeds
    }

    for key in sorted(rows_by_key):
        if key not in requested:
            output.append(rows_by_key[key])

    return output


def row_is_complete(
    row: Mapping[str, str],
    ego_id: str,
    pivot_egos: set[str],
    all_pairs_lp_egos: set[str],
    all_pairs_ilp_egos: set[str],
    min_max_lp_egos: set[str],
    d_hat: int,
    lambda_value: int,
) -> bool:
    required = [
        "complete_min_max_cc_max_disagreement",
        "complete_min_max_cc_runtime_seconds",
        "edge_min_max_cc_max_disagreement",
        "edge_min_max_cc_runtime_seconds",
    ]

    if ego_id in pivot_egos:
        required.extend([
            "complete_pivot_best_cost",
            "complete_pivot_average_cost",
            "edge_pivot_best_cost",
            "edge_pivot_average_cost",
        ])

    if ego_id in all_pairs_lp_egos:
        required.extend([
            "complete_lp_cost",
            "edge_all_pairs_lp_cost",
        ])

    if ego_id in all_pairs_ilp_egos:
        required.extend([
            "complete_ilp_cost",
            "edge_all_pairs_ilp_cost",
        ])

    if ego_id in min_max_lp_egos:
        required.extend([
            "complete_min_max_lp_cost",
            "complete_min_max_lp_runtime_seconds",
            "edge_min_max_lp_cost",
            "edge_min_max_lp_runtime_seconds",
        ])

    if any(not str(row.get(field, "")).strip() for field in required):
        return False

    try:
        complete_d_hat = int(
            float(row["complete_min_max_cc_d_hat"])
        )
        complete_lambda = int(
            float(row["complete_min_max_cc_lambda"])
        )
        edge_d_hat = int(
            float(row["edge_min_max_cc_d_hat"])
        )
        edge_lambda = int(
            float(row["edge_min_max_cc_lambda"])
        )
    except (KeyError, TypeError, ValueError):
        return False

    return (
        complete_d_hat == d_hat
        and edge_d_hat == d_hat
        and complete_lambda == lambda_value
        and edge_lambda == lambda_value
    )


def first_existing_complete_row(
    rows_by_key: Mapping[
        tuple[str, str, int],
        Mapping[str, str],
    ],
    ego_id: str,
) -> Mapping[str, str] | None:
    for key, row in rows_by_key.items():
        if key[0] == ego_id:
            return row
    return None


def load_or_compute_complete_results(
    matrix: np.ndarray,
    ego_id: str,
    existing_row: Mapping[str, str] | None,
    pivot_seeds: list[int],
    pivot_selected: bool,
    all_pairs_lp_selected: bool,
    all_pairs_ilp_selected: bool,
    min_max_lp_selected: bool,
    d_hat: int,
    lambda_value: int,
    min_max_lp_r: float,
    min_max_lp_r2: float,
    min_max_lp_method: int,
    all_pairs_time_limit: float | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Reuse complete values from the output when possible."""
    can_reuse_cc = (
        existing_row is not None
        and str(
            existing_row.get(
                "complete_min_max_cc_max_disagreement",
                "",
            )
        ).strip()
        and str(
            existing_row.get(
                "complete_min_max_cc_runtime_seconds",
                "",
            )
        ).strip()
        and str(
            existing_row.get(
                "complete_min_max_cc_d_hat",
                "",
            )
        ).strip()
        and str(
            existing_row.get(
                "complete_min_max_cc_lambda",
                "",
            )
        ).strip()
        and int(
            float(
                existing_row[
                    "complete_min_max_cc_d_hat"
                ]
            )
        ) == d_hat
        and int(
            float(
                existing_row[
                    "complete_min_max_cc_lambda"
                ]
            )
        ) == lambda_value
    )

    if can_reuse_cc:
        complete_cc = {
            "computed": True,
            "cluster_count": int(
                float(
                    existing_row[
                        "complete_min_max_cc_cluster_count"
                    ]
                )
            ),
            "max_disagreement": float(
                existing_row[
                    "complete_min_max_cc_max_disagreement"
                ]
            ),
            "d_hat": d_hat,
            "lambda": lambda_value,
            "runtime_seconds": float(
                existing_row[
                    "complete_min_max_cc_runtime_seconds"
                ]
            ),
        }
        print("  Reusing complete MinMaxCC.")
    else:
        print("  Computing complete MinMaxCC.")
        complete_cc = compute_min_max_cc_data(
            matrix,
            compute_min_max=True,
            param_1=d_hat,
            param_2=lambda_value,
        )

    can_reuse_pivot = (
        pivot_selected
        and existing_row is not None
        and str(
            existing_row.get(
                "complete_pivot_best_cost",
                "",
            )
        ).strip()
        and str(
            existing_row.get(
                "complete_pivot_average_cost",
                "",
            )
        ).strip()
    )

    if can_reuse_pivot:
        complete_pivot = empty_pivot_results()
        complete_pivot.update({
            "computed": True,
            "best_cost": float(
                existing_row[
                    "complete_pivot_best_cost"
                ]
            ),
            "average_cost": float(
                existing_row[
                    "complete_pivot_average_cost"
                ]
            ),
        })
        print("  Reusing complete Pivot.")
    elif pivot_selected:
        print("  Computing complete Pivot.")
        complete_pivot = run_pivot_multiple(
            matrix,
            pivot_seeds,
        )
    else:
        complete_pivot = empty_pivot_results()

    can_reuse_all_pairs = (
        existing_row is not None
        and (
            not all_pairs_lp_selected
            or str(
                existing_row.get(
                    "complete_lp_cost",
                    "",
                )
            ).strip()
        )
        and (
            not all_pairs_ilp_selected
            or str(
                existing_row.get(
                    "complete_ilp_cost",
                    "",
                )
            ).strip()
        )
    )

    if can_reuse_all_pairs:
        complete_all_pairs = {
            "lp_cost": (
                float(existing_row["complete_lp_cost"])
                if all_pairs_lp_selected
                else None
            ),
            "ilp_cost": (
                float(existing_row["complete_ilp_cost"])
                if all_pairs_ilp_selected
                else None
            ),
        }
        if all_pairs_lp_selected or all_pairs_ilp_selected:
            print("  Reusing complete all-pairs result(s).")
    elif all_pairs_lp_selected or all_pairs_ilp_selected:
        print("  Computing complete all-pairs result(s).")
        complete_all_pairs = compute_all_pairs_data(
            matrix,
            compute_lp=all_pairs_lp_selected,
            compute_ilp=all_pairs_ilp_selected,
            time_limit=all_pairs_time_limit,
        )
    else:
        complete_all_pairs = empty_all_pairs_result()

    can_reuse_min_max_lp = (
        min_max_lp_selected
        and existing_row is not None
        and str(
            existing_row.get(
                "complete_min_max_lp_cost",
                "",
            )
        ).strip()
        and str(
            existing_row.get(
                "complete_min_max_lp_runtime_seconds",
                "",
            )
        ).strip()
    )

    if can_reuse_min_max_lp:
        complete_min_max_lp = {
            "computed": True,
            "lp_cost": float(
                existing_row[
                    "complete_min_max_lp_cost"
                ]
            ),
            "r": float(
                existing_row.get(
                    "complete_min_max_lp_r",
                    min_max_lp_r,
                )
                or min_max_lp_r
            ),
            "r2": float(
                existing_row.get(
                    "complete_min_max_lp_r2",
                    min_max_lp_r2,
                )
                or min_max_lp_r2
            ),
            "method": int(
                float(
                    existing_row.get(
                        "complete_min_max_lp_method",
                        min_max_lp_method,
                    )
                    or min_max_lp_method
                )
            ),
            "norm": (
                existing_row.get(
                    "complete_min_max_lp_norm",
                    "inf",
                )
                or "inf"
            ),
            "lp_runtime_seconds": float(
                existing_row[
                    "complete_min_max_lp_runtime_seconds"
                ]
            ),
        }
        print("  Reusing complete MinMaxLP.")
    elif min_max_lp_selected:
        print("  Computing complete MinMaxLP.")
        complete_min_max_lp = compute_min_max_lp_data(
            matrix,
            compute_min_max_lp=True,
            r=min_max_lp_r,
            r2=min_max_lp_r2,
            method=min_max_lp_method,
            norm=np.inf,
        )
    else:
        complete_min_max_lp = empty_min_max_lp_result(
            min_max_lp_r,
            min_max_lp_r2,
            min_max_lp_method,
        )

    return (
        complete_pivot,
        complete_all_pairs,
        complete_cc,
        complete_min_max_lp,
    )


def make_row(
    ego_id: str,
    n: int,
    seed: int,
    p_delete: float,
    complete_pivot: Mapping[str, Any],
    edge_pivot: Mapping[str, Any],
    complete_all_pairs: Mapping[str, Any],
    edge_all_pairs: Mapping[str, Any],
    complete_cc: Mapping[str, Any],
    edge_cc: Mapping[str, Any],
    complete_min_max_lp: Mapping[str, Any],
    edge_min_max_lp: Mapping[str, Any],
    pivot_selected: bool,
    all_pairs_lp_selected: bool,
    all_pairs_ilp_selected: bool,
    min_max_lp_selected: bool,
) -> dict[str, str]:
    row = {field: "" for field in FIELDNAMES}

    row.update({
        "ego_id": ego_id,
        "n": str(n),
        "seed": str(seed),
        "p_delete": scalar_cell(p_delete),

        "complete_pivot_best_cost": (
            scalar_cell(complete_pivot.get("best_cost"))
            if pivot_selected
            else ""
        ),
        "complete_pivot_average_cost": (
            scalar_cell(
                complete_pivot.get("average_cost")
            )
            if pivot_selected
            else ""
        ),
        "edge_pivot_best_cost": (
            scalar_cell(edge_pivot.get("best_cost"))
            if pivot_selected
            else ""
        ),
        "edge_pivot_average_cost": (
            scalar_cell(edge_pivot.get("average_cost"))
            if pivot_selected
            else ""
        ),

        "complete_lp_cost": (
            scalar_cell(
                complete_all_pairs.get("lp_cost")
            )
            if all_pairs_lp_selected
            else ""
        ),
        "edge_all_pairs_lp_cost": (
            scalar_cell(edge_all_pairs.get("lp_cost"))
            if all_pairs_lp_selected
            else ""
        ),

        "complete_ilp_cost": (
            scalar_cell(
                complete_all_pairs.get("ilp_cost")
            )
            if all_pairs_ilp_selected
            else ""
        ),
        "edge_all_pairs_ilp_cost": (
            scalar_cell(edge_all_pairs.get("ilp_cost"))
            if all_pairs_ilp_selected
            else ""
        ),
    })

    row.update(
        flatten_min_max_cc("complete", complete_cc)
    )
    row.update(flatten_min_max_cc("edge", edge_cc))

    row.update(
        flatten_min_max_lp(
            "complete",
            complete_min_max_lp,
            selected=min_max_lp_selected,
        )
    )
    row.update(
        flatten_min_max_lp(
            "edge",
            edge_min_max_lp,
            selected=min_max_lp_selected,
        )
    )

    return row


def main() -> None:
    args = parse_args()

    output = resolve(args.output)
    ego_ids = parse_csv_list(args.ego_ids)
    p_delete_values = parse_float_list(
        args.p_delete_values
    )
    deletion_seeds = parse_integer_spec(args.seeds)
    pivot_seeds = parse_integer_spec(args.pivot_seeds)

    pivot_egos = parse_csv_set(args.pivot_egos)
    all_pairs_lp_egos = parse_csv_set(
        args.all_pairs_lp_egos
    )
    all_pairs_ilp_egos = parse_csv_set(
        args.all_pairs_ilp_egos
    )
    min_max_lp_egos = parse_csv_set(
        args.min_max_lp_egos
    )

    if not ego_ids:
        raise ValueError("At least one ego ID is required.")
    if args.d_hat < 0:
        raise ValueError("d_hat must be non-negative.")
    if args.lambda_value <= 4:
        raise ValueError("lambda must be greater than 4.")

    requested_egos = set(ego_ids)
    for option_name, selected_egos in [
        ("pivot-egos", pivot_egos),
        ("all-pairs-lp-egos", all_pairs_lp_egos),
        ("all-pairs-ilp-egos", all_pairs_ilp_egos),
        ("min-max-lp-egos", min_max_lp_egos),
    ]:
        unknown = sorted(selected_egos - requested_egos)
        if unknown:
            raise ValueError(
                f"--{option_name} contains ego IDs not present "
                f"in --ego-ids: {', '.join(unknown)}"
            )

    if output.exists() and args.overwrite:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        backup = output.with_name(
            f"{output.stem}_before_reproduction_"
            f"{timestamp}{output.suffix}"
        )
        shutil.copy2(output, backup)
        print("Backup:", backup)
        rows_by_key: dict[
            tuple[str, str, int],
            dict[str, str],
        ] = {}
    else:
        rows_by_key = read_existing_rows(output)

    total_requested = (
        len(ego_ids)
        * len(p_delete_values)
        * len(deletion_seeds)
    )
    existing_requested = sum(
        row_key(ego_id, p_delete, seed)
        in rows_by_key
        for ego_id in ego_ids
        for p_delete in p_delete_values
        for seed in deletion_seeds
    )

    print("Output:", output)
    print("Ego IDs:", ego_ids)
    print("p_delete values:", p_delete_values)
    print("Deletion seeds:", deletion_seeds)
    print("Pivot seeds:", pivot_seeds)
    print("Pivot egos:", sorted(pivot_egos))
    print(
        "All-pairs LP egos:",
        sorted(all_pairs_lp_egos),
    )
    print(
        "All-pairs ILP egos:",
        sorted(all_pairs_ilp_egos),
    )
    print("MinMaxLP egos:", sorted(min_max_lp_egos))
    print(
        f"MinMaxCC parameters: d_hat={args.d_hat}, "
        f"lambda={args.lambda_value}"
    )
    print("Requested rows:", total_requested)
    print("Existing requested rows:", existing_requested)

    newly_computed = 0
    skipped = 0
    failures = 0

    for ego_index, ego_id in enumerate(
        ego_ids,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"EGO {ego_id} "
            f"({ego_index}/{len(ego_ids)})"
        )
        print("=" * 78)

        try:
            complete_matrix = reconstruct_facebook_matrix(
                ego_id
            )
            n = int(complete_matrix.shape[0])
            print("Corrected n:", n)

            pivot_selected = ego_id in pivot_egos
            all_pairs_lp_selected = (
                ego_id in all_pairs_lp_egos
            )
            all_pairs_ilp_selected = (
                ego_id in all_pairs_ilp_egos
            )
            min_max_lp_selected = (
                ego_id in min_max_lp_egos
            )

            existing_complete_row = (
                first_existing_complete_row(
                    rows_by_key,
                    ego_id,
                )
            )

            (
                complete_pivot,
                complete_all_pairs,
                complete_cc,
                complete_min_max_lp,
            ) = load_or_compute_complete_results(
                matrix=complete_matrix,
                ego_id=ego_id,
                existing_row=existing_complete_row,
                pivot_seeds=pivot_seeds,
                pivot_selected=pivot_selected,
                all_pairs_lp_selected=(
                    all_pairs_lp_selected
                ),
                all_pairs_ilp_selected=(
                    all_pairs_ilp_selected
                ),
                min_max_lp_selected=(
                    min_max_lp_selected
                ),
                d_hat=args.d_hat,
                lambda_value=args.lambda_value,
                min_max_lp_r=args.min_max_lp_r,
                min_max_lp_r2=args.min_max_lp_r2,
                min_max_lp_method=(
                    args.min_max_lp_method
                ),
                all_pairs_time_limit=(
                    args.all_pairs_time_limit
                ),
            )

        except Exception as error:
            failures += 1
            print(
                f"FAILED while preparing ego {ego_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            if args.continue_on_error:
                continue
            raise

        for p_delete in p_delete_values:
            for seed in deletion_seeds:
                key = row_key(
                    ego_id,
                    p_delete,
                    seed,
                )
                existing_row = rows_by_key.get(key)

                if (
                    existing_row is not None
                    and row_is_complete(
                        existing_row,
                        ego_id=ego_id,
                        pivot_egos=pivot_egos,
                        all_pairs_lp_egos=(
                            all_pairs_lp_egos
                        ),
                        all_pairs_ilp_egos=(
                            all_pairs_ilp_egos
                        ),
                        min_max_lp_egos=(
                            min_max_lp_egos
                        ),
                        d_hat=args.d_hat,
                        lambda_value=(
                            args.lambda_value
                        ),
                    )
                ):
                    skipped += 1
                    print(
                        f"SKIP ego={ego_id} "
                        f"p_delete={p_delete:g} "
                        f"seed={seed}"
                    )
                    continue

                if (
                    args.limit is not None
                    and newly_computed >= args.limit
                ):
                    print(
                        "\nLimit reached. Saving progress."
                    )
                    atomic_write(
                        output,
                        ordered_rows(
                            rows_by_key,
                            ego_ids,
                            p_delete_values,
                            deletion_seeds,
                        ),
                    )
                    return

                label = (
                    f"ego={ego_id} "
                    f"p_delete={p_delete:g} "
                    f"seed={seed}"
                )
                print("\nRUN", label, flush=True)

                try:
                    edge_matrix, num_deleted = (
                        delete_edges(
                            complete_matrix,
                            p_delete,
                            seed,
                        )
                    )
                    print(
                        "  Deleted edges:",
                        num_deleted,
                        flush=True,
                    )

                    edge_cc = compute_min_max_cc_data(
                        edge_matrix,
                        compute_min_max=True,
                        param_1=args.d_hat,
                        param_2=args.lambda_value,
                    )
                    print(
                        "  MinMaxCC:",
                        edge_cc["max_disagreement"],
                        "| runtime:",
                        edge_cc["runtime_seconds"],
                        flush=True,
                    )

                    if pivot_selected:
                        edge_pivot = run_pivot_multiple(
                            edge_matrix,
                            pivot_seeds,
                        )
                        print(
                            "  Pivot best/average:",
                            edge_pivot["best_cost"],
                            edge_pivot["average_cost"],
                            flush=True,
                        )
                    else:
                        edge_pivot = (
                            empty_pivot_results()
                        )

                    if (
                        all_pairs_lp_selected
                        or all_pairs_ilp_selected
                    ):
                        edge_all_pairs = (
                            compute_all_pairs_data(
                                edge_matrix,
                                compute_lp=(
                                    all_pairs_lp_selected
                                ),
                                compute_ilp=(
                                    all_pairs_ilp_selected
                                ),
                                time_limit=(
                                    args.all_pairs_time_limit
                                ),
                            )
                        )
                        print(
                            "  All-pairs LP/ILP:",
                            edge_all_pairs["lp_cost"],
                            edge_all_pairs["ilp_cost"],
                            flush=True,
                        )
                    else:
                        edge_all_pairs = (
                            empty_all_pairs_result()
                        )

                    if min_max_lp_selected:
                        edge_min_max_lp = (
                            compute_min_max_lp_data(
                                edge_matrix,
                                compute_min_max_lp=True,
                                r=args.min_max_lp_r,
                                r2=args.min_max_lp_r2,
                                method=(
                                    args.min_max_lp_method
                                ),
                                norm=np.inf,
                            )
                        )
                        print(
                            "  MinMaxLP:",
                            edge_min_max_lp["lp_cost"],
                            "| runtime:",
                            edge_min_max_lp[
                                "lp_runtime_seconds"
                            ],
                            flush=True,
                        )
                    else:
                        edge_min_max_lp = (
                            empty_min_max_lp_result(
                                args.min_max_lp_r,
                                args.min_max_lp_r2,
                                args.min_max_lp_method,
                            )
                        )

                    rows_by_key[key] = make_row(
                        ego_id=ego_id,
                        n=n,
                        seed=seed,
                        p_delete=p_delete,
                        complete_pivot=(
                            complete_pivot
                        ),
                        edge_pivot=edge_pivot,
                        complete_all_pairs=(
                            complete_all_pairs
                        ),
                        edge_all_pairs=edge_all_pairs,
                        complete_cc=complete_cc,
                        edge_cc=edge_cc,
                        complete_min_max_lp=(
                            complete_min_max_lp
                        ),
                        edge_min_max_lp=(
                            edge_min_max_lp
                        ),
                        pivot_selected=pivot_selected,
                        all_pairs_lp_selected=(
                            all_pairs_lp_selected
                        ),
                        all_pairs_ilp_selected=(
                            all_pairs_ilp_selected
                        ),
                        min_max_lp_selected=(
                            min_max_lp_selected
                        ),
                    )

                    newly_computed += 1

                    # Save after every row so a long run can be resumed.
                    atomic_write(
                        output,
                        ordered_rows(
                            rows_by_key,
                            ego_ids,
                            p_delete_values,
                            deletion_seeds,
                        ),
                    )

                except Exception as error:
                    failures += 1
                    print(
                        f"FAILED {label}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        raise

    final_rows = ordered_rows(
        rows_by_key,
        ego_ids,
        p_delete_values,
        deletion_seeds,
    )
    atomic_write(output, final_rows)

    final_requested = sum(
        row_key(ego_id, p_delete, seed)
        in rows_by_key
        for ego_id in ego_ids
        for p_delete in p_delete_values
        for seed in deletion_seeds
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Output:", output)
    print("Requested rows present:", final_requested)
    print("Expected requested rows:", total_requested)
    print("New rows computed:", newly_computed)
    print("Rows skipped as complete:", skipped)
    print("Failures:", failures)

    if failures == 0 and final_requested != total_requested:
        raise RuntimeError(
            "The run finished without reported failures, but the "
            "output does not contain every requested row."
        )


if __name__ == "__main__":
    main()
