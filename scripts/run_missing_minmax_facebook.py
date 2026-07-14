#!/usr/bin/env python3
"""
Run missing min-max results for Facebook graphs, with a grid over min_max_cc
parameters.

Important:
- MinMaxLP is run only ONCE per original Facebook row for the complete graph
  and only ONCE per original Facebook row for the edge-deleted graph.
- min_max_cc is run for every (d_hat, lambda) pair.
- Full clusterings and disagreement vectors are NOT stored.
- Output is written to a separate expanded CSV, so the 16-row template stays safe.

Default input:
    results/processed/minmax_facebook_template_flat.csv

Default output:
    results/processed/minmax_facebook_grid_runs_flat.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from src.edge_deletion import delete_edges  # noqa: E402
from src.experiment_helpers import (  # noqa: E402
    compute_min_max_cc_data,
    compute_min_max_lp_data,
)
from src.facebook_sampling import (  # noqa: E402
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_circles,
    load_facebook_ego_edges,
)


DEFAULT_INPUT_CSV = REPO_ROOT / "results/processed/minmax_facebook_template_flat.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results/processed/minmax_facebook_grid_runs_flat.csv"

BASE_COLUMNS = [
    "ego_id",
    "n",
    "complete_pivot_best_cost",
    "complete_pivot_average_cost",
    "complete_ilp_cost",
    "complete_lp_cost",
    "p_delete",
    "edge_pivot_best_cost",
    "edge_pivot_average_cost",
    "edge_all_pairs_ilp_cost",
    "edge_all_pairs_lp_cost",
]

# No clustering columns and no disagreement-vector column.
MINMAX_SUFFIXES = [
    "min_max_cc_computed",
    "min_max_cc_cluster_count",
    "min_max_cc_max_disagreement",
    "min_max_cc_d_hat",
    "min_max_cc_lambda",
    "min_max_cc_runtime_seconds",

    "min_max_lp_computed",
    "min_max_lp_cost",
    "min_max_lp_rounding_cost",
    "min_max_lp_max_disagreement_vertex",
    "min_max_lp_cluster_count",
    "min_max_lp_r",
    "min_max_lp_r2",
    "min_max_lp_method",
    "min_max_lp_norm",
    "min_max_lp_runtime_seconds",
    "min_max_lp_rounding_runtime_seconds",
    "min_max_lp_total_runtime_seconds",
]

MINMAX_COLUMNS = [
    f"{prefix}_{suffix}"
    for prefix in ("complete", "edge")
    for suffix in MINMAX_SUFFIXES
]

OUTPUT_COLUMNS = BASE_COLUMNS + MINMAX_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Facebook min-max grid results without storing clusterings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)

    # MinMaxLP parameters. These are not part of the cc grid and are run once per row.
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--r2", type=float, default=0.4)
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument(
        "--norm",
        default="inf",
        help="Use 'inf' for max disagreement or a numeric p-norm.",
    )

    # min_max_cc parameter grid.
    parser.add_argument(
        "--lambda-values",
        default="5,8,12",
        help="Comma-separated lambda values for min_max_cc. Example: 5,8,12",
    )
    parser.add_argument(
        "--min-d-hat",
        type=int,
        default=1,
        help="Smallest d_hat power of 2 to include. Default is 1.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of original Facebook rows to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a row fails.",
    )
    return parser.parse_args()


def parse_norm(raw: str) -> float:
    if raw.lower() in {"inf", "infinity", "math.inf", "np.inf"}:
        return math.inf
    return float(raw)


def parse_required_int(value: str | None, name: str) -> int:
    if value in ("", None):
        raise ValueError(f"Missing required integer field: {name}")
    return int(float(value))


def parse_required_float(value: str | None, name: str) -> float:
    if value in ("", None):
        raise ValueError(f"Missing required float field: {name}")
    return float(value)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one lambda value is required.")
    if any(value < 4 for value in values):
        raise ValueError("All lambda values should be 4 or bigger.")
    return values


def scalar_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        return str(float(value))
    return str(value)


def max_positive_degree(S: np.ndarray) -> int:
    """
    Maximum number of positive observed edges incident to any node.
    Positive edges have value 1 in the signed matrix.
    """
    return int((S == 1).sum(axis=1).max())


def powers_of_two_up_to(max_value: int, min_value: int = 1) -> list[int]:
    """
    Powers of two up to max_value, starting at at least min_value.
    Example: max_value=17 gives [1, 2, 4, 8, 16].
    """
    if max_value < 1:
        return [1]

    values: list[int] = []
    current = 1

    while current <= max_value:
        if current >= min_value:
            values.append(current)
        current *= 2

    return values or [1]


def locate_facebook_file(ego_id: str, extension: str) -> Path:
    candidates = [
        REPO_ROOT / f"data/facebook/{ego_id}.{extension}",
        REPO_ROOT / f"data/facebook/facebook_3/{ego_id}.{extension}",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find Facebook {extension} file for ego {ego_id}. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def get_all_nodes_from_edges_and_circles(
    edge_nodes: set[Any],
    circles: list[dict[str, Any]],
) -> list[Any]:
    circle_nodes: set[Any] = set()

    for circle in circles:
        circle_nodes.update(circle["nodes"])

    return sorted(edge_nodes | circle_nodes)


def reconstruct_facebook_matrix(row: dict[str, str]) -> np.ndarray:
    ego_id = str(row.get("ego_id", "")).strip()

    if not ego_id:
        raise ValueError("Missing ego_id for Facebook row")

    edges_file = locate_facebook_file(ego_id, "edges")
    circles_file = locate_facebook_file(ego_id, "circles")

    edge_nodes, facebook_edges = load_facebook_ego_edges(str(edges_file))
    circles = load_facebook_circles(str(circles_file))

    all_nodes = get_all_nodes_from_edges_and_circles(edge_nodes, circles)

    matrix, _, _, _ = build_complete_signed_matrix_from_facebook_sample(
        all_nodes,
        facebook_edges,
    )

    return matrix


def flatten_results(
    prefix: str,
    cc_result: dict[str, Any],
    lp_result: dict[str, Any],
    d_hat: int,
    lambda_param: int,
) -> dict[str, str]:
    return {
        f"{prefix}_min_max_cc_computed": scalar_cell(
            cc_result["max_disagreement"] is not None
        ),
        f"{prefix}_min_max_cc_cluster_count": scalar_cell(
            cc_result["cluster_count"]
        ),
        f"{prefix}_min_max_cc_max_disagreement": scalar_cell(
            cc_result["max_disagreement"]
        ),
        f"{prefix}_min_max_cc_d_hat": scalar_cell(d_hat),
        f"{prefix}_min_max_cc_lambda": scalar_cell(lambda_param),
        f"{prefix}_min_max_cc_runtime_seconds": scalar_cell(
            cc_result["runtime_seconds"]
        ),

        f"{prefix}_min_max_lp_computed": scalar_cell(
            lp_result["lp_cost"] is not None
        ),
        f"{prefix}_min_max_lp_cost": scalar_cell(lp_result["lp_cost"]),
        f"{prefix}_min_max_lp_rounding_cost": scalar_cell(
            lp_result["rounding_cost"]
        ),
        f"{prefix}_min_max_lp_max_disagreement_vertex": scalar_cell(
            lp_result["max_disagreement_vertex"]
        ),
        f"{prefix}_min_max_lp_cluster_count": scalar_cell(
            lp_result["cluster_count"]
        ),
        f"{prefix}_min_max_lp_r": scalar_cell(lp_result["r"]),
        f"{prefix}_min_max_lp_r2": scalar_cell(lp_result["r2"]),
        f"{prefix}_min_max_lp_method": scalar_cell(lp_result["method"]),
        f"{prefix}_min_max_lp_norm": scalar_cell(lp_result["norm"]),
        f"{prefix}_min_max_lp_runtime_seconds": scalar_cell(
            lp_result["lp_runtime_seconds"]
        ),
        f"{prefix}_min_max_lp_rounding_runtime_seconds": scalar_cell(
            lp_result["rounding_runtime_seconds"]
        ),
        f"{prefix}_min_max_lp_total_runtime_seconds": scalar_cell(
            lp_result["total_runtime_seconds"]
        ),
    }


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def output_row_exists(
    output_rows: list[dict[str, str]],
    base_row: dict[str, str],
    d_hat: int,
    lambda_param: int,
) -> bool:
    """
    Resume support: skip a grid row if the output already has this
    ego_id, p_delete, d_hat, and lambda combination.
    """
    ego_id = base_row.get("ego_id", "")
    p_delete = base_row.get("p_delete", "")

    for row in output_rows:
        if (
            row.get("ego_id", "") == ego_id
            and row.get("p_delete", "") == p_delete
            and row.get("complete_min_max_cc_d_hat", "") == str(d_hat)
            and row.get("complete_min_max_cc_lambda", "") == str(lambda_param)
            and row.get("complete_min_max_lp_cost", "") != ""
            and row.get("edge_min_max_lp_cost", "") != ""
        ):
            return True

    return False


def load_existing_output(path: Path, overwrite: bool) -> list[dict[str, str]]:
    if overwrite or not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return list(reader)


def main() -> None:
    args = parse_args()
    norm = parse_norm(args.norm)
    lambda_values = parse_int_list(args.lambda_values)

    input_csv = args.input
    output_csv = args.output

    if not input_csv.is_absolute():
        input_csv = REPO_ROOT / input_csv
    if not output_csv.is_absolute():
        output_csv = REPO_ROOT / output_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if output_csv.exists() and args.overwrite:
        print(f"Overwriting existing output: {output_csv}")

    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")

        missing = [column for column in BASE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )

        input_rows = list(reader)

    output_rows = load_existing_output(output_csv, overwrite=args.overwrite)

    processed_original_rows = 0
    written_grid_rows = 0
    failures = 0

    for index, base_row in enumerate(input_rows):
        if args.limit is not None and processed_original_rows >= args.limit:
            break

        label = (
            f"source row {index + 1}/{len(input_rows)} | "
            f"ego_id={base_row.get('ego_id')} | "
            f"n={base_row.get('n')} | "
            f"p_delete={base_row.get('p_delete')}"
        )

        print("\n" + "=" * 80)
        print(label)
        print("=" * 80, flush=True)

        try:
            complete_matrix = reconstruct_facebook_matrix(base_row)

            p_delete = parse_required_float(base_row["p_delete"], "p_delete")
            # Your Facebook template no longer stores seed. The old full runs used seed 1.
            seed = parse_required_int(base_row.get("seed", "1"), "seed")

            edge_matrix, deleted_count = delete_edges(
                complete_matrix,
                p_delete,
                seed,
            )

            print(f"Deleted edges reconstructed: {deleted_count}", flush=True)

            max_d_hat = max(
                max_positive_degree(complete_matrix),
                max_positive_degree(edge_matrix),
            )
            d_hat_values = powers_of_two_up_to(
                max_d_hat,
                min_value=max(1, args.min_d_hat),
            )

            print(f"d_hat values: {d_hat_values}", flush=True)
            print(f"lambda values: {lambda_values}", flush=True)

            # MinMaxLP does not depend on d_hat or lambda.
            # Therefore, run it once for the complete graph and once for the
            # edge-deleted graph, then reuse the results for every cc grid row.
            print("Running MinMaxLP once for complete graph...", flush=True)
            complete_lp = compute_min_max_lp_data(
                complete_matrix,
                compute_min_max_lp=True,
                r=args.r,
                r2=args.r2,
                method=args.method,
                norm=norm,
            )

            print("Running MinMaxLP once for edge-deleted graph...", flush=True)
            edge_lp = compute_min_max_lp_data(
                edge_matrix,
                compute_min_max_lp=True,
                r=args.r,
                r2=args.r2,
                method=args.method,
                norm=norm,
            )

            for d_hat in d_hat_values:
                for lambda_param in lambda_values:
                    if output_row_exists(
                        output_rows,
                        base_row,
                        d_hat=d_hat,
                        lambda_param=lambda_param,
                    ):
                        print(
                            f"Skipping existing grid row: "
                            f"d_hat={d_hat}, lambda={lambda_param}",
                            flush=True,
                        )
                        continue

                    print(
                        f"Running min_max_cc for d_hat={d_hat}, "
                        f"lambda={lambda_param}",
                        flush=True,
                    )

                    complete_cc = compute_min_max_cc_data(
                        complete_matrix,
                        compute_min_max=True,
                        param_1=d_hat,
                        param_2=lambda_param,
                    )

                    edge_cc = compute_min_max_cc_data(
                        edge_matrix,
                        compute_min_max=True,
                        param_1=d_hat,
                        param_2=lambda_param,
                    )

                    new_row = {
                        column: base_row.get(column, "")
                        for column in BASE_COLUMNS
                    }

                    new_row.update(
                        flatten_results(
                            "complete",
                            complete_cc,
                            complete_lp,
                            d_hat,
                            lambda_param,
                        )
                    )

                    new_row.update(
                        flatten_results(
                            "edge",
                            edge_cc,
                            edge_lp,
                            d_hat,
                            lambda_param,
                        )
                    )

                    output_rows.append(new_row)
                    written_grid_rows += 1

                    atomic_write_csv(output_csv, OUTPUT_COLUMNS, output_rows)
                    print(
                        f"Saved checkpoint. Total new grid rows: {written_grid_rows}",
                        flush=True,
                    )

            processed_original_rows += 1

        except Exception as error:
            failures += 1
            print(f"ERROR: {error}", file=sys.stderr, flush=True)

            if not args.continue_on_error:
                raise

    print("\nFinished.")
    print(f"Processed original rows this run: {processed_original_rows}")
    print(f"New grid rows written this run: {written_grid_rows}")
    print(f"Failed original rows this run: {failures}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
