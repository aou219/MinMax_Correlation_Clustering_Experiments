#!/usr/bin/env python3
"""
Create a Facebook-only min-max template from results/processed/all_runs_flat.csv.

This template contains only the base Facebook rows. It does NOT store full
clusterings, disagreement vectors, graph_family, or file_path.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("results/processed/all_runs_flat.csv")
DEFAULT_OUTPUT = Path("results/processed/minmax_facebook_template_flat.csv")

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

# No full clustering columns and no disagreement-vector column.
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
        description="Create a Facebook-only min-max template."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def is_facebook_row(row: dict[str, str]) -> bool:
    return row.get("graph_family", "").strip().lower() == "facebook"


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Use --overwrite only when you intentionally want to recreate it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)

        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {args.input}")

        missing = [column for column in BASE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                "Required columns missing from all_runs_flat.csv: "
                + ", ".join(missing)
            )

        row_count = 0

        with args.output.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for source_row in reader:
                if not is_facebook_row(source_row):
                    continue

                output_row = {
                    column: source_row.get(column, "")
                    for column in BASE_COLUMNS
                }

                output_row.update({
                    column: ""
                    for column in MINMAX_COLUMNS
                })

                writer.writerow(output_row)
                row_count += 1

    print(f"Created {args.output}")
    print(f"Facebook rows: {row_count}")
    print(f"Columns: {len(OUTPUT_COLUMNS)}")


if __name__ == "__main__":
    main()
