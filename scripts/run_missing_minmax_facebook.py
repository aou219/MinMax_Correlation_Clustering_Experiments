"""
Run Facebook MinMaxLP and MinMaxCC experiments for multiple edge-deletion seeds.

Important:
- The complete Facebook graph is reconstructed once per ego graph.
- Complete MinMaxLP is computed once per ego graph.
- Complete MinMaxCC is computed once per (ego_id, d_hat, lambda).
- Edge deletion is repeated for every requested seed.
- Edge MinMaxLP is computed once per (ego_id, p_delete, seed).
- Edge MinMaxCC is computed for every
  (ego_id, p_delete, seed, d_hat, lambda).
- Existing rows are preserved and skipped, so this script can resume.
- Old output rows without a seed column are treated as seed 1.

Default seeds:
    1 through 30

Default ego order:
    3980, 698, 414, 686

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

DEFAULT_INPUT_CSV = (
    REPO_ROOT / "results/processed/all_runs_flat.csv"
)

DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "results/processed/research_tables/minmax_facebook_grid_runs_flat.csv"
)

BASE_COLUMNS = [
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
]

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
        description=(
            "Run Facebook MinMaxLP and MinMaxCC for multiple "
            "edge-deletion seeds."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)

    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--r2", type=float, default=0.4)
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument(
        "--norm",
        default="inf",
        help="Use 'inf' for max disagreement or a numeric p-norm.",
    )

    parser.add_argument(
        "--lambda-values",
        default="5,8,12",
        help="Comma-separated lambda values.",
    )
    parser.add_argument(
        "--min-d-hat",
        type=int,
        default=1,
        help="Smallest d_hat power of two to include.",
    )
    parser.add_argument(
        "--seeds",
        default="1-30",
        help=(
            "Seeds to run. Examples: '1-30', '2-30', or '1,4,7'. "
            "Use '2-30' when seed 1 is already complete."
        ),
    )
    parser.add_argument(
        "--ego-order",
        default="3980,698,414,686",
        help=(
            "Comma-separated Facebook ego IDs in processing order. "
            "Default: 3980,698,414,686."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of template rows to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output instead of resuming it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed seed.",
    )
    return parser.parse_args()


def parse_norm(raw: str) -> float:
    if raw.lower() in {"inf", "infinity", "math.inf", "np.inf"}:
        return math.inf
    return float(raw)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)

            if end < start:
                raise ValueError(f"Invalid seed range: {part}")

            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))

    seeds = sorted(set(seeds))

    if not seeds:
        raise ValueError("At least one seed is required.")

    return seeds


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


def max_positive_degree(matrix: np.ndarray) -> int:
    return int((matrix == 1).sum(axis=1).max())


def powers_of_two_up_to(max_value: int, min_value: int = 1) -> list[int]:
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


def reconstruct_facebook_matrix(ego_id: str) -> np.ndarray:
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


def normalize_existing_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {
        column: row.get(column, "")
        for column in OUTPUT_COLUMNS
    }

    if not str(normalized.get("seed", "")).strip():
        normalized["seed"] = "1"

    return normalized


def load_existing_output(
    path: Path,
    overwrite: bool,
) -> list[dict[str, str]]:
    if overwrite or not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [normalize_existing_row(row) for row in reader]


def output_key(
    ego_id: str,
    p_delete: str,
    seed: int,
    d_hat: int,
    lambda_param: int,
) -> tuple[str, str, str, str, str]:
    return (
        str(ego_id),
        f"{float(p_delete):.8f}",
        str(seed),
        str(d_hat),
        str(lambda_param),
    )


def seed_grid_is_complete(
    completed_keys: set[tuple[str, str, str, str, str]],
    ego_id: str,
    p_delete: float,
    seed: int,
    d_hat_values: list[int],
    lambda_values: list[int],
) -> bool:
    """
    Return True only when every expected (d_hat, lambda) row for this
    ego_id, p_delete, and seed is already present.

    This check happens before edge deletion and before solving MinMaxLP,
    so completed seeds are skipped without repeating expensive work.
    """
    return all(
        output_key(
            ego_id=ego_id,
            p_delete=str(p_delete),
            seed=seed,
            d_hat=d_hat,
            lambda_param=lambda_param,
        )
        in completed_keys
        for d_hat in d_hat_values
        for lambda_param in lambda_values
    )


def existing_output_keys(
    rows: list[dict[str, str]],
) -> set[tuple[str, str, str, str, str]]:
    keys = set()

    for row in rows:
        try:
            keys.add(
                output_key(
                    ego_id=row.get("ego_id", ""),
                    p_delete=row.get("p_delete", "0"),
                    seed=int(float(row.get("seed", "1") or "1")),
                    d_hat=int(
                        float(
                            row.get(
                                "complete_min_max_cc_d_hat",
                                "0",
                            )
                        )
                    ),
                    lambda_param=int(
                        float(
                            row.get(
                                "complete_min_max_cc_lambda",
                                "0",
                            )
                        )
                    ),
                )
            )
        except (TypeError, ValueError):
            continue

    return keys


def template_value(row: dict[str, str], column: str) -> str:
    return str(row.get(column, "")).strip()


def existing_lp_result(
    rows: list[dict[str, str]],
    ego_id: str,
    prefix: str,
) -> dict[str, Any] | None:
    """
    Reconstruct a MinMaxLP result dictionary from an existing output row.
    This prevents recomputing the complete-graph LP after a restart.
    """
    for row in rows:
        if str(row.get("ego_id", "")).strip() != str(ego_id):
            continue

        lp_cost = str(row.get(f"{prefix}_min_max_lp_cost", "")).strip()
        if not lp_cost:
            continue

        return {
            "lp_cost": float(lp_cost),
            "rounding_cost": float(
                row.get(f"{prefix}_min_max_lp_rounding_cost", 0) or 0
            ),
            "max_disagreement_vertex": float(
                row.get(
                    f"{prefix}_min_max_lp_max_disagreement_vertex",
                    0,
                )
                or 0
            ),
            "cluster_count": int(
                float(
                    row.get(f"{prefix}_min_max_lp_cluster_count", 0)
                    or 0
                )
            ),
            "r": float(row.get(f"{prefix}_min_max_lp_r", 0.4) or 0.4),
            "r2": float(row.get(f"{prefix}_min_max_lp_r2", 0.4) or 0.4),
            "method": int(
                float(row.get(f"{prefix}_min_max_lp_method", 0) or 0)
            ),
            "norm": row.get(f"{prefix}_min_max_lp_norm", "inf") or "inf",
            "lp_runtime_seconds": float(
                row.get(
                    f"{prefix}_min_max_lp_runtime_seconds",
                    0,
                )
                or 0
            ),
            "rounding_runtime_seconds": float(
                row.get(
                    f"{prefix}_min_max_lp_rounding_runtime_seconds",
                    0,
                )
                or 0
            ),
            "total_runtime_seconds": float(
                row.get(
                    f"{prefix}_min_max_lp_total_runtime_seconds",
                    0,
                )
                or 0
            ),
        }

    return None


def existing_complete_cc_results(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """
    Load already-computed complete-graph MinMaxCC results from the output CSV.
    The complete graph does not depend on p_delete or deletion seed.
    """
    cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    for row in rows:
        ego_id = str(row.get("ego_id", "")).strip()
        d_hat_raw = str(
            row.get("complete_min_max_cc_d_hat", "")
        ).strip()
        lambda_raw = str(
            row.get("complete_min_max_cc_lambda", "")
        ).strip()
        disagreement_raw = str(
            row.get("complete_min_max_cc_max_disagreement", "")
        ).strip()

        if not ego_id or not d_hat_raw or not lambda_raw or not disagreement_raw:
            continue

        key = (
            ego_id,
            int(float(d_hat_raw)),
            int(float(lambda_raw)),
        )

        if key in cache:
            continue

        cache[key] = {
            "cluster_count": int(
                float(
                    row.get(
                        "complete_min_max_cc_cluster_count",
                        0,
                    )
                    or 0
                )
            ),
            "max_disagreement": float(disagreement_raw),
            "runtime_seconds": float(
                row.get(
                    "complete_min_max_cc_runtime_seconds",
                    0,
                )
                or 0
            ),
        }

    return cache


def main() -> None:
    args = parse_args()
    norm = parse_norm(args.norm)
    lambda_values = parse_int_list(args.lambda_values)
    seeds = parse_seed_spec(args.seeds)
    ego_order = [
        ego_id.strip()
        for ego_id in args.ego_order.split(",")
        if ego_id.strip()
    ]

    input_csv = args.input
    output_csv = args.output

    if not input_csv.is_absolute():
        input_csv = REPO_ROOT / input_csv

    if not output_csv.is_absolute():
        output_csv = REPO_ROOT / output_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")

        required_input_columns = [
            column
            for column in BASE_COLUMNS
            if column != "seed"
        ]
        missing = [
            column
            for column in required_input_columns
            if column not in reader.fieldnames
        ]

        if missing:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(missing)
            )

        all_rows = list(reader)

        template_rows = [
            row
            for row in all_rows
            if row.get("graph_family", "").strip().lower() == "facebook"
            and str(row.get("ego_id", "")).strip() in ego_order
        ]

        order_index = {
            ego_id: index
            for index, ego_id in enumerate(ego_order)
        }

        template_rows.sort(
            key=lambda row: (
                order_index[str(row.get("ego_id", "")).strip()],
                float(row.get("p_delete", 0) or 0),
            )
        )

        if not template_rows:
            raise ValueError(
                "No requested Facebook rows found in all_runs_flat.csv."
            )

    output_rows = load_existing_output(
        output_csv,
        overwrite=args.overwrite,
    )
    completed_keys = existing_output_keys(output_rows)

    matrix_cache: dict[str, np.ndarray] = {}
    complete_lp_cache: dict[str, dict[str, Any]] = {}
    complete_cc_cache = existing_complete_cc_results(output_rows)

    for ego_id in ego_order:
        cached_lp = existing_lp_result(
            output_rows,
            ego_id=ego_id,
            prefix="complete",
        )
        if cached_lp is not None:
            complete_lp_cache[ego_id] = cached_lp

    processed_template_rows = 0
    new_grid_rows = 0
    failures = 0

    print("Input:", input_csv)
    print("Output:", output_csv)
    print("Seeds:", seeds)
    print("Ego order:", ego_order)
    print("Existing grid rows:", len(output_rows))
    print(
        "Complete LPs loaded from existing output:",
        sorted(complete_lp_cache.keys()),
    )

    for row_index, template_row in enumerate(template_rows):
        if (
            args.limit is not None
            and processed_template_rows >= args.limit
        ):
            break

        ego_id = template_value(template_row, "ego_id")
        p_delete_raw = template_value(template_row, "p_delete")

        if not ego_id:
            raise ValueError(
                f"Missing ego_id in template row {row_index + 1}"
            )

        if not p_delete_raw:
            raise ValueError(
                f"Missing p_delete in template row {row_index + 1}"
            )

        p_delete = float(p_delete_raw)

        print("\n" + "=" * 80)
        print(
            f"Template row {row_index + 1}/{len(template_rows)} | "
            f"ego_id={ego_id} | p_delete={p_delete}"
        )
        print("=" * 80)

        if ego_id not in matrix_cache:
            print("Reconstructing complete graph once...")
            matrix_cache[ego_id] = reconstruct_facebook_matrix(ego_id)

        complete_matrix = matrix_cache[ego_id]

        if ego_id not in complete_lp_cache:
            print(
                "No existing complete MinMaxLP found; computing it once..."
            )
            complete_lp_cache[ego_id] = compute_min_max_lp_data(
                complete_matrix,
                compute_min_max_lp=True,
                r=args.r,
                r2=args.r2,
                method=args.method,
                norm=norm,
            )
        else:
            print("Reusing existing complete MinMaxLP from output CSV.")

        complete_lp = complete_lp_cache[ego_id]

        # Edge deletion cannot increase the positive degree, so the d_hat grid
        # is determined by the complete graph and can be known before deletion.
        d_hat_values = powers_of_two_up_to(
            max_positive_degree(complete_matrix),
            min_value=max(1, args.min_d_hat),
        )

        for seed in seeds:
            print("\n" + "-" * 80)
            print(
                f"ego_id={ego_id} | p_delete={p_delete} | seed={seed}"
            )
            print("-" * 80)

            if seed_grid_is_complete(
                completed_keys=completed_keys,
                ego_id=ego_id,
                p_delete=p_delete,
                seed=seed,
                d_hat_values=d_hat_values,
                lambda_values=lambda_values,
            ):
                print(
                    "Skipping complete seed before deletion/LP: "
                    f"ego_id={ego_id}, p_delete={p_delete}, seed={seed}"
                )
                continue

            try:
                edge_matrix, deleted_count = delete_edges(
                    complete_matrix,
                    p_delete,
                    seed,
                )

                print("Deleted edges:", deleted_count)

                print(
                    "Running edge MinMaxLP once for this deletion...",
                    flush=True,
                )
                edge_lp = compute_min_max_lp_data(
                    edge_matrix,
                    compute_min_max_lp=True,
                    r=args.r,
                    r2=args.r2,
                    method=args.method,
                    norm=norm,
                )

                print("d_hat values:", d_hat_values)
                print("lambda values:", lambda_values)

                for d_hat in d_hat_values:
                    for lambda_param in lambda_values:
                        key = output_key(
                            ego_id=ego_id,
                            p_delete=str(p_delete),
                            seed=seed,
                            d_hat=d_hat,
                            lambda_param=lambda_param,
                        )

                        if key in completed_keys:
                            print(
                                "Skipping existing row: "
                                f"seed={seed}, d_hat={d_hat}, "
                                f"lambda={lambda_param}"
                            )
                            continue

                        complete_cache_key = (
                            ego_id,
                            d_hat,
                            lambda_param,
                        )

                        if complete_cache_key not in complete_cc_cache:
                            print(
                                "Running complete MinMaxCC once: "
                                f"d_hat={d_hat}, "
                                f"lambda={lambda_param}"
                            )
                            complete_cc_cache[
                                complete_cache_key
                            ] = compute_min_max_cc_data(
                                complete_matrix,
                                compute_min_max=True,
                                param_1=d_hat,
                                param_2=lambda_param,
                            )

                        complete_cc = complete_cc_cache[
                            complete_cache_key
                        ]

                        print(
                            "Running edge MinMaxCC: "
                            f"seed={seed}, d_hat={d_hat}, "
                            f"lambda={lambda_param}"
                        )
                        edge_cc = compute_min_max_cc_data(
                            edge_matrix,
                            compute_min_max=True,
                            param_1=d_hat,
                            param_2=lambda_param,
                        )

                        new_row = {
                            column: template_row.get(column, "")
                            for column in BASE_COLUMNS
                            if column != "seed"
                        }
                        new_row["seed"] = str(seed)

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

                        normalized_new_row = {
                            column: new_row.get(column, "")
                            for column in OUTPUT_COLUMNS
                        }

                        output_rows.append(normalized_new_row)
                        completed_keys.add(key)
                        new_grid_rows += 1

                        atomic_write_csv(
                            output_csv,
                            OUTPUT_COLUMNS,
                            output_rows,
                        )

                        print(
                            "Saved checkpoint. "
                            f"New grid rows: {new_grid_rows}",
                            flush=True,
                        )

            except Exception as error:
                failures += 1
                print(
                    f"ERROR for ego_id={ego_id}, "
                    f"p_delete={p_delete}, seed={seed}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

                if not args.continue_on_error:
                    raise

        processed_template_rows += 1

    print("\nFinished.")
    print("Processed template rows:", processed_template_rows)
    print("New grid rows written:", new_grid_rows)
    print("Failures:", failures)
    print("Output:", output_csv)


if __name__ == "__main__":
    main()