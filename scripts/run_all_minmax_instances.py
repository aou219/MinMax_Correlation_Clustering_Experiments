#!/usr/bin/env python3
"""Run min-max algorithms for every row in minmax_runs_flat.csv.
The script reconstructs each original complete signed graph, reproduces the
edge-deleted graph using the same ``p_delete`` and seed, runs both min-max
methods, and fills the corresponding CSV row.

Results are checkpointed after each completed row, so an interrupted run can
be resumed. By default, rows that already contain both complete and edge
min-max LP results are skipped.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
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
from src.graph_generation import (  # noqa: E402
    generate_clique_signed_graph,
    generate_signed_complete_graph,
)


DEFAULT_CSV = REPO_ROOT / "results/processed/minmax_runs_flat.csv"

MINMAX_SUFFIXES = [
    "min_max_cc_computed",
    "min_max_cc_clustering",
    "min_max_cc_cluster_count",
    "min_max_cc_max_disagreement",
    "min_max_cc_runtime_seconds",
    "min_max_lp_computed",
    "min_max_lp_cost",
    "min_max_lp_rounding_cost",
    "min_max_lp_max_disagreement_vertex",
    "min_max_lp_disagreement_vector",
    "min_max_lp_clustering",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run min-max algorithms for all rows in minmax_runs_flat.csv."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--r2", type=float, default=0.4)
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument(
        "--norm",
        default="inf",
        help="Use 'inf' for max disagreement or a numeric p-norm.",
    )
    parser.add_argument(
        "--only-family",
        choices=("random", "clique", "facebook"),
        help="Run only one graph family.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of unfinished rows to process in this invocation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute rows even when results are already present.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log a failing row and continue instead of stopping immediately.",
    )
    return parser.parse_args()


def parse_norm(raw: str) -> float:
    if raw.lower() in {"inf", "infinity", "math.inf", "np.inf"}:
        return math.inf
    return float(raw)


def parse_optional_int(value: str, name: str) -> int:
    if value == "":
        raise ValueError(f"Missing required integer field: {name}")
    return int(float(value))


def parse_optional_float(value: str, name: str) -> float:
    if value == "":
        raise ValueError(f"Missing required float field: {name}")
    return float(value)


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list, got: {value!r}")
    return parsed


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def scalar_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return str(value)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "experiments" in payload:
        payload = payload["experiments"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported JSON structure in {path}")
    return [record for record in payload if isinstance(record, dict)]


def values_match(left: Any, right: Any) -> bool:
    if left in (None, "") and right in (None, ""):
        return True
    try:
        return abs(float(left) - float(right)) < 1e-12
    except (TypeError, ValueError):
        return str(left) == str(right)


def find_source_record(row: dict[str, str]) -> dict[str, Any]:
    source_path = Path(row["file_path"])
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source result JSON not found: {source_path}")

    records = load_json_records(source_path)
    wanted_seed = row.get("seed", "")
    wanted_delete = row.get("p_delete", "")

    for record in records:
        params = record.get("graph_params", {})
        if values_match(params.get("seed"), wanted_seed) and values_match(
            params.get("p_delete"), wanted_delete
        ):
            return record

    raise LookupError(
        f"No JSON record matching seed={wanted_seed} and "
        f"p_delete={wanted_delete} in {source_path}"
    )


def get_all_nodes_from_edges_and_circles(
    edge_nodes: set[Any], circles: list[dict[str, Any]]
) -> list[Any]:
    circle_nodes: set[Any] = set()
    for circle in circles:
        circle_nodes.update(circle["nodes"])
    return sorted(edge_nodes | circle_nodes)


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


def reconstruct_complete_matrix(row: dict[str, str], record: dict[str, Any]):
    params = record.get("graph_params", {})
    family = row["graph_family"].strip().lower()
    seed = parse_optional_int(row["seed"], "seed")

    if family == "random":
        n = parse_optional_int(row["n"], "n")
        p_positive = parse_optional_float(row["p_positive"], "p_positive")
        return generate_signed_complete_graph(
            n=n,
            p_positive=p_positive,
            seed=seed,
        )

    if family == "clique":
        cluster_sizes = parse_list(
            params.get("cluster_sizes") or row.get("cluster_sizes", "")
        )
        p_pos_inside = params.get("p_pos_inside", 0.9)
        p_pos_between = params.get("p_pos_between", 0.1)
        matrix, _ = generate_clique_signed_graph(
            cluster_sizes=[int(value) for value in cluster_sizes],
            p_pos_inside=float(p_pos_inside),
            p_pos_between=float(p_pos_between),
            seed=seed,
        )
        return matrix

    if family == "facebook":
        ego_id = str(params.get("ego_id") or row.get("ego_id", "")).strip()
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

    raise ValueError(f"Unsupported graph_family: {family!r}")


def flatten_results(
    prefix: str,
    cc_result: dict[str, Any],
    lp_result: dict[str, Any],
) -> dict[str, str]:
    return {
        f"{prefix}_min_max_cc_computed": scalar_cell(
            cc_result["max_disagreement"] is not None
        ),
        f"{prefix}_min_max_cc_clustering": json_cell(cc_result["clustering"]),
        f"{prefix}_min_max_cc_cluster_count": scalar_cell(cc_result["cluster_count"]),
        f"{prefix}_min_max_cc_max_disagreement": scalar_cell(
            cc_result["max_disagreement"]
        ),
        f"{prefix}_min_max_cc_runtime_seconds": scalar_cell(
            cc_result["runtime_seconds"]
        ),
        f"{prefix}_min_max_lp_computed": scalar_cell(lp_result["lp_cost"] is not None),
        f"{prefix}_min_max_lp_cost": scalar_cell(lp_result["lp_cost"]),
        f"{prefix}_min_max_lp_rounding_cost": scalar_cell(lp_result["rounding_cost"]),
        f"{prefix}_min_max_lp_max_disagreement_vertex": scalar_cell(
            lp_result["max_disagreement_vertex"]
        ),
        f"{prefix}_min_max_lp_disagreement_vector": json_cell(
            lp_result["disagreement_vector"]
        ),
        f"{prefix}_min_max_lp_clustering": json_cell(lp_result["clustering"]),
        f"{prefix}_min_max_lp_cluster_count": scalar_cell(lp_result["cluster_count"]),
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


def row_is_complete(row: dict[str, str]) -> bool:
    return bool(row.get("complete_min_max_lp_cost")) and bool(
        row.get("edge_min_max_lp_cost")
    )


def main() -> None:
    args = parse_args()
    norm = parse_norm(args.norm)

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Min-max CSV not found: {csv_path}\n"
            "Create it first with scripts/make_minmax_flat_template.py."
        )

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        missing = [column for column in MINMAX_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                "The CSV is missing min-max columns. Recreate it with the template "
                "script. Missing: " + ", ".join(missing)
            )
        rows = list(reader)

    processed = 0
    failures = 0

    for index, row in enumerate(rows):
        family = row.get("graph_family", "").strip().lower()
        if args.only_family and family != args.only_family:
            continue
        if not args.force and row_is_complete(row):
            continue
        if args.limit is not None and processed >= args.limit:
            break

        label = (
            f"row {index + 1}/{len(rows)} | {family} | "
            f"file={row.get('file_name')} | seed={row.get('seed')} | "
            f"p_delete={row.get('p_delete')}"
        )
        print("\n" + "=" * 80)
        print(label)
        print("=" * 80, flush=True)

        try:
            source_record = find_source_record(row)
            complete_matrix = reconstruct_complete_matrix(row, source_record)
            p_delete = parse_optional_float(row["p_delete"], "p_delete")
            seed = parse_optional_int(row["seed"], "seed")
            edge_matrix, deleted_count = delete_edges(
                complete_matrix,
                p_delete,
                seed,
            )

            expected_deleted = row.get("edge_num_edges_deleted", "")
            if expected_deleted and int(float(expected_deleted)) != int(deleted_count):
                raise ValueError(
                    "Reconstructed edge deletion does not match the old run: "
                    f"expected {expected_deleted}, got {deleted_count}."
                )

            complete_cc = compute_min_max_cc_data(
                complete_matrix,
                compute_min_max=True,
                param_1=8,
                param_2=5,
            )
            complete_lp = compute_min_max_lp_data(
                complete_matrix,
                compute_min_max_lp=True,
                r=args.r,
                r2=args.r2,
                method=args.method,
                norm=norm,
            )
            edge_cc = compute_min_max_cc_data(
                edge_matrix,
                compute_min_max=True,
                param_1=8,
                param_2=5,
            )
            edge_lp = compute_min_max_lp_data(
                edge_matrix,
                compute_min_max_lp=True,
                r=args.r,
                r2=args.r2,
                method=args.method,
                norm=norm,
            )

            row.update(flatten_results("complete", complete_cc, complete_lp))
            row.update(flatten_results("edge", edge_cc, edge_lp))
            processed += 1

            atomic_write_csv(csv_path, fieldnames, rows)
            print(f"Saved checkpoint after {processed} completed row(s).", flush=True)

        except Exception as error:
            failures += 1
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise

    print("\nFinished.")
    print(f"Completed rows this run: {processed}")
    print(f"Failed rows this run: {failures}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
