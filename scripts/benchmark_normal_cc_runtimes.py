#!/usr/bin/env python3
"""
Benchmark ordinary correlation-clustering runtimes for the Facebook and clique
paper tables without changing any objective values in the main result tables.

Outputs
-------
results/research_tables/normal_cc_runtime_benchmarks_raw.csv
    One row per measured run.

results/research_tables/normal_cc_runtime_benchmarks.csv
    Paper-table-level averages:
      - Facebook: per ego_id, graph variant, and p_delete.
      - Clique: per n, graph variant, and p_delete.

results/research_tables/runtime_machine_specifications.json
    Hardware, operating-system, Python, NumPy, Gurobi, and Git information.

Runtime definitions
-------------------
Pivot:
    Wall-clock time of one call to the deterministic Pivot algorithm.
    Clustering-cost evaluation and graph construction are outside the timer.

Ordinary all-pairs LP:
    Gurobi's solver runtime (model.optimize runtime) is used for the paper
    summary when available. End-to-end wall-clock time of solve_all_pairs is
    also stored in the raw file.

The script is resumable. Without --restart, existing raw benchmark rows are
preserved and completed tasks are skipped.

Default scope
-------------
Facebook:
    - Pivot: all complete ego graphs, plus edge-deleted seed 1 for the four
      LP-eligible ego graphs.
    - LP: complete and edge-deleted seed 1 for ego IDs 414, 686, 698, 3980.
    - Pivot repetitions: seeds 1-30.
    - LP repetitions: 1. Increase to 3 when time permits.

Clique:
    - Graph seed 1 for every distinct clique configuration in all_runs_flat.
    - Pivot: complete and all four edge-deletion probabilities.
    - LP: complete graphs are benchmarked; existing edge LP solver runtimes
      are reused from all_runs_flat by default.

Run examples
------------
Fast Pivot-only benchmark:

    python scripts/benchmark_normal_cc_runtimes.py \
        --algorithms pivot \
        --pivot-runs 30 \
        --restart

Facebook LP benchmark only:

    python scripts/benchmark_normal_cc_runtimes.py \
        --dataset facebook \
        --algorithms lp \
        --lp-repetitions 1 \
        --restart

Full benchmark:

    caffeinate -i python scripts/benchmark_normal_cc_runtimes.py \
        --dataset all \
        --algorithms pivot,lp \
        --pivot-runs 30 \
        --lp-repetitions 1 \
        --restart
"""

from __future__ import annotations

# Keep numerical-library behavior stable before NumPy is imported.
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

if any(os.environ.get(k) != v for k, v in DETERMINISTIC_ENV.items()):
    env = os.environ.copy()
    env.update(DETERMINISTIC_ENV)
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "2026-07-22-normal-cc-runtime-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from src.all_pairs_solver import solve_all_pairs
from src.cost import calculate_clustering_cost
from src.edge_deletion import delete_edges
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)
from src.graph_generation import generate_clique_signed_graph


DEFAULT_FACEBOOK_TABLE = (
    REPO_ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
)
DEFAULT_CLIQUE_TABLE = (
    REPO_ROOT / "results/research_tables/archive/all_runs_flat.csv"
)
DEFAULT_RAW_OUTPUT = (
    REPO_ROOT / "results/research_tables/normal_cc_runtime_benchmarks_raw.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    REPO_ROOT / "results/research_tables/normal_cc_runtime_benchmarks.csv"
)
DEFAULT_MACHINE_OUTPUT = (
    REPO_ROOT / "results/research_tables/runtime_machine_specifications.json"
)

RAW_FIELDS = [
    "task_key",
    "dataset",
    "instance_id",
    "ego_id",
    "n",
    "graph_variant",
    "p_delete",
    "graph_seed",
    "deletion_seed",
    "cluster_sizes",
    "algorithm",
    "repetition",
    "pivot_seed",
    "cost",
    "wall_runtime_seconds",
    "solver_runtime_seconds",
    "runtime_source",
    "matrix_sha256",
    "created_utc",
]

SUMMARY_FIELDS = [
    "dataset",
    "ego_id",
    "n",
    "graph_variant",
    "p_delete",
    "algorithm",
    "number_of_runtime_observations",
    "average_runtime_seconds",
    "standard_deviation_seconds",
    "minimum_runtime_seconds",
    "maximum_runtime_seconds",
    "runtime_definition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["facebook", "clique", "all"],
        default="all",
    )
    parser.add_argument(
        "--algorithms",
        default="pivot,lp",
        help="Comma-separated selection: pivot, lp, or pivot,lp.",
    )
    parser.add_argument(
        "--pivot-runs",
        type=int,
        default=30,
        help="Number of deterministic Pivot seeds per graph.",
    )
    parser.add_argument(
        "--lp-repetitions",
        type=int,
        default=1,
        help="Number of LP solves per newly benchmarked graph.",
    )

    parser.add_argument(
        "--facebook-table",
        type=Path,
        default=DEFAULT_FACEBOOK_TABLE,
    )
    parser.add_argument(
        "--facebook-egos",
        default="all",
        help="Complete-graph Pivot ego IDs: all or comma-separated.",
    )
    parser.add_argument(
        "--facebook-lp-egos",
        default="414,686,698,3980",
    )
    parser.add_argument(
        "--facebook-edge-seed",
        type=int,
        default=1,
        help="Deletion seed used by the Facebook paper's edge rows.",
    )

    parser.add_argument(
        "--clique-table",
        type=Path,
        default=DEFAULT_CLIQUE_TABLE,
    )
    parser.add_argument(
        "--clique-graph-seeds",
        default="1",
        help="Clique graph/deletion seeds to benchmark, e.g. 1 or 1-3.",
    )
    parser.add_argument(
        "--clique-max-n",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--reuse-existing-clique-edge-lp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse edge_all_pairs_lp_runtime_seconds from all_runs_flat "
            "instead of rerunning clique edge LPs."
        ),
    )

    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--machine-output",
        type=Path,
        default=DEFAULT_MACHINE_OUTPUT,
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new measured/imported raw rows.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_int_spec(value: str) -> list[int]:
    out: set[int] = set()
    for part in split_csv(value):
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"Descending range is invalid: {part}")
            out.update(range(start, stop + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("At least one integer is required.")
    return sorted(out)


def parse_algorithms(value: str) -> set[str]:
    algorithms = {x.lower() for x in split_csv(value)}
    invalid = algorithms.difference({"pivot", "lp"})
    if invalid or not algorithms:
        raise ValueError(
            "Algorithms must be pivot, lp, or pivot,lp. Invalid: "
            + ", ".join(sorted(invalid))
        )
    return algorithms


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def probability_key(value: Any) -> str:
    return f"{float(value):.12g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
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
                fieldnames=list(fields),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_matrix(matrix: np.ndarray) -> str:
    array = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"|")
    digest.update(repr(tuple(array.shape)).encode())
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_run_pivot(
    signed_matrix: np.ndarray,
    seed: int,
) -> tuple[list[set[int]], list[int]]:
    rng = np.random.default_rng(seed)
    active = set(range(signed_matrix.shape[0]))
    clusters: list[set[int]] = []
    pivots: list[int] = []

    while active:
        ordered = sorted(active)
        pivot = int(rng.choice(np.asarray(ordered, dtype=int)))
        cluster = {pivot}
        for vertex in ordered:
            if vertex != pivot and signed_matrix[pivot, vertex] == 1:
                cluster.add(vertex)
        clusters.append(cluster)
        pivots.append(pivot)
        active.difference_update(cluster)

    return clusters, pivots


def locate_facebook_edges_file(ego_id: str) -> Path:
    candidates = [
        REPO_ROOT / f"data/facebook/{ego_id}.edges",
        REPO_ROOT / f"data/facebook/facebook_3/{ego_id}.edges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Facebook .edges file found for ego {ego_id}."
    )


def build_facebook_matrix(ego_id: str) -> np.ndarray:
    edge_file = locate_facebook_edges_file(ego_id)
    edge_nodes, facebook_edges = load_facebook_ego_edges(str(edge_file))
    ordered_nodes = sorted(edge_nodes)
    matrix, _, _, _ = build_complete_signed_matrix_from_facebook_sample(
        ordered_nodes,
        facebook_edges,
    )
    return matrix


def parse_cluster_sizes(value: Any, file_name: str = "") -> list[int]:
    text = str(value or "").strip()
    if text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                sizes = [int(x) for x in parsed]
                if sizes:
                    return sizes
        except (ValueError, SyntaxError, TypeError):
            pass

        match = re.fullmatch(r"\[?\s*(\d+)\s*x\s*(\d+)\s*\]?", text)
        if match:
            return [int(match.group(2))] * int(match.group(1))

        numbers = [int(x) for x in re.findall(r"\d+", text)]
        if numbers:
            return numbers

    stem = Path(str(file_name or "")).stem
    match = re.match(r"clq_n\d+_(.+)", stem)
    if not match:
        return []
    suffix = match.group(1)
    repeated = re.fullmatch(r"(\d+)x(\d+)", suffix)
    if repeated:
        return [int(repeated.group(2))] * int(repeated.group(1))
    return [int(x) for x in re.findall(r"\d+", suffix)]


def clique_graph_identity(row: Mapping[str, Any]) -> str:
    return (
        str(row.get("file_path", "")).strip()
        or str(row.get("file_name", "")).strip()
        or (
            f"n={row.get('n', '')}|"
            f"clusters={row.get('cluster_sizes', '')}"
        )
    )


def dispose_gurobi_environment() -> None:
    try:
        import gurobipy as gp
        gp.disposeDefaultEnv()
    except Exception:
        pass


def task_key(*parts: Any) -> str:
    return "|".join(str(part) for part in parts)


def benchmark_pivot_rows(
    *,
    dataset: str,
    instance_id: str,
    ego_id: str,
    n: int,
    variant: str,
    p_delete: float,
    graph_seed: int | str,
    deletion_seed: int | str,
    cluster_sizes: str,
    matrix: np.ndarray,
    pivot_seeds: Sequence[int],
    completed: set[str],
) -> Iterable[dict[str, Any]]:
    matrix_hash = sha256_matrix(matrix)

    for pivot_seed in pivot_seeds:
        key = task_key(
            dataset,
            instance_id,
            variant,
            probability_key(p_delete),
            "pivot",
            pivot_seed,
        )
        if key in completed:
            continue

        start = time.perf_counter()
        clusters, _ = deterministic_run_pivot(matrix, pivot_seed)
        elapsed = time.perf_counter() - start

        # Sanity-check cost, but keep it outside the runtime measurement.
        cost = calculate_clustering_cost(matrix, clusters)

        yield {
            "task_key": key,
            "dataset": dataset,
            "instance_id": instance_id,
            "ego_id": ego_id,
            "n": n,
            "graph_variant": variant,
            "p_delete": probability_key(p_delete),
            "graph_seed": graph_seed,
            "deletion_seed": deletion_seed,
            "cluster_sizes": cluster_sizes,
            "algorithm": "pivot",
            "repetition": "",
            "pivot_seed": pivot_seed,
            "cost": cost,
            "wall_runtime_seconds": round(elapsed, 9),
            "solver_runtime_seconds": "",
            "runtime_source": "measured_pure_pivot_wall_clock",
            "matrix_sha256": matrix_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }


def benchmark_lp_rows(
    *,
    dataset: str,
    instance_id: str,
    ego_id: str,
    n: int,
    variant: str,
    p_delete: float,
    graph_seed: int | str,
    deletion_seed: int | str,
    cluster_sizes: str,
    matrix: np.ndarray,
    repetitions: int,
    completed: set[str],
) -> Iterable[dict[str, Any]]:
    matrix_hash = sha256_matrix(matrix)

    for repetition in range(1, repetitions + 1):
        key = task_key(
            dataset,
            instance_id,
            variant,
            probability_key(p_delete),
            "lp",
            repetition,
        )
        if key in completed:
            continue

        start = time.perf_counter()
        cost, _, solve_info = solve_all_pairs(
            matrix,
            verbose=False,
            relax=True,
            return_x_values=False,
        )
        elapsed = time.perf_counter() - start

        yield {
            "task_key": key,
            "dataset": dataset,
            "instance_id": instance_id,
            "ego_id": ego_id,
            "n": n,
            "graph_variant": variant,
            "p_delete": probability_key(p_delete),
            "graph_seed": graph_seed,
            "deletion_seed": deletion_seed,
            "cluster_sizes": cluster_sizes,
            "algorithm": "lp",
            "repetition": repetition,
            "pivot_seed": "",
            "cost": cost,
            "wall_runtime_seconds": round(elapsed, 9),
            "solver_runtime_seconds": round(
                float(solve_info["runtime_seconds"]),
                9,
            ),
            "runtime_source": "measured_all_pairs_lp",
            "matrix_sha256": matrix_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        dispose_gurobi_environment()


def imported_clique_edge_lp_row(
    row: Mapping[str, Any],
    completed: set[str],
) -> dict[str, Any] | None:
    runtime = to_float(row.get("edge_all_pairs_lp_runtime_seconds"))
    if runtime is None:
        return None

    graph_id = clique_graph_identity(row)
    n = int(float(row["n"]))
    seed = int(float(row.get("seed", "1")))
    p_delete = float(row["p_delete"])
    key = task_key(
        "clique",
        graph_id,
        seed,
        "edge_deleted",
        probability_key(p_delete),
        "lp",
        "existing",
    )
    if key in completed:
        return None

    sizes = parse_cluster_sizes(
        row.get("cluster_sizes"),
        row.get("file_name", ""),
    )

    return {
        "task_key": key,
        "dataset": "clique",
        "instance_id": graph_id,
        "ego_id": "",
        "n": n,
        "graph_variant": "edge_deleted",
        "p_delete": probability_key(p_delete),
        "graph_seed": seed,
        "deletion_seed": seed,
        "cluster_sizes": json.dumps(sizes),
        "algorithm": "lp",
        "repetition": "existing",
        "pivot_seed": "",
        "cost": row.get("edge_all_pairs_lp_cost", ""),
        "wall_runtime_seconds": "",
        "solver_runtime_seconds": round(runtime, 9),
        "runtime_source": "existing_all_runs_flat_solver_runtime",
        "matrix_sha256": "",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def summarize(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)

    for row in raw_rows:
        dataset = str(row["dataset"])
        algorithm = str(row["algorithm"])
        variant = str(row["graph_variant"])
        p_delete = probability_key(row["p_delete"])
        n = int(float(row["n"]))

        if algorithm == "pivot":
            runtime = to_float(row.get("wall_runtime_seconds"))
            definition = "pure Pivot wall-clock time per run"
        else:
            runtime = to_float(row.get("solver_runtime_seconds"))
            if runtime is None:
                runtime = to_float(row.get("wall_runtime_seconds"))
                definition = "all-pairs LP end-to-end wall-clock time"
            else:
                definition = "Gurobi all-pairs LP solver runtime"

        if runtime is None:
            continue

        if dataset == "facebook":
            ego_id = str(row.get("ego_id", "")).strip()
            group_key = (
                dataset,
                ego_id,
                n,
                variant,
                p_delete,
                algorithm,
                definition,
            )
        else:
            group_key = (
                dataset,
                "",
                n,
                variant,
                p_delete,
                algorithm,
                definition,
            )

        groups[group_key].append(runtime)

    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        dataset, ego_id, n, variant, p_delete, algorithm, definition = key
        output.append(
            {
                "dataset": dataset,
                "ego_id": ego_id,
                "n": n,
                "graph_variant": variant,
                "p_delete": p_delete,
                "algorithm": algorithm,
                "number_of_runtime_observations": len(values),
                "average_runtime_seconds": round(statistics.fmean(values), 9),
                "standard_deviation_seconds": (
                    round(statistics.stdev(values), 9)
                    if len(values) >= 2
                    else 0.0
                ),
                "minimum_runtime_seconds": round(min(values), 9),
                "maximum_runtime_seconds": round(max(values), 9),
                "runtime_definition": definition,
            }
        )

    output.sort(
        key=lambda row: (
            row["dataset"],
            int(row["n"]),
            int(row["ego_id"]) if row["ego_id"] else -1,
            0 if row["graph_variant"] == "complete" else 1,
            float(row["p_delete"]),
            row["algorithm"],
        )
    )
    return output


def safe_command(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            list(command),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def machine_specifications() -> dict[str, Any]:
    info: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine_architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "script_version": SCRIPT_VERSION,
        "git_commit": safe_command(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
        ),
        "git_branch": safe_command(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"]
        ),
    }

    if platform.system() == "Darwin":
        info["cpu_brand"] = safe_command(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        )
        memory_bytes = safe_command(["sysctl", "-n", "hw.memsize"])
        if memory_bytes:
            try:
                info["memory_bytes"] = int(memory_bytes)
                info["memory_gib"] = round(
                    int(memory_bytes) / (1024 ** 3),
                    3,
                )
            except ValueError:
                info["memory_bytes"] = memory_bytes
        info["mac_hardware_overview"] = safe_command(
            ["system_profiler", "SPHardwareDataType"]
        )
        info["mac_software_overview"] = safe_command(
            ["system_profiler", "SPSoftwareDataType"]
        )

    try:
        import gurobipy as gp
        info["gurobipy_version"] = getattr(gp, "__version__", None)
        info["gurobi_version"] = ".".join(
            str(x) for x in gp.gurobi.version()
        )
    except Exception as error:
        info["gurobi_error"] = f"{type(error).__name__}: {error}"

    return info


def main() -> None:
    args = parse_args()

    if args.pivot_runs < 1:
        raise ValueError("--pivot-runs must be at least 1.")
    if args.lp_repetitions < 1:
        raise ValueError("--lp-repetitions must be at least 1.")

    algorithms = parse_algorithms(args.algorithms)
    raw_path = resolve(args.raw_output)
    summary_path = resolve(args.summary_output)
    machine_path = resolve(args.machine_output)

    if args.restart:
        for path in (raw_path, summary_path, machine_path):
            if path.exists():
                path.unlink()

    raw_rows = read_csv(raw_path) if raw_path.exists() else []
    completed = {
        str(row.get("task_key", "")).strip()
        for row in raw_rows
        if str(row.get("task_key", "")).strip()
    }

    atomic_write_json(machine_path, machine_specifications())

    new_count = 0
    failures: list[str] = []

    def add_row(row: dict[str, Any]) -> bool:
        nonlocal new_count
        if args.limit is not None and new_count >= args.limit:
            return False
        raw_rows.append(row)
        completed.add(str(row["task_key"]))
        new_count += 1
        atomic_write_csv(raw_path, raw_rows, RAW_FIELDS)
        atomic_write_csv(summary_path, summarize(raw_rows), SUMMARY_FIELDS)
        print(
            f"SAVED {row['dataset']} {row['graph_variant']} "
            f"{row['algorithm']} | n={row['n']} "
            f"p_delete={row['p_delete']}",
            flush=True,
        )
        return True

    def limit_reached() -> bool:
        return args.limit is not None and new_count >= args.limit

    # ------------------------------------------------------------------
    # Facebook
    # ------------------------------------------------------------------
    if args.dataset in {"facebook", "all"} and not limit_reached():
        facebook_rows = read_csv(resolve(args.facebook_table))
        available_egos = sorted(
            {
                str(int(float(row["ego_id"])))
                for row in facebook_rows
                if str(row.get("ego_id", "")).strip()
            },
            key=int,
        )

        if str(args.facebook_egos).strip().lower() == "all":
            complete_egos = available_egos
        else:
            complete_egos = [
                str(int(x)) for x in split_csv(args.facebook_egos)
            ]

        lp_egos = {
            str(int(x)) for x in split_csv(args.facebook_lp_egos)
        }

        p_values_by_ego: dict[str, list[float]] = defaultdict(list)
        n_by_ego: dict[str, int] = {}
        for row in facebook_rows:
            ego = str(int(float(row["ego_id"])))
            n_by_ego[ego] = int(float(row["n"]))
            if int(float(row["seed"])) == args.facebook_edge_seed:
                p = float(row["p_delete"])
                if p not in p_values_by_ego[ego]:
                    p_values_by_ego[ego].append(p)

        for ego in complete_egos:
            if limit_reached():
                break
            try:
                complete_matrix = build_facebook_matrix(ego)
                n = int(complete_matrix.shape[0])
                if n_by_ego.get(ego) not in (None, n):
                    raise ValueError(
                        f"Facebook n mismatch for ego {ego}: "
                        f"matrix={n}, table={n_by_ego.get(ego)}"
                    )

                if "pivot" in algorithms:
                    for row in benchmark_pivot_rows(
                        dataset="facebook",
                        instance_id=f"ego_{ego}",
                        ego_id=ego,
                        n=n,
                        variant="complete",
                        p_delete=0.0,
                        graph_seed="",
                        deletion_seed="",
                        cluster_sizes="",
                        matrix=complete_matrix,
                        pivot_seeds=range(1, args.pivot_runs + 1),
                        completed=completed,
                    ):
                        if not add_row(row):
                            break

                if "lp" in algorithms and ego in lp_egos and not limit_reached():
                    for row in benchmark_lp_rows(
                        dataset="facebook",
                        instance_id=f"ego_{ego}",
                        ego_id=ego,
                        n=n,
                        variant="complete",
                        p_delete=0.0,
                        graph_seed="",
                        deletion_seed="",
                        cluster_sizes="",
                        matrix=complete_matrix,
                        repetitions=args.lp_repetitions,
                        completed=completed,
                    ):
                        if not add_row(row):
                            break

                # Edge rows only exist in the paper for LP-eligible ego IDs.
                if ego in lp_egos:
                    for p_delete in sorted(p_values_by_ego[ego]):
                        if limit_reached():
                            break
                        edge_matrix, _ = delete_edges(
                            complete_matrix.copy(),
                            p_delete,
                            args.facebook_edge_seed,
                        )

                        if "pivot" in algorithms:
                            for row in benchmark_pivot_rows(
                                dataset="facebook",
                                instance_id=(
                                    f"ego_{ego}_p{probability_key(p_delete)}"
                                    f"_seed{args.facebook_edge_seed}"
                                ),
                                ego_id=ego,
                                n=n,
                                variant="edge_deleted",
                                p_delete=p_delete,
                                graph_seed="",
                                deletion_seed=args.facebook_edge_seed,
                                cluster_sizes="",
                                matrix=edge_matrix,
                                pivot_seeds=range(
                                    1,
                                    args.pivot_runs + 1,
                                ),
                                completed=completed,
                            ):
                                if not add_row(row):
                                    break

                        if "lp" in algorithms and not limit_reached():
                            for row in benchmark_lp_rows(
                                dataset="facebook",
                                instance_id=(
                                    f"ego_{ego}_p{probability_key(p_delete)}"
                                    f"_seed{args.facebook_edge_seed}"
                                ),
                                ego_id=ego,
                                n=n,
                                variant="edge_deleted",
                                p_delete=p_delete,
                                graph_seed="",
                                deletion_seed=args.facebook_edge_seed,
                                cluster_sizes="",
                                matrix=edge_matrix,
                                repetitions=args.lp_repetitions,
                                completed=completed,
                            ):
                                if not add_row(row):
                                    break

            except Exception as error:
                message = (
                    f"Facebook ego {ego}: "
                    f"{type(error).__name__}: {error}"
                )
                failures.append(message)
                print("FAILED:", message, file=sys.stderr)
                if not args.continue_on_error:
                    raise

    # ------------------------------------------------------------------
    # Clique
    # ------------------------------------------------------------------
    if args.dataset in {"clique", "all"} and not limit_reached():
        clique_rows = [
            row
            for row in read_csv(resolve(args.clique_table))
            if str(
                row.get("graph_family", row.get("graph_type", ""))
            ).strip().lower() == "clique"
        ]
        selected_seeds = set(parse_int_spec(args.clique_graph_seeds))

        if (
            "lp" in algorithms
            and args.reuse_existing_clique_edge_lp
        ):
            for source_row in clique_rows:
                if limit_reached():
                    break
                seed = int(float(source_row.get("seed", "1")))
                n = int(float(source_row["n"]))
                if seed not in selected_seeds:
                    continue
                if args.clique_max_n is not None and n > args.clique_max_n:
                    continue
                imported = imported_clique_edge_lp_row(
                    source_row,
                    completed,
                )
                if imported is not None:
                    add_row(imported)

        # One reconstruction per graph configuration and graph seed.
        complete_specs: dict[tuple[Any, ...], dict[str, Any]] = {}
        edge_specs: dict[tuple[Any, ...], dict[str, Any]] = {}

        for row in clique_rows:
            seed = int(float(row.get("seed", "1")))
            if seed not in selected_seeds:
                continue

            n = int(float(row["n"]))
            if args.clique_max_n is not None and n > args.clique_max_n:
                continue

            sizes = parse_cluster_sizes(
                row.get("cluster_sizes"),
                row.get("file_name", ""),
            )
            if not sizes or sum(sizes) != n:
                raise ValueError(
                    f"Could not reconstruct clique sizes for "
                    f"{clique_graph_identity(row)}"
                )

            graph_id = clique_graph_identity(row)
            p_inside = (
                to_float(row.get("p_pos_inside"))
                or 0.9
            )
            p_between = (
                to_float(row.get("p_pos_between"))
                or 0.1
            )
            complete_key = (
                graph_id,
                seed,
                tuple(sizes),
                p_inside,
                p_between,
            )
            complete_specs.setdefault(
                complete_key,
                {
                    "graph_id": graph_id,
                    "seed": seed,
                    "n": n,
                    "sizes": sizes,
                    "p_inside": p_inside,
                    "p_between": p_between,
                },
            )

            p_delete = float(row["p_delete"])
            edge_key = (*complete_key, probability_key(p_delete))
            edge_specs.setdefault(
                edge_key,
                {
                    **complete_specs[complete_key],
                    "p_delete": p_delete,
                },
            )

        for spec in complete_specs.values():
            if limit_reached():
                break
            try:
                complete_matrix, _ = generate_clique_signed_graph(
                    cluster_sizes=spec["sizes"],
                    p_pos_inside=spec["p_inside"],
                    p_pos_between=spec["p_between"],
                    seed=spec["seed"],
                )
                instance_id = (
                    f"{spec['graph_id']}|seed={spec['seed']}"
                )
                sizes_text = json.dumps(spec["sizes"])

                if "pivot" in algorithms:
                    for row in benchmark_pivot_rows(
                        dataset="clique",
                        instance_id=instance_id,
                        ego_id="",
                        n=spec["n"],
                        variant="complete",
                        p_delete=0.0,
                        graph_seed=spec["seed"],
                        deletion_seed="",
                        cluster_sizes=sizes_text,
                        matrix=complete_matrix,
                        pivot_seeds=range(1, args.pivot_runs + 1),
                        completed=completed,
                    ):
                        if not add_row(row):
                            break

                if "lp" in algorithms and not limit_reached():
                    for row in benchmark_lp_rows(
                        dataset="clique",
                        instance_id=instance_id,
                        ego_id="",
                        n=spec["n"],
                        variant="complete",
                        p_delete=0.0,
                        graph_seed=spec["seed"],
                        deletion_seed="",
                        cluster_sizes=sizes_text,
                        matrix=complete_matrix,
                        repetitions=args.lp_repetitions,
                        completed=completed,
                    ):
                        if not add_row(row):
                            break

            except Exception as error:
                message = (
                    f"Clique complete {spec['graph_id']} seed={spec['seed']}: "
                    f"{type(error).__name__}: {error}"
                )
                failures.append(message)
                print("FAILED:", message, file=sys.stderr)
                if not args.continue_on_error:
                    raise

        for spec in edge_specs.values():
            if limit_reached():
                break
            try:
                complete_matrix, _ = generate_clique_signed_graph(
                    cluster_sizes=spec["sizes"],
                    p_pos_inside=spec["p_inside"],
                    p_pos_between=spec["p_between"],
                    seed=spec["seed"],
                )
                edge_matrix, _ = delete_edges(
                    complete_matrix.copy(),
                    spec["p_delete"],
                    spec["seed"],
                )
                instance_id = (
                    f"{spec['graph_id']}|seed={spec['seed']}|"
                    f"p={probability_key(spec['p_delete'])}"
                )
                sizes_text = json.dumps(spec["sizes"])

                if "pivot" in algorithms:
                    for row in benchmark_pivot_rows(
                        dataset="clique",
                        instance_id=instance_id,
                        ego_id="",
                        n=spec["n"],
                        variant="edge_deleted",
                        p_delete=spec["p_delete"],
                        graph_seed=spec["seed"],
                        deletion_seed=spec["seed"],
                        cluster_sizes=sizes_text,
                        matrix=edge_matrix,
                        pivot_seeds=range(1, args.pivot_runs + 1),
                        completed=completed,
                    ):
                        if not add_row(row):
                            break

                if (
                    "lp" in algorithms
                    and not args.reuse_existing_clique_edge_lp
                    and not limit_reached()
                ):
                    for row in benchmark_lp_rows(
                        dataset="clique",
                        instance_id=instance_id,
                        ego_id="",
                        n=spec["n"],
                        variant="edge_deleted",
                        p_delete=spec["p_delete"],
                        graph_seed=spec["seed"],
                        deletion_seed=spec["seed"],
                        cluster_sizes=sizes_text,
                        matrix=edge_matrix,
                        repetitions=args.lp_repetitions,
                        completed=completed,
                    ):
                        if not add_row(row):
                            break

            except Exception as error:
                message = (
                    f"Clique edge {spec['graph_id']} seed={spec['seed']} "
                    f"p={spec['p_delete']}: "
                    f"{type(error).__name__}: {error}"
                )
                failures.append(message)
                print("FAILED:", message, file=sys.stderr)
                if not args.continue_on_error:
                    raise

    summary_rows = summarize(raw_rows)
    atomic_write_csv(raw_path, raw_rows, RAW_FIELDS)
    atomic_write_csv(summary_path, summary_rows, SUMMARY_FIELDS)

    print("\n" + "=" * 78)
    print("RUNTIME BENCHMARK FINISHED")
    print("=" * 78)
    print("New raw observations:", new_count)
    print("Total raw observations:", len(raw_rows))
    print("Summary rows:", len(summary_rows))
    print("Failures:", len(failures))
    print("Raw output:", raw_path)
    print("Summary output:", summary_path)
    print("Machine specifications:", machine_path)

    if failures:
        print("\nFailures:")
        for failure in failures:
            print("-", failure)


if __name__ == "__main__":
    main()
