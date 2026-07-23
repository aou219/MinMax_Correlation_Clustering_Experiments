#!/usr/bin/env python3
"""Read-only MinMaxLP reproducibility check for Facebook ego 3980.

Reruns one complete graph and 120 edge-deleted graphs, then compares the
stored deterministic MinMaxLP outputs with the independent rerun. The main
experiment CSV is never modified. Runtime values are recorded but not compared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.edge_deletion import delete_edges
from src.experiment_helpers import compute_min_max_lp_data
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)

DEFAULT_TABLE = REPO_ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
DEFAULT_REPORT = REPO_ROOT / "results/research_tables/reproducibility/minmaxlp_ego3980_check.csv"
DEFAULT_SUMMARY = REPO_ROOT / "results/research_tables/reproducibility/minmaxlp_ego3980_check_summary.json"
FIELDS = [
    "lp_cost", "rounding_cost", "max_disagreement_vertex", "cluster_count",
    "r", "r2", "method", "norm",
]
REPORT_FIELDS = [
    "task", "ego_id", "n", "graph_variant", "p_delete", "seed",
    "deleted_edges", "matrix_sha256",
    "stored_lp_cost", "rerun_lp_cost",
    "stored_rounding_cost", "rerun_rounding_cost",
    "stored_max_disagreement_vertex", "rerun_max_disagreement_vertex",
    "stored_cluster_count", "rerun_cluster_count",
    "stored_r", "rerun_r", "stored_r2", "rerun_r2",
    "stored_method", "rerun_method", "stored_norm", "rerun_norm",
    "rerun_lp_runtime_seconds", "rerun_rounding_runtime_seconds",
    "rerun_total_runtime_seconds", "comparisons", "matches",
    "failed_fields", "status", "checked_utc",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--ego-id", default="3980")
    p.add_argument("--p-delete-values", default="0.05,0.15,0.25,0.4")
    p.add_argument("--seeds", default="1-30")
    p.add_argument("--r", type=float, default=0.4)
    p.add_argument("--r2", type=float, default=0.4)
    p.add_argument("--method", type=int, default=2)
    p.add_argument("--abs-tol", type=float, default=1e-6)
    p.add_argument("--rel-tol", type=float, default=1e-8)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def pkey(value: Any) -> str:
    return f"{float(value):.12g}"


def parse_floats(raw: str) -> list[float]:
    values = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not values:
        raise ValueError("No p_delete values selected")
    return values


def parse_ints(raw: str) -> list[int]:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            if b < a:
                raise ValueError(f"Invalid range: {part}")
            values.update(range(a, b + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError("No seeds selected")
    return sorted(values)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_matrix(matrix: np.ndarray) -> str:
    a = np.ascontiguousarray(matrix)
    h = hashlib.sha256()
    # Same convention as scripts/run_experiment.py.
    h.update(str(a.shape).encode())
    h.update(b"|")
    h.update(str(a.dtype).encode())
    h.update(b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header: {path}")
        return list(reader)


def locate_edges(ego: str) -> Path:
    candidates = [
        REPO_ROOT / f"data/facebook/{ego}.edges",
        REPO_ROOT / f"data/facebook/facebook_3/{ego}.edges",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No .edges file for ego {ego}")


def complete_matrix(ego: str) -> tuple[np.ndarray, Path]:
    path = locate_edges(ego)
    edge_nodes, edges = load_facebook_ego_edges(str(path))
    nodes = sorted(edge_nodes)
    matrix, _, _, _ = build_complete_signed_matrix_from_facebook_sample(nodes, edges)
    return matrix, path


def fnum(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def inum(value: Any) -> int | None:
    value = fnum(value)
    return None if value is None else int(round(value))


def norm(value: Any) -> str:
    text = str(value).strip().lower()
    return "inf" if text in {"inf", "+inf", "infinity", "np.inf", "math.inf"} else text


def stored(row: Mapping[str, str], scope: str) -> dict[str, Any]:
    p = f"{scope}_min_max_lp"
    return {
        "lp_cost": row.get(f"{p}_cost", ""),
        "rounding_cost": row.get(f"{p}_rounding_cost", ""),
        "max_disagreement_vertex": row.get(f"{p}_max_disagreement_vertex", ""),
        "cluster_count": row.get(f"{p}_cluster_count", ""),
        "r": row.get(f"{p}_r", ""),
        "r2": row.get(f"{p}_r2", ""),
        "method": row.get(f"{p}_method", ""),
        "norm": row.get(f"{p}_norm", ""),
    }


def rerun(result: Mapping[str, Any]) -> dict[str, Any]:
    return {field: result.get(field) for field in FIELDS}


def equal(field: str, left: Any, right: Any, abs_tol: float, rel_tol: float) -> bool:
    if field in {"lp_cost", "rounding_cost", "r", "r2"}:
        a, b = fnum(left), fnum(right)
        return a is not None and b is not None and math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol)
    if field in {"max_disagreement_vertex", "cluster_count", "method"}:
        a, b = inum(left), inum(right)
        return a is not None and b is not None and a == b
    if field == "norm":
        return norm(left) == norm(right)
    raise ValueError(field)


def dispose() -> None:
    try:
        import gurobipy as gp
        gp.disposeDefaultEnv()
    except Exception:
        pass


def compare(task: str, ego: str, n: int, variant: str, p_delete: float, seed: int | str,
            deleted: int, matrix: np.ndarray, old: Mapping[str, Any], result: Mapping[str, Any],
            abs_tol: float, rel_tol: float) -> dict[str, Any]:
    new = rerun(result)
    failed = [field for field in FIELDS if not equal(field, old[field], new[field], abs_tol, rel_tol)]
    return {
        "task": task, "ego_id": ego, "n": n, "graph_variant": variant,
        "p_delete": pkey(p_delete), "seed": seed, "deleted_edges": deleted,
        "matrix_sha256": sha_matrix(matrix),
        "stored_lp_cost": old["lp_cost"], "rerun_lp_cost": new["lp_cost"],
        "stored_rounding_cost": old["rounding_cost"], "rerun_rounding_cost": new["rounding_cost"],
        "stored_max_disagreement_vertex": old["max_disagreement_vertex"],
        "rerun_max_disagreement_vertex": new["max_disagreement_vertex"],
        "stored_cluster_count": old["cluster_count"], "rerun_cluster_count": new["cluster_count"],
        "stored_r": old["r"], "rerun_r": new["r"], "stored_r2": old["r2"], "rerun_r2": new["r2"],
        "stored_method": old["method"], "rerun_method": new["method"],
        "stored_norm": old["norm"], "rerun_norm": new["norm"],
        "rerun_lp_runtime_seconds": result.get("lp_runtime_seconds", ""),
        "rerun_rounding_runtime_seconds": result.get("rounding_runtime_seconds", ""),
        "rerun_total_runtime_seconds": result.get("total_runtime_seconds", ""),
        "comparisons": len(FIELDS), "matches": len(FIELDS) - len(failed),
        "failed_fields": ",".join(failed), "status": "PASS" if not failed else "FAIL",
        "checked_utc": utc(),
    }


def git(*items: str) -> str | None:
    try:
        return subprocess.check_output(["git", *items], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main() -> None:
    a = args()
    table = resolve(a.table)
    report = resolve(a.report)
    summary = resolve(a.summary)
    p_values = parse_floats(a.p_delete_values)
    seeds = parse_ints(a.seeds)
    expected = 1 + len(p_values) * len(seeds)
    table_hash_before = sha_file(table)

    rows = read_table(table)
    selected = [row for row in rows if str(row.get("ego_id", "")).strip() == str(a.ego_id)
                and pkey(row.get("p_delete", "")) in {pkey(x) for x in p_values}
                and int(float(row.get("seed", 0))) in set(seeds)]
    if len(selected) != len(p_values) * len(seeds):
        raise ValueError(f"Expected {len(p_values) * len(seeds)} ego rows, found {len(selected)}")
    row_map = {(pkey(row["p_delete"]), int(float(row["seed"]))): row for row in selected}
    if len(row_map) != len(selected):
        raise ValueError("Duplicate ego/p_delete/seed rows")

    matrix, edge_file = complete_matrix(str(a.ego_id))
    n = matrix.shape[0]
    if {int(float(row["n"])) for row in selected} != {n}:
        raise ValueError("Stored and reconstructed graph sizes differ")

    # Check that all stored complete values agree and parameters are correct.
    first = selected[0]
    complete_ref = stored(first, "complete")
    for row in selected:
        for field in FIELDS:
            if not equal(field, complete_ref[field], stored(row, "complete")[field], a.abs_tol, a.rel_tol):
                raise ValueError(f"Inconsistent complete field across rows: {field}")
        for scope in ("complete", "edge"):
            values = stored(row, scope)
            if not math.isclose(float(values["r"]), a.r, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Stored {scope} r is not {a.r}")
            if not math.isclose(float(values["r2"]), a.r2, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Stored {scope} r2 is not {a.r2}")
            if int(float(values["method"])) != a.method:
                raise ValueError(f"Stored {scope} method is not {a.method}")
            if norm(values["norm"]) != "inf":
                raise ValueError(f"Stored {scope} norm is not inf")

    print("=" * 78)
    print("MINMAXLP REPRODUCIBILITY CHECK — EGO 3980")
    print("=" * 78)
    print("Instances:", expected, "(1 complete +", expected - 1, "edge-deleted)")
    print("Parameters:", f"r={a.r}, r2={a.r2}, method={a.method}, norm=inf")
    print("Complete matrix SHA-256:", sha_matrix(matrix))
    print("Main table is read-only.")

    results: list[dict[str, Any]] = []

    def save() -> None:
        atomic_csv(report, results)

    try:
        print("\nRUN complete graph")
        result = compute_min_max_lp_data(matrix, True, r=a.r, r2=a.r2, method=a.method, norm=np.inf)
        item = compare("complete|minmaxlp|3980", str(a.ego_id), n, "complete", 0.0, "", 0,
                       matrix, complete_ref, result, a.abs_tol, a.rel_tol)
        results.append(item)
        save()
        print("Status:", item["status"], item["failed_fields"])
    finally:
        dispose()

    for p_delete in p_values:
        for seed in seeds:
            row = row_map[(pkey(p_delete), seed)]
            edge_matrix, deleted = delete_edges(matrix.copy(), p_delete, seed)
            print(f"RUN p_delete={p_delete}, seed={seed}, deleted={deleted}")
            try:
                result = compute_min_max_lp_data(edge_matrix, True, r=a.r, r2=a.r2, method=a.method, norm=np.inf)
                item = compare(f"edge|minmaxlp|3980|{pkey(p_delete)}|{seed}", str(a.ego_id), n,
                               "edge_deleted", p_delete, seed, deleted, edge_matrix, stored(row, "edge"),
                               result, a.abs_tol, a.rel_tol)
            except Exception as error:
                item = {
                    "task": f"edge|minmaxlp|3980|{pkey(p_delete)}|{seed}", "ego_id": str(a.ego_id),
                    "n": n, "graph_variant": "edge_deleted", "p_delete": pkey(p_delete), "seed": seed,
                    "deleted_edges": deleted, "matrix_sha256": sha_matrix(edge_matrix), "comparisons": 0,
                    "matches": 0, "failed_fields": f"{type(error).__name__}: {error}", "status": "ERROR",
                    "checked_utc": utc(),
                }
                if not a.continue_on_error:
                    results.append(item)
                    save()
                    raise
            finally:
                dispose()
            results.append(item)
            save()
            print("Status:", item["status"], item.get("failed_fields", ""))

    table_hash_after = sha_file(table)
    passed = sum(row["status"] == "PASS" for row in results)
    failed = sum(row["status"] == "FAIL" for row in results)
    errors = sum(row["status"] == "ERROR" for row in results)
    comparisons = sum(int(row.get("comparisons", 0)) for row in results)
    matches = sum(int(row.get("matches", 0)) for row in results)
    source_changed = table_hash_before != table_hash_after
    status = "passed" if len(results) == expected and failed == 0 and errors == 0 and not source_changed else "failed"

    versions: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    try:
        import gurobipy as gp
        versions["gurobipy"] = getattr(gp, "__version__", None)
        versions["gurobi"] = ".".join(str(x) for x in gp.gurobi.version())
    except Exception as error:
        versions["gurobi_error"] = str(error)

    info = {
        "status": status, "checked_utc": utc(), "ego_id": str(a.ego_id),
        "expected_instances": expected, "checked_instances": len(results),
        "passed_instances": passed, "failed_instances": failed, "error_instances": errors,
        "field_comparisons": comparisons, "field_matches": matches,
        "field_mismatches": comparisons - matches, "runtime_compared": False,
        "parameters": {"r": a.r, "r2": a.r2, "method": a.method, "norm": "inf",
                       "abs_tolerance": a.abs_tol, "rel_tolerance": a.rel_tol},
        "preprocessing": "sorted endpoints occurring in the .edges file only",
        "source_table": str(table), "source_table_sha256_before": table_hash_before,
        "source_table_sha256_after": table_hash_after, "source_table_changed": source_changed,
        "edges_file": str(edge_file), "edges_file_sha256": sha_file(edge_file),
        "complete_matrix_sha256": sha_matrix(matrix),
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--porcelain")), "versions": versions,
        "report": str(report),
    }
    atomic_json(summary, info)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Status:", status)
    print("Instances:", f"{passed} passed, {failed} failed, {errors} errors")
    print("Field comparisons:", f"{matches}/{comparisons} matched")
    print("Source table changed:", source_changed)
    print("Report:", report)
    print("Summary:", summary)
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
