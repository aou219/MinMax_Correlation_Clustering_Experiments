#!/usr/bin/env python3
"""
Insert measured normal correlation-clustering runtimes into the already
generated Facebook and clique paper tables.

Run make_paper_tables.py first, then run this script.

The script only changes:
    pivot_runtime_seconds_average
    lp_runtime_seconds_average

A timestamped backup of each paper table is made automatically.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BENCHMARK = (
    REPO_ROOT / "results/research_tables/normal_cc_runtime_benchmarks.csv"
)
DEFAULT_FACEBOOK = (
    REPO_ROOT
    / "results/research_tables/facebook_correlation_clustering_table.csv"
)
DEFAULT_CLIQUE = (
    REPO_ROOT
    / "results/research_tables/clique_correlation_clustering_table.csv"
)

RUNTIME_COLUMNS = [
    "pivot_runtime_seconds_average",
    "lp_runtime_seconds_average",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--facebook-table", type=Path, default=DEFAULT_FACEBOOK)
    parser.add_argument("--clique-table", type=Path, default=DEFAULT_CLIQUE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def probability_key(value: Any) -> str:
    return f"{float(value):.12g}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def atomic_write(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
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


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(
        f"{path.stem}_before_runtime_merge_{stamp}{path.suffix}"
    )
    shutil.copy2(path, target)
    return target


def runtime_maps(
    benchmark_rows: Sequence[Mapping[str, str]],
) -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    dict[tuple[int, str, str], dict[str, str]],
]:
    facebook: dict[tuple[str, str, str], dict[str, str]] = {}
    clique: dict[tuple[int, str, str], dict[str, str]] = {}

    for row in benchmark_rows:
        algorithm = str(row.get("algorithm", "")).strip()
        if algorithm not in {"pivot", "lp"}:
            continue

        value = str(row.get("average_runtime_seconds", "")).strip()
        if not value:
            continue

        dataset = str(row.get("dataset", "")).strip()
        variant = str(row.get("graph_variant", "")).strip()
        p_delete = probability_key(row.get("p_delete", 0))

        if dataset == "facebook":
            key = (
                str(int(float(row["ego_id"]))),
                variant,
                p_delete,
            )
            facebook.setdefault(key, {})[algorithm] = value

        elif dataset == "clique":
            key = (
                int(float(row["n"])),
                variant,
                p_delete,
            )
            clique.setdefault(key, {})[algorithm] = value

    return facebook, clique


def ensure_columns(fields: list[str]) -> list[str]:
    output = list(fields)
    for column in RUNTIME_COLUMNS:
        if column not in output:
            output.append(column)
    return output


def update_facebook(
    rows: list[dict[str, str]],
    mapping: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[int, list[str]]:
    updated = 0
    missing: list[str] = []

    for row in rows:
        key = (
            str(int(float(row["ego_id"]))),
            str(row.get("graph_variant", "")).strip(),
            probability_key(row.get("p_delete", 0)),
        )
        values = mapping.get(key)

        if values is None:
            missing.append(
                f"facebook ego={key[0]} variant={key[1]} p={key[2]}"
            )
            continue

        row["pivot_runtime_seconds_average"] = values.get("pivot", "")
        row["lp_runtime_seconds_average"] = values.get("lp", "")
        updated += 1

    return updated, missing


def update_clique(
    rows: list[dict[str, str]],
    mapping: dict[tuple[int, str, str], dict[str, str]],
) -> tuple[int, list[str]]:
    updated = 0
    missing: list[str] = []

    for row in rows:
        p_delete = probability_key(row.get("p_delete", 0))
        variant = "complete" if float(p_delete) == 0 else "edge_deleted"
        key = (
            int(float(row["n"])),
            variant,
            p_delete,
        )
        values = mapping.get(key)

        if values is None:
            missing.append(
                f"clique n={key[0]} variant={key[1]} p={key[2]}"
            )
            continue

        row["pivot_runtime_seconds_average"] = values.get("pivot", "")
        row["lp_runtime_seconds_average"] = values.get("lp", "")
        updated += 1

    return updated, missing


def main() -> None:
    args = parse_args()
    benchmark_path = resolve(args.benchmark)
    facebook_path = resolve(args.facebook_table)
    clique_path = resolve(args.clique_table)

    _, benchmark_rows = read_csv(benchmark_path)
    facebook_map, clique_map = runtime_maps(benchmark_rows)

    fb_fields, fb_rows = read_csv(facebook_path)
    cl_fields, cl_rows = read_csv(clique_path)

    fb_fields = ensure_columns(fb_fields)
    cl_fields = ensure_columns(cl_fields)

    fb_updated, fb_missing = update_facebook(fb_rows, facebook_map)
    cl_updated, cl_missing = update_clique(cl_rows, clique_map)

    print("Facebook rows updated:", fb_updated, "/", len(fb_rows))
    print("Clique rows updated:", cl_updated, "/", len(cl_rows))
    print("Facebook benchmark matches missing:", len(fb_missing))
    print("Clique benchmark matches missing:", len(cl_missing))

    if fb_missing:
        print("\nUnmatched Facebook rows:")
        for item in fb_missing:
            print("-", item)

    if cl_missing:
        print("\nUnmatched clique rows:")
        for item in cl_missing:
            print("-", item)

    if args.dry_run:
        print("\nDRY RUN: no files changed.")
        return

    fb_backup = backup(facebook_path)
    cl_backup = backup(clique_path)

    atomic_write(facebook_path, fb_fields, fb_rows)
    atomic_write(clique_path, cl_fields, cl_rows)

    print("\nFacebook backup:", fb_backup)
    print("Clique backup:", cl_backup)
    print("Facebook table:", facebook_path)
    print("Clique table:", clique_path)


if __name__ == "__main__":
    main()
