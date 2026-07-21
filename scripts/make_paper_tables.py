#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
DEFAULT_MINMAX_OUTPUT = REPO_ROOT / "results/research_tables/facebook_minmax_table.csv"
DEFAULT_CC_OUTPUT = REPO_ROOT / "results/research_tables/facebook_correlation_clustering_table.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--minmax-output", type=Path, default=DEFAULT_MINMAX_OUTPUT)
    p.add_argument("--cc-output", type=Path, default=DEFAULT_CC_OUTPUT)
    p.add_argument("--d-hat", type=int, default=8)
    p.add_argument("--lambda-value", type=int, default=5)
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def rounded(value: float | None, digits: int = 6) -> str | float:
    return "" if value is None else round(value, digits)


def average(values: Iterable[Any]) -> float | None:
    nums = [x for x in (to_float(v) for v in values) if x is not None]
    return None if not nums else sum(nums) / len(nums)


def ratio(cc: Any, lp: Any) -> float | None:
    cc_num, lp_num = to_float(cc), to_float(lp)
    if cc_num is None or lp_num is None or lp_num == 0:
        return None
    return cc_num / lp_num


def get_seed(row: dict[str, str]) -> str:
    seed = str(row.get("seed", "")).strip()
    return seed or "1"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fixed_rows(rows: list[dict[str, str]], d_hat: int, lam: int) -> list[dict[str, str]]:
    out = []
    for row in rows:
        d = to_float(row.get("edge_min_max_cc_d_hat"))
        l = to_float(row.get("edge_min_max_cc_lambda"))
        if d is not None and l is not None and int(d) == d_hat and int(l) == lam:
            out.append(row)
    if not out:
        raise ValueError(f"No rows found for d_hat={d_hat}, lambda={lam}")
    return out


def make_minmax_table(rows: list[dict[str, str]], d_hat: int, lam: int) -> list[dict[str, Any]]:
    complete_by_ego: dict[str, dict[str, str]] = {}
    edge_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    metadata: dict[tuple[str, str], dict[str, str]] = {}

    for row in rows:
        ego = str(row.get("ego_id", "")).strip()
        if not ego:
            continue
        n = str(row.get("n", "")).strip()
        seed = get_seed(row)
        p_delete = str(row.get("p_delete", "")).strip()

        cd = to_float(row.get("complete_min_max_cc_d_hat"))
        cl = to_float(row.get("complete_min_max_cc_lambda"))
        if cd is not None and cl is not None and int(cd) == d_hat and int(cl) == lam:
            complete_by_ego.setdefault(ego, row)

        cc = to_float(row.get("edge_min_max_cc_max_disagreement"))
        if cc is None or not p_delete:
            continue
        lp = to_float(row.get("edge_min_max_lp_cost"))
        r = ratio(cc, lp)
        key = (ego, p_delete)
        metadata.setdefault(key, {"n": n})
        obs = {
            "seed": seed,
            "cc": cc,
            "lp": lp,
            "ratio": r,
            "minmaxcc_runtime": row.get(
                "edge_min_max_cc_runtime_seconds", ""
            ),
            "lp_runtime": row.get(
                "edge_min_max_lp_runtime_seconds", ""
            ),
        }
        old = edge_groups[key].get(seed)
        if old is None:
            edge_groups[key][seed] = obs
        elif abs(old["cc"] - cc) > 1e-8:
            raise ValueError(f"Conflicting duplicate for ego={ego}, p_delete={p_delete}, seed={seed}")

    output: list[dict[str, Any]] = []

    for ego, row in complete_by_ego.items():
        cc = to_float(row.get("complete_min_max_cc_max_disagreement"))
        lp = to_float(row.get("complete_min_max_lp_cost"))
        r = ratio(cc, lp)
        output.append({
            "ego_id": ego,
            "n": str(row.get("n", "")).strip(),
            "p_delete": 0,
            "graph_variant": "complete",
            "d_hat": d_hat,
            "lambda": lam,
            "number_of_seeds": 1,
            "minmaxcc_best": rounded(cc),
            "minmaxcc_average": rounded(cc),
            "minmaxcc_worst": rounded(cc),
            "min_max_lp_cost_best": rounded(lp),
            "min_max_lp_cost_average": rounded(lp),
            "min_max_lp_cost_worst": rounded(lp),
            "minmaxcc_best_to_lp_ratio": rounded(r),
            "minmaxcc_average_to_lp_ratio": rounded(r),
            "minmaxcc_worst_to_lp_ratio": rounded(r),
            "minmaxcc_runtime_seconds_average": rounded(
                to_float(
                    row.get("complete_min_max_cc_runtime_seconds")
                )
            ),
            "min_max_lp_runtime_seconds_average": rounded(
                to_float(
                    row.get("complete_min_max_lp_runtime_seconds")
                )
            ),
        })

    for key, by_seed in edge_groups.items():
        ego, p_delete = key
        obs = list(by_seed.values())
        ratio_obs = [x for x in obs if x["ratio"] is not None]

        if ratio_obs:
            best = min(ratio_obs, key=lambda x: (x["ratio"], x["cc"], int(float(x["seed"]))))
            worst = max(ratio_obs, key=lambda x: (x["ratio"], x["cc"], -int(float(x["seed"]))))
        else:
            best = min(obs, key=lambda x: (x["cc"], int(float(x["seed"]))))
            worst = max(obs, key=lambda x: (x["cc"], -int(float(x["seed"]))))

        output.append({
            "ego_id": ego,
            "n": metadata[key]["n"],
            "p_delete": p_delete,
            "graph_variant": "edge_deleted",
            "d_hat": d_hat,
            "lambda": lam,
            "number_of_seeds": len(obs),
            "minmaxcc_best": rounded(best["cc"]),
            "minmaxcc_average": rounded(average(x["cc"] for x in obs)),
            "minmaxcc_worst": rounded(worst["cc"]),
            "min_max_lp_cost_best": rounded(best["lp"]),
            "min_max_lp_cost_average": rounded(average(x["lp"] for x in obs)),
            "min_max_lp_cost_worst": rounded(worst["lp"]),
            "minmaxcc_best_to_lp_ratio": rounded(best["ratio"]),
            "minmaxcc_average_to_lp_ratio": rounded(average(x["ratio"] for x in ratio_obs)),
            "minmaxcc_worst_to_lp_ratio": rounded(worst["ratio"]),
            "minmaxcc_runtime_seconds_average": rounded(
                average(x["minmaxcc_runtime"] for x in obs)
            ),
            "min_max_lp_runtime_seconds_average": rounded(
                average(x["lp_runtime"] for x in obs)
            ),
        })

    output.sort(
        key=lambda r: (
            int(float(r["n"])),
            int(float(r["ego_id"])),
            0 if r["graph_variant"] == "complete" else 1,
            float(r["p_delete"]),
        )
    )
    return output


def make_cc_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Build the standard correlation-clustering table.

    Complete graph:
        The complete graph is independent of the deletion seed. We keep one
        consistent complete Pivot/LP pair per ego graph.

    Edge-deleted graph:
        The LP currently available in the experiment table was computed for
        deletion seed 1. Therefore the reported edge Pivot result is selected
        explicitly from deletion seed 1 as well.

        This guarantees that:

            edge_pivot_best_cost

        and:

            edge_all_pairs_lp_cost

        come from exactly the same edge-deleted graph.

    Pivot results for deletion seeds 2..30 remain in the raw experiment CSV.
    They are not mixed with the seed-1 LP in this paper table.
    """

    complete_rows: dict[str, dict[str, Any]] = {}
    seed_one_rows: dict[tuple[str, str], dict[str, Any]] = {}

    def numeric_seed(row: dict[str, str]) -> int | None:
        value = to_float(row.get("seed"))
        return None if value is None else int(value)

    def set_consistent(
        current: dict[str, Any],
        key: str,
        value: float | None,
        context: str,
    ) -> None:
        if value is None:
            return

        old = current.get(key)

        if old is None:
            current[key] = value
            return

        if abs(float(old) - value) > 1e-8:
            raise ValueError(
                f"Conflicting {key} values for {context}: "
                f"{old} versus {value}"
            )

    for row in rows:
        ego = str(row.get("ego_id", "")).strip()
        p_delete = str(row.get("p_delete", "")).strip()
        n = str(row.get("n", "")).strip()

        if not ego:
            continue

        # Complete Pivot and LP are repeated in the flat table. Verify that
        # the repeated values are consistent rather than silently taking an
        # arbitrary row.
        complete = complete_rows.setdefault(
            ego,
            {
                "ego_id": ego,
                "n": n,
                "p_delete": 0,
                "graph_variant": "complete",
                "paired_deletion_seed": "",
                "pivot_best_cost": None,
                "lp_cost": None,
            },
        )

        set_consistent(
            complete,
            "pivot_best_cost",
            to_float(row.get("complete_pivot_best_cost")),
            f"complete ego={ego}",
        )
        set_consistent(
            complete,
            "lp_cost",
            to_float(row.get("complete_lp_cost")),
            f"complete ego={ego}",
        )

        # For an edge-deleted comparison, only deletion seed 1 is eligible.
        if numeric_seed(row) != 1 or not p_delete:
            continue

        key = (ego, p_delete)
        edge = seed_one_rows.setdefault(
            key,
            {
                "ego_id": ego,
                "n": n,
                "p_delete": p_delete,
                "graph_variant": "edge_deleted",
                "paired_deletion_seed": 1,
                "pivot_best_cost": None,
                "lp_cost": None,
            },
        )

        set_consistent(
            edge,
            "pivot_best_cost",
            to_float(row.get("edge_pivot_best_cost")),
            f"edge ego={ego}, p_delete={p_delete}, seed=1",
        )
        set_consistent(
            edge,
            "lp_cost",
            to_float(row.get("edge_all_pairs_lp_cost")),
            f"edge ego={ego}, p_delete={p_delete}, seed=1",
        )

    output: list[dict[str, Any]] = []

    for ego, row in complete_rows.items():
        pivot = row.pop("pivot_best_cost")
        lp = row.pop("lp_cost")

        row["pivot_best_cost"] = rounded(pivot)
        row["lp_cost"] = rounded(lp)
        row["pivot_to_lp_ratio"] = rounded(ratio(pivot, lp))
        output.append(row)

    for (ego, p_delete), row in seed_one_rows.items():
        pivot = row.pop("pivot_best_cost")
        lp = row.pop("lp_cost")

        if pivot is None:
            raise ValueError(
                "Missing seed-1 edge Pivot result for "
                f"ego={ego}, p_delete={p_delete}. "
                "Run update_pivot_results.py first."
            )

        if lp is None:
            raise ValueError(
                "Missing seed-1 all-pairs LP result for "
                f"ego={ego}, p_delete={p_delete}."
            )

        row["pivot_best_cost"] = rounded(pivot)
        row["lp_cost"] = rounded(lp)
        row["pivot_to_lp_ratio"] = rounded(ratio(pivot, lp))
        output.append(row)

    output.sort(
        key=lambda r: (
            int(float(r["n"])),
            int(float(r["ego_id"])),
            0 if r["graph_variant"] == "complete" else 1,
            float(r["p_delete"]),
        )
    )

    return output

def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    minmax_output = resolve(args.minmax_output)
    cc_output = resolve(args.cc_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    all_rows = read_rows(input_path)
    selected_rows = fixed_rows(all_rows, args.d_hat, args.lambda_value)
    minmax_rows = make_minmax_table(selected_rows, args.d_hat, args.lambda_value)
    cc_rows = make_cc_table(all_rows)

    minmax_fields = [
        "ego_id", "n", "p_delete",
        "minmaxcc_best",
        "min_max_lp_cost_best",
        "minmaxcc_best_to_lp_ratio",

        "minmaxcc_average",
        "min_max_lp_cost_average",
        "minmaxcc_average_to_lp_ratio",

        "minmaxcc_worst",
        "min_max_lp_cost_worst",
        "minmaxcc_worst_to_lp_ratio",
        "minmaxcc_runtime_seconds_average",
        "min_max_lp_runtime_seconds_average",
    ]
    cc_fields = [
        "ego_id",
        "n",
        "p_delete",
        "graph_variant",
        "paired_deletion_seed",
        "pivot_best_cost",
        "lp_cost",
        "pivot_to_lp_ratio",
    ]

    write_rows(minmax_output, minmax_rows, minmax_fields)
    write_rows(cc_output, cc_rows, cc_fields)

    print("Input:", input_path)
    print("Input rows:", len(all_rows))
    print(f"Fixed parameters: d_hat={args.d_hat}, lambda={args.lambda_value}")
    print("MinMax table:", minmax_output)
    print("MinMax rows:", len(minmax_rows))
    print("Correlation table:", cc_output)
    print("Correlation rows:", len(cc_rows))


if __name__ == "__main__":
    main()
