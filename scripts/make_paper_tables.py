#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/processed/research_tables/minmax_facebook_grid_runs_flat.csv"
DEFAULT_MINMAX_OUTPUT = REPO_ROOT / "results/processed/research_tables/facebook_minmax_table.csv"
DEFAULT_CC_OUTPUT = REPO_ROOT / "results/processed/research_tables/facebook_correlation_clustering_table.csv"


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
            "lp_rounding": row.get("edge_min_max_lp_rounding_cost", ""),
            "lp_runtime": row.get("edge_min_max_lp_runtime_seconds", ""),
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
            "best_seed": "",
            "worst_seed": "",
            "min_max_lp_rounding_cost_best_seed": row.get("complete_min_max_lp_rounding_cost", ""),
            "min_max_lp_runtime_seconds_best_seed": row.get("complete_min_max_lp_runtime_seconds", ""),
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
            "best_seed": best["seed"] if best["ratio"] is not None else "",
            "worst_seed": worst["seed"] if worst["ratio"] is not None else "",
            "min_max_lp_rounding_cost_best_seed": best["lp_rounding"] if best["ratio"] is not None else "",
            "min_max_lp_runtime_seconds_best_seed": best["lp_runtime"] if best["ratio"] is not None else "",
        })

    output.sort(key=lambda r: (int(float(r["ego_id"])), 0 if r["graph_variant"] == "complete" else 1, float(r["p_delete"])))
    return output


def make_cc_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    complete: dict[str, dict[str, Any]] = {}
    edge: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"n": "", "pivot": {}, "lp": {}})

    for row in rows:
        ego = str(row.get("ego_id", "")).strip()
        if not ego:
            continue
        n = str(row.get("n", "")).strip()
        seed = get_seed(row)
        p_delete = str(row.get("p_delete", "")).strip()

        cp = to_float(row.get("complete_pivot_best_cost"))
        clp = to_float(row.get("complete_lp_cost"))
        current = complete.setdefault(ego, {
            "ego_id": ego,
            "n": n,
            "p_delete": 0,
            "graph_variant": "complete",
            "number_of_seeds": 1,
            "pivot_best_cost_average": "",
            "lp_cost_average": "",
        })
        if current["pivot_best_cost_average"] == "" and cp is not None:
            current["pivot_best_cost_average"] = rounded(cp)
        if current["lp_cost_average"] == "" and clp is not None:
            current["lp_cost_average"] = rounded(clp)

        group = edge[(ego, p_delete)]
        group["n"] = n
        ep = to_float(row.get("edge_pivot_best_cost"))
        elp = to_float(row.get("edge_all_pairs_lp_cost"))
        if ep is not None:
            group["pivot"].setdefault(seed, ep)
        if elp is not None:
            group["lp"].setdefault(seed, elp)

    output = list(complete.values())
    for (ego, p_delete), group in edge.items():
        seed_ids = set(group["pivot"]) | set(group["lp"])
        output.append({
            "ego_id": ego,
            "n": group["n"],
            "p_delete": p_delete,
            "graph_variant": "edge_deleted",
            "number_of_seeds": len(seed_ids),
            "pivot_best_cost_average": rounded(average(group["pivot"].values())),
            "lp_cost_average": rounded(average(group["lp"].values())),
        })

    output.sort(key=lambda r: (int(float(r["ego_id"])), 0 if r["graph_variant"] == "complete" else 1, float(r["p_delete"])))
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
        "ego_id", "n", "p_delete", "graph_variant", "d_hat", "lambda", "number_of_seeds",
        "minmaxcc_best",
        "min_max_lp_cost_best",
        "minmaxcc_best_to_lp_ratio",

        "minmaxcc_average",
        "min_max_lp_cost_average",
        "minmaxcc_average_to_lp_ratio",

        "minmaxcc_worst",
        "min_max_lp_cost_worst",
        "minmaxcc_worst_to_lp_ratio",
        "best_seed", "worst_seed",
        "min_max_lp_rounding_cost_best_seed", "min_max_lp_runtime_seconds_best_seed",
    ]
    cc_fields = [
        "ego_id", "n", "p_delete", "graph_variant", "number_of_seeds",
        "pivot_best_cost_average", "lp_cost_average",
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
