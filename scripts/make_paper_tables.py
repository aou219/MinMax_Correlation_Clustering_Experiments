#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
DEFAULT_MINMAX_OUTPUT = REPO_ROOT / "results/research_tables/facebook_minmax_table.csv"
DEFAULT_CC_OUTPUT = REPO_ROOT / "results/research_tables/facebook_correlation_clustering_table.csv"
DEFAULT_CLIQUE_INPUT = REPO_ROOT / "results/research_tables/archive/all_runs_flat.csv"
DEFAULT_CLIQUE_CC_OUTPUT = REPO_ROOT / "results/research_tables/clique_correlation_clustering_table.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--minmax-output", type=Path, default=DEFAULT_MINMAX_OUTPUT)
    p.add_argument("--cc-output", type=Path, default=DEFAULT_CC_OUTPUT)
    p.add_argument("--clique-input", type=Path, default=DEFAULT_CLIQUE_INPUT)
    p.add_argument("--clique-cc-output", type=Path, default=DEFAULT_CLIQUE_CC_OUTPUT)
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


def first_numeric(
    row: dict[str, str],
    *columns: str,
) -> float | None:
    """Return the first available numeric value from candidate columns."""
    for column in columns:
        value = to_float(row.get(column))
        if value is not None:
            return value
    return None


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
    Build the Facebook ordinary correlation-clustering table.

    Complete graph rows:
      - include every ego graph that has complete Pivot results;
      - always report Pivot best and average cost;
      - compute approximation ratios only when a complete LP result exists;
      - otherwise leave both approximation columns empty.

    Edge-deleted rows:
      - include deletion seed 1 when a paired ordinary Pivot and all-pairs LP
        result is available;
      - report Pivot costs and approximation ratios for that same graph.

    Runtime columns:
      - are filled when explicit ordinary Pivot/LP runtime columns exist;
      - otherwise they are left empty.
    """

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
        elif abs(float(old) - value) > 1e-8:
            raise ValueError(
                f"Conflicting {key} values for {context}: "
                f"{old} versus {value}"
            )

    complete_rows: dict[str, dict[str, Any]] = {}
    seed_one_rows: dict[tuple[str, str], dict[str, Any]] = {}

    complete_pivot_runtime: dict[str, list[float]] = defaultdict(list)
    complete_lp_runtime: dict[str, list[float]] = defaultdict(list)
    edge_pivot_runtime: dict[tuple[str, str], list[float]] = defaultdict(list)
    edge_lp_runtime: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in rows:
        ego = str(row.get("ego_id", "")).strip()
        if not ego:
            continue

        p_delete = str(row.get("p_delete", "")).strip()
        n = str(row.get("n", "")).strip()

        complete = complete_rows.setdefault(
            ego,
            {
                "ego_id": ego,
                "n": n,
                "p_delete": 0,
                "graph_variant": "complete",
                "pivot_best": None,
                "pivot_average": None,
                "lp": None,
            },
        )

        set_consistent(
            complete,
            "pivot_best",
            to_float(row.get("complete_pivot_best_cost")),
            f"complete ego={ego}",
        )
        set_consistent(
            complete,
            "pivot_average",
            to_float(row.get("complete_pivot_average_cost")),
            f"complete ego={ego}",
        )
        set_consistent(
            complete,
            "lp",
            to_float(row.get("complete_lp_cost")),
            f"complete ego={ego}",
        )

        complete_pivot_time = first_numeric(
            row,
            "complete_pivot_runtime_seconds_average",
            "complete_pivot_average_runtime_seconds",
            "complete_pivot_runtime_seconds",
        )
        if complete_pivot_time is not None:
            complete_pivot_runtime[ego].append(complete_pivot_time)

        complete_lp_time = first_numeric(
            row,
            "complete_all_pairs_lp_runtime_seconds",
            "complete_lp_runtime_seconds",
        )
        if complete_lp_time is not None:
            complete_lp_runtime[ego].append(complete_lp_time)

        if p_delete:
            runtime_key = (ego, p_delete)

            edge_pivot_time = first_numeric(
                row,
                "edge_pivot_runtime_seconds_average",
                "edge_pivot_average_runtime_seconds",
                "edge_pivot_runtime_seconds",
            )
            if edge_pivot_time is not None:
                edge_pivot_runtime[runtime_key].append(edge_pivot_time)

            edge_lp_time = first_numeric(
                row,
                "edge_all_pairs_lp_runtime_seconds",
                "edge_lp_runtime_seconds",
            )
            if edge_lp_time is not None:
                edge_lp_runtime[runtime_key].append(edge_lp_time)

        # Edge-deleted paper rows remain paired at deletion seed 1.
        if numeric_seed(row) != 1 or not p_delete:
            continue

        edge_best = to_float(row.get("edge_pivot_best_cost"))
        edge_average = to_float(row.get("edge_pivot_average_cost"))
        edge_lp = to_float(row.get("edge_all_pairs_lp_cost"))

        # Large ego graphs without an ordinary LP do not get edge rows.
        if edge_lp is None:
            continue

        if edge_best is None or edge_average is None:
            raise ValueError(
                "Missing paired edge Pivot result for "
                f"ego={ego}, p_delete={p_delete}, seed=1."
            )

        key = (ego, p_delete)
        edge = seed_one_rows.setdefault(
            key,
            {
                "ego_id": ego,
                "n": n,
                "p_delete": p_delete,
                "graph_variant": "edge_deleted",
                "pivot_best": None,
                "pivot_average": None,
                "lp": None,
            },
        )

        set_consistent(
            edge,
            "pivot_best",
            edge_best,
            f"edge ego={ego}, p_delete={p_delete}, seed=1",
        )
        set_consistent(
            edge,
            "pivot_average",
            edge_average,
            f"edge ego={ego}, p_delete={p_delete}, seed=1",
        )
        set_consistent(
            edge,
            "lp",
            edge_lp,
            f"edge ego={ego}, p_delete={p_delete}, seed=1",
        )

    output: list[dict[str, Any]] = []

    def append_row(
        source_row: dict[str, Any],
        context: str,
        pivot_times: Iterable[Any],
        lp_times: Iterable[Any],
        require_lp: bool,
    ) -> None:
        best = source_row["pivot_best"]
        average_cost = source_row["pivot_average"]
        lp = source_row["lp"]

        if best is None:
            raise ValueError(f"Missing best Pivot result for {context}.")
        if average_cost is None:
            raise ValueError(f"Missing average Pivot result for {context}.")
        if require_lp and lp is None:
            raise ValueError(f"Missing all-pairs LP result for {context}.")

        row = {
            key: value
            for key, value in source_row.items()
            if key not in {"pivot_best", "pivot_average", "lp"}
        }

        row["pivot_best_cost"] = rounded(best)
        row["pivot_average_cost"] = rounded(average_cost)

        # For the large complete ego graphs, LP is unavailable, so these
        # fields intentionally remain empty.
        row["averagepivot_approximation"] = rounded(
            ratio(average_cost, lp)
        )
        row["bestpivot_approximation"] = rounded(
            ratio(best, lp)
        )

        row["pivot_runtime_seconds_average"] = rounded(
            average(pivot_times)
        )
        row["lp_runtime_seconds_average"] = rounded(
            average(lp_times)
        )
        output.append(row)

    for ego, row in complete_rows.items():
        # Ignore ego graphs for which neither complete Pivot value exists.
        if (
            row["pivot_best"] is None
            and row["pivot_average"] is None
        ):
            continue

        append_row(
            row,
            f"complete ego={ego}",
            complete_pivot_runtime[ego],
            complete_lp_runtime[ego],
            require_lp=False,
        )

    for (ego, p_delete), row in seed_one_rows.items():
        append_row(
            row,
            f"edge ego={ego}, p_delete={p_delete}, seed=1",
            edge_pivot_runtime[(ego, p_delete)],
            edge_lp_runtime[(ego, p_delete)],
            require_lp=True,
        )

    output.sort(
        key=lambda row: (
            int(float(row["n"])),
            int(float(row["ego_id"])),
            0 if row["graph_variant"] == "complete" else 1,
            float(row["p_delete"]),
        )
    )
    return output

def parse_cluster_sizes(value: Any, file_name: str = "") -> list[int]:
    """
    Parse a clique decomposition such as "[5, 5]", "2x10", or a clique
    filename such as "clq_n20_7_7_6.json".
    """
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

        sizes = [int(x) for x in re.findall(r"\d+", text)]
        if sizes:
            return sizes

    stem = Path(str(file_name or "")).stem
    match = re.match(r"clq_n\d+_(.+)", stem)
    if not match:
        return []

    suffix = match.group(1)
    repeated = re.fullmatch(r"(\d+)x(\d+)", suffix)
    if repeated:
        return [int(repeated.group(2))] * int(repeated.group(1))

    return [int(x) for x in re.findall(r"\d+", suffix)]


def clique_balance_label(sizes: list[int]) -> str:
    """
    Balanced clique decompositions have equal or nearly equal clique sizes.
    """
    if not sizes:
        raise ValueError("Cannot determine clique balance without cluster sizes.")
    return "balanced" if max(sizes) - min(sizes) <= 1 else "unbalanced"


def make_clique_cc_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Build the clique ordinary correlation-clustering table.

    Balanced and unbalanced clique instances are merged. The output contains
    one row per combination of n and p_delete.

    Runtime columns are averaged over every available instance in the group.
    The current all_runs_flat.csv already contains
    ``edge_all_pairs_lp_runtime_seconds``. Pivot runtime and complete LP
    runtime are included whenever explicit runtime columns are present.
    """

    def graph_identity(row: dict[str, str]) -> str:
        return (
            str(row.get("file_path", "")).strip()
            or str(row.get("file_name", "")).strip()
            or f'n={row.get("n", "")}|clusters={row.get("cluster_sizes", "")}'
        )

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
        elif abs(float(old) - value) > 1e-8:
            raise ValueError(
                f"Conflicting {key} values for {context}: "
                f"{old} versus {value}"
            )

    clique_rows = [
        row
        for row in rows
        if str(row.get("graph_family", "")).strip().lower() == "clique"
        or str(row.get("graph_type", "")).strip().lower() == "clique"
    ]

    if not clique_rows:
        raise ValueError("No clique rows found in the clique input table.")

    complete_by_graph_seed: dict[
        tuple[str, str, int],
        dict[str, Any],
    ] = {}

    edge_groups: dict[
        tuple[int, str],
        list[dict[str, float | None]],
    ] = defaultdict(list)

    for row in clique_rows:
        n_value = to_float(row.get("n"))
        if n_value is None:
            raise ValueError(
                f"Missing n for clique row: {row.get('file_name', '')}"
            )
        n = int(n_value)

        graph_id = graph_identity(row)
        seed = get_seed(row)

        complete_key = (graph_id, seed, n)
        complete = complete_by_graph_seed.setdefault(
            complete_key,
            {
                "best": None,
                "average": None,
                "lp": None,
                "pivot_runtime_samples": [],
                "lp_runtime_samples": [],
            },
        )

        set_consistent(
            complete,
            "best",
            to_float(row.get("complete_pivot_best_cost")),
            f"complete clique graph={graph_id}, seed={seed}",
        )
        set_consistent(
            complete,
            "average",
            to_float(row.get("complete_pivot_average_cost")),
            f"complete clique graph={graph_id}, seed={seed}",
        )
        set_consistent(
            complete,
            "lp",
            to_float(row.get("complete_lp_cost")),
            f"complete clique graph={graph_id}, seed={seed}",
        )

        complete_pivot_time = first_numeric(
            row,
            "complete_pivot_runtime_seconds_average",
            "complete_pivot_average_runtime_seconds",
            "complete_pivot_runtime_seconds",
        )
        if complete_pivot_time is not None:
            complete["pivot_runtime_samples"].append(
                complete_pivot_time
            )

        complete_lp_time = first_numeric(
            row,
            "complete_all_pairs_lp_runtime_seconds",
            "complete_lp_runtime_seconds",
        )
        if complete_lp_time is not None:
            complete["lp_runtime_samples"].append(
                complete_lp_time
            )

        p_delete = str(row.get("p_delete", "")).strip()
        edge_best = to_float(row.get("edge_pivot_best_cost"))
        edge_average = to_float(row.get("edge_pivot_average_cost"))
        edge_lp = to_float(row.get("edge_all_pairs_lp_cost"))

        if (
            not p_delete
            and edge_best is None
            and edge_average is None
            and edge_lp is None
        ):
            continue

        if not p_delete:
            raise ValueError(
                f"Missing p_delete for edge clique graph={graph_id}."
            )

        if edge_best is None or edge_average is None or edge_lp is None:
            raise ValueError(
                "Incomplete paired edge result for "
                f"clique graph={graph_id}, p_delete={p_delete}, seed={seed}."
            )

        edge_pivot_time = first_numeric(
            row,
            "edge_pivot_runtime_seconds_average",
            "edge_pivot_average_runtime_seconds",
            "edge_pivot_runtime_seconds",
        )
        edge_lp_time = first_numeric(
            row,
            "edge_all_pairs_lp_runtime_seconds",
            "edge_lp_runtime_seconds",
        )

        edge_groups[(n, p_delete)].append(
            {
                "average_ratio": ratio(edge_average, edge_lp),
                "best_ratio": ratio(edge_best, edge_lp),
                "pivot_runtime": edge_pivot_time,
                "lp_runtime": edge_lp_time,
            }
        )

    complete_groups: dict[
        int,
        list[dict[str, float | None]],
    ] = defaultdict(list)

    for (graph_id, seed, n), values in complete_by_graph_seed.items():
        best = values["best"]
        average_cost = values["average"]
        lp = values["lp"]

        if best is None or average_cost is None or lp is None:
            raise ValueError(
                "Incomplete complete Pivot/LP result for "
                f"clique graph={graph_id}, seed={seed}."
            )

        complete_groups[n].append(
            {
                "average_ratio": ratio(average_cost, lp),
                "best_ratio": ratio(best, lp),
                "pivot_runtime": average(
                    values["pivot_runtime_samples"]
                ),
                "lp_runtime": average(
                    values["lp_runtime_samples"]
                ),
            }
        )

    output: list[dict[str, Any]] = []

    def append_group(
        n: int,
        p_delete: str | int,
        observations: list[dict[str, float | None]],
    ) -> None:
        average_ratios = [
            float(obs["average_ratio"])
            for obs in observations
            if obs["average_ratio"] is not None
        ]
        best_ratios = [
            float(obs["best_ratio"])
            for obs in observations
            if obs["best_ratio"] is not None
        ]

        output.append(
            {
                "n": n,
                "p_delete": p_delete,
                "averagepivot_approximation": (
                    rounded(average(average_ratios))
                    if average_ratios
                    else ""
                ),
                "bestpivot_approximation": (
                    rounded(average(best_ratios))
                    if best_ratios
                    else ""
                ),
                "pivot_runtime_seconds_average": rounded(
                    average(
                        obs["pivot_runtime"]
                        for obs in observations
                    )
                ),
                "lp_runtime_seconds_average": rounded(
                    average(
                        obs["lp_runtime"]
                        for obs in observations
                    )
                ),
            }
        )

    for n, observations in complete_groups.items():
        append_group(n, 0, observations)

    for (n, p_delete), observations in edge_groups.items():
        append_group(n, p_delete, observations)

    output.sort(
        key=lambda row: (
            int(float(row["n"])),
            float(row["p_delete"]),
        )
    )
    return output

def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    minmax_output = resolve(args.minmax_output)
    cc_output = resolve(args.cc_output)
    clique_input = resolve(args.clique_input)
    clique_cc_output = resolve(args.clique_cc_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not clique_input.exists():
        raise FileNotFoundError(f"Clique input not found: {clique_input}")

    all_rows = read_rows(input_path)
    selected_rows = fixed_rows(all_rows, args.d_hat, args.lambda_value)
    minmax_rows = make_minmax_table(selected_rows, args.d_hat, args.lambda_value)
    cc_rows = make_cc_table(all_rows)
    clique_rows = read_rows(clique_input)
    clique_cc_rows = make_clique_cc_table(clique_rows)

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
        "pivot_best_cost",
        "pivot_average_cost",
        "averagepivot_approximation",
        "bestpivot_approximation",
        "pivot_runtime_seconds_average",
        "lp_runtime_seconds_average",
    ]
    clique_cc_fields = [
        "n",
        "p_delete",
        "averagepivot_approximation",
        "bestpivot_approximation",
        "pivot_runtime_seconds_average",
        "lp_runtime_seconds_average",
    ]

    write_rows(minmax_output, minmax_rows, minmax_fields)
    write_rows(cc_output, cc_rows, cc_fields)
    write_rows(clique_cc_output, clique_cc_rows, clique_cc_fields)

    print("Input:", input_path)
    print("Input rows:", len(all_rows))
    print(f"Fixed parameters: d_hat={args.d_hat}, lambda={args.lambda_value}")
    print("MinMax table:", minmax_output)
    print("MinMax rows:", len(minmax_rows))
    print("Correlation table:", cc_output)
    print("Correlation rows:", len(cc_rows))
    print("Clique input:", clique_input)
    print("Clique correlation table:", clique_cc_output)
    print("Clique correlation rows:", len(clique_cc_rows))


if __name__ == "__main__":
    main()
