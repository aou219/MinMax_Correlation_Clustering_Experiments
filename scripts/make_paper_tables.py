#!/usr/bin/env python3
"""
Generate the Facebook MinMax, Facebook ordinary correlation-clustering,
and clique ordinary correlation-clustering paper tables.

This version keeps make_paper_tables_corrected.py as the baseline and changes
only the requested aggregation/output rules:
- no graph_variant or include_in_figures columns;
- MinMaxCC costs and ratios are summarized independently;
- ordinary Facebook Pivot edge rows aggregate the 30 deletion seeds using
  per-seed best-of-100 and mean-of-100 Pivot summaries;
- clique approximation logic remains unchanged;
- ordinary Facebook/clique runtimes are merged directly from the separate
  runtime benchmark, so the generated tables are never silently blank.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
DEFAULT_MINMAX_OUTPUT = REPO_ROOT / "results/research_tables/facebook_minmax_table.csv"
DEFAULT_CC_OUTPUT = REPO_ROOT / "results/research_tables/facebook_correlation_clustering_table.csv"
DEFAULT_CLIQUE_INPUT = REPO_ROOT / "results/research_tables/archive/all_runs_flat.csv"
DEFAULT_CLIQUE_CC_OUTPUT = REPO_ROOT / "results/research_tables/clique_correlation_clustering_table.csv"
DEFAULT_RUNTIME_BENCHMARK = REPO_ROOT / "results/research_tables/normal_cc_runtime_benchmarks.csv"


# Davies, Moseley, and Newman (ICML 2023), Table 1.
# The paper reports a true LP objective only for these five small Facebook
# instances. In our local experiment table, FB 414/686/698/3980 already have
# same-instance LP values; FB 348 uses the published complete-graph LP value.
DAVIES_COMPLETE_MINMAX_LP = {
    "348": 39.13,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--minmax-output", type=Path, default=DEFAULT_MINMAX_OUTPUT)
    p.add_argument("--cc-output", type=Path, default=DEFAULT_CC_OUTPUT)
    p.add_argument("--clique-input", type=Path, default=DEFAULT_CLIQUE_INPUT)
    p.add_argument("--clique-cc-output", type=Path, default=DEFAULT_CLIQUE_CC_OUTPUT)
    p.add_argument(
        "--runtime-benchmark",
        type=Path,
        default=DEFAULT_RUNTIME_BENCHMARK,
        help=(
            "Separate ordinary Pivot/LP runtime benchmark. These values are "
            "merged directly into the Facebook and clique CC paper tables."
        ),
    )
    p.add_argument(
        "--allow-missing-runtimes",
        action="store_true",
        help="Write tables even if an expected ordinary CC runtime is missing.",
    )
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



def probability_key(value: Any) -> str:
    return f"{float(value):.12g}"


def runtime_maps(
    benchmark_rows: Sequence[Mapping[str, str]],
) -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    dict[tuple[int, str, str], dict[str, str]],
]:
    """Index separate ordinary-CC runtime benchmark values."""
    facebook: dict[tuple[str, str, str], dict[str, str]] = {}
    clique: dict[tuple[int, str, str], dict[str, str]] = {}

    for row in benchmark_rows:
        algorithm = str(row.get("algorithm", "")).strip().lower()
        if algorithm not in {"pivot", "lp"}:
            continue

        runtime = str(
            row.get("average_runtime_seconds", "")
        ).strip()
        if not runtime:
            continue

        dataset = str(row.get("dataset", "")).strip().lower()
        variant = str(
            row.get("graph_variant", "")
        ).strip().lower()
        p_delete = probability_key(row.get("p_delete", 0))

        if dataset == "facebook":
            ego_value = to_float(row.get("ego_id"))
            if ego_value is None:
                continue
            key = (
                str(int(ego_value)),
                variant,
                p_delete,
            )
            facebook.setdefault(key, {})[algorithm] = runtime

        elif dataset == "clique":
            n_value = to_float(row.get("n"))
            if n_value is None:
                continue
            key = (
                int(n_value),
                variant,
                p_delete,
            )
            clique.setdefault(key, {})[algorithm] = runtime

    return facebook, clique


def has_ordinary_approximation(
    row: Mapping[str, Any],
) -> bool:
    return any(
        str(row.get(column, "")).strip()
        for column in (
            "averagepivot_approximation",
            "bestpivot_approximation",
        )
    )


def merge_runtime_benchmarks(
    facebook_rows: list[dict[str, Any]],
    clique_rows: list[dict[str, Any]],
    benchmark_rows: Sequence[Mapping[str, str]],
    *,
    allow_missing: bool,
) -> None:
    """Populate every expected ordinary Pivot/LP runtime from the benchmark."""
    facebook_map, clique_map = runtime_maps(benchmark_rows)
    missing: list[str] = []

    for row in facebook_rows:
        p_delete = probability_key(row.get("p_delete", 0))
        variant = (
            "complete"
            if abs(float(p_delete)) < 1e-12
            else "edge_deleted"
        )
        key = (
            str(int(float(row["ego_id"]))),
            variant,
            p_delete,
        )
        values = facebook_map.get(key, {})

        row["pivot_runtime_seconds_average"] = values.get(
            "pivot",
            "",
        )
        row["lp_runtime_seconds_average"] = values.get(
            "lp",
            "",
        )

        if not str(
            row["pivot_runtime_seconds_average"]
        ).strip():
            missing.append(
                "Facebook Pivot runtime: "
                f"ego={key[0]}, p_delete={key[2]}"
            )

        if (
            has_ordinary_approximation(row)
            and not str(
                row["lp_runtime_seconds_average"]
            ).strip()
        ):
            missing.append(
                "Facebook LP runtime: "
                f"ego={key[0]}, p_delete={key[2]}"
            )

    for row in clique_rows:
        p_delete = probability_key(row.get("p_delete", 0))
        variant = (
            "complete"
            if abs(float(p_delete)) < 1e-12
            else "edge_deleted"
        )
        key = (
            int(float(row["n"])),
            variant,
            p_delete,
        )
        values = clique_map.get(key, {})

        row["pivot_runtime_seconds_average"] = values.get(
            "pivot",
            "",
        )
        row["lp_runtime_seconds_average"] = values.get(
            "lp",
            "",
        )

        if not str(
            row["pivot_runtime_seconds_average"]
        ).strip():
            missing.append(
                "Clique Pivot runtime: "
                f"n={key[0]}, p_delete={key[2]}"
            )
        if (
            has_ordinary_approximation(row)
            and not str(
                row["lp_runtime_seconds_average"]
            ).strip()
        ):
            missing.append(
                "Clique LP runtime: "
                f"n={key[0]}, p_delete={key[2]}"
            )

    if missing:
        message = (
            "Expected ordinary CC runtimes are missing from the "
            "runtime benchmark:\n- "
            + "\n- ".join(missing)
        )
        if not allow_missing:
            raise ValueError(message)
        print("WARNING:", message)


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


def make_minmax_table(
    rows: list[dict[str, str]],
    d_hat: int,
    lam: int,
) -> list[dict[str, Any]]:
    """
    Build the Facebook MinMaxCC table.

    Aggregation over deletion seeds
    --------------------------------
    Costs and approximation ratios are summarized independently:

    * minmaxcc_cost_best/average/worst are the minimum, arithmetic mean,
      and maximum MinMaxCC objective values over the deletion seeds.
    * minmaxcc_ratio_best/average/worst are the minimum, arithmetic mean,
      and maximum of the per-seed ratios
      MinMaxCC(seed) / MinMaxLP-reference(seed).
    * MinMaxLP cost columns independently report the minimum, arithmetic
      mean, and maximum LP-reference values.

    Consequently, the seed attaining the best cost does not need to be the
    seed attaining the best ratio.

    LP-reference policy
    -------------------
    * Prefer a locally solved MinMaxLP value for the same graph instance.
    * For FB348, use the published complete-graph LP objective 39.13 from
      Davies et al. as an explicitly marked external reference.
    * Leave LP and ratio fields empty where no true LP objective is available.
    """

    complete_by_ego: dict[str, dict[str, str]] = {}
    edge_groups: dict[
        tuple[str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)
    metadata: dict[tuple[str, str], dict[str, str]] = {}

    def complete_reference(
        ego: str,
        row: dict[str, str],
    ) -> tuple[float | None, str]:
        local_lp = to_float(row.get("complete_min_max_lp_cost"))
        if local_lp is not None:
            return local_lp, "computed_complete_same_instance"

        published_lp = DAVIES_COMPLETE_MINMAX_LP.get(ego)
        if published_lp is not None:
            return published_lp, "davies2023_complete_graph_lp"

        return None, "unavailable"

    def edge_reference(
        ego: str,
        row: dict[str, str],
    ) -> tuple[float | None, str]:
        local_lp = to_float(row.get("edge_min_max_lp_cost"))
        if local_lp is not None:
            return local_lp, "computed_edge_same_instance"

        published_lp = DAVIES_COMPLETE_MINMAX_LP.get(ego)
        if published_lp is not None:
            return published_lp, "davies2023_complete_graph_lp"

        return None, "unavailable"

    def validate_duplicate(
        old: dict[str, Any],
        new: dict[str, Any],
        context: str,
    ) -> None:
        for field in ("cc", "lp", "ratio"):
            old_value = to_float(old.get(field))
            new_value = to_float(new.get(field))
            if old_value is None and new_value is None:
                continue
            if (
                old_value is None
                or new_value is None
                or abs(old_value - new_value) > 1e-8
            ):
                raise ValueError(
                    f"Conflicting duplicate {field} for {context}: "
                    f"{old.get(field)} versus {new.get(field)}"
                )

    for row in rows:
        ego = str(row.get("ego_id", "")).strip()
        if not ego:
            continue

        n = str(row.get("n", "")).strip()
        seed = get_seed(row)
        p_delete = str(row.get("p_delete", "")).strip()

        complete_d_hat = to_float(
            row.get("complete_min_max_cc_d_hat")
        )
        complete_lambda = to_float(
            row.get("complete_min_max_cc_lambda")
        )
        if (
            complete_d_hat is not None
            and complete_lambda is not None
            and int(complete_d_hat) == d_hat
            and int(complete_lambda) == lam
        ):
            complete_by_ego.setdefault(ego, row)

        cc = to_float(
            row.get("edge_min_max_cc_max_disagreement")
        )
        if cc is None or not p_delete:
            continue

        lp, lp_source = edge_reference(ego, row)
        observation = {
            "seed": seed,
            "cc": cc,
            "lp": lp,
            "ratio": ratio(cc, lp),
            "lp_reference_source": lp_source,
            "minmaxcc_runtime": row.get(
                "edge_min_max_cc_runtime_seconds",
                "",
            ),
            "lp_runtime": (
                row.get(
                    "edge_min_max_lp_runtime_seconds",
                    "",
                )
                if lp_source == "computed_edge_same_instance"
                else ""
            ),
            "lp_total_runtime": (
                row.get(
                    "edge_min_max_lp_total_runtime_seconds",
                    "",
                )
                if lp_source == "computed_edge_same_instance"
                else ""
            ),
        }

        key = (ego, p_delete)
        metadata.setdefault(key, {"n": n})
        old = edge_groups[key].get(seed)
        if old is None:
            edge_groups[key][seed] = observation
        else:
            validate_duplicate(
                old,
                observation,
                (
                    f"ego={ego}, p_delete={p_delete}, "
                    f"seed={seed}"
                ),
            )

    output: list[dict[str, Any]] = []

    for ego, row in complete_by_ego.items():
        cc = to_float(
            row.get("complete_min_max_cc_max_disagreement")
        )
        if cc is None:
            continue

        lp, lp_source = complete_reference(ego, row)
        approximation = ratio(cc, lp)

        output.append({
            "ego_id": ego,
            "n": str(row.get("n", "")).strip(),
            "p_delete": 0,
            "d_hat": d_hat,
            "lambda": lam,
            "number_of_seeds": 1,
            "minmaxcc_cost_best": rounded(cc),
            "minmaxcc_cost_average": rounded(cc),
            "minmaxcc_cost_worst": rounded(cc),
            "min_max_lp_cost_minimum": rounded(lp),
            "min_max_lp_cost_average": rounded(lp),
            "min_max_lp_cost_maximum": rounded(lp),
            "minmaxcc_ratio_best": rounded(approximation),
            "minmaxcc_ratio_average": rounded(approximation),
            "minmaxcc_ratio_worst": rounded(approximation),
            "minmaxcc_runtime_seconds_average": rounded(
                to_float(
                    row.get(
                        "complete_min_max_cc_runtime_seconds"
                    )
                )
            ),
            "min_max_lp_runtime_seconds_average": rounded(
                first_numeric(
                    row,
                    "complete_min_max_lp_total_runtime_seconds",
                    "complete_min_max_lp_runtime_seconds",
                )
                if lp_source
                == "computed_complete_same_instance"
                else None
            ),
            "lp_reference_source": lp_source,
        })

    for key, by_seed in edge_groups.items():
        ego, p_delete = key
        observations = list(by_seed.values())

        cc_values = [
            float(observation["cc"])
            for observation in observations
        ]
        lp_values = [
            float(observation["lp"])
            for observation in observations
            if observation["lp"] is not None
        ]
        ratio_values = [
            float(observation["ratio"])
            for observation in observations
            if observation["ratio"] is not None
        ]

        sources = {
            str(observation["lp_reference_source"])
            for observation in observations
        }
        if len(sources) != 1:
            raise ValueError(
                "Inconsistent LP-reference metadata for "
                f"ego={ego}, p_delete={p_delete}: "
                + ", ".join(sorted(sources))
            )

        total_runtime_values = [
            to_float(observation["lp_total_runtime"])
            for observation in observations
        ]
        if (
            total_runtime_values
            and all(
                value is not None
                for value in total_runtime_values
            )
        ):
            minmax_lp_runtime = average(total_runtime_values)
        else:
            minmax_lp_runtime = average(
                observation["lp_runtime"]
                for observation in observations
            )

        output.append({
            "ego_id": ego,
            "n": metadata[key]["n"],
            "p_delete": p_delete,
            "d_hat": d_hat,
            "lambda": lam,
            "number_of_seeds": len(observations),
            "minmaxcc_cost_best": rounded(min(cc_values)),
            "minmaxcc_cost_average": rounded(
                average(cc_values)
            ),
            "minmaxcc_cost_worst": rounded(max(cc_values)),
            "min_max_lp_cost_minimum": (
                rounded(min(lp_values))
                if lp_values
                else ""
            ),
            "min_max_lp_cost_average": (
                rounded(average(lp_values))
                if lp_values
                else ""
            ),
            "min_max_lp_cost_maximum": (
                rounded(max(lp_values))
                if lp_values
                else ""
            ),
            "minmaxcc_ratio_best": (
                rounded(min(ratio_values))
                if ratio_values
                else ""
            ),
            "minmaxcc_ratio_average": (
                rounded(average(ratio_values))
                if ratio_values
                else ""
            ),
            "minmaxcc_ratio_worst": (
                rounded(max(ratio_values))
                if ratio_values
                else ""
            ),
            "minmaxcc_runtime_seconds_average": rounded(
                average(
                    observation["minmaxcc_runtime"]
                    for observation in observations
                )
            ),
            "min_max_lp_runtime_seconds_average": rounded(
                minmax_lp_runtime
            ),
            "lp_reference_source": next(iter(sources)),
        })

    output.sort(
        key=lambda row: (
            int(float(row["n"])),
            int(float(row["ego_id"])),
            float(row["p_delete"]),
        )
    )
    return output


def make_cc_table(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Build the Facebook ordinary correlation-clustering table.

    Complete graph
    --------------
    There is one graph instance per ego ID:
    * Pivot best cost is the best result among the 100 Pivot runs.
    * Pivot average cost is the mean over the 100 Pivot runs.
    * Both approximation values use the complete-graph LP.

    Edge-deleted graphs
    -------------------
    For every deletion seed, the stored Pivot best and average costs summarize
    that seed's 100 Pivot runs. The paper row then aggregates over all
    available deletion seeds:
    * pivot_best_cost is the mean of the per-seed best-of-100 costs;
    * pivot_average_cost is the mean of the per-seed 100-run means;
    * bestpivot_approximation is the mean of
      per-seed best-of-100 cost / same-instance LP;
    * averagepivot_approximation is the mean of
      per-seed 100-run mean cost / same-instance LP.

    Complete Pivot-only ego graphs remain in the table with empty LP,
    approximation, and LP-runtime cells.
    """

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
    edge_groups: dict[
        tuple[str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    complete_pivot_runtime: dict[
        str,
        list[float],
    ] = defaultdict(list)
    complete_lp_runtime: dict[
        str,
        list[float],
    ] = defaultdict(list)
    edge_pivot_runtime: dict[
        tuple[str, str],
        list[float],
    ] = defaultdict(list)
    edge_lp_runtime: dict[
        tuple[str, str],
        list[float],
    ] = defaultdict(list)

    for row in rows:
        ego = str(row.get("ego_id", "")).strip()
        if not ego:
            continue

        n = str(row.get("n", "")).strip()
        p_delete = str(row.get("p_delete", "")).strip()
        seed = get_seed(row)

        complete = complete_rows.setdefault(
            ego,
            {
                "ego_id": ego,
                "n": n,
                "p_delete": 0,
                "pivot_best": None,
                "pivot_average": None,
                "lp": None,
            },
        )

        set_consistent(
            complete,
            "pivot_best",
            to_float(
                row.get("complete_pivot_best_cost")
            ),
            f"complete ego={ego}",
        )
        set_consistent(
            complete,
            "pivot_average",
            to_float(
                row.get("complete_pivot_average_cost")
            ),
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
            complete_pivot_runtime[ego].append(
                complete_pivot_time
            )

        complete_lp_time = first_numeric(
            row,
            "complete_all_pairs_lp_runtime_seconds",
            "complete_lp_runtime_seconds",
        )
        if complete_lp_time is not None:
            complete_lp_runtime[ego].append(
                complete_lp_time
            )

        if not p_delete:
            continue

        edge_best = to_float(
            row.get("edge_pivot_best_cost")
        )
        edge_average = to_float(
            row.get("edge_pivot_average_cost")
        )
        edge_lp = to_float(
            row.get("edge_all_pairs_lp_cost")
        )

        runtime_key = (ego, p_delete)
        edge_pivot_time = first_numeric(
            row,
            "edge_pivot_runtime_seconds_average",
            "edge_pivot_average_runtime_seconds",
            "edge_pivot_runtime_seconds",
        )
        if edge_pivot_time is not None:
            edge_pivot_runtime[runtime_key].append(
                edge_pivot_time
            )

        edge_lp_time = first_numeric(
            row,
            "edge_all_pairs_lp_runtime_seconds",
            "edge_lp_runtime_seconds",
        )
        if edge_lp_time is not None:
            edge_lp_runtime[runtime_key].append(
                edge_lp_time
            )

        # No edge paper row is possible without a same-instance LP.
        if edge_lp is None:
            continue
        if edge_best is None or edge_average is None:
            raise ValueError(
                "Missing paired edge Pivot result for "
                f"ego={ego}, p_delete={p_delete}, "
                f"seed={seed}."
            )

        observation = {
            "seed": seed,
            "n": n,
            "pivot_best": edge_best,
            "pivot_average": edge_average,
            "lp": edge_lp,
            "best_ratio": ratio(edge_best, edge_lp),
            "average_ratio": ratio(
                edge_average,
                edge_lp,
            ),
        }

        old = edge_groups[runtime_key].get(seed)
        if old is None:
            edge_groups[runtime_key][seed] = observation
        else:
            for field in (
                "pivot_best",
                "pivot_average",
                "lp",
            ):
                if (
                    abs(
                        float(old[field])
                        - float(observation[field])
                    )
                    > 1e-8
                ):
                    raise ValueError(
                        f"Conflicting duplicate {field} for "
                        f"ego={ego}, p_delete={p_delete}, "
                        f"seed={seed}."
                    )

    output: list[dict[str, Any]] = []

    for ego, source in complete_rows.items():
        best = source["pivot_best"]
        average_cost = source["pivot_average"]
        lp = source["lp"]

        if best is None and average_cost is None:
            continue
        if best is None or average_cost is None:
            raise ValueError(
                f"Incomplete complete Pivot result for ego={ego}."
            )

        output.append({
            "ego_id": ego,
            "n": source["n"],
            "p_delete": 0,
            "number_of_seeds": 1,
            "pivot_best_cost": rounded(best),
            "pivot_average_cost": rounded(average_cost),
            "averagepivot_approximation": rounded(
                ratio(average_cost, lp)
            ),
            "bestpivot_approximation": rounded(
                ratio(best, lp)
            ),
            "pivot_runtime_seconds_average": rounded(
                average(complete_pivot_runtime[ego])
            ),
            "lp_runtime_seconds_average": rounded(
                average(complete_lp_runtime[ego])
            ),
        })

    for key, by_seed in edge_groups.items():
        ego, p_delete = key
        observations = list(by_seed.values())
        if not observations:
            continue

        best_costs = [
            observation["pivot_best"]
            for observation in observations
        ]
        average_costs = [
            observation["pivot_average"]
            for observation in observations
        ]
        best_ratios = [
            observation["best_ratio"]
            for observation in observations
        ]
        average_ratios = [
            observation["average_ratio"]
            for observation in observations
        ]

        output.append({
            "ego_id": ego,
            "n": observations[0]["n"],
            "p_delete": p_delete,
            "number_of_seeds": len(observations),
            "pivot_best_cost": rounded(
                average(best_costs)
            ),
            "pivot_average_cost": rounded(
                average(average_costs)
            ),
            "averagepivot_approximation": rounded(
                average(average_ratios)
            ),
            "bestpivot_approximation": rounded(
                average(best_ratios)
            ),
            "pivot_runtime_seconds_average": rounded(
                average(edge_pivot_runtime[key])
            ),
            "lp_runtime_seconds_average": rounded(
                average(edge_lp_runtime[key])
            ),
        })

    output.sort(
        key=lambda row: (
            int(float(row["n"])),
            int(float(row["ego_id"])),
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
    runtime_benchmark = resolve(args.runtime_benchmark)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not clique_input.exists():
        raise FileNotFoundError(f"Clique input not found: {clique_input}")
    if not runtime_benchmark.exists():
        raise FileNotFoundError(
            f"Runtime benchmark not found: {runtime_benchmark}"
        )

    all_rows = read_rows(input_path)
    selected_rows = fixed_rows(all_rows, args.d_hat, args.lambda_value)
    minmax_rows = make_minmax_table(selected_rows, args.d_hat, args.lambda_value)
    cc_rows = make_cc_table(all_rows)
    clique_rows = read_rows(clique_input)
    clique_cc_rows = make_clique_cc_table(clique_rows)
    runtime_rows = read_rows(runtime_benchmark)
    merge_runtime_benchmarks(
        cc_rows,
        clique_cc_rows,
        runtime_rows,
        allow_missing=args.allow_missing_runtimes,
    )

    minmax_fields = [
        "ego_id",
        "n",
        "p_delete",
        "d_hat",
        "lambda",
        "number_of_seeds",
        "minmaxcc_cost_best",
        "minmaxcc_cost_average",
        "minmaxcc_cost_worst",
        "min_max_lp_cost_minimum",
        "min_max_lp_cost_average",
        "min_max_lp_cost_maximum",
        "minmaxcc_ratio_best",
        "minmaxcc_ratio_average",
        "minmaxcc_ratio_worst",
        "minmaxcc_runtime_seconds_average",
        "min_max_lp_runtime_seconds_average",
        "lp_reference_source",
    ]
    cc_fields = [
        "ego_id",
        "n",
        "p_delete",
        "number_of_seeds",
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
    print("Runtime benchmark:", runtime_benchmark)
    print("Clique correlation table:", clique_cc_output)
    print("Clique correlation rows:", len(clique_cc_rows))


if __name__ == "__main__":
    main()
