#!/usr/bin/env python3
"""Compute Pivot-to-LP statistics for balanced and unbalanced clique graphs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/research_tables/clique_runs_flat.csv"

OUTPUT = (
    ROOT
    / "results/research_tables/clique_pivot_lp_ratio_statistics.csv"
)

Q_VALUES = [0.05, 0.15, 0.25, 0.40]

BALANCE_ORDER = {
    "balanced": 0,
    "unbalanced": 1,
}


def parse_cluster_sizes(
    value: object,
    file_name: object = "",
) -> list[int]:
    """Parse clique sizes from cluster_sizes or the graph filename."""

    text = str(value or "").strip()

    if text and text.lower() not in {"nan", "none"}:
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple)):
                sizes = [int(size) for size in parsed]

                if sizes:
                    return sizes

        except (ValueError, SyntaxError, TypeError):
            pass

        repeated = re.fullmatch(
            r"\[?\s*(\d+)\s*x\s*(\d+)\s*\]?",
            text,
        )

        if repeated:
            number_of_cliques = int(repeated.group(1))
            clique_size = int(repeated.group(2))

            return [clique_size] * number_of_cliques

        sizes = [
            int(number)
            for number in re.findall(r"\d+", text)
        ]

        if sizes:
            return sizes

    stem = Path(str(file_name or "")).stem

    match = re.match(r"clq_n\d+_(.+)", stem)

    if not match:
        return []

    suffix = match.group(1)

    repeated = re.fullmatch(r"(\d+)x(\d+)", suffix)

    if repeated:
        number_of_cliques = int(repeated.group(1))
        clique_size = int(repeated.group(2))

        return [clique_size] * number_of_cliques

    return [
        int(number)
        for number in re.findall(r"\d+", suffix)
    ]


def clique_balance_label(sizes: list[int]) -> str:
    """
    Classify a clique decomposition.

    Balanced means that the largest and smallest clique sizes differ
    by at most one vertex.
    """

    if not sizes:
        return "unknown"

    if max(sizes) - min(sizes) <= 1:
        return "balanced"

    return "unbalanced"


def summarize(values: pd.Series) -> dict[str, float | int]:
    """Calculate mean, sample SD, and a two-sided 95% t-interval."""

    values = pd.to_numeric(values, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    count = len(values)

    if count == 0:
        return {
            "runs": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(values.mean())

    if count > 1:
        std = float(values.std(ddof=1))
        standard_error = std / np.sqrt(count)
        margin = float(
            t.ppf(0.975, df=count - 1)
            * standard_error
        )
    else:
        std = np.nan
        margin = np.nan

    return {
        "runs": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    df = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
    )

    required_columns = [
        "n",
        "seed",
        "p_delete",
        "cluster_sizes",
        "edge_pivot_average_cost",
        "edge_all_pairs_lp_cost",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # clique_runs_flat.csv should already contain only clique graphs.
    # Check this when graph_family is available.
    if "graph_family" in df.columns:
        non_clique_rows = ~(
            df["graph_family"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("clique")
        )

        if non_clique_rows.any():
            raise ValueError(
                "The input contains "
                f"{int(non_clique_rows.sum())} non-clique rows."
            )

    numeric_columns = [
        "n",
        "seed",
        "p_delete",
        "edge_pivot_average_cost",
        "edge_all_pairs_lp_cost",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Keep only the edge-deletion probabilities used in the analysis.
    df = df[
        df["p_delete"].apply(
            lambda value: (
                pd.notna(value)
                and any(
                    np.isclose(value, q)
                    for q in Q_VALUES
                )
            )
        )
    ].copy()

    file_names = (
        df["file_name"]
        if "file_name" in df.columns
        else pd.Series("", index=df.index)
    )

    df["clique_sizes"] = [
        parse_cluster_sizes(
            cluster_sizes,
            file_name,
        )
        for cluster_sizes, file_name in zip(
            df["cluster_sizes"],
            file_names,
        )
    ]

    df["balance"] = df["clique_sizes"].apply(
        clique_balance_label
    )

    unknown_rows = df[
        df["balance"] == "unknown"
    ]

    if not unknown_rows.empty:
        columns = ["n", "cluster_sizes"]

        if "file_name" in unknown_rows.columns:
            columns.append("file_name")

        examples = unknown_rows[
            columns
        ].head(10)

        raise ValueError(
            "Could not determine balanced/unbalanced for "
            f"{len(unknown_rows)} rows.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # Build a stable identifier for each clique configuration.
    if "file_name" in df.columns:
        df["graph_id"] = (
            df["file_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        df["graph_id"] = ""

    fallback_id = (
        "n="
        + df["n"].astype("Int64").astype(str)
        + "|sizes="
        + df["clique_sizes"].astype(str)
    )

    df["graph_id"] = df["graph_id"].mask(
        df["graph_id"].eq(""),
        fallback_id,
    )

    rows_before_deduplication = len(df)

    # Keep one result per graph configuration, seed, and deletion probability.
    df = df.drop_duplicates(
        subset=[
            "graph_id",
            "seed",
            "p_delete",
        ]
    ).copy()

    # Pivot average cost divided by the LP lower bound.
    lp_cost = df[
        "edge_all_pairs_lp_cost"
    ].replace(0, np.nan)

    df["pivot_average_to_lp"] = (
        df["edge_pivot_average_cost"]
        / lp_cost
    )

    rows: list[dict[str, object]] = []

    # Produce one row for each existing n/balance combination.
    # All seeds and selected p_delete values are combined.
    for (n, balance), group in df.groupby(
        ["n", "balance"],
        sort=True,
        dropna=False,
    ):
        statistics = summarize(
            group["pivot_average_to_lp"]
        )

        rows.append({
            "n": int(n),
            "balance": balance,
            "ratio": "pivot_average_to_lp",
            **statistics,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["balance_order"] = result[
            "balance"
        ].map(BALANCE_ORDER)

        result = (
            result
            .sort_values(
                ["n", "balance_order"]
            )
            .drop(columns="balance_order")
            .reset_index(drop=True)
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"Input: {INPUT}")
    print(f"Output: {OUTPUT}")
    print(
        "Rows before deduplication: "
        f"{rows_before_deduplication}"
    )
    print(
        "Unique edge graph instances: "
        f"{len(df)}"
    )
    print(f"Summary rows: {len(result)}")
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()