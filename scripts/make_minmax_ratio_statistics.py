#!/usr/bin/env python3
"""Compute Pivot-to-LP statistics with one summary row per clique graph size n."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results/research_tables/clique_runs_flat.csv"
)

OUTPUT = (
    ROOT
    / "results/research_tables/clique_pivot_lp_ratio_statistics.csv"
)

Q_VALUES = [0.05, 0.15, 0.25, 0.40]


def summarize(values: pd.Series) -> dict[str, float | int]:
    """Calculate mean, sample SD, and a two-sided 95% t-interval."""

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = (
        values
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

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
            t.ppf(
                0.975,
                df=count - 1,
            )
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

    # clique_runs_flat.csv should contain only clique graphs.
    if "graph_family" in df.columns:
        clique_mask = (
            df["graph_family"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("clique")
        )

        if not clique_mask.all():
            number_of_non_clique_rows = int(
                (~clique_mask).sum()
            )

            raise ValueError(
                "The input contains "
                f"{number_of_non_clique_rows} non-clique rows."
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

    # Remove rows whose essential identifiers are missing.
    df = df.dropna(
        subset=[
            "n",
            "seed",
            "p_delete",
        ]
    ).copy()

    # Keep only the edge-deletion probabilities used in the analysis.
    df = df[
        df["p_delete"].apply(
            lambda value: any(
                np.isclose(value, q)
                for q in Q_VALUES
            )
        )
    ].copy()

    # Identify each separate clique configuration.
    if "file_name" in df.columns:
        df["graph_id"] = (
            df["file_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        df["graph_id"] = ""

    # Use n and cluster_sizes as a fallback when file_name is unavailable.
    if "cluster_sizes" in df.columns:
        cluster_sizes = (
            df["cluster_sizes"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        cluster_sizes = pd.Series(
            "",
            index=df.index,
        )

    fallback_id = (
        "n="
        + df["n"].astype("Int64").astype(str)
        + "|sizes="
        + cluster_sizes
    )

    df["graph_id"] = df["graph_id"].mask(
        df["graph_id"].eq(""),
        fallback_id,
    )

    rows_before_deduplication = len(df)

    # Pivot results can be repeated for different MinMax parameter settings.
    # Keep one Pivot result per graph configuration, seed, and p_delete.
    df = df.drop_duplicates(
        subset=[
            "graph_id",
            "seed",
            "p_delete",
        ]
    ).copy()

    # Avoid undefined ratios when the LP objective is zero.
    lp_cost = df[
        "edge_all_pairs_lp_cost"
    ].replace(0, np.nan)

    # Compute the ratio separately for every graph instance.
    df["pivot_average_to_lp"] = (
        df["edge_pivot_average_cost"]
        / lp_cost
    )

    rows: list[dict[str, object]] = []

    # Produce exactly one summary row per n.
    # This combines all clique configurations, seeds, and selected
    # p_delete values belonging to the same graph size.
    for n, group in df.groupby(
        "n",
        sort=True,
        dropna=False,
    ):
        statistics = summarize(
            group["pivot_average_to_lp"]
        )

        rows.append({
            "n": int(n),
            "ratio": "pivot_average_to_lp",
            **statistics,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = (
            result
            .sort_values("n")
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