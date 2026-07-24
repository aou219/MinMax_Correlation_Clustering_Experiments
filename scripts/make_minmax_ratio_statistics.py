#!/usr/bin/env python3
"""Compute mean, standard deviation, and 95% CI of MinMax ratios."""

from pathlib import Path

import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
OUTPUT = ROOT / "results/research_tables/facebook_minmax_ratio_statistics.csv"

EGO_IDS = [3980, 698, 414, 686]
Q_VALUES = [0.05, 0.15, 0.25, 0.40]


def summarize(group: pd.DataFrame, column: str) -> pd.Series:
    values = group[column].dropna()
    count = len(values)
    mean = values.mean()
    std = values.std(ddof=1)

    if count > 1:
        margin = t.ppf(0.975, count - 1) * std / count**0.5
    else:
        margin = float("nan")

    return pd.Series({
        "runs": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    })


def main() -> None:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")

    # Keep the parameter setting used in the paper plots.
    df = df[
        df["ego_id"].isin(EGO_IDS)
        & df["p_delete"].isin(Q_VALUES)
        & (df["edge_min_max_cc_d_hat"] == 8)
        & (df["edge_min_max_cc_lambda"] == 5)
        & (df["edge_min_max_lp_r"] == 0.4)
        & (df["edge_min_max_lp_r2"] == 0.4)
        & (df["edge_min_max_lp_method"] == 2)
    ].copy()

    # Approximation ratios for every individual run.
    df["minmaxcc_to_lp"] = (
        df["edge_min_max_cc_max_disagreement"]
        / df["edge_min_max_lp_cost"]
    )
    df["minmaxcc_to_lp_rounding"] = (
        df["edge_min_max_cc_max_disagreement"]
        / df["edge_min_max_lp_rounding_cost"]
    )

    rows = []
    groups = df.groupby(["ego_id", "n", "p_delete"])

    for keys, group in groups:
        ego_id, n, p_delete = keys

        for ratio_name in [
            "minmaxcc_to_lp",
            "minmaxcc_to_lp_rounding",
        ]:
            stats = summarize(group, ratio_name)
            rows.append({
                "ego_id": ego_id,
                "n": int(n),
                "p_delete": p_delete,
                "ratio": ratio_name,
                **stats.to_dict(),
            })

    result = pd.DataFrame(rows).sort_values(
        ["ratio", "n", "p_delete"]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("Input:", INPUT)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
