#!/usr/bin/env python3
"""
Plot average MinMax costs versus graph size.

The plot contains exactly three lines:
1. MinMaxCC average clustering cost.
2. MinMaxLP average objective value.
3. 48*sqrt(n) size reference.

Costs are averaged over the selected edge-deletion probabilities and seeds.
Transparent bands show the full observed minimum-to-maximum range.
The input CSV is only read and is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT
    / "results/research_tables/minmax_facebook_grid_runs_flat.csv"
)

DEFAULT_OUTPUT = (
    ROOT
    / "results/figures/research_figures"
    / "facebook_minmax_average_cost_vs_n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average MinMax costs versus graph size."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--q-values",
        default="0.05,0.15,0.25,0.4",
        help="Comma-separated edge-deletion probabilities.",
    )
    parser.add_argument(
        "--d-hat",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lambda-value",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--method",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ego-ids",
        default="3980,698,414,686",
        help="Comma-separated Facebook ego IDs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def close(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(
        pd.to_numeric(series, errors="coerce"),
        value,
        rtol=0,
        atol=1e-10,
    )


def average_by_n(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    return (
        frame.dropna(subset=["n", column])
        .groupby("n")[column]
        .agg(mean="mean", minimum="min", maximum="max")
        .reset_index()
        .sort_values("n")
    )


def main() -> None:
    args = parse_args()

    input_path = resolve(args.input)
    output_path = resolve(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    required = {
        "ego_id",
        "n",
        "p_delete",
        "edge_min_max_cc_d_hat",
        "edge_min_max_cc_lambda",
        "edge_min_max_cc_max_disagreement",
        "edge_min_max_lp_cost",
        "edge_min_max_lp_r",
        "edge_min_max_lp_r2",
        "edge_min_max_lp_method",
    }

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    q_values = [
        float(value.strip())
        for value in args.q_values.split(",")
        if value.strip()
    ]

    ego_ids = {
        int(value.strip())
        for value in args.ego_ids.split(",")
        if value.strip()
    }

    frame = frame[
        frame["p_delete"].isin(q_values)
        & frame["ego_id"].isin(ego_ids)
    ].copy()

    cc_rows = frame[
        close(
            frame["edge_min_max_cc_d_hat"],
            args.d_hat,
        )
        & close(
            frame["edge_min_max_cc_lambda"],
            args.lambda_value,
        )
    ].copy()

    lp_rows = frame[
        close(
            frame["edge_min_max_lp_r"],
            args.r,
        )
        & close(
            frame["edge_min_max_lp_r2"],
            args.r2,
        )
        & close(
            frame["edge_min_max_lp_method"],
            args.method,
        )
    ].copy()

    cc_data = average_by_n(
        cc_rows,
        "edge_min_max_cc_max_disagreement",
    )
    lp_data = average_by_n(
        lp_rows,
        "edge_min_max_lp_cost",
    )

    if cc_data.empty:
        raise ValueError(
            "No MinMaxCC cost values were found."
        )

    all_n = np.array(
        sorted(
            set(cc_data["n"])
            | set(lp_data["n"])
        ),
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    ax.fill_between(
        cc_data["n"],
        cc_data["minimum"],
        cc_data["maximum"],
        color="#4285F4",
        alpha=0.16,
        linewidth=0,
    )

    if not lp_data.empty:
        ax.fill_between(
            lp_data["n"],
            lp_data["minimum"],
            lp_data["maximum"],
            color="#EA4335",
            alpha=0.16,
            linewidth=0,
        )

    ax.plot(
        cc_data["n"],
        cc_data["mean"],
        marker="o",
        linestyle=":",
        linewidth=1.9,
        markersize=5.5,
        color="#4285F4",
        label="MinMaxCC",
    )

    if not lp_data.empty:
        ax.plot(
            lp_data["n"],
            lp_data["mean"],
            marker="o",
            linestyle=":",
            linewidth=1.9,
            markersize=5.5,
            color="#EA4335",
            label="MinMaxLP",
        )

    ax.plot(
        all_n,
        48 * np.sqrt(all_n),
        marker="o",
        linestyle=":",
        linewidth=1.9,
        markersize=5.5,
        color="#FF6D01",
        label="48√n",
    )

    ax.set_xlabel("Number of vertices, n")
    ax.set_ylabel(
        "Average maximum-disagreement cost"
    )

    ax.set_yscale("log")

    ax.grid(
        axis="y",
        alpha=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.4,
        columnspacing=1.0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Input:", input_path)
    print("PNG:", png_path)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
