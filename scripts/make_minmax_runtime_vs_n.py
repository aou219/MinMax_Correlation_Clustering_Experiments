#!/usr/bin/env python3
"""Plot MinMaxCC and MinMaxLP average runtime versus graph size."""

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
    / "facebook_minmax_runtime_vs_n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--q-values",
        default="0.05,0.15,0.25,0.4",
    )
    parser.add_argument("--d-hat", type=int, default=8)
    parser.add_argument("--lambda-value", type=int, default=5)
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--r2", type=float, default=0.4)
    parser.add_argument("--method", type=int, default=2)
    parser.add_argument(
        "--lp-ego-ids",
        default="3980,698,414,686",
        help="Ego IDs for which MinMaxLP was run.",
    )
    parser.add_argument(
        "--exclude-runtime-above",
        type=float,
        default=4000.0,
    )
    parser.add_argument("--dpi", type=int, default=300)
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


def average_runtime(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    return (
        frame.dropna(subset=["n", column])
        .groupby("n", as_index=False)[column]
        .mean()
        .sort_values("n")
    )


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    frame = pd.read_csv(input_path, encoding="utf-8-sig")

    required = {
        "ego_id",
        "n",
        "p_delete",
        "edge_min_max_cc_d_hat",
        "edge_min_max_cc_lambda",
        "edge_min_max_cc_runtime_seconds",
        "edge_min_max_lp_runtime_seconds",
        "edge_min_max_lp_r",
        "edge_min_max_lp_r2",
        "edge_min_max_lp_method",
    }

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
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
    lp_ego_ids = {
        int(value.strip())
        for value in args.lp_ego_ids.split(",")
        if value.strip()
    }

    frame = frame[frame["p_delete"].isin(q_values)].copy()

    # MinMaxCC uses every Facebook ego graph available in the CSV.
    cc_rows = frame[
        close(frame["edge_min_max_cc_d_hat"], args.d_hat)
        & close(
            frame["edge_min_max_cc_lambda"],
            args.lambda_value,
        )
    ].copy()

    # MinMaxLP is restricted to the four ego graphs where LP was run.
    lp_rows = frame[
        frame["ego_id"].isin(lp_ego_ids)
        & close(frame["edge_min_max_lp_r"], args.r)
        & close(frame["edge_min_max_lp_r2"], args.r2)
        & close(
            frame["edge_min_max_lp_method"],
            args.method,
        )
    ].copy()

    cc_rows.loc[
        cc_rows["edge_min_max_cc_runtime_seconds"]
        > args.exclude_runtime_above,
        "edge_min_max_cc_runtime_seconds",
    ] = np.nan

    lp_rows.loc[
        lp_rows["edge_min_max_lp_runtime_seconds"]
        > args.exclude_runtime_above,
        "edge_min_max_lp_runtime_seconds",
    ] = np.nan

    cc_data = average_runtime(
        cc_rows,
        "edge_min_max_cc_runtime_seconds",
    )
    lp_data = average_runtime(
        lp_rows,
        "edge_min_max_lp_runtime_seconds",
    )

    if cc_data.empty:
        raise ValueError("No MinMaxCC runtime values were found.")
    if lp_data.empty:
        raise ValueError("No MinMaxLP runtime values were found.")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    ax.plot(
        cc_data["n"],
        cc_data["edge_min_max_cc_runtime_seconds"],
        marker="o",
        linestyle="--",
        linewidth=1.9,
        markersize=5.5,
        color="#4285F4",
        label="MinMaxCC",
    )

    ax.plot(
        lp_data["n"],
        lp_data["edge_min_max_lp_runtime_seconds"],
        marker="o",
        linestyle="--",
        linewidth=1.9,
        markersize=5.5,
        color="#EA4335",
        label="MinMaxLP",
    )

    ax.set_xlabel("Number of vertices, n")
    ax.set_ylabel("Average runtime (seconds)")

    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.5,
        columnspacing=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    print("MinMaxCC graph sizes:", cc_data["n"].tolist())
    print("MinMaxLP graph sizes:", lp_data["n"].tolist())
    print("PNG:", png_path)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
