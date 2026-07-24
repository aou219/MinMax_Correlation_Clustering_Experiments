#!/usr/bin/env python3
"""
Create two grouped MinMax ratio bar plots:

1. MinMaxCC / MinMaxLP.
2. MinMaxCC / LP-rounding clustering cost.

The script writes two PNG files and two PDF files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results/research_tables/facebook_minmax_table.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/figures/research_figures"
    / "facebook_minmax_average_ratio_grouped_bars"
)

LP_RATIO_COLUMN = "minmaxcc_ratio_average"
ROUNDING_RATIO_COLUMN = (
    "minmaxcc_to_lp_rounding_clustering_ratio_average"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lp-ratio-column",
        default=LP_RATIO_COLUMN,
    )
    parser.add_argument(
        "--rounding-ratio-column",
        default=ROUNDING_RATIO_COLUMN,
    )
    parser.add_argument(
        "--q-values",
        default="0,0.05,0.15,0.25,0.4",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def prepare_data(
    frame: pd.DataFrame,
    ratio_column: str,
    q_values: list[float],
) -> pd.DataFrame:
    data = frame.copy()

    for column in ["n", "p_delete", ratio_column]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=["n", "p_delete", ratio_column]
    )
    data = data[data["p_delete"].isin(q_values)]

    return (
        data.groupby(["n", "p_delete"], as_index=False)[
            ratio_column
        ]
        .mean()
    )


def save_grouped_bar_plot(
    *,
    plot_data: pd.DataFrame,
    ratio_column: str,
    q_values: list[float],
    output_path: Path,
    ylabel: str,
    dpi: int,
    add_equal_cost_line: bool,
) -> None:
    if plot_data.empty:
        print(f"SKIP {output_path.name}: no values for {ratio_column}")
        return

    n_values = sorted(plot_data["n"].astype(int).unique())
    x = np.arange(len(n_values))
    width = 0.15

    colors = {
        0.0: "#4285F4",
        0.05: "#EA4335",
        0.15: "#F9AB00",
        0.25: "#34A853",
        0.4: "#FF6D01",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for index, q in enumerate(q_values):
        values = (
            plot_data[
                np.isclose(plot_data["p_delete"], q)
            ]
            .set_index("n")[ratio_column]
            .reindex(n_values)
        )

        positions = x + (
            index - (len(q_values) - 1) / 2
        ) * width

        ax.bar(
            positions,
            values.to_numpy(),
            width=width,
            color=colors.get(q),
            label=f"q={q:g}",
        )

    if add_equal_cost_line:
        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Equal clustering cost",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in n_values])
    ax.set_xlabel("Number of vertices, n")
    ax.set_ylabel(ylabel)

    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.1,
        columnspacing=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(
        output_path.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    print("PNG:", output_path.with_suffix(".png"))
    print("PDF:", output_path.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    frame = pd.read_csv(input_path, encoding="utf-8-sig")

    required = {
        "n",
        "p_delete",
        args.lp_ratio_column,
        args.rounding_ratio_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    q_values = [
        float(value.strip())
        for value in args.q_values.split(",")
        if value.strip()
    ]

    lp_data = prepare_data(
        frame,
        args.lp_ratio_column,
        q_values,
    )
    rounding_data = prepare_data(
        frame,
        args.rounding_ratio_column,
        q_values,
    )

    rounding_output = output_path.with_name(
        output_path.name + "_lp_rounding"
    )

    print("Input:", input_path)

    save_grouped_bar_plot(
        plot_data=lp_data,
        ratio_column=args.lp_ratio_column,
        q_values=q_values,
        output_path=output_path,
        ylabel=(
            "Average approximation ratio "
            "(MinMaxCC / MinMaxLP)"
        ),
        dpi=args.dpi,
        add_equal_cost_line=False,
    )

    save_grouped_bar_plot(
        plot_data=rounding_data,
        ratio_column=args.rounding_ratio_column,
        q_values=q_values,
        output_path=rounding_output,
        ylabel=(
            "Average clustering-cost ratio "
            "(MinMaxCC / LP rounding)"
        ),
        dpi=args.dpi,
        add_equal_cost_line=False,
    )


if __name__ == "__main__":
    main()
