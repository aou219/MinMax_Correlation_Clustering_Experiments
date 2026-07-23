#!/usr/bin/env python3
"""
Create two publication-style Facebook MinMaxCC/LP approximation-ratio figures.

Each figure contains only ONE average line.

The dark-blue line shows the aggregated average approximation ratio.
The light-green shaded area shows the full aggregated best-to-worst range.

Aggregation
-----------
Figure 1, x = n:
    For every graph size n, merge all p_delete rows:
    - line: mean of minmaxcc_ratio_average
    - lower range: minimum minmaxcc_ratio_best
    - upper range: maximum minmaxcc_ratio_worst

Figure 2, x = p_delete:
    For every p_delete, merge all ego graphs:
    - line: mean of minmaxcc_ratio_average
    - lower range: minimum minmaxcc_ratio_best
    - upper range: maximum minmaxcc_ratio_worst

Input:
    results/research_tables/facebook_minmax_table.csv

Output:
    results/figures/research_figures/
        7_approximation_ratio_range_vs_n.png
        7_approximation_ratio_range_vs_n.pdf
        8_approximation_ratio_range_vs_p_delete.png
        8_approximation_ratio_range_vs_p_delete.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    REPO_ROOT
    / "results/research_tables/facebook_minmax_table.csv"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results/figures/research_figures"
)

BEST_RATIO = "minmaxcc_ratio_best"
AVERAGE_RATIO = "minmaxcc_ratio_average"
WORST_RATIO = "minmaxcc_ratio_worst"

# Professional publication colors.
AVERAGE_LINE_COLOR = "#1F4E79"  # dark blue
RANGE_COLOR = "#B7E4C7"         # light green
RANGE_ALPHA = 0.38
MARKER_EDGE_COLOR = "#163A5C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create two single-line approximation-ratio figures with "
            "light-green best-to-worst ranges."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--include-external-reference-rows",
        action="store_true",
        help=(
            "Include rows whose ratios use an external or complete-graph "
            "reference. By default only same-instance locally solved LP "
            "ratios are plotted."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_data(
    path: Path,
    *,
    include_external_reference_rows: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "ego_id",
        "n",
        "p_delete",
        BEST_RATIO,
        AVERAGE_RATIO,
        WORST_RATIO,
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing)
        )

    numeric_columns = [
        "ego_id",
        "n",
        "p_delete",
        BEST_RATIO,
        AVERAGE_RATIO,
        WORST_RATIO,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "ego_id",
            "n",
            "p_delete",
            BEST_RATIO,
            AVERAGE_RATIO,
            WORST_RATIO,
        ]
    ).copy()

    if df.empty:
        raise ValueError(
            "No rows contain all required approximation-ratio values."
        )

    df["ego_id"] = df["ego_id"].astype(int)
    df["n"] = df["n"].astype(int)

    # Defensive handling in case best and worst appear in reverse order.
    df["ratio_lower"] = df[[BEST_RATIO, WORST_RATIO]].min(axis=1)
    df["ratio_upper"] = df[[BEST_RATIO, WORST_RATIO]].max(axis=1)

    # The MinMax table marks rows based on whether the denominator is a
    # locally solved LP for the same graph instance. External complete-graph
    # references remain in the table but are excluded from figures by default.
    if not include_external_reference_rows:
        if "lp_reference_source" not in df.columns:
            raise ValueError(
                "The input table is missing lp_reference_source, which is "
                "required to distinguish local same-instance LP ratios from "
                "external reference ratios."
            )
        sources = df["lp_reference_source"].fillna("").astype(str)
        df = df.loc[sources.str.startswith("computed_")].copy()

    if df.empty:
        raise ValueError(
            "No figure-eligible rows remain after applying the LP-reference "
            "filter."
        )

    return df


def apply_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
        "lines.markersize": 6.5,
    })


def format_p_delete(value: float) -> str:
    if abs(value) < 1e-12:
        return "0 (complete)"
    return f"{value:g}"


def aggregate_by_x(
    df: pd.DataFrame,
    x_column: str,
) -> pd.DataFrame:
    """
    Merge all rows at each x value into one average and one full range.

    Each source row receives equal weight in the mean.
    """
    return (
        df.groupby(x_column, as_index=False)
        .agg(
            average_ratio=(AVERAGE_RATIO, "mean"),
            lower_ratio=("ratio_lower", "min"),
            upper_ratio=("ratio_upper", "max"),
            number_of_rows=(AVERAGE_RATIO, "size"),
        )
        .sort_values(x_column)
        .reset_index(drop=True)
    )


def finish_axis(
    ax,
    title: str,
    xlabel: str,
) -> None:
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(
        "Approximation ratio (MinMaxCC / MinMaxLP)",
        labelpad=8,
    )
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.margins(x=0.03)


def add_single_line_and_range(
    ax,
    x,
    average,
    lower,
    upper,
) -> None:
    ax.fill_between(
        x,
        lower,
        upper,
        color=RANGE_COLOR,
        alpha=RANGE_ALPHA,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        x,
        average,
        color=AVERAGE_LINE_COLOR,
        marker="o",
        linestyle="-",
        markerfacecolor="white",
        markeredgecolor=MARKER_EDGE_COLOR,
        markeredgewidth=1.4,
        zorder=3,
    )


def add_legend(ax) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=AVERAGE_LINE_COLOR,
            marker="o",
            linestyle="-",
            markerfacecolor="white",
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=1.4,
            label="Mean approximation ratio",
        ),
        Patch(
            facecolor=RANGE_COLOR,
            edgecolor="none",
            alpha=0.65,
            label="Full best–worst range",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="best",
    )


def save_figure(
    fig,
    output_stem: Path,
) -> None:
    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    fig.tight_layout()
    fig.savefig(
        png_path,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def plot_ratio_range_vs_n(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    One merged line across graph size n.

    All p_delete values are merged at each n.
    """
    table = aggregate_by_x(df, "n")

    x = table["n"].to_numpy()
    average = table["average_ratio"].to_numpy()
    lower = table["lower_ratio"].to_numpy()
    upper = table["upper_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    add_single_line_and_range(
        ax,
        x,
        average,
        lower,
        upper,
    )

    ax.set_xticks(x)

    finish_axis(
        ax,
        title=(
            "MinMaxCC approximation ratio by Facebook graph size"
        ),
        xlabel="Number of nodes, n",
    )
    add_legend(ax)

    save_figure(
        fig,
        output_dir
        / "7_approximation_ratio_range_vs_n",
    )


def plot_ratio_range_vs_p_delete(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    One merged line across p_delete.

    All ego graphs are merged at each p_delete.
    """
    table = aggregate_by_x(df, "p_delete")

    x = table["p_delete"].to_numpy()
    average = table["average_ratio"].to_numpy()
    lower = table["lower_ratio"].to_numpy()
    upper = table["upper_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    add_single_line_and_range(
        ax,
        x,
        average,
        lower,
        upper,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_p_delete(float(value)) for value in x]
    )

    finish_axis(
        ax,
        title=(
            "MinMaxCC approximation ratio by edge-deletion probability"
        ),
        xlabel="Edge-deletion probability, p_delete",
    )
    add_legend(ax)

    save_figure(
        fig,
        output_dir
        / "8_approximation_ratio_range_vs_p_delete",
    )


def main() -> None:
    args = parse_args()

    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)

    apply_publication_style()
    data = load_data(
        input_path,
        include_external_reference_rows=(
            args.include_external_reference_rows
        ),
    )

    print("Input:", input_path)
    print("Rows used:", len(data))
    print("Output directory:", output_dir)

    n_table = aggregate_by_x(data, "n")
    p_delete_table = aggregate_by_x(data, "p_delete")

    print("\nAggregated by n:")
    print(n_table.to_string(index=False))

    print("\nAggregated by p_delete:")
    print(p_delete_table.to_string(index=False))

    plot_ratio_range_vs_n(
        data,
        output_dir,
    )
    plot_ratio_range_vs_p_delete(
        data,
        output_dir,
    )

    print("\nDone. Created 2 single-line figures in PNG and PDF format.")


if __name__ == "__main__":
    main()
