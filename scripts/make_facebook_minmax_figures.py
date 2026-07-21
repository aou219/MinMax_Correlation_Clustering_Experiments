#!/usr/bin/env python3
"""
Create six Facebook MinMaxCC/LP research figures.

Input:
    results/processed/research_tables/facebook_minmax_table.csv

Output:
    results/processed/figures/research_figures/

Figures:
1. n (x) versus average MinMaxCC/LP ratio (y), one line per p_delete
2. n (x) versus worst MinMaxCC/LP ratio (y), one line per p_delete
3. n (x) versus best MinMaxCC/LP ratio (y), one line per p_delete
4. p_delete (x) versus average MinMaxCC/LP ratio (y), one line per ego_id
5. p_delete (x) versus worst MinMaxCC/LP ratio (y), one line per ego_id
6. p_delete (x) versus best MinMaxCC/LP ratio (y), one line per ego_id

Rows without a corresponding LP ratio are skipped automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

RATIO_COLUMNS = {
    "average": "minmaxcc_average_to_lp_ratio",
    "worst": "minmaxcc_worst_to_lp_ratio",
    "best": "minmaxcc_best_to_lp_ratio",
}

RAINBOW_COLORS = [
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#f1c40f",  # yellow
    "#2ca02c",  # green
    "#1f77b4",  # blue
    "#4b0082",  # indigo
    "#8a2be2",  # violet
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create six Facebook MinMaxCC/LP research figures."
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
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "ego_id",
        "n",
        "p_delete",
        *RATIO_COLUMNS.values(),
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
        *RATIO_COLUMNS.values(),
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["ego_id", "n", "p_delete"]).copy()
    df["ego_id"] = df["ego_id"].astype(int)
    df["n"] = df["n"].astype(int)

    return df


def ordered_colors(values) -> dict:
    ordered = list(values)

    if len(ordered) > len(RAINBOW_COLORS):
        raise ValueError(
            f"Need {len(ordered)} colors, but the fixed rainbow palette "
            f"contains only {len(RAINBOW_COLORS)} colors."
        )

    return {
        value: RAINBOW_COLORS[index]
        for index, value in enumerate(ordered)
    }


def format_p_delete(value: float) -> str:
    if abs(value) < 1e-12:
        return "0 (complete)"
    return f"{value:g}"


def finish_and_save(
    fig,
    ax,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)


def plot_ratio_by_n(
    df: pd.DataFrame,
    metric_name: str,
    ratio_column: str,
    output_path: Path,
) -> None:
    """
    x-axis: n
    y-axis: approximation ratio
    one line per p_delete
    """
    plot_df = df.dropna(subset=[ratio_column]).copy()

    p_values = sorted(plot_df["p_delete"].unique())
    color_map = ordered_colors(p_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for p_delete in p_values:
        line = (
            plot_df[plot_df["p_delete"].eq(p_delete)]
            .sort_values(["n", "ego_id"])
        )

        if line.empty:
            continue

        ax.plot(
            line["n"],
            line[ratio_column],
            marker="o",
            linewidth=2,
            color=color_map[p_delete],
            label=f"p_delete={format_p_delete(float(p_delete))}",
        )

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(
            f"No numeric values found in column {ratio_column}."
        )

    ax.set_xticks(sorted(plot_df["n"].unique()))

    finish_and_save(
        fig,
        ax,
        output_path,
        title=(
            f"{metric_name.capitalize()} MinMaxCC/LP approximation ratio "
            "by graph size"
        ),
        xlabel="Number of nodes (n)",
        ylabel=f"{metric_name.capitalize()} MinMaxCC / LP ratio",
    )


def plot_ratio_by_p_delete(
    df: pd.DataFrame,
    metric_name: str,
    ratio_column: str,
    output_path: Path,
) -> None:
    """
    x-axis: p_delete
    y-axis: approximation ratio
    one line per ego_id
    """
    plot_df = df.dropna(subset=[ratio_column]).copy()

    ego_ids = sorted(
        plot_df["ego_id"].unique(),
        key=lambda ego_id: (
            int(
                plot_df.loc[
                    plot_df["ego_id"].eq(ego_id),
                    "n",
                ].iloc[0]
            ),
            int(ego_id),
        ),
    )
    color_map = ordered_colors(ego_ids)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for ego_id in ego_ids:
        line = (
            plot_df[plot_df["ego_id"].eq(ego_id)]
            .sort_values("p_delete")
        )

        if line.empty:
            continue

        n_value = int(line["n"].iloc[0])

        ax.plot(
            line["p_delete"],
            line[ratio_column],
            marker="o",
            linewidth=2,
            color=color_map[ego_id],
            label=f"FB {int(ego_id)} (n={n_value})",
        )

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(
            f"No numeric values found in column {ratio_column}."
        )

    p_ticks = sorted(plot_df["p_delete"].unique())
    ax.set_xticks(p_ticks)
    ax.set_xticklabels(
        [format_p_delete(float(value)) for value in p_ticks]
    )

    finish_and_save(
        fig,
        ax,
        output_path,
        title=(
            f"{metric_name.capitalize()} MinMaxCC/LP approximation ratio "
            "by edge-deletion probability"
        ),
        xlabel="p_delete",
        ylabel=f"{metric_name.capitalize()} MinMaxCC / LP ratio",
    )


def main() -> None:
    args = parse_args()

    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)

    df = load_data(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Input:", input_path)
    print("Output directory:", output_dir)
    print("Rows:", len(df))

    for number, metric_name in enumerate(
        ["average", "worst", "best"],
        start=1,
    ):
        plot_ratio_by_n(
            df,
            metric_name,
            RATIO_COLUMNS[metric_name],
            output_dir
            / f"{number}_{metric_name}_ratio_vs_n_by_p_delete.png",
        )

    for number, metric_name in enumerate(
        ["average", "worst", "best"],
        start=4,
    ):
        plot_ratio_by_p_delete(
            df,
            metric_name,
            RATIO_COLUMNS[metric_name],
            output_dir
            / f"{number}_{metric_name}_ratio_vs_p_delete_by_ego.png",
        )

    print("\nDone. Created 6 figures.")


if __name__ == "__main__":
    main()
