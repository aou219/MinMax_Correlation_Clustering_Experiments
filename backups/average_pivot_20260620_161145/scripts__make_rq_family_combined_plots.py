from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FLAT = ROOT / "results" / "processed" / "all_runs_flat.csv"

FIG_OUT = ROOT / "figures" / "rq_family_combined_plots"
FIG_OUT.mkdir(parents=True, exist_ok=True)

PDELETE_ORDER = [0.05, 0.15, 0.25, 0.40]

AVG_PIVOT_RATIO_COL = "edge_average_pivot_approx_with4"
AVG_PIVOT_COST_COL = "edge_pivot_average_cost"
LP_RATIO_COL = "edge_lp_ratio_with4"
LP_COST_COL = "edge_lp_with4_cost"
EDGE_ILP_COL = "edge_ilp_with4_cost"

RANDOM_COLOR_MAP = {
    0.2: "red",
    0.3: "orange",
    0.4: "gold",
    0.5: "green",
    0.6: "blue",
    0.7: "indigo",
    0.8: "violet",
}

CLIQUE_COLOR_MAP = {
    "balanced": "tab:blue",
    "non-balanced": "tab:orange",
}


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def infer_family(row):
    name = str(row.get("file_name", "")).lower()
    graph_type = str(row.get("graph_type", "")).lower()
    family = str(row.get("graph_family", "")).lower()

    if family in {"random", "clique", "facebook"}:
        return family
    if "random" in name or graph_type == "random":
        return "random"
    if "clq" in name or "clique" in graph_type:
        return "clique"
    if "fb_" in name or "facebook" in name or "facebook" in graph_type:
        return "facebook"
    return "other"


def parse_clique_sizes(row):
    raw = str(row.get("cluster_sizes", "")).strip()
    file_name = str(row.get("file_name", "")).strip()

    def parse_text(text):
        text = text.replace(".json", "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return []

        m = re.fullmatch(r"(\d+)x(\d+)", text)
        if m:
            count = int(m.group(1))
            size = int(m.group(2))
            return [size] * count

        return [int(x) for x in re.findall(r"\d+", text)]

    sizes = parse_text(raw)
    if sizes:
        return sizes

    stem = Path(file_name).stem
    m = re.match(r"clq_n\d+_(.+)", stem)
    if m:
        return parse_text(m.group(1))

    return []


def clique_balance_label(sizes):
    if not sizes:
        return "non-balanced"

    sizes = sorted(int(x) for x in sizes)
    # balanced = clique sizes almost equal
    if max(sizes) - min(sizes) <= 1:
        return "balanced"
    return "non-balanced"


def load_data():
    if not FLAT.exists():
        raise SystemExit(
            "Missing results/processed/all_runs_flat.csv. "
            "Run scripts/make_all_runs_flat.py first."
        )

    df = pd.read_csv(FLAT)
    df["graph_family"] = df.apply(infer_family, axis=1)

    numeric_cols = [
        "n",
        "seed",
        "p_delete",
        "p_positive",
        AVG_PIVOT_RATIO_COL,
        AVG_PIVOT_COST_COL,
        LP_RATIO_COL,
        LP_COST_COL,
        EDGE_ILP_COL,
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    if "same_clustering_4_cycle" not in df.columns:
        df["same_clustering_4_cycle"] = ""

    same = df["same_clustering_4_cycle"].astype(str).str.lower().str.strip()
    df["fourcycle_clustering_changed"] = np.where(
        same.eq("false"),
        1.0,
        np.where(same.eq("true"), 0.0, np.nan),
    )

    # Average Pivot ratio for plotting
    df["avg_pivot_ratio_plot"] = df[AVG_PIVOT_RATIO_COL]
    zero_ilp = df[EDGE_ILP_COL].eq(0)
    zero_avg_pivot = df[AVG_PIVOT_COST_COL].eq(0)
    df.loc[
        df["avg_pivot_ratio_plot"].isna() & zero_ilp & zero_avg_pivot,
        "avg_pivot_ratio_plot"
    ] = 1.0

    # LP ratio for plotting
    df["lp_ratio_plot"] = df[LP_RATIO_COL]
    zero_lp = df[LP_COST_COL].eq(0)
    df.loc[
        df["lp_ratio_plot"].isna() & zero_ilp & zero_lp,
        "lp_ratio_plot"
    ] = 1.0

    sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_balance"] = sizes.apply(clique_balance_label)

    return df


def aggregate_random(df):
    random_df = df[df["graph_family"].eq("random")].copy()

    return (
        random_df
        .groupby(["p_delete", "p_positive", "n"], dropna=False)
        .agg(
            avg_pivot_approx=("avg_pivot_ratio_plot", "mean"),
            lp_ratio=("lp_ratio_plot", "mean"),
            clustering_changed_fraction=("fourcycle_clustering_changed", "mean"),
        )
        .reset_index()
    )


def aggregate_clique(df):
    clique_df = df[df["graph_family"].eq("clique")].copy()

    return (
        clique_df
        .groupby(["p_delete", "clique_balance", "n"], dropna=False)
        .agg(
            avg_pivot_approx=("avg_pivot_ratio_plot", "mean"),
            lp_ratio=("lp_ratio_plot", "mean"),
            clustering_changed_fraction=("fourcycle_clustering_changed", "mean"),
        )
        .reset_index()
    )


def plot_combined_pdelete(
    table,
    family,
    line_col,
    y_col,
    title,
    ylabel,
    filename,
    y_ref=None,
    color_map=None,
    line_order=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    legend_handles = {}
    legend_labels = {}

    for ax, pdel in zip(axes, PDELETE_ORDER):
        sub = table[np.isclose(table["p_delete"], pdel, equal_nan=False)].copy()

        if sub.empty:
            ax.text(0.5, 0.5, f"No data for p_delete={pdel}", ha="center", va="center")
            ax.set_title(f"p_delete={pdel:.2f}")
            ax.set_xlabel("n nodes")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            continue

        if line_order is not None:
            line_values = [lv for lv in line_order if lv in set(sub[line_col].dropna().tolist())]
        else:
            if line_col == "p_positive":
                line_values = sorted(sub[line_col].dropna().unique())
            else:
                line_values = sorted(sub[line_col].dropna().astype(str).unique())

        for line_value in line_values:
            if line_col == "p_positive":
                line_sub = sub[np.isclose(sub[line_col], line_value, equal_nan=False)].sort_values("n")
                label = f"p_pos={line_value:.1f}"
                color = color_map.get(round(float(line_value), 1), None) if color_map else None
            else:
                line_sub = sub[sub[line_col].astype(str).eq(str(line_value))].sort_values("n")
                label = str(line_value)
                color = color_map.get(str(line_value), None) if color_map else None

            if line_sub[y_col].notna().sum() == 0:
                continue

            handle, = ax.plot(
                line_sub["n"],
                line_sub[y_col],
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )

            legend_handles[label] = handle
            legend_labels[label] = label

        if y_ref is not None:
            ax.axhline(y_ref, linestyle="--", linewidth=1, color="gray")

        ax.set_title(f"p_delete={pdel:.2f}")
        ax.set_xlabel("n nodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{family}: {title}", fontsize=15)

    labels = list(legend_labels.values())
    handles = [legend_handles[label] for label in labels]

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=9,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    out_path = FIG_OUT / filename
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_random_plots(random_table):
    plot_combined_pdelete(
        table=random_table,
        family="random",
        line_col="p_positive",
        y_col="avg_pivot_approx",
        title="Average Pivot approximation by n",
        ylabel="Average Pivot / ILP cost",
        filename="random_01_average_pivot_all_pdelete.png",
        y_ref=1,
        color_map=RANDOM_COLOR_MAP,
    )

    plot_combined_pdelete(
        table=random_table,
        family="random",
        line_col="p_positive",
        y_col="lp_ratio",
        title="LP relaxation ratio by n",
        ylabel="LP / ILP ratio",
        filename="random_02_lp_ratio_all_pdelete.png",
        y_ref=1,
        color_map=RANDOM_COLOR_MAP,
    )

    plot_combined_pdelete(
        table=random_table,
        family="random",
        line_col="p_positive",
        y_col="clustering_changed_fraction",
        title="4-cycle constraints changed clustering by n",
        ylabel="Fraction changed",
        filename="random_03_fourcycle_clustering_change_all_pdelete.png",
        y_ref=0,
        color_map=RANDOM_COLOR_MAP,
    )


def make_clique_plots(clique_table):
    plot_combined_pdelete(
        table=clique_table,
        family="clique",
        line_col="clique_balance",
        y_col="avg_pivot_approx",
        title="Average Pivot approximation by n",
        ylabel="Average Pivot / ILP cost",
        filename="clique_01_average_pivot_all_pdelete.png",
        y_ref=1,
        color_map=CLIQUE_COLOR_MAP,
        line_order=["balanced", "non-balanced"],
    )

    plot_combined_pdelete(
        table=clique_table,
        family="clique",
        line_col="clique_balance",
        y_col="lp_ratio",
        title="LP relaxation ratio by n",
        ylabel="LP / ILP ratio",
        filename="clique_02_lp_ratio_all_pdelete.png",
        y_ref=1,
        color_map=CLIQUE_COLOR_MAP,
        line_order=["balanced", "non-balanced"],
    )

    plot_combined_pdelete(
        table=clique_table,
        family="clique",
        line_col="clique_balance",
        y_col="clustering_changed_fraction",
        title="4-cycle constraints changed clustering by n",
        ylabel="Fraction changed",
        filename="clique_03_fourcycle_clustering_change_all_pdelete.png",
        y_ref=0,
        color_map=CLIQUE_COLOR_MAP,
        line_order=["balanced", "non-balanced"],
    )


def main():
    df = load_data()

    random_table = aggregate_random(df)
    clique_table = aggregate_clique(df)

    make_random_plots(random_table)
    make_clique_plots(clique_table)

    print("")
    print("Done.")
    print(f"All plots are saved in: {FIG_OUT}")
    print("")
    print("Created 6 PNG files:")
    print("- random_01_average_pivot_all_pdelete.png")
    print("- random_02_lp_ratio_all_pdelete.png")
    print("- random_03_fourcycle_clustering_change_all_pdelete.png")
    print("- clique_01_average_pivot_all_pdelete.png")
    print("- clique_02_lp_ratio_all_pdelete.png")
    print("- clique_03_fourcycle_clustering_change_all_pdelete.png")


if __name__ == "__main__":
    main()
