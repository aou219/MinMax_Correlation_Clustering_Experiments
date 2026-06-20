from pathlib import Path
from math import comb
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FLAT = ROOT / "results" / "processed" / "all_runs_flat.csv"

FIG_OUT = ROOT / "figures" / "thesis_plots"
COMBINED_OUT = ROOT / "figures" / "rq_family_combined_plots"

# oude dubbele/losse outputmappen verwijderen
OLD_DIRS = [
    ROOT / "results" / "processed" / "plots" / "final_thesis_plots",
    ROOT / "results" / "processed" / "plots" / "rq_family_line_plots",
    ROOT / "results" / "processed" / "plots" / "rq_family_combined_plots",
    ROOT / "figures" / "rq_family_line_plots",
    COMBINED_OUT,
]

for old_dir in OLD_DIRS:
    if old_dir.exists():
        shutil.rmtree(old_dir)
        print(f"Deleted old folder: {old_dir}")

# oude final thesis png's verwijderen, maar map zelf houden
FIG_OUT.mkdir(parents=True, exist_ok=True)
for old_png in FIG_OUT.glob("*.png"):
    old_png.unlink()
    print(f"Deleted old figure: {old_png}")

COMBINED_OUT.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ["random", "clique", "facebook"]
PDELETE_ORDER = [0.05, 0.15, 0.25, 0.40]

AVG_PIVOT_COMPLETE_RATIO_COL = "complete_average_pivot_approx"
AVG_PIVOT_EDGE_RATIO_COL = "edge_average_pivot_approx_with4"
AVG_PIVOT_COMPLETE_COST_COL = "complete_pivot_average_cost"
AVG_PIVOT_EDGE_COST_COL = "edge_pivot_average_cost"

LP_COMPLETE_RATIO_COL = "complete_lp_ratio"
LP_EDGE_RATIO_COL = "edge_lp_ratio_with4"

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


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


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
        "complete_ilp_cost",
        "edge_ilp_with4_cost",
        "edge_ilp_without4_cost",
        "complete_lp_cost",
        "edge_lp_with4_cost",
        "complete_primal_cost",
        "complete_dual_cost",
        "edge_primal_cost",
        "edge_dual_cost",
        AVG_PIVOT_COMPLETE_RATIO_COL,
        AVG_PIVOT_EDGE_RATIO_COL,
        AVG_PIVOT_COMPLETE_COST_COL,
        AVG_PIVOT_EDGE_COST_COL,
        LP_COMPLETE_RATIO_COL,
        LP_EDGE_RATIO_COL,
        "complete_bad_triangles_total",
        "edge_bad_triangles_total",
        "edge_bad_4_cycles_count",
        "edge_num_edges_deleted",
        "runtime_seconds",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    if "same_clustering_4_cycle" not in df.columns:
        df["same_clustering_4_cycle"] = ""

    df = df[df["graph_family"].isin(FAMILY_ORDER)].copy()

    # ratio-fix bij cost 0: als ILP=0 en methode=0, dan ratio = 1
    df["complete_avg_pivot_ratio_plot"] = df[AVG_PIVOT_COMPLETE_RATIO_COL]
    zero_complete_ilp = df["complete_ilp_cost"].eq(0)
    zero_complete_avg_pivot = df[AVG_PIVOT_COMPLETE_COST_COL].eq(0)
    df.loc[
        df["complete_avg_pivot_ratio_plot"].isna() & zero_complete_ilp & zero_complete_avg_pivot,
        "complete_avg_pivot_ratio_plot"
    ] = 1.0

    df["edge_avg_pivot_ratio_plot"] = df[AVG_PIVOT_EDGE_RATIO_COL]
    zero_edge_ilp = df["edge_ilp_with4_cost"].eq(0)
    zero_edge_avg_pivot = df[AVG_PIVOT_EDGE_COST_COL].eq(0)
    df.loc[
        df["edge_avg_pivot_ratio_plot"].isna() & zero_edge_ilp & zero_edge_avg_pivot,
        "edge_avg_pivot_ratio_plot"
    ] = 1.0

    df["complete_lp_ratio_plot"] = df[LP_COMPLETE_RATIO_COL]
    zero_complete_lp = df["complete_lp_cost"].eq(0)
    df.loc[
        df["complete_lp_ratio_plot"].isna() & zero_complete_ilp & zero_complete_lp,
        "complete_lp_ratio_plot"
    ] = 1.0

    df["edge_lp_ratio_plot"] = df[LP_EDGE_RATIO_COL]
    zero_edge_lp = df["edge_lp_with4_cost"].eq(0)
    df.loc[
        df["edge_lp_ratio_plot"].isna() & zero_edge_ilp & zero_edge_lp,
        "edge_lp_ratio_plot"
    ] = 1.0

    df["unit_id"] = (
        df["graph_family"].astype(str)
        + "|"
        + df["file_name"].astype(str)
        + "|seed="
        + df["seed"].astype(str)
    )

    df["complete_edges"] = df["n"].apply(
        lambda x: comb(int(x), 2) if pd.notna(x) and int(x) >= 2 else np.nan
    )
    df["remaining_edges"] = df["complete_edges"] - df["edge_num_edges_deleted"]
    df["possible_triangles"] = df["n"].apply(
        lambda x: comb(int(x), 3) if pd.notna(x) and int(x) >= 3 else np.nan
    )

    df["complete_cost_per_edge"] = df["complete_ilp_cost"] / df["complete_edges"]
    df["edge_cost_per_remaining_edge"] = df["edge_ilp_with4_cost"] / df["remaining_edges"]
    df["normalized_new_over_complete_cost"] = (
        df["edge_cost_per_remaining_edge"] / df["complete_cost_per_edge"]
    )

    df["complete_bad_triangle_density"] = (
        df["complete_bad_triangles_total"] / df["possible_triangles"]
    )
    df["edge_bad_triangle_density"] = (
        df["edge_bad_triangles_total"] / df["possible_triangles"]
    )

    df["bad4_per_1000_remaining_edges"] = (
        1000 * df["edge_bad_4_cycles_count"] / df["remaining_edges"]
    )

    df["complete_lp_gap"] = 1 - df["complete_lp_ratio_plot"]
    df["edge_lp_gap"] = 1 - df["edge_lp_ratio_plot"]

    same = df["same_clustering_4_cycle"].astype(str).str.lower().str.strip()
    df["fourcycle_clustering_known"] = same.isin(["true", "false"])
    df["fourcycle_clustering_changed"] = np.where(
        same.eq("false"),
        1.0,
        np.where(same.eq("true"), 0.0, np.nan),
    )

    sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_balance"] = sizes.apply(clique_balance_label)

    return df


def summarize(g):
    known = g[g["fourcycle_clustering_known"]]

    return pd.Series({
        "runs": len(g),
        "avg_normalized_new_over_complete_cost": g["normalized_new_over_complete_cost"].mean(),

        "avg_complete_bad_triangle_density": g["complete_bad_triangle_density"].mean(),
        "avg_edge_bad_triangle_density": g["edge_bad_triangle_density"].mean(),

        "avg_bad_4_cycles": g["edge_bad_4_cycles_count"].mean(),
        "avg_bad4_per_1000_remaining_edges": g["bad4_per_1000_remaining_edges"].mean(),

        "fourcycle_clustering_changed_fraction": (
            known["fourcycle_clustering_changed"].mean() if len(known) else np.nan
        ),

        "avg_complete_pivot_ratio": g["complete_avg_pivot_ratio_plot"].mean(),
        "avg_edge_pivot_ratio": g["edge_avg_pivot_ratio_plot"].mean(),

        "avg_complete_lp_ratio": g["complete_lp_ratio_plot"].mean(),
        "avg_edge_lp_ratio": g["edge_lp_ratio_plot"].mean(),
        "avg_complete_lp_gap": g["complete_lp_gap"].mean(),
        "avg_edge_lp_gap": g["edge_lp_gap"].mean(),

        "median_runtime_seconds": g["runtime_seconds"].median(),
    })


def family_pdelete_table(df):
    return (
        df.groupby(["graph_family", "p_delete"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["graph_family", "p_delete"])
    )


def by_size_table(df):
    return (
        df.groupby(["graph_family", "n", "p_delete"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["graph_family", "n", "p_delete"])
    )


def save_final_figure(fig, filename):
    path = FIG_OUT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_family_lines(table, y_col, title, ylabel, filename, y_ref=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fam in FAMILY_ORDER:
        sub = table[table["graph_family"] == fam].sort_values("p_delete")
        if len(sub):
            ax.plot(sub["p_delete"], sub[y_col], marker="o", linewidth=2, label=fam)

    if y_ref is not None:
        ax.axhline(y_ref, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PDELETE_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_final_figure(fig, filename)


def plot_complete_vs_new(table, complete_col, new_col, title, ylabel, filename, y_ref=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fam in FAMILY_ORDER:
        sub = table[table["graph_family"] == fam].sort_values("p_delete")
        if len(sub):
            ax.plot(
                sub["p_delete"],
                sub[complete_col],
                linestyle="--",
                marker="o",
                linewidth=2,
                label=f"{fam} complete",
            )
            ax.plot(
                sub["p_delete"],
                sub[new_col],
                marker="o",
                linewidth=2,
                label=f"{fam} new",
            )

    if y_ref is not None:
        ax.axhline(y_ref, linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PDELETE_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    save_final_figure(fig, filename)


def plot_size_lines(table, y_col, title, ylabel, filename):
    grouped = (
        table.groupby(["graph_family", "n"], dropna=False)
        .agg(value=(y_col, "mean"))
        .reset_index()
        .sort_values(["graph_family", "n"])
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fam in FAMILY_ORDER:
        sub = grouped[grouped["graph_family"] == fam]
        if len(sub):
            ax.plot(sub["n"], sub["value"], marker="o", linewidth=2, label=fam)

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_final_figure(fig, filename)


def make_final_thesis_plots(df):
    fam = family_pdelete_table(df)
    size = by_size_table(df)

    plot_family_lines(
        fam,
        "avg_normalized_new_over_complete_cost",
        "RQ1: Normalized cost after edge deletion",
        "Normalized new/complete cost",
        "01_rq1_normalized_cost_ratio.png",
        y_ref=1,
    )

    plot_family_lines(
        fam,
        "avg_bad4_per_1000_remaining_edges",
        "RQ1: Bad 4-cycle density after edge deletion",
        "Bad 4-cycles per 1000 remaining edges",
        "02_rq1_bad4_density.png",
    )

    plot_complete_vs_new(
        fam,
        "avg_complete_bad_triangle_density",
        "avg_edge_bad_triangle_density",
        "RQ1/RQ2: Bad triangle density, complete vs new",
        "Bad triangle density",
        "03_bad_triangle_density_complete_vs_new.png",
    )

    plot_complete_vs_new(
        fam,
        "avg_complete_pivot_ratio",
        "avg_edge_pivot_ratio",
        "RQ2: Average Pivot approximation, complete vs new",
        "Average Pivot / ILP",
        "04_rq2_average_pivot_complete_vs_new.png",
        y_ref=1,
    )

    plot_size_lines(
        size,
        "median_runtime_seconds",
        "RQ2/RQ3: ILP runtime by graph size",
        "Median runtime seconds",
        "05_rq2_rq3_runtime_by_size.png",
    )

    plot_complete_vs_new(
        fam,
        "avg_complete_lp_gap",
        "avg_edge_lp_gap",
        "RQ3: LP gap, complete vs new",
        "LP gap = 1 - LP/ILP",
        "06_rq3_lp_gap_complete_vs_new.png",
        y_ref=0,
    )


def aggregate_random(df):
    random_df = df[df["graph_family"].eq("random")].copy()

    return (
        random_df
        .groupby(["p_delete", "p_positive", "n"], dropna=False)
        .agg(
            avg_pivot_approx=("edge_avg_pivot_ratio_plot", "mean"),
            lp_ratio=("edge_lp_ratio_plot", "mean"),
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
            avg_pivot_approx=("edge_avg_pivot_ratio_plot", "mean"),
            lp_ratio=("edge_lp_ratio_plot", "mean"),
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
            present = set(sub[line_col].dropna().astype(str).tolist())
            line_values = [lv for lv in line_order if str(lv) in present]
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

    out_path = COMBINED_OUT / filename
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_combined_plots(df):
    random_table = aggregate_random(df)
    clique_table = aggregate_clique(df)

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

    make_final_thesis_plots(df)
    make_combined_plots(df)

    print("")
    print("Done.")
    print(f"Final thesis plots saved in: {FIG_OUT}")
    print(f"Combined random/clique plots saved in: {COMBINED_OUT}")
    print("")
    print("Pivot figures now use AVERAGE Pivot, not best Pivot.")


if __name__ == "__main__":
    main()
