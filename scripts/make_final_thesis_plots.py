from pathlib import Path
from math import comb
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FLAT = ROOT / "results" / "processed" / "all_runs_flat.csv"

OUT = ROOT / "results" / "processed" / "plots" / "final_thesis_plots"
FIG_OUT = ROOT / "figures" / "thesis_plots"

OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ["random", "clique", "facebook"]
PDELETE_ORDER = [0.05, 0.15, 0.25, 0.40]
USE_ONLY_COMPLETE_PDELETE_UNITS = True


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def infer_family(row):
    name = str(row.get("file_name", "")).lower()
    graph_type = str(row.get("graph_type", "")).lower()

    if "random" in name or graph_type == "random":
        return "random"
    if "clq" in name or "clique" in graph_type:
        return "clique"
    if "fb_" in name or "facebook" in name or "facebook" in graph_type:
        return "facebook"

    return row.get("graph_family", "other")


def parse_clique_sizes(row):
    raw = str(row.get("cluster_sizes", "")).strip()
    file_name = str(row.get("file_name", "")).strip()

    def parse_text(s):
        s = s.replace(".json", "").strip()

        m = re.fullmatch(r"(\d+)x(\d+)", s)
        if m:
            count = int(m.group(1))
            size = int(m.group(2))
            return [size] * count

        nums = [int(x) for x in re.findall(r"\d+", s)]
        return nums

    if raw and raw not in {"None", "nan", "NaN", ""}:
        parsed = parse_text(raw)
        if parsed:
            return parsed

    stem = Path(file_name).stem
    m = re.match(r"clq_n\d+_(.+)", stem)
    if m:
        return parse_text(m.group(1))

    return []


def load_data():
    if not FLAT.exists():
        raise SystemExit("Missing results/processed/all_runs_flat.csv. Run scripts/make_all_runs_flat.py first.")

    df = pd.read_csv(FLAT)
    df["graph_family"] = df.apply(infer_family, axis=1)

    numeric_cols = [
        "n", "seed", "p_delete", "p_positive",
        "complete_ilp_cost", "complete_lp_cost",
        "edge_ilp_with4_cost", "edge_ilp_without4_cost",
        "edge_lp_with4_cost",
        "complete_lp_ratio", "edge_lp_ratio_with4",
        "complete_best_pivot_approx", "edge_best_pivot_approx_with4",
        "complete_bad_triangles_total", "edge_bad_triangles_total",
        "edge_bad_4_cycles_count", "edge_num_edges_deleted",
        "runtime_seconds",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    if "same_clustering_4_cycle" not in df.columns:
        df["same_clustering_4_cycle"] = ""

    df = df[df["graph_family"].isin(FAMILY_ORDER)].copy()
    df["p_delete_num"] = df["p_delete"]

    df["unit_id"] = (
        df["graph_family"].astype(str)
        + "|"
        + df["file_name"].astype(str)
        + "|seed="
        + df["seed"].astype(str)
    )

    if USE_ONLY_COMPLETE_PDELETE_UNITS:
        complete_units = []
        for unit, g in df.groupby("unit_id"):
            pset = set(round(x, 2) for x in g["p_delete_num"].dropna().unique())
            if set(PDELETE_ORDER).issubset(pset):
                complete_units.append(unit)
        before = len(df)
        df = df[df["unit_id"].isin(complete_units)].copy()
        print(f"Using complete p_delete units only: {len(complete_units)} units")
        print(f"Rows kept: {len(df)}/{before}")

    df["complete_edges"] = df["n"].apply(lambda x: comb(int(x), 2) if pd.notna(x) and int(x) >= 2 else np.nan)
    df["remaining_edges"] = df["complete_edges"] - df["edge_num_edges_deleted"]
    df["possible_triangles"] = df["n"].apply(lambda x: comb(int(x), 3) if pd.notna(x) and int(x) >= 3 else np.nan)

    df["complete_cost_per_edge"] = df["complete_ilp_cost"] / df["complete_edges"]
    df["edge_cost_per_remaining_edge"] = df["edge_ilp_with4_cost"] / df["remaining_edges"]
    df["normalized_new_over_complete_cost"] = df["edge_cost_per_remaining_edge"] / df["complete_cost_per_edge"]

    df["complete_bad_triangle_density"] = df["complete_bad_triangles_total"] / df["possible_triangles"]
    df["edge_bad_triangle_density"] = df["edge_bad_triangles_total"] / df["possible_triangles"]
    df["bad_triangle_removed_fraction"] = (
        (df["complete_bad_triangles_total"] - df["edge_bad_triangles_total"])
        / df["complete_bad_triangles_total"]
    )

    df["bad4_per_1000_remaining_edges"] = 1000 * df["edge_bad_4_cycles_count"] / df["remaining_edges"]

    df["complete_lp_gap"] = 1 - df["complete_lp_ratio"]
    df["edge_lp_gap"] = 1 - df["edge_lp_ratio_with4"]

    df["complete_pivot_excess"] = df["complete_best_pivot_approx"] - 1
    df["edge_pivot_excess"] = df["edge_best_pivot_approx_with4"] - 1

    same = df["same_clustering_4_cycle"].astype(str).str.lower().str.strip()
    df["fourcycle_clustering_known"] = same.isin(["true", "false"])
    df["fourcycle_clustering_changed"] = same.eq("false")

    sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_sizes_label"] = sizes.apply(lambda xs: "-".join(str(x) for x in xs) if xs else "")
    df["num_cliques"] = sizes.apply(lambda xs: len(xs) if xs else np.nan)
    df["largest_clique_size"] = sizes.apply(lambda xs: max(xs) if xs else np.nan)
    df["smallest_clique_size"] = sizes.apply(lambda xs: min(xs) if xs else np.nan)
    df["clique_imbalance_ratio"] = df["largest_clique_size"] / df["smallest_clique_size"]

    return df


def summarize(g):
    known = g[g["fourcycle_clustering_known"]]

    return pd.Series({
        "runs": len(g),
        "avg_n": g["n"].mean() if "n" in g.columns else np.nan,
        "avg_complete_ilp_cost": g["complete_ilp_cost"].mean(),
        "avg_edge_ilp_cost": g["edge_ilp_with4_cost"].mean(),
        "avg_complete_cost_per_edge": g["complete_cost_per_edge"].mean(),
        "avg_edge_cost_per_remaining_edge": g["edge_cost_per_remaining_edge"].mean(),
        "avg_normalized_new_over_complete_cost": g["normalized_new_over_complete_cost"].mean(),

        "avg_complete_bad_triangle_density": g["complete_bad_triangle_density"].mean(),
        "avg_edge_bad_triangle_density": g["edge_bad_triangle_density"].mean(),
        "avg_bad_triangle_removed_fraction": g["bad_triangle_removed_fraction"].mean(),

        "avg_bad_4_cycles": g["edge_bad_4_cycles_count"].mean(),
        "avg_bad4_per_1000_remaining_edges": g["bad4_per_1000_remaining_edges"].mean(),
        "fourcycle_clustering_changed_fraction": known["fourcycle_clustering_changed"].mean() if len(known) else np.nan,

        "avg_complete_pivot_ratio": g["complete_best_pivot_approx"].mean(),
        "avg_edge_pivot_ratio": g["edge_best_pivot_approx_with4"].mean(),

        "avg_complete_lp_ratio": g["complete_lp_ratio"].mean(),
        "avg_edge_lp_ratio": g["edge_lp_ratio_with4"].mean(),
        "avg_complete_lp_gap": g["complete_lp_gap"].mean(),
        "avg_edge_lp_gap": g["edge_lp_gap"].mean(),

        "median_runtime_seconds": g["runtime_seconds"].median(),
    })


def family_pdelete_table(df):
    return (
        df.groupby(["graph_family", "p_delete_num"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["graph_family", "p_delete_num"])
    )


def by_size_table(df):
    return (
        df.groupby(["graph_family", "n", "p_delete_num"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["graph_family", "n", "p_delete_num"])
    )


def save_figure(fig, filename):
    p1 = OUT / filename
    p2 = FIG_OUT / filename
    fig.tight_layout()
    fig.savefig(p1, dpi=250)
    fig.savefig(p2, dpi=250)
    plt.close(fig)
    print("Saved:", p1)
    print("Saved:", p2)


def plot_family_lines(table, y_col, title, ylabel, filename, y_ref=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for fam in FAMILY_ORDER:
        sub = table[table["graph_family"] == fam].sort_values("p_delete_num")
        if len(sub):
            ax.plot(sub["p_delete_num"], sub[y_col], marker="o", label=fam)

    if y_ref is not None:
        ax.axhline(y_ref, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PDELETE_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(fig, filename)


def plot_complete_vs_new(table, complete_col, new_col, title, ylabel, filename, y_ref=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fam in FAMILY_ORDER:
        sub = table[table["graph_family"] == fam].sort_values("p_delete_num")
        if len(sub):
            ax.plot(sub["p_delete_num"], sub[complete_col], linestyle="--", marker="o", label=f"{fam} complete")
            ax.plot(sub["p_delete_num"], sub[new_col], marker="o", label=f"{fam} new")

    if y_ref is not None:
        ax.axhline(y_ref, linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PDELETE_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    save_figure(fig, filename)


def plot_size_lines(table, y_col, title, ylabel, filename, y_ref=None):
    grouped = (
        table.groupby(["graph_family", "n"], dropna=False)
        .agg(value=(y_col, "mean"))
        .reset_index()
        .sort_values(["graph_family", "n"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for fam in FAMILY_ORDER:
        sub = grouped[grouped["graph_family"] == fam]
        if len(sub):
            ax.plot(sub["n"], sub["value"], marker="o", label=fam)

    if y_ref is not None:
        ax.axhline(y_ref, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(fig, filename)


def main():
    df = load_data()
    fam = family_pdelete_table(df)
    size = by_size_table(df)

    # 6 final thesis figures only
    plot_family_lines(
        fam,
        "avg_normalized_new_over_complete_cost",
        "RQ1: Normalized cost after edge deletion",
        "Normalized new/complete cost",
        "01_rq1_normalized_cost.png",
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
        "RQ1/RQ2: Bad triangle density, complete vs edge-deleted",
        "Bad triangle density",
        "03_bad_triangle_density_complete_vs_new.png",
    )

    plot_complete_vs_new(
        fam,
        "avg_complete_pivot_ratio",
        "avg_edge_pivot_ratio",
        "RQ2: Pivot approximation, complete vs edge-deleted",
        "Pivot/ILP",
        "04_rq2_pivot_complete_vs_new.png",
        y_ref=1,
    )

    plot_size_lines(
        size,
        "median_runtime_seconds",
        "RQ2: ILP runtime by graph size",
        "Median runtime seconds",
        "05_rq2_runtime_by_size.png",
    )

    plot_complete_vs_new(
        fam,
        "avg_complete_lp_gap",
        "avg_edge_lp_gap",
        "RQ3: LP gap, complete vs edge-deleted",
        "LP gap = 1 - LP/ILP",
        "06_rq3_lp_gap_complete_vs_new.png",
        y_ref=0,
    )

    print("")
    print("Done. Final plots are in:")
    print("-", OUT)
    print("-", FIG_OUT)


if __name__ == "__main__":
    main()
