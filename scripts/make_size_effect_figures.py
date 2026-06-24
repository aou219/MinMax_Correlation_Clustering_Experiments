from pathlib import Path
import argparse
import re
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "processed" / "all_runs_flat.csv"
DEFAULT_OUT = ROOT / "results" / "processed" / "figures" / "size_effect"

FAMILY_ORDER = ["random", "clique", "facebook"]
P_DELETE_ORDER = [0.05, 0.15, 0.25, 0.40]

P_DELETE_COLORS = {
    0.05: "#1f77b4",  # blue
    0.15: "#2ca02c",  # green
    0.25: "#ff7f0e",  # orange
    0.40: "#d62728",  # red
}

METRICS = {
    "bad_4_cycles": {
        "title": "Size effect in edge-deleted graphs: Bad 4-cycle count",
        "ylabel": "Bad 4-cycles",
        "edge_col": "edge_bad_4_cycles",
        "complete_col": "complete_bad_4_cycles",
    },
    "disjoint_bad_triangle_ratio": {
        "title": "Size effect in edge-deleted graphs: Edge-disjoint bad triangle lower-bound ratio",
        "ylabel": "Max disjoint bad triangles / ILP",
        "edge_col": "edge_disjoint_bad_triangle_ratio",
        "complete_col": "complete_disjoint_bad_triangle_ratio",
    },
    "ilp_ratio": {
        "title": "Size effect in edge-deleted graphs: ILP cost ratio",
        "ylabel": "Edge-deleted ILP / complete ILP",
        "edge_col": "edge_ilp_ratio",
        "complete_col": "complete_ilp_ratio",
    },
    "lp_integrality_ratio": {
        "title": "Size effect in edge-deleted graphs: LP integrality ratio",
        "ylabel": "LP / ILP",
        "edge_col": "edge_lp_integrality_ratio",
        "complete_col": "complete_lp_integrality_ratio",
    },
    "pivot_approx": {
        "title": "Size effect in edge-deleted graphs: Pivot approximation ratio",
        "ylabel": "Pivot / ILP",
        "edge_col": "edge_pivot_approx",
        "complete_col": "complete_pivot_approx",
    },
    "sparse_vs_real_ilp_ratio": {
        "title": "Size effect in edge-deleted graphs: Sparse ILP / real ILP ratio",
        "ylabel": "Sparse ILP / all-pairs ILP",
        "edge_col": "edge_sparse_vs_real_ilp_ratio",
        "complete_col": "complete_sparse_vs_real_ilp_ratio",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRICS), default=sorted(METRICS))
    return parser.parse_args()


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def numeric_column(df, names):
    for name in names:
        if name in df.columns:
            values = to_num(df[name])
            if values.notna().sum() > 0:
                return values
    return pd.Series(np.nan, index=df.index)


def safe_ratio(num, den):
    return np.where(den.gt(0), num / den, np.nan)


def infer_family(row):
    graph_family = str(row.get("graph_family", "")).lower()
    graph_type = str(row.get("graph_type", "")).lower()
    file_name = str(row.get("file_name", "")).lower()
    file_path = str(row.get("file_path", "")).lower()

    if graph_family in FAMILY_ORDER:
        return graph_family
    if graph_type == "random" or "random" in file_name or "random" in file_path:
        return "random"
    if graph_type == "clique" or "clq" in file_name or "clq" in file_path:
        return "clique"
    if "facebook" in graph_type or "facebook" in file_path or "fb_" in file_name:
        return "facebook"
    return "other"


def parse_clique_sizes(row):
    raw = str(row.get("cluster_sizes", "")).strip()
    file_name = str(row.get("file_name", "")).strip()
    file_path = str(row.get("file_path", "")).strip()

    def parse_text(text):
        text = text.replace(".json", "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return []

        match = re.fullmatch(r"\[?\s*(\d+)\s*x\s*(\d+)\s*\]?", text)
        if match:
            return [int(match.group(2))] * int(match.group(1))

        return [int(x) for x in re.findall(r"\d+", text)]

    sizes = parse_text(raw)
    if sizes:
        return sizes

    stem = Path(file_name or file_path).stem
    match = re.match(r"clq_n\d+_(.+)", stem)
    if match:
        return parse_text(match.group(1))

    return []


def clique_balance_label(sizes):
    if not sizes:
        return "unknown"
    if len(set(sizes)) == 1:
        return "balanced"
    return "unbalanced"


def prepare_data(csv_path):
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in ["n", "seed", "p_delete", "p_positive", "ego_id"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    df["graph_family"] = df.apply(infer_family, axis=1)
    df = df[df["graph_family"].isin(FAMILY_ORDER)].copy()

    complete_ilp = numeric_column(df, ["complete_ilp_cost", "complete_sparse_ilp_cost"])
    complete_lp = numeric_column(df, ["complete_lp_cost", "complete_sparse_lp_cost"])
    complete_pivot = numeric_column(df, ["complete_pivot_average_cost"])
    complete_disjoint = numeric_column(df, ["complete_bad_triangles_max_disjoint"])

    all_pairs_ilp = numeric_column(df, ["edge_all_pairs_ilp_cost"])
    sparse_ilp = numeric_column(df, ["edge_ilp_with4_cost", "edge_sparse_ilp_with4_cost"])

    edge_ilp = all_pairs_ilp.copy()
    edge_ilp.loc[edge_ilp.isna()] = sparse_ilp.loc[edge_ilp.isna()]

    all_pairs_lp = numeric_column(df, ["edge_all_pairs_lp_cost"])
    sparse_lp = numeric_column(df, ["edge_lp_with4_cost", "edge_sparse_lp_with4_cost"])

    edge_lp = all_pairs_lp.copy()
    edge_lp.loc[edge_lp.isna()] = sparse_lp.loc[edge_lp.isna()]

    edge_lp_ratio = numeric_column(df, ["edge_all_pairs_lp_to_ilp_ratio"])
    sparse_lp_ratio = numeric_column(df, ["edge_lp_ratio_with4", "edge_sparse_lp_to_ilp_ratio_with4"])
    edge_lp_ratio.loc[edge_lp_ratio.isna()] = sparse_lp_ratio.loc[edge_lp_ratio.isna()]

    edge_pivot = numeric_column(df, ["edge_pivot_average_cost"])
    edge_disjoint = numeric_column(df, ["edge_bad_triangles_max_disjoint"])
    edge_bad4 = numeric_column(df, ["edge_bad_4_cycles_count"])

    df["complete_bad_4_cycles"] = 0.0
    df["edge_bad_4_cycles"] = edge_bad4

    df["complete_disjoint_bad_triangle_ratio"] = safe_ratio(complete_disjoint, complete_ilp)
    df["edge_disjoint_bad_triangle_ratio"] = safe_ratio(edge_disjoint, edge_ilp)

    df["complete_lp_integrality_ratio"] = safe_ratio(complete_lp, complete_ilp)
    df["edge_lp_integrality_ratio"] = edge_lp_ratio
    missing_lp_ratio = df["edge_lp_integrality_ratio"].isna()
    df.loc[missing_lp_ratio, "edge_lp_integrality_ratio"] = safe_ratio(edge_lp, edge_ilp)[missing_lp_ratio]

    df["complete_pivot_approx"] = safe_ratio(complete_pivot, complete_ilp)
    df["edge_pivot_approx"] = safe_ratio(edge_pivot, edge_ilp)

    df["complete_ilp_ratio"] = 1.0
    df["edge_ilp_ratio"] = safe_ratio(edge_ilp, complete_ilp)

    df["complete_sparse_vs_real_ilp_ratio"] = 1.0
    df["edge_sparse_vs_real_ilp_ratio"] = safe_ratio(sparse_ilp, all_pairs_ilp)

    clique_sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_balance"] = clique_sizes.apply(clique_balance_label)

    return df


def mean_table(df, group_cols, value_col):
    return (
        df.groupby(group_cols, dropna=False)
        .agg(value=(value_col, "mean"), runs=(value_col, "count"))
        .reset_index()
        .sort_values(group_cols)
    )


def pdelete_key(value):
    value = round(float(value), 2)
    return P_DELETE_ORDER.index(value) if value in P_DELETE_ORDER else 999


def complete_baseline(df, complete_col):
    table = mean_table(df, ["n"], complete_col)
    return table[table["value"].notna()].sort_values("n")


def plot_size_lines(ax, df, edge_col, complete_col, title, ylabel):
    edge = mean_table(df, ["n", "p_delete"], edge_col)
    edge = edge[edge["value"].notna()]
    comp = complete_baseline(df, complete_col)

    for p_delete in sorted(edge["p_delete"].dropna().unique(), key=pdelete_key):
        line = edge[edge["p_delete"].eq(p_delete)].sort_values("n")
        color = P_DELETE_COLORS.get(round(float(p_delete), 2), "black")

        ax.plot(
            line["n"],
            line["value"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"p_delete={float(p_delete):.2f}",
        )

    if not comp.empty:
        ax.plot(
            comp["n"],
            comp["value"],
            linestyle="--",
            linewidth=2.0,
            color="black",
            label="complete graph baseline",
        )

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["n"].dropna().unique()))
    ax.grid(True, alpha=0.28)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def remove_old_metric_folder(out_dir, metric_name):
    old_folder = out_dir / metric_name
    if old_folder.exists() and old_folder.is_dir():
        shutil.rmtree(old_folder)
        print(f"Removed old folder: {old_folder}")


def plot_metric(df, metric_name, metric, out_dir):
    panels = [
        ("random edge-deleted graphs", df[df["graph_family"].eq("random")].copy()),
        ("clique balanced edge-deleted graphs", df[df["graph_family"].eq("clique") & df["clique_balance"].eq("balanced")].copy()),
        ("clique unbalanced edge-deleted graphs", df[df["graph_family"].eq("clique") & df["clique_balance"].eq("unbalanced")].copy()),
        ("facebook edge-deleted graphs", df[df["graph_family"].eq("facebook")].copy()),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for ax, (title, sub) in zip(axes, panels):
        if sub.empty:
            ax.set_visible(False)
            continue

        plot_size_lines(
            ax,
            sub,
            metric["edge_col"],
            metric["complete_col"],
            title,
            metric["ylabel"],
        )

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="color = edge deletion probability",
            loc="lower center",
            ncol=5,
            fontsize=9,
        )

    fig.suptitle(metric["title"], fontsize=16)
    save(fig, out_dir / f"{metric_name}.png")


def main():
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    df = prepare_data(csv_path)

    for metric_name in args.metrics:
        metric = METRICS[metric_name]
        remove_old_metric_folder(out_dir, metric_name)
        plot_metric(df, metric_name, metric, out_dir)

    print(f"\nDone. Size-effect figures saved in: {out_dir}")


if __name__ == "__main__":
    main()
