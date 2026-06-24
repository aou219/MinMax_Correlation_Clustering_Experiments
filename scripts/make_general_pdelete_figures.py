from pathlib import Path
import argparse
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "processed" / "all_runs_flat.csv"
OUT = ROOT / "results" / "processed" / "figures" / "p_delete_effect" / "general"

P_DELETE_ORDER = [0.05, 0.15, 0.25, 0.40]
FAMILY_ORDER = ["random", "clique", "facebook"]

P_POS_COLORS = {
    0.2: "#d62728",  # red
    0.3: "#ff7f0e",  # orange
    0.4: "#f1c40f",  # yellow
    0.5: "#2ca02c",  # green
    0.6: "#1f77b4",  # blue
    0.7: "#4b0082",  # indigo
    0.8: "#8a2be2",  # violet
}

CLIQUE_COLORS = {
    "balanced": "#1f77b4",
    "unbalanced": "#d62728",
}

FAMILY_COLORS = {
    "random": "#d62728",
    "clique": "#1f77b4",
    "facebook": "#9467bd",
}

FACEBOOK_COLORS = {
    "ego 414": "#d62728",
    "ego 698": "#9467bd",
    "ego 3980": "#1f77b4",
}

FACEBOOK_EXCLUDE = {"ego 686"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild p_delete_effect/general figures. Aggregates by first averaging "
            "within each n, then averaging across n so each size has equal weight."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def numeric_column(df, names):
    for name in names:
        if name not in df.columns:
            continue
        values = to_num(df[name])
        if values.notna().sum() > 0:
            return values
    return pd.Series(np.nan, index=df.index)


def safe_ratio(num, den, zero_zero_as_one=False):
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = den.gt(0)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    if zero_zero_as_one:
        exact_zero = den.eq(0) & num.eq(0)
        result.loc[exact_zero] = 1.0
    return result


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
    complete_disjoint = numeric_column(df, ["complete_bad_triangles_max_disjoint"])

    edge_all_pairs_ilp = numeric_column(df, ["edge_all_pairs_ilp_cost"])
    edge_sparse_ilp = numeric_column(df, ["edge_ilp_with4_cost", "edge_sparse_ilp_with4_cost"])
    edge_ilp = edge_all_pairs_ilp.copy()
    edge_ilp.loc[edge_ilp.isna()] = edge_sparse_ilp.loc[edge_ilp.isna()]

    edge_all_pairs_lp = numeric_column(df, ["edge_all_pairs_lp_cost"])
    edge_sparse_lp = numeric_column(df, ["edge_lp_with4_cost", "edge_sparse_lp_with4_cost"])
    edge_lp = edge_all_pairs_lp.copy()
    edge_lp.loc[edge_lp.isna()] = edge_sparse_lp.loc[edge_lp.isna()]

    edge_lp_ratio = numeric_column(df, ["edge_all_pairs_lp_to_ilp_ratio"])
    sparse_lp_ratio = numeric_column(df, ["edge_lp_ratio_with4", "edge_sparse_lp_to_ilp_ratio_with4"])
    edge_lp_ratio.loc[edge_lp_ratio.isna()] = sparse_lp_ratio.loc[edge_lp_ratio.isna()]

    complete_pivot_avg = numeric_column(df, ["complete_pivot_average_cost"])
    edge_pivot_avg = numeric_column(df, ["edge_pivot_average_cost"])
    complete_pivot_best = numeric_column(df, ["complete_pivot_best_cost", "complete_pivot_average_cost"])
    edge_pivot_best = numeric_column(df, ["edge_pivot_best_cost", "edge_pivot_average_cost"])

    df["complete_bad_4_cycles"] = 0.0
    df["edge_bad_4_cycles"] = numeric_column(df, ["edge_bad_4_cycles_count"])

    df["complete_disjoint_bad_triangle_ratio"] = safe_ratio(complete_disjoint, complete_ilp)
    df["edge_disjoint_bad_triangle_ratio"] = safe_ratio(
        numeric_column(df, ["edge_bad_triangles_max_disjoint"]),
        edge_ilp,
    )

    df["complete_lp_integrality_ratio"] = safe_ratio(complete_lp, complete_ilp, zero_zero_as_one=True)
    df["edge_lp_integrality_ratio"] = edge_lp_ratio
    missing_lp_ratio = df["edge_lp_integrality_ratio"].isna()
    df.loc[missing_lp_ratio, "edge_lp_integrality_ratio"] = safe_ratio(
        edge_lp,
        edge_ilp,
        zero_zero_as_one=True,
    )[missing_lp_ratio]

    df["complete_average_pivot_approx"] = safe_ratio(
        complete_pivot_avg,
        complete_ilp,
        zero_zero_as_one=True,
    )
    df["edge_average_pivot_approx"] = safe_ratio(
        edge_pivot_avg,
        edge_ilp,
        zero_zero_as_one=True,
    )
    df["complete_best_pivot_approx"] = safe_ratio(
        complete_pivot_best,
        complete_ilp,
        zero_zero_as_one=True,
    )
    df["edge_best_pivot_approx"] = safe_ratio(
        edge_pivot_best,
        edge_ilp,
        zero_zero_as_one=True,
    )

    df["complete_ilp_ratio"] = 1.0
    df["edge_ilp_ratio"] = safe_ratio(edge_ilp, complete_ilp)

    clique_sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_balance"] = clique_sizes.apply(clique_balance_label)
    df["ego_label"] = df["ego_id"].apply(
        lambda x: f"ego {int(x)}" if pd.notna(x) else "ego unknown"
    )

    df = df[~df["ego_label"].isin(FACEBOOK_EXCLUDE)].copy()
    return df


METRICS = [
    {
        "folder": "bad_4_cycles",
        "title": "Bad 4-cycles after edge deletion",
        "ylabel": "Number of bad 4-cycles",
        "edge_col": "edge_bad_4_cycles",
        "complete_col": None,
        "style_legend": False,
    },
    {
        "folder": "disjoint_bad_triangle_ratio",
        "title": "Edge-disjoint bad triangle lower-bound ratio",
        "ylabel": "Max disjoint bad triangles / ILP",
        "edge_col": "edge_disjoint_bad_triangle_ratio",
        "complete_col": "complete_disjoint_bad_triangle_ratio",
        "style_legend": True,
    },
    {
        "folder": "ilp_ratio",
        "title": "Deleted graph ILP / complete graph ILP",
        "ylabel": "Deleted graph ILP / complete graph ILP",
        "edge_col": "edge_ilp_ratio",
        "complete_col": None,
        "style_legend": False,
    },
    {
        "folder": "lp_integrality_ratio",
        "title": "LP / ILP ratio",
        "ylabel": "LP / ILP",
        "edge_col": "edge_lp_integrality_ratio",
        "complete_col": "complete_lp_integrality_ratio",
        "style_legend": True,
    },
    {
        "folder": "pivot_approx/average_pivot_approx",
        "title": "Average Pivot approximation ratio",
        "ylabel": "Average Pivot / ILP",
        "edge_col": "edge_average_pivot_approx",
        "complete_col": "complete_average_pivot_approx",
        "style_legend": True,
    },
    {
        "folder": "pivot_approx/best_pivot_approx",
        "title": "Best Pivot approximation ratio",
        "ylabel": "Best Pivot / ILP",
        "edge_col": "edge_best_pivot_approx",
        "complete_col": "complete_best_pivot_approx",
        "style_legend": True,
    },
]


def equal_n_table(df, group_cols, edge_col, complete_col=None):
    value_cols = {"edge_value": edge_col}
    if complete_col:
        value_cols["complete_value"] = complete_col

    by_n = (
        df.groupby(group_cols + ["p_delete", "n"], dropna=False)
        .agg(**{name: (col, "mean") for name, col in value_cols.items()})
        .reset_index()
    )

    return (
        by_n.groupby(group_cols + ["p_delete"], dropna=False)
        .agg(**{name: (name, "mean") for name in value_cols})
        .reset_index()
        .sort_values(group_cols + ["p_delete"])
    )


def finish_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(P_DELETE_ORDER)
    ax.grid(True, alpha=0.28)


def style_handles():
    return [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="edge-deleted"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="complete"),
    ]


def add_legend(ax, handles, metric, title=None, loc="best", fontsize=9):
    legend_handles = list(handles)
    if metric["style_legend"]:
        legend_handles += style_handles()
    ax.legend(
        handles=legend_handles,
        title=title,
        loc=loc,
        fontsize=fontsize,
        title_fontsize=fontsize,
    )


def plot_line_pair(ax, table, color, label, metric, linewidth=2.0):
    line = table.sort_values("p_delete")
    if line.empty or line["edge_value"].notna().sum() == 0:
        return None

    ax.plot(
        line["p_delete"],
        line["edge_value"],
        marker="o",
        linewidth=linewidth,
        color=color,
    )

    if metric["complete_col"] and "complete_value" in line:
        comp = line[line["complete_value"].notna()]
        if not comp.empty:
            ax.plot(
                comp["p_delete"],
                comp["complete_value"],
                linestyle="--",
                linewidth=linewidth * 0.9,
                color=color,
            )

    return Line2D([0], [0], color=color, marker="o", linewidth=linewidth, label=label)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_random(df, metric):
    sub = df[df["graph_family"].eq("random")].copy()
    table = equal_n_table(sub, ["p_positive"], metric["edge_col"], metric["complete_col"])

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for p_pos in sorted(table["p_positive"].dropna().unique()):
        color = P_POS_COLORS.get(round(float(p_pos), 1), "black")
        line = table[np.isclose(table["p_positive"], p_pos, equal_nan=False)]
        handle = plot_line_pair(ax, line, color, f"p_pos={p_pos:.1f}", metric)
        if handle:
            handles.append(handle)

    finish_axis(ax, f"{metric['title']} - random, all n", metric["ylabel"])
    add_legend(ax, handles, metric, title="color = p_pos", loc="best")
    save(fig, OUT / metric["folder"] / "random.png")


def plot_clique(df, metric):
    sub = df[df["graph_family"].eq("clique") & df["clique_balance"].isin(["balanced", "unbalanced"])].copy()
    table = equal_n_table(sub, ["clique_balance"], metric["edge_col"], metric["complete_col"])

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for label in ["balanced", "unbalanced"]:
        line = table[table["clique_balance"].eq(label)]
        handle = plot_line_pair(ax, line, CLIQUE_COLORS[label], label, metric)
        if handle:
            handles.append(handle)

    finish_axis(ax, f"{metric['title']} - clique, all n", metric["ylabel"])
    add_legend(ax, handles, metric, title="color = clique type", loc="best")
    save(fig, OUT / metric["folder"] / "clique.png")


def plot_facebook(df, metric):
    sub = df[df["graph_family"].eq("facebook")].copy()
    table = equal_n_table(sub, ["ego_label"], metric["edge_col"], metric["complete_col"])

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for ego in ["ego 414", "ego 698", "ego 3980"]:
        line = table[table["ego_label"].eq(ego)]
        handle = plot_line_pair(ax, line, FACEBOOK_COLORS[ego], ego, metric)
        if handle:
            handles.append(handle)

    finish_axis(ax, f"{metric['title']} - facebook", metric["ylabel"])
    add_legend(ax, handles, metric, title="color = ego graph", loc="best")
    save(fig, OUT / metric["folder"] / "facebook.png")


def plot_compare(df, metric):
    table = equal_n_table(df, ["graph_family"], metric["edge_col"], metric["complete_col"])

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for family in FAMILY_ORDER:
        line = table[table["graph_family"].eq(family)]
        handle = plot_line_pair(ax, line, FAMILY_COLORS[family], family, metric, linewidth=2.2)
        if handle:
            handles.append(handle)

    finish_axis(ax, f"{metric['title']} - graph family comparison", metric["ylabel"])
    add_legend(ax, handles, metric, loc="best")
    save(fig, OUT / metric["folder"] / "compare.png")


def remove_old_direct_pivot_files():
    pivot_dir = OUT / "pivot_approx"
    for name in ["random.png", "clique.png", "facebook.png", "compare.png"]:
        path = pivot_dir / name
        if path.exists():
            path.unlink()
            print(f"Deleted old direct pivot file: {path}")


def main():
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    df = prepare_data(csv_path)

    remove_old_direct_pivot_files()

    for metric in METRICS:
        plot_random(df, metric)
        plot_clique(df, metric)
        plot_facebook(df, metric)
        plot_compare(df, metric)

    print("\nDone. Rebuilt p_delete_effect/general figures with equal-n aggregation.")


if __name__ == "__main__":
    main()
