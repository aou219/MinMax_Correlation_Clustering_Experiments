from pathlib import Path
import argparse
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "results" / "processed" / "all_runs_flat.csv"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "processed" / "figures"

FAMILY_ORDER = ["random", "clique", "facebook"]

METHODS = {
    "sparse-ilp": {
        "label": "Sparse ILP",
        "complete_ilp": ["complete_sparse_ilp_cost", "complete_ilp_cost"],
        "edge_ilp": ["edge_sparse_ilp_with4_cost", "edge_ilp_with4_cost"],
        "edge_lp": ["edge_sparse_lp_with4_cost", "edge_lp_with4_cost"],
        "lp_ratio": ["edge_sparse_lp_to_ilp_ratio_with4", "edge_lp_ratio_with4"],
    },
    "all-pairs-ilp": {
        "label": "All-pairs ILP",
        "complete_ilp": ["complete_sparse_ilp_cost", "complete_ilp_cost"],
        "edge_ilp": ["edge_all_pairs_ilp_cost"],
        "edge_lp": ["edge_all_pairs_lp_cost"],
        "lp_ratio": ["edge_all_pairs_lp_to_ilp_ratio"],
    },
}

METRICS = {
    "lp_integrality_ratio": {
        "column": "lp_integrality_ratio",
        "ylabel": "LP / ILP ratio after edge deletion",
        "title": "LP integrality ratio",
        "reference": 1.0,
    },
    "pivot_approx": {
        "column": "pivot_approx",
        "ylabel": "Average Pivot cost / ILP cost after edge deletion",
        "title": "Pivot approximation ratio",
        "reference": 1.0,
    },
    "bad_4_cycles": {
        "column": "edge_bad_4_cycles_count",
        "ylabel": "Number of bad 4-cycles",
        "title": "Bad 4-cycles after edge deletion",
        "reference": None,
    },
    "disjoint_bad_triangle_ratio": {
        "column": "disjoint_bad_triangle_ratio",
        "ylabel": "Max disjoint bad triangles / ILP cost",
        "title": "Disjoint bad triangle lower-bound ratio",
        "reference": 1.0,
    },
    "ilp_ratio": {
        "column": "ilp_ratio",
        "ylabel": "Edge-deleted ILP cost / complete ILP cost",
        "title": "ILP cost ratio after edge deletion",
        "reference": 1.0,
    },
}


def display_path(path):
    try:
        return path.relative_to(ROOT_DIR)
    except ValueError:
        return path


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name

    return None


def numeric_column(df, names):
    for name in names:
        if name not in df.columns:
            continue

        values = to_num(df[name])

        if values.notna().sum() > 0:
            return values

    return pd.Series(np.nan, index=df.index)


def infer_family(row):
    file_name = str(row.get("file_name", "")).lower()
    file_path = str(row.get("file_path", "")).lower()
    graph_type = str(row.get("graph_type", "")).lower()
    graph_family = str(row.get("graph_family", "")).lower()

    if graph_family in FAMILY_ORDER:
        return graph_family
    if "random" in file_name or "random" in file_path or graph_type == "random":
        return "random"
    if "clq" in file_name or "clq" in file_path or graph_type == "clique":
        return "clique"
    if (
        "fb_" in file_name
        or "fb_" in file_path
        or "facebook" in file_name
        or "facebook" in file_path
        or "facebook" in graph_type
    ):
        return "facebook"

    return "other"


def parse_clique_label(row):
    raw = str(row.get("cluster_sizes", "")).strip()
    file_name = str(row.get("file_name", "")).strip()

    if raw and raw.lower() not in {"nan", "none"}:
        nums = re.findall(r"\d+", raw)
        if nums:
            return "-".join(nums)

    stem = Path(file_name).stem
    match = re.match(r"clq_n\d+_(.+)", stem)
    if match:
        return match.group(1).replace("_", "-")

    return "unknown"


def load_base_data(csv_path):
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in [
        "n",
        "seed",
        "p_delete",
        "p_positive",
        "edge_bad_4_cycles_count",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    df["graph_family"] = df.apply(infer_family, axis=1)
    df = df[df["graph_family"].isin(FAMILY_ORDER)].copy()
    df["clique_label"] = df.apply(parse_clique_label, axis=1)
    df["ego_label"] = df["ego_id"].fillna("").astype(str) if "ego_id" in df.columns else ""

    return df


def add_method_metrics(base_df, method_info):
    df = base_df.copy()

    complete_ilp = numeric_column(df, method_info["complete_ilp"])
    edge_ilp = numeric_column(df, method_info["edge_ilp"])
    edge_lp = numeric_column(df, method_info["edge_lp"])
    edge_lp_ratio = numeric_column(df, method_info["lp_ratio"])
    edge_pivot_cost = numeric_column(df, ["edge_pivot_average_cost"])
    disjoint_bad_triangles = numeric_column(
        df,
        ["edge_bad_triangles_max_disjoint", "edge_bad_triangles_min_disjoint"],
    )

    df["lp_integrality_ratio"] = edge_lp_ratio
    df.loc[
        df["lp_integrality_ratio"].isna() & edge_ilp.eq(0) & edge_lp.eq(0),
        "lp_integrality_ratio",
    ] = 1.0

    df["pivot_approx"] = np.where(edge_ilp.gt(0), edge_pivot_cost / edge_ilp, np.nan)
    df.loc[
        df["pivot_approx"].isna() & edge_ilp.eq(0) & edge_pivot_cost.eq(0),
        "pivot_approx",
    ] = 1.0

    df["disjoint_bad_triangle_ratio"] = np.where(
        edge_ilp.gt(0),
        disjoint_bad_triangles / edge_ilp,
        np.nan,
    )

    df["ilp_ratio"] = np.where(
        complete_ilp.gt(0),
        edge_ilp / complete_ilp,
        np.nan,
    )

    return df


def aggregate_for_family(df, family, metric_col):
    sub = df[df["graph_family"].eq(family)].copy()

    if sub.empty:
        return pd.DataFrame()

    if family == "random":
        group_cols = ["p_delete", "p_positive"]
        label_col = "p_positive"
    elif family == "clique":
        group_cols = ["p_delete", "n"]
        label_col = "n"
    else:
        group_cols = ["p_delete", "ego_label"]
        label_col = "ego_label"

    grouped = (
        sub.groupby(group_cols, dropna=False)
        .agg(value=(metric_col, "mean"), runs=(metric_col, "count"))
        .reset_index()
        .sort_values(group_cols)
    )

    grouped["line_label"] = grouped[label_col].apply(format_line_label(family))
    return grouped


def format_line_label(family):
    def formatter(value):
        if pd.isna(value) or str(value).lower() == "nan":
            return "unknown"
        if family == "random":
            return f"p_pos={float(value):.1f}"
        if family == "clique":
            return f"n={int(float(value))}"
        return f"ego={str(value).replace('.0', '')}"

    return formatter


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {display_path(path)}")


def plot_family(df, family, method_label, metric_name, metric_info, out_dir):
    metric_col = metric_info["column"]
    table = aggregate_for_family(df, family, metric_col)

    if table.empty or table["value"].notna().sum() == 0:
        print(f"Skipping {out_dir.name}/{metric_name}/{family}: no data")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for label, line in table.groupby("line_label"):
        line = line.sort_values("p_delete")
        ax.plot(
            line["p_delete"],
            line["value"],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    if metric_info["reference"] is not None:
        ax.axhline(metric_info["reference"], linestyle="--", linewidth=1.0, color="black", alpha=0.55)

    ax.set_title(f"{method_label}: {metric_info['title']} - {family}")
    ax.set_xlabel("p_delete")
    ax.set_ylabel(metric_info["ylabel"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    save_figure(fig, out_dir / metric_name / f"{family}.png")


def plot_compare(df, method_label, metric_name, metric_info, out_dir):
    metric_col = metric_info["column"]

    table = (
        df.groupby(["graph_family", "p_delete"], dropna=False)
        .agg(value=(metric_col, "mean"), runs=(metric_col, "count"))
        .reset_index()
        .sort_values(["graph_family", "p_delete"])
    )

    if table.empty or table["value"].notna().sum() == 0:
        print(f"Skipping {out_dir.name}/{metric_name}/compare: no data")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for family in FAMILY_ORDER:
        line = table[table["graph_family"].eq(family)].sort_values("p_delete")

        if line.empty or line["value"].notna().sum() == 0:
            continue

        ax.plot(
            line["p_delete"],
            line["value"],
            marker="o",
            linewidth=2.0,
            label=family,
        )

    if metric_info["reference"] is not None:
        ax.axhline(metric_info["reference"], linestyle="--", linewidth=1.0, color="black", alpha=0.55)

    ax.set_title(f"{method_label}: {metric_info['title']} - graph family comparison")
    ax.set_xlabel("p_delete")
    ax.set_ylabel(metric_info["ylabel"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    save_figure(fig, out_dir / metric_name / "compare.png")


def write_summary_table(method_tables, out_dir):
    table_dir = out_dir.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for method_name, df in method_tables.items():
        for metric_name, metric_info in METRICS.items():
            metric_col = metric_info["column"]
            grouped = (
                df.groupby("graph_family", dropna=False)[metric_col]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
            )
            grouped.insert(0, "metric", metric_name)
            grouped.insert(0, "method", method_name)
            rows.append(grouped)

    summary = pd.concat(rows, ignore_index=True)
    path = table_dir / "thesis_figure_metric_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {display_path(path)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis figures from saved all_runs_flat.csv results. "
            "This script does not run ILP/Gurobi."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to all_runs_flat.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for figure folders.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS.keys()),
        default=sorted(METHODS.keys()),
        help="Which ILP result types to plot.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRICS.keys()),
        default=sorted(METRICS.keys()),
        help="Metric folders to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else ROOT_DIR / args.csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir

    base_df = load_base_data(csv_path)
    method_tables = {}

    for method_name in args.methods:
        method_info = METHODS[method_name]
        method_out_dir = out_dir / method_name
        df = add_method_metrics(base_df, method_info)
        method_tables[method_name] = df

        for metric_name in args.metrics:
            metric_info = METRICS[metric_name]

            for family in FAMILY_ORDER:
                plot_family(
                    df=df,
                    family=family,
                    method_label=method_info["label"],
                    metric_name=metric_name,
                    metric_info=metric_info,
                    out_dir=method_out_dir,
                )

            plot_compare(
                df=df,
                method_label=method_info["label"],
                metric_name=metric_name,
                metric_info=metric_info,
                out_dir=method_out_dir,
            )

    write_summary_table(method_tables, out_dir)
    print(f"\nDone. Figures saved in: {display_path(out_dir)}")


if __name__ == "__main__":
    main()
