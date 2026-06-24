from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(".")
CSV_PATH = ROOT / "results" / "processed" / "all_runs_flat.csv"
OUT = ROOT / "results" / "processed" / "figures" / "general"
P_DELETE_ORDER = [0.05, 0.15, 0.25, 0.40]

COLORS = {
    "balanced": "#1f77b4",
    "unbalanced": "#d62728",
}


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def first_numeric(df, names):
    for name in names:
        if name in df.columns:
            values = to_num(df[name])
            if values.notna().sum() > 0:
                return values
    return pd.Series(np.nan, index=df.index)


def safe_ratio(num, den):
    return np.where(den.gt(0), num / den, np.nan)


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


def mean_table(df, group_cols, value_col):
    return (
        df.groupby(group_cols, dropna=False)
        .agg(value=(value_col, "mean"), runs=(value_col, "count"))
        .reset_index()
        .sort_values(group_cols)
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def prepare_data():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    for col in ["n", "seed", "p_delete"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    graph_family = df.get("graph_family", "").astype(str).str.lower()
    graph_type = df.get("graph_type", "").astype(str).str.lower()
    file_name = df.get("file_name", "").astype(str).str.lower()
    file_path = df.get("file_path", "").astype(str).str.lower()

    clique_mask = (
        graph_family.eq("clique")
        | graph_type.eq("clique")
        | file_name.str.contains("clq", na=False)
        | file_path.str.contains("clq", na=False)
    )
    df = df[clique_mask].copy()

    clique_sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_balance"] = clique_sizes.apply(clique_balance_label)
    df = df[df["clique_balance"].isin(["balanced", "unbalanced"])].copy()

    complete_ilp = first_numeric(df, ["complete_ilp_cost", "complete_sparse_ilp_cost"])
    complete_lp = first_numeric(df, ["complete_lp_cost", "complete_sparse_lp_cost"])
    complete_pivot = first_numeric(df, ["complete_pivot_average_cost"])
    complete_disjoint = first_numeric(df, ["complete_bad_triangles_max_disjoint"])

    all_pairs_ilp = first_numeric(df, ["edge_all_pairs_ilp_cost"])
    all_pairs_lp = first_numeric(df, ["edge_all_pairs_lp_cost"])
    all_pairs_lp_ratio = first_numeric(df, ["edge_all_pairs_lp_to_ilp_ratio"])
    edge_pivot = first_numeric(df, ["edge_pivot_average_cost"])
    edge_disjoint = first_numeric(df, ["edge_bad_triangles_max_disjoint"])
    edge_bad4 = first_numeric(df, ["edge_bad_4_cycles_count"])

    df["complete_bad_4_cycles"] = 0.0
    df["edge_bad_4_cycles"] = edge_bad4

    df["complete_disjoint_bad_triangle_ratio"] = safe_ratio(complete_disjoint, complete_ilp)
    df["edge_disjoint_bad_triangle_ratio"] = safe_ratio(edge_disjoint, all_pairs_ilp)

    df["complete_lp_integrality_ratio"] = safe_ratio(complete_lp, complete_ilp)
    df["edge_lp_integrality_ratio"] = all_pairs_lp_ratio
    missing_lp_ratio = df["edge_lp_integrality_ratio"].isna()
    df.loc[missing_lp_ratio, "edge_lp_integrality_ratio"] = safe_ratio(
        all_pairs_lp, all_pairs_ilp
    )[missing_lp_ratio]

    df["complete_pivot_approx"] = safe_ratio(complete_pivot, complete_ilp)
    df["edge_pivot_approx"] = safe_ratio(edge_pivot, all_pairs_ilp)

    df["complete_ilp_ratio"] = 1.0
    df["edge_ilp_ratio"] = safe_ratio(all_pairs_ilp, complete_ilp)

    return df


METRICS = {
    "bad_4_cycles": {
        "title": "Bad 4-cycle count - clique",
        "ylabel": "Bad 4-cycles",
        "edge_col": "edge_bad_4_cycles",
        "complete_col": "complete_bad_4_cycles",
    },
    "disjoint_bad_triangle_ratio": {
        "title": "Edge-disjoint bad triangle lower bound ratio - clique",
        "ylabel": "Max disjoint bad triangles / all-pairs ILP",
        "edge_col": "edge_disjoint_bad_triangle_ratio",
        "complete_col": "complete_disjoint_bad_triangle_ratio",
    },
    "ilp_ratio": {
        "title": "ILP cost ratio after edge deletion - clique",
        "ylabel": "Edge-deleted all-pairs ILP / complete ILP",
        "edge_col": "edge_ilp_ratio",
        "complete_col": "complete_ilp_ratio",
    },
    "lp_integrality_ratio": {
        "title": "LP / all-pairs ILP ratio - clique",
        "ylabel": "LP / all-pairs ILP",
        "edge_col": "edge_lp_integrality_ratio",
        "complete_col": "complete_lp_integrality_ratio",
    },
    "pivot_approx": {
        "title": "Pivot approximation ratio - clique",
        "ylabel": "Pivot / all-pairs ILP",
        "edge_col": "edge_pivot_approx",
        "complete_col": "complete_pivot_approx",
    },
}


def plot_metric(df, metric_name, metric):
    fig, ax = plt.subplots(figsize=(10, 6))

    edge = mean_table(df, ["p_delete", "clique_balance"], metric["edge_col"])
    comp = mean_table(df, ["p_delete", "clique_balance"], metric["complete_col"])

    handles = []

    for label in ["balanced", "unbalanced"]:
        e = edge[edge["clique_balance"].eq(label)].sort_values("p_delete")
        c = comp[comp["clique_balance"].eq(label)].sort_values("p_delete")

        if e.empty or e["value"].notna().sum() == 0:
            continue

        color = COLORS[label]

        ax.plot(
            e["p_delete"],
            e["value"],
            marker="o",
            linewidth=2.2,
            color=color,
        )

        if not c.empty and c["value"].notna().sum() > 0:
            ax.plot(
                c["p_delete"],
                c["value"],
                linestyle="--",
                linewidth=2.0,
                color=color,
            )

        handles.append(
            Line2D([0], [0], color=color, marker="o", linewidth=2.2, label=label)
        )

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="edge-deleted"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="complete"),
    ]

    ax.set_title(metric["title"], fontsize=16)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(P_DELETE_ORDER)
    ax.grid(True, alpha=0.28)
    ax.legend(handles=handles + style_handles, title="color = clique type", fontsize=9)

    save(fig, OUT / metric_name / "clique.png")


def main():
    df = prepare_data()

    for metric_name, metric in METRICS.items():
        plot_metric(df, metric_name, metric)

    print(f"\nDone. General clique balance figures saved in: {OUT}")


if __name__ == "__main__":
    main()
