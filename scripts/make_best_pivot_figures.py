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
PDELETE_OUT = ROOT / "results" / "processed" / "figures" / "p_delete_effect"
SIZE_OUT = ROOT / "results" / "processed" / "figures" / "size_effect"

P_DELETE_ORDER = [0.05, 0.15, 0.25, 0.40]

P_POS_COLORS = {
    0.2: "#d62728",  # red
    0.3: "#ff7f0e",  # orange
    0.4: "#f1c40f",  # yellow
    0.5: "#2ca02c",  # green
    0.6: "#1f77b4",  # blue
    0.7: "#4b0082",  # indigo
    0.8: "#8a2be2",  # violet
}

RANDOM_N_COLORS = {
    10: "#d62728",
    20: "#ff7f0e",
    30: "#2ca02c",
}

CLIQUE_N_COLORS = {
    10: "#1f77b4",
    20: "#9467bd",
    30: "#8c564b",
    100: "#e377c2",
}

FACEBOOK_EGO_COLORS = {
    "ego 414": "#17becf",
    "ego 686": "#2ca02c",
    "ego 698": "#bcbd22",
    "ego 3980": "#7f7f7f",
}

CLIQUE_BALANCE_COLORS = {
    "balanced": "#1f77b4",
    "unbalanced": "#d62728",
}

P_DELETE_COLORS = {
    0.05: "#1f77b4",  # blue
    0.15: "#2ca02c",  # green
    0.25: "#ff7f0e",  # orange
    0.40: "#d62728",  # red
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate only the pivot approximation figures, split into average "
            "Pivot and best Pivot output folders."
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


def safe_ratio(num, den):
    return np.where(den.gt(0), num / den, np.nan)


def infer_family(row):
    graph_family = str(row.get("graph_family", "")).lower()
    graph_type = str(row.get("graph_type", "")).lower()
    file_name = str(row.get("file_name", "")).lower()
    file_path = str(row.get("file_path", "")).lower()

    if graph_family in {"random", "clique", "facebook"}:
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
    df = df[df["graph_family"].isin(["random", "clique", "facebook"])].copy()

    complete_ilp = numeric_column(df, ["complete_ilp_cost", "complete_sparse_ilp_cost"])
    edge_ilp = numeric_column(df, ["edge_all_pairs_ilp_cost"])
    fallback_edge_ilp = numeric_column(df, ["edge_ilp_with4_cost", "edge_sparse_ilp_with4_cost"])
    missing_edge_ilp = edge_ilp.isna()
    edge_ilp.loc[missing_edge_ilp] = fallback_edge_ilp.loc[missing_edge_ilp]

    complete_pivot_best = numeric_column(
        df,
        ["complete_pivot_best_cost", "complete_pivot_average_cost"],
    )
    edge_pivot_best = numeric_column(
        df,
        ["edge_pivot_best_cost", "edge_pivot_average_cost"],
    )
    complete_pivot_average = numeric_column(
        df,
        ["complete_pivot_average_cost", "complete_pivot_best_cost"],
    )
    edge_pivot_average = numeric_column(
        df,
        ["edge_pivot_average_cost", "edge_pivot_best_cost"],
    )

    df["complete_best_pivot_approx"] = safe_ratio(complete_pivot_best, complete_ilp)
    df["edge_best_pivot_approx"] = safe_ratio(edge_pivot_best, edge_ilp)
    df["complete_average_pivot_approx"] = safe_ratio(complete_pivot_average, complete_ilp)
    df["edge_average_pivot_approx"] = safe_ratio(edge_pivot_average, edge_ilp)

    clique_sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_balance"] = clique_sizes.apply(clique_balance_label)
    df["ego_label"] = df["ego_id"].apply(
        lambda x: f"ego {int(x)}" if pd.notna(x) else "ego unknown"
    )

    return df


def mean_table(df, group_cols, value_col):
    return (
        df.groupby(group_cols, dropna=False)
        .agg(value=(value_col, "mean"), runs=(value_col, "count"))
        .reset_index()
        .sort_values(group_cols)
    )


def finish_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(P_DELETE_ORDER)
    ax.grid(True, alpha=0.28)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def style_handles():
    return [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="edge-deleted"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="complete"),
    ]


def add_legend(ax, series_handles, title=None, loc="best", fontsize=8):
    ax.legend(
        handles=list(series_handles) + style_handles(),
        title=title,
        loc=loc,
        fontsize=fontsize,
        title_fontsize=fontsize,
    )


def metric_name(mode):
    return "Best Pivot" if mode == "best" else "Average Pivot"


def edge_metric(mode):
    return f"edge_{mode}_pivot_approx"


def complete_metric(mode):
    return f"complete_{mode}_pivot_approx"


def mode_folder(mode):
    return f"{mode}_pivot_approx"


def pdelete_pivot_path(scope, mode, filename):
    return PDELETE_OUT / scope / "pivot_approx" / mode_folder(mode) / filename


def size_pivot_path(mode):
    return SIZE_OUT / mode_folder(mode) / "pivot_approx.png"


def plot_pair(ax, x, edge_y, complete_y, color, label, linewidth=1.8):
    ax.plot(x, edge_y, marker="o", linewidth=linewidth, color=color)
    if complete_y is not None and len(complete_y) > 0:
        ax.plot(x, complete_y, linestyle="--", linewidth=linewidth * 0.9, color=color)
    return Line2D([0], [0], color=color, marker="o", linewidth=linewidth, label=label)


def plot_general_random(df, mode):
    sub = df[df["graph_family"].eq("random")].copy()
    label = metric_name(mode)
    edge = mean_table(sub, ["p_delete", "p_positive"], edge_metric(mode))
    comp = mean_table(sub, ["p_delete", "p_positive"], complete_metric(mode))

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for p_pos in sorted(edge["p_positive"].dropna().unique()):
        e = edge[edge["p_positive"].eq(p_pos)].sort_values("p_delete")
        c = comp[comp["p_positive"].eq(p_pos)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        color = P_POS_COLORS.get(round(float(p_pos), 1), "black")
        handles.append(
            plot_pair(
                ax,
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                color,
                f"p_pos={p_pos:.1f}",
                linewidth=2.0,
            )
        )

    finish_axis(ax, f"{label} approximation ratio - random", f"{label} / all-pairs ILP")
    add_legend(ax, handles, title="color = p_pos", loc="best", fontsize=9)
    save(fig, pdelete_pivot_path("general", mode, "random.png"))


def plot_general_clique(df, mode):
    sub = df[df["graph_family"].eq("clique")].copy()
    label = metric_name(mode)
    edge = mean_table(sub, ["p_delete", "clique_balance"], edge_metric(mode))
    comp = mean_table(sub, ["p_delete", "clique_balance"], complete_metric(mode))

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for label in ["balanced", "unbalanced"]:
        e = edge[edge["clique_balance"].eq(label)].sort_values("p_delete")
        c = comp[comp["clique_balance"].eq(label)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        color = CLIQUE_BALANCE_COLORS[label]
        handles.append(
            plot_pair(
                ax,
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                color,
                label,
                linewidth=2.0,
            )
        )

    finish_axis(ax, f"{label} approximation ratio - clique", f"{label} / all-pairs ILP")
    add_legend(ax, handles, title="color = clique type", loc="best", fontsize=9)
    save(fig, pdelete_pivot_path("general", mode, "clique.png"))


def plot_general_facebook(df, mode):
    sub = df[df["graph_family"].eq("facebook")].copy()
    label = metric_name(mode)
    edge = mean_table(sub, ["p_delete", "ego_label"], edge_metric(mode))
    comp = mean_table(sub, ["p_delete", "ego_label"], complete_metric(mode))

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for ego in sorted(edge["ego_label"].dropna().unique()):
        e = edge[edge["ego_label"].eq(ego)].sort_values("p_delete")
        c = comp[comp["ego_label"].eq(ego)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        color = FACEBOOK_EGO_COLORS.get(ego, "black")
        handles.append(
            plot_pair(
                ax,
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                color,
                ego,
                linewidth=2.0,
            )
        )

    finish_axis(ax, f"{label} approximation ratio - facebook", f"{label} / all-pairs ILP")
    add_legend(ax, handles, title="color = ego graph", loc="best", fontsize=9)
    save(fig, pdelete_pivot_path("general", mode, "facebook.png"))


def plot_general_compare(df, mode):
    label = metric_name(mode)
    table = (
        df.groupby(["graph_family", "p_delete"], dropna=False)
        .agg(value=(edge_metric(mode), "mean"), runs=(edge_metric(mode), "count"))
        .reset_index()
        .sort_values(["graph_family", "p_delete"])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    family_colors = {
        "random": "#1f77b4",
        "clique": "#ff7f0e",
        "facebook": "#2ca02c",
    }

    for family in ["random", "clique", "facebook"]:
        line = table[table["graph_family"].eq(family)].sort_values("p_delete")
        if line.empty or line["value"].notna().sum() == 0:
            continue
        ax.plot(
            line["p_delete"],
            line["value"],
            marker="o",
            linewidth=2.2,
            color=family_colors[family],
            label=family,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.55)
    finish_axis(
        ax,
        f"{label} approximation ratio - graph family comparison",
        f"{label} / all-pairs ILP after edge deletion",
    )
    ax.legend(fontsize=9)
    save(fig, pdelete_pivot_path("general", mode, "compare.png"))


def plot_specific_random(df, mode):
    sub = df[df["graph_family"].eq("random") & df["n"].isin([10, 20, 30])].copy()
    label = metric_name(mode)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    handles = []
    p_values = sorted(sub["p_positive"].dropna().unique())

    for ax, n in zip(axes, [10, 20, 30]):
        part = sub[sub["n"].eq(n)]
        edge = mean_table(part, ["p_delete", "p_positive"], edge_metric(mode))
        comp = mean_table(part, ["p_delete", "p_positive"], complete_metric(mode))

        for p_pos in p_values:
            e = edge[edge["p_positive"].eq(p_pos)].sort_values("p_delete")
            c = comp[comp["p_positive"].eq(p_pos)].sort_values("p_delete")
            if e.empty or e["value"].notna().sum() == 0:
                continue
            color = P_POS_COLORS.get(round(float(p_pos), 1), "black")
            handle = plot_pair(
                ax,
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                color,
                f"p={p_pos:.1f}",
            )
            if n == 10:
                handles.append(handle)

        finish_axis(ax, f"random n={n}", f"{label} / all-pairs ILP")

    add_legend(axes[-1], handles, title="color = p_pos", loc="best", fontsize=8)
    fig.suptitle(f"{label} approximation ratio - random")
    save(fig, pdelete_pivot_path("specific", mode, "random.png"))


def plot_specific_clique(df, mode):
    sub = df[df["graph_family"].eq("clique") & df["n"].isin([10, 20, 30, 100])].copy()
    label = metric_name(mode)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    handles = []

    for ax, n in zip(axes, [10, 20, 30, 100]):
        part = sub[sub["n"].eq(n)]
        edge = mean_table(part, ["p_delete", "clique_balance"], edge_metric(mode))
        comp = mean_table(part, ["p_delete", "clique_balance"], complete_metric(mode))

        for label in ["balanced", "unbalanced"]:
            e = edge[edge["clique_balance"].eq(label)].sort_values("p_delete")
            c = comp[comp["clique_balance"].eq(label)].sort_values("p_delete")
            if e.empty or e["value"].notna().sum() == 0:
                continue
            color = CLIQUE_BALANCE_COLORS[label]
            handle = plot_pair(
                ax,
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                color,
                label,
            )
            if n == 10:
                handles.append(handle)

        finish_axis(ax, f"clique n={n}", f"{label} / all-pairs ILP")

    add_legend(axes[-1], handles, title="color = clique type", loc="best", fontsize=8)
    fig.suptitle(f"{label} approximation ratio - clique")
    save(fig, pdelete_pivot_path("specific", mode, "clique.png"))


def plot_specific_facebook(df, mode):
    sub = df[df["graph_family"].eq("facebook")].copy()
    label = metric_name(mode)
    edge = mean_table(sub, ["p_delete", "ego_label"], edge_metric(mode))
    comp = mean_table(sub, ["p_delete", "ego_label"], complete_metric(mode))

    fig, ax = plt.subplots(figsize=(12, 7))
    handles = []

    for ego in sorted(edge["ego_label"].dropna().unique()):
        e = edge[edge["ego_label"].eq(ego)].sort_values("p_delete")
        c = comp[comp["ego_label"].eq(ego)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        color = FACEBOOK_EGO_COLORS.get(ego, "black")
        handles.append(
            plot_pair(
                ax,
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                color,
                ego,
                linewidth=2.3,
            )
        )

    finish_axis(ax, "facebook ego graphs", f"{label} / all-pairs ILP")
    add_legend(ax, handles, title="color = ego graph", loc="best", fontsize=9)
    fig.suptitle(f"{label} approximation ratio - facebook", fontsize=18)
    save(fig, pdelete_pivot_path("specific", mode, "facebook.png"))


def plot_specific_compare(df, mode):
    label = metric_name(mode)
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))

    random_df = df[df["graph_family"].eq("random") & df["n"].isin([10, 20, 30])].copy()
    random_edge = mean_table(random_df, ["p_delete", "n"], edge_metric(mode))
    random_comp = mean_table(random_df, ["p_delete", "n"], complete_metric(mode))
    random_handles = []
    for n in [10, 20, 30]:
        e = random_edge[random_edge["n"].eq(n)].sort_values("p_delete")
        c = random_comp[random_comp["n"].eq(n)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        random_handles.append(
            plot_pair(
                axes[0],
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                RANDOM_N_COLORS[n],
                f"random n={n}",
            )
        )
    finish_axis(axes[0], "random all n", f"{label} / all-pairs ILP")
    add_legend(axes[0], random_handles, title="color = graph size", loc="best", fontsize=8)

    clique_df = df[df["graph_family"].eq("clique") & df["n"].isin([10, 20, 30, 100])].copy()
    clique_edge = mean_table(clique_df, ["p_delete", "n"], edge_metric(mode))
    clique_comp = mean_table(clique_df, ["p_delete", "n"], complete_metric(mode))
    clique_handles = []
    for n in [10, 20, 30, 100]:
        e = clique_edge[clique_edge["n"].eq(n)].sort_values("p_delete")
        c = clique_comp[clique_comp["n"].eq(n)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        clique_handles.append(
            plot_pair(
                axes[1],
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                CLIQUE_N_COLORS[n],
                f"clique n={n}",
            )
        )
    finish_axis(axes[1], "clique all n", f"{label} / all-pairs ILP")
    add_legend(axes[1], clique_handles, title="color = graph size", loc="best", fontsize=8)

    facebook_df = df[df["graph_family"].eq("facebook")].copy()
    fb_edge = mean_table(facebook_df, ["p_delete", "ego_label"], edge_metric(mode))
    fb_comp = mean_table(facebook_df, ["p_delete", "ego_label"], complete_metric(mode))
    fb_handles = []
    for ego in sorted(fb_edge["ego_label"].dropna().unique()):
        e = fb_edge[fb_edge["ego_label"].eq(ego)].sort_values("p_delete")
        c = fb_comp[fb_comp["ego_label"].eq(ego)].sort_values("p_delete")
        if e.empty or e["value"].notna().sum() == 0:
            continue
        fb_handles.append(
            plot_pair(
                axes[2],
                e["p_delete"],
                e["value"],
                c["value"] if not c.empty else None,
                FACEBOOK_EGO_COLORS.get(ego, "black"),
                ego,
            )
        )
    finish_axis(axes[2], "facebook ego graphs", f"{label} / all-pairs ILP")
    add_legend(axes[2], fb_handles, title="color = ego graph", loc="best", fontsize=8)

    fig.suptitle(f"{label} approximation ratio - compare")
    save(fig, pdelete_pivot_path("specific", mode, "compare.png"))


def pdelete_key(value):
    value = round(float(value), 2)
    return P_DELETE_ORDER.index(value) if value in P_DELETE_ORDER else 999


def size_complete_baseline(df, mode):
    table = mean_table(df, ["n"], complete_metric(mode))
    table = table[table["value"].notna()]
    return table.sort_values("n")


def plot_size_lines(ax, df, title, mode):
    label = metric_name(mode)
    edge = mean_table(df, ["n", "p_delete"], edge_metric(mode))
    edge = edge[edge["value"].notna()]
    comp = size_complete_baseline(df, mode)

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
    ax.set_ylabel(f"{label} / all-pairs ILP")
    ax.set_xticks(sorted(df["n"].dropna().unique()))
    ax.grid(True, alpha=0.28)


def plot_size_effect(df, mode):
    label = metric_name(mode)
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
        plot_size_lines(ax, sub, title, mode)
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

    fig.suptitle(
        f"Size effect in edge-deleted graphs: {label} approximation ratio",
        fontsize=16,
    )
    save(fig, size_pivot_path(mode))


def main():
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    df = prepare_data(csv_path)

    for mode in ["average", "best"]:
        plot_general_random(df, mode)
        plot_general_clique(df, mode)
        plot_general_facebook(df, mode)
        plot_general_compare(df, mode)

        plot_specific_random(df, mode)
        plot_specific_clique(df, mode)
        plot_specific_facebook(df, mode)
        plot_specific_compare(df, mode)

        plot_size_effect(df, mode)

    print("\nDone. Pivot figures were written to average and best Pivot folders.")


if __name__ == "__main__":
    main()
