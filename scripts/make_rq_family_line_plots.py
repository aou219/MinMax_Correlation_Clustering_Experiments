from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FLAT = ROOT / "results" / "processed" / "all_runs_flat.csv"

OUT = ROOT / "results" / "processed" / "plots" / "rq_family_line_plots"
FIG_OUT = ROOT / "figures" / "rq_family_line_plots"

PDELETE_ORDER = [0.05, 0.15, 0.25, 0.40]

PIVOT_COL = "edge_best_pivot_approx_with4"
LP_COL = "edge_lp_ratio_with4"
PIVOT_PLOT_COL = "edge_best_pivot_approx_plot"
LP_PLOT_COL = "edge_lp_ratio_plot"
CLUSTERING_COL = "fourcycle_clustering_changed"

OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def save_figure(fig, filename):
    p1 = OUT / filename
    p2 = FIG_OUT / filename
    fig.tight_layout()
    fig.savefig(p1, dpi=250)
    fig.savefig(p2, dpi=250)
    plt.close(fig)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


def parse_clique_sizes(row):
    raw = str(row.get("cluster_sizes", "")).strip()
    file_name = str(row.get("file_name", "")).strip()

    def parse_text(text):
        text = text.replace(".json", "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return []

        m = re.fullmatch(r"\[?\s*(\d+)\s*x\s*(\d+)\s*\]?", text)
        if m:
            count = int(m.group(1))
            size = int(m.group(2))
            return [size] * count

        nums = [int(x) for x in re.findall(r"\d+", text)]
        return nums

    sizes = parse_text(raw)
    if sizes:
        return sizes

    stem = Path(file_name).stem
    m = re.match(r"clq_n\d+_(.+)", stem)
    if m:
        return parse_text(m.group(1))

    return []


def clique_structure_label(sizes):
    if not sizes:
        return "unknown structure"

    sizes = sorted([int(x) for x in sizes], reverse=True)
    k = len(sizes)
    largest = max(sizes)
    smallest = min(sizes)
    ratio = largest / smallest if smallest else np.nan

    if k == 1:
        return "1 clique"

    if largest == smallest:
        return f"{k} equal cliques"

    if k == 2 and largest - smallest <= 1:
        return "2 near-equal cliques"

    if ratio <= 1.30:
        return f"{k} balanced cliques"

    if ratio <= 2.00:
        return f"{k} mildly imbalanced cliques"

    return f"{k} imbalanced cliques"


def load_data():
    if not FLAT.exists():
        raise SystemExit(
            "Missing results/processed/all_runs_flat.csv. "
            "Run scripts/make_all_runs_flat.py first."
        )

    df = pd.read_csv(FLAT)

    needed_numeric = [
        "n", "seed", "p_delete", "p_positive",
        PIVOT_COL, LP_COL,
        "edge_bad_4_cycles_count",
        "edge_ilp_with4_cost",
        "edge_ilp_without4_cost",
        "edge_pivot_best_cost",
        "edge_lp_with4_cost",
        "complete_ilp_cost",
        "complete_lp_ratio",
        "runtime_seconds",
    ]

    for col in needed_numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = to_num(df[col])

    if "graph_family" not in df.columns:
        df["graph_family"] = ""

    if "same_clustering_4_cycle" not in df.columns:
        df["same_clustering_4_cycle"] = ""

    same = df["same_clustering_4_cycle"].astype(str).str.lower().str.strip()

    df["fourcycle_clustering_known"] = same.isin(["true", "false"])
    df["fourcycle_clustering_changed"] = np.where(
        same.eq("false"),
        1.0,
        np.where(same.eq("true"), 0.0, np.nan),
    )

    # Als ILP-cost 0 is en Pivot/LP ook 0 is, krijg je normaal 0/0 = NaN.
    # Voor de grafiek zetten we dat op 1, omdat de methode dan exact optimaal is.
    df[PIVOT_PLOT_COL] = df[PIVOT_COL]
    zero_ilp = df["edge_ilp_with4_cost"].eq(0)
    zero_pivot = df["edge_pivot_best_cost"].eq(0)
    df.loc[df[PIVOT_PLOT_COL].isna() & zero_ilp & zero_pivot, PIVOT_PLOT_COL] = 1.0

    df[LP_PLOT_COL] = df[LP_COL]
    zero_lp = df["edge_lp_with4_cost"].eq(0)
    df.loc[df[LP_PLOT_COL].isna() & zero_ilp & zero_lp, LP_PLOT_COL] = 1.0

    df["fourcycle_objective_changed"] = np.where(
        df["edge_ilp_with4_cost"].notna() & df["edge_ilp_without4_cost"].notna(),
        (df["edge_ilp_with4_cost"] != df["edge_ilp_without4_cost"]).astype(float),
        np.nan,
    )

    sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_structure"] = sizes.apply(clique_structure_label)
    df["clique_sizes_exact"] = sizes.apply(lambda xs: "-".join(map(str, xs)) if xs else "")

    return df


def aggregate_random(df):
    random_df = df[df["graph_family"].astype(str).str.lower().eq("random")].copy()

    grouped = (
        random_df
        .groupby(["p_delete", "p_positive", "n"], dropna=False)
        .agg(
            runs=("file_name", "count"),
            pivot_approx=(PIVOT_PLOT_COL, "mean"),
            lp_ratio=(LP_PLOT_COL, "mean"),
            clustering_changed_fraction=(CLUSTERING_COL, "mean"),
            objective_changed_fraction=("fourcycle_objective_changed", "mean"),
            bad4_count=("edge_bad_4_cycles_count", "mean"),
        )
        .reset_index()
    )

    return grouped


def aggregate_clique(df):
    clique_df = df[df["graph_family"].astype(str).str.lower().eq("clique")].copy()

    grouped = (
        clique_df
        .groupby(["p_delete", "clique_structure", "n"], dropna=False)
        .agg(
            runs=("file_name", "count"),
            pivot_approx=(PIVOT_PLOT_COL, "mean"),
            lp_ratio=(LP_PLOT_COL, "mean"),
            clustering_changed_fraction=(CLUSTERING_COL, "mean"),
            objective_changed_fraction=("fourcycle_objective_changed", "mean"),
            bad4_count=("edge_bad_4_cycles_count", "mean"),
        )
        .reset_index()
    )

    return grouped


def plot_lines_by_pdelete(
    table,
    family,
    line_col,
    y_col,
    title_y,
    ylabel,
    filename_prefix,
    y_ref=None,
):
    for pdel in PDELETE_ORDER:
        sub = table[np.isclose(table["p_delete"], pdel, equal_nan=False)].copy()

        if sub.empty:
            print(f"Skipping {family} {filename_prefix} p_delete={pdel}: no data")
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))

        if line_col == "p_positive":
            line_values = sorted(sub[line_col].dropna().unique())
        else:
            line_values = sorted(sub[line_col].dropna().astype(str).unique())

        for line_value in line_values:
            if line_col == "p_positive":
                line_sub = sub[np.isclose(sub[line_col], line_value, equal_nan=False)].sort_values("n")
                label = f"p_pos={line_value:.1f}"
            else:
                line_sub = sub[sub[line_col].astype(str).eq(str(line_value))].sort_values("n")
                label = str(line_value)

            if line_sub[y_col].notna().sum() == 0:
                continue

            ax.plot(
                line_sub["n"],
                line_sub[y_col],
                marker="o",
                linewidth=1.8,
                label=label,
            )

        if y_ref is not None:
            ax.axhline(y_ref, linestyle="--", linewidth=1)

        ax.set_title(f"{family}: {title_y} by n, p_delete={pdel:.2f}")
        ax.set_xlabel("n nodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        safe_pdel = str(pdel).replace(".", "")
        filename = f"{filename_prefix}_{family}_pdelete_{safe_pdel}.png"

        save_figure(fig, filename)


def make_random_plots(random_table):
    # 4 grafieken: Pivot approximation, x-as n, lijnen per p_positive.
    plot_lines_by_pdelete(
        table=random_table,
        family="random",
        line_col="p_positive",
        y_col="pivot_approx",
        title_y="Pivot approximation after edge deletion",
        ylabel="Best Pivot / ILP cost",
        filename_prefix="01_pivot_approx",
        y_ref=1,
    )

    # 4 grafieken: LP relaxation ratio, x-as n, lijnen per p_positive.
    plot_lines_by_pdelete(
        table=random_table,
        family="random",
        line_col="p_positive",
        y_col="lp_ratio",
        title_y="LP relaxation ratio after edge deletion",
        ylabel="LP / ILP ratio with 4-cycle constraints",
        filename_prefix="02_lp_ratio",
        y_ref=1,
    )

    # 4 grafieken: hoe vaak 4-cycle constraints de clustering veranderen.
    plot_lines_by_pdelete(
        table=random_table,
        family="random",
        line_col="p_positive",
        y_col="clustering_changed_fraction",
        title_y="4-cycle constraints changed clustering",
        ylabel="Fraction of runs with different clustering",
        filename_prefix="03_fourcycle_clustering_change",
        y_ref=0,
    )


def make_clique_plots(clique_table):
    # 4 grafieken: Pivot approximation, x-as n, lijnen per clique-structuur.
    plot_lines_by_pdelete(
        table=clique_table,
        family="clique",
        line_col="clique_structure",
        y_col="pivot_approx",
        title_y="Pivot approximation after edge deletion",
        ylabel="Best Pivot / ILP cost",
        filename_prefix="04_pivot_approx",
        y_ref=1,
    )

    # 4 grafieken: LP relaxation ratio, x-as n, lijnen per clique-structuur.
    plot_lines_by_pdelete(
        table=clique_table,
        family="clique",
        line_col="clique_structure",
        y_col="lp_ratio",
        title_y="LP relaxation ratio after edge deletion",
        ylabel="LP / ILP ratio with 4-cycle constraints",
        filename_prefix="05_lp_ratio",
        y_ref=1,
    )

    # 4 grafieken: hoe vaak 4-cycle constraints de clustering veranderen.
    plot_lines_by_pdelete(
        table=clique_table,
        family="clique",
        line_col="clique_structure",
        y_col="clustering_changed_fraction",
        title_y="4-cycle constraints changed clustering",
        ylabel="Fraction of runs with different clustering",
        filename_prefix="06_fourcycle_clustering_change",
        y_ref=0,
    )


def main():
    df = load_data()

    random_table = aggregate_random(df)
    clique_table = aggregate_clique(df)

    random_table.to_csv(OUT / "random_line_plot_data.csv", index=False)
    clique_table.to_csv(OUT / "clique_line_plot_data.csv", index=False)

    random_table.to_csv(FIG_OUT / "random_line_plot_data.csv", index=False)
    clique_table.to_csv(FIG_OUT / "clique_line_plot_data.csv", index=False)

    make_random_plots(random_table)
    make_clique_plots(clique_table)

    print("")
    print("Done. Plots and plot-data tables are saved in:")
    print(f"- {OUT}")
    print(f"- {FIG_OUT}")
    print("")
    print("Random plots: 12 total = 3 metrics x 4 p_delete values.")
    print("Clique plots: 12 total = 3 metrics x 4 p_delete values.")


if __name__ == "__main__":
    main()
