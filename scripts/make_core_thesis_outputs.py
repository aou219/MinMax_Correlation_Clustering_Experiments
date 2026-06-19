from pathlib import Path
from math import comb
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FLAT = ROOT / "results" / "processed" / "all_runs_flat.csv"

REPORT_DIR = ROOT / "results" / "processed" / "reports"
PLOT_DIR = ROOT / "results" / "processed" / "plots" / "core_thesis_figures"
FIG_DIR = ROOT / "figures" / "thesis_plots"
TABLE_DIR = ROOT / "results" / "processed" / "tables" / "core_thesis_tables"
REPORT = REPORT_DIR / "core_thesis_results_report.md"

FAMILY_ORDER = ["random", "clique", "facebook"]
PDELETE_ORDER = [0.05, 0.15, 0.25, 0.40]
USE_ONLY_COMPLETE_PDELETE_UNITS = True


# ============================================================
# Reset only generated output from previous thesis-analysis attempts
# ============================================================

def reset_outputs():
    folders_to_remove = [
        REPORT_DIR,
        ROOT / "results" / "processed" / "plots",
        FIG_DIR,
        TABLE_DIR,
        ROOT / "results" / "processed" / "tables" / "final_thesis_tables",
        ROOT / "results" / "processed" / "tables" / "rq_report",
        ROOT / "results" / "processed" / "tables" / "edge_deletion_main",
    ]

    for path in folders_to_remove:
        if path.exists():
            shutil.rmtree(path)
            print("Deleted:", path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def fmt(x, digits=3):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_pct(x, digits=1):
    try:
        if pd.isna(x):
            return "-"
        return f"{100 * float(x):.{digits}f}%"
    except Exception:
        return str(x)


def md_table(df, cols, labels=None, percent_cols=None, max_rows=None):
    if labels is None:
        labels = cols
    percent_cols = set(percent_cols or [])

    view = df[cols].copy()
    view.columns = labels

    if max_rows is not None:
        view = view.head(max_rows)

    lines = []
    lines.append("| " + " | ".join(view.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(view.columns)) + " |")

    for _, row in view.iterrows():
        vals = []
        for i, v in enumerate(row):
            original = cols[i]
            if isinstance(v, (float, np.floating)):
                vals.append(fmt_pct(v) if original in percent_cols else fmt(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


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


def safe_comb(n, k):
    if pd.isna(n):
        return np.nan
    n = int(n)
    if n < k:
        return np.nan
    return comb(n, k)


def add_derived_columns(df):
    df["complete_edges"] = df["n"].apply(lambda x: safe_comb(x, 2))
    df["remaining_edges"] = df["complete_edges"] - df["edge_num_edges_deleted"]
    df["possible_triangles"] = df["n"].apply(lambda x: safe_comb(x, 3))

    df["complete_cost_per_edge"] = df["complete_ilp_cost"] / df["complete_edges"]
    df["edge_cost_per_remaining_edge"] = df["edge_ilp_with4_cost"] / df["remaining_edges"]
    df["normalized_new_over_complete_cost"] = (
        df["edge_cost_per_remaining_edge"] / df["complete_cost_per_edge"]
    )
    df["cost_reduction_fraction"] = (
        (df["complete_ilp_cost"] - df["edge_ilp_with4_cost"]) / df["complete_ilp_cost"]
    )
    df["cost_ratio_new_over_complete"] = df["edge_ilp_with4_cost"] / df["complete_ilp_cost"]

    df["complete_bad_triangle_density"] = df["complete_bad_triangles_total"] / df["possible_triangles"]
    df["edge_bad_triangle_density"] = df["edge_bad_triangles_total"] / df["possible_triangles"]
    df["bad_triangle_removed_fraction"] = (
        (df["complete_bad_triangles_total"] - df["edge_bad_triangles_total"])
        / df["complete_bad_triangles_total"]
    )

    df["bad4_per_1000_remaining_edges"] = (
        1000 * df["edge_bad_4_cycles_count"] / df["remaining_edges"]
    )

    df["complete_lp_gap"] = 1 - df["complete_lp_ratio"]
    df["edge_lp_gap"] = 1 - df["edge_lp_ratio_with4"]

    df["complete_pivot_excess"] = df["complete_best_pivot_approx"] - 1
    df["edge_pivot_excess"] = df["edge_best_pivot_approx_with4"] - 1

    df["fourcycle_cost_gap"] = df["edge_ilp_with4_cost"] - df["edge_ilp_without4_cost"]
    df["fourcycle_cost_gap_per_remaining_edge"] = df["fourcycle_cost_gap"] / df["remaining_edges"]

    df["fourcycle_cost_changed"] = (
        df["edge_ilp_without4_cost"].notna()
        & df["edge_ilp_with4_cost"].notna()
        & ((df["edge_ilp_with4_cost"] - df["edge_ilp_without4_cost"]).abs() > 1e-9)
    )

    same = df["same_clustering_4_cycle"].astype(str).str.lower().str.strip()
    df["fourcycle_clustering_known"] = same.isin(["true", "false"])
    df["fourcycle_clustering_changed"] = same.eq("false")

    df["same_cost_different_clustering"] = (
        df["edge_ilp_without4_cost"].notna()
        & df["edge_ilp_with4_cost"].notna()
        & ((df["edge_ilp_with4_cost"] - df["edge_ilp_without4_cost"]).abs() <= 1e-9)
        & df["fourcycle_clustering_changed"]
    )

    sizes = df.apply(parse_clique_sizes, axis=1)
    df["clique_sizes_label"] = sizes.apply(lambda xs: "-".join(str(x) for x in xs) if xs else "")
    df["num_cliques"] = sizes.apply(lambda xs: len(xs) if xs else np.nan)
    df["largest_clique_size"] = sizes.apply(lambda xs: max(xs) if xs else np.nan)
    df["smallest_clique_size"] = sizes.apply(lambda xs: min(xs) if xs else np.nan)
    df["clique_imbalance_ratio"] = df["largest_clique_size"] / df["smallest_clique_size"]

    return df


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

    return add_derived_columns(df)


def aggregate(df, keys):
    g = df.groupby(keys, dropna=False)

    out = g.agg(
        runs=("file_name", "count"),
        avg_n=("n", "mean"),

        complete_ilp=("complete_ilp_cost", "mean"),
        new_ilp=("edge_ilp_with4_cost", "mean"),
        cost_ratio=("cost_ratio_new_over_complete", "mean"),
        complete_cost_per_edge=("complete_cost_per_edge", "mean"),
        new_cost_per_edge=("edge_cost_per_remaining_edge", "mean"),
        normalized_cost_ratio=("normalized_new_over_complete_cost", "mean"),
        cost_reduction=("cost_reduction_fraction", "mean"),

        complete_bad_triangles=("complete_bad_triangles_total", "mean"),
        new_bad_triangles=("edge_bad_triangles_total", "mean"),
        complete_bad_triangle_density=("complete_bad_triangle_density", "mean"),
        new_bad_triangle_density=("edge_bad_triangle_density", "mean"),
        bad_triangles_removed=("bad_triangle_removed_fraction", "mean"),

        bad_4_cycles=("edge_bad_4_cycles_count", "mean"),
        bad4_per_1000_remaining_edges=("bad4_per_1000_remaining_edges", "mean"),

        fourcycle_cost_gap=("fourcycle_cost_gap", "mean"),
        fourcycle_cost_gap_per_edge=("fourcycle_cost_gap_per_remaining_edge", "mean"),
        fourcycle_cost_changed=("fourcycle_cost_changed", "mean"),
        same_cost_different_clustering=("same_cost_different_clustering", "mean"),

        pivot_complete=("complete_best_pivot_approx", "mean"),
        pivot_new=("edge_best_pivot_approx_with4", "mean"),
        pivot_excess_new=("edge_pivot_excess", "mean"),

        lp_complete_cost=("complete_lp_cost", "mean"),
        lp_new_cost=("edge_lp_with4_cost", "mean"),
        lp_complete=("complete_lp_ratio", "mean"),
        lp_new=("edge_lp_ratio_with4", "mean"),
        lp_gap_complete=("complete_lp_gap", "mean"),
        lp_gap_new=("edge_lp_gap", "mean"),

        median_runtime=("runtime_seconds", "median"),
        max_runtime=("runtime_seconds", "max"),
    ).reset_index()

    known = df[df["fourcycle_clustering_known"]]
    if len(known):
        known_group = known.groupby(keys, dropna=False).agg(
            fourcycle_clustering_changed=("fourcycle_clustering_changed", "mean")
        ).reset_index()
        out = out.merge(known_group, on=keys, how="left")
    else:
        out["fourcycle_clustering_changed"] = np.nan

    return out


def pcol(prefix, p):
    return f"{prefix}_p{p:.2f}"


def family_wide(base, metrics):
    result = base[["graph_family"]].drop_duplicates().copy()
    for metric, prefix in metrics:
        wide = (
            base.pivot(index="graph_family", columns="p_delete_num", values=metric)
            .reset_index()
        )
        wide.columns = ["graph_family"] + [pcol(prefix, float(c)) for c in wide.columns[1:]]
        for p in PDELETE_ORDER:
            col = pcol(prefix, p)
            if col not in wide.columns:
                wide[col] = np.nan
        result = result.merge(wide[["graph_family"] + [pcol(prefix, p) for p in PDELETE_ORDER]], on="graph_family", how="left")
    return result


def add_complete_family_values(fam_table, cols):
    complete = fam_table.groupby("graph_family", dropna=False).agg(**{
        f"{new_name}": (old_name, "mean") for old_name, new_name in cols
    }).reset_index()
    return complete


def make_tables(df):
    fam = aggregate(df, ["graph_family", "p_delete_num"])
    size_p = aggregate(df, ["graph_family", "n", "p_delete_num"])

    # ============================================================
    # RQ1 — compact tables
    # ============================================================

    rq1_cost = add_complete_family_values(
        fam,
        [
            ("complete_ilp", "complete_ilp_mean"),
            ("complete_cost_per_edge", "complete_cost_per_edge_mean"),
        ],
    )
    rq1_cost = rq1_cost.merge(
        family_wide(
            fam,
            [
                ("new_ilp", "new_ilp"),
                ("cost_ratio", "cost_ratio"),
                ("new_cost_per_edge", "new_cost_per_edge"),
                ("normalized_cost_ratio", "normalized_cost_ratio"),
            ],
        ),
        on="graph_family",
        how="left",
    )

    rq1_structure = add_complete_family_values(
        fam,
        [
            ("complete_bad_triangle_density", "complete_bad_triangle_density_mean"),
        ],
    )
    rq1_structure = rq1_structure.merge(
        family_wide(
            fam,
            [
                ("new_bad_triangle_density", "new_bad_triangle_density"),
                ("bad_triangles_removed", "bad_triangles_removed"),
                ("bad4_per_1000_remaining_edges", "bad4_density"),
            ],
        ),
        on="graph_family",
        how="left",
    )

    rq1_fourcycle = family_wide(
        fam,
        [
            ("bad_4_cycles", "bad4_count"),
            ("fourcycle_cost_changed", "cost_changed"),
            ("fourcycle_clustering_changed", "clustering_changed"),
            ("same_cost_different_clustering", "same_cost_diff_clustering"),
        ],
    )

    # ============================================================
    # RQ2 — compact tables
    # ============================================================

    rq2_family = add_complete_family_values(
        fam,
        [
            ("pivot_complete", "pivot_complete_mean"),
            ("lp_complete", "lp_complete_mean"),
            ("complete_bad_triangle_density", "complete_bad_triangle_density_mean"),
        ],
    )
    rq2_family = rq2_family.merge(
        family_wide(
            fam,
            [
                ("pivot_new", "pivot_new"),
                ("lp_new", "lp_new"),
                ("normalized_cost_ratio", "normalized_cost_ratio"),
                ("median_runtime", "median_runtime"),
            ],
        ),
        on="graph_family",
        how="left",
    )

    size_wide = size_p[["graph_family", "n", "runs"]].groupby(["graph_family", "n"], dropna=False).agg(
        runs=("runs", "sum")
    ).reset_index()

    for metric, prefix in [
        ("pivot_new", "pivot_new"),
        ("lp_new", "lp_new"),
        ("normalized_cost_ratio", "normalized_cost_ratio"),
        ("median_runtime", "median_runtime"),
    ]:
        wide = size_p.pivot(index=["graph_family", "n"], columns="p_delete_num", values=metric).reset_index()
        wide.columns = ["graph_family", "n"] + [pcol(prefix, float(c)) for c in wide.columns[2:]]
        keep = ["graph_family", "n"] + [pcol(prefix, p) for p in [0.05, 0.40] if pcol(prefix, p) in wide.columns]
        size_wide = size_wide.merge(wide[keep], on=["graph_family", "n"], how="left")

    rq2_size = size_wide.sort_values(["graph_family", "n"])

    detail_rows = []

    # random p_positive summary at strongest deletion
    random_040 = df[(df["graph_family"].eq("random")) & (df["p_delete_num"].round(2).eq(0.40))]
    if len(random_040):
        rtab = aggregate(random_040, ["p_positive"])
        for _, r in rtab.iterrows():
            detail_rows.append({
                "input_type": "random",
                "input_setting": f"p+={fmt(r['p_positive'])}",
                "avg_n": r["avg_n"],
                "p_delete": 0.40,
                "pivot_new": r["pivot_new"],
                "lp_new": r["lp_new"],
                "normalized_cost_ratio": r["normalized_cost_ratio"],
                "complete_bad_triangle_density": r["complete_bad_triangle_density"],
                "new_bad_triangle_density": r["new_bad_triangle_density"],
                "median_runtime": r["median_runtime"],
            })

    # clique structure at strongest deletion, summarized by structure
    clique_040 = df[(df["graph_family"].eq("clique")) & (df["p_delete_num"].round(2).eq(0.40))]
    if len(clique_040):
        ctab = aggregate(clique_040, ["clique_sizes_label", "num_cliques", "largest_clique_size", "clique_imbalance_ratio"])
        for _, r in ctab.iterrows():
            detail_rows.append({
                "input_type": "clique",
                "input_setting": f"{r['clique_sizes_label']}",
                "avg_n": r["avg_n"],
                "p_delete": 0.40,
                "pivot_new": r["pivot_new"],
                "lp_new": r["lp_new"],
                "normalized_cost_ratio": r["normalized_cost_ratio"],
                "complete_bad_triangle_density": r["complete_bad_triangle_density"],
                "new_bad_triangle_density": r["new_bad_triangle_density"],
                "median_runtime": r["median_runtime"],
            })

    # facebook ego at strongest deletion
    facebook_040 = df[(df["graph_family"].eq("facebook")) & (df["p_delete_num"].round(2).eq(0.40))]
    if len(facebook_040) and "ego_id" in facebook_040.columns:
        ftab = aggregate(facebook_040, ["ego_id"])
        for _, r in ftab.iterrows():
            detail_rows.append({
                "input_type": "facebook",
                "input_setting": f"ego={r['ego_id']}",
                "avg_n": r["avg_n"],
                "p_delete": 0.40,
                "pivot_new": r["pivot_new"],
                "lp_new": r["lp_new"],
                "normalized_cost_ratio": r["normalized_cost_ratio"],
                "complete_bad_triangle_density": r["complete_bad_triangle_density"],
                "new_bad_triangle_density": r["new_bad_triangle_density"],
                "median_runtime": r["median_runtime"],
            })

    rq2_detail = pd.DataFrame(detail_rows).sort_values(["input_type", "avg_n", "input_setting"])

    # ============================================================
    # RQ3 — compact tables
    # ============================================================

    rq3_family = add_complete_family_values(
        fam,
        [
            ("lp_complete", "lp_complete_mean"),
            ("lp_gap_complete", "lp_gap_complete_mean"),
        ],
    )
    rq3_family = rq3_family.merge(
        family_wide(
            fam,
            [
                ("lp_new", "lp_new"),
                ("lp_gap_new", "lp_gap_new"),
            ],
        ),
        on="graph_family",
        how="left",
    )

    rq3_size = size_wide[["graph_family", "n", "runs"]].copy()
    for metric, prefix in [
        ("lp_new", "lp_new"),
        ("lp_gap_new", "lp_gap_new"),
    ]:
        wide = size_p.pivot(index=["graph_family", "n"], columns="p_delete_num", values=metric).reset_index()
        wide.columns = ["graph_family", "n"] + [pcol(prefix, float(c)) for c in wide.columns[2:]]
        keep = ["graph_family", "n"] + [pcol(prefix, p) for p in [0.05, 0.40] if pcol(prefix, p) in wide.columns]
        rq3_size = rq3_size.merge(wide[keep], on=["graph_family", "n"], how="left")

    worst = df.sort_values("edge_lp_gap", ascending=False).head(12).copy()
    rq3_worst = worst[[
        "graph_family", "file_name", "n", "seed", "p_delete_num",
        "complete_lp_gap", "edge_lp_gap",
        "complete_lp_ratio", "edge_lp_ratio_with4",
        "complete_bad_triangle_density", "edge_bad_triangle_density",
    ]].copy()

    tables = {
        "rq1_1_cost_effect_compact.csv": rq1_cost,
        "rq1_2_structural_conflicts_compact.csv": rq1_structure,
        "rq1_3_bad4_constraint_effect_compact.csv": rq1_fourcycle,
        "rq2_1_method_performance_by_family_compact.csv": rq2_family,
        "rq2_2_graph_size_effect_compact.csv": rq2_size,
        "rq2_3_input_structure_details_p040.csv": rq2_detail,
        "rq3_1_lp_gap_by_family_compact.csv": rq3_family,
        "rq3_2_lp_gap_by_size_compact.csv": rq3_size,
        "rq3_3_worst_lp_gap_cases.csv": rq3_worst,
    }

    for name, table in tables.items():
        table.to_csv(TABLE_DIR / name, index=False)
        print("Saved:", TABLE_DIR / name)

    return tables


def save_fig(fig, filename):
    fig.tight_layout()
    p1 = PLOT_DIR / filename
    p2 = FIG_DIR / filename
    fig.savefig(p1, dpi=250)
    fig.savefig(p2, dpi=250)
    plt.close(fig)
    print("Saved:", p1)
    print("Saved:", p2)


def plot_family_lines(fam, y_col, title, ylabel, filename, y_ref=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for family in FAMILY_ORDER:
        sub = fam[fam["graph_family"].eq(family)].sort_values("p_delete_num")
        if len(sub):
            ax.plot(sub["p_delete_num"], sub[y_col], marker="o", label=family)
    if y_ref is not None:
        ax.axhline(y_ref, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PDELETE_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, filename)


def plot_complete_new(fam, complete_col, new_col, title, ylabel, filename, y_ref=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family in FAMILY_ORDER:
        sub = fam[fam["graph_family"].eq(family)].sort_values("p_delete_num")
        if len(sub):
            ax.plot(sub["p_delete_num"], sub[complete_col], linestyle="--", marker="o", label=f"{family} complete")
            ax.plot(sub["p_delete_num"], sub[new_col], marker="o", label=f"{family} new")
    if y_ref is not None:
        ax.axhline(y_ref, linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("p_delete")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PDELETE_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, filename)


def plot_size_runtime(size):
    fig, ax = plt.subplots(figsize=(8, 5))
    size_mean = size.groupby(["graph_family", "n"], dropna=False).agg(
        median_runtime=("median_runtime", "mean")
    ).reset_index()
    for family in FAMILY_ORDER:
        sub = size_mean[size_mean["graph_family"].eq(family)].sort_values("n")
        if len(sub):
            ax.plot(sub["n"], sub["median_runtime"], marker="o", label=family)
    ax.set_title("RQ2: ILP runtime by graph size")
    ax.set_xlabel("n")
    ax.set_ylabel("Median runtime seconds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, "05_rq2_runtime_by_size.png")


def make_plots(df):
    fam = aggregate(df, ["graph_family", "p_delete_num"])
    size = aggregate(df, ["graph_family", "n", "p_delete_num"])

    plot_family_lines(
        fam,
        "normalized_cost_ratio",
        "RQ1: Normalized cost after edge deletion",
        "Normalized new/complete cost",
        "01_rq1_normalized_cost_ratio.png",
        y_ref=1,
    )

    plot_family_lines(
        fam,
        "bad4_per_1000_remaining_edges",
        "RQ1: Bad 4-cycle density after edge deletion",
        "Bad 4-cycles per 1000 remaining edges",
        "02_rq1_bad4_density.png",
    )

    plot_complete_new(
        fam,
        "complete_bad_triangle_density",
        "new_bad_triangle_density",
        "RQ1/RQ2: Bad triangle density, complete vs new",
        "Bad triangle density",
        "03_bad_triangle_density_complete_vs_new.png",
    )

    plot_complete_new(
        fam,
        "pivot_complete",
        "pivot_new",
        "RQ2: Pivot approximation, complete vs new",
        "Pivot/ILP",
        "04_rq2_pivot_complete_vs_new.png",
        y_ref=1,
    )

    plot_size_runtime(size)

    plot_complete_new(
        fam,
        "lp_gap_complete",
        "lp_gap_new",
        "RQ3: LP gap, complete vs new",
        "LP gap = 1 - LP/ILP",
        "06_rq3_lp_gap_complete_vs_new.png",
        y_ref=0,
    )


def write_report(tables):
    lines = []
    lines.append("# Core thesis results")
    lines.append("")
    lines.append("This report contains only the compact tables selected for the thesis: three tables per research question.")
    lines.append("")
    lines.append("The full CSV versions are saved in `results/processed/tables/core_thesis_tables/`.")
    lines.append("The final plots are saved in `results/processed/plots/core_thesis_figures/` and copied to `figures/thesis_plots/`.")
    lines.append("")
    lines.append(f"Only complete p_delete units used: **{USE_ONLY_COMPLETE_PDELETE_UNITS}**.")
    lines.append("")

    lines.append("## Final figures")
    lines.append("")
    lines.append("1. `01_rq1_normalized_cost_ratio.png`")
    lines.append("2. `02_rq1_bad4_density.png`")
    lines.append("3. `03_bad_triangle_density_complete_vs_new.png`")
    lines.append("4. `04_rq2_pivot_complete_vs_new.png`")
    lines.append("5. `05_rq2_runtime_by_size.png`")
    lines.append("6. `06_rq3_lp_gap_complete_vs_new.png`")
    lines.append("")

    # RQ1
    lines.append("## RQ1 — Edge deletion")
    lines.append("")
    lines.append("### Table RQ1.1 — Cost effect and cost ratios")
    lines.append("")
    rq1 = tables["rq1_1_cost_effect_compact.csv"]
    lines.append(md_table(rq1, list(rq1.columns), percent_cols=[c for c in rq1.columns if c.startswith("cost_reduction")]))
    lines.append("")

    lines.append("### Table RQ1.2 — Structural conflict changes")
    lines.append("")
    rq1s = tables["rq1_2_structural_conflicts_compact.csv"]
    lines.append(md_table(rq1s, list(rq1s.columns), percent_cols=[c for c in rq1s.columns if c.startswith("bad_triangles_removed")]))
    lines.append("")

    lines.append("### Table RQ1.3 — Bad 4-cycle constraint effect")
    lines.append("")
    rq1f = tables["rq1_3_bad4_constraint_effect_compact.csv"]
    percent_cols = [c for c in rq1f.columns if c.startswith("cost_changed") or c.startswith("clustering_changed") or c.startswith("same_cost")]
    lines.append(md_table(rq1f, list(rq1f.columns), percent_cols=percent_cols))
    lines.append("")

    # RQ2
    lines.append("## RQ2 — Input structure")
    lines.append("")
    lines.append("### Table RQ2.1 — Method performance by graph family")
    lines.append("")
    rq2 = tables["rq2_1_method_performance_by_family_compact.csv"]
    lines.append(md_table(rq2, list(rq2.columns)))
    lines.append("")

    lines.append("### Table RQ2.2 — Graph size as input to method output")
    lines.append("")
    rq2s = tables["rq2_2_graph_size_effect_compact.csv"]
    lines.append(md_table(rq2s, list(rq2s.columns), max_rows=30))
    lines.append("")
    lines.append("Full table saved as `rq2_2_graph_size_effect_compact.csv`.")
    lines.append("")

    lines.append("### Table RQ2.3 — Input-structure details at p_delete = 0.40")
    lines.append("")
    rq2d = tables["rq2_3_input_structure_details_p040.csv"]
    lines.append(md_table(rq2d, list(rq2d.columns), max_rows=30))
    lines.append("")
    lines.append("Full table saved as `rq2_3_input_structure_details_p040.csv`.")
    lines.append("")

    # RQ3
    lines.append("## RQ3 — LP vs ILP")
    lines.append("")
    lines.append("### Table RQ3.1 — LP gap by graph family")
    lines.append("")
    rq3 = tables["rq3_1_lp_gap_by_family_compact.csv"]
    lines.append(md_table(rq3, list(rq3.columns)))
    lines.append("")

    lines.append("### Table RQ3.2 — LP gap by graph size")
    lines.append("")
    rq3s = tables["rq3_2_lp_gap_by_size_compact.csv"]
    lines.append(md_table(rq3s, list(rq3s.columns), max_rows=30))
    lines.append("")
    lines.append("Full table saved as `rq3_2_lp_gap_by_size_compact.csv`.")
    lines.append("")

    lines.append("### Table RQ3.3 — Worst LP gap cases")
    lines.append("")
    rq3w = tables["rq3_3_worst_lp_gap_cases.csv"]
    lines.append(md_table(rq3w, list(rq3w.columns)))
    lines.append("")

    lines.append("## Created CSV tables")
    lines.append("")
    for name in tables:
        lines.append(f"- `results/processed/tables/core_thesis_tables/{name}`")

    REPORT.write_text("\n".join(lines))
    print("Saved report:", REPORT)


def main():
    reset_outputs()
    df = load_data()
    tables = make_tables(df)
    make_plots(df)
    write_report(tables)

    print("")
    print("Done. Open these:")
    print("open results/processed/reports/core_thesis_results_report.md")
    print("open results/processed/plots/core_thesis_figures")
    print("open figures/thesis_plots")


if __name__ == "__main__":
    main()
