from pathlib import Path
from math import comb
import re

import numpy as np
import pandas as pd

ROOT = Path(".")
FLAT = ROOT / "results" / "processed" / "all_runs_flat.csv"

TABLE_DIR = ROOT / "results" / "processed" / "tables" / "final_thesis_tables"
REPORT = ROOT / "results" / "processed" / "reports" / "final_thesis_results_report.md"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ["random", "clique", "facebook"]
PDELETE_ORDER = [0.05, 0.15, 0.25, 0.40]
USE_ONLY_COMPLETE_PDELETE_UNITS = True


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
    df["cost_reduction_fraction"] = (df["complete_ilp_cost"] - df["edge_ilp_with4_cost"]) / df["complete_ilp_cost"]

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


def summarize(g):
    both_costs = g[g["edge_ilp_without4_cost"].notna() & g["edge_ilp_with4_cost"].notna()]
    known = g[g["fourcycle_clustering_known"]]

    return pd.Series({
        "runs": len(g),

        "avg_n": g["n"].mean() if "n" in g.columns else np.nan,
        "avg_complete_ilp_cost": g["complete_ilp_cost"].mean(),
        "avg_edge_ilp_cost": g["edge_ilp_with4_cost"].mean(),
        "avg_complete_cost_per_edge": g["complete_cost_per_edge"].mean(),
        "avg_edge_cost_per_remaining_edge": g["edge_cost_per_remaining_edge"].mean(),
        "avg_normalized_new_over_complete_cost": g["normalized_new_over_complete_cost"].mean(),
        "avg_cost_reduction_fraction": g["cost_reduction_fraction"].mean(),

        "avg_complete_bad_triangles": g["complete_bad_triangles_total"].mean(),
        "avg_edge_bad_triangles": g["edge_bad_triangles_total"].mean(),
        "avg_complete_bad_triangle_density": g["complete_bad_triangle_density"].mean(),
        "avg_edge_bad_triangle_density": g["edge_bad_triangle_density"].mean(),
        "avg_bad_triangle_removed_fraction": g["bad_triangle_removed_fraction"].mean(),

        "avg_bad_4_cycles": g["edge_bad_4_cycles_count"].mean(),
        "avg_bad4_per_1000_remaining_edges": g["bad4_per_1000_remaining_edges"].mean(),
        "avg_fourcycle_cost_gap": g["fourcycle_cost_gap"].mean(),
        "avg_fourcycle_cost_gap_per_remaining_edge": g["fourcycle_cost_gap_per_remaining_edge"].mean(),
        "fourcycle_cost_changed_fraction": both_costs["fourcycle_cost_changed"].mean() if len(both_costs) else np.nan,
        "fourcycle_clustering_changed_fraction": known["fourcycle_clustering_changed"].mean() if len(known) else np.nan,
        "same_cost_different_clustering_fraction": both_costs["same_cost_different_clustering"].mean() if len(both_costs) else np.nan,

        "avg_complete_pivot_ratio": g["complete_best_pivot_approx"].mean(),
        "avg_edge_pivot_ratio": g["edge_best_pivot_approx_with4"].mean(),
        "avg_complete_lp_ratio": g["complete_lp_ratio"].mean(),
        "avg_edge_lp_ratio": g["edge_lp_ratio_with4"].mean(),
        "avg_complete_lp_gap": g["complete_lp_gap"].mean(),
        "avg_edge_lp_gap": g["edge_lp_gap"].mean(),

        "avg_complete_ilp_for_lp": g["complete_ilp_cost"].mean(),
        "avg_complete_lp_cost": g["complete_lp_cost"].mean(),
        "avg_edge_lp_cost": g["edge_lp_with4_cost"].mean(),

        "median_runtime_seconds": g["runtime_seconds"].median(),
        "max_runtime_seconds": g["runtime_seconds"].max(),
    })


def main():
    df = load_data()

    fam = (
        df.groupby(["graph_family", "p_delete_num"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["graph_family", "p_delete_num"])
    )

    size = (
        df.groupby(["graph_family", "n", "p_delete_num"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["graph_family", "n", "p_delete_num"])
    )

    # RQ1: 3 tables
    rq1_cost = fam[[
        "graph_family", "p_delete_num", "runs",
        "avg_complete_ilp_cost", "avg_edge_ilp_cost",
        "avg_complete_cost_per_edge", "avg_edge_cost_per_remaining_edge",
        "avg_normalized_new_over_complete_cost", "avg_cost_reduction_fraction",
    ]].copy()

    rq1_structure = fam[[
        "graph_family", "p_delete_num",
        "avg_complete_bad_triangles", "avg_edge_bad_triangles",
        "avg_complete_bad_triangle_density", "avg_edge_bad_triangle_density",
        "avg_bad_triangle_removed_fraction", "avg_bad4_per_1000_remaining_edges",
    ]].copy()

    rq1_fourcycles = fam[[
        "graph_family", "p_delete_num",
        "avg_bad_4_cycles", "avg_bad4_per_1000_remaining_edges",
        "avg_fourcycle_cost_gap", "avg_fourcycle_cost_gap_per_remaining_edge",
        "fourcycle_cost_changed_fraction",
        "fourcycle_clustering_changed_fraction",
        "same_cost_different_clustering_fraction",
    ]].copy()

    # RQ2: 3 tables
    rq2_family = fam[[
        "graph_family", "p_delete_num", "runs",
        "avg_complete_pivot_ratio", "avg_edge_pivot_ratio",
        "avg_complete_lp_ratio", "avg_edge_lp_ratio",
        "avg_complete_bad_triangle_density", "avg_edge_bad_triangle_density",
        "avg_normalized_new_over_complete_cost",
    ]].copy()

    rq2_size_runtime = size[[
        "graph_family", "n", "p_delete_num", "runs",
        "avg_complete_pivot_ratio", "avg_edge_pivot_ratio",
        "avg_complete_lp_ratio", "avg_edge_lp_ratio",
        "median_runtime_seconds", "max_runtime_seconds",
    ]].copy()

    random_detail = (
        df[df["graph_family"].eq("random")]
        .groupby(["graph_family", "n", "p_positive", "p_delete_num"], dropna=False)
        .apply(summarize)
        .reset_index()
    )

    clique_detail = (
        df[df["graph_family"].eq("clique")]
        .groupby(
            ["graph_family", "n", "clique_sizes_label", "num_cliques", "largest_clique_size", "clique_imbalance_ratio", "p_delete_num"],
            dropna=False,
        )
        .apply(summarize)
        .reset_index()
    )

    facebook_detail = (
        df[df["graph_family"].eq("facebook")]
        .groupby(["graph_family", "n", "ego_id", "p_delete_num"], dropna=False)
        .apply(summarize)
        .reset_index()
    )

    detail_rows = []

    for _, r in random_detail.iterrows():
        detail_rows.append({
            "graph_family": "random",
            "input_description": f"random n={int(r['n'])}, p+={fmt(r['p_positive'])}",
            "n": r["n"],
            "p_delete": r["p_delete_num"],
            "p_positive": r["p_positive"],
            "clique_sizes_label": "",
            "num_cliques": np.nan,
            "largest_clique_size": np.nan,
            "clique_imbalance_ratio": np.nan,
            "ego_id": "",
            "avg_edge_pivot_ratio": r["avg_edge_pivot_ratio"],
            "avg_edge_lp_ratio": r["avg_edge_lp_ratio"],
            "avg_complete_bad_triangle_density": r["avg_complete_bad_triangle_density"],
            "avg_edge_bad_triangle_density": r["avg_edge_bad_triangle_density"],
            "median_runtime_seconds": r["median_runtime_seconds"],
        })

    for _, r in clique_detail.iterrows():
        detail_rows.append({
            "graph_family": "clique",
            "input_description": f"clique {r['clique_sizes_label']}",
            "n": r["n"],
            "p_delete": r["p_delete_num"],
            "p_positive": np.nan,
            "clique_sizes_label": r["clique_sizes_label"],
            "num_cliques": r["num_cliques"],
            "largest_clique_size": r["largest_clique_size"],
            "clique_imbalance_ratio": r["clique_imbalance_ratio"],
            "ego_id": "",
            "avg_edge_pivot_ratio": r["avg_edge_pivot_ratio"],
            "avg_edge_lp_ratio": r["avg_edge_lp_ratio"],
            "avg_complete_bad_triangle_density": r["avg_complete_bad_triangle_density"],
            "avg_edge_bad_triangle_density": r["avg_edge_bad_triangle_density"],
            "median_runtime_seconds": r["median_runtime_seconds"],
        })

    for _, r in facebook_detail.iterrows():
        detail_rows.append({
            "graph_family": "facebook",
            "input_description": f"facebook ego {r.get('ego_id', '')}",
            "n": r["n"],
            "p_delete": r["p_delete_num"],
            "p_positive": np.nan,
            "clique_sizes_label": "",
            "num_cliques": np.nan,
            "largest_clique_size": np.nan,
            "clique_imbalance_ratio": np.nan,
            "ego_id": r.get("ego_id", ""),
            "avg_edge_pivot_ratio": r["avg_edge_pivot_ratio"],
            "avg_edge_lp_ratio": r["avg_edge_lp_ratio"],
            "avg_complete_bad_triangle_density": r["avg_complete_bad_triangle_density"],
            "avg_edge_bad_triangle_density": r["avg_edge_bad_triangle_density"],
            "median_runtime_seconds": r["median_runtime_seconds"],
        })

    rq2_input_details = pd.DataFrame(detail_rows).sort_values(["graph_family", "n", "input_description", "p_delete"])

    # RQ3: 3 tables
    rq3_family = fam[[
        "graph_family", "p_delete_num", "runs",
        "avg_complete_ilp_for_lp", "avg_complete_lp_cost",
        "avg_complete_lp_ratio", "avg_complete_lp_gap",
        "avg_edge_ilp_cost", "avg_edge_lp_cost",
        "avg_edge_lp_ratio", "avg_edge_lp_gap",
    ]].copy()

    rq3_size = size[[
        "graph_family", "n", "p_delete_num", "runs",
        "avg_complete_lp_ratio", "avg_edge_lp_ratio",
        "avg_complete_lp_gap", "avg_edge_lp_gap",
        "avg_complete_ilp_cost", "avg_edge_ilp_cost",
    ]].copy()

    worst_lp = df.copy()
    worst_lp = worst_lp.sort_values("edge_lp_gap", ascending=False).head(30)
    rq3_worst_lp = worst_lp[[
        "graph_family", "file_name", "n", "seed", "p_delete_num",
        "complete_lp_gap", "edge_lp_gap",
        "complete_lp_ratio", "edge_lp_ratio_with4",
        "complete_bad_triangle_density", "edge_bad_triangle_density",
    ]].copy()

    # Save tables
    tables = {
        "rq1_1_cost_effect.csv": rq1_cost,
        "rq1_2_structural_conflicts.csv": rq1_structure,
        "rq1_3_bad4_constraint_effect.csv": rq1_fourcycles,
        "rq2_1_method_performance_by_family.csv": rq2_family,
        "rq2_2_graph_size_and_runtime.csv": rq2_size_runtime,
        "rq2_3_input_structure_details.csv": rq2_input_details,
        "rq3_1_lp_vs_ilp_by_family.csv": rq3_family,
        "rq3_2_lp_vs_ilp_by_size.csv": rq3_size,
        "rq3_3_worst_lp_gap_cases.csv": rq3_worst_lp,
    }

    for name, table in tables.items():
        table.to_csv(TABLE_DIR / name, index=False)
        print("Saved:", TABLE_DIR / name)

    # Write report
    lines = []
    lines.append("# Final thesis results report")
    lines.append("")
    lines.append("This report keeps only the strongest tables for the thesis: three tables per research question.")
    lines.append("")
    lines.append(f"Only complete p_delete units used: **{USE_ONLY_COMPLETE_PDELETE_UNITS}**.")
    lines.append("")

    lines.append("## Final figures to use")
    lines.append("")
    lines.append("1. `01_rq1_normalized_cost.png`")
    lines.append("2. `02_rq1_bad4_density.png`")
    lines.append("3. `03_bad_triangle_density_complete_vs_new.png`")
    lines.append("4. `04_rq2_pivot_complete_vs_new.png`")
    lines.append("5. `05_rq2_runtime_by_size.png`")
    lines.append("6. `06_rq3_lp_gap_complete_vs_new.png`")
    lines.append("")

    lines.append("## RQ1 — Edge deletion")
    lines.append("")
    lines.append("### Table RQ1.1 — Cost effect of edge deletion")
    lines.append("")
    lines.append(md_table(
        rq1_cost,
        [
            "graph_family", "p_delete_num", "runs",
            "avg_complete_ilp_cost", "avg_edge_ilp_cost",
            "avg_complete_cost_per_edge", "avg_edge_cost_per_remaining_edge",
            "avg_normalized_new_over_complete_cost", "avg_cost_reduction_fraction",
        ],
        [
            "family", "p_delete", "runs",
            "complete ILP", "new ILP",
            "complete cost/edge", "new cost/edge",
            "normalized new/complete", "cost reduction",
        ],
        percent_cols={"avg_cost_reduction_fraction"},
    ))
    lines.append("")

    lines.append("### Table RQ1.2 — Structural conflict changes")
    lines.append("")
    lines.append(md_table(
        rq1_structure,
        [
            "graph_family", "p_delete_num",
            "avg_complete_bad_triangles", "avg_edge_bad_triangles",
            "avg_complete_bad_triangle_density", "avg_edge_bad_triangle_density",
            "avg_bad_triangle_removed_fraction", "avg_bad4_per_1000_remaining_edges",
        ],
        [
            "family", "p_delete",
            "bad triangles complete", "bad triangles new",
            "bad triangle density complete", "bad triangle density new",
            "bad triangles removed", "bad4/1000 rem edges",
        ],
        percent_cols={"avg_bad_triangle_removed_fraction"},
    ))
    lines.append("")

    lines.append("### Table RQ1.3 — Bad 4-cycle constraint effect")
    lines.append("")
    lines.append(md_table(
        rq1_fourcycles,
        [
            "graph_family", "p_delete_num",
            "avg_bad_4_cycles", "avg_bad4_per_1000_remaining_edges",
            "avg_fourcycle_cost_gap", "avg_fourcycle_cost_gap_per_remaining_edge",
            "fourcycle_cost_changed_fraction",
            "fourcycle_clustering_changed_fraction",
            "same_cost_different_clustering_fraction",
        ],
        [
            "family", "p_delete",
            "bad 4-cycles", "bad4/1000 rem edges",
            "4cycle cost gap", "4cycle cost gap/edge",
            "cost changed", "clustering changed", "same cost diff clustering",
        ],
        percent_cols={
            "fourcycle_cost_changed_fraction",
            "fourcycle_clustering_changed_fraction",
            "same_cost_different_clustering_fraction",
        },
    ))
    lines.append("")

    lines.append("## RQ2 — Input structure")
    lines.append("")
    lines.append("### Table RQ2.1 — Method performance by graph family")
    lines.append("")
    lines.append(md_table(
        rq2_family,
        [
            "graph_family", "p_delete_num", "runs",
            "avg_complete_pivot_ratio", "avg_edge_pivot_ratio",
            "avg_complete_lp_ratio", "avg_edge_lp_ratio",
            "avg_complete_bad_triangle_density", "avg_edge_bad_triangle_density",
            "avg_normalized_new_over_complete_cost",
        ],
        [
            "family", "p_delete", "runs",
            "Pivot complete", "Pivot new",
            "LP complete", "LP new",
            "bad triangle dens complete", "bad triangle dens new",
            "normalized cost",
        ],
    ))
    lines.append("")

    lines.append("### Table RQ2.2 — Graph size and runtime")
    lines.append("")
    lines.append(md_table(
        rq2_size_runtime,
        [
            "graph_family", "n", "p_delete_num", "runs",
            "avg_complete_pivot_ratio", "avg_edge_pivot_ratio",
            "avg_complete_lp_ratio", "avg_edge_lp_ratio",
            "median_runtime_seconds", "max_runtime_seconds",
        ],
        [
            "family", "n", "p_delete", "runs",
            "Pivot complete", "Pivot new",
            "LP complete", "LP new",
            "median runtime", "max runtime",
        ],
        max_rows=60,
    ))
    lines.append("")
    lines.append("The full table is saved as `rq2_2_graph_size_and_runtime.csv`.")
    lines.append("")

    lines.append("### Table RQ2.3 — Input structure details")
    lines.append("")
    lines.append(md_table(
        rq2_input_details,
        [
            "graph_family", "input_description", "n", "p_delete",
            "p_positive", "clique_sizes_label", "num_cliques",
            "largest_clique_size", "clique_imbalance_ratio", "ego_id",
            "avg_edge_pivot_ratio", "avg_edge_lp_ratio",
            "avg_complete_bad_triangle_density", "avg_edge_bad_triangle_density",
            "median_runtime_seconds",
        ],
        [
            "family", "input", "n", "p_delete",
            "p+", "clique sizes", "# cliques",
            "largest clique", "imbalance", "ego",
            "Pivot new", "LP new",
            "bad triangle dens complete", "bad triangle dens new",
            "median runtime",
        ],
        max_rows=60,
    ))
    lines.append("")
    lines.append("The full table is saved as `rq2_3_input_structure_details.csv`.")
    lines.append("")

    lines.append("## RQ3 — LP vs ILP")
    lines.append("")
    lines.append("### Table RQ3.1 — LP vs ILP by graph family")
    lines.append("")
    lines.append(md_table(
        rq3_family,
        [
            "graph_family", "p_delete_num", "runs",
            "avg_complete_ilp_for_lp", "avg_complete_lp_cost",
            "avg_complete_lp_ratio", "avg_complete_lp_gap",
            "avg_edge_ilp_cost", "avg_edge_lp_cost",
            "avg_edge_lp_ratio", "avg_edge_lp_gap",
        ],
        [
            "family", "p_delete", "runs",
            "complete ILP", "complete LP",
            "LP/ILP complete", "LP gap complete",
            "new ILP", "new LP",
            "LP/ILP new", "LP gap new",
        ],
    ))
    lines.append("")

    lines.append("### Table RQ3.2 — LP vs ILP by graph size")
    lines.append("")
    lines.append(md_table(
        rq3_size,
        [
            "graph_family", "n", "p_delete_num", "runs",
            "avg_complete_lp_ratio", "avg_edge_lp_ratio",
            "avg_complete_lp_gap", "avg_edge_lp_gap",
            "avg_complete_ilp_cost", "avg_edge_ilp_cost",
        ],
        [
            "family", "n", "p_delete", "runs",
            "LP/ILP complete", "LP/ILP new",
            "LP gap complete", "LP gap new",
            "complete ILP", "new ILP",
        ],
        max_rows=60,
    ))
    lines.append("")
    lines.append("The full table is saved as `rq3_2_lp_vs_ilp_by_size.csv`.")
    lines.append("")

    lines.append("### Table RQ3.3 — Worst LP gap cases")
    lines.append("")
    lines.append(md_table(
        rq3_worst_lp,
        [
            "graph_family", "file_name", "n", "seed", "p_delete_num",
            "complete_lp_gap", "edge_lp_gap",
            "complete_lp_ratio", "edge_lp_ratio_with4",
            "complete_bad_triangle_density", "edge_bad_triangle_density",
        ],
        [
            "family", "file", "n", "seed", "p_delete",
            "LP gap complete", "LP gap new",
            "LP/ILP complete", "LP/ILP new",
            "bad triangle dens complete", "bad triangle dens new",
        ],
    ))
    lines.append("")

    lines.append("## Created tables")
    lines.append("")
    for name in tables:
        lines.append(f"- `results/processed/tables/final_thesis_tables/{name}`")

    REPORT.write_text("\n".join(lines))
    print("Saved report:", REPORT)


if __name__ == "__main__":
    main()
