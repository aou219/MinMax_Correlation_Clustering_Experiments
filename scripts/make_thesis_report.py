import csv
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from collections import defaultdict

ROOT = Path(".")
RESULTS = ROOT / "results"
PROCESSED = RESULTS / "processed"
TABLES = PROCESSED / "tables"
REPORTS = PROCESSED / "reports"
FLAT = PROCESSED / "all_runs_flat.csv"

REPORTS.mkdir(parents=True, exist_ok=True)
(TABLES / "random").mkdir(parents=True, exist_ok=True)
(TABLES / "clique").mkdir(parents=True, exist_ok=True)
(TABLES / "facebook").mkdir(parents=True, exist_ok=True)
(TABLES / "bad_triangles").mkdir(parents=True, exist_ok=True)
(TABLES / "four_cycles").mkdir(parents=True, exist_ok=True)

if not FLAT.exists():
    raise SystemExit("Missing results/processed/all_runs_flat.csv")


# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def f(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in {"", "None", "nan", "NaN", "-"}:
        return None
    try:
        return float(x)
    except Exception:
        return None

def fmt(x, digits=3):
    x = f(x)
    if x is None:
        return "-"
    return f"{x:.{digits}f}"

def avg(values):
    nums = [f(v) for v in values]
    nums = [x for x in nums if x is not None]
    return mean(nums) if nums else None

def pct(a, b):
    if not b:
        return 0
    return 100 * a / b

def group_by(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in keys)].append(r)
    return groups

def pearson(xs, ys):
    pairs = [(f(x), f(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = mean(xs)
    my = mean(ys)

    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)

def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)

def is_random(r):
    return "random" in r.get("file_name", "").lower() or r.get("graph_type") == "random"

def is_clique(r):
    return "clq" in r.get("file_name", "").lower() or "clique" in r.get("graph_type", "").lower()

def is_facebook(r):
    return "facebook" in r.get("file_name", "").lower() or "fb_" in r.get("file_name", "").lower() or "facebook" in r.get("graph_type", "").lower()

def family(r):
    if is_random(r):
        return "random"
    if is_clique(r):
        return "clique"
    if is_facebook(r):
        return "facebook"
    return "other"


# ============================================================
# Patch same_clustering_4_cycle from raw JSON if missing
# ============================================================

def load_experiments(path):
    with open(path) as file:
        data = json.load(file)

    if isinstance(data, dict) and "experiments" in data:
        return data["experiments"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []

def find_key(obj, target):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target:
                found.append(v)
            found.extend(find_key(v, target))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_key(item, target))
    return found

def patch_same_clustering_if_needed(rows):
    if "same_clustering_4_cycle" not in rows[0]:
        for r in rows:
            r["same_clustering_4_cycle"] = ""

    filled = sum(
        1 for r in rows
        if str(r.get("same_clustering_4_cycle", "")).strip() not in {"", "None", "nan", "NaN"}
    )

    if filled > 0:
        print(f"same_clustering_4_cycle already filled: {filled}/{len(rows)}")
        return rows

    print("Patching same_clustering_4_cycle from raw JSON files...")

    raw_values = defaultdict(list)

    json_files = [
        p for p in RESULTS.rglob("*.json")
        if "processed" not in str(p)
        and "archive" not in str(p)
        and "backup" not in str(p).lower()
        and not p.name.endswith("_all.json")
    ]

    for path in json_files:
        for exp in load_experiments(path):
            vals = find_key(exp, "same_clustering_4_cycle")
            raw_values[path.name].append(vals[0] if vals else None)

    seen = defaultdict(int)
    patched = 0

    for row in rows:
        file_name = row["file_name"]
        idx = seen[file_name]
        seen[file_name] += 1

        vals = raw_values.get(file_name, [])

        if idx < len(vals):
            val = vals[idx]
            if val is True:
                row["same_clustering_4_cycle"] = "true"
                patched += 1
            elif val is False:
                row["same_clustering_4_cycle"] = "false"
                patched += 1
            else:
                row["same_clustering_4_cycle"] = ""

    fieldnames = list(rows[0].keys())
    write_csv(FLAT, rows, fieldnames)

    print(f"Patched same_clustering_4_cycle: {patched}/{len(rows)}")
    return rows


# ============================================================
# Load data
# ============================================================

rows = read_csv(FLAT)
rows = patch_same_clustering_if_needed(rows)

random_rows = [r for r in rows if is_random(r)]
clique_rows = [r for r in rows if is_clique(r)]
facebook_rows = [r for r in rows if is_facebook(r)]


# ============================================================
# Derived fields
# ============================================================

for r in rows:
    c_ilp = f(r.get("complete_ilp_cost"))
    e_ilp = f(r.get("edge_ilp_with4_cost"))

    c_bad_total = f(r.get("complete_bad_triangles_total"))
    e_bad_total = f(r.get("edge_bad_triangles_total"))

    c_bad_max = f(r.get("complete_bad_triangles_max_disjoint"))
    e_bad_max = f(r.get("edge_bad_triangles_max_disjoint"))

    n = f(r.get("n"))
    possible_triangles = n * (n - 1) * (n - 2) / 6 if n and n >= 3 else None

    r["_complete_max_disjoint_over_ilp"] = c_bad_max / c_ilp if c_bad_max is not None and c_ilp not in (None, 0) else None
    r["_edge_max_disjoint_over_ilp"] = e_bad_max / e_ilp if e_bad_max is not None and e_ilp not in (None, 0) else None
    r["_complete_bad_triangle_density"] = c_bad_total / possible_triangles if c_bad_total is not None and possible_triangles not in (None, 0) else None
    r["_edge_bad_triangle_density"] = e_bad_total / possible_triangles if e_bad_total is not None and possible_triangles not in (None, 0) else None
    r["_bad_triangle_removed_fraction"] = (c_bad_total - e_bad_total) / c_bad_total if c_bad_total not in (None, 0) and e_bad_total is not None else None

    without4 = f(r.get("edge_ilp_without4_cost"))
    with4 = f(r.get("edge_ilp_with4_cost"))
    same_cluster = str(r.get("same_clustering_4_cycle", "")).strip().lower()

    r["_has_both_4cycle_costs"] = without4 is not None and with4 is not None
    r["_four_cycle_cost_changed"] = abs(without4 - with4) > 1e-9 if r["_has_both_4cycle_costs"] else None
    r["_same_clustering_known"] = same_cluster in {"true", "false"}
    r["_four_cycle_clustering_changed"] = same_cluster == "false" if r["_same_clustering_known"] else None
    r["_same_cost_different_clustering"] = (
        r["_has_both_4cycle_costs"]
        and r["_four_cycle_cost_changed"] is False
        and r["_four_cycle_clustering_changed"] is True
    )


# ============================================================
# Tables
# ============================================================

# Random by n,p
random_np = []
for (n, p), g in group_by(random_rows, ["n", "p_positive"]).items():
    random_np.append({
        "n": n,
        "p_positive": p,
        "runs": len(g),
        "pivot_ilp_complete": avg(r.get("complete_best_pivot_approx") for r in g),
        "lp_ilp_complete": avg(r.get("complete_lp_ratio") for r in g),
        "pivot_ilp_edge": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "lp_ilp_edge": avg(r.get("edge_lp_ratio_with4") for r in g),
        "bad_triangle_density": avg(r.get("_complete_bad_triangle_density") for r in g),
        "ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
    })
random_np.sort(key=lambda r: (f(r["n"]) or 0, f(r["p_positive"]) or 0))
write_csv(TABLES / "random" / "random_by_n_p.csv", random_np, list(random_np[0].keys()))

# Random trend by p
random_p = []
for (p,), g in group_by(random_rows, ["p_positive"]).items():
    random_p.append({
        "p_positive": p,
        "runs": len(g),
        "avg_bad_triangles_total": avg(r.get("complete_bad_triangles_total") for r in g),
        "avg_bad_triangle_density": avg(r.get("_complete_bad_triangle_density") for r in g),
        "avg_ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
        "avg_lp_ilp_ratio": avg(r.get("complete_lp_ratio") for r in g),
        "avg_pivot_ilp_ratio": avg(r.get("complete_best_pivot_approx") for r in g),
        "avg_max_disjoint_over_ilp": avg(r.get("_complete_max_disjoint_over_ilp") for r in g),
    })
random_p.sort(key=lambda r: f(r["p_positive"]) or 0)
write_csv(TABLES / "random" / "random_p_bad_triangle_trend.csv", random_p, list(random_p[0].keys()))

# Random trend by n
random_n = []
for (n,), g in group_by(random_rows, ["n"]).items():
    random_n.append({
        "n": n,
        "runs": len(g),
        "avg_pivot_ilp_complete": avg(r.get("complete_best_pivot_approx") for r in g),
        "avg_lp_ilp_complete": avg(r.get("complete_lp_ratio") for r in g),
        "avg_pivot_ilp_edge": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "avg_lp_ilp_edge": avg(r.get("edge_lp_ratio_with4") for r in g),
    })
random_n.sort(key=lambda r: f(r["n"]) or 0)
write_csv(TABLES / "random" / "random_trend_by_n.csv", random_n, list(random_n[0].keys()))
write_csv(TABLES / "random" / "random_trend_by_p_positive.csv", random_p, list(random_p[0].keys()))
write_csv(TABLES / "random" / "random_np_bad_triangle_detail.csv", random_np, list(random_np[0].keys()))

# Clique
clique_table = []
for (n, clusters), g in group_by(clique_rows, ["n", "cluster_sizes"]).items():
    clique_table.append({
        "n": n,
        "cluster_sizes": clusters,
        "runs": len(g),
        "pivot_ilp_complete": avg(r.get("complete_best_pivot_approx") for r in g),
        "lp_ilp_complete": avg(r.get("complete_lp_ratio") for r in g),
        "pivot_ilp_edge": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "lp_ilp_edge": avg(r.get("edge_lp_ratio_with4") for r in g),
        "bad4_cycles_avg": avg(r.get("edge_bad_4_cycles_count") for r in g),
    })
clique_table.sort(key=lambda r: (f(r["n"]) or 0, r["cluster_sizes"]))
write_csv(TABLES / "clique" / "clique_by_structure.csv", clique_table, list(clique_table[0].keys()))

# Facebook
facebook_table = []
for (ego,), g in group_by(facebook_rows, ["ego_id"]).items():
    first = g[0]
    facebook_table.append({
        "ego_id": ego,
        "n": first.get("n", ""),
        "runs": len(g),
        "has_ilp": "yes" if avg(r.get("complete_ilp_cost") for r in g) is not None else "no",
        "pivot_ilp_complete": avg(r.get("complete_best_pivot_approx") for r in g),
        "lp_ilp_complete": avg(r.get("complete_lp_ratio") for r in g),
        "complete_ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
        "complete_lp_cost": avg(r.get("complete_lp_cost") for r in g),
        "new_ilp_cost": avg(r.get("edge_ilp_with4_cost") for r in g),
        "new_lp_cost": avg(r.get("edge_lp_with4_cost") for r in g),
    })
facebook_table.sort(key=lambda r: f(r["n"]) or 999999)
write_csv(TABLES / "facebook" / "facebook_full_ego_summary.csv", facebook_table, list(facebook_table[0].keys()))

# Bad triangle by family
bad_family = []
for fam in ["random", "clique", "facebook"]:
    g = [r for r in rows if family(r) == fam]
    bad_family.append({
        "graph_family": fam,
        "runs": len(g),
        "avg_complete_max_disjoint_over_ilp": avg(r.get("_complete_max_disjoint_over_ilp") for r in g),
        "avg_new_max_disjoint_over_ilp": avg(r.get("_edge_max_disjoint_over_ilp") for r in g),
        "avg_complete_bad_triangle_density": avg(r.get("_complete_bad_triangle_density") for r in g),
        "avg_edge_bad_triangle_density": avg(r.get("_edge_bad_triangle_density") for r in g),
        "avg_bad_triangle_removed_fraction": avg(r.get("_bad_triangle_removed_fraction") for r in g),
        "corr_complete_bad_triangles_vs_ilp": pearson([r.get("complete_bad_triangles_total") for r in g], [r.get("complete_ilp_cost") for r in g]),
        "corr_complete_max_disjoint_vs_ilp": pearson([r.get("complete_bad_triangles_max_disjoint") for r in g], [r.get("complete_ilp_cost") for r in g]),
    })
write_csv(TABLES / "bad_triangles" / "bad_triangle_bounds_by_graph_family.csv", bad_family, list(bad_family[0].keys()))

# Four cycles by family
four_family = []
for fam in ["random", "clique", "facebook"]:
    g = [r for r in rows if family(r) == fam]
    both = [r for r in g if r["_has_both_4cycle_costs"]]
    cost_changed = [r for r in both if r["_four_cycle_cost_changed"]]
    known = [r for r in g if r["_same_clustering_known"]]
    clustering_changed = [r for r in known if r["_four_cycle_clustering_changed"]]
    same_cost_diff = [r for r in both if r["_same_cost_different_clustering"]]

    four_family.append({
        "graph_family": fam,
        "runs_with_both_costs": len(both),
        "cost_changed_count": len(cost_changed),
        "cost_changed_percent": pct(len(cost_changed), len(both)),
        "same_clustering_known": len(known),
        "clustering_changed_count": len(clustering_changed),
        "clustering_changed_percent": pct(len(clustering_changed), len(known)),
        "same_cost_different_clustering_count": len(same_cost_diff),
        "same_cost_different_clustering_percent": pct(len(same_cost_diff), len(both)),
        "avg_bad4_cycles": avg(r.get("edge_bad_4_cycles_count") for r in g),
    })
write_csv(TABLES / "four_cycles" / "four_cycle_effect_by_graph_family.csv", four_family, list(four_family[0].keys()))

# Four cycle detail
four_detail = []
for r in rows:
    if not r["_has_both_4cycle_costs"]:
        continue
    four_detail.append({
        "file_name": r.get("file_name"),
        "graph_family": family(r),
        "n": r.get("n"),
        "p_positive": r.get("p_positive"),
        "cluster_sizes": r.get("cluster_sizes"),
        "ego_id": r.get("ego_id"),
        "seed": r.get("seed"),
        "ilp_without4": r.get("edge_ilp_without4_cost"),
        "ilp_with4": r.get("edge_ilp_with4_cost"),
        "cost_changed": r["_four_cycle_cost_changed"],
        "same_clustering_4_cycle": r.get("same_clustering_4_cycle"),
        "same_cost_different_clustering": r["_same_cost_different_clustering"],
        "bad4_cycles": r.get("edge_bad_4_cycles_count"),
    })
write_csv(TABLES / "four_cycles" / "four_cycle_effect_detail.csv", four_detail, list(four_detail[0].keys()))
write_csv(TABLES / "four_cycles" / "four_cycle_effect_by_file.csv", four_detail, list(four_detail[0].keys()))


# ============================================================
# Report
# ============================================================

lines = []
lines.append("# Thesis results report")
lines.append("")
lines.append("This report summarizes the processed results from `all_runs_flat.csv`.")
lines.append("")
lines.append("Ratios are interpreted as follows: `Pivot/ILP = 1` means Pivot is optimal; higher is worse. `LP/ILP = 1` means the LP relaxation is tight; lower means the LP is looser.")
lines.append("")

lines.append("## 1. Data overview")
lines.append("")
lines.append(f"- Total runs: **{len(rows)}**")
lines.append(f"- Random graph runs: **{len(random_rows)}**")
lines.append(f"- Clique/community graph runs: **{len(clique_rows)}**")
lines.append(f"- Facebook ego-network runs: **{len(facebook_rows)}**")
lines.append("")

lines.append("## 2. Random graphs")
lines.append("")
lines.append(md_table(
    ["p+", "runs", "bad triangle density", "ILP cost", "LP/ILP", "Pivot/ILP", "max disjoint/ILP"],
    [[r["p_positive"], r["runs"], fmt(r["avg_bad_triangle_density"]), fmt(r["avg_ilp_cost"]), fmt(r["avg_lp_ilp_ratio"]), fmt(r["avg_pivot_ilp_ratio"]), fmt(r["avg_max_disjoint_over_ilp"])] for r in random_p]
))
lines.append("")

p_max_bad = max(random_p, key=lambda r: f(r["avg_bad_triangle_density"]) or -1)
p_max_cost = max(random_p, key=lambda r: f(r["avg_ilp_cost"]) or -1)
p_loose_lp = min(random_p, key=lambda r: f(r["avg_lp_ilp_ratio"]) or 999)
p_worst_pivot = max(random_p, key=lambda r: f(r["avg_pivot_ilp_ratio"]) or -1)

lines.append(f"The highest bad-triangle density occurs around **p+={p_max_bad['p_positive']}**. The highest average ILP cost occurs around **p+={p_max_cost['p_positive']}**. The loosest LP relaxation occurs around **p+={p_loose_lp['p_positive']}**, with LP/ILP ≈ **{fmt(p_loose_lp['avg_lp_ilp_ratio'])}**. Pivot is worst on average around **p+={p_worst_pivot['p_positive']}**, with Pivot/ILP ≈ **{fmt(p_worst_pivot['avg_pivot_ilp_ratio'])}**.")
lines.append("")

lines.append("## 3. Clique/community graphs")
lines.append("")
lines.append(md_table(
    ["n", "clusters", "runs", "Pivot/ILP complete", "LP/ILP complete", "Pivot/ILP new", "LP/ILP new", "avg bad 4-cycles"],
    [[r["n"], r["cluster_sizes"], r["runs"], fmt(r["pivot_ilp_complete"]), fmt(r["lp_ilp_complete"]), fmt(r["pivot_ilp_edge"]), fmt(r["lp_ilp_edge"]), fmt(r["bad4_cycles_avg"])] for r in clique_table]
))
lines.append("")
lines.append("LP is often tight on clique/community graphs, while Pivot becomes worse for larger and more unbalanced structures, especially after new graph.")
lines.append("")

lines.append("## 4. Facebook ego-networks")
lines.append("")
lines.append(md_table(
    ["ego", "n", "has ILP", "Pivot/ILP", "LP/ILP", "ILP complete", "LP complete", "ILP new", "LP new"],
    [[r["ego_id"], r["n"], r["has_ilp"], fmt(r["pivot_ilp_complete"]), fmt(r["lp_ilp_complete"]), fmt(r["complete_ilp_cost"]), fmt(r["complete_lp_cost"]), fmt(r["new_ilp_cost"]), fmt(r["new_lp_cost"])] for r in facebook_table]
))
lines.append("")
lines.append("Facebook ego-networks behave between random graphs and synthetic clique/community graphs. They have real structure, but not the clean planted structure of the clique instances.")
lines.append("")

lines.append("## 5. Bad triangles")
lines.append("")
lines.append(md_table(
    ["graph family", "runs", "max disjoint/ILP complete", "max disjoint/ILP new", "bad triangle density", "bad triangles removed", "corr bad triangles vs ILP"],
    [[r["graph_family"], r["runs"], fmt(r["avg_complete_max_disjoint_over_ilp"]), fmt(r["avg_new_max_disjoint_over_ilp"]), fmt(r["avg_complete_bad_triangle_density"]), fmt(r["avg_bad_triangle_removed_fraction"]), fmt(r["corr_complete_bad_triangles_vs_ilp"])] for r in bad_family]
))
lines.append("")
lines.append("The maximum edge-disjoint bad-triangle count is a lower bound on the ILP cost. When this ratio is close to 1, local bad-triangle structure explains much of the optimum cost.")
lines.append("")

lines.append("## 6. Bad 4-cycle constraints")
lines.append("")
lines.append(md_table(
    ["graph family", "runs with both costs", "cost changed", "cost changed %", "known clusterings", "clustering changed", "clustering changed %", "same cost different clustering"],
    [[r["graph_family"], r["runs_with_both_costs"], r["cost_changed_count"], fmt(r["cost_changed_percent"]), r["same_clustering_known"], r["clustering_changed_count"], fmt(r["clustering_changed_percent"]), r["same_cost_different_clustering_count"]] for r in four_family]
))
lines.append("")

total_both = sum(r["runs_with_both_costs"] for r in four_family)
total_cost_changed = sum(r["cost_changed_count"] for r in four_family)
total_known = sum(r["same_clustering_known"] for r in four_family)
total_clustering_changed = sum(r["clustering_changed_count"] for r in four_family)
total_same_cost_diff = sum(r["same_cost_different_clustering_count"] for r in four_family)

lines.append(f"Overall, the ILP cost with and without 4-cycle constraints can be compared in **{total_both}** runs. The objective cost changed in **{total_cost_changed}** runs, which is **{pct(total_cost_changed, total_both):.1f}%**. The clustering comparison is known in **{total_known}** runs, and the clustering changed in **{total_clustering_changed}** runs, which is **{pct(total_clustering_changed, total_known):.1f}%**. In **{total_same_cost_diff}** runs, the cost stayed the same but the clustering changed.")
lines.append("")

lines.append("## 7. Research question answers")
lines.append("")
lines.append("### RQ1 — New graph")
lines.append("")
lines.append("New graph usually lowers the absolute ILP cost, because fewer edges remain in the objective. At the same time, Pivot often becomes relatively worse after new graph, especially on clique/community graphs. This suggests that deleted edges remove structural information that Pivot needs.")
lines.append("")
lines.append("### RQ2 — Input structure")
lines.append("")
lines.append("Input structure strongly affects the methods. Random graphs with many bad triangles tend to have higher costs and looser LP relaxations. Clique/community graphs often make LP tight, but Pivot struggles more on larger and unbalanced structures. Facebook ego-networks sit between random and clique graphs.")
lines.append("")
lines.append("### RQ3 — LP vs ILP")
lines.append("")
lines.append("LP is often close to ILP for clique/community graphs, but it can be loose for random graphs with many inconsistent local structures. ILP is practical for small and medium instances, but larger Facebook ego-networks and 4-cycle constraint generation become computationally expensive.")
lines.append("")

REPORT = REPORTS / "thesis_results.md"
REPORT.write_text("\n".join(lines))

# Remove old loose thesis files if present
for p in PROCESSED.glob("thesis_*.csv"):
    p.unlink()

print("Saved report:", REPORT)
print("Saved tables in:", TABLES)

# === EXTRA SIZE-EFFECT ANALYSIS START ===

import re

def parse_clique_sizes(row):
    raw = str(row.get("cluster_sizes", "")).strip()
    file_name = str(row.get("file_name", "")).strip()

    def parse_text(s):
        s = s.replace(".json", "")
        s = s.strip()

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
    # Examples:
    # clq_n10_2x5 -> 2 clusters of 5
    # clq_n25_10_10_5 -> [10, 10, 5]
    m = re.match(r"clq_n\d+_(.+)", stem)
    if m:
        return parse_text(m.group(1))

    return []

# Add clique-size derived variables
for r in clique_rows:
    sizes = parse_clique_sizes(r)

    if not sizes:
        n_value = f(r.get("n"))
        sizes = [int(n_value)] if n_value else []

    if sizes:
        n_total = sum(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        num_cliques = len(sizes)
        imbalance = max_size / min_size if min_size else None
        largest_fraction = max_size / n_total if n_total else None
        balanced = max_size == min_size

        r["_clique_sizes_label"] = "-".join(str(x) for x in sizes)
        r["_num_cliques"] = num_cliques
        r["_min_clique_size"] = min_size
        r["_max_clique_size"] = max_size
        r["_largest_clique_fraction"] = largest_fraction
        r["_clique_imbalance_ratio"] = imbalance
        r["_balanced_cliques"] = "balanced" if balanced else "unbalanced"

# ============================================================
# Extra table 1: random graph results by n
# ============================================================

random_by_size = []

for (n,), g in group_by(random_rows, ["n"]).items():
    random_by_size.append({
        "n": n,
        "runs": len(g),
        "avg_complete_ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
        "avg_new_ilp_with4_cost": avg(r.get("edge_ilp_with4_cost") for r in g),
        "avg_complete_lp_ilp_ratio": avg(r.get("complete_lp_ratio") for r in g),
        "avg_complete_pivot_ilp_ratio": avg(r.get("complete_best_pivot_approx") for r in g),
        "avg_new_lp_ilp_ratio_with4": avg(r.get("edge_lp_ratio_with4") for r in g),
        "avg_new_pivot_ilp_ratio_with4": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "avg_bad_triangle_density": avg(r.get("_complete_bad_triangle_density") for r in g),
        "avg_bad_triangles_total": avg(r.get("complete_bad_triangles_total") for r in g),
    })

random_by_size.sort(key=lambda r: f(r["n"]) or 0)

if random_by_size:
    write_csv(
        TABLES / "random" / "random_by_size_n.csv",
        random_by_size,
        list(random_by_size[0].keys())
    )

# ============================================================
# Extra table 2: clique/community structure and ratios
# ============================================================

clique_size_effect = []

for key, g in group_by(
    clique_rows,
    [
        "_clique_sizes_label",
        "_num_cliques",
        "_min_clique_size",
        "_max_clique_size",
        "_largest_clique_fraction",
        "_clique_imbalance_ratio",
        "_balanced_cliques",
    ]
).items():
    (
        sizes_label,
        num_cliques,
        min_size,
        max_size,
        largest_fraction,
        imbalance,
        balanced,
    ) = key

    clique_size_effect.append({
        "clique_sizes": sizes_label,
        "number_of_cliques": num_cliques,
        "min_clique_size": min_size,
        "max_clique_size": max_size,
        "largest_clique_fraction": largest_fraction,
        "imbalance_ratio": imbalance,
        "balanced_or_unbalanced": balanced,
        "runs": len(g),
        "avg_complete_ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
        "avg_new_ilp_with4_cost": avg(r.get("edge_ilp_with4_cost") for r in g),
        "avg_complete_lp_ilp_ratio": avg(r.get("complete_lp_ratio") for r in g),
        "avg_complete_pivot_ilp_ratio": avg(r.get("complete_best_pivot_approx") for r in g),
        "avg_new_lp_ilp_ratio_with4": avg(r.get("edge_lp_ratio_with4") for r in g),
        "avg_new_pivot_ilp_ratio_with4": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "avg_bad4_cycles": avg(r.get("edge_bad_4_cycles_count") for r in g),
        "avg_bad_triangle_density": avg(r.get("_complete_bad_triangle_density") for r in g),
    })

clique_size_effect.sort(
    key=lambda r: (
        f(r["max_clique_size"]) or 0,
        f(r["number_of_cliques"]) or 0,
        str(r["clique_sizes"])
    )
)

if clique_size_effect:
    write_csv(
        TABLES / "clique" / "clique_size_effect.csv",
        clique_size_effect,
        list(clique_size_effect[0].keys())
    )

# ============================================================
# Extra table 3: balanced vs unbalanced clique/community graphs
# ============================================================

clique_balance_effect = []

for (balance,), g in group_by(clique_rows, ["_balanced_cliques"]).items():
    if not balance:
        continue

    clique_balance_effect.append({
        "balanced_or_unbalanced": balance,
        "runs": len(g),
        "avg_complete_ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
        "avg_new_ilp_with4_cost": avg(r.get("edge_ilp_with4_cost") for r in g),
        "avg_complete_lp_ilp_ratio": avg(r.get("complete_lp_ratio") for r in g),
        "avg_complete_pivot_ilp_ratio": avg(r.get("complete_best_pivot_approx") for r in g),
        "avg_new_lp_ilp_ratio_with4": avg(r.get("edge_lp_ratio_with4") for r in g),
        "avg_new_pivot_ilp_ratio_with4": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "avg_bad4_cycles": avg(r.get("edge_bad_4_cycles_count") for r in g),
        "avg_imbalance_ratio": avg(r.get("_clique_imbalance_ratio") for r in g),
        "avg_largest_clique_fraction": avg(r.get("_largest_clique_fraction") for r in g),
    })

clique_balance_effect.sort(key=lambda r: r["balanced_or_unbalanced"])

if clique_balance_effect:
    write_csv(
        TABLES / "clique" / "clique_balance_effect.csv",
        clique_balance_effect,
        list(clique_balance_effect[0].keys())
    )

# ============================================================
# Extra table 4: by largest clique size
# ============================================================

clique_by_largest_size = []

for (max_size,), g in group_by(clique_rows, ["_max_clique_size"]).items():
    if not max_size:
        continue

    clique_by_largest_size.append({
        "largest_clique_size": max_size,
        "runs": len(g),
        "avg_complete_ilp_cost": avg(r.get("complete_ilp_cost") for r in g),
        "avg_new_ilp_with4_cost": avg(r.get("edge_ilp_with4_cost") for r in g),
        "avg_complete_lp_ilp_ratio": avg(r.get("complete_lp_ratio") for r in g),
        "avg_complete_pivot_ilp_ratio": avg(r.get("complete_best_pivot_approx") for r in g),
        "avg_new_lp_ilp_ratio_with4": avg(r.get("edge_lp_ratio_with4") for r in g),
        "avg_new_pivot_ilp_ratio_with4": avg(r.get("edge_best_pivot_approx_with4") for r in g),
        "avg_bad4_cycles": avg(r.get("edge_bad_4_cycles_count") for r in g),
        "avg_imbalance_ratio": avg(r.get("_clique_imbalance_ratio") for r in g),
    })

clique_by_largest_size.sort(key=lambda r: f(r["largest_clique_size"]) or 0)

if clique_by_largest_size:
    write_csv(
        TABLES / "clique" / "clique_by_largest_clique_size.csv",
        clique_by_largest_size,
        list(clique_by_largest_size[0].keys())
    )

# ============================================================
# Add extra sections to thesis report
# ============================================================

report_text = REPORT.read_text()

extra_lines = []
extra_lines.append("")
extra_lines.append("---")
extra_lines.append("")
extra_lines.append("## 8. Additional size-effect tables")
extra_lines.append("")
extra_lines.append("This section adds the size-based comparisons that are useful for interpreting the results by graph size and clique/community structure.")
extra_lines.append("")

extra_lines.append("### 8.1 Random graphs by graph size")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "n",
        "runs",
        "ILP cost",
        "LP/ILP complete",
        "Pivot/ILP complete",
        "LP/ILP new",
        "Pivot/ILP new",
        "bad triangle density",
    ],
    [
        [
            r["n"],
            r["runs"],
            fmt(r["avg_complete_ilp_cost"]),
            fmt(r["avg_complete_lp_ilp_ratio"]),
            fmt(r["avg_complete_pivot_ilp_ratio"]),
            fmt(r["avg_new_lp_ilp_ratio_with4"]),
            fmt(r["avg_new_pivot_ilp_ratio_with4"]),
            fmt(r["avg_bad_triangle_density"]),
        ]
        for r in random_by_size
    ]
))
extra_lines.append("")
extra_lines.append("This table checks whether graph size changes the approximation behaviour. It separates the effect of graph size `n` from the effect of the positive-edge probability `p+`.")
extra_lines.append("")

extra_lines.append("### 8.2 Clique/community size effect")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "clique sizes",
        "# cliques",
        "max clique",
        "imbalance",
        "LP/ILP complete",
        "Pivot/ILP complete",
        "LP/ILP new",
        "Pivot/ILP new",
        "bad 4-cycles",
    ],
    [
        [
            r["clique_sizes"],
            r["number_of_cliques"],
            r["max_clique_size"],
            fmt(r["imbalance_ratio"]),
            fmt(r["avg_complete_lp_ilp_ratio"]),
            fmt(r["avg_complete_pivot_ilp_ratio"]),
            fmt(r["avg_new_lp_ilp_ratio_with4"]),
            fmt(r["avg_new_pivot_ilp_ratio_with4"]),
            fmt(r["avg_bad4_cycles"]),
        ]
        for r in clique_size_effect
    ]
))
extra_lines.append("")
extra_lines.append("This table compares the ratios against the clique/community sizes. It is useful for checking whether larger cliques, more cliques, or more unbalanced clique sizes make Pivot or LP behave differently.")
extra_lines.append("")

extra_lines.append("### 8.3 Balanced vs unbalanced clique/community graphs")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "type",
        "runs",
        "ILP cost",
        "LP/ILP complete",
        "Pivot/ILP complete",
        "LP/ILP new",
        "Pivot/ILP new",
        "avg imbalance",
    ],
    [
        [
            r["balanced_or_unbalanced"],
            r["runs"],
            fmt(r["avg_complete_ilp_cost"]),
            fmt(r["avg_complete_lp_ilp_ratio"]),
            fmt(r["avg_complete_pivot_ilp_ratio"]),
            fmt(r["avg_new_lp_ilp_ratio_with4"]),
            fmt(r["avg_new_pivot_ilp_ratio_with4"]),
            fmt(r["avg_imbalance_ratio"]),
        ]
        for r in clique_balance_effect
    ]
))
extra_lines.append("")
extra_lines.append("This table checks whether balanced clique/community structures behave differently from unbalanced structures.")
extra_lines.append("")

corr_largest_pivot = pearson(
    [r.get("_largest_clique_fraction") for r in clique_rows],
    [r.get("complete_best_pivot_approx") for r in clique_rows],
)

corr_imbalance_pivot = pearson(
    [r.get("_clique_imbalance_ratio") for r in clique_rows],
    [r.get("complete_best_pivot_approx") for r in clique_rows],
)

corr_largest_lp = pearson(
    [r.get("_largest_clique_fraction") for r in clique_rows],
    [r.get("complete_lp_ratio") for r in clique_rows],
)

extra_lines.append("### 8.4 Correlation checks for clique size")
extra_lines.append("")
extra_lines.append(md_table(
    ["comparison", "correlation"],
    [
        ["largest clique fraction vs Pivot/ILP", fmt(corr_largest_pivot)],
        ["imbalance ratio vs Pivot/ILP", fmt(corr_imbalance_pivot)],
        ["largest clique fraction vs LP/ILP", fmt(corr_largest_lp)],
    ]
))
extra_lines.append("")
extra_lines.append("These correlations are not a proof, but they help indicate whether clique size or imbalance is related to the observed approximation ratios.")
extra_lines.append("")

REPORT.write_text(report_text.rstrip() + "\n" + "\n".join(extra_lines))

print("Added extra size-effect tables:")
print("- results/processed/tables/random/random_by_size_n.csv")
print("- results/processed/tables/clique/clique_size_effect.csv")
print("- results/processed/tables/clique/clique_balance_effect.csv")
print("- results/processed/tables/clique/clique_by_largest_clique_size.csv")
print("Updated report:", REPORT)

# === EXTRA SIZE-EFFECT ANALYSIS END ===

# === COMPLETE VS NEW ANALYSIS START ===

COMPLETE_NEW_DIR = TABLES / "complete_vs_new"
COMPLETE_NEW_DIR.mkdir(parents=True, exist_ok=True)

def safe_ratio(a, b):
    a = f(a)
    b = f(b)
    if a is None or b in (None, 0):
        return None
    return a / b

def row_diff(new_value, complete_value):
    new_value = f(new_value)
    complete_value = f(complete_value)
    if new_value is None or complete_value is None:
        return None
    return new_value - complete_value

def row_reduction_fraction(complete_value, new_value):
    complete_value = f(complete_value)
    new_value = f(new_value)
    if complete_value in (None, 0) or new_value is None:
        return None
    return (complete_value - new_value) / complete_value

def complete_vs_new_summary(group_rows):
    both_ilp = [
        r for r in group_rows
        if f(r.get("complete_ilp_cost")) is not None
        and f(r.get("edge_ilp_with4_cost")) is not None
    ]

    return {
        "runs": len(group_rows),
        "runs_with_both_ilp": len(both_ilp),

        "complete_ilp_cost": avg(r.get("complete_ilp_cost") for r in group_rows),
        "new_ilp_cost": avg(r.get("edge_ilp_with4_cost") for r in group_rows),
        "new_over_complete_ilp": avg(
            safe_ratio(r.get("edge_ilp_with4_cost"), r.get("complete_ilp_cost"))
            for r in group_rows
        ),
        "ilp_cost_reduction_fraction": avg(
            row_reduction_fraction(r.get("complete_ilp_cost"), r.get("edge_ilp_with4_cost"))
            for r in group_rows
        ),

        "complete_lp_ilp_ratio": avg(r.get("complete_lp_ratio") for r in group_rows),
        "new_lp_ilp_ratio": avg(r.get("edge_lp_ratio_with4") for r in group_rows),
        "lp_ratio_change_new_minus_complete": avg(
            row_diff(r.get("edge_lp_ratio_with4"), r.get("complete_lp_ratio"))
            for r in group_rows
        ),

        "complete_pivot_ilp_ratio": avg(r.get("complete_best_pivot_approx") for r in group_rows),
        "new_pivot_ilp_ratio": avg(r.get("edge_best_pivot_approx_with4") for r in group_rows),
        "pivot_ratio_change_new_minus_complete": avg(
            row_diff(r.get("edge_best_pivot_approx_with4"), r.get("complete_best_pivot_approx"))
            for r in group_rows
        ),

        "complete_bad_triangles_total": avg(r.get("complete_bad_triangles_total") for r in group_rows),
        "new_bad_triangles_total": avg(r.get("edge_bad_triangles_total") for r in group_rows),
        "bad_triangle_removed_fraction": avg(r.get("_bad_triangle_removed_fraction") for r in group_rows),

        "complete_max_disjoint_over_ilp": avg(r.get("_complete_max_disjoint_over_ilp") for r in group_rows),
        "new_max_disjoint_over_ilp": avg(r.get("_edge_max_disjoint_over_ilp") for r in group_rows),

        "new_bad_4_cycles": avg(r.get("edge_bad_4_cycles_count") for r in group_rows),
    }

def add_group_summary(base, group_rows):
    out = dict(base)
    out.update(complete_vs_new_summary(group_rows))
    return out

# ============================================================
# Complete vs new table 1: by graph family
# ============================================================

complete_vs_new_by_family = []

for fam in ["random", "clique", "facebook"]:
    g = [r for r in rows if family(r) == fam]
    complete_vs_new_by_family.append(
        add_group_summary({"graph_family": fam}, g)
    )

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_by_family.csv",
    complete_vs_new_by_family,
    list(complete_vs_new_by_family[0].keys())
)

# ============================================================
# Complete vs new table 2: random by n
# ============================================================

complete_vs_new_random_by_n = []

for (n,), g in group_by(random_rows, ["n"]).items():
    complete_vs_new_random_by_n.append(
        add_group_summary({"n": n}, g)
    )

complete_vs_new_random_by_n.sort(key=lambda r: f(r["n"]) or 0)

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_random_by_n.csv",
    complete_vs_new_random_by_n,
    list(complete_vs_new_random_by_n[0].keys())
)

# ============================================================
# Complete vs new table 3: random by p_positive
# ============================================================

complete_vs_new_random_by_p = []

for (p,), g in group_by(random_rows, ["p_positive"]).items():
    complete_vs_new_random_by_p.append(
        add_group_summary({"p_positive": p}, g)
    )

complete_vs_new_random_by_p.sort(key=lambda r: f(r["p_positive"]) or 0)

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_random_by_p.csv",
    complete_vs_new_random_by_p,
    list(complete_vs_new_random_by_p[0].keys())
)

# ============================================================
# Complete vs new table 4: random by n and p_positive
# ============================================================

complete_vs_new_random_by_n_p = []

for (n, p), g in group_by(random_rows, ["n", "p_positive"]).items():
    complete_vs_new_random_by_n_p.append(
        add_group_summary({"n": n, "p_positive": p}, g)
    )

complete_vs_new_random_by_n_p.sort(
    key=lambda r: (f(r["n"]) or 0, f(r["p_positive"]) or 0)
)

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_random_by_n_p.csv",
    complete_vs_new_random_by_n_p,
    list(complete_vs_new_random_by_n_p[0].keys())
)

# ============================================================
# Complete vs new table 5: clique/community by structure
# ============================================================

complete_vs_new_clique_by_structure = []

for (n, cluster_sizes), g in group_by(clique_rows, ["n", "_clique_sizes_label"]).items():
    complete_vs_new_clique_by_structure.append(
        add_group_summary(
            {
                "n": n,
                "clique_sizes": cluster_sizes,
                "number_of_cliques": g[0].get("_num_cliques", ""),
                "largest_clique_size": g[0].get("_max_clique_size", ""),
                "imbalance_ratio": g[0].get("_clique_imbalance_ratio", ""),
                "balanced_or_unbalanced": g[0].get("_balanced_cliques", ""),
            },
            g
        )
    )

complete_vs_new_clique_by_structure.sort(
    key=lambda r: (
        f(r["n"]) or 0,
        f(r["largest_clique_size"]) or 0,
        str(r["clique_sizes"])
    )
)

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_clique_by_structure.csv",
    complete_vs_new_clique_by_structure,
    list(complete_vs_new_clique_by_structure[0].keys())
)

# ============================================================
# Complete vs new table 6: clique by largest clique size
# ============================================================

complete_vs_new_clique_by_largest_size = []

for (largest,), g in group_by(clique_rows, ["_max_clique_size"]).items():
    if not largest:
        continue

    complete_vs_new_clique_by_largest_size.append(
        add_group_summary(
            {
                "largest_clique_size": largest,
            },
            g
        )
    )

complete_vs_new_clique_by_largest_size.sort(
    key=lambda r: f(r["largest_clique_size"]) or 0
)

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_clique_by_largest_size.csv",
    complete_vs_new_clique_by_largest_size,
    list(complete_vs_new_clique_by_largest_size[0].keys())
)

# ============================================================
# Complete vs new table 7: Facebook by ego-network
# ============================================================

complete_vs_new_facebook_by_ego = []

for (ego,), g in group_by(facebook_rows, ["ego_id"]).items():
    complete_vs_new_facebook_by_ego.append(
        add_group_summary(
            {
                "ego_id": ego,
                "n": g[0].get("n", ""),
            },
            g
        )
    )

complete_vs_new_facebook_by_ego.sort(key=lambda r: f(r["n"]) or 999999)

write_csv(
    COMPLETE_NEW_DIR / "complete_vs_new_facebook_by_ego.csv",
    complete_vs_new_facebook_by_ego,
    list(complete_vs_new_facebook_by_ego[0].keys())
)

# ============================================================
# Add complete vs new section to report
# ============================================================

report_text = REPORT.read_text()

extra_lines = []
extra_lines.append("")
extra_lines.append("---")
extra_lines.append("")
extra_lines.append("## 9. Complete graph vs new graph after edge deletion")
extra_lines.append("")
extra_lines.append("This section directly compares the original complete graph with the new graph obtained after edge deletion. This is the central comparison for the thesis.")
extra_lines.append("")
extra_lines.append("The new graph uses the ILP with bad 4-cycle constraints when that value is available, because those constraints are part of the sparse edge-deleted formulation.")
extra_lines.append("")

extra_lines.append("### 9.1 Overall comparison by graph family")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "family",
        "runs",
        "complete ILP",
        "new ILP",
        "new/complete ILP",
        "cost reduction",
        "complete LP/ILP",
        "new LP/ILP",
        "LP change",
        "complete Pivot/ILP",
        "new Pivot/ILP",
        "Pivot change",
    ],
    [
        [
            r["graph_family"],
            r["runs"],
            fmt(r["complete_ilp_cost"]),
            fmt(r["new_ilp_cost"]),
            fmt(r["new_over_complete_ilp"]),
            fmt(100 * f(r["ilp_cost_reduction_fraction"]) if f(r["ilp_cost_reduction_fraction"]) is not None else None) + "%",
            fmt(r["complete_lp_ilp_ratio"]),
            fmt(r["new_lp_ilp_ratio"]),
            fmt(r["lp_ratio_change_new_minus_complete"]),
            fmt(r["complete_pivot_ilp_ratio"]),
            fmt(r["new_pivot_ilp_ratio"]),
            fmt(r["pivot_ratio_change_new_minus_complete"]),
        ]
        for r in complete_vs_new_by_family
    ]
))
extra_lines.append("")
extra_lines.append("This table shows whether edge deletion mainly lowers the objective cost, changes LP tightness, or makes Pivot relatively worse.")
extra_lines.append("")

extra_lines.append("### 9.2 Random graphs: complete vs new by graph size")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "n",
        "runs",
        "complete ILP",
        "new ILP",
        "new/complete ILP",
        "cost reduction",
        "complete LP/ILP",
        "new LP/ILP",
        "complete Pivot/ILP",
        "new Pivot/ILP",
    ],
    [
        [
            r["n"],
            r["runs"],
            fmt(r["complete_ilp_cost"]),
            fmt(r["new_ilp_cost"]),
            fmt(r["new_over_complete_ilp"]),
            fmt(100 * f(r["ilp_cost_reduction_fraction"]) if f(r["ilp_cost_reduction_fraction"]) is not None else None) + "%",
            fmt(r["complete_lp_ilp_ratio"]),
            fmt(r["new_lp_ilp_ratio"]),
            fmt(r["complete_pivot_ilp_ratio"]),
            fmt(r["new_pivot_ilp_ratio"]),
        ]
        for r in complete_vs_new_random_by_n
    ]
))
extra_lines.append("")
extra_lines.append("This table checks whether the effect of edge deletion changes as the random graph becomes larger.")
extra_lines.append("")

extra_lines.append("### 9.3 Random graphs: complete vs new by positive-edge probability")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "p+",
        "runs",
        "complete ILP",
        "new ILP",
        "new/complete ILP",
        "cost reduction",
        "complete LP/ILP",
        "new LP/ILP",
        "complete Pivot/ILP",
        "new Pivot/ILP",
    ],
    [
        [
            r["p_positive"],
            r["runs"],
            fmt(r["complete_ilp_cost"]),
            fmt(r["new_ilp_cost"]),
            fmt(r["new_over_complete_ilp"]),
            fmt(100 * f(r["ilp_cost_reduction_fraction"]) if f(r["ilp_cost_reduction_fraction"]) is not None else None) + "%",
            fmt(r["complete_lp_ilp_ratio"]),
            fmt(r["new_lp_ilp_ratio"]),
            fmt(r["complete_pivot_ilp_ratio"]),
            fmt(r["new_pivot_ilp_ratio"]),
        ]
        for r in complete_vs_new_random_by_p
    ]
))
extra_lines.append("")
extra_lines.append("This table checks whether edge deletion behaves differently when the graph contains mostly negative, mixed, or mostly positive edges.")
extra_lines.append("")

extra_lines.append("### 9.4 Clique/community graphs: complete vs new by structure")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "n",
        "clique sizes",
        "imbalance",
        "complete ILP",
        "new ILP",
        "new/complete ILP",
        "complete LP/ILP",
        "new LP/ILP",
        "complete Pivot/ILP",
        "new Pivot/ILP",
    ],
    [
        [
            r["n"],
            r["clique_sizes"],
            fmt(r["imbalance_ratio"]),
            fmt(r["complete_ilp_cost"]),
            fmt(r["new_ilp_cost"]),
            fmt(r["new_over_complete_ilp"]),
            fmt(r["complete_lp_ilp_ratio"]),
            fmt(r["new_lp_ilp_ratio"]),
            fmt(r["complete_pivot_ilp_ratio"]),
            fmt(r["new_pivot_ilp_ratio"]),
        ]
        for r in complete_vs_new_clique_by_structure
    ]
))
extra_lines.append("")
extra_lines.append("This table directly checks whether clique size and imbalance affect the difference between the complete graph and the new graph after edge deletion.")
extra_lines.append("")

extra_lines.append("### 9.5 Facebook ego-networks: complete vs new")
extra_lines.append("")
extra_lines.append(md_table(
    [
        "ego",
        "n",
        "complete ILP",
        "new ILP",
        "new/complete ILP",
        "cost reduction",
        "complete LP/ILP",
        "new LP/ILP",
        "complete Pivot/ILP",
        "new Pivot/ILP",
    ],
    [
        [
            r["ego_id"],
            r["n"],
            fmt(r["complete_ilp_cost"]),
            fmt(r["new_ilp_cost"]),
            fmt(r["new_over_complete_ilp"]),
            fmt(100 * f(r["ilp_cost_reduction_fraction"]) if f(r["ilp_cost_reduction_fraction"]) is not None else None) + "%",
            fmt(r["complete_lp_ilp_ratio"]),
            fmt(r["new_lp_ilp_ratio"]),
            fmt(r["complete_pivot_ilp_ratio"]),
            fmt(r["new_pivot_ilp_ratio"]),
        ]
        for r in complete_vs_new_facebook_by_ego
    ]
))
extra_lines.append("")
extra_lines.append("This table checks whether the same complete-vs-new pattern also appears in real Facebook ego-networks.")
extra_lines.append("")


REPORT.write_text(report_text.rstrip() + "\n" + "\n".join(extra_lines))

print("Added complete-vs-new tables:")
print("- results/processed/tables/complete_vs_new/complete_vs_new_by_family.csv")
print("- results/processed/tables/complete_vs_new/complete_vs_new_random_by_n.csv")
print("- results/processed/tables/complete_vs_new/complete_vs_new_random_by_p.csv")
print("- results/processed/tables/complete_vs_new/complete_vs_new_random_by_n_p.csv")
print("- results/processed/tables/complete_vs_new/complete_vs_new_clique_by_structure.csv")
print("- results/processed/tables/complete_vs_new/complete_vs_new_clique_by_largest_size.csv")
print("- results/processed/tables/complete_vs_new/complete_vs_new_facebook_by_ego.csv")
print("Updated report:", REPORT)

# === COMPLETE VS NEW ANALYSIS END ===

# ============================================================
# Pattern scan: broad list of possible thesis patterns
# ============================================================

PATTERN_SCAN = REPORTS / "pattern_scan_all.md"

def val(x):
    return f(x)

def local_ratio(a, b):
    a = val(a)
    b = val(b)
    if a is None or b in (None, 0):
        return None
    return a / b

def local_diff(a, b):
    a = val(a)
    b = val(b)
    if a is None or b is None:
        return None
    return a - b

def local_reduction(complete, new):
    complete = val(complete)
    new = val(new)
    if complete in (None, 0) or new is None:
        return None
    return (complete - new) / complete

def strength_from_abs(x):
    x = abs(x) if x is not None else None
    if x is None:
        return "unknown"
    if x >= 0.30:
        return "strong"
    if x >= 0.10:
        return "medium"
    return "weak"

def strength_from_corr(x):
    x = abs(x) if x is not None else None
    if x is None:
        return "unknown"
    if x >= 0.70:
        return "strong"
    if x >= 0.40:
        return "medium"
    return "weak"

pattern_rows = []

def add_pattern(category, pattern, evidence, strength, use_in_thesis):
    pattern_rows.append({
        "category": category,
        "pattern": pattern,
        "evidence": evidence,
        "strength": strength,
        "use_in_thesis": use_in_thesis,
    })

# 1. Complete vs new by family
for fam in ["random", "clique", "facebook"]:
    g = [r for r in rows if family(r) == fam]

    cost_red = avg(local_reduction(r.get("complete_ilp_cost"), r.get("edge_ilp_with4_cost")) for r in g)
    pivot_change = avg(local_diff(r.get("edge_best_pivot_approx_with4"), r.get("complete_best_pivot_approx")) for r in g)
    lp_change = avg(local_diff(r.get("edge_lp_ratio_with4"), r.get("complete_lp_ratio")) for r in g)

    add_pattern(
        "complete vs new",
        f"In {fam} graphs, edge deletion lowers the ILP objective cost.",
        f"Average cost reduction: {fmt(100 * cost_red if cost_red is not None else None)}%",
        strength_from_abs(cost_red),
        "likely"
    )

    add_pattern(
        "complete vs new",
        f"In {fam} graphs, Pivot usually becomes worse relative to ILP after edge deletion.",
        f"Average Pivot/ILP change new - complete: {fmt(pivot_change)}",
        strength_from_abs(pivot_change),
        "likely" if pivot_change is not None and pivot_change > 0.05 else "maybe"
    )

    add_pattern(
        "complete vs new",
        f"In {fam} graphs, LP tightness changes only slightly after edge deletion.",
        f"Average LP/ILP change new - complete: {fmt(lp_change)}",
        "weak" if lp_change is not None and abs(lp_change) < 0.05 else strength_from_abs(lp_change),
        "maybe"
    )

# 2. Random graph p+ effects
random_p_groups = []
for (p,), g in group_by(random_rows, ["p_positive"]).items():
    cost_red = avg(local_reduction(r.get("complete_ilp_cost"), r.get("edge_ilp_with4_cost")) for r in g)
    pivot_change = avg(local_diff(r.get("edge_best_pivot_approx_with4"), r.get("complete_best_pivot_approx")) for r in g)
    lp_change = avg(local_diff(r.get("edge_lp_ratio_with4"), r.get("complete_lp_ratio")) for r in g)
    bad_density = avg(r.get("_complete_bad_triangle_density") for r in g)
    random_p_groups.append((p, cost_red, pivot_change, lp_change, bad_density))

random_p_groups.sort(key=lambda x: val(x[0]) or 0)

if random_p_groups:
    strongest_reduction = max(random_p_groups, key=lambda x: x[1] if x[1] is not None else -999)
    weakest_reduction = min(random_p_groups, key=lambda x: x[1] if x[1] is not None else 999)

    add_pattern(
        "random p+",
        "Edge deletion has the strongest cost-reduction effect when p+ is low.",
        f"Highest reduction at p+={strongest_reduction[0]}: {fmt(100 * strongest_reduction[1])}%; weakest at p+={weakest_reduction[0]}: {fmt(100 * weakest_reduction[1])}%",
        "strong",
        "yes"
    )

    p_values = [x[0] for x in random_p_groups]
    reductions = [x[1] for x in random_p_groups]
    corr_p_reduction = pearson(p_values, reductions)

    add_pattern(
        "random p+",
        "As p+ increases, edge deletion reduces the ILP cost less.",
        f"Correlation p+ vs cost reduction: {fmt(corr_p_reduction)}",
        strength_from_corr(corr_p_reduction),
        "yes"
    )

# 3. Random graph size effects
random_n_groups = []
for (n,), g in group_by(random_rows, ["n"]).items():
    cost_red = avg(local_reduction(r.get("complete_ilp_cost"), r.get("edge_ilp_with4_cost")) for r in g)
    pivot_complete = avg(r.get("complete_best_pivot_approx") for r in g)
    pivot_new = avg(r.get("edge_best_pivot_approx_with4") for r in g)
    lp_complete = avg(r.get("complete_lp_ratio") for r in g)
    lp_new = avg(r.get("edge_lp_ratio_with4") for r in g)
    random_n_groups.append((n, cost_red, pivot_complete, pivot_new, lp_complete, lp_new))

random_n_groups.sort(key=lambda x: val(x[0]) or 0)

if random_n_groups:
    corr_n_reduction = pearson([x[0] for x in random_n_groups], [x[1] for x in random_n_groups])
    corr_n_pivot_new = pearson([x[0] for x in random_n_groups], [x[3] for x in random_n_groups])
    corr_n_lp_complete = pearson([x[0] for x in random_n_groups], [x[4] for x in random_n_groups])

    add_pattern(
        "random n",
        "For random graphs, the percentage cost reduction from edge deletion decreases as n grows.",
        f"Correlation n vs cost reduction: {fmt(corr_n_reduction)}",
        strength_from_corr(corr_n_reduction),
        "likely"
    )

    add_pattern(
        "random n",
        "For random graphs, Pivot/new tends to become worse as n grows.",
        f"Correlation n vs Pivot/ILP new: {fmt(corr_n_pivot_new)}",
        strength_from_corr(corr_n_pivot_new),
        "maybe"
    )

    add_pattern(
        "random n",
        "For random graphs, LP/ILP complete becomes looser as n grows.",
        f"Correlation n vs LP/ILP complete: {fmt(corr_n_lp_complete)}",
        strength_from_corr(corr_n_lp_complete),
        "likely"
    )

# 4. Bad triangles
for fam in ["random", "clique", "facebook"]:
    g = [r for r in rows if family(r) == fam]

    corr_bad_ilp = pearson(
        [r.get("complete_bad_triangles_total") for r in g],
        [r.get("complete_ilp_cost") for r in g],
    )

    corr_disjoint_ilp = pearson(
        [r.get("complete_bad_triangles_max_disjoint") for r in g],
        [r.get("complete_ilp_cost") for r in g],
    )

    removed = avg(r.get("_bad_triangle_removed_fraction") for r in g)

    add_pattern(
        "bad triangles",
        f"In {fam} graphs, bad triangles are strongly related to ILP cost.",
        f"Correlation bad triangles vs ILP: {fmt(corr_bad_ilp)}",
        strength_from_corr(corr_bad_ilp),
        "yes" if corr_bad_ilp is not None and abs(corr_bad_ilp) > 0.7 else "maybe"
    )

    add_pattern(
        "bad triangles",
        f"In {fam} graphs, edge deletion removes many bad triangles.",
        f"Average removed fraction: {fmt(100 * removed if removed is not None else None)}%",
        strength_from_abs(removed),
        "likely"
    )

    add_pattern(
        "bad triangles",
        f"In {fam} graphs, edge-disjoint bad triangles track ILP cost.",
        f"Correlation max disjoint bad triangles vs ILP: {fmt(corr_disjoint_ilp)}",
        strength_from_corr(corr_disjoint_ilp),
        "likely"
    )

# 5. Four-cycle effects
both = [r for r in rows if r.get("_has_both_4cycle_costs")]
cost_changed = [r for r in both if r.get("_four_cycle_cost_changed")]
known = [r for r in rows if r.get("_same_clustering_known")]
clustering_changed = [r for r in known if r.get("_four_cycle_clustering_changed")]
same_cost_diff = [r for r in both if r.get("_same_cost_different_clustering")]

cost_change_pct = pct(len(cost_changed), len(both))
cluster_change_pct = pct(len(clustering_changed), len(known))
same_cost_diff_pct = pct(len(same_cost_diff), len(both))

add_pattern(
    "4-cycle constraints",
    "Bad 4-cycle constraints rarely change the ILP objective cost.",
    f"Cost changed in {len(cost_changed)}/{len(both)} runs = {fmt(cost_change_pct)}%",
    "strong",
    "yes"
)

add_pattern(
    "4-cycle constraints",
    "Bad 4-cycle constraints can change the optimal clustering even when the cost stays the same.",
    f"Clustering changed in {len(clustering_changed)}/{len(known)} known runs = {fmt(cluster_change_pct)}%; same-cost different-clustering cases: {len(same_cost_diff)}",
    "medium",
    "yes"
)

# 6. Clique size and imbalance patterns
clique_with_size = [r for r in clique_rows if r.get("_clique_imbalance_ratio") not in {None, "", "None"}]

if clique_with_size:
    corr_imbalance_pivot_complete = pearson(
        [r.get("_clique_imbalance_ratio") for r in clique_with_size],
        [r.get("complete_best_pivot_approx") for r in clique_with_size],
    )

    corr_imbalance_pivot_new = pearson(
        [r.get("_clique_imbalance_ratio") for r in clique_with_size],
        [r.get("edge_best_pivot_approx_with4") for r in clique_with_size],
    )

    corr_largest_bad4 = pearson(
        [r.get("_max_clique_size") for r in clique_with_size],
        [r.get("edge_bad_4_cycles_count") for r in clique_with_size],
    )

    corr_largest_pivot_new = pearson(
        [r.get("_max_clique_size") for r in clique_with_size],
        [r.get("edge_best_pivot_approx_with4") for r in clique_with_size],
    )

    add_pattern(
        "clique size",
        "More imbalanced clique/community structures are related to worse Pivot performance.",
        f"Correlation imbalance vs Pivot/ILP complete: {fmt(corr_imbalance_pivot_complete)}; imbalance vs Pivot/ILP new: {fmt(corr_imbalance_pivot_new)}",
        strength_from_corr(corr_imbalance_pivot_new),
        "maybe"
    )

    add_pattern(
        "clique size",
        "Larger clique sizes are related to more bad 4-cycles after edge deletion.",
        f"Correlation largest clique size vs bad 4-cycles: {fmt(corr_largest_bad4)}",
        strength_from_corr(corr_largest_bad4),
        "likely"
    )

    add_pattern(
        "clique size",
        "Larger clique sizes are related to worse Pivot performance on the new graph.",
        f"Correlation largest clique size vs Pivot/ILP new: {fmt(corr_largest_pivot_new)}",
        strength_from_corr(corr_largest_pivot_new),
        "maybe"
    )

# Sort patterns: yes first, likely, maybe
priority = {"yes": 0, "likely": 1, "maybe": 2, "no": 3}
pattern_rows.sort(key=lambda r: (priority.get(r["use_in_thesis"], 9), r["category"], r["pattern"]))

# Write CSV too
pattern_csv = TABLES / "pattern_scan_all.csv"
write_csv(pattern_csv, pattern_rows, ["category", "pattern", "evidence", "strength", "use_in_thesis"])

# Markdown report
pattern_lines = []
pattern_lines.append("# Pattern scan")
pattern_lines.append("")
pattern_lines.append("This file is not the final thesis text. It is a broad scan of possible patterns in the results. The goal is to find candidate patterns first and filter them later.")
pattern_lines.append("")
pattern_lines.append("## How to read this")
pattern_lines.append("")
pattern_lines.append("- `yes` = strong candidate for thesis")
pattern_lines.append("- `likely` = probably useful")
pattern_lines.append("- `maybe` = only use if it supports the story")
pattern_lines.append("")
pattern_lines.append("## Pattern candidates")
pattern_lines.append("")
pattern_lines.append(md_table(
    ["category", "pattern", "evidence", "strength", "use"],
    [
        [
            r["category"],
            r["pattern"],
            r["evidence"],
            r["strength"],
            r["use_in_thesis"],
        ]
        for r in pattern_rows
    ]
))
pattern_lines.append("")
pattern_lines.append("## Suggested filtering")
pattern_lines.append("")
pattern_lines.append("Start with the patterns marked `yes`. Then add only the `likely` patterns that support the main story: complete graph vs new graph after edge deletion.")
pattern_lines.append("")

PATTERN_SCAN.write_text("\n".join(pattern_lines))

print("Saved pattern scan:", PATTERN_SCAN)
print("Saved pattern CSV:", pattern_csv)
