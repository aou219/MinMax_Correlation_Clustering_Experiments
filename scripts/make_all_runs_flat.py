import csv
import json
import shutil
from pathlib import Path

ROOT = Path(".")
RESULTS = ROOT / "results"
PROCESSED = RESULTS / "processed"
FLAT = PROCESSED / "all_runs_flat.csv"

EXPECTED_PDELS = ["0.05", "0.15", "0.25", "0.40"]

PROCESSED.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def clean_value(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return json.dumps(x)
    return x


def safe_get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def safe_ratio(a, b):
    try:
        if a is None or b is None or b == 0:
            return None
        return a / b
    except Exception:
        return None


def graph_family(path, shared, exp):
    name = path.name.lower()
    graph_type = str(shared.get("graph_type") or exp.get("graph_params", {}).get("graph_type") or "").lower()

    if "random" in name or graph_type == "random":
        return "random"
    if "clq" in name or "clique" in graph_type:
        return "clique"
    if "facebook" in name or "fb_" in name or "facebook" in graph_type:
        return "facebook"
    return "other"


def expand_experiments(data):
    if isinstance(data, dict) and "experiments" in data:
        return data.get("shared_graph_params", {}), data["experiments"]
    if isinstance(data, list):
        return {}, data
    if isinstance(data, dict):
        return {}, [data]
    return {}, []


def make_row(path, shared, exp, p_key, p_result):
    gp = {}
    gp.update(shared or {})
    gp.update(exp.get("graph_params", {}) or {})

    complete = exp.get("complete_graph", {}) or {}
    edge = (p_result or {}).get("edge_deleted_graph", {}) or {}
    approx = (p_result or {}).get("approximations", {}) or {}

    complete_approx = safe_get(exp, "approximations", "complete_graph") or {}
    if not complete_approx:
        complete_ilp = safe_get(complete, "ilp", "cost")
        complete_approx = {
            "best_pivot_approximation": safe_ratio(safe_get(complete, "pivot", "best_cost"), complete_ilp),
            "average_pivot_approximation": safe_ratio(safe_get(complete, "pivot", "average_cost"), complete_ilp),
            "lp_relaxation_ratio": safe_ratio(safe_get(complete, "lp_relaxation", "cost"), complete_ilp),
            "bad_triangle_primal_ratio": safe_ratio(safe_get(complete, "bad_triangle_lp_bounds", "primal_cost"), complete_ilp),
            "bad_triangle_dual_ratio": safe_ratio(safe_get(complete, "bad_triangle_lp_bounds", "dual_cost"), complete_ilp),
            "min_disjoint_bad_triangle_ratio": safe_ratio(safe_get(complete, "bad_triangles", "min_edge_disjoint_count"), complete_ilp),
            "max_disjoint_bad_triangle_ratio": safe_ratio(safe_get(complete, "bad_triangles", "max_edge_disjoint_count"), complete_ilp),
        }

    edge_ilp_with4 = safe_get(edge, "ilp", "with_4_cycles", "cost")
    edge_approx = approx or {}
    if not edge_approx and edge_ilp_with4 is not None:
        edge_approx = {
            "best_pivot_approximation_with_4_cycles": safe_ratio(safe_get(edge, "pivot", "best_cost"), edge_ilp_with4),
            "average_pivot_approximation_with_4_cycles": safe_ratio(safe_get(edge, "pivot", "average_cost"), edge_ilp_with4),
            "lp_relaxation_ratio_with_4_cycles": safe_ratio(safe_get(edge, "lp_relaxation", "with_4_cycles", "cost"), edge_ilp_with4),
            "bad_triangle_primal_ratio": safe_ratio(safe_get(edge, "bad_triangle_lp_bounds", "primal_cost"), edge_ilp_with4),
            "bad_triangle_dual_ratio": safe_ratio(safe_get(edge, "bad_triangle_lp_bounds", "dual_cost"), edge_ilp_with4),
            "min_disjoint_bad_triangle_ratio": safe_ratio(safe_get(edge, "bad_triangles", "min_edge_disjoint_count"), edge_ilp_with4),
            "max_disjoint_bad_triangle_ratio": safe_ratio(safe_get(edge, "bad_triangles", "max_edge_disjoint_count"), edge_ilp_with4),
        }

    row = {
        "file_name": path.name,
        "file_path": str(path),
        "graph_family": graph_family(path, shared, exp),

        "graph_type": gp.get("graph_type"),
        "n": gp.get("n") or gp.get("num_nodes"),
        "seed": gp.get("seed"),
        "p_delete": p_key,
        "p_positive": gp.get("p_positive"),
        "cluster_sizes": gp.get("cluster_sizes") or gp.get("true_cluster_sizes"),
        "ego_id": gp.get("ego_id"),

        "complete_pivot_best_cost": safe_get(complete, "pivot", "best_cost"),
        "complete_pivot_average_cost": safe_get(complete, "pivot", "average_cost"),
        "complete_bad_triangles_total": safe_get(complete, "bad_triangles", "total_count"),
        "complete_bad_triangles_min_disjoint": safe_get(complete, "bad_triangles", "min_edge_disjoint_count"),
        "complete_bad_triangles_max_disjoint": safe_get(complete, "bad_triangles", "max_edge_disjoint_count"),
        "complete_ilp_cost": safe_get(complete, "ilp", "cost"),
        "complete_lp_cost": safe_get(complete, "lp_relaxation", "cost"),
        "complete_primal_cost": safe_get(complete, "bad_triangle_lp_bounds", "primal_cost"),
        "complete_dual_cost": safe_get(complete, "bad_triangle_lp_bounds", "dual_cost"),

        "complete_best_pivot_approx": complete_approx.get("best_pivot_approximation"),
        "complete_average_pivot_approx": complete_approx.get("average_pivot_approximation"),
        "complete_lp_ratio": complete_approx.get("lp_relaxation_ratio"),
        "complete_bad_triangle_primal_ratio": complete_approx.get("bad_triangle_primal_ratio"),
        "complete_bad_triangle_dual_ratio": complete_approx.get("bad_triangle_dual_ratio"),
        "complete_min_disjoint_bad_triangle_ratio": complete_approx.get("min_disjoint_bad_triangle_ratio"),
        "complete_max_disjoint_bad_triangle_ratio": complete_approx.get("max_disjoint_bad_triangle_ratio"),

        "edge_num_edges_deleted": safe_get(edge, "num_edges_deleted"),
        "edge_pivot_best_cost": safe_get(edge, "pivot", "best_cost"),
        "edge_pivot_average_cost": safe_get(edge, "pivot", "average_cost"),
        "edge_bad_triangles_total": safe_get(edge, "bad_triangles", "total_count"),
        "edge_bad_triangles_min_disjoint": safe_get(edge, "bad_triangles", "min_edge_disjoint_count"),
        "edge_bad_triangles_max_disjoint": safe_get(edge, "bad_triangles", "max_edge_disjoint_count"),

        "edge_ilp_without4_cost": safe_get(edge, "ilp", "without_4_cycles", "cost"),
        "edge_ilp_with4_cost": safe_get(edge, "ilp", "with_4_cycles", "cost"),
        "edge_bad_4_cycles_count": safe_get(edge, "ilp", "with_4_cycles", "bad_4_cycles_count"),
        "same_clustering_4_cycle": safe_get(edge, "ilp", "same_clustering_4_cycle"),

        "edge_lp_without4_cost": safe_get(edge, "lp_relaxation", "without_4_cycles", "cost"),
        "edge_lp_with4_cost": safe_get(edge, "lp_relaxation", "with_4_cycles", "cost"),
        "edge_primal_cost": safe_get(edge, "bad_triangle_lp_bounds", "primal_cost"),
        "edge_dual_cost": safe_get(edge, "bad_triangle_lp_bounds", "dual_cost"),

        "edge_best_pivot_approx_with4": edge_approx.get("best_pivot_approximation_with_4_cycles"),
        "edge_average_pivot_approx_with4": edge_approx.get("average_pivot_approximation_with_4_cycles"),
        "edge_lp_ratio_with4": edge_approx.get("lp_relaxation_ratio_with_4_cycles"),
        "edge_bad_triangle_primal_ratio": edge_approx.get("bad_triangle_primal_ratio"),
        "edge_bad_triangle_dual_ratio": edge_approx.get("bad_triangle_dual_ratio"),
        "edge_min_disjoint_bad_triangle_ratio": edge_approx.get("min_disjoint_bad_triangle_ratio"),
        "edge_max_disjoint_bad_triangle_ratio": edge_approx.get("max_disjoint_bad_triangle_ratio"),

        "runtime_seconds": (p_result or {}).get("runtime_seconds") or exp.get("runtime_seconds"),
    }

    return {k: clean_value(v) for k, v in row.items()}


def main():
    json_files = [
        p for p in RESULTS.rglob("*.json")
        if "processed" not in str(p)
        and "backup" not in str(p).lower()
        and "archive" not in str(p).lower()
        and not p.name.endswith("_all.json")
    ]

    rows = []

    for path in sorted(json_files):
        data = load_json(path)
        shared, experiments = expand_experiments(data)

        for exp in experiments:
            p_delete_results = exp.get("p_delete_results")

            if isinstance(p_delete_results, dict) and p_delete_results:
                for p_key in sorted(p_delete_results.keys(), key=lambda x: float(x)):
                    rows.append(make_row(path, shared, exp, p_key, p_delete_results[p_key]))
            else:
                # fallback oude structuur
                gp = {}
                gp.update(shared or {})
                gp.update(exp.get("graph_params", {}) or {})
                p_key = gp.get("p_delete") or ""
                fake_p_result = {
                    "edge_deleted_graph": exp.get("edge_deleted_graph", {}),
                    "approximations": safe_get(exp, "approximations", "edge_deleted_graph") or exp.get("approximations", {}),
                    "runtime_seconds": exp.get("runtime_seconds"),
                }
                rows.append(make_row(path, shared, exp, p_key, fake_p_result))

    if not rows:
        raise SystemExit("Geen rows gevonden. Check je results folder.")

    if FLAT.exists():
        backup = FLAT.with_suffix(".csv.bak")
        shutil.copy2(FLAT, backup)
        print("Backup old flat:", backup)

    fieldnames = list(rows[0].keys())

    with open(FLAT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        fam = r.get("graph_family", "other")
        counts[fam] = counts.get(fam, 0) + 1

    print("Saved:", FLAT)
    print("Total rows:", len(rows))
    print("Counts:", counts)


if __name__ == "__main__":
    main()
