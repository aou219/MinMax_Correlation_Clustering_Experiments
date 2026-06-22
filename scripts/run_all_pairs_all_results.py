import argparse
import csv
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path


SCRIPT_LOCATION = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_LOCATION.parent
    if SCRIPT_LOCATION.name == "scripts"
    else SCRIPT_LOCATION
)
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

for directory in (SCRIPTS_DIR, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


FAMILY_ORDER = {
    "random": 0,
    "clique": 1,
    "facebook_full_ego": 2,
}

FACEBOOK_ORDER = {
    "3980": 0,
    "698": 1,
    "414": 2,
    "686": 3,
}

CSV_COLUMNS = [
    "edge_all_pairs_ilp_cost",
    "edge_all_pairs_lp_cost",
    "edge_all_pairs_lp_to_ilp_ratio",
    "edge_all_pairs_ilp_runtime_seconds",
    "edge_all_pairs_lp_runtime_seconds",
    "edge_all_pairs_ilp_optimal",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all-pairs LP/ILP for every saved edge-deleted graph."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only show task counts and ordering; do not solve or modify files.",
    )
    parser.add_argument(
        "--csv-every",
        type=int,
        default=25,
        help="Refresh all_runs_flat.csv after this many completed tasks.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def atomic_write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=4)
    with temporary.open(encoding="utf-8") as check_file:
        json.load(check_file)
    os.replace(temporary, path)


def active_result_files():
    patterns = [
        "results/experiments_results_random/**/*.json",
        "results/experiments_results_clique/**/*.json",
        "results/experiments_results_facebook/**/*.json",
    ]
    files = []
    for pattern in patterns:
        files.extend(REPO_ROOT.glob(pattern))
    return sorted(set(files), key=file_sort_key)


def file_metadata(path):
    data = load_json(path)
    shared = data.get("shared_graph_params", {}) if isinstance(data, dict) else {}
    family = str(shared.get("graph_type", ""))
    n = int(shared.get("num_nodes", shared.get("n", 0)) or 0)
    ego_id = str(shared.get("ego_id", ""))
    p_positive = float(shared.get("p_positive", -1) or -1)
    cluster_sizes = tuple(shared.get("cluster_sizes", []) or [])
    return family, n, ego_id, p_positive, cluster_sizes


def file_sort_key(path):
    try:
        family, n, ego_id, p_positive, cluster_sizes = file_metadata(path)
    except Exception:
        return 99, 0, "", (), path.name

    family_index = FAMILY_ORDER.get(family, 99)
    if family == "facebook_full_ego":
        return family_index, FACEBOOK_ORDER.get(ego_id, 99), n, path.name
    if family == "random":
        return family_index, n, p_positive, path.name
    return family_index, n, cluster_sizes, path.name


def experiments_from_data(data):
    if isinstance(data, dict) and "experiments" in data:
        return data.get("shared_graph_params", {}), data["experiments"]
    if isinstance(data, list):
        return {}, data
    if isinstance(data, dict):
        return {}, [data]
    return {}, []


def p_delete_items(experiment):
    results = experiment.get("p_delete_results")
    if isinstance(results, dict) and results:
        return sorted(results.items(), key=lambda item: float(item[0]))

    graph_params = experiment.get("graph_params", {}) or {}
    p_delete = graph_params.get("p_delete")
    if p_delete is None:
        return []
    return [(str(p_delete), experiment)]


def count_tasks(files):
    counts = {"random": 0, "clique": 0, "facebook_full_ego": 0}
    total = 0
    for path in files:
        data = load_json(path)
        shared, experiments = experiments_from_data(data)
        family = str(shared.get("graph_type", ""))
        for experiment in experiments:
            count = len(p_delete_items(experiment))
            counts[family] = counts.get(family, 0) + count
            total += count
    return total, counts


def create_initial_backup(files, csv_path):
    backup_dir = REPO_ROOT / "backups"
    backup_dir.mkdir(exist_ok=True)
    existing = sorted(backup_dir.glob("before_all_pairs_run_*.zip"))
    if existing:
        print(f"Backup already exists: {existing[-1].relative_to(REPO_ROOT)}", flush=True)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = backup_dir / f"before_all_pairs_run_{timestamp}.zip"
    backup_files = list(files)
    if csv_path.exists():
        backup_files.append(csv_path)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in backup_files:
            archive.write(path, path.relative_to(REPO_ROOT))

    with zipfile.ZipFile(output, "r") as archive:
        broken = archive.testzip()
    if broken is not None:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Backup verification failed at {broken}")
    print(f"Backup created: {output.relative_to(REPO_ROOT)}", flush=True)


def all_facebook_nodes(edge_nodes, circles):
    circle_nodes = set()
    for circle in circles:
        circle_nodes.update(circle["nodes"])
    return sorted(edge_nodes | circle_nodes)


def build_complete_graph(shared, experiment):
    from facebook_sampling import (
        build_complete_signed_matrix_from_facebook_sample,
        load_facebook_circles,
        load_facebook_ego_edges,
    )
    from graph_generation import (
        generate_clique_signed_graph,
        generate_signed_complete_graph,
    )

    graph_params = {}
    graph_params.update(shared or {})
    graph_params.update(experiment.get("graph_params", {}) or {})
    family = str(graph_params.get("graph_type", shared.get("graph_type", "")))
    seed = int(graph_params["seed"])

    if family == "random":
        n = int(graph_params.get("num_nodes", graph_params.get("n")))
        p_positive = float(graph_params["p_positive"])
        matrix = generate_signed_complete_graph(
            n=n,
            p_positive=p_positive,
            seed=seed,
        )
        details = f"n={n}, p_positive={p_positive}, seed={seed}"
        return matrix, family, seed, details

    if family == "clique":
        cluster_sizes = list(graph_params["cluster_sizes"])
        p_inside = float(graph_params.get("p_pos_inside", 0.9))
        p_between = float(graph_params.get("p_pos_between", 0.1))
        matrix, _ = generate_clique_signed_graph(
            cluster_sizes=cluster_sizes,
            p_pos_inside=p_inside,
            p_pos_between=p_between,
            seed=seed,
        )
        details = (
            f"n={sum(cluster_sizes)}, cluster_sizes={cluster_sizes}, "
            f"p_inside={p_inside}, p_between={p_between}, seed={seed}"
        )
        return matrix, family, seed, details

    if family == "facebook_full_ego":
        ego_id = str(graph_params["ego_id"])
        edges_path = REPO_ROOT / "data" / "facebook" / f"{ego_id}.edges"
        circles_path = REPO_ROOT / "data" / "facebook" / f"{ego_id}.circles"
        edge_nodes, facebook_edges = load_facebook_ego_edges(str(edges_path))
        circles = load_facebook_circles(str(circles_path))
        nodes = all_facebook_nodes(edge_nodes, circles)
        matrix, _, _, _ = build_complete_signed_matrix_from_facebook_sample(
            nodes,
            facebook_edges,
        )
        details = f"n={len(nodes)}, ego_id={ego_id}, seed={seed}"
        return matrix, family, seed, details

    raise ValueError(f"Unsupported graph type: {family}")


def edge_result_from_p_result(p_result):
    if isinstance(p_result, dict) and "edge_deleted_graph" in p_result:
        return p_result["edge_deleted_graph"]
    if isinstance(p_result, dict):
        return p_result.setdefault("edge_deleted_graph", {})
    raise ValueError("Invalid p_delete result")


def existing_cost(edge_result, key):
    value = edge_result.get(key)
    if isinstance(value, dict) and value.get("cost") is not None:
        return float(value["cost"])
    return None


def save_solve_result(edge_result, key, cost, info):
    edge_result[key] = {
        "cost": float(cost),
        "optimal": bool(info["is_optimal"]),
        "runtime_seconds": float(info["runtime_seconds"]),
    }
    if info.get("mip_gap") is not None:
        edge_result[key]["mip_gap"] = float(info["mip_gap"])


def mark_ego_686_ilp_skipped(edge_result):
    if "all_pairs_ilp" not in edge_result:
        edge_result["all_pairs_ilp"] = {
            "status": "skipped",
            "reason": "ILP intentionally skipped for Facebook ego 686; LP only.",
        }


def update_ratio(p_result, edge_result):
    lp_cost = existing_cost(edge_result, "all_pairs_lp_relaxation")
    ilp_cost = existing_cost(edge_result, "all_pairs_ilp")
    approximations = p_result.setdefault("approximations", {})
    if not isinstance(approximations, dict):
        approximations = {}
        p_result["approximations"] = approximations
    if lp_cost is None or ilp_cost is None or ilp_cost == 0:
        approximations["all_pairs_lp_to_ilp_ratio"] = None
    else:
        approximations["all_pairs_lp_to_ilp_ratio"] = lp_cost / ilp_cost


def normalized_cluster_sizes(value):
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return json.dumps(parsed, separators=(",", ":"))
    except Exception:
        pass
    return str(value).replace(" ", "")


def normalized_number(value):
    if value in (None, ""):
        return ""
    return f"{float(value):.10g}"


def task_key(family, n, seed, p_delete, p_positive="", cluster_sizes="", ego_id=""):
    return (
        str(family),
        str(int(float(n))),
        str(int(float(seed))),
        normalized_number(p_delete),
        normalized_number(p_positive),
        normalized_cluster_sizes(cluster_sizes),
        str(ego_id or "").replace(".0", ""),
    )


def collect_all_pairs_results(files):
    results = {}
    for path in files:
        data = load_json(path)
        shared, experiments = experiments_from_data(data)
        for experiment in experiments:
            params = {}
            params.update(shared or {})
            params.update(experiment.get("graph_params", {}) or {})
            family = params.get("graph_type", "")
            n = params.get("num_nodes", params.get("n", ""))
            seed = params.get("seed", "")
            p_positive = params.get("p_positive", "")
            cluster_sizes = params.get("cluster_sizes", "")
            ego_id = params.get("ego_id", "")
            for p_key, p_result in p_delete_items(experiment):
                edge_result = edge_result_from_p_result(p_result)
                ilp = edge_result.get("all_pairs_ilp", {})
                lp = edge_result.get("all_pairs_lp_relaxation", {})
                approximations = p_result.get("approximations", {}) or {}
                key = task_key(
                    family, n, seed, p_key, p_positive, cluster_sizes, ego_id
                )
                results[key] = {
                    "edge_all_pairs_ilp_cost": ilp.get("cost", ""),
                    "edge_all_pairs_lp_cost": lp.get("cost", ""),
                    "edge_all_pairs_lp_to_ilp_ratio": approximations.get(
                        "all_pairs_lp_to_ilp_ratio", ""
                    ),
                    "edge_all_pairs_ilp_runtime_seconds": ilp.get(
                        "runtime_seconds", ""
                    ),
                    "edge_all_pairs_lp_runtime_seconds": lp.get(
                        "runtime_seconds", ""
                    ),
                    "edge_all_pairs_ilp_optimal": ilp.get("optimal", ""),
                }
    return results


def update_flat_csv(files):
    csv_path = REPO_ROOT / "results" / "processed" / "all_runs_flat.csv"
    if not csv_path.exists():
        print("CSV not found; JSON checkpoints are safe, CSV update skipped.", flush=True)
        return

    result_map = collect_all_pairs_results(files)
    with csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for column in CSV_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    matched = 0
    for row in rows:
        key = task_key(
            row.get("graph_type", ""),
            row.get("n", ""),
            row.get("seed", ""),
            row.get("p_delete", ""),
            row.get("p_positive", ""),
            row.get("cluster_sizes", ""),
            row.get("ego_id", ""),
        )
        values = result_map.get(key)
        if values is None:
            continue
        row.update(values)
        matched += 1

    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    print(f"CSV refreshed: {matched}/{len(rows)} rows matched.", flush=True)


def describe_order(files):
    print("Execution order:", flush=True)
    for path in files:
        family, n, ego_id, p_positive, cluster_sizes = file_metadata(path)
        if family == "random":
            label = f"random n={n}, p_positive={p_positive}"
        elif family == "clique":
            label = f"clique n={n}, cluster_sizes={list(cluster_sizes)}"
        else:
            suffix = " (LP only)" if ego_id == "686" else ""
            label = f"facebook ego={ego_id}, n={n}{suffix}"
        print(f"- {label}: {path.relative_to(REPO_ROOT)}", flush=True)


def main():
    args = parse_args()
    files = active_result_files()
    total, counts = count_tasks(files)
    print(f"Total edge-deleted runs: {total}", flush=True)
    print(
        f"random={counts.get('random', 0)}, "
        f"clique={counts.get('clique', 0)}, "
        f"facebook={counts.get('facebook_full_ego', 0)}",
        flush=True,
    )
    describe_order(files)
    if args.list_only:
        return

    from all_pairs_solver import solve_all_pairs
    from edge_deletion import delete_edges

    csv_path = REPO_ROOT / "results" / "processed" / "all_runs_flat.csv"
    create_initial_backup(files, csv_path)
    completed = 0

    for path in files:
        data = load_json(path)
        shared, experiments = experiments_from_data(data)
        experiments.sort(key=lambda exp: int(exp.get("graph_params", {}).get("seed", 0)))

        for experiment in experiments:
            complete, family, seed, details = build_complete_graph(shared, experiment)
            ego_id = str((shared or {}).get("ego_id", ""))

            for p_key, p_result in p_delete_items(experiment):
                p_delete = float(p_key)
                completed += 1
                edge_result = edge_result_from_p_result(p_result)
                incomplete, _ = delete_edges(complete, p_delete=p_delete, seed=seed)

                print("", flush=True)
                print(
                    f"[{completed}/{total}] {family}: {details}, p_delete={p_delete}",
                    flush=True,
                )

                lp_cost = existing_cost(edge_result, "all_pairs_lp_relaxation")
                if lp_cost is None:
                    print("  Solving all-pairs LP...", flush=True)
                    lp_cost, _, lp_info = solve_all_pairs(
                        incomplete,
                        verbose=False,
                        relax=True,
                        return_x_values=False,
                    )
                    save_solve_result(
                        edge_result,
                        "all_pairs_lp_relaxation",
                        lp_cost,
                        lp_info,
                    )
                    atomic_write_json(path, data)
                    print(
                        f"  LP={lp_cost:.6g} "
                        f"({lp_info['runtime_seconds']:.2f}s) saved",
                        flush=True,
                    )
                else:
                    print(f"  LP={lp_cost:.6g} already saved", flush=True)

                if family == "facebook_full_ego" and ego_id == "686":
                    mark_ego_686_ilp_skipped(edge_result)
                    update_ratio(p_result, edge_result)
                    atomic_write_json(path, data)
                    print("  ILP skipped for ego 686 (LP only)", flush=True)
                else:
                    ilp_cost = existing_cost(edge_result, "all_pairs_ilp")
                    if ilp_cost is None:
                        print("  Solving all-pairs ILP...", flush=True)
                        ilp_cost, _, ilp_info = solve_all_pairs(
                            incomplete,
                            verbose=False,
                            relax=False,
                            return_x_values=False,
                        )
                        save_solve_result(
                            edge_result,
                            "all_pairs_ilp",
                            ilp_cost,
                            ilp_info,
                        )
                        update_ratio(p_result, edge_result)
                        atomic_write_json(path, data)
                        print(
                            f"  ILP={ilp_cost:.6g}, "
                            f"optimal={ilp_info['is_optimal']} "
                            f"({ilp_info['runtime_seconds']:.2f}s) saved",
                            flush=True,
                        )
                    else:
                        update_ratio(p_result, edge_result)
                        atomic_write_json(path, data)
                        print(f"  ILP={ilp_cost:.6g} already saved", flush=True)

                if args.csv_every > 0 and completed % args.csv_every == 0:
                    update_flat_csv(files)

    update_flat_csv(files)
    print("", flush=True)
    print(f"DONE: {completed}/{total} edge-deleted runs processed.", flush=True)


if __name__ == "__main__":
    main()
