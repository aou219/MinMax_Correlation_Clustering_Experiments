from pathlib import Path
import sys
import json
import shutil
import time
import copy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import experiment_helpers as eh

from facebook_sampling import (
    load_facebook_ego_edges,
    load_facebook_circles,
    build_complete_signed_matrix_from_facebook_sample,
)

from experiment_facebook import get_all_nodes_from_edges_and_circles


RESULTS_DIR = ROOT / "results" / "experiments_results_facebook" / "full"

TARGET_FILES = [
    "fb_ego414_full.json",
    "fb_ego686_full_without_ilp.json",
    "fb_ego698_full.json",
    "fb_ego3980_full.json",
]

P_DELETE_VALUES = [0.05, 0.15, 0.25, 0.40]


def p_delete_key(p_delete):
    return f"{p_delete:.2f}"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_seed(exp):
    return exp.get("graph_params", {}).get("seed")


def count_items(x):
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    return len(x)


def safe_ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def get_first_experiment(data):
    experiments = data.get("experiments", [])
    if not experiments:
        raise ValueError("Geen experiments gevonden in JSON.")
    return experiments[0]


def load_facebook_matrix(ego_id):
    edges_file = ROOT / "data" / "facebook" / f"{ego_id}.edges"
    circles_file = ROOT / "data" / "facebook" / f"{ego_id}.circles"

    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file niet gevonden: {edges_file}")

    if not circles_file.exists():
        raise FileNotFoundError(f"Circles file niet gevonden: {circles_file}")

    # Exact dezelfde aanpak als experiment_facebook.py
    edge_nodes, facebook_edges = load_facebook_ego_edges(str(edges_file))
    circles = load_facebook_circles(str(circles_file))

    all_nodes = get_all_nodes_from_edges_and_circles(
        edge_nodes=edge_nodes,
        circles=circles
    )

    S, node_to_index, positive_count, negative_count = build_complete_signed_matrix_from_facebook_sample(
        all_nodes,
        facebook_edges
    )

    return S


def run_edge_deleted_only(S, p_delete, seed, pivot_seeds, with_ilp=True):
    start_time = time.time()
    n = S.shape[0]

    # Alleen edge deletion op de bestaande complete graph matrix
    S_new, num_edges_deleted = eh.delete_edges(S, p_delete, seed)

    # Edge-deleted graph: Pivot
    pivot_results_new = eh.run_pivot_multiple(S_new, pivot_seeds)

    # Edge-deleted graph: bad triangles
    all_bad_triangles_new = eh.find_bad_triangles(S_new)
    edge_to_triangles_new = eh.make_edge_to_triangle_map(all_bad_triangles_new)

    min_disjoint_bad_triangles_new = eh.find_edge_disjoint_bad_triangles_min(
        copy.deepcopy(edge_to_triangles_new)
    )
    min_num_bad_triangles_new = eh.count_bad_triangles(min_disjoint_bad_triangles_new)

    max_disjoint_bad_triangles_new = eh.find_edge_disjoint_bad_triangles_max(
        copy.deepcopy(edge_to_triangles_new)
    )
    max_num_bad_triangles_new = eh.count_bad_triangles(max_disjoint_bad_triangles_new)

    edge_deleted_graph = {
        "num_edges_deleted": num_edges_deleted,
        "pivot": {
            "best_cost": pivot_results_new["best_cost"],
            "average_cost": pivot_results_new["average_cost"],
        },
        "bad_triangles": {
            "total_count": len(all_bad_triangles_new),
            "min_edge_disjoint_count": min_num_bad_triangles_new,
            "max_edge_disjoint_count": max_num_bad_triangles_new,
        },
    }

    ilp_cost_new_with4 = None

    if with_ilp:
        # Edge-deleted graph: ILP without 4-cycles
        ilp_cost_new_no4, ilp_x_values_new_no4, bad_cycles_ilp_new_no4 = eh.solve_ilp(
            S_new,
            verbose=False,
            relax=False,
            add_four_cycles=False
        )

        ilp_clusters_new_no4 = eh.find_ilp_clusters(ilp_x_values_new_no4, n)

        # Edge-deleted graph: ILP with 4-cycles
        ilp_cost_new_with4, ilp_x_values_new_with4, bad_cycles_ilp_new_with4 = eh.solve_ilp(
            S_new,
            verbose=False,
            relax=False,
            add_four_cycles=True
        )

        ilp_clusters_new_with4 = eh.find_ilp_clusters(ilp_x_values_new_with4, n)

        same_clustering_4_cycle = eh.same_clustering(
            ilp_clusters_new_no4,
            ilp_clusters_new_with4
        )

        edge_deleted_graph["ilp"] = {
            "without_4_cycles": {
                "cost": ilp_cost_new_no4
            },
            "with_4_cycles": {
                "cost": ilp_cost_new_with4,
                "bad_4_cycles_count": count_items(bad_cycles_ilp_new_with4)
            },
            "same_clustering_4_cycle": same_clustering_4_cycle
        }

    # Edge-deleted graph: LP relaxation without 4-cycles
    lp_cost_new_no4, lp_x_values_new_no4, bad_cycles_lp_new_no4 = eh.solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=False
    )

    # Edge-deleted graph: LP relaxation with 4-cycles
    lp_cost_new_with4, lp_x_values_new_with4, bad_cycles_lp_new_with4 = eh.solve_ilp(
        S_new,
        verbose=False,
        relax=True,
        add_four_cycles=True
    )

    edge_deleted_graph["lp_relaxation"] = {
        "without_4_cycles": {
            "cost": lp_cost_new_no4
        },
        "with_4_cycles": {
            "cost": lp_cost_new_with4
        }
    }

    # Edge-deleted graph: bad-triangle LP bounds
    primal_cost_new, primal_x_values_new = eh.solve_primal(
        S_new,
        all_bad_triangles_new,
        verbose=False
    )

    dual_cost_new, dual_x_values_new = eh.solve_dual(
        S_new,
        all_bad_triangles_new,
        verbose=False
    )

    edge_deleted_graph["bad_triangle_lp_bounds"] = {
        "primal_cost": primal_cost_new,
        "dual_cost": dual_cost_new
    }

    runtime_seconds = time.time() - start_time

    approximations = None

    if with_ilp and ilp_cost_new_with4 is not None:
        approximations = {
            "best_pivot_approximation_with_4_cycles": safe_ratio(
                pivot_results_new["best_cost"],
                ilp_cost_new_with4
            ),
            "average_pivot_approximation_with_4_cycles": safe_ratio(
                pivot_results_new["average_cost"],
                ilp_cost_new_with4
            ),
            "lp_relaxation_ratio_with_4_cycles": safe_ratio(
                lp_cost_new_with4,
                ilp_cost_new_with4
            ),
            "bad_triangle_primal_ratio": safe_ratio(
                primal_cost_new,
                ilp_cost_new_with4
            ),
            "bad_triangle_dual_ratio": safe_ratio(
                dual_cost_new,
                ilp_cost_new_with4
            ),
            "min_disjoint_bad_triangle_ratio": safe_ratio(
                min_num_bad_triangles_new,
                ilp_cost_new_with4
            ),
            "max_disjoint_bad_triangle_ratio": safe_ratio(
                max_num_bad_triangles_new,
                ilp_cost_new_with4
            ),
        }

    return {
        "edge_deleted_graph": edge_deleted_graph,
        "approximations": approximations,
        "runtime_seconds": runtime_seconds,
    }


def save_pdelete_result(results_file, p_delete, edge_only_result):
    data = read_json(results_file)
    exp = get_first_experiment(data)

    key = p_delete_key(p_delete)

    exp.setdefault("p_delete_results", {})

    if key in exp["p_delete_results"]:
        print(f"SKIP save: {results_file.name} already has p_delete {key}")
        return

    exp["p_delete_results"][key] = {
        "p_delete": p_delete,
        "edge_deleted_graph": edge_only_result["edge_deleted_graph"],
        "approximations": edge_only_result["approximations"],
        "runtime_seconds": edge_only_result["runtime_seconds"],
    }

    data.setdefault("shared_graph_params", {})
    data["shared_graph_params"].pop("p_delete", None)

    p_delete_values = data["shared_graph_params"].get("p_delete_values", [])
    p_delete_values = sorted(set([float(x) for x in p_delete_values] + [float(p_delete)]))
    data["shared_graph_params"]["p_delete_values"] = p_delete_values

    write_json(results_file, data)


def collect_jobs():
    jobs = []

    for filename in TARGET_FILES:
        path = RESULTS_DIR / filename

        if not path.exists():
            print("SKIP missing results file:", path)
            continue

        data = read_json(path)
        exp = get_first_experiment(data)

        shared = data.get("shared_graph_params", {})

        ego_id = str(shared.get("ego_id"))
        num_nodes = int(shared.get("num_nodes"))
        pivot_seeds = shared.get("pivot_seeds", list(range(1, 11)))
        seed = get_seed(exp)

        existing_keys = set(exp.get("p_delete_results", {}).keys())

        with_ilp = True

        if "without_ilp" in filename:
            with_ilp = False

        if exp.get("complete_graph", {}).get("ilp") is None:
            with_ilp = False

        for p_delete in P_DELETE_VALUES:
            key = p_delete_key(p_delete)

            if key in existing_keys:
                continue

            jobs.append({
                "path": path,
                "filename": filename,
                "ego_id": ego_id,
                "num_nodes": num_nodes,
                "seed": seed,
                "p_delete": p_delete,
                "pivot_seeds": pivot_seeds,
                "with_ilp": with_ilp,
            })

    # Kleinste graphs eerst
    jobs = sorted(
        jobs,
        key=lambda job: (
            int(job["num_nodes"]),
            int(job["ego_id"]),
            float(job["p_delete"])
        )
    )

    return jobs


def print_summary(jobs):
    print("\n" + "=" * 90)
    print("FACEBOOK MISSING P_DELETE — EDGE-DELETED ONLY")
    print("=" * 90)

    if not jobs:
        print("Geen jobs gevonden. Alles is al compleet.")
        return

    for job in jobs:
        mode = "with ILP" if job["with_ilp"] else "without ILP"
        print(
            f"{job['filename']:<35} ego={job['ego_id']:<5} "
            f"nodes={job['num_nodes']:<4} seed={job['seed']} "
            f"p_delete={job['p_delete']:<4} {mode}"
        )

    print("\nTotal runs:", len(jobs))


def backup_results():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / "results" / f"backup_before_facebook_missing_pdelete_{timestamp}"
    backup_target = backup_dir / "experiments_results_facebook" / "full"
    backup_target.mkdir(parents=True, exist_ok=True)

    for filename in TARGET_FILES:
        src = RESULTS_DIR / filename
        if src.exists():
            shutil.copy2(src, backup_target / filename)

    print("Backup gemaakt:", backup_dir)


def main():
    run_mode = "--run" in sys.argv

    jobs = collect_jobs()
    print_summary(jobs)

    if not jobs:
        return

    if not run_mode:
        print("\nDRY RUN ONLY.")
        print("Er is nog niks uitgevoerd.")
        print("Echt runnen met:")
        print("python -u scripts/run_missing_facebook_pdelete_edge_only.py --run")
        return

    backup_results()

    matrix_cache = {}

    for i, job in enumerate(jobs, start=1):
        ego_id = job["ego_id"]

        if ego_id not in matrix_cache:
            print(f"\nLoading Facebook matrix for ego {ego_id}...")
            matrix_cache[ego_id] = load_facebook_matrix(ego_id)

        S = matrix_cache[ego_id]

        print("\n" + "=" * 90)
        print(f"RUN {i}/{len(jobs)}")
        print("file:", job["path"])
        print("ego_id:", ego_id)
        print("nodes:", job["num_nodes"])
        print("seed:", job["seed"])
        print("p_delete:", job["p_delete"])
        print("mode:", "with ILP" if job["with_ilp"] else "without ILP")
        print("Complete graph wordt NIET opnieuw geanalyseerd.")
        print("=" * 90)

        result = run_edge_deleted_only(
            S=S,
            p_delete=job["p_delete"],
            seed=job["seed"],
            pivot_seeds=job["pivot_seeds"],
            with_ilp=job["with_ilp"],
        )

        save_pdelete_result(
            results_file=job["path"],
            p_delete=job["p_delete"],
            edge_only_result=result,
        )

        print(
            f"Saved {job['filename']} p_delete={job['p_delete']} "
            f"runtime={result['runtime_seconds']:.2f} sec"
        )

    print("\nDONE.")


if __name__ == "__main__":
    main()
