from pathlib import Path
import sys
import json
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_generation import generate_clique_signed_graph
from experiment_helpers import (
    run_full_experiment,
    print_standard_results,
    build_saveable_results
)

TARGET_NS = [10, 15, 20, 25]
TARGET_SEEDS = list(range(1, 41))
P_DELETE_VALUES = [0.05, 0.15, 0.25, 0.40]
PIVOT_SEEDS = list(range(1, 11))

BASE_DIR = ROOT / "results" / "experiments_results_clique"


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


def parse_cluster_sizes(path, n):
    stem = path.stem
    prefix = f"clq_n{n}_"

    if not stem.startswith(prefix):
        raise ValueError(f"Kan cluster sizes niet parsen uit bestandsnaam: {path.name}")

    tail = stem.replace(prefix, "")
    cluster_sizes = []

    for part in tail.split("_"):
        if "x" in part:
            count, size = part.split("x")
            cluster_sizes.extend([int(size)] * int(count))
        else:
            cluster_sizes.append(int(part))

    if sum(cluster_sizes) != n:
        raise ValueError(
            f"Cluster sizes kloppen niet voor {path.name}: {cluster_sizes}, som={sum(cluster_sizes)}, n={n}"
        )

    return cluster_sizes


def get_param(data, first_exp, names, default):
    sources = [
        data.get("shared_graph_params", {}),
        first_exp.get("graph_params", {}) if first_exp else {},
    ]

    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]

    return default


def backup_target_dirs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / "results" / f"backup_before_clique_seed_1_40_{timestamp}" / "experiments_results_clique"

    for n in TARGET_NS:
        src = BASE_DIR / f"n{n}"
        dst = backup_dir / f"n{n}"

        if src.exists():
            shutil.copytree(src, dst)

    print("Backup gemaakt:")
    print(backup_dir.parent)
    print()


def collect_jobs():
    jobs = []

    for n in TARGET_NS:
        folder = BASE_DIR / f"n{n}"

        if not folder.exists():
            print("SKIP missing folder:", folder)
            continue

        for path in sorted(folder.glob(f"clq_n{n}_*.json")):
            data = read_json(path)
            experiments = data.get("experiments", [])

            first_exp = experiments[0] if experiments else None
            cluster_sizes = parse_cluster_sizes(path, n)

            p_pos_inside = float(get_param(
                data,
                first_exp,
                ["p_pos_inside", "p_positive_inside", "positive_inside_probability"],
                0.9
            ))

            p_pos_between = float(get_param(
                data,
                first_exp,
                ["p_pos_between", "p_positive_between", "positive_between_probability"],
                0.1
            ))

            seed_map = {
                get_seed(exp): exp
                for exp in experiments
                if get_seed(exp) is not None
            }

            for seed in TARGET_SEEDS:
                exp = seed_map.get(seed)
                existing_keys = set(exp.get("p_delete_results", {}).keys()) if exp else set()

                for p_delete in P_DELETE_VALUES:
                    key = p_delete_key(p_delete)

                    # Niet opnieuw runnen als deze seed + p_delete al bestaat
                    if key in existing_keys:
                        continue

                    jobs.append({
                        "path": path,
                        "n": n,
                        "cluster_sizes": cluster_sizes,
                        "p_pos_inside": p_pos_inside,
                        "p_pos_between": p_pos_between,
                        "seed": seed,
                        "p_delete": p_delete,
                    })

    return jobs


def save_into_current_json(results_file, results, p_delete):
    data = read_json(results_file)

    seed = results["graph_params"]["seed"]
    key = p_delete_key(p_delete)

    data.setdefault("shared_graph_params", {})
    data["shared_graph_params"].pop("p_delete", None)

    p_delete_values = data["shared_graph_params"].get("p_delete_values", [])
    p_delete_values = sorted(set([float(x) for x in p_delete_values] + [float(p_delete)]))
    data["shared_graph_params"]["p_delete_values"] = p_delete_values

    data.setdefault("experiments", [])

    seed_map = {
        get_seed(exp): exp
        for exp in data["experiments"]
        if get_seed(exp) is not None
    }

    if seed not in seed_map:
        data["experiments"].append({
            "graph_params": {
                "seed": seed
            },
            "complete_graph": results.get("complete_graph"),
            "complete_graph_approximations": results.get("approximations", {}).get("complete_graph"),
            "p_delete_results": {}
        })

        seed_map = {
            get_seed(exp): exp
            for exp in data["experiments"]
            if get_seed(exp) is not None
        }

    exp = seed_map[seed]

    if exp.get("complete_graph") is None:
        exp["complete_graph"] = results.get("complete_graph")

    if exp.get("complete_graph_approximations") is None:
        exp["complete_graph_approximations"] = results.get("approximations", {}).get("complete_graph")

    exp.setdefault("p_delete_results", {})

    # Extra veiligheid: nooit bestaande p_delete overschrijven
    if key in exp["p_delete_results"]:
        print(f"SKIP save: seed {seed} already has p_delete {key} in {results_file.name}")
        return

    exp["p_delete_results"][key] = {
        "p_delete": p_delete,
        "edge_deleted_graph": results.get("edge_deleted_graph"),
        "approximations": results.get("approximations", {}).get("edge_deleted_graph"),
        "runtime_seconds": results.get("runtime_seconds")
    }

    data["experiments"] = sorted(
        data["experiments"],
        key=lambda exp: get_seed(exp) if get_seed(exp) is not None else 10**9
    )

    write_json(results_file, data)


def print_job_summary(jobs):
    print("\n" + "=" * 100)
    print("CLIQUE SEED 1-40 JOBS")
    print("=" * 100)

    if not jobs:
        print("Geen jobs gevonden. Alles lijkt al compleet.")
        return

    summary = {}

    for job in jobs:
        key = (
            job["n"],
            job["path"].name,
            job["p_delete"],
        )
        summary.setdefault(key, []).append(job["seed"])

    for (n, filename, p_delete), seeds in sorted(summary.items()):
        seeds = sorted(seeds)
        print(
            f"n={n:<3} p_delete={p_delete:<4} "
            f"count={len(seeds):<3} seeds={min(seeds)}-{max(seeds):<3} file={filename}"
        )

    print("\nTotal runs:", len(jobs))


def main():
    run_mode = "--run" in sys.argv

    jobs = collect_jobs()
    print_job_summary(jobs)

    if not jobs:
        return

    if not run_mode:
        print("\nDRY RUN ONLY.")
        print("Er is nog niks uitgevoerd.")
        print("Gebruik dit om echt te runnen:")
        print("python -u scripts/run_clique_missing_seeds_1_40_n10_n25.py --run")
        return

    backup_target_dirs()

    for i, job in enumerate(jobs, start=1):
        path = job["path"]
        n = job["n"]
        cluster_sizes = job["cluster_sizes"]
        p_pos_inside = job["p_pos_inside"]
        p_pos_between = job["p_pos_between"]
        seed = job["seed"]
        p_delete = job["p_delete"]

        print("\n" + "=" * 100)
        print(f"RUN {i}/{len(jobs)}")
        print("file:", path)
        print("n:", n)
        print("cluster_sizes:", cluster_sizes)
        print("p_pos_inside:", p_pos_inside)
        print("p_pos_between:", p_pos_between)
        print("seed:", seed)
        print("p_delete:", p_delete)
        print("=" * 100)

        generated = generate_clique_signed_graph(
            cluster_sizes=cluster_sizes,
            p_pos_inside=p_pos_inside,
            p_pos_between=p_pos_between,
            p_delete_inside=0.0,
            p_delete_between=0.0,
            seed=seed
        )

        if isinstance(generated, tuple):
            S = generated[0]
        else:
            S = generated

        experiment_data = run_full_experiment(
            S=S,
            p_delete=p_delete,
            seed=seed,
            pivot_seeds=PIVOT_SEEDS
        )

        graph_params = {
            "graph_type": "clique",
            "num_nodes": n,
            "cluster_sizes": cluster_sizes,
            "p_pos_inside": p_pos_inside,
            "p_pos_between": p_pos_between,
            "seed": seed,
            "pivot_seeds": PIVOT_SEEDS,
            "p_delete": p_delete,
            "num_edges_deleted": experiment_data["num_edges_deleted"]
        }

        print_standard_results(
            graph_type="clique",
            graph_params=graph_params,
            experiment_data=experiment_data
        )

        results = build_saveable_results(
            graph_params=graph_params,
            experiment_data=experiment_data
        )

        save_into_current_json(
            results_file=path,
            results=results,
            p_delete=p_delete
        )

        print(f"Saved seed {seed}, p_delete {p_delete} into {path}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
