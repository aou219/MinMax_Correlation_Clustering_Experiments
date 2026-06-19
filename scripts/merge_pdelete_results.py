from pathlib import Path
import json


RESULTS_ROOT = Path("results")

P_DELETE_DIRS = {
    "p_delete_005": 0.05,
    "p_delete_015": 0.15,
    "p_delete_025": 0.25,
    "p_delete_040": 0.40,
}

FAMILIES = [
    "experiments_results_random",
    "experiments_results_clique",
    "experiments_results_facebook",
]


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def pdelete_key(p_delete):
    """
    0.05 -> "0.05"
    0.15 -> "0.15"
    0.25 -> "0.25"
    0.40 -> "0.40"
    """
    return f"{p_delete:.2f}"


def remove_p_delete_from_shared_params(shared):
    """
    In the merged file, p_delete is no longer a single shared parameter,
    because every file contains multiple p_delete values.
    """
    clean = dict(shared)
    clean.pop("p_delete", None)
    return clean


def get_seed(experiment):
    return experiment.get("graph_params", {}).get("seed")


def merge_file_group(source_files, output_file):
    """
    Merge files from several p_delete folders into one file.

    Input example:
    results/p_delete_005/experiments_results_clique/n10/clq_n10_2x5.json
    results/p_delete_015/experiments_results_clique/n10/clq_n10_2x5.json
    results/p_delete_025/experiments_results_clique/n10/clq_n10_2x5.json
    results/p_delete_040/experiments_results_clique/n10/clq_n10_2x5.json

    Output:
    results/experiments_results_clique/n10/clq_n10_2x5.json
    """

    merged = {
        "shared_graph_params": None,
        "experiments": []
    }

    experiments_by_seed = {}
    p_delete_values_found = []

    for p_delete, source_file in sorted(source_files, key=lambda x: x[0]):
        data = read_json(source_file)

        shared = data.get("shared_graph_params", {})
        clean_shared = remove_p_delete_from_shared_params(shared)

        if merged["shared_graph_params"] is None:
            merged["shared_graph_params"] = clean_shared
        else:
            if merged["shared_graph_params"] != clean_shared:
                print(f"WARNING: shared params differ in {source_file}")

        p_delete_values_found.append(p_delete)

        for experiment in data.get("experiments", []):
            seed = get_seed(experiment)

            if seed is None:
                print(f"WARNING: experiment without seed in {source_file}")
                continue

            if seed not in experiments_by_seed:
                experiments_by_seed[seed] = {
                    "graph_params": {
                        "seed": seed
                    },
                    "complete_graph": experiment.get("complete_graph"),
                    "complete_graph_approximations": (
                        experiment.get("approximations", {}).get("complete_graph")
                    ),
                    "p_delete_results": {}
                }
            else:
                existing_complete = experiments_by_seed[seed].get("complete_graph")
                new_complete = experiment.get("complete_graph")

                if existing_complete != new_complete:
                    print(
                        f"WARNING: complete_graph differs for seed {seed} "
                        f"in {source_file}"
                    )

                existing_complete_approx = experiments_by_seed[seed].get(
                    "complete_graph_approximations"
                )
                new_complete_approx = experiment.get("approximations", {}).get(
                    "complete_graph"
                )

                if existing_complete_approx != new_complete_approx:
                    print(
                        f"WARNING: complete_graph approximations differ for seed {seed} "
                        f"in {source_file}"
                    )

            key = pdelete_key(p_delete)

            experiments_by_seed[seed]["p_delete_results"][key] = {
                "p_delete": p_delete,
                "edge_deleted_graph": experiment.get("edge_deleted_graph"),
                "approximations": (
                    experiment.get("approximations", {}).get("edge_deleted_graph")
                ),
                "runtime_seconds": experiment.get("runtime_seconds")
            }

    if merged["shared_graph_params"] is None:
        merged["shared_graph_params"] = {}

    merged["shared_graph_params"]["p_delete_values"] = sorted(set(p_delete_values_found))

    merged["experiments"] = [
        experiments_by_seed[seed]
        for seed in sorted(experiments_by_seed)
    ]

    write_json(output_file, merged)


def collect_groups():
    """
    Group files by family and relative path.
    This works for random, clique, and facebook.

    Example:
    results/p_delete_005/experiments_results_random/n30/random_n30_p05.json
    results/p_delete_015/experiments_results_random/n30/random_n30_p05.json

    become:
    results/experiments_results_random/n30/random_n30_p05.json
    """

    groups = {}

    for p_delete_dir, p_delete in P_DELETE_DIRS.items():
        base = RESULTS_ROOT / p_delete_dir

        if not base.exists():
            print(f"SKIP missing folder: {base}")
            continue

        for family in FAMILIES:
            family_dir = base / family

            if not family_dir.exists():
                continue

            for source_file in family_dir.rglob("*.json"):
                relative_path = source_file.relative_to(family_dir)
                output_file = RESULTS_ROOT / family / relative_path

                groups.setdefault(output_file, [])
                groups[output_file].append((p_delete, source_file))

    return groups


def main():
    groups = collect_groups()

    print(f"Found {len(groups)} merged output files to create.")

    for output_file, source_files in sorted(groups.items()):
        print("\n" + "=" * 70)
        print(f"Creating: {output_file}")
        print("From:")

        for p_delete, source_file in sorted(source_files, key=lambda x: x[0]):
            print(f"  p_delete={p_delete}: {source_file}")

        merge_file_group(source_files, output_file)

    print("\nDONE.")
    print("Merged structure created under:")
    print("  results/experiments_results_random")
    print("  results/experiments_results_clique")
    print("  results/experiments_results_facebook")


if __name__ == "__main__":
    main()