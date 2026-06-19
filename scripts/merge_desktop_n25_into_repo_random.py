from pathlib import Path
import json
import shutil
from datetime import datetime


# Source: map op je bureaublad met de nieuwe n25 files
SOURCE_DIR_CANDIDATES = [
    Path.home() / "Desktop" / "n25",
    Path.home() / "Bureaublad" / "n25",
]

# Target: bestaande repo-map met p_delete=0.15
TARGET_DIR = Path("results/experiments_results_random/n25")


def find_source_dir():
    for path in SOURCE_DIR_CANDIDATES:
        if path.exists():
            return path

    print("Kon source map niet vinden.")
    print("Ik heb gezocht op:")

    for path in SOURCE_DIR_CANDIDATES:
        print(" ", path)

    raise SystemExit


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_seed(exp):
    return exp.get("graph_params", {}).get("seed")


def backup_target():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("results") / f"backup_before_merge_desktop_n25_{timestamp}"

    shutil.copytree(
        TARGET_DIR,
        backup_dir / "experiments_results_random" / "n25"
    )

    print("Backup gemaakt:")
    print(backup_dir)


def merge_file(source_file, target_file):
    source_data = read_json(source_file)
    target_data = read_json(target_file)

    # shared_graph_params updaten
    target_data.setdefault("shared_graph_params", {})
    target_data["shared_graph_params"].pop("p_delete", None)

    source_values = source_data.get("shared_graph_params", {}).get("p_delete_values", [])
    target_values = target_data.get("shared_graph_params", {}).get("p_delete_values", [])

    all_values = sorted(set(float(x) for x in source_values + target_values))
    target_data["shared_graph_params"]["p_delete_values"] = all_values

    target_experiments = target_data.setdefault("experiments", [])
    source_experiments = source_data.get("experiments", [])

    target_by_seed = {
        get_seed(exp): exp
        for exp in target_experiments
        if get_seed(exp) is not None
    }

    added_seeds = 0
    added_pdelete_entries = 0
    skipped_existing_entries = 0

    for source_exp in source_experiments:
        seed = get_seed(source_exp)

        if seed is None:
            continue

        # Als seed nog niet bestaat in target, maak nieuwe instance aan
        if seed not in target_by_seed:
            new_exp = {
                "graph_params": {
                    "seed": seed
                },
                "complete_graph": source_exp.get("complete_graph"),
                "complete_graph_approximations": source_exp.get("complete_graph_approximations"),
                "p_delete_results": {}
            }

            target_experiments.append(new_exp)
            target_by_seed[seed] = new_exp
            added_seeds += 1

        target_exp = target_by_seed[seed]

        # Complete graph info aanvullen als die ontbreekt
        if target_exp.get("complete_graph") is None:
            target_exp["complete_graph"] = source_exp.get("complete_graph")

        if target_exp.get("complete_graph_approximations") is None:
            target_exp["complete_graph_approximations"] = source_exp.get("complete_graph_approximations")

        target_exp.setdefault("p_delete_results", {})

        # Alle p_delete resultaten uit source toevoegen aan dezelfde seed-instance
        for p_delete_key, p_delete_data in source_exp.get("p_delete_results", {}).items():
            if p_delete_key in target_exp["p_delete_results"]:
                # Bestaande 0.15 bijvoorbeeld niet overschrijven
                skipped_existing_entries += 1
                continue

            target_exp["p_delete_results"][p_delete_key] = p_delete_data
            added_pdelete_entries += 1

    target_data["experiments"] = sorted(
        target_experiments,
        key=lambda exp: get_seed(exp) if get_seed(exp) is not None else 10**9
    )

    write_json(target_file, target_data)

    return added_seeds, added_pdelete_entries, skipped_existing_entries


def main():
    source_dir = find_source_dir()

    print("Source map:")
    print(source_dir)

    print("\nTarget map:")
    print(TARGET_DIR)

    if not TARGET_DIR.exists():
        print("\nTarget map bestaat niet:")
        print(TARGET_DIR)
        raise SystemExit

    backup_target()

    total_added_seeds = 0
    total_added_pdelete_entries = 0
    total_skipped_existing_entries = 0

    source_files = sorted(source_dir.glob("random_n25_p*.json"))

    if not source_files:
        print("\nGeen random_n25_p*.json files gevonden in:")
        print(source_dir)
        raise SystemExit

    for source_file in source_files:
        target_file = TARGET_DIR / source_file.name

        print("\n" + "=" * 80)
        print("Source:", source_file)
        print("Target:", target_file)

        if not target_file.exists():
            print("SKIP: target file bestaat niet.")
            continue

        added_seeds, added_pdelete_entries, skipped_existing_entries = merge_file(
            source_file=source_file,
            target_file=target_file
        )

        total_added_seeds += added_seeds
        total_added_pdelete_entries += added_pdelete_entries
        total_skipped_existing_entries += skipped_existing_entries

        print("Added seeds:", added_seeds)
        print("Added p_delete entries:", added_pdelete_entries)
        print("Skipped existing entries:", skipped_existing_entries)

    print("\nDONE.")
    print("Total added seeds:", total_added_seeds)
    print("Total added p_delete entries:", total_added_pdelete_entries)
    print("Total skipped existing entries:", total_skipped_existing_entries)


if __name__ == "__main__":
    main()
