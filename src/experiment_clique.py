"""Run the complete, resumable clique experiment grid."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiment_helpers import (
    P_DELETE_VALUES,
    PIVOT_SEEDS,
    format_runtime,
    has_complete_result,
    has_p_delete_result,
    initialize_file,
    missing_summary,
    run_complete_graph,
    run_edge_deleted_graph,
    save_complete_result,
    save_p_delete_result,
)
from graph_generation import generate_clique_signed_graph


CLIQUE_CONFIGS = {
    10: (
        ("2x5", [5, 5]),
        ("4_3_3", [4, 3, 3]),
        ("5_3_2", [5, 3, 2]),
    ),
    15: (
        ("3x5", [5, 5, 5]),
        ("5_5_3_2", [5, 5, 3, 2]),
        ("8_7", [8, 7]),
    ),
    20: (
        ("2x10", [10, 10]),
        ("4x5", [5, 5, 5, 5]),
        ("7_7_6", [7, 7, 6]),
    ),
    25: (
        ("10_10_5", [10, 10, 5]),
        ("12_7_6", [12, 7, 6]),
        ("13_12", [13, 12]),
        ("9_8_8", [9, 8, 8]),
    ),
    30: (
        ("15_10_5", [15, 10, 5]),
        ("20_5_5", [20, 5, 5]),
        ("2x15", [15, 15]),
        ("3x10", [10, 10, 10]),
    ),
    100: (
        ("10x10", [10] * 10),
        ("4x25", [25] * 4),
        ("60_25_10_5", [60, 25, 10, 5]),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=CLIQUE_CONFIGS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--p-delete", type=float, choices=P_DELETE_VALUES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report missing runs without computing or writing anything.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    n_values = [args.n] if args.n else list(CLIQUE_CONFIGS)
    total_complete_missing = 0
    total_p_delete_missing = 0

    for n in n_values:
        seeds = [args.seed] if args.seed else list(
            range(1, 21) if n == 100 else range(1, 51)
        )
        p_deletes = (
            [args.p_delete] if args.p_delete is not None
            else list(P_DELETE_VALUES)
        )

        for tag, cluster_sizes in CLIQUE_CONFIGS[n]:
            output = (
                root
                / "results"
                / "experiments_results_clique"
                / f"n{n}"
                / f"clq_n{n}_{tag}.json"
            )
            shared = {
                "graph_type": "clique",
                "num_nodes": n,
                "cluster_sizes": cluster_sizes,
                "p_pos_between": 0.1,
                "pivot_seeds": list(PIVOT_SEEDS),
                "p_pos_inside": 0.9,
                "p_delete_values": list(P_DELETE_VALUES),
            }

            if args.dry_run:
                missing = missing_summary(output, seeds, p_deletes)
                total_complete_missing += missing["complete_graphs"]
                total_p_delete_missing += missing["p_delete_results"]
                print(
                    f"{output.name}: "
                    f"complete graphs missing="
                    f"{missing['complete_graphs']}, "
                    f"p_delete results missing="
                    f"{missing['p_delete_results']}"
                )
                continue

            initialize_file(output, shared, seeds)

            for seed in seeds:
                S, _ = generate_clique_signed_graph(
                    cluster_sizes=cluster_sizes,
                    p_pos_inside=0.9,
                    p_pos_between=0.1,
                    seed=seed,
                )

                if not has_complete_result(output, seed):
                    started = time.time()
                    complete, approx, sparse_approx = run_complete_graph(S)
                    save_complete_result(
                        output, seed, complete, approx, sparse_approx
                    )

                    print(
                        f"{output.name}, seed={seed}, complete graph: "
                        f"{format_runtime(time.time() - started)}",
                        flush=True,
                    )

                for p_delete in p_deletes:
                    if has_p_delete_result(output, seed, p_delete):
                        continue
                    result = run_edge_deleted_graph(S, p_delete, seed)
                    save_p_delete_result(output, seed, p_delete, result)
                    print(
                        f"{output.name}, seed={seed}, "
                        f"p_delete={p_delete:.2f}: "
                        f"{format_runtime(result['runtime_seconds'])}",
                        flush=True,
                    )

    if args.dry_run:
        print(
            "\nDRY RUN TOTAL: "
            f"complete graphs missing={total_complete_missing}, "
            f"p_delete results missing={total_p_delete_missing}"
        )


if __name__ == "__main__":
    main()
