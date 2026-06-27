"""Run the complete, resumable random signed graph experiment grid."""

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
from graph_generation import generate_signed_complete_graph


N_VALUES = (5, 10, 15, 20, 25, 30)
P_POSITIVE_VALUES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=N_VALUES)
    parser.add_argument("--p-positive", type=float, choices=P_POSITIVE_VALUES)
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
    n_values = [args.n] if args.n else list(N_VALUES)
    p_values = (
        [args.p_positive] if args.p_positive is not None
        else list(P_POSITIVE_VALUES)
    )
    seeds = [args.seed] if args.seed else list(range(1, 51))
    p_deletes = (
        [args.p_delete] if args.p_delete is not None
        else list(P_DELETE_VALUES)
    )
    total_complete_missing = 0
    total_p_delete_missing = 0

    for n in n_values:
        for p_positive in p_values:
            tag = f"p{int(round(p_positive * 10)):02d}"
            output = (
                root
                / "results"
                / "experiments_results_random"
                / f"n{n}"
                / f"random_n{n}_{tag}.json"
            )
            shared = {
                "graph_type": "random",
                "num_nodes": n,
                "p_positive": p_positive,
                "pivot_seeds": list(PIVOT_SEEDS),
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
                S = generate_signed_complete_graph(
                    n=n,
                    p_positive=p_positive,
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
