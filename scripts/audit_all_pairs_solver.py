import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from all_pairs_solver import solve_all_pairs
from edge_deletion import delete_edges
from graph_generation import generate_signed_complete_graph
from ilp_solver import solve_ilp


def bad_four_cycle_graph():
    """Return a +,+,+,- cycle with both diagonals missing."""
    S = np.zeros((4, 4), dtype=int)
    for i, j, sign in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (0, 3, -1),
    ):
        S[i, j] = sign
        S[j, i] = sign
    return S


def solve_comparison(case_name, S):
    sparse_ilp, _, _ = solve_ilp(
        S, verbose=False, relax=False, add_four_cycles=False
    )
    sparse_ilp_with4, _, _ = solve_ilp(
        S, verbose=False, relax=False, add_four_cycles=True
    )
    all_pairs_ilp, _, ilp_info = solve_all_pairs(
        S, verbose=False, relax=False
    )

    sparse_lp, _, _ = solve_ilp(
        S, verbose=False, relax=True, add_four_cycles=False
    )
    sparse_lp_with4, _, _ = solve_ilp(
        S, verbose=False, relax=True, add_four_cycles=True
    )
    all_pairs_lp, _, lp_info = solve_all_pairs(
        S, verbose=False, relax=True
    )

    n = S.shape[0]
    observed_edges = int(np.count_nonzero(np.triu(S, k=1)))
    return {
        "case": case_name,
        "n": n,
        "observed_edges": observed_edges,
        "all_pairs_variables": n * (n - 1) // 2,
        "sparse_ilp": sparse_ilp,
        "sparse_ilp_with4": sparse_ilp_with4,
        "all_pairs_ilp": all_pairs_ilp,
        "sparse_lp": sparse_lp,
        "sparse_lp_with4": sparse_lp_with4,
        "all_pairs_lp": all_pairs_lp,
        "all_pairs_ilp_optimal": ilp_info["is_optimal"],
        "all_pairs_ilp_runtime_seconds": ilp_info["runtime_seconds"],
        "all_pairs_lp_runtime_seconds": lp_info["runtime_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare the existing sparse solver with the all-pairs solver."
    )
    parser.add_argument("--random-n", type=int, default=10)
    parser.add_argument("--graph-seed", type=int, default=1)
    parser.add_argument("--delete-seed", type=int, default=1)
    parser.add_argument("--p-positive", type=float, default=0.5)
    parser.add_argument("--p-delete", type=float, default=0.25)
    args = parser.parse_args()

    complete = generate_signed_complete_graph(
        n=args.random_n,
        p_positive=args.p_positive,
        seed=args.graph_seed,
    )
    incomplete, _ = delete_edges(
        complete,
        p_delete=args.p_delete,
        seed=args.delete_seed,
    )

    rows = [
        solve_comparison("bad_four_cycle", bad_four_cycle_graph()),
        solve_comparison("random_edge_deleted", incomplete),
    ]

    output_dir = REPO_ROOT / "results" / "all_pairs_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"all_pairs_comparison_{timestamp}.csv"

    with output_path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['case']}: sparse ILP={row['sparse_ilp']}, "
            f"sparse ILP+4={row['sparse_ilp_with4']}, "
            f"all-pairs ILP={row['all_pairs_ilp']}"
        )
    print(f"New audit output: {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
