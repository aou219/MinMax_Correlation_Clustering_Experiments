#!/usr/bin/env python3
"""Run clique experiments and write the results to one CSV table."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment_helpers as h
from src.graph_generation import generate_clique_signed_graph


OUTPUT_TABLE = (
    ROOT / "results/research_tables/clique_reproduction.csv"
)

# Experiment settings
CLUSTER_SIZE_CASES = [[5, 6]]
P_POS_INSIDE_VALUES = [0.9]
P_POS_BETWEEN_VALUES = [0.1]
GRAPH_SEEDS = [1]

# p_delete = 0 means the complete graph.
P_DELETE_VALUES = [0.0, 0.05]

PIVOT_SEEDS = range(1, 11)


def main():
    rows = []

    for cluster_sizes in CLUSTER_SIZE_CASES:
        n = sum(cluster_sizes)
        cluster_sizes_text = "_".join(map(str, cluster_sizes))
        case_id = f"clq_n{n}_{cluster_sizes_text}"

        for p_inside in P_POS_INSIDE_VALUES:
            for p_between in P_POS_BETWEEN_VALUES:
                for seed in GRAPH_SEEDS:
                    matrix, _ = generate_clique_signed_graph(
                        cluster_sizes=cluster_sizes,
                        p_pos_inside=p_inside,
                        p_pos_between=p_between,
                        seed=seed,
                    )

                    for p_delete in P_DELETE_VALUES:
                        print(
                            f"run case={case_id}, seed={seed}, "
                            f"p_delete={p_delete}"
                        )

                        result = h.run_full_experiment(
                            matrix.copy(),
                            p_delete=p_delete,
                            seed=seed,
                            pivot_seeds=PIVOT_SEEDS,
                            compute_pivot=True,
                            compute_normal_lp=True,
                            compute_min_max=False,
                            compute_min_max_lp=False,
                        )

                        complete_graph = p_delete == 0

                        pivot = result[
                            "pivot_results"
                            if complete_graph
                            else "pivot_results_new"
                        ]
                        lp_cost = result[
                            "normal_lp_cost"
                            if complete_graph
                            else "normal_lp_cost_new"
                        ]
                        lp_info = result[
                            "normal_lp_info"
                            if complete_graph
                            else "normal_lp_info_new"
                        ]

                        rows.append({
                            "case_id": case_id,
                            "n": n,
                            "seed": seed,
                            "p_delete": p_delete,
                            "p_pos_inside": p_inside,
                            "p_pos_between": p_between,
                            "cluster_sizes": cluster_sizes_text,
                            "num_edges_deleted": (
                                0
                                if complete_graph
                                else result["num_edges_deleted"]
                            ),
                            "pivot_best_cost": pivot["best_cost"],
                            "pivot_average_cost": pivot["average_cost"],
                            "lp_cost": lp_cost,
                            "lp_runtime_seconds": (
                                lp_info or {}
                            ).get("runtime_seconds", ""),
                        })

                        del result
                        gc.collect()

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_TABLE, index=False)

    print(f"Done. Output: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
