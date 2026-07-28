#!/usr/bin/env python3
"""Run Facebook experiments and write the results to one CSV table."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment_helpers as h
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)


OUTPUT_TABLE = (
    ROOT / "results/output_tables/facebook_output.csv"
)

# Experiment settings
EGO_IDS = ["3980"]

# p_delete = 0 means the complete graph.
P_DELETE_VALUES = [0.0, 0.05]
DELETION_SEEDS = [1]

PIVOT_SEEDS = range(1, 101)

D_HAT = 8
LAMBDA = 5
MINMAX_LP_R = 0.4
MINMAX_LP_R2 = 0.4
MINMAX_LP_METHOD = 2


def load_facebook_graph(ego_id):
    candidates = [
        ROOT / "data/facebook" / f"{ego_id}.edges",
        ROOT / "data/facebook/facebook_3" / f"{ego_id}.edges",
    ]

    edges_file = next(
        (path for path in candidates if path.exists()),
        None,
    )

    if edges_file is None:
        raise FileNotFoundError(
            f"No .edges file found for Facebook ego {ego_id}."
        )

    nodes, edges = load_facebook_ego_edges(
        str(edges_file)
    )

    matrix, _, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            sorted(nodes),
            edges,
        )
    )

    return matrix


def main():
    rows = []

    for ego_id in EGO_IDS:
        matrix = load_facebook_graph(ego_id)
        n = len(matrix)

        for p_delete in P_DELETE_VALUES:
            seeds = [0] if p_delete == 0 else DELETION_SEEDS

            for seed in seeds:
                print(
                    f"run ego={ego_id}, seed={seed}, "
                    f"p_delete={p_delete}"
                )

                result = h.run_full_experiment(
                    matrix.copy(),
                    p_delete=p_delete,
                    seed=seed,
                    pivot_seeds=PIVOT_SEEDS,
                    compute_pivot=True,
                    compute_normal_lp=True,
                    compute_min_max=True,
                    compute_min_max_lp=True,
                    min_max_cc_param_1=D_HAT,
                    min_max_cc_param_2=LAMBDA,
                    min_max_lp_r=MINMAX_LP_R,
                    min_max_lp_r2=MINMAX_LP_R2,
                    min_max_lp_method=MINMAX_LP_METHOD,
                )

                complete_graph = p_delete == 0
                suffix = "" if complete_graph else "_new"

                pivot = result[f"pivot_results{suffix}"]
                normal_lp_info = result[
                    f"normal_lp_info{suffix}"
                ] or {}
                minmax_cc = result[
                    f"min_max_cc_results{suffix}"
                ]
                minmax_lp = result[
                    f"min_max_lp_results{suffix}"
                ]

                ratio_key = (
                    "complete_min_max_approximation_ratio"
                    if complete_graph
                    else "edge_min_max_approximation_ratio"
                )

                rows.append({
                    "ego_id": ego_id,
                    "n": n,
                    "seed": seed,
                    "p_delete": p_delete,
                    "num_edges_deleted": (
                        0
                        if complete_graph
                        else result["num_edges_deleted"]
                    ),
                    "pivot_best_cost": pivot["best_cost"],
                    "pivot_average_cost": pivot["average_cost"],
                    "lp_cost": result[
                        f"normal_lp_cost{suffix}"
                    ],
                    "lp_runtime_seconds": normal_lp_info.get(
                        "runtime_seconds",
                        "",
                    ),
                    "minmax_cc_cluster_count": minmax_cc[
                        "cluster_count"
                    ],
                    "minmax_cc_max_disagreement": minmax_cc[
                        "max_disagreement"
                    ],
                    "minmax_cc_d_hat": minmax_cc["d_hat"],
                    "minmax_cc_lambda": minmax_cc["lambda"],
                    "minmax_cc_runtime_seconds": minmax_cc[
                        "runtime_seconds"
                    ],
                    "minmax_lp_cost": minmax_lp["lp_cost"],
                    "minmax_lp_rounding_cost": minmax_lp[
                        "rounding_cost"
                    ],
                    "minmax_lp_cluster_count": minmax_lp[
                        "cluster_count"
                    ],
                    "minmax_lp_max_disagreement_vertex": (
                        minmax_lp["max_disagreement_vertex"]
                    ),
                    "minmax_lp_r": minmax_lp["r"],
                    "minmax_lp_r2": minmax_lp["r2"],
                    "minmax_lp_method": minmax_lp["method"],
                    "minmax_lp_norm": minmax_lp["norm"],
                    "minmax_lp_runtime_seconds": minmax_lp[
                        "lp_runtime_seconds"
                    ],
                    "minmax_lp_rounding_runtime_seconds": (
                        minmax_lp["rounding_runtime_seconds"]
                    ),
                    "minmax_lp_total_runtime_seconds": minmax_lp[
                        "total_runtime_seconds"
                    ],
                    "minmax_cc_to_lp_ratio": result[ratio_key],
                })

                del result
                gc.collect()

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_TABLE, index=False)

    print(f"Done. Output: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
