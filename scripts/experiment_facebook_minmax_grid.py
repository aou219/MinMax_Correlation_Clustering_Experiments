import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np

from src.facebook_sampling import (
    load_facebook_ego_edges,
    load_facebook_circles,
    build_complete_signed_matrix_from_facebook_sample,
)

from src.edge_deletion import delete_edges
from src.pivot import run_pivot
from src.cost import calculate_clustering_cost
from src.min_max import min_max_cc, max_disagreement
from src.min_max_lp import (
    MinMaxLP,
    cluster as min_max_lp_cluster,
    LocalObj,
    DegreeDist,
)


# ============================================================
# Defaults: edit these by hand or override with CLI flags
# ============================================================

EGO_IDS = ["348"]
P_DELETE_VALUES = [0.05, 0.15, 0.25, 0.40]
SEEDS = [1]
PIVOT_SEEDS = list(range(1, 51))

D_HAT_MIN = 1
LAMBDA_VALUES = [5, 8, 12]

MIN_MAX_LP_R = 0.4
MIN_MAX_LP_R2 = 0.4
MIN_MAX_LP_METHOD = 2
MIN_MAX_LP_NORM = np.inf

DEFAULT_OUTPUT = "results/processed/minmax_facebook_experiment_flat.csv"


# ============================================================
# Flat output columns
# No correlation-clustering all-pairs LP/ILP columns are included.
# ============================================================

FIELDNAMES = [
    "ego_id",
    "n",

    "complete_pivot_best_cost",
    "complete_pivot_average_cost",

    "p_delete",

    "edge_pivot_best_cost",
    "edge_pivot_average_cost",

    "complete_min_max_cc_computed",
    "complete_min_max_cc_cluster_count",
    "complete_min_max_cc_max_disagreement",
    "complete_min_max_cc_d_hat",
    "complete_min_max_cc_lambda",
    "complete_min_max_cc_runtime_seconds",

    "complete_min_max_lp_computed",
    "complete_min_max_lp_cost",
    "complete_min_max_lp_rounding_cost",
    "complete_min_max_lp_max_disagreement_vertex",
    "complete_min_max_lp_cluster_count",
    "complete_min_max_lp_r",
    "complete_min_max_lp_r2",
    "complete_min_max_lp_method",
    "complete_min_max_lp_norm",
    "complete_min_max_lp_runtime_seconds",
    "complete_min_max_lp_rounding_runtime_seconds",
    "complete_min_max_lp_total_runtime_seconds",

    "edge_min_max_cc_computed",
    "edge_min_max_cc_cluster_count",
    "edge_min_max_cc_max_disagreement",
    "edge_min_max_cc_d_hat",
    "edge_min_max_cc_lambda",
    "edge_min_max_cc_runtime_seconds",

    "edge_min_max_lp_computed",
    "edge_min_max_lp_cost",
    "edge_min_max_lp_rounding_cost",
    "edge_min_max_lp_max_disagreement_vertex",
    "edge_min_max_lp_cluster_count",
    "edge_min_max_lp_r",
    "edge_min_max_lp_r2",
    "edge_min_max_lp_method",
    "edge_min_max_lp_norm",
    "edge_min_max_lp_runtime_seconds",
    "edge_min_max_lp_rounding_runtime_seconds",
    "edge_min_max_lp_total_runtime_seconds",
]


# ============================================================
# Small helpers
# ============================================================

def parse_list(raw, cast=str):
    if raw is None:
        return None
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def norm_for_csv(norm):
    return "inf" if norm == np.inf or norm == math.inf else norm


def get_all_nodes_from_edges_and_circles(edge_nodes, circles):
    circle_nodes = set()
    for circle in circles:
        circle_nodes.update(circle["nodes"])
    return sorted(edge_nodes | circle_nodes)


def facebook_file_path(ego_id, suffix):
    candidates = [
        Path(f"data/facebook/facebook_3/{ego_id}.{suffix}"),
        Path(f"data/facebook/{ego_id}.{suffix}"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Could not find Facebook file for ego_id={ego_id}, suffix={suffix}. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def max_positive_degree(S):
    return int((S == 1).sum(axis=1).max())


def powers_of_two_up_to(max_value, min_value=1):
    values = []
    current = 1
    while current <= max_value:
        if current >= min_value:
            values.append(current)
        current *= 2
    return values


def run_pivot_multiple(S, pivot_seeds):
    best_cost = None
    total_cost = 0.0

    for pivot_seed in pivot_seeds:
        clusters, _ = run_pivot(S, pivot_seed)
        cost = calculate_clustering_cost(S, clusters)
        total_cost += cost

        if best_cost is None or cost < best_cost:
            best_cost = cost

    average_cost = total_cost / len(pivot_seeds)

    return {
        "best_cost": best_cost,
        "average_cost": average_cost,
    }


def compute_min_max_cc(S, d_hat, lambda_value):
    start = time.time()
    clustering = min_max_cc(S, d_hat, lambda_value)
    runtime = time.time() - start

    return {
        "computed": True,
        "cluster_count": len(clustering),
        "max_disagreement": max_disagreement(clustering, S),
        "d_hat": d_hat,
        "lambda": lambda_value,
        "runtime_seconds": round(runtime, 6),
    }


def compute_min_max_lp(S, r, r2, method, norm):
    total_start = time.time()

    (
        lp_cost,
        distances,
        L_t_vals,
        neighborsR,
        neighborsR2,
        lp_runtime,
    ) = MinMaxLP(S, r, r2, method)

    clustering, rounding_runtime = min_max_lp_cluster(
        distances,
        L_t_vals,
        neighborsR,
        neighborsR2,
        r,
        r2,
    )

    pos_degrees = DegreeDist(S)

    (
        _disagreement_vector,
        rounding_cost,
        max_disagreement_vertex,
    ) = LocalObj(
        S,
        clustering,
        pos_degrees,
        norm,
    )

    total_runtime = time.time() - total_start

    return {
        "computed": True,
        "lp_cost": lp_cost,
        "rounding_cost": rounding_cost,
        "max_disagreement_vertex": max_disagreement_vertex,
        "cluster_count": len(clustering),
        "r": r,
        "r2": r2,
        "method": method,
        "norm": norm_for_csv(norm),
        "lp_runtime_seconds": round(lp_runtime, 6),
        "rounding_runtime_seconds": round(rounding_runtime, 6),
        "total_runtime_seconds": round(total_runtime, 6),
    }


def empty_min_max_lp(r, r2, method, norm):
    return {
        "computed": False,
        "lp_cost": "",
        "rounding_cost": "",
        "max_disagreement_vertex": "",
        "cluster_count": "",
        "r": r,
        "r2": r2,
        "method": method,
        "norm": norm_for_csv(norm),
        "lp_runtime_seconds": "",
        "rounding_runtime_seconds": "",
        "total_runtime_seconds": "",
    }


def load_existing_rows(output_file):
    if not os.path.exists(output_file):
        return []

    with open(output_file, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def row_key(row):
    return (
        str(row.get("ego_id", "")),
        f"{float(row.get('p_delete', 0.0)):.8f}",
        str(row.get("complete_min_max_cc_d_hat", "")),
        str(row.get("complete_min_max_cc_lambda", "")),
    )


def make_key(ego_id, p_delete, d_hat, lambda_value):
    return (
        str(ego_id),
        f"{float(p_delete):.8f}",
        str(d_hat),
        str(lambda_value),
    )


def save_rows(output_file, rows):
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(
    ego_id,
    n,
    p_delete,
    complete_pivot,
    edge_pivot,
    complete_cc,
    edge_cc,
    complete_lp,
    edge_lp,
):
    return {
        "ego_id": ego_id,
        "n": n,

        "complete_pivot_best_cost": complete_pivot["best_cost"],
        "complete_pivot_average_cost": round(complete_pivot["average_cost"], 6),

        "p_delete": p_delete,

        "edge_pivot_best_cost": edge_pivot["best_cost"],
        "edge_pivot_average_cost": round(edge_pivot["average_cost"], 6),

        "complete_min_max_cc_computed": complete_cc["computed"],
        "complete_min_max_cc_cluster_count": complete_cc["cluster_count"],
        "complete_min_max_cc_max_disagreement": complete_cc["max_disagreement"],
        "complete_min_max_cc_d_hat": complete_cc["d_hat"],
        "complete_min_max_cc_lambda": complete_cc["lambda"],
        "complete_min_max_cc_runtime_seconds": complete_cc["runtime_seconds"],

        "complete_min_max_lp_computed": complete_lp["computed"],
        "complete_min_max_lp_cost": complete_lp["lp_cost"],
        "complete_min_max_lp_rounding_cost": complete_lp["rounding_cost"],
        "complete_min_max_lp_max_disagreement_vertex": complete_lp["max_disagreement_vertex"],
        "complete_min_max_lp_cluster_count": complete_lp["cluster_count"],
        "complete_min_max_lp_r": complete_lp["r"],
        "complete_min_max_lp_r2": complete_lp["r2"],
        "complete_min_max_lp_method": complete_lp["method"],
        "complete_min_max_lp_norm": complete_lp["norm"],
        "complete_min_max_lp_runtime_seconds": complete_lp["lp_runtime_seconds"],
        "complete_min_max_lp_rounding_runtime_seconds": complete_lp["rounding_runtime_seconds"],
        "complete_min_max_lp_total_runtime_seconds": complete_lp["total_runtime_seconds"],

        "edge_min_max_cc_computed": edge_cc["computed"],
        "edge_min_max_cc_cluster_count": edge_cc["cluster_count"],
        "edge_min_max_cc_max_disagreement": edge_cc["max_disagreement"],
        "edge_min_max_cc_d_hat": edge_cc["d_hat"],
        "edge_min_max_cc_lambda": edge_cc["lambda"],
        "edge_min_max_cc_runtime_seconds": edge_cc["runtime_seconds"],

        "edge_min_max_lp_computed": edge_lp["computed"],
        "edge_min_max_lp_cost": edge_lp["lp_cost"],
        "edge_min_max_lp_rounding_cost": edge_lp["rounding_cost"],
        "edge_min_max_lp_max_disagreement_vertex": edge_lp["max_disagreement_vertex"],
        "edge_min_max_lp_cluster_count": edge_lp["cluster_count"],
        "edge_min_max_lp_r": edge_lp["r"],
        "edge_min_max_lp_r2": edge_lp["r2"],
        "edge_min_max_lp_method": edge_lp["method"],
        "edge_min_max_lp_norm": edge_lp["norm"],
        "edge_min_max_lp_runtime_seconds": edge_lp["lp_runtime_seconds"],
        "edge_min_max_lp_rounding_runtime_seconds": edge_lp["rounding_runtime_seconds"],
        "edge_min_max_lp_total_runtime_seconds": edge_lp["total_runtime_seconds"],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Facebook pivot + MinMaxLP + MinMaxCC grid experiments. "
            "Does NOT run correlation-clustering all-pairs LP/ILP."
        )
    )

    parser.add_argument("--ego-ids", default=",".join(EGO_IDS))
    parser.add_argument("--p-delete-values", default=",".join(str(x) for x in P_DELETE_VALUES))
    parser.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    parser.add_argument("--pivot-seeds", default=",".join(str(x) for x in PIVOT_SEEDS))
    parser.add_argument("--lambda-values", default=",".join(str(x) for x in LAMBDA_VALUES))
    parser.add_argument("--min-d-hat", type=int, default=D_HAT_MIN)

    parser.add_argument("--r", type=float, default=MIN_MAX_LP_R)
    parser.add_argument("--r2", type=float, default=MIN_MAX_LP_R2)
    parser.add_argument("--method", type=int, default=MIN_MAX_LP_METHOD)

    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-lp", action="store_true")

    args = parser.parse_args()

    ego_ids = parse_list(args.ego_ids, str)
    p_delete_values = parse_list(args.p_delete_values, float)
    seeds = parse_list(args.seeds, int)
    pivot_seeds = parse_list(args.pivot_seeds, int)
    lambda_values = parse_list(args.lambda_values, int)
    norm = MIN_MAX_LP_NORM

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    output_rows = load_existing_rows(args.output)
    existing_keys = {row_key(row) for row in output_rows}

    print("Output file:", args.output)
    print("Existing rows:", len(output_rows))
    print("Ego ids:", ego_ids)
    print("p_delete values:", p_delete_values)
    print("seeds:", seeds)
    print("lambda values:", lambda_values)
    print("MinMaxLP method:", args.method)
    print("Skip LP:", args.skip_lp)

    total_new_rows = 0

    for ego_id in ego_ids:
        print("\n" + "=" * 80)
        print(f"Loading Facebook ego graph {ego_id}")
        print("=" * 80)

        edges_file = facebook_file_path(ego_id, "edges")
        circles_file = facebook_file_path(ego_id, "circles")

        edge_nodes, facebook_edges = load_facebook_ego_edges(edges_file)
        circles = load_facebook_circles(circles_file)

        all_nodes = get_all_nodes_from_edges_and_circles(edge_nodes, circles)
        n = len(all_nodes)

        complete_S, _node_to_index, _positive_count, _negative_count = (
            build_complete_signed_matrix_from_facebook_sample(
                all_nodes,
                facebook_edges,
            )
        )

        print(f"n={n}")
        print("Running complete pivot once...")
        complete_pivot = run_pivot_multiple(complete_S, pivot_seeds)

        if args.skip_lp:
            complete_lp = empty_min_max_lp(args.r, args.r2, args.method, norm)
        else:
            print("Running complete MinMaxLP once...")
            complete_lp = compute_min_max_lp(
                complete_S,
                r=args.r,
                r2=args.r2,
                method=args.method,
                norm=norm,
            )

        for p_delete in p_delete_values:
            for seed in seeds:
                print("\n" + "-" * 80)
                print(f"ego_id={ego_id} | p_delete={p_delete} | seed={seed}")
                print("-" * 80)

                edge_S, num_edges_deleted = delete_edges(complete_S, p_delete, seed)
                print("Deleted edges:", num_edges_deleted)

                print("Running edge-deleted pivot once...")
                edge_pivot = run_pivot_multiple(edge_S, pivot_seeds)

                if args.skip_lp:
                    edge_lp = empty_min_max_lp(args.r, args.r2, args.method, norm)
                else:
                    print("Running edge-deleted MinMaxLP once...")
                    edge_lp = compute_min_max_lp(
                        edge_S,
                        r=args.r,
                        r2=args.r2,
                        method=args.method,
                        norm=norm,
                    )

                max_d_hat = max(
                    max_positive_degree(complete_S),
                    max_positive_degree(edge_S),
                )
                d_hat_values = powers_of_two_up_to(
                    max_d_hat,
                    min_value=max(1, args.min_d_hat),
                )

                print("d_hat values:", d_hat_values)

                for d_hat in d_hat_values:
                    for lambda_value in lambda_values:
                        key = make_key(ego_id, p_delete, d_hat, lambda_value)
                        if key in existing_keys:
                            print(f"Skipping existing row: d_hat={d_hat}, lambda={lambda_value}")
                            continue

                        print(f"Running MinMaxCC: d_hat={d_hat}, lambda={lambda_value}")

                        complete_cc = compute_min_max_cc(
                            complete_S,
                            d_hat=d_hat,
                            lambda_value=lambda_value,
                        )
                        edge_cc = compute_min_max_cc(
                            edge_S,
                            d_hat=d_hat,
                            lambda_value=lambda_value,
                        )

                        flat_row = flatten_result(
                            ego_id=ego_id,
                            n=n,
                            p_delete=p_delete,
                            complete_pivot=complete_pivot,
                            edge_pivot=edge_pivot,
                            complete_cc=complete_cc,
                            edge_cc=edge_cc,
                            complete_lp=complete_lp,
                            edge_lp=edge_lp,
                        )

                        output_rows.append(flat_row)
                        existing_keys.add(key)
                        total_new_rows += 1

                        save_rows(args.output, output_rows)
                        print(f"Saved checkpoint. Total new rows: {total_new_rows}")

    print("\nDone.")
    print("Total new rows:", total_new_rows)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
