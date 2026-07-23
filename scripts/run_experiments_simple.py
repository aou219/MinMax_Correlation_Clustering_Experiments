#!/usr/bin/env python3
"""Small runner for the final Facebook experiments."""

import csv
import gc
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from src import experiment_helpers as h
from src.edge_deletion import delete_edges
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_ego_edges,
)


# ---------- settings ----------

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/research_tables/minmax_facebook_grid_runs_flat.csv"

# Smallest n first.
EGO_IDS = ["3980", "698", "414", "686"]
P_DELETE = [0.05, 0.15, 0.25, 0.4]
SEEDS = range(1, 31)
PIVOT_SEEDS = range(1, 101)

RUN_PIVOT = False
RUN_NORMAL_LP = False
RUN_MINMAX_CC = False
RUN_MINMAX_LP = True       # Includes rounding.

RUN_COMPLETE = True
RUN_DELETED = True
OVERWRITE_FINISHED = False

D_HAT = 8
LAMBDA = 5
R = 0.4
R2 = 0.4
METHOD = 2


# ---------- files ----------

CLUSTER_COLUMNS = [
    "complete_min_max_lp_clustering_json",
    "edge_min_max_lp_clustering_json",
]


def read_csv():
    with TABLE.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for column in CLUSTER_COLUMNS:
            if column not in fields:
                fields.append(column)
        rows = [
            {field: row.get(field, "") for field in fields}
            for row in reader
        ]
    return fields, rows


def write_csv(fields, rows):
    fd, tmp = tempfile.mkstemp(
        prefix=TABLE.name + ".", suffix=".tmp", dir=TABLE.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, TABLE)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def backup():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = TABLE.with_name(f"{TABLE.stem}_backup_{stamp}.csv")
    shutil.copy2(TABLE, path)
    print("Backup:", path)


def filled(rows, columns):
    return rows and all(
        str(row.get(column, "")).strip()
        for row in rows
        for column in columns
    )


def set_values(rows, values):
    for row in rows:
        for column, output in values.items():
            if isinstance(output, np.generic):
                output = output.item()
            row[column] = "" if output is None else str(output)


# ---------- graph ----------

def edges_file(ego_id):
    for path in [
        ROOT / f"data/facebook/{ego_id}.edges",
        ROOT / f"data/facebook/facebook_3/{ego_id}.edges",
    ]:
        if path.exists():
            return path
    raise FileNotFoundError(f"No .edges file for ego {ego_id}")


def complete_graph(ego_id):
    nodes, edges = load_facebook_ego_edges(str(edges_file(ego_id)))
    matrix, node_to_index, _, _ = (
        build_complete_signed_matrix_from_facebook_sample(
            sorted(nodes), edges
        )
    )
    index_to_node = {
        int(index): node for node, index in node_to_index.items()
    }
    return matrix, index_to_node


def cluster_json(clustering, index_to_node):
    clusters = [
        [index_to_node[int(vertex)] for vertex in cluster]
        for cluster in clustering
    ]
    return json.dumps(clusters, ensure_ascii=False, separators=(",", ":"))


# ---------- algorithms ----------

def run_pivot(matrix):
    return h.run_pivot_multiple(matrix, PIVOT_SEEDS)


def run_normal_lp(matrix):
    return h.compute_all_pairs_data(
        matrix, compute_lp=True, compute_ilp=False
    )


def run_minmax_cc(matrix):
    return h.compute_min_max_cc_data(
        matrix, compute_min_max=True, param_1=D_HAT, param_2=LAMBDA
    )


def run_minmax_lp(matrix):
    result = h.compute_min_max_lp_data(
        matrix,
        compute_min_max_lp=True,
        r=R,
        r2=R2,
        method=METHOD,
        norm=np.inf,
    )
    required = [
        "lp_cost",
        "clustering",
        "cluster_count",
        "rounding_cost",
        "max_disagreement_vertex",
        "lp_runtime_seconds",
        "rounding_runtime_seconds",
        "total_runtime_seconds",
    ]
    missing = [key for key in required if result.get(key) is None]
    if missing:
        raise RuntimeError("Missing MinMaxLP output: " + ", ".join(missing))
    return result


ALGORITHMS = [
    ("pivot", RUN_PIVOT, run_pivot),
    ("normal_lp", RUN_NORMAL_LP, run_normal_lp),
    ("minmax_cc", RUN_MINMAX_CC, run_minmax_cc),
    ("minmax_lp", RUN_MINMAX_LP, run_minmax_lp),
]


def output_columns(name, scope):
    if name == "pivot":
        return [
            f"{scope}_pivot_best_cost",
            f"{scope}_pivot_average_cost",
        ]
    if name == "normal_lp":
        return [
            "complete_lp_cost"
            if scope == "complete"
            else "edge_all_pairs_lp_cost"
        ]
    if name == "minmax_cc":
        p = f"{scope}_min_max_cc"
        return [
            f"{p}_computed", f"{p}_cluster_count",
            f"{p}_max_disagreement", f"{p}_d_hat",
            f"{p}_lambda", f"{p}_runtime_seconds",
        ]

    p = f"{scope}_min_max_lp"
    return [
        f"{p}_computed", f"{p}_cost", f"{p}_rounding_cost",
        f"{p}_max_disagreement_vertex", f"{p}_cluster_count",
        f"{p}_clustering_json", f"{p}_r", f"{p}_r2",
        f"{p}_method", f"{p}_norm", f"{p}_runtime_seconds",
        f"{p}_rounding_runtime_seconds", f"{p}_total_runtime_seconds",
    ]


def save_result(name, scope, rows, result, index_to_node):
    if name == "pivot":
        values = {
            f"{scope}_pivot_best_cost": result["best_cost"],
            f"{scope}_pivot_average_cost": result["average_cost"],
        }
    elif name == "normal_lp":
        column = (
            "complete_lp_cost"
            if scope == "complete"
            else "edge_all_pairs_lp_cost"
        )
        values = {column: result["lp_cost"]}
    elif name == "minmax_cc":
        p = f"{scope}_min_max_cc"
        values = {
            f"{p}_computed": result["computed"],
            f"{p}_cluster_count": result["cluster_count"],
            f"{p}_max_disagreement": result["max_disagreement"],
            f"{p}_d_hat": result["d_hat"],
            f"{p}_lambda": result["lambda"],
            f"{p}_runtime_seconds": result["runtime_seconds"],
        }
    else:
        p = f"{scope}_min_max_lp"
        values = {
            f"{p}_computed": result["computed"],
            f"{p}_cost": result["lp_cost"],
            f"{p}_rounding_cost": result["rounding_cost"],
            f"{p}_max_disagreement_vertex":
                result["max_disagreement_vertex"],
            f"{p}_cluster_count": result["cluster_count"],
            f"{p}_clustering_json":
                cluster_json(result["clustering"], index_to_node),
            f"{p}_r": result["r"],
            f"{p}_r2": result["r2"],
            f"{p}_method": result["method"],
            f"{p}_norm": result["norm"],
            f"{p}_runtime_seconds": result["lp_runtime_seconds"],
            f"{p}_rounding_runtime_seconds":
                result["rounding_runtime_seconds"],
            f"{p}_total_runtime_seconds":
                result["total_runtime_seconds"],
        }
    set_values(rows, values)


def cleanup():
    gc.collect()
    try:
        import gurobipy as gp
        gp.disposeDefaultEnv()
    except Exception:
        pass


# ---------- main loop ----------

def main():
    fields, rows = read_csv()
    backup()

    enabled = [(name, function) for name, on, function in ALGORITHMS if on]
    print("Enabled:", ", ".join(name for name, _ in enabled))
    print("Ego order:", ", ".join(EGO_IDS))

    for ego_id in EGO_IDS:
        print(f"\n=== ego {ego_id} ===")
        complete, index_to_node = complete_graph(ego_id)
        ego_rows = [row for row in rows if row["ego_id"].strip() == ego_id]

        for row in ego_rows:
            row["n"] = str(len(complete))

        if RUN_COMPLETE:
            for name, function in enabled:
                columns = output_columns(name, "complete")
                if filled(ego_rows, columns) and not OVERWRITE_FINISHED:
                    print("skip complete", name)
                    continue
                print("run complete", name)
                result = function(complete)
                save_result(name, "complete", ego_rows, result, index_to_node)
                write_csv(fields, rows)
                cleanup()

        if RUN_DELETED:
            for p_delete in P_DELETE:
                for seed in SEEDS:
                    matches = [
                        row for row in ego_rows
                        if abs(float(row["p_delete"]) - p_delete) < 1e-12
                        and int(float(row["seed"])) == seed
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Missing/duplicate row: {ego_id}, "
                            f"{p_delete}, seed {seed}"
                        )
                    target = matches[0]
                    deleted = None

                    for name, function in enabled:
                        columns = output_columns(name, "edge")
                        if filled([target], columns) and not OVERWRITE_FINISHED:
                            continue
                        if deleted is None:
                            deleted, _ = delete_edges(
                                complete.copy(), p_delete, seed
                            )
                        print(
                            f"run {name}: ego={ego_id}, "
                            f"p={p_delete}, seed={seed}"
                        )
                        result = function(deleted)
                        save_result(
                            name, "edge", [target], result, index_to_node
                        )
                        write_csv(fields, rows)
                        cleanup()

        del complete
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
