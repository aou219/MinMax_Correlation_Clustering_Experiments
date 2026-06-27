"""Run the complete, resumable Facebook ego graph experiment grid."""

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
    run_complete_graph,
    run_edge_deleted_graph,
    save_complete_result,
    save_p_delete_result,
)
from facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_circles,
    load_facebook_ego_edges,
)


EGO_IDS = ("414", "686", "698", "3980")
EGO_WITHOUT_OPTIMIZATION = {"686"}


def all_nodes(edge_nodes, circles):
    circle_nodes = set()
    for circle in circles:
        circle_nodes.update(circle["nodes"])
    return sorted(edge_nodes | circle_nodes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego-id", choices=EGO_IDS)
    parser.add_argument("--p-delete", type=float, choices=P_DELETE_VALUES)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ego_ids = [args.ego_id] if args.ego_id else list(EGO_IDS)
    p_deletes = (
        [args.p_delete] if args.p_delete is not None
        else list(P_DELETE_VALUES)
    )

    for ego_id in ego_ids:
        edges_path = root / "data" / "facebook" / f"{ego_id}.edges"
        circles_path = root / "data" / "facebook" / f"{ego_id}.circles"
        edge_nodes, facebook_edges = load_facebook_ego_edges(str(edges_path))
        circles = load_facebook_circles(str(circles_path))
        nodes = all_nodes(edge_nodes, circles)
        S, _, positive_count, negative_count = (
            build_complete_signed_matrix_from_facebook_sample(
                nodes, facebook_edges
            )
        )

        include_optimization = ego_id not in EGO_WITHOUT_OPTIMIZATION
        suffix = "_full" if include_optimization else "_full_without_ilp"
        output = (
            root
            / "results"
            / "experiments_results_facebook"
            / "full"
            / f"fb_ego{ego_id}{suffix}.json"
        )
        shared = {
            "graph_type": "facebook_full_ego",
            "ego_id": ego_id,
            "num_nodes": len(nodes),
            "num_facebook_edges": len(facebook_edges),
            "num_circles": len(circles),
            "circle_sizes": sorted(
                [len(circle["nodes"]) for circle in circles],
                reverse=True,
            ),
            "pivot_seeds": list(PIVOT_SEEDS),
            "p_delete_values": list(P_DELETE_VALUES),
            "positive_edges": positive_count,
            "negative_edges": negative_count,
            "sample_type": "full_ego_network",
            "signing_rule": {
                "existing_facebook_friendship": "+1",
                "missing_friendship_inside_full_ego_network": "-1",
                "deleted_edge": "0",
            },
            "optimization_included": include_optimization,
        }
        initialize_file(output, shared, seeds=[1])

        if not has_complete_result(output, 1):
            started = time.time()
            complete, approx, sparse_approx = run_complete_graph(
                S,
                include_optimization=include_optimization,
            )
            save_complete_result(
                output, 1, complete, approx, sparse_approx
            )
            print(
                f"{output.name}, complete graph: "
                f"{format_runtime(time.time() - started)}",
                flush=True,
            )

        for p_delete in p_deletes:
            if has_p_delete_result(output, 1, p_delete):
                continue
            result = run_edge_deleted_graph(
                S,
                p_delete,
                seed=1,
                include_optimization=include_optimization,
            )
            save_p_delete_result(output, 1, p_delete, result)
            print(
                f"{output.name}, p_delete={p_delete:.2f}: "
                f"{format_runtime(result['runtime_seconds'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
