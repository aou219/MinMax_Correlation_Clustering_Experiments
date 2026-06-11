import random
import numpy as np


def load_facebook_ego_edges(edges_file):
    """
    Load one Facebook ego-network edge file.

    Expected format:
        node1 node2
    """
    edges = set()
    nodes = set()

    with open(edges_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            if u == v:
                continue

            edge = tuple(sorted((u, v)))
            edges.add(edge)
            nodes.add(u)
            nodes.add(v)

    return nodes, edges


def load_facebook_circles(circles_file):
    """
    Load one Facebook circles file.

    Expected format:
        circle_name node1 node2 node3 ...
    """
    circles = []

    with open(circles_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            circle_name = parts[0]
            circle_nodes = [int(node) for node in parts[1:]]

            circles.append({
                "name": circle_name,
                "nodes": circle_nodes
            })

    return circles


def sample_nodes_from_circles(
    circles,
    num_circles=4,
    nodes_per_circle=25,
    seed=None
):
    """
    Sample nodes from multiple Facebook circles.

    The function tries to avoid overlapping nodes, so that the sampled
    circles are clearer as communities.
    """
    rng = random.Random(seed)

    circles_sorted = sorted(
        circles,
        key=lambda circle: len(circle["nodes"]),
        reverse=True
    )

    selected_circles = []
    sampled_nodes = []
    used_nodes = set()

    for circle in circles_sorted:
        available_nodes = [
            node for node in circle["nodes"]
            if node not in used_nodes
        ]

        if len(available_nodes) < nodes_per_circle:
            continue

        chosen_nodes = rng.sample(available_nodes, nodes_per_circle)

        selected_circles.append({
            "name": circle["name"],
            "nodes": chosen_nodes
        })

        sampled_nodes.extend(chosen_nodes)
        used_nodes.update(chosen_nodes)

        if len(selected_circles) == num_circles:
            break

    if len(selected_circles) < num_circles:
        raise ValueError("Not enough large non-overlapping circles found.")

    return sampled_nodes, selected_circles


def build_complete_signed_matrix_from_facebook_sample(sampled_nodes, facebook_edges):
    """
    Build a complete signed graph from sampled Facebook nodes.

    Rule:
        existing Facebook friendship edge = +1
        missing friendship edge = -1
    """
    n = len(sampled_nodes)
    node_to_index = {node: index for index, node in enumerate(sampled_nodes)}

    S = np.zeros((n, n), dtype=int)

    positive_count = 0
    negative_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            original_u = sampled_nodes[i]
            original_v = sampled_nodes[j]

            edge = tuple(sorted((original_u, original_v)))

            if edge in facebook_edges:
                sign = 1
                positive_count += 1
            else:
                sign = -1
                negative_count += 1

            S[i, j] = sign
            S[j, i] = sign

    return S, node_to_index, positive_count, negative_count
