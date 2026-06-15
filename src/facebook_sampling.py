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

import random

def sample_nodes_from_circles(circles, cluster_sizes, seed=None):
    rng = random.Random(seed)

    circles_sorted = sorted(
        circles,
        key=lambda c: len(c["nodes"]),
        reverse=True
    )

    selected_circles = []
    sampled_nodes = []
    used_nodes = set()

    circle_idx = 0

    for n_nodes in cluster_sizes:

        # zoek best matching circle (greedy, but realistic)
        found = False

        for _ in range(len(circles_sorted)):

            circle = circles_sorted[circle_idx % len(circles_sorted)]
            circle_idx += 1

            available_nodes = circle["nodes"]

            # kleine overlap toegestaan → alleen soft filtering
            candidate_nodes = list(set(available_nodes))

            if len(candidate_nodes) < n_nodes:
                continue

            chosen = rng.sample(candidate_nodes, n_nodes)

            selected_circles.append({
                "name": circle["name"],
                "nodes": chosen
            })

            sampled_nodes.extend(chosen)
            used_nodes.update(chosen)

            found = True
            break

        if not found:
            raise ValueError(f"Could not sample {n_nodes} nodes realistically")

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
