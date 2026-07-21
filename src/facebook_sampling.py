import numpy as np


def load_facebook_ego_edges(edges_file):
    """
    Load one Facebook ego-network edge file.

    Expected format:
        node1 node2
    """
    edges = set()
    nodes = set()

    with open(edges_file, "r", encoding="utf-8") as f:
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


def build_complete_signed_matrix_from_facebook_sample(
    sampled_nodes,
    facebook_edges,
):
    """
    Build a complete signed graph from Facebook edge-file nodes.

    Rule:
        existing Facebook friendship edge = +1
        missing friendship edge = -1
        diagonal = 0

    Important:
        sampled_nodes should contain only nodes occurring in the
        Facebook .edges file when reproducing the paper instances.
    """
    sampled_nodes = list(sampled_nodes)
    n = len(sampled_nodes)

    if len(set(sampled_nodes)) != n:
        raise ValueError("sampled_nodes contains duplicate node IDs")

    node_to_index = {
        node: index
        for index, node in enumerate(sampled_nodes)
    }

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

