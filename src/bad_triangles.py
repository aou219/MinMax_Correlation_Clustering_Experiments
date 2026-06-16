import heapq
import numpy as np
import heapq

def is_bad_triangle(S, i, j, k):
    """Return True if the triangle (i,j,k) is a bad triangle."""
    return(
        (S[i, j] == 1 and S[i, k] == 1 and S[j, k] == -1)
        or (S[i,j] == 1 and S[i,k]==-1 and S[j,k]== 1)
        or (S[i,j] == -1 and S[i,k]==1 and S[j,k]==1)
    )

def find_bad_triangles(S):
    """Return a list of all bad triangles in the graph S."""
    n = S.shape[0]
    bad_triangle_list = []
    # To keep an order, we force the first node to be the smallest, then the second node, then the last.
    # This way 012 will get iterated and 021 won't
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1, n):
                if(is_bad_triangle(S, i, j, k)):
                    bad_triangle_list.append((i,j,k))
    return bad_triangle_list

def count_bad_triangles(bad_triangle_list):
    return len(bad_triangle_list)

def make_edge_to_triangle_map(bad_triangles):
    edge_to_triangles = {}
    for tri in bad_triangles:
        i, j, k = tri
        for u, v in [(i,j), (i,k), (j,k)]:
            edge = tuple(sorted((u,v)))
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = set()
            edge_to_triangles[edge].add(tri)
    return edge_to_triangles

def _triangle_edges(triangle):
    """Return the three sorted edges of a triangle."""
    i, j, k = triangle

    return [
        tuple(sorted((i, j))),
        tuple(sorted((i, k))),
        tuple(sorted((j, k))),
    ]


def _greedy_edge_disjoint_bad_triangles(edge_to_triangles, choose="most"):
    """
    Greedy construction of an edge-disjoint set of bad triangles.

    choose="most":
        Pick an edge that is contained in the most remaining bad triangles.

    choose="least":
        Pick an edge that is contained in the fewest remaining bad triangles.

    The returned triangles are edge-disjoint.
    The input edge_to_triangles is not modified.
    """
    local_edge_to_triangles = {
        edge: set(triangles)
        for edge, triangles in edge_to_triangles.items()
    }

    remaining_triangles = set()

    for triangles in local_edge_to_triangles.values():
        remaining_triangles.update(triangles)

    selected_triangles = []

    while remaining_triangles:
        edge_counts = []

        for edge, triangles in local_edge_to_triangles.items():
            count = len(triangles & remaining_triangles)

            if count > 0:
                edge_counts.append((count, edge))

        if not edge_counts:
            break

        if choose == "most":
            _, selected_edge = max(edge_counts, key=lambda x: (x[0], x[1]))
        elif choose == "least":
            _, selected_edge = min(edge_counts, key=lambda x: (x[0], x[1]))
        else:
            raise ValueError("choose must be 'most' or 'least'")

        candidates = sorted(
            local_edge_to_triangles[selected_edge] & remaining_triangles
        )

        if not candidates:
            continue

        triangle = candidates[0]
        selected_triangles.append(triangle)

        # Remove every remaining bad triangle that shares an edge
        # with the selected triangle.
        for edge in _triangle_edges(triangle):
            remaining_triangles -= local_edge_to_triangles.get(edge, set())

    return selected_triangles


def find_edge_disjoint_bad_triangles_min(edge_to_triangles):
    """
    Greedy low edge-disjoint bad-triangle bound.

    It starts with edges that occur in many bad triangles.
    This often gives a smaller edge-disjoint set.
    """
    return _greedy_edge_disjoint_bad_triangles(
        edge_to_triangles,
        choose="most"
    )


def find_edge_disjoint_bad_triangles_max(edge_to_triangles):
    """
    Greedy high edge-disjoint bad-triangle bound.

    It starts with edges that occur in few bad triangles.
    This often gives a larger edge-disjoint set.
    """
    return _greedy_edge_disjoint_bad_triangles(
        edge_to_triangles,
        choose="least"
    )
