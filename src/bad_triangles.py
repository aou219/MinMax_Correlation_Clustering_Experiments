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

def find_edge_disjoint_bad_triangles_min(edge_to_triangles):
    """
    Search for edge-disjoint bad triangles by selecting edges with the MOST
    bad triangles first. Once a triangle is selected, all triangles sharing
    its edges are removed. This typically produces a lower bound on the
    number of edge-disjoint bad triangles.

    Note: It can be a bit counter-intuitive — if you pick edges with the most
    bad triangles first, you tend to end up with the **minimum number of
    edge-disjoint triangles**. Conversely, picking edges with the fewest
    bad triangles first tends to give a higher count of disjoint triangles.
    """

    # Max-heap: (-num_triangles, edge)
    heap = [(-len(tris), edge) for edge, tris in edge_to_triangles.items()]
    heapq.heapify(heap)

    disjoint_triangles = set()
    used_triangles = set()

    while heap:
        neg_count, edge = heapq.heappop(heap)
        if -neg_count == 0:
            continue
        # Pick any triangle using this edge
        candidates = [tri for tri in edge_to_triangles[edge] if tri not in used_triangles]
        if not candidates:
            continue
        tri = candidates[0]
        disjoint_triangles.add(tri)
        used_triangles.add(tri)
        # Remove this triangle from all edges it touches
        i, j, k = tri
        for u, v in [(i,j), (i,k), (j,k)]:
            e = tuple(sorted((u,v)))
            for t in edge_to_triangles[e]:
                if t != tri:
                    used_triangles.add(t)
            edge_to_triangles[e] = set()  # all triangles touching this edge are now "used"

        # Rebuild heap
        heap = [(-len(tris), e) for e, tris in edge_to_triangles.items() if len(tris) > 0]
        heapq.heapify(heap)

    return list(disjoint_triangles)

def find_edge_disjoint_bad_triangles_max(edge_to_triangles):
    """
    Search for edge-disjoint bad triangles by selecting edges with the LEAST
    bad triangles first. Once a triangle is selected, all triangles sharing
    its edges are removed. This typically produces a lower bound on the
    number of edge-disjoint bad triangles.

    Note: It can be a bit counter-intuitive — if you pick edges with the most
    bad triangles first, you tend to end up with the **minimum number of
    edge-disjoint triangles**. Conversely, picking edges with the fewest
    bad triangles first tends to give a higher count of disjoint triangles.
    """
    # Min-heap: (num_triangles, edge)
    heap = [(len(tris), edge) for edge, tris in edge_to_triangles.items()]
    heapq.heapify(heap)

    disjoint_triangles = set()
    used_triangles = set()

    while heap:
        count, edge = heapq.heappop(heap)
        if count == 0:
            continue
        # Pick any triangle using this edge that hasn't been used yet
        candidates = [tri for tri in edge_to_triangles[edge] if tri not in used_triangles]
        if not candidates:
            continue
        tri = candidates[0]
        disjoint_triangles.add(tri)
        used_triangles.add(tri)
        # Mark all triangles touching this edge as used
        i, j, k = tri
        for u, v in [(i, j), (i, k), (j, k)]:
            e = tuple(sorted((u, v)))
            for t in edge_to_triangles[e]:
                used_triangles.add(t)
            edge_to_triangles[e] = set()  # clear triangles for this edge

        # Rebuild heap with updated counts
        heap = [(len(tris), e) for e, tris in edge_to_triangles.items() if len(tris) > 0]
        heapq.heapify(heap)

    return list(disjoint_triangles)