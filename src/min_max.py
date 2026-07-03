import numpy as np
from collections import deque
import math

def min_max_cc(adj, d_hat, lam):
    """
    Min-Max Correlation Clustering  (O(n^2) algorithm).
 
    Parameters
    ----------
    adj   : (n, n) int ndarray
              +1 = positive edge, -1 = negative edge, 0 = missing/deleted.
    d_hat : float
              Threshold parameter.  Set to  max(c * log n,  OPT(G')).
    lam   : float
              Lambda.  Must satisfy  lam > 4 / (1 - q).  Use 5.0 for q = 0.
 
    Returns
    -------
    partition : list of sets of int
    """
    n   = adj.shape[0]
    pos = (adj == 1).astype(np.int32)      # 0/1 positive-adjacency matrix
 
    # |N+(u) ∩ N+(v)| for all pairs — one BLAS matmul
    inter   = pos @ pos.T                  # shape (n, n)
    pos_deg = inter.diagonal().copy()      # pos_deg[v] = |N+(v)|
 
    # High-degree set S
    S = np.where(pos_deg > lam * d_hat)[0]
 
    # Similarity graph H: (u,v) in H iff u in S and |N+(u) ∩ N+(v)| > 2*d_hat
    adj_H = [set() for _ in range(n)]
    for u in S:
        for v in np.where(inter[u] > 2 * d_hat)[0]:
            adj_H[u].add(int(v))
            adj_H[int(v)].add(int(u))
 
    # Connected components of H (BFS)
    visited   = np.zeros(n, dtype=bool)
    partition = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        if not adj_H[start]:              # isolated vertex → singleton
            partition.append({start})
            continue
        comp  = set()
        queue = deque([start])
        while queue:
            u = queue.popleft()
            comp.add(u)
            for w in adj_H[u]:
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)
        partition.append(comp)
 
    return partition
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def vertex_disagreement(v, cluster, adj):
    """Disagreement of vertex v given its cluster (a set)."""
    dis = 0
    for u in range(adj.shape[0]):
        if u == v:
            continue
        same = u in cluster
        if adj[v, u] ==  1 and not same:
            dis += 1
        if adj[v, u] == -1 and same:
            dis += 1
    return dis
 
 
def max_disagreement(partition, adj):
    """Max disagreement over all vertices."""
    vtoc = {v: c for c in partition for v in c}
    return max(vertex_disagreement(v, vtoc[v], adj) for v in range(adj.shape[0]))
 