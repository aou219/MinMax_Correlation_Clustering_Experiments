import numpy as np

def delete_edges(S, p_delete, seed):
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    S_deleted = S.copy()
    edges_deleted_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_delete:
                S_deleted[i, j] = 0
                S_deleted[j, i] = 0  # keep symmetry
                edges_deleted_count+=1

    return S_deleted, edges_deleted_count