# correlation-clustering-experiments
Experiments with Pivot, LP rounding, and ILP formulations for correlation clustering on complete and edge-deleted signed graphs.

## Current implementation

The current code generates a complete signed graph, runs the Pivot algorithm on it and calculates the cost.
The signed graph is first represented as an adjacency matrix `S`, where:

- `1` represents a positive edge;
- `-1` represents a negative edge;
- `0` represents no edge or the diagonal.

The parameter `n=8` determines the number of vertices.
The parameter `p_positive=0.5` means that every edge has a 50% probability of being positive and a 50% probability of being negative.
The parameter `seed=42` is used to make the random graph generation and Pivot choices reproducible.

After generating the graph, the code:

1. converts the matrix to a NetworkX graph for visualization;
2. runs the Pivot algorithm on the signed matrix;
3. computes the clustering cost;
4. prints the clusters, selected pivots, and cost;
5. draws the clustered signed graph.

The clustering cost is defined as the number of negative edges inside clusters plus the number of positive edges between different clusters.