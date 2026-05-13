# Correlation-clustering-experiments

Experiments with Pivot, LP rounding, and ILP formulations for correlation clustering on complete and edge-deleted signed graphs.

## Current implementation

The current code:

- Generates a complete signed graph (adjacency matrix).
- Runs the Pivot algorithm.
- Calculates the cost.
- Finds all bad triangles.
- Counts the amount of bad triangles for an edge.
- Draws the signed graph, with the pivot order shown above the node and the clusters found by the color of the node.

## Graph representation

The signed graph is first represented as an adjacency matrix `S`, where:

- `1` represents a positive edge;
- `-1` represents a negative edge;
- `0` represents no edge or the diagonal.

## Parameters

The parameter `n=8` determines the number of vertices.
The parameter `p_positive=0.5` means that every edge has a 50% probability of being positive and a 50% probability of being negative.
The parameter `seed=42` is used to make the random graph generation and Pivot choices reproducible.
The parameter `n1 = 3` and `n2 = 1` are the nodes from the edge for which we want to count the number of containing bad triangles.

## Cost definition

The clustering cost is defined as the number of negative edges inside clusters plus the number of positive edges between different clusters.

## Bad triangles

A bad triangle is a triple of edges with exactly two positive edges and one negative edge. Such a triple of edges always has a cost of at least 1, so it is useful for lower-bound analysis.

## Current workflow

After generating the graph, the code:

1. converts the matrix to a NetworkX graph for visualization;
2. runs the Pivot algorithm on the signed matrix;
3. computes the clustering cost;
4. prints the clusters, selected pivots, and cost and bad-triangle information;
5. draws the clustered signed graph.

## Running the code

It is recommended to work in a virtual environment.

### macOS / Linux

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

```

Install the required packages:

```bash

pip install -r requirements.txt

```

Run the current experiment:

```bash

python src/experiments.py

```

### Windows

Create and activate a virtual environment:

```bash

python -m venv .venv
.venv\Scripts\activate

```

Install the required packages:

```bash

pip install -r requirements.txt

```

Run the current experiment:

```bash

python src\experiments.py

```

The main experiment settings can be changed in src/experiments.py.