# Correlation-clustering-experiments

Experiments with Pivot, ILP, LP relaxations, bad triangles, edge-disjoint bad triangles, and bad 4-cycle constraints for correlation clustering on complete and edge-deleted signed graphs.

## Current implementation

The current code:

- Generates a complete signed graph as an adjacency matrix.
- Runs the Pivot algorithm on the complete graph.
- Computes the Pivot clustering cost.
- Finds all bad triangles.
- Computes minimum and maximum edge-disjoint bad triangle counts.
- Solves the ILP formulation for the complete graph.
- Solves the LP relaxation for the complete graph.
- Solves primal and dual LP formulations based on bad triangles.
- Deletes edges from the complete graph to create an incomplete graph.
- Runs the same algorithms on the incomplete graph.
- Finds the remaining bad triangles after edge deletion.
- Computes minimum and maximum edge-disjoint bad triangle counts after edge deletion.
- Detects bad 4-cycles in the incomplete graph.
- Solves the ILP and LP relaxation on the incomplete graph both with and without bad 4-cycle constraints.
- Checks how many bad 4-cycle constraints are violated by the solution without these constraints.
- Saves the results to a JSON file.
- Draws the complete and edge-deleted graphs with the Pivot and ILP clusterings.

## Graph representation

The signed graph is represented as an adjacency matrix `S`, where:

- `1` represents a positive edge;
- `-1` represents a negative edge;
- `0` represents no edge or the diagonal.

In the complete graph, every pair of different vertices has either a positive or a negative edge. In the incomplete graph, some edges are deleted and are set to `0`.

Deleted edges are treated as missing or unobserved edges. They do not contribute to the objective function and they do not receive Gurobi decision variables in the ILP formulation. This means that the optimization model only uses observed edges directly.

## Parameters

The main experiment settings can be changed in `src/experiments.py`.

The most important parameters are:

- `n`: number of vertices;
- `p_positive`: probability that an edge is positive;
- `p_delete`: probability that an edge is deleted when constructing the incomplete graph;
- `seed`: random seed used for reproducibility.

For example:

```python
n = 24
p_positive = 0.5
p_delete = 0.25
seed = None
````

When `seed=None`, a new random graph is generated each time the experiment is run.

## Cost definition

The clustering cost is defined as the number of disagreements:

* a positive edge contributes cost `1` if its endpoints are placed in different clusters;
* a negative edge contributes cost `1` if its endpoints are placed in the same cluster.

Deleted edges do not contribute to the cost.

## Bad triangles

A bad triangle is a triangle with exactly two positive edges and one negative edge. Such a triangle always forces at least one disagreement in any clustering.

Bad triangles are useful because they give local evidence that some clustering mistake is unavoidable. However, simply counting all bad triangles does not directly give a lower bound on the optimal clustering cost, because different bad triangles can share edges. One mistaken edge can then account for several bad triangles at the same time.

For this reason, the code also considers edge-disjoint bad triangles.

## Edge-disjoint bad triangles

Two bad triangles are edge-disjoint if they do not share any edge. A set of edge-disjoint bad triangles can be used as a lower bound on the optimal clustering cost, because each bad triangle forces at least one disagreement and the triangles cannot all be charged to the same mistaken edge.

The code computes two different edge-disjoint bad triangle counts:

* `min_bad_triangles_count`;
* `max_bad_triangles_count`.

These values are computed using different deterministic selection procedures. They are used to compare how different choices of edge-disjoint bad triangles affect the resulting lower-bound estimate.

The maximum edge-disjoint bad triangle count is usually more informative as a lower bound, because it tries to find a larger set of disjoint bad triangles. The minimum count is included for comparison and to study how sensitive the lower-bound estimate is to the selection procedure.

The same computations are performed on both the complete graph and the edge-deleted graph.

## ILP formulation

The main ILP formulation used in this project is a sparse ILP formulation. In this formulation, Gurobi variables are only created for observed edges. Deleted edges do not receive variables and are not used directly in the model.

This is important for incomplete graphs, because deleted edges should not contribute to the cost and should not directly constrain the optimization problem. However, this also means that some triangle constraints disappear when one or more edges of a triangle are missing. Therefore, the implementation also checks bad 4-cycles as an additional structure that can remain after edge deletion.

## LP relaxation

The ILP can also be solved as an LP relaxation by allowing variables to take fractional values between `0` and `1`. The LP relaxation is used to obtain a lower bound on the optimal ILP cost.

The experiments compare the LP relaxation on both the complete graph and the incomplete graph. For the incomplete graph, the LP relaxation is also solved both with and without bad 4-cycle constraints.

## Bad-triangle LP bounds

Besides the ILP relaxation, the project also includes primal and dual LP formulations based on bad triangles.

These LPs use the set of bad triangles to compute lower bounds on the clustering cost. The primal and dual values are printed for both the complete graph and the incomplete graph. In the experiments, these values can be compared with the ILP optimum, the LP relaxation, and the edge-disjoint bad triangle counts.

## Bad 4-cycles

In incomplete graphs, deleting edges can remove bad triangles, even though a larger inconsistent structure remains. For this reason, the code also detects bad 4-cycles.

A bad 4-cycle is considered when:

* all four cycle edges are present;
* exactly one of the four cycle edges is negative;
* both diagonals are missing.

The experiments compare the ILP and LP relaxation on the incomplete graph:

1. without bad 4-cycle constraints;
2. with bad 4-cycle constraints.

The code also checks how many bad 4-cycle constraints are violated by the solution obtained without adding these constraints.

This makes it possible to distinguish between:

* the number of bad 4-cycles detected in the graph;
* the number of bad 4-cycle constraints actually violated by the solution;
* whether adding the constraints changes the objective cost or only changes the feasible solution structure.

## Current workflow

After generating the complete graph, the code:

1. converts the matrix to a NetworkX graph for visualization;
2. runs the Pivot algorithm;
3. computes the Pivot cost;
4. finds all bad triangles;
5. computes minimum and maximum edge-disjoint bad triangle counts;
6. solves the ILP;
7. solves the LP relaxation;
8. solves primal and dual LP formulations based on bad triangles.

After that, the code creates an incomplete graph by deleting edges. On this new graph, the code:

1. runs the Pivot algorithm again;
2. computes the Pivot cost;
3. finds all remaining bad triangles;
4. computes minimum and maximum edge-disjoint bad triangle counts;
5. solves the ILP without bad 4-cycle constraints;
6. solves the ILP with bad 4-cycle constraints;
7. solves the LP relaxation without bad 4-cycle constraints;
8. solves the LP relaxation with bad 4-cycle constraints;
9. detects bad 4-cycles;
10. checks how many bad 4-cycle constraints are violated by the no-4-cycle solution;
11. solves primal and dual LP formulations based on the remaining bad triangles;
12. saves the results to a JSON file;
13. draws the complete and incomplete graph clusterings.

## Output

The experiment prints results for both the complete graph and the incomplete graph.

For the complete graph, the output includes:

* Pivot cost;
* minimum and maximum edge-disjoint bad triangle counts;
* bad-triangle LP primal and dual values;
* ILP optimal cost;
* LP relaxation cost.

For the incomplete graph, the output includes:

* number of deleted edges;
* Pivot cost;
* minimum and maximum edge-disjoint bad triangle counts;
* bad-triangle LP primal and dual values;
* ILP cost without bad 4-cycle constraints;
* ILP cost with bad 4-cycle constraints;
* number of detected bad 4-cycles;
* number of violated bad 4-cycle constraints in the ILP solution without bad 4-cycle constraints;
* LP relaxation cost without bad 4-cycle constraints;
* LP relaxation cost with bad 4-cycle constraints;
* number of violated bad 4-cycle constraints in the LP solution without bad 4-cycle constraints.

## Results file

Experiment results are stored in:

```bash
results/experiments_results.json
```

The results are appended to this file after each run. The JSON structure separates:

* graph parameters;
* complete graph results;
* incomplete graph results;
* Pivot results;
* bad triangle counts;
* edge-disjoint bad triangle counts;
* ILP results;
* LP relaxation results;
* bad-triangle LP bounds;
* bad 4-cycle counts;
* violated bad 4-cycle counts.

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

## Notes

The ILP formulation used in this project is sparse: deleted edges are not assigned Gurobi variables. This means that deleted edges do not contribute to the cost and do not directly constrain the clustering.

Bad 4-cycle constraints are added to capture some inconsistencies that may remain after bad triangles disappear due to edge deletion. In random instances, adding these constraints does not always change the objective cost. However, the constraints can still remove violated solutions or change the feasible solution structure.
