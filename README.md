# Correlation Clustering under Random Edge Deletion

This repository contains the code and experiment data for a BSc thesis on the
robustness of correlation clustering methods when signed edges are deleted.
It compares the randomized Pivot algorithm, integer and linear programming
formulations, and lower bounds based on bad triangles. It also studies whether
bad 4-cycle constraints recover information that is lost after edge deletion.

The experiments cover three graph families:

- independently signed random complete graphs;
- synthetic graphs with planted clique structure;
- Facebook ego-networks from the SNAP Facebook social-circles dataset.

## Problem

For a signed graph `G = (V, E+ union E-)`, correlation clustering seeks a
partition of the vertices that minimizes disagreements:

- a positive edge whose endpoints are assigned to different clusters;
- a negative edge whose endpoints are assigned to the same cluster.

Let `x_ij = 0` when vertices `i` and `j` are in the same cluster and
`x_ij = 1` otherwise. The objective is:

```text
minimize
    sum(x_ij                 for each positive edge (i,j))
  + sum(1 - x_ij             for each negative edge (i,j))
```

Random edge deletion changes an observed edge to a missing edge. The tested
deletion probabilities are `0.05`, `0.15`, `0.25`, and `0.40`.

## Methods

### Pivot

Pivot selects an unclustered vertex and clusters it with its observed positive
neighbours. Because its output depends on the pivot order, each instance is run
with 100 pivot seeds. Both the best and average disagreement costs are stored.

### Full ILP and LP

The all-pairs formulation contains a variable for every vertex pair. Triangle
inequalities enforce a valid clustering. Binary variables give the Full ILP
optimum; variables in the interval `[0, 1]` give the LP lower bound.

### Observed-edge formulations

The sparse formulations use only observed edges. They are evaluated both
without and with constraints for induced bad 4-cycles: cycles with three
positive edges, one negative edge, and two missing diagonals.

### Bad-triangle lower bound

A bad triangle contains two positive edges and one negative edge. Every valid
clustering must disagree with at least one of its edges. A greedy set of
edge-disjoint bad triangles therefore gives a lower bound on the optimum.

## Experimental Data

### Random signed graphs

- graph sizes `n = 5, 10, 15, 20, 25, 30`
- positive-edge probabilities `p_pos = 0.2, 0.3, ..., 0.8`
- 50 graph seeds per `(n, p_pos)` setting
- 2,100 complete instances and 8,400 edge-deleted runs

### Planted-clique graphs

- graph sizes `n = 10, 15, 20, 25, 30, 100`
- positive-edge probability 0.9 inside planted cliques
- positive-edge probability 0.1 between planted cliques
- balanced and unbalanced clique-size decompositions
- 50 seeds for `n <= 30` and 20 seeds for `n = 100`
- 910 complete instances and 3,640 edge-deleted runs

### Facebook ego-networks

- ego IDs 414, 686, 698, and 3980
- an observed friendship is positive;
- a missing friendship between included vertices is negative;
- the ego vertex itself is excluded;
- 4 complete instances and 16 edge-deleted runs.

Full optimization results are available for three ego-networks, giving 12
Facebook runs for analyses that require an ILP optimum.

## Repository Layout

```text
.
├── data/facebook/                         SNAP ego-network input files
├── results/
│   ├── experiments_results_random/        raw random-graph JSON results
│   ├── experiments_results_clique/        raw planted-clique JSON results
│   ├── experiments_results_facebook/      raw Facebook JSON results
│   └── processed/
│       ├── all_runs_flat.csv               combined analysis table
│       ├── figures/                        generated thesis figures
│       └── tables/                         generated summary tables
├── scripts/
│   ├── make_all_runs_flat.py               JSON-to-CSV processing
│   ├── make_general_pdelete_figures.py     aggregate family figures
│   └── make_specific_pdelete_figures.py    parameter-specific figures
└── src/
    ├── all_pairs_solver.py                 Full ILP and LP formulations
    ├── bad_triangles.py                    bad-triangle routines
    ├── edge_deletion.py                    random edge deletion
    ├── experiment_helpers.py               shared experiment pipeline
    ├── experiment_random.py                one random instance
    ├── experiment_clique.py                one planted-clique instance
    ├── experiment_facebook.py              one Facebook instance
    ├── graph_generation.py                 synthetic graph generators
    ├── ilp_solver.py                       observed-edge formulations
    ├── lp_formulations.py                  LP routines
    └── pivot.py                            Pivot implementation
```

The repository contains additional analysis scripts used during development.
The three scripts listed above form the shortest route from the stored JSON
experiments to the main processed dataset and figures.

## Installation

Python 3.11 or a compatible recent Python version is recommended. Create a
virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The optimization code uses `gurobipy`. Running the ILP and LP experiments
requires a working Gurobi installation and license. Academic licenses are
available separately from Gurobi.

## Running One Instance

The files in `src/experiment_random.py`, `src/experiment_clique.py`, and
`src/experiment_facebook.py` contain a parameter block near the top of their
main section. Set the graph parameters there and run the file from the
repository root:

```bash
python src/experiment_random.py
python src/experiment_clique.py
python src/experiment_facebook.py
```

These entry points are intended for inspecting or testing one configuration.
Large instances, especially the all-pairs ILP and bad 4-cycle computations,
can require substantial time and memory. Set `draw_graph = False` when figures
for an individual run are not needed.

## Rebuilding the Processed Dataset

The raw experiment records are stored as JSON below the three
`results/experiments_results_*` directories. Rebuild the combined CSV with:

```bash
python scripts/make_all_runs_flat.py
```

This writes:

```text
results/processed/all_runs_flat.csv
```

If that file already exists, the script first creates
`results/processed/all_runs_flat.csv.bak`.

For the dataset used in the thesis, the command should report:

```text
random:   8400 rows
clique:   3640 rows
facebook:   16 rows
total:   12056 rows
```

Only 12 Facebook rows contain comparable Full ILP values.

## Rebuilding the Main Figures

After rebuilding `all_runs_flat.csv`, generate the aggregate and detailed
edge-deletion figures with:

```bash
python scripts/make_general_pdelete_figures.py
python scripts/make_specific_pdelete_figures.py
```

The outputs are written to:

```text
results/processed/figures/p_delete_effect/general/
results/processed/figures/p_delete_effect/specific/
```

Both scripts accept a different input CSV when needed:

```bash
python scripts/make_general_pdelete_figures.py --csv path/to/results.csv
```

The general plots first average within each graph size and then across graph
sizes. This prevents graph sizes with more tested configurations from receiving
more weight. The specific plots retain distinctions such as
`p_pos`, clique balance, graph size, and Facebook ego ID.

## Interpreting the Main Ratios

### Best-Pivot approximation ratio

```text
best Pivot cost over 100 runs / Full ILP cost
```

A value of 1 means that the best Pivot run found an optimal cost. A value of
1.2 means that its cost was 20% above the optimum. Lower is better.

### LP integrality ratio

```text
LP lower bound / Full ILP cost
```

A value close to 1 indicates a tight LP relaxation. Lower values indicate a
larger gap between the LP lower bound and the integer optimum.

### Greedy bad-triangle bound ratio

```text
number of greedily selected edge-disjoint bad triangles / Full ILP cost
```

A value close to 1 means that this local lower bound accounts for most of the
optimal cost. Because the implementation is greedy, it need not find the
largest possible edge-disjoint set.

### Observed-edge ILP ratio

```text
observed-edge ILP objective / Full ILP objective
```

A value of 1 denotes equal objective values. The experiments compare this
ratio before and after adding bad 4-cycle constraints.

## Reproducibility Notes

- Graph generation and edge deletion use stored integer seeds.
- Each deletion probability is applied separately to its corresponding
  complete graph instance.
- Raw JSON files retain the graph parameters and results used to construct the
  processed CSV.
- Gurobi version, hardware, and available threads can affect runtime, but an
  optimal solve should not change the reported objective value.
- Facebook ego 686 is retained in the dataset but excluded from comparisons
  that require an available Full ILP optimum.

## Citation

This repository accompanies the BSc thesis *Robustness of Pivot, LP, and ILP
Correlation Clustering under Random Edge Deletion* by Amira Ouchene. A formal
citation can be added here after the final thesis version is archived.
