# Correlation Clustering Experiments

This repository contains experiments for a thesis project on **correlation clustering** on signed graphs.

The project compares Pivot, ILP, LP relaxations, bad triangles, edge-disjoint bad triangles, edge deletion, and bad 4-cycle constraints on different graph families.

The experiments are run on:

- random signed graphs
- synthetic clique/community graphs
- real-world Facebook ego-networks from the SNAP Facebook social circles dataset

---

## Research Questions

### RQ1 — Edge deletion

How does the clustering cost and clustering structure change when edges are removed?

### RQ2 — Input structure

For which types of input graphs do Pivot, LP, and ILP perform better or worse?

The project compares:

- random graphs
- clique/community graphs
- Facebook ego-networks

### RQ3 — LP vs ILP

How close is the LP relaxation to the ILP optimum, and when is ILP still practically solvable?

---

## Problem Setting

The project studies correlation clustering on signed graphs.

For a pair of vertices \(i,j\), the clustering variable is:

- \(x_{ij} = 0\): vertices are in the same cluster
- \(x_{ij} = 1\): vertices are in different clusters

The objective is to minimize disagreements:

- a positive edge contributes cost 1 if its endpoints are placed in different clusters
- a negative edge contributes cost 1 if its endpoints are placed in the same cluster

The cost is:

\[
\sum_{(i,j)\in E^+} x_{ij}
+
\sum_{(i,j)\in E^-} (1 - x_{ij})
\]

Lower cost is better.

---

## Methods

### Pivot heuristic

The randomized Pivot algorithm is used as an approximation algorithm. Multiple Pivot seeds are tested, and both the best and average Pivot costs are stored.

### ILP

The integer linear programming formulation is used to compute the optimal clustering cost when this is computationally feasible.

The ILP uses triangle inequalities to enforce valid clustering structure.

### LP relaxation

The LP relaxation allows variables \(x_{ij}\) to take fractional values between 0 and 1. This gives a lower bound on the ILP optimum.

The ratio `LP/ILP` measures how tight the LP relaxation is.

### Bad triangles

A bad triangle is a triangle with exactly two positive edges and one negative edge.

At least one edge in every bad triangle must be violated in any clustering. Therefore, bad triangles help explain why some instances have higher optimum cost.

The project computes:

- total number of bad triangles
- greedy minimum edge-disjoint bad triangle count
- greedy maximum edge-disjoint bad triangle count
- ratios comparing edge-disjoint bad triangles to the ILP cost

### Bad 4-cycle constraints

For sparse edge-deleted graphs, bad 4-cycle constraints are added and compared.

A bad 4-cycle is used when:

- all four cycle edges are present
- exactly one cycle edge is negative
- both diagonals are missing

The experiments compare sparse ILP and LP solutions:

- without bad 4-cycle constraints
- with bad 4-cycle constraints

This makes it possible to test whether the constraints change:

- the objective cost
- the clustering structure
- or both

---

## Data

### Random graphs

Random signed complete graphs are generated for:

- \(n = 5, 10, 15, 20, 25, 30\)
- \(p^+ = 0.2, 0.3, ..., 0.8\)

Here, \(p^+\) is the probability that an edge is positive.

### Clique/community graphs

Synthetic clique/community graphs are generated with planted cluster structures.

Examples:

- balanced structures: `2x15`, `3x10`, `4x25`
- unbalanced structures: `15_10_5`, `20_5_5`, `60_25_10_5`

### Facebook ego-networks

The Facebook experiments use full ego-networks from the SNAP Facebook social circles dataset.

Each ego-network contains:

- the ego user's friends
- friendship edges between those friends

The ego node itself is excluded, because otherwise it becomes a dominant supernode connected to almost everyone.

For the signed graph construction:

- existing friendships are treated as positive edges
- missing friendships inside the ego-network are treated as negative edges

The Facebook experiments are real-world structured cases, but they are not meant to represent the entire Facebook graph.

---

## Repository Structure

```text
.
├── data
│   └── facebook
├── results
│   ├── experiments_results_clique
│   ├── experiments_results_facebook
│   ├── experiments_results_random
│   ├── figures
│   ├── processed
│   └── raw
├── scripts
│   ├── make_deep_thesis_patterns.py
│   └── make_thesis_results_file.py
└── src
    ├── bad_triangles.py
    ├── check_facebook_circles.py
    ├── cost.py
    ├── draw_graphs.py
    ├── draw_graphs_clique.py
    ├── edge_deletion.py
    ├── experiment_clique.py
    ├── experiment_facebook.py
    ├── experiment_helpers.py
    ├── experiment_random.py
    ├── facebook_sampling.py
    ├── graph_generation.py
    ├── ilp_solver.py
    ├── lp_formulations.py
    └── pivot.py
```

---

## Raw Results

Raw JSON results are stored in:

```text
results/experiments_results_random/
results/experiments_results_clique/
results/experiments_results_facebook/
```

The raw results are grouped by graph family and graph size.

### Random results

```text
results/experiments_results_random/n5/
results/experiments_results_random/n10/
results/experiments_results_random/n15/
results/experiments_results_random/n20/
results/experiments_results_random/n25/
results/experiments_results_random/n30/
```

### Clique/community results

```text
results/experiments_results_clique/n10/
results/experiments_results_clique/n15/
results/experiments_results_clique/n20/
results/experiments_results_clique/n25/
results/experiments_results_clique/n30/
results/experiments_results_clique/n100/
```

### Facebook results

```text
results/experiments_results_facebook/full/
```

---

## Processed Thesis Results

Processed thesis-ready results are stored in:

```text
results/processed/
```

Start with these two files:

```text
results/processed/reports/01_key_results.md
results/processed/reports/02_deep_patterns.md
```

### Main reports

```text
results/processed/reports/01_key_results.md
```

Main result overview per research question.

```text
results/processed/reports/02_deep_patterns.md
```

Deeper analysis of bad triangles, edge-disjoint bad triangles, p_positive, edge deletion, and bad 4-cycles.

---

## Processed Tables

### Random graphs

```text
results/processed/tables/random/random_by_n_p.csv
results/processed/tables/random/random_trend_by_n.csv
results/processed/tables/random/random_trend_by_p_positive.csv
results/processed/tables/random/random_p_bad_triangle_trend.csv
results/processed/tables/random/random_np_bad_triangle_detail.csv
```

These tables study how graph size \(n\) and \(p^+\) affect Pivot, LP, ILP, bad triangles, and cost.

### Clique/community graphs

```text
results/processed/tables/clique/clique_by_structure.csv
```

This table compares balanced and unbalanced planted community structures.

### Facebook ego-networks

```text
results/processed/tables/facebook/facebook_full_ego_summary.csv
```

This table summarizes the Facebook full ego-network results.

### Bad triangles

```text
results/processed/tables/bad_triangles/bad_triangle_bounds_by_graph_family.csv
```

This table compares edge-disjoint bad triangle lower bounds with ILP costs.

### Four-cycle constraints

```text
results/processed/tables/four_cycles/four_cycle_effect_by_graph_family.csv
results/processed/tables/four_cycles/four_cycle_effect_by_file.csv
results/processed/tables/four_cycles/four_cycle_effect_detail.csv
```

These tables show how often 4-cycle constraints change the cost or produce a different clustering.

---

## How to Read the Ratios

### Pivot/ILP

```text
Pivot/ILP = 1.000
```

means Pivot found an optimal solution.

```text
Pivot/ILP = 1.400
```

means Pivot cost is 40% higher than the ILP optimum.

Higher is worse.

### LP/ILP

```text
LP/ILP = 1.000
```

means the LP relaxation is tight and matches the ILP optimum.

```text
LP/ILP = 0.700
```

means the LP lower bound is only 70% of the ILP optimum.

Closer to 1 is better.

### Max edge-disjoint bad triangles / ILP

```text
max edge-disjoint bad triangles / ILP = 1.000
```

means the greedy edge-disjoint bad-triangle lower bound fully explains the ILP cost.

A lower value means that the ILP cost is not fully explained by local disjoint bad triangles and may depend more on global graph structure.

---

## Main Findings

### Edge deletion

Edge deletion usually lowers the absolute ILP cost, because fewer edges remain in the objective.

However, Pivot often becomes relatively worse after edge deletion, especially on clique/community graphs. This suggests that deleted edges remove useful structural information.

### Random graphs

For random graphs, the difficulty depends strongly on \(p^+\).

The number of bad triangles changes with \(p^+\). Since a bad triangle has exactly two positive edges and one negative edge, inconsistent local structure is most common when positive edges are frequent but negative edges still occur often enough.

Random graphs with many bad triangles tend to have higher ILP costs and looser LP relaxations.

### Clique/community graphs

LP performs very well on clique/community graphs and often exactly matches the ILP optimum.

Pivot performs worse on larger and more unbalanced community structures, especially after edge deletion.

### Facebook ego-networks

Facebook ego-networks behave between random graphs and synthetic clique/community graphs.

They have real social structure, but not the clean planted structure of the synthetic clique instances.

For small and medium ego-networks, ILP can still be solved. Larger ego-networks become computationally expensive, especially when 4-cycle constraints are included.

### Bad triangles

Bad triangles help explain why some graphs have higher clustering cost.

The maximum edge-disjoint bad-triangle count gives a lower bound on the ILP cost. When this bound is close to the ILP cost, local bad-triangle structure explains much of the optimum.

When it is far from the ILP cost, the difficulty is caused by more global structure.

### 4-cycle constraints

For edge-deleted sparse graphs, the experiments compare the ILP cost **without** bad 4-cycle constraints to the ILP cost **with** bad 4-cycle constraints.

This comparison is important because bad 4-cycle constraints do not always change the objective value. In many cases, the optimal cost stays the same, but the selected clustering can still change.

In the processed results:

- ILP costs with and without 4-cycle constraints can be compared in `2624` runs.
- The objective cost changed in `89` runs, which is `3.4%` of comparable runs.
- The objective cost stayed the same in `2535` runs.
- The clustering comparison `same_clustering_4_cycle` is available in `2624` runs.
- The clustering stayed the same in `2022` runs.
- The clustering changed in `602` runs, which is `22.9%` of runs with known clustering comparison.
- A same-cost but different-clustering outcome occurred in `518` runs.

This means that 4-cycle constraints usually do not change the ILP objective cost, but they can still affect the structure of the optimal clustering. Therefore, the analysis considers both the numerical cost difference and the clustering difference.

---

## Reproducing Processed Results

The processed thesis summaries are generated from:

```text
results/processed/all_runs_flat.csv
```

To regenerate the processed reports and tables, run:

```bash
python3 scripts/make_thesis_results_file.py
python3 scripts/make_deep_thesis_patterns.py
```

The main output files are:

```text
results/processed/reports/01_key_results.md
results/processed/reports/02_deep_patterns.md
```

---

## Running Experiments

Example commands:

```bash
python3 src/experiment_random.py
python3 src/experiment_clique.py
python3 src/experiment_facebook.py
```

Some experiments, especially ILP or 4-cycle experiments on larger Facebook ego-networks, can take a long time.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The ILP and LP experiments use Gurobi, so a working Gurobi installation and license may be required.

---

## Notes

Generated Python cache files such as `__pycache__` are not part of the experiment results and can be ignored.

The most important files for thesis writing are:

```text
results/processed/reports/01_key_results.md
results/processed/reports/02_deep_patterns.md
```
MD