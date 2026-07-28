# MinMax Correlation Clustering Experiments

This repository contains the code used to run the clique and Facebook experiments for correlation clustering.

The experiment runners use the shared function `run_full_experiment` from `src/experiment_helpers.py`.

## Algorithms

The clique experiments run:

- Pivot
- the LP relaxation for normal correlation clustering

The Facebook experiments run:

- Pivot
- the LP relaxation for normal correlation clustering
- MinMaxCC
- MinMaxLP and its rounding algorithm

For all experiments, `p_delete = 0` represents the complete graph. A positive value of `p_delete` represents the fraction of edges deleted before running the algorithms.

## Requirements

- Python 3
- the packages listed in `requirements.txt`
- Gurobi with a valid license

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Data

The Facebook data is stored in:

```text
data/facebook/
```

The Facebook runner searches for each ego-network edge file in:

```text
data/facebook/<ego_id>.edges
data/facebook/facebook_3/<ego_id>.edges
```

## Running the clique experiments

Run:

```bash
python scripts/run_experiment_clique.py
```

The results are written to:

```text
results/output_tables/clique_output.csv
```

Experiment settings can be changed at the top of `scripts/run_experiment_clique.py`:

```python
CLUSTER_SIZE_CASES = [[5, 6]]
P_POS_INSIDE_VALUES = [0.9]
P_POS_BETWEEN_VALUES = [0.1]
GRAPH_SEEDS = [1]
P_DELETE_VALUES = [0.0, 0.05]
PIVOT_SEEDS = range(1, 11)
```

Each combination of clique sizes, probabilities, graph seed, and deletion probability produces one row in the output table.

## Running the Facebook experiments

Run:

```bash
python scripts/run_experiment_facebook.py
```

The results are written to:

```text
results/output_tables/facebook_output.csv
```

Experiment settings can be changed at the top of `scripts/run_experiment_facebook.py`:

```python
EGO_IDS = ["3980"]
P_DELETE_VALUES = [0.0, 0.05]
DELETION_SEEDS = [1]
PIVOT_SEEDS = range(1, 101)

D_HAT = 8
LAMBDA = 5
MINMAX_LP_R = 0.4
MINMAX_LP_R2 = 0.4
MINMAX_LP_METHOD = 2
```

For `p_delete = 0`, the runner writes one complete-graph row. For positive deletion probabilities, it runs every seed in `DELETION_SEEDS`.

To run the four Facebook ego networks used in the experiments:

```python
EGO_IDS = ["3980", "698", "414", "686"]
```

A larger experiment grid can be configured with:

```python
P_DELETE_VALUES = [0.0, 0.05, 0.15, 0.25, 0.40]
DELETION_SEEDS = range(1, 31)
```

## Output tables

The runners write one row per graph instance and deletion setting.

The clique output includes:

- graph parameters
- number of deleted edges
- Pivot best and average cost
- normal correlation-clustering LP cost
- normal LP runtime

The Facebook output additionally includes:

- MinMaxCC cluster count, maximum disagreement, parameters, and runtime
- MinMaxLP cost, rounded solution statistics, parameters, and runtimes
- the MinMaxCC-to-MinMaxLP ratio

The output CSV is overwritten each time its runner is executed.

## Project structure

```text
data/facebook/                 Facebook ego-network data
scripts/run_experiment_clique.py
scripts/run_experiment_facebook.py
src/experiment_helpers.py     Shared experiment pipeline
src/normal_lp.py              Normal correlation-clustering LP
src/pivot.py                   Pivot algorithm
src/min_max.py                 MinMaxCC
src/min_max_lp.py              MinMaxLP and rounding
results/output_tables/         Newly generated experiment outputs
results/research_tables/       Tables used for the paper
```

The experiment runners do not read from `results/research_tables/`.

## Quick verification

Check that the main files compile:

```bash
python -m py_compile \
  src/normal_lp.py \
  src/experiment_helpers.py \
  scripts/run_experiment_clique.py \
  scripts/run_experiment_facebook.py
```

Then run both quick configurations:

```bash
python scripts/run_experiment_clique.py
python scripts/run_experiment_facebook.py
```
