# MinMax Correlation Clustering Experiments

This repository contains the code and results for our experiments on ordinary
correlation clustering and MinMax correlation clustering.

We use Facebook ego networks and synthetic signed clique graphs.

## Setup

```bash
git clone https://github.com/aou219/MinMax_Correlation_Clustering_Experiments.git
cd MinMax_Correlation_Clustering_Experiments

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Gurobi and a valid Gurobi license are needed for the LP runs.

## Main folders

```text
data/facebook/                     Facebook ego-network files
src/                               algorithms and helper functions
scripts/run_experiment.py          runs the Facebook experiments
scripts/make_paper_tables.py       creates the final tables
scripts/make_facebook_minmax_figures.py
scripts/make_facebook_approximation_range_figures.py
scripts/benchmark_normal_cc_runtimes.py
scripts/capture_machine_specifications.py
results/research_tables/           input and output tables
results/figures/research_figures/  generated figures
results/reports.md                 explanation of all table columns
```

## Facebook preprocessing

We use these Facebook ego IDs:

```text
0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980
```

For every ego graph, we only keep vertices that occur in the `.edges` file.
Vertices that only occur in circle or feature files are not included.

The signed matrix is built as follows:

- friendship edge: `+1`
- non-friendship pair: `-1`
- deleted edge: `0`
- diagonal: `0`

The `.edges` files can be stored in either of these locations:

```text
data/facebook/<ego_id>.edges
data/facebook/facebook_3/<ego_id>.edges
```

## Experiment setup

We use these edge-deletion probabilities:

```text
0.05, 0.15, 0.25, 0.4
```

For every probability, we use deletion seeds `1-30`.

A deletion seed creates a different edge-deleted graph.

### Pivot

Pivot is run 100 times on every graph instance, using Pivot seeds `1-100`.

For every deletion seed, we save:

- the best cost from the 100 Pivot runs;
- the average cost from the 100 Pivot runs.

### MinMaxCC

MinMaxCC is deterministic for a fixed graph and parameter setting, so it is run
once per graph instance.

### Synthetic clique graphs

The clique graphs use:

```text
p_pos_inside = 0.9
p_pos_between = 0.1
```

The exact clique settings and graph seeds are stored in:

```text
results/research_tables/archive/all_runs_flat.csv
```

## Parameters

For MinMaxCC, we tested:

```text
lambda = 5, 8, 12
d_hat = powers of two from 1 up to the graph's maximum positive degree
```

The final setting is:

```text
d_hat = 8
lambda = 5
```

We chose this pair by comparing the MinMaxCC/MinMaxLP ratios across the tested
instances. We first looked at how often a parameter pair gave the best ratio,
then at its mean ratio, then at how often it gave the worst ratio. Remaining
ties were broken by choosing the smaller `d_hat` and then the smaller `lambda`.

The same final setting is used for all reported graphs.

For MinMaxLP and its rounding step, we use:

```text
r = 0.4
r2 = 0.4
norm = infinity
Gurobi Method = 2
Crossover = 0
```

The local MinMaxLP runs are done for:

```text
414, 686, 698, 3980
```

## Metrics

### Ordinary correlation clustering

The cost counts:

- positive pairs placed in different clusters;
- negative pairs placed in the same cluster.

Deleted pairs do not contribute to the cost.

The approximation ratio is:

```text
Pivot cost / ordinary LP objective
```

For edge-deleted Facebook graphs, the ratio is first calculated separately for
every deletion seed using the LP from the same graph instance. The final table
then reports the average of those per-seed ratios.

### MinMax correlation clustering

The MinMaxCC objective is the largest number of disagreements at any vertex.

The approximation ratio is:

```text
MinMaxCC objective / MinMaxLP objective
```

Costs and ratios are summarized separately:

```text
minmaxcc_cost_best     = minimum cost
minmaxcc_cost_average  = average cost
minmaxcc_cost_worst    = maximum cost

minmaxcc_ratio_best    = minimum per-seed ratio
minmaxcc_ratio_average = average per-seed ratio
minmaxcc_ratio_worst   = maximum per-seed ratio
```

The seed with the best cost does not always have the best ratio, because the LP
value can also change between deletion seeds.

More details about every output column are in `results/reports.md`.

## Run MinMaxLP with rounding

```bash
caffeinate -i python scripts/run_experiment.py   --table results/research_tables/minmax_facebook_grid_runs_flat.csv   --mode minmax   --minmax-components lp   --ego-ids 414,686,698,3980   --minmax-lp-egos 414,686,698,3980   --p-delete-values 0.05,0.15,0.25,0.4   --seeds 1-30   --min-max-lp-r 0.4   --min-max-lp-r2 0.4   --min-max-lp-method 2   --progress results/research_tables/minmax_lp_rounding_progress.json   --manifest results/research_tables/minmax_lp_rounding_manifest.json   --restart   --continue-on-error   --memory-cleanup gurobi
```

This saves:

- LP objective;
- rounding cost;
- maximum-disagreement vertex;
- cluster count;
- LP runtime;
- rounding runtime;
- total runtime;
- matrix hashes and parameters.

To continue an interrupted run, use the same command without `--restart`.

## Runtime benchmark

Run the machine-specification script first:

```bash
python scripts/capture_machine_specifications.py
```

Then run the ordinary Pivot benchmark:

```bash
caffeinate -i python scripts/benchmark_normal_cc_runtimes.py   --dataset all   --algorithms pivot   --pivot-runs 30   --restart
```

Then run the ordinary LP benchmark:

```bash
caffeinate -i python scripts/benchmark_normal_cc_runtimes.py   --dataset all   --algorithms lp   --lp-repetitions 1
```

Runtime definitions:

- Pivot: time for one Pivot call;
- ordinary LP: Gurobi solver runtime when available;
- MinMaxCC: time for the MinMaxCC call;
- MinMaxLP total runtime: LP runtime plus rounding runtime.

Do not run runtime benchmarks while another solver experiment is active.

The machine details are saved in:

```text
results/research_tables/runtime_machine_specifications.json
```

The experiments were run on an Apple M2 machine with 8 physical and 8 logical
cores, using Gurobi 13.0.2. The JSON file contains the exact RAM, operating
system, Python, NumPy, Gurobi and Git information.

## Create the final tables and figures

```bash
python scripts/make_paper_tables.py   --d-hat 8   --lambda-value 5

python scripts/make_facebook_minmax_figures.py
python scripts/make_facebook_approximation_range_figures.py
```

`make_paper_tables.py` already adds the ordinary Pivot and LP runtimes from
`normal_cc_runtime_benchmarks.csv`.

The final tables are:

```text
results/research_tables/facebook_correlation_clustering_table.csv
results/research_tables/clique_correlation_clustering_table.csv
results/research_tables/facebook_minmax_table.csv
```

The figures are saved in:

```text
results/figures/research_figures/
```

The MinMax figures only use local same-instance LP comparisons by default.

## Reproducibility settings

| Setting | Value |
|---|---|
| Facebook ego IDs | `0,107,348,414,686,698,1684,1912,3437,3980` |
| Deletion probabilities | `0.05,0.15,0.25,0.4` |
| Deletion seeds | `1-30` |
| Pivot seeds | `1-100` |
| Pivot runtime repetitions | `30` |
| Ordinary LP runtime repetitions | `1` |
| Tested `lambda` values | `5,8,12` |
| Tested `d_hat` values | powers of two up to maximum positive degree |
| Final `d_hat`, `lambda` | `8`, `5` |
| MinMaxLP `r`, `r2` | `0.4`, `0.4` |
| MinMaxLP norm | infinity |
| Gurobi Method, Crossover | `2`, `0` |
| Local Facebook LP ego IDs | `414,686,698,3980` |

The runs use fixed seeds and sorted vertex order.
