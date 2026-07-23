# MinMax Correlation Clustering Experiments

Code for the experiments on ordinary correlation clustering and MinMax
correlation clustering on Facebook ego networks and synthetic clique graphs.

The repository contains the experiment implementations, the processed result
tables used by the paper, and the scripts that turn those results into paper
tables and figures.

## Experiment flow

```text
Facebook .edges files / synthetic clique parameters
                    |
                    v
       experiment and Pivot scripts
                    |
                    v
results/research_tables/minmax_facebook_grid_runs_flat.csv
                    |
                    v
        scripts/make_paper_tables.py
                    |
                    v
  paper tables in results/research_tables/
                    |
                    v
             figure scripts
                    |
                    v
 figures in results/figures/research_figures/
```

The Facebook flat table is the main source of truth. Solver and Pivot scripts
update this table, and the table-generation scripts derive the paper outputs
from it.

## Setup

Run commands from the repository root.

```bash
git clone https://github.com/aou219/MinMax_Correlation_Clustering_Experiments.git
cd MinMax_Correlation_Clustering_Experiments

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy scipy networkx pandas matplotlib gurobipy
```

The LP experiments require a working Gurobi installation and license.

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

On macOS, long commands can be prefixed with `caffeinate -i` to prevent the
machine from sleeping.

## Data

Facebook ego-network edge files are read from either of these locations:

```text
data/facebook/<ego_id>.edges
data/facebook/facebook_3/<ego_id>.edges
```

The experiments use the following ego IDs:

```text
0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980
```

For each ego graph, the vertex set is the sorted set of endpoints appearing in
the `.edges` file. Friendship edges are encoded as `+1`, non-edges as `-1`,
and deleted edges as `0`.

## Reproducing the Facebook experiments

The scripts below update:

```text
results/research_tables/minmax_facebook_grid_runs_flat.csv
```

The reported experiments use:

```text
p_delete:       0.05, 0.15, 0.25, 0.4
deletion seeds: 1-30
Pivot seeds:    1-100
```

### 1. Pivot on the LP-sized ego graphs

This runs complete and edge-deleted Pivot experiments for the four ego graphs
used in the ordinary LP comparison.

```bash
caffeinate -i python scripts/update_pivot_results_all_egos.py \
  --table results/research_tables/minmax_facebook_grid_runs_flat.csv \
  --ego-ids 414,686,698,3980 \
  --pivot-seeds 1-100 \
  --restart
```

### 2. Pivot on the remaining complete ego graphs

The larger ego graphs are evaluated with Pivot on the complete graph only.

```bash
caffeinate -i python scripts/update_complete_pivot_big_egos.py \
  --table results/research_tables/minmax_facebook_grid_runs_flat.csv \
  --pivot-seeds 1-100 \
  --restart
```

### 3. Ordinary correlation-clustering LP

```bash
caffeinate -i python scripts/run_experiment.py \
  --table results/research_tables/minmax_facebook_grid_runs_flat.csv \
  --mode normal \
  --ego-ids 414,686,698,3980 \
  --normal-lp-egos 414,686,698,3980 \
  --p-delete-values 0.05,0.15,0.25,0.4 \
  --seeds 1-30 \
  --progress results/research_tables/normal_lp_progress.json \
  --manifest results/research_tables/normal_lp_manifest.json \
  --restart \
  --continue-on-error \
  --memory-cleanup gurobi
```

### 4. MinMaxCC

```bash
caffeinate -i python scripts/run_experiment.py \
  --table results/research_tables/minmax_facebook_grid_runs_flat.csv \
  --mode minmax \
  --minmax-components cc \
  --ego-ids all \
  --minmax-cc-egos all \
  --p-delete-values 0.05,0.15,0.25,0.4 \
  --seeds 1-30 \
  --d-hat 8 \
  --lambda-value 5 \
  --progress results/research_tables/minmax_cc_progress.json \
  --manifest results/research_tables/minmax_cc_manifest.json \
  --restart \
  --continue-on-error
```

### 5. MinMaxLP and rounding

```bash
caffeinate -i python scripts/run_experiment.py \
  --table results/research_tables/minmax_facebook_grid_runs_flat.csv \
  --mode minmax \
  --minmax-components lp \
  --ego-ids 414,686,698,3980 \
  --minmax-lp-egos 414,686,698,3980 \
  --p-delete-values 0.05,0.15,0.25,0.4 \
  --seeds 1-30 \
  --min-max-lp-r 0.4 \
  --min-max-lp-r2 0.4 \
  --min-max-lp-method 2 \
  --progress results/research_tables/minmax_lp_progress.json \
  --manifest results/research_tables/minmax_lp_manifest.json \
  --restart \
  --continue-on-error \
  --memory-cleanup gurobi
```

## Using `run_experiment.py`

The experiment runner updates ordinary LP, MinMaxCC, and MinMaxLP results.
Pivot is handled by the two Pivot updater scripts above.

Show all options with:

```bash
python scripts/run_experiment.py --help
```

The main modes are:

| Command | Runs |
|---|---|
| `--mode normal` | ordinary all-pairs LP |
| `--mode minmax --minmax-components cc` | MinMaxCC |
| `--mode minmax --minmax-components lp` | MinMaxLP and rounding |
| `--mode minmax --minmax-components cc,lp` | both MinMax methods |
| `--mode all` | ordinary LP and selected MinMax methods |

Useful options:

| Option | Purpose |
|---|---|
| `--dry-run` | print the planned work without solving |
| `--restart` | recompute the selected configuration from the beginning |
| `--progress PATH` | checkpoint completed tasks |
| `--manifest PATH` | record parameters, hashes, versions, and failures |
| `--continue-on-error` | continue after a failed task |
| `--limit N` | run only the first `N` new edge-deleted instances |
| `--memory-cleanup gurobi` | dispose the Gurobi environment between solves |

A small dry run is useful before starting a full job:

```bash
python scripts/run_experiment.py \
  --mode minmax \
  --minmax-components cc \
  --ego-ids 3980 \
  --minmax-cc-egos 3980 \
  --p-delete-values 0.05 \
  --seeds 1 \
  --d-hat 8 \
  --lambda-value 5 \
  --dry-run
```

## Interrupting and resuming

The runner saves the table and progress after every completed task.

To resume an interrupted `run_experiment.py` job, run the same command again
without `--restart`. Keep the same input table, parameters, progress path, and
manifest path.

Use `--restart` only when the selected results should be recomputed and
overwritten.

## Building the paper tables

After the experiment table has been updated, run:

```bash
python scripts/make_paper_tables.py \
  --d-hat 8 \
  --lambda-value 5
```

This produces:

```text
results/research_tables/facebook_minmax_table.csv
results/research_tables/facebook_correlation_clustering_table.csv
results/research_tables/clique_correlation_clustering_table.csv
```

The ordinary correlation-clustering tables report Pivot costs and, when an LP
lower bound is available, Pivot-to-LP approximation ratios. The MinMax table
contains the best, average, and worst MinMaxCC results over the deletion seeds,
together with the corresponding MinMaxLP values and ratios.

## Building the figures

```bash
python scripts/make_facebook_minmax_figures.py
python scripts/make_facebook_approximation_range_figures.py
```

Figures are written to:

```text
results/figures/research_figures/
```

The approximation-range figures show the average value as a line and the
best-to-worst range as a shaded band.

## Runtime measurements

MinMaxCC and MinMaxLP runtimes are stored by `run_experiment.py`. Ordinary
Pivot and LP runtimes are measured separately so the runtime benchmark does
not change the experiment results.

```bash
python scripts/capture_machine_specifications.py

caffeinate -i python scripts/benchmark_normal_cc_runtimes.py \
  --dataset all \
  --algorithms pivot \
  --pivot-runs 30 \
  --restart

caffeinate -i python scripts/benchmark_normal_cc_runtimes.py \
  --dataset all \
  --algorithms lp \
  --lp-repetitions 1

python scripts/apply_runtime_benchmarks.py
```

Run runtime benchmarks when no other experiment is using the machine.

## Why the experiments are reproducible

The experiment pipeline fixes the parts that can otherwise change between
runs:

- Facebook vertices are taken only from sorted `.edges` endpoints.
- Edge deletion uses recorded probabilities and seeds.
- Pivot uses NumPy `default_rng`, fixed seeds, and sorted active vertices.
- The selected parameters are passed explicitly in the reproduction commands.
- Long runs use progress files and atomic CSV checkpoints.
- Manifests record the configuration, graph and code hashes, software versions,
  Git information, completed tasks, and failures.
- Paper tables and figures are generated from saved CSV results rather than
  copied manually.
- Machine and software details can be saved with
  `capture_machine_specifications.py`.

## Repository structure

```text
data/facebook/
    Facebook ego-network files

src/
    algorithm implementations and graph utilities

scripts/run_experiment.py
    ordinary LP, MinMaxCC, and MinMaxLP experiment runner

scripts/update_pivot_results_all_egos.py
    complete and edge-deleted Pivot experiments

scripts/update_complete_pivot_big_egos.py
    complete-only Pivot experiments for larger ego graphs

scripts/make_paper_tables.py
    creates the paper tables from the flat result tables

scripts/make_facebook_minmax_figures.py
scripts/make_facebook_approximation_range_figures.py
    creates the Facebook figures

results/research_tables/
    raw, processed, progress, manifest, and runtime tables

results/figures/research_figures/
    generated paper figures
```

## Main outputs

| File | Description |
|---|---|
| `minmax_facebook_grid_runs_flat.csv` | full Facebook experiment grid |
| `facebook_minmax_table.csv` | MinMax paper table |
| `facebook_correlation_clustering_table.csv` | Facebook Pivot/LP paper table |
| `clique_correlation_clustering_table.csv` | synthetic clique paper table |
| `*_progress.json` | resumable task state |
| `*_manifest.json` | experiment configuration and verification metadata |
