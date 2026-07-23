# Table and Figure Guide

This file explains what every table column means and how the values are
calculated.

## General notes

- `p_delete = 0` means the complete graph.
- `p_delete > 0` means an edge-deleted graph.
- An empty cell means that the value was not available or was not calculated.
- The final MinMax table reports ratios instead of repeating raw costs.
- A deletion seed creates a different graph instance.
- Pivot is run 100 times on every Facebook graph instance.

## Runtime coverage

This section shows exactly where runtime values are available in the final
tables.

The counts below apply after running the final `make_paper_tables.py`.

| Algorithm and dataset | Runtime available for | Runtime not available for | How it was measured |
|---|---|---|---|
| Ordinary Pivot — Facebook | All 26 rows in `facebook_correlation_clustering_table.csv`. This includes the 10 complete ego graphs and the 16 edge-deleted rows for ego IDs `414, 686, 698, 3980`. | None of the final Facebook ordinary-Pivot rows. | 30 separately timed Pivot calls. The table stores the average time for one Pivot call. For an edge-deleted setting, the runtime benchmark uses deletion seed 1 for that `ego_id, p_delete` combination. |
| Ordinary LP — Facebook | 20 of 26 rows: the complete and four edge-deleted settings for ego IDs `414, 686, 698, 3980`. | The six complete rows for ego IDs `0, 107, 348, 1684, 1912, 3437`. These LPs were not locally solved or benchmarked. | One LP solve per benchmark graph. The table uses Gurobi solver runtime when available. |
| Ordinary Pivot — clique graphs | All 30 `n, p_delete` groups in `clique_correlation_clustering_table.csv`. | None of the final clique Pivot rows. | The benchmark table stores the average Pivot runtime for each grouped `n, p_delete` setting. |
| Ordinary LP — clique graphs | All 30 `n, p_delete` groups in `clique_correlation_clustering_table.csv`. | None of the final clique LP rows. | The table uses the LP runtime stored in the runtime benchmark summary. Existing archived clique LP timings are reused where appropriate. |
| MinMaxCC — Facebook | All 50 rows in `facebook_minmax_table.csv`: 10 ego IDs, each with one complete and four edge-deleted settings. | None. | Complete row: one MinMaxCC run. Edge-deleted row: average runtime over the 30 deletion seeds. |
| MinMaxLP + rounding — Facebook | 20 of 50 rows: the complete and four edge-deleted settings for ego IDs `414, 686, 698, 3980`. | The 30 rows for ego IDs `0, 107, 348, 1684, 1912, 3437`. Ego ID `348` uses an external Davies LP objective, so it has no local runtime. The other large ego IDs have no valid local MinMaxLP result. | Complete row: total runtime of one local run. Edge-deleted row: average total runtime over 30 deletion seeds. Total runtime is LP solve time plus rounding time. |

### Where the runtime values come from

Ordinary Pivot and ordinary LP runtimes come from:

```text
results/research_tables/normal_cc_runtime_benchmarks.csv
```

MinMaxCC and MinMaxLP runtimes are stored directly in:

```text
results/research_tables/minmax_facebook_grid_runs_flat.csv
```

The final `make_paper_tables.py` reads both sources and writes the runtime
columns into the three paper tables.

A clearly interrupted MinMaxLP runtime can be excluded from only the runtime
average with:

```text
--exclude-minmax-runtime-above 4000
```

This does not remove the run's objective, clustering, rounding result, or ratio.

### Why some runtime cells are empty

An empty runtime cell does not mean the algorithm failed. It means that the
corresponding algorithm was not run locally for that graph, or that only an
external objective value was available.

In particular:

- the six larger ordinary Facebook graphs only have complete Pivot results, so
  their ordinary LP runtimes are empty;
- ego ID `348` has a Davies MinMaxLP objective but no local MinMaxLP runtime;
- the other larger Facebook ego IDs have neither a local MinMaxLP objective nor
  a local MinMaxLP runtime.

## `facebook_correlation_clustering_table.csv`

This table contains the ordinary correlation-clustering results for the
Facebook graphs.

For complete graphs, there is one row per ego ID.

For edge-deleted graphs, one row combines the available deletion seeds for one
ego ID and one `p_delete` value.

| Column | Meaning |
|---|---|
| `ego_id` | Facebook ego-network ID. |
| `n` | Number of vertices after keeping only endpoints from the `.edges` file. |
| `p_delete` | Edge-deletion probability. `0` means complete graph. |
| `number_of_seeds` | Number of deletion seeds used in the row. |
| `pivot_best_cost` | Complete graph: best cost from 100 Pivot runs. Edge-deleted graph: average over deletion seeds of each seed's best-of-100 Pivot cost. |
| `pivot_average_cost` | Complete graph: average cost from 100 Pivot runs. Edge-deleted graph: average over deletion seeds of each seed's mean-of-100 Pivot cost. |
| `averagepivot_approximation` | Complete graph: average Pivot cost divided by the complete LP. Edge-deleted graph: average of the per-seed mean-Pivot/LP ratios. |
| `bestpivot_approximation` | Complete graph: best Pivot cost divided by the complete LP. Edge-deleted graph: average of the per-seed best-Pivot/LP ratios. |
| `pivot_runtime_seconds_average` | Pivot runtime from `normal_cc_runtime_benchmarks.csv`. |
| `lp_runtime_seconds_average` | Ordinary LP runtime from `normal_cc_runtime_benchmarks.csv`. Empty when no LP was benchmarked. |

For edge-deleted rows, the ratio is always calculated using Pivot and LP values
from the same deletion seed.

## `clique_correlation_clustering_table.csv`

This table contains ordinary correlation-clustering results for the synthetic
clique graphs.

Rows are grouped by `n` and `p_delete`.

| Column | Meaning |
|---|---|
| `n` | Number of vertices. |
| `p_delete` | Edge-deletion probability. `0` means complete graph. |
| `averagepivot_approximation` | Average across the grouped clique instances of `(average Pivot cost) / LP`. |
| `bestpivot_approximation` | Average across the grouped clique instances of `(best Pivot cost) / LP`. |
| `pivot_runtime_seconds_average` | Average Pivot runtime for that `n, p_delete` group. |
| `lp_runtime_seconds_average` | Average ordinary LP runtime for that `n, p_delete` group. |

## `facebook_minmax_table.csv`

This table contains the final Facebook MinMax comparisons.

A complete row has `p_delete = 0`. An edge-deleted row combines the available
deletion seeds, normally 30. The fixed MinMaxCC setting is `d_hat = 8` and
`lambda = 5`; these values are documented in the README instead of being
repeated in every row.

The table does not include raw MinMaxCC costs, raw LP objectives, raw rounding
costs, or `number_of_seeds`. Those values remain available in
`minmax_facebook_grid_runs_flat.csv`.

| Column | Meaning |
|---|---|
| `ego_id` | Facebook ego-network ID. |
| `n` | Number of vertices after the corrected preprocessing. |
| `p_delete` | Edge-deletion probability. `0` means complete graph. |
| `minmaxcc_ratio_best` | Minimum per-seed ratio `MinMaxCC objective / MinMaxLP objective`. |
| `minmaxcc_ratio_average` | Average of the per-seed ratios `MinMaxCC objective / MinMaxLP objective`. |
| `minmaxcc_ratio_worst` | Maximum per-seed ratio `MinMaxCC objective / MinMaxLP objective`. |
| `minmaxcc_to_lp_rounding_clustering_ratio_best` | Minimum per-seed ratio `MinMaxCC objective / LP-rounding clustering cost`. |
| `minmaxcc_to_lp_rounding_clustering_ratio_average` | Average of the per-seed ratios `MinMaxCC objective / LP-rounding clustering cost`. |
| `minmaxcc_to_lp_rounding_clustering_ratio_worst` | Maximum per-seed ratio `MinMaxCC objective / LP-rounding clustering cost`. |
| `minmaxcc_runtime_seconds_average` | Complete graph: runtime of the MinMaxCC run. Edge-deleted graph: average MinMaxCC runtime over deletion seeds. |
| `min_max_lp_runtime_seconds_average` | Complete graph: local LP-plus-rounding total runtime. Edge-deleted graph: average local total runtime over deletion seeds after any explicitly excluded interrupted runtime sample. |
| `lp_reference_source` | Shows where the LP value used in the first ratio came from. |

### How to read the two ratios

The LP ratio uses the fractional MinMaxLP objective as a lower bound.

The LP-rounding ratio compares two actual clusterings:

```text
MinMaxCC objective / LP-rounding clustering cost
```

For this ratio:

- below `1`: MinMaxCC has the lower maximum-disagreement cost;
- above `1`: the LP-rounding clustering has the lower cost;
- equal to `1`: both clusterings have the same cost.

For both ratio families, `best` is the minimum ratio, `average` is the
arithmetic mean, and `worst` is the maximum ratio over deletion seeds.

The LP-rounding ratio is only available where LP rounding was run locally. It
is blank for external Davies-reference rows.

### `lp_reference_source`

| Value | Meaning |
|---|---|
| `computed_complete_same_instance` | LP was solved locally on the same complete graph. |
| `computed_edge_same_instance` | LP was solved locally on the same edge-deleted graph. |
| `davies2023_complete_graph_lp` | LP value came from Davies et al.; no local rounding comparison is available. |
| `unavailable` | No valid LP value was available. |


## Figures

### Figures 1-6

`make_facebook_minmax_figures.py` uses:

- average: `minmaxcc_ratio_average`
- worst: `minmaxcc_ratio_worst`
- best: `minmaxcc_ratio_best`

### Figures 7-8

`make_facebook_approximation_range_figures.py` uses:

- line: average of `minmaxcc_ratio_average`
- lower band: minimum of `minmaxcc_ratio_best`
- upper band: maximum of `minmaxcc_ratio_worst`

The figure scripts only use rows where `lp_reference_source` starts with
`computed_`. This means the final figures only use local LP values from the
same graph instances.

The current figures use `minmaxcc_ratio_best`, `minmaxcc_ratio_average`, and
`minmaxcc_ratio_worst`. The LP-rounding comparison columns are added to the
table but are not automatically plotted by the existing figure scripts.

## Commands

```bash
python scripts/make_paper_tables.py \
  --d-hat 8 \
  --lambda-value 5 \
  --exclude-minmax-runtime-above 4000

python scripts/make_facebook_minmax_figures.py
python scripts/make_facebook_approximation_range_figures.py
```

`make_paper_tables.py` already adds the ordinary runtimes. Do not run
`apply_runtime_benchmarks.py` afterward.
