# Table and Figure Guide

This file explains what every table column means and how the values are
calculated.

## General notes

- `p_delete = 0` means the complete graph.
- `p_delete > 0` means an edge-deleted graph.
- An empty cell means that the value was not available or was not calculated.
- Costs and ratios are summarized separately.
- A deletion seed creates a different graph instance.
- Pivot is run 100 times on every Facebook graph instance.

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

This table contains the Facebook MinMaxCC and MinMaxLP results.

A complete row has `p_delete = 0`.

An edge-deleted row combines the available deletion seeds, normally 30.

| Column | Meaning |
|---|---|
| `ego_id` | Facebook ego-network ID. |
| `n` | Number of vertices after the corrected preprocessing. |
| `p_delete` | Edge-deletion probability. `0` means complete graph. |
| `d_hat` | Final `d_hat` used by MinMaxCC. |
| `lambda` | Final `lambda` used by MinMaxCC. |
| `number_of_seeds` | Number of deletion seeds represented by the row. |
| `minmaxcc_cost_best` | Minimum MinMaxCC cost over the represented seeds. |
| `minmaxcc_cost_average` | Average MinMaxCC cost over the represented seeds. |
| `minmaxcc_cost_worst` | Maximum MinMaxCC cost over the represented seeds. |
| `min_max_lp_cost_minimum` | Minimum available MinMaxLP value over the represented seeds. |
| `min_max_lp_cost_average` | Average available MinMaxLP value over the represented seeds. |
| `min_max_lp_cost_maximum` | Maximum available MinMaxLP value over the represented seeds. |
| `minmaxcc_ratio_best` | Minimum per-seed `MinMaxCC / MinMaxLP` ratio. |
| `minmaxcc_ratio_average` | Average of the per-seed `MinMaxCC / MinMaxLP` ratios. |
| `minmaxcc_ratio_worst` | Maximum per-seed `MinMaxCC / MinMaxLP` ratio. |
| `minmaxcc_runtime_seconds_average` | Complete graph: runtime of that MinMaxCC run. Edge-deleted graph: average MinMaxCC runtime over deletion seeds. |
| `min_max_lp_runtime_seconds_average` | Average total MinMaxLP runtime. Total runtime is LP runtime plus rounding runtime. Empty for external or unavailable LP values. |
| `lp_reference_source` | Shows where the LP value came from. |

### `lp_reference_source`

| Value | Meaning |
|---|---|
| `computed_complete_same_instance` | LP was solved locally on the same complete graph. |
| `computed_edge_same_instance` | LP was solved locally on the same edge-deleted graph. |
| `davies2023_complete_graph_lp` | LP value came from Davies et al. |
| `unavailable` | No valid LP value was available. |

Costs and ratios are independent. The seed with the lowest cost can be
different from the seed with the lowest ratio.

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

## Commands

```bash
python scripts/make_paper_tables.py   --d-hat 8   --lambda-value 5

python scripts/make_facebook_minmax_figures.py
python scripts/make_facebook_approximation_range_figures.py
```

`make_paper_tables.py` already adds the ordinary runtimes. Do not run
`apply_runtime_benchmarks.py` afterward.
