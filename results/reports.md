# Paper Table and Figure Definitions

This document describes the three generated paper tables, the meaning of every
column, the aggregation rules, and the columns used by the Facebook MinMax
figures.

## General conventions

- `p_delete = 0` denotes the complete graph.
- `p_delete > 0` denotes an edge-deleted graph.
- Empty cells mean that the value was unavailable or was not computed.
- Costs and approximation ratios are aggregated separately. A seed attaining
  the best cost does not necessarily attain the best ratio.
- For Facebook edge-deletion experiments, a deletion seed defines a distinct
  graph instance.
- For Pivot, each graph instance is run with 100 Pivot seeds. Thus, each
  deletion seed has a best-of-100 Pivot cost and a mean-of-100 Pivot cost.

---

## `facebook_correlation_clustering_table.csv`

Ordinary correlation clustering results for the Facebook ego graphs.

### Row definition

- A row with `p_delete = 0` represents one complete ego graph.
- A row with `p_delete > 0` aggregates all available deletion seeds for one
  ego graph and one deletion probability.
- Complete ego graphs for which only Pivot was run remain in the table. Their
  LP, approximation-ratio, and LP-runtime cells are empty.

### Columns

| Column | Definition |
|---|---|
| `ego_id` | Facebook ego-network identifier. |
| `n` | Number of vertices after corrected preprocessing. Only vertices occurring as endpoints in the `.edges` file are included. |
| `p_delete` | Edge-deletion probability. Zero means the complete graph. |
| `number_of_seeds` | Number of deletion seeds represented by the row. This is `1` for a complete graph and normally `30` for an edge-deleted row. |
| `pivot_best_cost` | Complete row: minimum cost among the 100 Pivot runs. Edge row: arithmetic mean, over deletion seeds, of each seed's best-of-100 Pivot cost. |
| `pivot_average_cost` | Complete row: arithmetic mean cost of the 100 Pivot runs. Edge row: arithmetic mean, over deletion seeds, of each seed's mean cost across its 100 Pivot runs. |
| `averagepivot_approximation` | Complete row: `pivot_average_cost / complete LP`. Edge row: arithmetic mean over deletion seeds of `(per-seed mean-of-100 Pivot cost) / (same-instance LP)`. It is not computed as a ratio of two aggregated means. |
| `bestpivot_approximation` | Complete row: `pivot_best_cost / complete LP`. Edge row: arithmetic mean over deletion seeds of `(per-seed best-of-100 Pivot cost) / (same-instance LP)`. |
| `pivot_runtime_seconds_average` | Runtime imported from `normal_cc_runtime_benchmarks.csv`. Pivot timing measures one Pivot call; graph construction and cost evaluation are outside the timer. The current Facebook edge benchmark uses deletion seed 1 for each `ego_id, p_delete` combination. |
| `lp_runtime_seconds_average` | Runtime imported from `normal_cc_runtime_benchmarks.csv`. The benchmark summary uses Gurobi solver runtime when available. It is empty when no ordinary LP was benchmarked. |

---

## `clique_correlation_clustering_table.csv`

Ordinary correlation clustering results for the synthetic clique graphs.

### Row definition

There is one row for each pair `(n, p_delete)`. Balanced and unbalanced clique
instances, graph configurations, and available graph seeds in that group are
merged.

### Columns

| Column | Definition |
|---|---|
| `n` | Number of vertices in the clique graph. |
| `p_delete` | Edge-deletion probability. Zero means the complete graph. |
| `averagepivot_approximation` | Arithmetic mean across the grouped clique instances of `(instance mean Pivot cost) / (same-instance ordinary LP)`. |
| `bestpivot_approximation` | Arithmetic mean across the grouped clique instances of `(instance best Pivot cost) / (same-instance ordinary LP)`. It is not the single most favorable ratio from the group. |
| `pivot_runtime_seconds_average` | Average Pivot runtime for the corresponding `n, p_delete` benchmark group, imported from `normal_cc_runtime_benchmarks.csv`. |
| `lp_runtime_seconds_average` | Average ordinary LP solver runtime for the corresponding `n, p_delete` benchmark group, imported from `normal_cc_runtime_benchmarks.csv`. |

---

## `facebook_minmax_table.csv`

MinMaxCC results and MinMaxLP-reference comparisons for the Facebook ego
graphs.

### Row definition

- A row with `p_delete = 0` represents the complete ego graph.
- A row with `p_delete > 0` aggregates the available deletion seeds, normally
  30.
- MinMaxCC costs, MinMaxLP-reference values, and approximation ratios are
  summarized independently.

### Columns

| Column | Definition |
|---|---|
| `ego_id` | Facebook ego-network identifier. |
| `n` | Number of vertices after corrected `.edges`-endpoint preprocessing. |
| `p_delete` | Edge-deletion probability. Zero means the complete graph. |
| `d_hat` | `d_hat` parameter used by MinMaxCC. |
| `lambda` | `lambda` parameter used by MinMaxCC. |
| `number_of_seeds` | Number of deletion seeds represented by the row. This is `1` for complete graphs and normally `30` for edge-deleted rows. |
| `minmaxcc_cost_best` | Minimum MinMaxCC objective value over the represented deletion seeds. |
| `minmaxcc_cost_average` | Arithmetic mean MinMaxCC objective value over the represented deletion seeds. |
| `minmaxcc_cost_worst` | Maximum MinMaxCC objective value over the represented deletion seeds. |
| `min_max_lp_cost_minimum` | Minimum available MinMaxLP-reference value over the represented seeds. |
| `min_max_lp_cost_average` | Arithmetic mean of the available MinMaxLP-reference values over the represented seeds. |
| `min_max_lp_cost_maximum` | Maximum available MinMaxLP-reference value over the represented seeds. |
| `minmaxcc_ratio_best` | Minimum of the per-seed ratios `MinMaxCC(seed) / MinMaxLP-reference(seed)`. |
| `minmaxcc_ratio_average` | Arithmetic mean of the per-seed ratios `MinMaxCC(seed) / MinMaxLP-reference(seed)`. This is not `average cost / average LP`. |
| `minmaxcc_ratio_worst` | Maximum of the per-seed ratios `MinMaxCC(seed) / MinMaxLP-reference(seed)`. |
| `minmaxcc_runtime_seconds_average` | Complete row: runtime of the complete MinMaxCC run. Edge row: arithmetic mean of the MinMaxCC runtimes over deletion seeds. |
| `min_max_lp_runtime_seconds_average` | For locally computed MinMaxLP results, the table uses total runtime when all required total-runtime values are present; otherwise it falls back to LP-solve runtime. The value is empty for external or unavailable LP references. Until the LP-plus-rounding rerun is complete, this column may therefore represent LP-only runtime. |
| `lp_reference_source` | Origin and comparability of the LP reference. Values are described below. |

### `lp_reference_source` values

| Value | Meaning |
|---|---|
| `computed_complete_same_instance` | MinMaxLP was solved locally on the same complete graph instance. |
| `computed_edge_same_instance` | MinMaxLP was solved locally on the same edge-deleted graph instance. |
| `davies2023_complete_graph_lp` | The complete-graph LP objective reported by Davies et al. is used as an external reference. For edge-deleted rows this is not a same-instance LP. |
| `unavailable` | No valid MinMaxLP objective was available; LP and ratio cells are empty. |

### Important interpretation rule

The cost and ratio columns are deliberately independent. For example, the seed
with the smallest MinMaxCC cost can differ from the seed with the smallest
MinMaxCC/MinMaxLP ratio because the LP denominator also varies by seed.

---

## Facebook MinMax figures

The current figure scripts are compatible with the current
`facebook_minmax_table.csv`.

### Detailed figures 1–6

`make_facebook_minmax_figures.py` uses:

- average figures: `minmaxcc_ratio_average`
- worst figures: `minmaxcc_ratio_worst`
- best figures: `minmaxcc_ratio_best`

### Range figures 7–8

`make_facebook_approximation_range_figures.py` uses:

- line: aggregated mean of `minmaxcc_ratio_average`
- lower band: aggregated minimum of `minmaxcc_ratio_best`
- upper band: aggregated maximum of `minmaxcc_ratio_worst`

By default, both scripts include only rows whose `lp_reference_source` begins
with `computed_`. Therefore, figures use locally solved, same-instance LP
comparisons and exclude Davies external-reference and unavailable rows.

Do not pass `--include-external-reference-rows` for the final paper figures.

---

## Generation commands

```bash
python scripts/make_paper_tables.py \
  --d-hat 8 \
  --lambda-value 5

python scripts/make_facebook_minmax_figures.py
python scripts/make_facebook_approximation_range_figures.py
```

