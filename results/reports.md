# All-pairs Pivot Approximation Outliers

This report summarizes cases where the average Pivot cost is more than 3 times the all-pairs ILP optimum:

```text
pivot_all_pairs_ratio = edge_pivot_average_cost / edge_all_pairs_ilp_cost
```

## Overall summary

| Metric | Value |
| --- | --- |
| Total outlier cases | 491 |
| Clique outlier cases | 463 |
| Random outlier cases | 28 |
| Mean ratio among outliers | 3.60 |
| Maximum ratio | 12.97 |
| Worst case | `clq_n15_8_7.json`, seed `37`, `p_delete=0.40` |

## Outliers by edge deletion probability

| `p_delete` | Outlier cases |
| --- | --- |
| 0.05 | 13 |
| 0.15 | 70 |
| 0.25 | 164 |
| 0.40 | 244 |

## Clique balance summary

| Category | Hits | Total clique runs | Hit rate | Mean outlier ratio | Max outlier ratio |
| --- | --- | --- | --- | --- | --- |
| balanced | 184 | 1390 | 13.2% | 3.70 | 5.76 |
| near_balanced | 109 | 1000 | 10.9% | 3.71 | 12.97 |
| imbalanced | 170 | 1295 | 13.1% | 3.44 | 6.24 |

## Clique structure detail

| File | Category | Hits | Total runs | Hit rate | Mean outlier ratio | Max outlier ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `clq_n30_20_5_5.json` | imbalanced | 69 | 200 | 34.5% | 3.41 | 4.46 |
| `clq_n30_2x15.json` | balanced | 64 | 200 | 32.0% | 3.50 | 4.73 |
| `clq_n100_60_25_10_5.json` | imbalanced | 28 | 95 | 29.5% | 3.23 | 3.54 |
| `clq_n20_2x10.json` | balanced | 58 | 200 | 29.0% | 3.60 | 5.71 |
| `clq_n25_13_12.json` | near_balanced | 52 | 200 | 26.0% | 3.53 | 5.76 |
| `clq_n10_2x5.json` | balanced | 45 | 200 | 22.5% | 4.10 | 5.75 |
| `clq_n15_8_7.json` | near_balanced | 33 | 200 | 16.5% | 4.12 | 12.97 |
| `clq_n10_5_3_2.json` | imbalanced | 25 | 200 | 12.5% | 3.73 | 5.84 |
| `clq_n30_15_10_5.json` | imbalanced | 19 | 200 | 9.5% | 3.26 | 3.72 |
| `clq_n25_12_7_6.json` | imbalanced | 13 | 200 | 6.5% | 3.14 | 3.38 |
| `clq_n20_7_7_6.json` | near_balanced | 12 | 200 | 6.0% | 3.44 | 4.28 |
| `clq_n25_10_10_5.json` | imbalanced | 9 | 200 | 4.5% | 3.33 | 3.65 |
| `clq_n15_3x5.json` | balanced | 8 | 200 | 4.0% | 3.91 | 5.76 |
| `clq_n10_4_3_3.json` | near_balanced | 7 | 200 | 3.5% | 4.01 | 5.17 |
| `clq_n15_5_5_3_2.json` | imbalanced | 7 | 200 | 3.5% | 4.61 | 6.24 |
| `clq_n30_3x10.json` | balanced | 7 | 200 | 3.5% | 3.25 | 3.72 |
| `clq_n25_9_8_8.json` | near_balanced | 5 | 200 | 2.5% | 3.23 | 3.35 |
| `clq_n20_4x5.json` | balanced | 2 | 200 | 1.0% | 4.87 | 5.20 |
| `clq_n100_10x10.json` | balanced | 0 | 95 | 0.0% | 0.00 | 0.00 |
| `clq_n100_4x25.json` | balanced | 0 | 95 | 0.0% | 0.00 | 0.00 |

## Interpretation

The outliers are concentrated in clique-based instances: 463 of the 491 cases are clique graphs.

Balanced two-clique instances can show many high-ratio cases, but the effect is not limited to balanced cliques. The safer conclusion is that high Pivot/all-pairs ILP ratios are common in clique-structured instances, especially after substantial edge deletion.
