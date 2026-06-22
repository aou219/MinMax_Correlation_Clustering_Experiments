# All-pairs thesis results

Runs: 12,101. All-pairs LP: 12,101. Optimal all-pairs ILP: 12,097. Facebook ego 686 has LP only (4 runs).

Complete-graph sparse LP/ILP values are also the all-pairs values because every vertex pair is observed.
For edge-deleted graphs, the report retains the sparse formulation and adds the exact all-pairs formulation.

## Main validation

All LP bounds satisfy LP <= ILP. All reported ILPs are optimal. The all-pairs objective exceeds the sparse+4 objective in 31 runs.

## Outputs

Nine CSV tables are in `tables/`; seven PNG figures are in `figures/`.
The all-pairs LP/ILP ratio is undefined when the exact ILP objective is zero; those rows are omitted from ratio means.
