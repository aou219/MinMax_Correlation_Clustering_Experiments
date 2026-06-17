# Thesis results report

This report summarizes the processed results from `all_runs_flat.csv`.

Ratios are interpreted as follows: `Pivot/ILP = 1` means Pivot is optimal; higher is worse. `LP/ILP = 1` means the LP relaxation is tight; lower means the LP is looser.

## 1. Data overview

- Total runs: **2625**
- Random graph runs: **2100**
- Clique/community graph runs: **521**
- Facebook ego-network runs: **4**

## 2. Random graphs

| p+ | runs | bad triangle density | ILP cost | LP/ILP | Pivot/ILP | max disjoint/ILP |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 300 | 0.094 | 23.323 | 0.829 | 1.189 | 0.758 |
| 0.3 | 300 | 0.183 | 36.437 | 0.816 | 1.210 | 0.762 |
| 0.4 | 300 | 0.277 | 48.390 | 0.823 | 1.203 | 0.771 |
| 0.5 | 300 | 0.367 | 57.977 | 0.852 | 1.176 | 0.796 |
| 0.6 | 300 | 0.433 | 62.060 | 0.913 | 1.175 | 0.849 |
| 0.7 | 300 | 0.449 | 52.317 | 0.994 | 1.234 | 0.949 |
| 0.8 | 300 | 0.387 | 35.683 | 1.000 | 1.292 | 0.987 |

The highest bad-triangle density occurs around **p+=0.7**. The highest average ILP cost occurs around **p+=0.6**. The loosest LP relaxation occurs around **p+=0.3**, with LP/ILP ≈ **0.816**. Pivot is worst on average around **p+=0.8**, with Pivot/ILP ≈ **1.292**.

## 3. Clique/community graphs

| n | clusters | runs | Pivot/ILP complete | LP/ILP complete | Pivot/ILP new | LP/ILP new | avg bad 4-cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2x5 | 10 | 1.080 | 1.000 | 1.367 | 1.000 | 0.900 |
| 10 | 3_2_5 | 10 | 1.100 | 0.988 | 1.200 | 1.000 | 0.500 |
| 10 | 3_3_4 | 10 | 1.020 | 0.986 | 1.200 | 1.000 | 0.300 |
| 15 | 3x5 | 10 | 1.234 | 1.000 | 1.598 | 1.000 | 2.800 |
| 15 | 5_2_5_3 | 10 | 1.193 | 0.987 | 1.353 | 0.990 | 2.600 |
| 15 | 7_8 | 10 | 1.195 | 1.000 | 1.560 | 1.000 | 7.100 |
| 20 | 2x10 | 10 | 1.206 | 1.000 | 1.658 | 1.000 | 17.500 |
| 20 | 4x5 | 10 | 1.274 | 1.000 | 1.443 | 1.000 | 5.700 |
| 20 | 7_6_7 | 10 | 1.290 | 1.000 | 1.460 | 1.000 | 7.900 |
| 25 | 12_13 | 10 | 1.246 | 1.000 | 1.878 | 1.000 | 61.200 |
| 25 | 12_6_7 | 10 | 1.477 | 1.000 | 1.801 | 1.000 | 35.500 |
| 25 | 5_10_10 | 10 | 1.357 | 1.000 | 1.679 | 1.000 | 35.500 |
| 25 | 8_8_9 | 10 | 1.428 | 1.000 | 1.642 | 1.000 | 32.200 |
| 30 | 15_10_5 | 60 | 1.514 | 1.000 | 1.903 | 1.000 | 88.017 |
| 30 | 20_5_5 | 60 | 1.490 | 1.000 | 2.004 | 1.000 | 127.167 |
| 30 | 2x15 | 60 | 1.413 | 1.000 | 2.048 | 1.000 | 129.333 |
| 30 | 3x10 | 60 | 1.460 | 1.000 | 1.833 | 1.000 | 69.283 |
| 100 | 10x10 | 50 | 1.535 | 0.862 | 1.656 | 0.861 | 2173.740 |
| 100 | 4x25 | 51 | 1.828 | 1.000 | 2.097 | 1.000 | 7462.686 |
| 100 | 60_25_10_5 | 50 | 1.948 | 1.000 | 2.398 | 1.000 | 16209.860 |

LP is often tight on clique/community graphs, while Pivot becomes worse for larger and more unbalanced structures, especially after new graph.

## 4. Facebook ego-networks

| ego | n | has ILP | Pivot/ILP | LP/ILP | ILP complete | LP complete | ILP new | LP new |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3980 | 58 | yes | 1.173 | 0.913 | 75.000 | 68.500 | 67.000 | 60.500 |
| 698 | 64 | yes | 1.227 | 0.969 | 97.000 | 94.000 | 82.000 | 79.000 |
| 414 | 155 | yes | 1.374 | 0.927 | 797.000 | 739.000 | 670.000 | 628.500 |
| 686 | 170 | no | - | - | - | 826.500 | - | 711.000 |

Facebook ego-networks behave between random graphs and synthetic clique/community graphs. They have real structure, but not the clean planted structure of the clique instances.

## 5. Bad triangles

| graph family | runs | max disjoint/ILP complete | max disjoint/ILP new | bad triangle density | bad triangles removed | corr bad triangles vs ILP |
| --- | --- | --- | --- | --- | --- | --- |
| random | 2100 | 0.841 | 0.845 | 0.313 | 0.569 | 0.948 |
| clique | 521 | 0.948 | 0.954 | 0.137 | 0.382 | 0.920 |
| facebook | 4 | 0.849 | 0.835 | 0.027 | 0.353 | 1.000 |

The maximum edge-disjoint bad-triangle count is a lower bound on the ILP cost. When this ratio is close to 1, local bad-triangle structure explains much of the optimum cost.

## 6. Bad 4-cycle constraints

| graph family | runs with both costs | cost changed | cost changed % | known clusterings | clustering changed | clustering changed % | same cost different clustering |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2100 | 85 | 4.048 | 2100 | 584 | 27.810 | 503 |
| clique | 521 | 3 | 0.576 | 521 | 16 | 3.071 | 14 |
| facebook | 3 | 1 | 33.333 | 3 | 2 | 66.667 | 1 |

Overall, the ILP cost with and without 4-cycle constraints can be compared in **2624** runs. The objective cost changed in **89** runs, which is **3.4%**. The clustering comparison is known in **2624** runs, and the clustering changed in **602** runs, which is **22.9%**. In **518** runs, the cost stayed the same but the clustering changed.

## 7. Research question answers

### RQ1 — New graph

New graph usually lowers the absolute ILP cost, because fewer edges remain in the objective. At the same time, Pivot often becomes relatively worse after new graph, especially on clique/community graphs. This suggests that deleted edges remove structural information that Pivot needs.

### RQ2 — Input structure

Input structure strongly affects the methods. Random graphs with many bad triangles tend to have higher costs and looser LP relaxations. Clique/community graphs often make LP tight, but Pivot struggles more on larger and unbalanced structures. Facebook ego-networks sit between random and clique graphs.

### RQ3 — LP vs ILP

LP is often close to ILP for clique/community graphs, but it can be loose for random graphs with many inconsistent local structures. ILP is practical for small and medium instances, but larger Facebook ego-networks and 4-cycle constraint generation become computationally expensive.

---

## 8. Additional size-effect tables

This section adds the size-based comparisons that are useful for interpreting the results by graph size and clique/community structure.

### 8.1 Random graphs by graph size

| n | runs | ILP cost | LP/ILP complete | Pivot/ILP complete | LP/ILP new | Pivot/ILP new | bad triangle density |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 350 | 1.509 | 0.988 | 1.111 | 0.995 | 1.171 | 0.312 |
| 10 | 350 | 9.120 | 0.960 | 1.141 | 0.980 | 1.195 | 0.310 |
| 15 | 350 | 24.057 | 0.902 | 1.211 | 0.940 | 1.236 | 0.312 |
| 20 | 350 | 46.186 | 0.865 | 1.248 | 0.907 | 1.270 | 0.314 |
| 25 | 350 | 76.380 | 0.833 | 1.275 | 0.873 | 1.296 | 0.315 |
| 30 | 350 | 113.766 | 0.812 | 1.267 | 0.847 | 1.314 | 0.314 |

This table checks whether graph size changes the approximation behaviour. It separates the effect of graph size `n` from the effect of the positive-edge probability `p+`.

### 8.2 Clique/community size effect

| clique sizes | # cliques | max clique | imbalance | LP/ILP complete | Pivot/ILP complete | LP/ILP new | Pivot/ILP new | bad 4-cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-3-4 | 3 | 4 | 1.333 | 0.986 | 1.020 | 1.000 | 1.200 | 0.300 |
| 5-5 | 2 | 5 | 1.000 | 1.000 | 1.080 | 1.000 | 1.367 | 0.900 |
| 3-2-5 | 3 | 5 | 2.500 | 0.988 | 1.100 | 1.000 | 1.200 | 0.500 |
| 5-5-5 | 3 | 5 | 1.000 | 1.000 | 1.234 | 1.000 | 1.598 | 2.800 |
| 5-2-5-3 | 4 | 5 | 2.500 | 0.987 | 1.193 | 0.990 | 1.353 | 2.600 |
| 5-5-5-5 | 4 | 5 | 1.000 | 1.000 | 1.274 | 1.000 | 1.443 | 5.700 |
| 7-6-7 | 3 | 7 | 1.167 | 1.000 | 1.290 | 1.000 | 1.460 | 7.900 |
| 7-8 | 2 | 8 | 1.143 | 1.000 | 1.195 | 1.000 | 1.560 | 7.100 |
| 8-8-9 | 3 | 9 | 1.125 | 1.000 | 1.428 | 1.000 | 1.642 | 32.200 |
| 10-10 | 2 | 10 | 1.000 | 1.000 | 1.206 | 1.000 | 1.658 | 17.500 |
| 10-10-10 | 3 | 10 | 1.000 | 1.000 | 1.460 | 1.000 | 1.833 | 69.283 |
| 5-10-10 | 3 | 10 | 2.000 | 1.000 | 1.357 | 1.000 | 1.679 | 35.500 |
| 10-10-10-10-10-10-10-10-10-10 | 10 | 10 | 1.000 | 0.862 | 1.535 | 0.861 | 1.656 | 2173.740 |
| 12-6-7 | 3 | 12 | 2.000 | 1.000 | 1.477 | 1.000 | 1.801 | 35.500 |
| 12-13 | 2 | 13 | 1.083 | 1.000 | 1.246 | 1.000 | 1.878 | 61.200 |
| 15-15 | 2 | 15 | 1.000 | 1.000 | 1.413 | 1.000 | 2.048 | 129.333 |
| 15-10-5 | 3 | 15 | 3.000 | 1.000 | 1.514 | 1.000 | 1.903 | 88.017 |
| 20-5-5 | 3 | 20 | 4.000 | 1.000 | 1.490 | 1.000 | 2.004 | 127.167 |
| 25-25-25-25 | 4 | 25 | 1.000 | 1.000 | 1.828 | 1.000 | 2.097 | 7462.686 |
| 60-25-10-5 | 4 | 60 | 12.000 | 1.000 | 1.948 | 1.000 | 2.398 | 16209.860 |

This table compares the ratios against the clique/community sizes. It is useful for checking whether larger cliques, more cliques, or more unbalanced clique sizes make Pivot or LP behave differently.

### 8.3 Balanced vs unbalanced clique/community graphs

| type | runs | ILP cost | LP/ILP complete | Pivot/ILP complete | LP/ILP new | Pivot/ILP new | avg imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 261 | 213.395 | 0.974 | 1.495 | 0.973 | 1.851 | 1.000 |
| unbalanced | 260 | 120.904 | 0.998 | 1.502 | 1.000 | 1.892 | 4.494 |

This table checks whether balanced clique/community structures behave differently from unbalanced structures.

### 8.4 Correlation checks for clique size

| comparison | correlation |
| --- | --- |
| largest clique fraction vs Pivot/ILP | 0.001 |
| imbalance ratio vs Pivot/ILP | 0.449 |
| largest clique fraction vs LP/ILP | 0.629 |

These correlations are not a proof, but they help indicate whether clique size or imbalance is related to the observed approximation ratios.

---

## 9. Complete graph vs new graph after edge deletion

This section directly compares the original complete graph with the new graph obtained after edge deletion. This is the central comparison for the thesis.

The new graph uses the ILP with bad 4-cycle constraints when that value is available, because those constraints are part of the sparse edge-deleted formulation.

### 9.1 Overall comparison by graph family

| family | runs | complete ILP | new ILP | new/complete ILP | cost reduction | complete LP/ILP | new LP/ILP | LP change | complete Pivot/ILP | new Pivot/ILP | Pivot change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2100 | 45.170 | 31.743 | 0.631 | 36.910% | 0.891 | 0.917 | 0.032 | 1.211 | 1.253 | 0.031 |
| clique | 521 | 167.238 | 141.676 | 0.847 | 15.343% | 0.986 | 0.987 | 0.001 | 1.499 | 1.872 | 0.373 |
| facebook | 4 | 323.000 | 273.000 | 0.860 | 14.022% | 0.937 | 0.935 | -0.002 | 1.258 | 1.439 | 0.181 |

This table shows whether edge deletion mainly lowers the objective cost, changes LP tightness, or makes Pivot relatively worse.

### 9.2 Random graphs: complete vs new by graph size

| n | runs | complete ILP | new ILP | new/complete ILP | cost reduction | complete LP/ILP | new LP/ILP | complete Pivot/ILP | new Pivot/ILP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 350 | 1.509 | 0.851 | 0.532 | 46.773% | 0.988 | 0.995 | 1.111 | 1.171 |
| 10 | 350 | 9.120 | 6.020 | 0.609 | 39.122% | 0.960 | 0.980 | 1.141 | 1.195 |
| 15 | 350 | 24.057 | 16.700 | 0.642 | 35.768% | 0.902 | 0.940 | 1.211 | 1.236 |
| 20 | 350 | 46.186 | 32.529 | 0.658 | 34.245% | 0.865 | 0.907 | 1.248 | 1.270 |
| 25 | 350 | 76.380 | 53.866 | 0.662 | 33.784% | 0.833 | 0.873 | 1.275 | 1.296 |
| 30 | 350 | 113.766 | 80.494 | 0.667 | 33.262% | 0.812 | 0.847 | 1.267 | 1.314 |

This table checks whether the effect of edge deletion changes as the random graph becomes larger.

### 9.3 Random graphs: complete vs new by positive-edge probability

| p+ | runs | complete ILP | new ILP | new/complete ILP | cost reduction | complete LP/ILP | new LP/ILP | complete Pivot/ILP | new Pivot/ILP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 300 | 23.323 | 3.770 | 0.149 | 85.074% | 0.829 | 0.961 | 1.189 | 1.111 |
| 0.3 | 300 | 36.437 | 16.213 | 0.407 | 59.278% | 0.816 | 0.880 | 1.210 | 1.193 |
| 0.4 | 300 | 48.390 | 28.787 | 0.561 | 43.905% | 0.823 | 0.854 | 1.203 | 1.207 |
| 0.5 | 300 | 57.977 | 40.150 | 0.649 | 35.131% | 0.852 | 0.858 | 1.176 | 1.214 |
| 0.6 | 300 | 62.060 | 48.837 | 0.745 | 25.479% | 0.913 | 0.888 | 1.175 | 1.220 |
| 0.7 | 300 | 52.317 | 49.020 | 0.891 | 10.868% | 0.994 | 0.979 | 1.234 | 1.266 |
| 0.8 | 300 | 35.683 | 35.427 | 0.963 | 3.678% | 1.000 | 1.000 | 1.292 | 1.507 |

This table checks whether edge deletion behaves differently when the graph contains mostly negative, mixed, or mostly positive edges.

### 9.4 Clique/community graphs: complete vs new by structure

| n | clique sizes | imbalance | complete ILP | new ILP | new/complete ILP | complete LP/ILP | new LP/ILP | complete Pivot/ILP | new Pivot/ILP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 3-3-4 | 1.333 | 3.900 | 3.400 | 0.871 | 0.986 | 1.000 | 1.020 | 1.200 |
| 10 | 3-2-5 | 2.500 | 3.300 | 2.900 | 0.867 | 0.988 | 1.000 | 1.100 | 1.200 |
| 10 | 5-5 | 1.000 | 3.200 | 3.000 | 0.955 | 1.000 | 1.000 | 1.080 | 1.367 |
| 15 | 5-2-5-3 | 2.500 | 10.300 | 8.900 | 0.879 | 0.987 | 0.990 | 1.193 | 1.353 |
| 15 | 5-5-5 | 1.000 | 9.400 | 8.200 | 0.872 | 1.000 | 1.000 | 1.234 | 1.598 |
| 15 | 7-8 | 1.143 | 11.100 | 9.800 | 0.890 | 1.000 | 1.000 | 1.195 | 1.560 |
| 20 | 5-5-5-5 | 1.000 | 17.500 | 14.600 | 0.839 | 1.000 | 1.000 | 1.274 | 1.443 |
| 20 | 7-6-7 | 1.167 | 16.500 | 13.500 | 0.825 | 1.000 | 1.000 | 1.290 | 1.460 |
| 20 | 10-10 | 1.000 | 17.800 | 15.300 | 0.867 | 1.000 | 1.000 | 1.206 | 1.658 |
| 25 | 8-8-9 | 1.125 | 28.700 | 24.500 | 0.853 | 1.000 | 1.000 | 1.428 | 1.642 |
| 25 | 5-10-10 | 2.000 | 26.600 | 22.700 | 0.856 | 1.000 | 1.000 | 1.357 | 1.679 |
| 25 | 12-6-7 | 2.000 | 27.900 | 23.800 | 0.853 | 1.000 | 1.000 | 1.477 | 1.801 |
| 25 | 12-13 | 1.083 | 27.800 | 23.900 | 0.859 | 1.000 | 1.000 | 1.246 | 1.878 |
| 30 | 10-10-10 | 1.000 | 41.550 | 34.550 | 0.832 | 1.000 | 1.000 | 1.460 | 1.833 |
| 30 | 15-10-5 | 3.000 | 41.883 | 34.800 | 0.832 | 1.000 | 1.000 | 1.514 | 1.903 |
| 30 | 15-15 | 1.000 | 42.117 | 35.000 | 0.832 | 1.000 | 1.000 | 1.413 | 2.048 |
| 30 | 20-5-5 | 4.000 | 42.750 | 35.817 | 0.838 | 1.000 | 1.000 | 1.490 | 2.004 |
| 100 | 10-10-10-10-10-10-10-10-10-10 | 1.000 | 497.220 | 422.260 | 0.849 | 0.862 | 0.861 | 1.535 | 1.656 |
| 100 | 25-25-25-25 | 1.000 | 496.784 | 421.765 | 0.849 | 1.000 | 1.000 | 1.828 | 2.097 |
| 100 | 60-25-10-5 | 12.000 | 495.920 | 420.700 | 0.848 | 1.000 | 1.000 | 1.948 | 2.398 |

This table directly checks whether clique size and imbalance affect the difference between the complete graph and the new graph after edge deletion.

### 9.5 Facebook ego-networks: complete vs new

| ego | n | complete ILP | new ILP | new/complete ILP | cost reduction | complete LP/ILP | new LP/ILP | complete Pivot/ILP | new Pivot/ILP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3980 | 58 | 75.000 | 67.000 | 0.893 | 10.667% | 0.913 | 0.903 | 1.173 | 1.388 |
| 698 | 64 | 97.000 | 82.000 | 0.845 | 15.464% | 0.969 | 0.963 | 1.227 | 1.451 |
| 414 | 155 | 797.000 | 670.000 | 0.841 | 15.935% | 0.927 | 0.938 | 1.374 | 1.479 |
| 686 | 170 | - | - | - | -% | - | - | - | - |

This table checks whether the same complete-vs-new pattern also appears in real Facebook ego-networks.
