from pathlib import Path

text = r"""# 4. Computational Experiments — Copy-ready checklist and paper text

This document covers **only Section 4: Computational Experiments**.  
For each checklist question, it gives:

1. the checklist answer;
2. the exact text that can be copied into the paper or supplementary material;


---

## 4.1 Does this paper include computational experiments?

**Checklist answer: YES**

---

## 4.2 Does the paper state the number and range of values tried for every hyperparameter and the criterion used to select the final setting?

**Checklist answer: NO**

---

## 4.3 Is all preprocessing code included?

**Checklist answer: YES**

I saw this in the paper already
---

## 4.4 Is all source code needed to conduct and analyse the experiments included?

**Checklist answer: YES**

---

## 4.5 Will all experimental code be made publicly available with a research-use license?

**Checklist answer: YES only when the public repository contains an explicit license**
I will do this later.

---

## 4.6 Does the implementation of the new method contain comments and references to the corresponding paper steps?

**Checklist answer: NO**

The MinMaxCC implementation must contain comments identifying the corresponding algorithm step, equation, theorem, or pseudocode line for this to be `yes` .
---

## 4.7 Are random seeds described sufficiently to reproduce the results?

**Checklist answer: YES if there is something like this in the paper:**

### Copy into the paper

> Random edge deletion uses seeds \(1,\ldots,30\). Thus, every graph and deletion-probability pair is evaluated on exactly 30 independently generated deleted instances. MinMaxCC is deterministic for a fixed deleted graph and fixed parameter pair and is therefore run once on each generated instance. For the ordinary Pivot baseline, seeds \(1,\ldots,100\) are used on every reported graph instance; both the minimum and arithmetic mean of the 100 Pivot costs are recorded. Comparisons between MinMaxCC and LP rounding are paired by graph identifier, deletion probability, and deletion seed, so both methods are evaluated on the same deleted graph.

### Exact run counts

- Ten Facebook ego graphs are used for MinMaxCC.
- Four deletion probabilities are used.
- Thirty deletion seeds are used per graph and deletion probability.
- Therefore, MinMaxCC is evaluated on
  \[
  10\times4\times30=1200
  \]
  edge-deleted Facebook instances.
- The ten complete Facebook graphs are also evaluated, giving \(1210\) complete-plus-deleted MinMaxCC graph instances.
- Local MinMaxLP is evaluated on four ego graphs: \(414,686,698,\) and \(3980\).
- For local MinMaxLP, the edge-deleted experiment contains
  \[
  4\times4\times30=480
  \]
  LP solves.
- Four additional complete-graph LP solves are performed, giving \(484\) local MinMaxLP solves in total.
- Pivot uses 100 seeds per reported graph instance.

---

## 4.8 Is the computing infrastructure specified?

**Checklist answer: NO**


### Add something like this

>The experiments were conducted on a MacBook Air with an Apple M2 processor, comprising 8 CPU cores (4 performance and 4 efficiency cores), 8 GiB of memory, and the ARM64 architecture. The machine ran macOS 13.3 (22E252). The experiments used Python 3.11.8, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, NetworkX 3.6.1, Matplotlib 3.11.0, Gurobi Optimizer 13.0.2, and gurobipy 13.0.2.  The reported MinMaxLP runs used Gurobi Method \(2\) with Crossover \(0\), and Gurobi was permitted to use up to 8 threads
The checklist should remain **PARTIAL** until the bracketed values are replaced with the recorded exact values.

---

## 4.9 Are all evaluation metrics formally defined and motivated?

**Checklist answer: YES after the following text is included**

### Copy into the paper

> For a clustering \(\mathcal C\), the ordinary correlation-clustering cost is the number of positive pairs placed in different clusters plus the number of negative pairs placed in the same cluster. For MinMax correlation clustering, the disagreement of a vertex is the number of incident positive pairs cut by \(\mathcal C\) plus the number of incident negative pairs retained inside its cluster. The MinMaxCC objective is the maximum disagreement over all vertices,
> \[
> \operatorname{cost}_{\max}(\mathcal C)
> =\max_{v\in V}\operatorname{disagree}(v,\mathcal C).
> \]
> This objective measures worst-case per-vertex clustering quality rather than only total clustering quality.

### Copy into the paper for approximation ratios

> Whenever a locally computed MinMaxLP value is available for the same deleted instance, the approximation ratio is
> \[
> \rho
> =
> \frac{\operatorname{cost}_{\max}(\mathcal C_{\mathrm{MinMaxCC}})}
> {\operatorname{OPT}_{\mathrm{MinMaxLP}}}.
> \]
> Smaller ratios indicate solutions closer to the LP lower bound. Ratios are calculated separately for every deleted instance before aggregation; they are not obtained by dividing two aggregated means.

### Copy into the paper for LP rounding

> The MinMaxLP objective is a lower bound and does not itself define a clustering. The LP-rounding cost is instead the maximum-disagreement cost of the feasible clustering returned by the rounding algorithm and can therefore be compared directly with the MinMaxCC clustering cost.

### Required special-case paragraph for ego ID 348

> For Facebook ego graph 348, the deleted-instance MinMaxLP problems could not be solved on the local machine. We therefore use the complete-graph MinMaxLP objective \(39.13\), as reported by Davies et al., as an external reference value. This same value is used as the denominator for every ratio reported for ego graph 348. Consequently, these values are ratios to an externally reported complete-graph LP reference and must not be interpreted as approximation ratios relative to the LP optimum of each deleted instance.

### Copy into the paper for runtimes

> MinMaxCC runtime is the wall-clock time of the MinMaxCC algorithm call. MinMaxLP runtime is the Gurobi LP-solver runtime. LP-rounding runtime is measured separately, and MinMaxLP total runtime is the sum of LP-solver and rounding runtimes. Pivot runtime measures one Pivot call, excluding graph construction and subsequent cost evaluation. Ordinary LP runtime uses Gurobi solver time when available.

---

## 4.10 Is the number of algorithm runs stated for every reported result?

**Checklist answer: YES. I think I read it but look if this is all stated**


> For every Facebook ego graph and every deletion probability in \(\{0.05,0.15,0.25,0.40\}\), we generated 30 deleted instances using seeds \(1,\ldots,30\). MinMaxCC was run once per generated instance because it is deterministic for a fixed graph and fixed parameters. Local MinMaxLP and LP rounding were run on the same 30 deleted instances for each of the four ego graphs \(414,686,698,\) and \(3980\), resulting in 480 edge-deleted LP solves and four complete-graph LP solves. Pivot costs were computed using 100 Pivot seeds per reported instance. The separate Pivot runtime benchmark used 30 timing repetitions, while the ordinary LP runtime benchmark used one LP repetition per newly benchmarked graph.

---

## 4.11 Does the analysis report variation or distributional information beyond a single mean or median?

**Checklist answer: YES**

### Copy into the paper

> For every graph and deletion-probability pair, the analysis uses \(N=30\) independently generated deleted instances. We report the arithmetic mean approximation ratio, the sample standard deviation,
> \[
> s=\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(r_i-\bar r)^2},
> \]
> and a two-sided 95% Student-\(t\) confidence interval for the mean,
> \[
> \bar r\pm t_{0.975,N-1}\frac{s}{\sqrt N}.
> \]
> The reported confidence-interval limits describe uncertainty in the estimated mean; they are not intended to contain 95% of the individual observations. For descriptive plots, averages are computed across the selected deletion probabilities and seeds. Where minimum and maximum values are shown, they represent the complete observed range rather than a confidence interval.

### Exact table description

> The statistical summary contains, for every graph and deletion probability, the number of successful runs (`runs`), the mean ratio (`mean`), the sample standard deviation (`std`), the lower 95% confidence limit (`ci95_low`), and the upper 95% confidence limit (`ci95_high`).

---

## 4.12 Are performance differences evaluated using an appropriate statistical test?

**Checklist answer: YES if the LP-rounding comparison is included in the paper or supplement**

### Copy into the paper

> We compare the MinMaxCC clustering cost with the LP-rounding clustering cost using paired, two-sided Wilcoxon signed-rank tests. Each pair consists of results obtained on the same Facebook ego graph, deletion probability, and deletion seed. We conduct 16 tests, corresponding to four ego graphs and four deletion probabilities. To control the family-wise error rate, we apply a Bonferroni correction, resulting in a corrected significance threshold of
> \[
> \alpha_{\mathrm{corrected}}=\frac{0.05}{16}=0.003125.
> \]
> All 16 observed \(p\)-values are smaller than \(0.003125\), so every tested difference remains statistically significant after correction. We define the paired difference as
> \[
> \operatorname{cost}_{\mathrm{MinMaxCC}}
> -
> \operatorname{cost}_{\mathrm{LP\ rounding}}.
> \]
> A positive value therefore means that LP rounding has the lower cost, while a negative value means that MinMaxCC has the lower cost. LP rounding has a lower median cost in 13 of the 16 settings. MinMaxCC has a lower median cost for ego graph 686 at deletion probabilities \(0.05\), \(0.15\), and \(0.25\).

### Exact statistical results

| Ego ID | \(p_{\mathrm{delete}}\) | Paired runs | Median difference | \(p\)-value |
|---:|---:|---:|---:|---:|
| 414 | 0.05 | 30 | 3.0 | \(1.0261173547595263\times10^{-4}\) |
| 414 | 0.15 | 30 | 2.5 | \(7.717298875479337\times10^{-6}\) |
| 414 | 0.25 | 30 | 4.0 | \(7.983183771979704\times10^{-6}\) |
| 414 | 0.40 | 30 | 13.0 | \(1.6447303239552733\times10^{-6}\) |
| 686 | 0.05 | 30 | -9.0 | \(3.428548539642235\times10^{-6}\) |
| 686 | 0.15 | 30 | -9.5 | \(1.6752383396794876\times10^{-6}\) |
| 686 | 0.25 | 30 | -8.0 | \(1.6762987141123663\times10^{-6}\) |
| 686 | 0.40 | 30 | 4.0 | \(3.100953776514089\times10^{-5}\) |
| 698 | 0.05 | 30 | 10.0 | \(1.5470299066026792\times10^{-6}\) |
| 698 | 0.15 | 30 | 10.0 | \(1.5410884614417123\times10^{-6}\) |
| 698 | 0.25 | 30 | 10.0 | \(1.557972322784545\times10^{-6}\) |
| 698 | 0.40 | 30 | 7.0 | \(1.492314604383744\times10^{-6}\) |
| 3980 | 0.05 | 30 | 4.5 | \(1.4169661684349726\times10^{-6}\) |
| 3980 | 0.15 | 30 | 5.0 | \(1.5155622868388145\times10^{-6}\) |
| 3980 | 0.25 | 30 | 4.0 | \(1.3342807477653781\times10^{-6}\) |
| 3980 | 0.40 | 30 | 4.0 | \(1.5420773746302333\times10^{-6}\) |

### Important limitation

Do not use this Wilcoxon test to compare MinMaxCC directly with the raw MinMaxLP objective. The LP objective is a lower bound rather than a feasible clustering produced by a competing algorithm.

When LP rounding is completely omitted from the paper and supplementary material, this checklist answer should not claim that the above comparison is part of the paper.

---

## 4.13 Are all final hyperparameters listed?

**Checklist answer: YES after the following table or equivalent paragraph is included**

### Copy into the paper

| Component | Parameter | Final value |
|---|---|---:|
| Facebook deletion | \(p_{\mathrm{delete}}\) | \(0.05,0.15,0.25,0.40\) |
| Facebook deletion | deletion seeds | \(1,\ldots,30\) |
| MinMaxCC | \(\hat d\) | \(8\) |
| MinMaxCC | \(\lambda\) | \(5\) |
| MinMaxLP | \(r\) | \(0.4\) |
| MinMaxLP | \(r_2\) | \(0.4\) |
| MinMaxLP | norm | infinity |
| Gurobi | Method | \(2\) |
| Gurobi | Crossover | \(0\) |
| Pivot cost experiments | Pivot seeds | \(1,\ldots,100\) |
| Pivot runtime benchmark | repetitions | \(30\) |
| Ordinary LP runtime benchmark | repetitions | \(1\) |
| Statistical summaries | runs per graph/deletion pair | \(30\) |
| Significance testing | test | paired two-sided Wilcoxon signed-rank |
| Significance testing | number of tests | \(16\) |
| Significance testing | corrected threshold | \(0.003125\) |

### Copy below the table

> The local MinMaxLP and LP-rounding experiments were conducted for ego IDs \(414\), \(686\), \(698\), and \(3980\), containing \(150\), \(168\), \(61\), and \(52\) vertices, respectively, after corrected preprocessing. MinMaxCC was evaluated on all ten Facebook ego graphs. For ego ID 348, the externally reported complete-graph LP value \(39.13\) was used as the reference denominator for all reported ratios because the deleted-graph LP instances could not be solved locally.

---

# Final checklist answers for Section 4

| Question | Current answer |
|---|---|
| 4.1 Computational experiments included | **YES** |
| 4.2 Parameter ranges and selection criterion | **NO** |
| 4.3 Preprocessing code included | **YES** |
| 4.4 Conducting and analysis code included | **YES** |
| 4.5 Public code with research-use license | **PARTIAL** — change to **YES** after an explicit license is added to the public repository |
| 4.6 Code comments linked to the paper | **NO** |
| 4.7 Random seeds reproducible | **YES after the provided seed and run-count paragraph is included in the paper** |
| 4.8 Computing infrastructure | **NO in the current paper** — change to **YES** after the provided machine and software specification paragraph is included |
| 4.9 Metrics formally defined and motivated | **YES after the provided metric definitions and the ego-348 qualification are included** |
| 4.10 Number of runs stated | **YES, provided that all run counts in the supplied paragraph are already stated in the paper** |
| 4.11 Variation and confidence intervals | **YES** |
| 4.12 Appropriate significance tests | **YES only when the LP-rounding comparison and Wilcoxon analysis are included; otherwise NA** |
| 4.13 Final hyperparameters listed | **YES after the provided hyperparameter table is included** |
"""

path = Path("/mnt/data/AAAI27_section_4_copy_ready.md")
path.write_text(text, encoding="utf-8")
print(path)
