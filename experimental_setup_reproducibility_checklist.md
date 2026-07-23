# Experimental Setup and Reproducibility Checklist

## Experimental setup

### Datasets

**Facebook ego networks**

- Ego IDs: `0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980`
- Vertex set: sorted unique endpoints in each `.edges` file
- Circle-only vertices excluded
- Friendship edge: `+1`
- Non-edge: `-1`
- Deleted edge: `0`

**Synthetic clique graphs**

- Planted clique structure
- `p_pos_inside = 0.9`
- `p_pos_between = 0.1`
- Balanced and unbalanced cluster configurations
- Graph-generation seeds stored

### Edge deletion

- `p_delete = 0.05, 0.15, 0.25, 0.4`
- Deletion seeds: `1-30`
- Complete graph: `p_delete = 0`

### Ordinary correlation clustering

**Pivot**

- NumPy `default_rng`
- Sorted active vertices
- Pivot seeds: `1-100`
- Report best and average cost
- Large Facebook ego graphs: complete graph only

**All-pairs LP**

- Gurobi metric LP relaxation
- Facebook ego IDs: `414, 686, 698, 3980`
- Complete and edge-deleted instances
- Approximation ratio: `Pivot cost / LP lower bound`

### MinMaxCC

- `d_hat = 8`
- `lambda = 5`
- Complete and edge-deleted Facebook graphs
- Report maximum disagreement, best/average/worst ratios, and runtime

### MinMaxLP and rounding

- Ego IDs: `414, 686, 698, 3980`
- `r = 0.4`
- `r2 = 0.4`
- Infinity norm
- Gurobi Method `2`
- Crossover `0`
- Store LP, rounding, and total runtime

### Repetitions

- 4 deletion probabilities
- 30 deletion seeds
- 100 Pivot seeds per result instance
- Runtime benchmark: 30 Pivot runs and 1-3 LP repetitions

### Evaluation metrics

- ordinary disagreement cost
- maximum vertex disagreement
- LP lower bound
- approximation ratio
- best, average, and worst result
- runtime

### Reproducibility mechanisms

- fixed preprocessing
- sorted node order
- fixed seeds
- deterministic Pivot
- atomic CSV checkpoints
- progress JSON files
- manifests with parameters and hashes
- matrix hashes
- tables and figures generated from CSV results
- independent reproducibility check for ego `3980`

## Checklist: dataset usage

- 3.1: `yes`
- 3.2: `yes`
- 3.3: `NA`
- 3.4: `NA`
- 3.5: `yes`
- 3.6: `yes`
- 3.7: `NA`

## Checklist: computational experiments

- 4.1: `yes`
- 4.2: `partial`
- 4.3: `partial`
- 4.4: `partial`
- 4.5: `partial`
- 4.6: `partial`
- 4.7: `yes`
- 4.8: `partial`
- 4.9: `partial`
- 4.10: `yes`
- 4.11: `yes`
- 4.12: `no`
- 4.13: `partial`
