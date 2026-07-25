from scipy.stats import wilcoxon
import pandas as pd

df = pd.read_csv(
    "results/research_tables/minmax_facebook_grid_runs_flat.csv"
)

for (ego_id, p_delete), group in df.groupby(
    ["ego_id", "p_delete"]
):
    paired = group[
        [
            "edge_min_max_cc_max_disagreement",
            "edge_min_max_lp_rounding_cost",
        ]
    ].dropna()

    if len(paired) < 2:
        continue

    difference = (
        paired["edge_min_max_cc_max_disagreement"]
        - paired["edge_min_max_lp_rounding_cost"]
    )

    statistic, p_value = wilcoxon(difference)

    print(
        ego_id,
        p_delete,
        "runs =", len(paired),
        "median difference =", difference.median(),
        "p-value =", p_value,
    )