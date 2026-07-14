import csv
from pathlib import Path


INPUT_FILE = Path("results/processed/minmax_facebook_grid_runs_flat.csv")
OUTPUT_DIR = Path("results/processed/thesis_tables")

MINMAX_OUTPUT = OUTPUT_DIR / "facebook_minmax_table.csv"
CC_OUTPUT = OUTPUT_DIR / "facebook_correlation_clustering_table.csv"


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def ratio(value, bound):
    value = to_float(value)
    bound = to_float(bound)
    if value is None or bound is None or bound == 0:
        return None
    return value / bound


def ratio_distance_from_1(value, bound):
    r = ratio(value, bound)
    if r is None:
        return None
    return abs(r - 1.0)


def better_candidate(current_best, candidate, score_key):
    candidate_score = candidate.get(score_key)

    if candidate_score is None:
        return False

    if current_best is None:
        return True

    current_score = current_best.get(score_key)
    if current_score is None:
        return True

    return candidate_score < current_score


def make_minmax_table(rows):
    """
    Complete graph:
        one row per ego_id, with p_delete = 0

    Edge-deleted graph:
        one row per ego_id and p_delete

    In both cases, choose the MinMaxCC row whose ratio
    MinMaxCC / MinMaxLP is closest to 1.
    """

    best = {}

    for row in rows:
        ego_id = row.get("ego_id", "").strip()
        original_p_delete = row.get("p_delete", "").strip()
        n = row.get("n", "").strip()

        # Complete graph candidate
        # Complete graph does not depend on p_delete, so force p_delete to 0.
        complete_ratio = ratio(
            row.get("complete_min_max_cc_max_disagreement"),
            row.get("complete_min_max_lp_cost"),
        )
        complete_score = ratio_distance_from_1(
            row.get("complete_min_max_cc_max_disagreement"),
            row.get("complete_min_max_lp_cost"),
        )

        complete_candidate = {
            "ego_id": ego_id,
            "n": n,
            "p_delete": "0",
            "graph_variant": "complete",
            "min_max_cc_max_disagreement": row.get("complete_min_max_cc_max_disagreement", ""),
            "min_max_lp_cost": row.get("complete_min_max_lp_cost", ""),
            "min_max_cc_to_lp_ratio": "" if complete_ratio is None else round(complete_ratio, 4),
            "d_hat": row.get("complete_min_max_cc_d_hat", ""),
            "lambda": row.get("complete_min_max_cc_lambda", ""),
            "min_max_lp_rounding_cost": row.get("complete_min_max_lp_rounding_cost", ""),
            "min_max_lp_runtime_seconds": row.get("complete_min_max_lp_runtime_seconds", ""),
            "_score": complete_score,
        }

        complete_key = (ego_id, "0", "complete")
        if better_candidate(best.get(complete_key), complete_candidate, "_score"):
            best[complete_key] = complete_candidate

        # Edge-deleted graph candidate
        # This one does depend on p_delete.
        edge_ratio = ratio(
            row.get("edge_min_max_cc_max_disagreement"),
            row.get("edge_min_max_lp_cost"),
        )
        edge_score = ratio_distance_from_1(
            row.get("edge_min_max_cc_max_disagreement"),
            row.get("edge_min_max_lp_cost"),
        )

        edge_candidate = {
            "ego_id": ego_id,
            "n": n,
            "p_delete": original_p_delete,
            "graph_variant": "edge_deleted",
            "min_max_cc_max_disagreement": row.get("edge_min_max_cc_max_disagreement", ""),
            "min_max_lp_cost": row.get("edge_min_max_lp_cost", ""),
            "min_max_cc_to_lp_ratio": "" if edge_ratio is None else round(edge_ratio, 4),
            "d_hat": row.get("edge_min_max_cc_d_hat", ""),
            "lambda": row.get("edge_min_max_cc_lambda", ""),
            "min_max_lp_rounding_cost": row.get("edge_min_max_lp_rounding_cost", ""),
            "min_max_lp_runtime_seconds": row.get("edge_min_max_lp_runtime_seconds", ""),
            "_score": edge_score,
        }

        edge_key = (ego_id, original_p_delete, "edge_deleted")
        if better_candidate(best.get(edge_key), edge_candidate, "_score"):
            best[edge_key] = edge_candidate

    output_rows = []
    for key in sorted(best.keys(), key=lambda x: (int(x[0]), float(x[1]), x[2])):
        clean_row = dict(best[key])
        clean_row.pop("_score", None)
        output_rows.append(clean_row)

    return output_rows
def make_correlation_clustering_table(rows):
    """
    For each ego_id and p_delete, keep one deduplicated row.

    The min-max grid repeats pivot and LP values for every d_hat/lambda,
    so we only keep the first row for each ego_id and p_delete.
    """

    seen = set()
    output_rows = []

    for row in rows:
        ego_id = row.get("ego_id", "").strip()
        p_delete = row.get("p_delete", "").strip()
        key = (ego_id, p_delete)

        if key in seen:
            continue

        seen.add(key)

        output_rows.append({
            "ego_id": ego_id,
            "n": row.get("n", ""),
            "p_delete": p_delete,

            "complete_pivot_best_cost": row.get("complete_pivot_best_cost", ""),
            "complete_lp_cost": row.get("complete_lp_cost", ""),

            "edge_pivot_best_cost": row.get("edge_pivot_best_cost", ""),
            "edge_all_pairs_lp_cost": row.get("edge_all_pairs_lp_cost", ""),
        })

    output_rows.sort(key=lambda r: (int(r["ego_id"]), float(r["p_delete"])))
    return output_rows


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    rows = read_rows(INPUT_FILE)
    print("Read rows:", len(rows))
    print("Input:", INPUT_FILE)

    minmax_rows = make_minmax_table(rows)
    cc_rows = make_correlation_clustering_table(rows)

    minmax_fieldnames = [
        "ego_id",
        "n",
        "p_delete",
        "graph_variant",
        "min_max_cc_max_disagreement",
        "min_max_lp_cost",
        "min_max_cc_to_lp_ratio",
        "d_hat",
        "lambda",
        "min_max_lp_rounding_cost",
        "min_max_lp_runtime_seconds",
    ]

    cc_fieldnames = [
        "ego_id",
        "n",
        "p_delete",
        "complete_pivot_best_cost",
        "complete_lp_cost",
        "edge_pivot_best_cost",
        "edge_all_pairs_lp_cost",
    ]

    write_rows(MINMAX_OUTPUT, minmax_rows, minmax_fieldnames)
    write_rows(CC_OUTPUT, cc_rows, cc_fieldnames)

    print("Wrote min-max table:", MINMAX_OUTPUT)
    print("Rows:", len(minmax_rows))

    print("Wrote correlation clustering table:", CC_OUTPUT)
    print("Rows:", len(cc_rows))


if __name__ == "__main__":
    main()