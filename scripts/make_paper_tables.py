import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path(
    "results/processed/research_tables/"
    "minmax_facebook_grid_runs_flat.csv"
)

MINMAX_OUTPUT = Path(
    "results/processed/research_tables/"
    "facebook_minmax_table.csv"
)

CC_OUTPUT = Path(
    "results/processed/research_tables/"
    "facebook_correlation_clustering_table.csv"
)

def to_float(value):
    """Convert a CSV value to float, or return None."""
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def get_seed(row):
    """
    Return the stored deletion seed.

    Older input files without a seed column are treated as seed 1.
    """
    seed = str(row.get("seed", "")).strip()
    return seed if seed else "1"


def read_rows(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def average(values):
    numeric_values = [
        number
        for number in (
            to_float(value)
            for value in values
        )
        if number is not None
    ]

    if not numeric_values:
        return None

    return sum(numeric_values) / len(numeric_values)


def ratio(value, bound):
    value = to_float(value)
    bound = to_float(bound)

    if value is None or bound is None or bound == 0:
        return None

    return value / bound


def ratio_distance_from_one(value, bound):
    current_ratio = ratio(value, bound)

    if current_ratio is None:
        return None

    return abs(current_ratio - 1.0)


def numeric_or_infinity(value):
    number = to_float(value)

    if number is None:
        return float("inf")

    return number


def is_better_candidate(current_best, candidate):
    """
    Select the MinMaxCC result whose ratio to its corresponding
    MinMaxLP value is closest to 1.

    Ties are resolved deterministically using:
      1. lower ratio distance from 1;
      2. lower MinMaxCC value;
      3. lower seed;
      4. lower d_hat;
      5. lower lambda.
    """
    if candidate["_ratio_distance"] is None:
        return False

    if current_best is None:
        return True

    candidate_key = (
        candidate["_ratio_distance"],
        numeric_or_infinity(candidate["minmaxcc_best"]),
        numeric_or_infinity(candidate["_seed"]),
        numeric_or_infinity(candidate["d_hat"]),
        numeric_or_infinity(candidate["lambda"]),
    )

    current_key = (
        current_best["_ratio_distance"],
        numeric_or_infinity(current_best["minmaxcc_best"]),
        numeric_or_infinity(current_best["_seed"]),
        numeric_or_infinity(current_best["d_hat"]),
        numeric_or_infinity(current_best["lambda"]),
    )

    return candidate_key < current_key


def make_minmax_table(rows):
    """
    Create 20 aggregated Facebook MinMax rows:

      - 4 complete-graph rows:
            one per ego_id

      - 16 edge-deleted rows:
            one per ego_id and p_delete

    For every edge-deleted group:

      minmaxcc_best
          The MinMaxCC value whose MinMaxCC / MinMaxLP ratio
          is closest to 1 across all seeds and all tested
          d_hat/lambda combinations.

      min_max_lp_cost_best
          The MinMaxLP value belonging to the same seed and
          instance as the selected minmaxcc_best value.

      minmaxcc_average
          Average MinMaxCC value across all seeds and all
          d_hat/lambda combinations.

      min_max_lp_cost_average
          Average MinMaxLP value across seeds. Each seed is
          counted once because its LP value is repeated for
          every d_hat/lambda row in the grid.

    The seed belonging to the best result is used internally
    but is not written to the output table.
    """

    best_candidates = {}

    # All unique MinMaxCC observations:
    # group_key -> observation_key -> value
    cc_values = defaultdict(dict)

    # One MinMaxLP value per seed:
    # group_key -> seed -> value
    lp_values_by_seed = defaultdict(dict)

    for row in rows:
        ego_id = str(row.get("ego_id", "")).strip()

        if not ego_id:
            continue

        n = str(row.get("n", "")).strip()
        seed = get_seed(row)
        p_delete = str(row.get("p_delete", "")).strip()

        # ============================================================
        # Complete graph
        # ============================================================

        complete_key = (
            ego_id,
            "0",
            "complete",
        )

        complete_cc = row.get(
            "complete_min_max_cc_max_disagreement",
            "",
        )

        complete_lp = row.get(
            "complete_min_max_lp_cost",
            "",
        )

        complete_d_hat = row.get(
            "complete_min_max_cc_d_hat",
            "",
        )

        complete_lambda = row.get(
            "complete_min_max_cc_lambda",
            "",
        )

        # The complete graph is copied into many seed/p_delete rows.
        # Deduplicate it using its d_hat/lambda combination.
        complete_cc_observation_key = (
            str(complete_d_hat).strip(),
            str(complete_lambda).strip(),
        )

        if to_float(complete_cc) is not None:
            cc_values[complete_key].setdefault(
                complete_cc_observation_key,
                complete_cc,
            )

        # There is only one complete-graph LP instance per ego_id.
        if to_float(complete_lp) is not None:
            lp_values_by_seed[complete_key].setdefault(
                "complete",
                complete_lp,
            )

        complete_ratio = ratio(
            complete_cc,
            complete_lp,
        )

        complete_candidate = {
            "ego_id": ego_id,
            "n": n,
            "p_delete": "0",
            "graph_variant": "complete",
            "minmaxcc_best": complete_cc,
            "minmaxcc_average": "",
            "min_max_lp_cost_best": complete_lp,
            "min_max_lp_cost_average": "",
            "min_max_cc_to_lp_ratio": (
                ""
                if complete_ratio is None
                else round(complete_ratio, 6)
            ),
            "d_hat": complete_d_hat,
            "lambda": complete_lambda,
            "min_max_lp_rounding_cost": row.get(
                "complete_min_max_lp_rounding_cost",
                "",
            ),
            "min_max_lp_runtime_seconds": row.get(
                "complete_min_max_lp_runtime_seconds",
                "",
            ),
            "_ratio_distance": ratio_distance_from_one(
                complete_cc,
                complete_lp,
            ),
            "_seed": "0",
        }

        if is_better_candidate(
            best_candidates.get(complete_key),
            complete_candidate,
        ):
            best_candidates[complete_key] = complete_candidate

        # ============================================================
        # Edge-deleted graph
        # ============================================================

        edge_key = (
            ego_id,
            p_delete,
            "edge_deleted",
        )

        edge_cc = row.get(
            "edge_min_max_cc_max_disagreement",
            "",
        )

        edge_lp = row.get(
            "edge_min_max_lp_cost",
            "",
        )

        edge_d_hat = row.get(
            "edge_min_max_cc_d_hat",
            "",
        )

        edge_lambda = row.get(
            "edge_min_max_cc_lambda",
            "",
        )

        # One CC observation per seed, d_hat and lambda.
        edge_cc_observation_key = (
            seed,
            str(edge_d_hat).strip(),
            str(edge_lambda).strip(),
        )

        if to_float(edge_cc) is not None:
            cc_values[edge_key].setdefault(
                edge_cc_observation_key,
                edge_cc,
            )

        # The same edge LP is repeated for every d_hat/lambda
        # combination. Count it only once per seed.
        if to_float(edge_lp) is not None:
            existing_lp = lp_values_by_seed[edge_key].get(seed)

            if existing_lp is None:
                lp_values_by_seed[edge_key][seed] = edge_lp
            else:
                existing_number = to_float(existing_lp)
                new_number = to_float(edge_lp)

                if (
                    existing_number is not None
                    and new_number is not None
                    and abs(existing_number - new_number) > 1e-8
                ):
                    raise ValueError(
                        "Conflicting edge MinMaxLP values for "
                        f"ego_id={ego_id}, p_delete={p_delete}, "
                        f"seed={seed}: {existing_lp} versus {edge_lp}"
                    )

        edge_ratio = ratio(
            edge_cc,
            edge_lp,
        )

        edge_candidate = {
            "ego_id": ego_id,
            "n": n,
            "p_delete": p_delete,
            "graph_variant": "edge_deleted",
            "minmaxcc_best": edge_cc,
            "minmaxcc_average": "",
            "min_max_lp_cost_best": edge_lp,
            "min_max_lp_cost_average": "",
            "min_max_cc_to_lp_ratio": (
                ""
                if edge_ratio is None
                else round(edge_ratio, 6)
            ),
            "d_hat": edge_d_hat,
            "lambda": edge_lambda,
            "min_max_lp_rounding_cost": row.get(
                "edge_min_max_lp_rounding_cost",
                "",
            ),
            "min_max_lp_runtime_seconds": row.get(
                "edge_min_max_lp_runtime_seconds",
                "",
            ),
            "_ratio_distance": ratio_distance_from_one(
                edge_cc,
                edge_lp,
            ),
            "_seed": seed,
        }

        if is_better_candidate(
            best_candidates.get(edge_key),
            edge_candidate,
        ):
            best_candidates[edge_key] = edge_candidate

    output_rows = []

    for group_key in sorted(
        best_candidates,
        key=lambda item: (
            int(item[0]),
            0 if item[2] == "complete" else 1,
            float(item[1]),
        ),
    ):
        output_row = dict(
            best_candidates[group_key]
        )

        average_cc = average(
            cc_values[group_key].values()
        )

        average_lp = average(
            lp_values_by_seed[group_key].values()
        )

        output_row["minmaxcc_average"] = (
            ""
            if average_cc is None
            else round(average_cc, 6)
        )

        output_row["min_max_lp_cost_average"] = (
            ""
            if average_lp is None
            else round(average_lp, 6)
        )

        output_row.pop("_ratio_distance", None)
        output_row.pop("_seed", None)

        output_rows.append(output_row)

    return output_rows


def make_correlation_clustering_table(rows):
    """
    Create 20 correlation-clustering rows:

      - one complete row per ego_id;
      - one edge-deleted row per ego_id and p_delete.

    Edge-deleted Pivot and all-pairs LP values are averaged over
    the seeds. Grid repetitions over d_hat/lambda are deduplicated.
    """

    complete_values = {}

    edge_values = defaultdict(
        lambda: {
            "n": "",
            "pivot_by_seed": {},
            "lp_by_seed": {},
        }
    )

    for row in rows:
        ego_id = str(row.get("ego_id", "")).strip()

        if not ego_id:
            continue

        n = str(row.get("n", "")).strip()
        seed = get_seed(row)
        p_delete = str(row.get("p_delete", "")).strip()

        if ego_id not in complete_values:
            complete_values[ego_id] = {
                "ego_id": ego_id,
                "n": n,
                "p_delete": "0",
                "graph_variant": "complete",
                "pivot_best_cost_average": row.get(
                    "complete_pivot_best_cost",
                    "",
                ),
                "lp_cost_average": row.get(
                    "complete_lp_cost",
                    "",
                ),
            }

        edge_key = (
            ego_id,
            p_delete,
        )

        group = edge_values[edge_key]
        group["n"] = n

        edge_pivot = row.get(
            "edge_pivot_best_cost",
            "",
        )

        edge_lp = row.get(
            "edge_all_pairs_lp_cost",
            "",
        )

        if to_float(edge_pivot) is not None:
            group["pivot_by_seed"].setdefault(
                seed,
                edge_pivot,
            )

        if to_float(edge_lp) is not None:
            group["lp_by_seed"].setdefault(
                seed,
                edge_lp,
            )

    output_rows = list(
        complete_values.values()
    )

    for (
        ego_id,
        p_delete,
    ), group in edge_values.items():
        pivot_average = average(
            group["pivot_by_seed"].values()
        )

        lp_average = average(
            group["lp_by_seed"].values()
        )

        output_rows.append({
            "ego_id": ego_id,
            "n": group["n"],
            "p_delete": p_delete,
            "graph_variant": "edge_deleted",
            "pivot_best_cost_average": (
                ""
                if pivot_average is None
                else round(pivot_average, 6)
            ),
            "lp_cost_average": (
                ""
                if lp_average is None
                else round(lp_average, 6)
            ),
        })

    output_rows.sort(
        key=lambda row: (
            int(row["ego_id"]),
            0 if row["graph_variant"] == "complete" else 1,
            float(row["p_delete"]),
        )
    )

    return output_rows


def validate_output(minmax_rows, correlation_rows):
    expected_ego_ids = {
        "414",
        "686",
        "698",
        "3980",
    }

    expected_p_delete_values = {
        "0.05",
        "0.15",
        "0.25",
        "0.4",
    }

    expected_total_rows = (
        len(expected_ego_ids)
        + len(expected_ego_ids) * len(expected_p_delete_values)
    )

    if len(minmax_rows) != expected_total_rows:
        raise ValueError(
            f"Expected {expected_total_rows} MinMax rows, "
            f"but generated {len(minmax_rows)}."
        )

    if len(correlation_rows) != expected_total_rows:
        raise ValueError(
            f"Expected {expected_total_rows} correlation-clustering rows, "
            f"but generated {len(correlation_rows)}."
        )

    for table_name, table_rows in [
        ("MinMax", minmax_rows),
        ("correlation clustering", correlation_rows),
    ]:
        complete_rows = [
            row
            for row in table_rows
            if row["graph_variant"] == "complete"
        ]

        edge_rows = [
            row
            for row in table_rows
            if row["graph_variant"] == "edge_deleted"
        ]

        complete_ego_ids = {
            row["ego_id"]
            for row in complete_rows
        }

        if complete_ego_ids != expected_ego_ids:
            raise ValueError(
                f"{table_name} complete ego IDs are incomplete: "
                f"{sorted(complete_ego_ids)}"
            )

        found_edge_combinations = {
            (
                row["ego_id"],
                row["p_delete"],
            )
            for row in edge_rows
        }

        expected_edge_combinations = {
            (
                ego_id,
                p_delete,
            )
            for ego_id in expected_ego_ids
            for p_delete in expected_p_delete_values
        }

        if found_edge_combinations != expected_edge_combinations:
            missing = (
                expected_edge_combinations
                - found_edge_combinations
            )

            extra = (
                found_edge_combinations
                - expected_edge_combinations
            )

            raise ValueError(
                f"{table_name} edge combinations are incomplete. "
                f"Missing: {sorted(missing)}. "
                f"Extra: {sorted(extra)}."
            )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    rows = read_rows(INPUT_FILE)

    print("Input:", INPUT_FILE)
    print("Read grid rows:", len(rows))

    minmax_rows = make_minmax_table(rows)

    correlation_rows = (
        make_correlation_clustering_table(rows)
    )

    validate_output(
        minmax_rows,
        correlation_rows,
    )

    minmax_fieldnames = [
        "ego_id",
        "n",
        "p_delete",
        "graph_variant",
        "minmaxcc_best",
        "minmaxcc_average",
        "min_max_lp_cost_best",
        "min_max_lp_cost_average",
        "min_max_cc_to_lp_ratio",
        "d_hat",
        "lambda",
        "min_max_lp_rounding_cost",
        "min_max_lp_runtime_seconds",
    ]

    correlation_fieldnames = [
        "ego_id",
        "n",
        "p_delete",
        "graph_variant",
        "pivot_best_cost_average",
        "lp_cost_average",
    ]

    write_rows(
        MINMAX_OUTPUT,
        minmax_rows,
        minmax_fieldnames,
    )

    write_rows(
        CC_OUTPUT,
        correlation_rows,
        correlation_fieldnames,
    )

    print()
    print("Wrote MinMax table:")
    print(MINMAX_OUTPUT)
    print("Rows:", len(minmax_rows))

    print()
    print("Wrote correlation-clustering table:")
    print(CC_OUTPUT)
    print("Rows:", len(correlation_rows))


if __name__ == "__main__":
    main()