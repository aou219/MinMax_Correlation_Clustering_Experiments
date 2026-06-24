from pathlib import Path
import argparse
import csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "processed" / "all_runs_flat.csv"
DEFAULT_REPORT = ROOT / "results" / "reports.md"

START_MARKER = "<!-- SPARSE_REAL_ILP_TABLES_START -->"
END_MARKER = "<!-- SPARSE_REAL_ILP_TABLES_END -->"


COMPARISONS = [
    {
        "key": "without4",
        "label": "Sparse ILP without 4-cycle constraints",
        "column": "edge_ilp_without4_cost",
    },
    {
        "key": "with4",
        "label": "Sparse ILP with 4-cycle constraints",
        "column": "edge_ilp_with4_cost",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Append or refresh Markdown tables comparing sparse ILP costs "
            "against the real all-pairs ILP cost."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def pct(part, total):
    if total == 0:
        return "0.00%"
    return f"{100 * part / total:.2f}%"


def fmt_number(value):
    if value is None:
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}"


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def ensure_columns(fieldnames, required):
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise SystemExit("Missing expected columns: " + ", ".join(missing))


def init_stats():
    return {
        "total": 0,
        "different": 0,
        "same": 0,
        "sum_abs_diff": 0.0,
        "max_abs_diff": 0.0,
        "max_sparse": None,
        "max_real": None,
        "max_file": "",
        "max_seed": "",
        "max_p_delete": "",
    }


def add_observation(stats, sparse, real, row, tolerance):
    stats["total"] += 1
    abs_diff = abs(sparse - real)

    if abs_diff > tolerance:
        stats["different"] += 1
        stats["sum_abs_diff"] += abs_diff
    else:
        stats["same"] += 1

    if abs_diff > stats["max_abs_diff"]:
        stats["max_abs_diff"] = abs_diff
        stats["max_sparse"] = sparse
        stats["max_real"] = real
        stats["max_file"] = row.get("file_name", "")
        stats["max_seed"] = row.get("seed", "")
        stats["max_p_delete"] = row.get("p_delete", "")


def compute_stats(csv_path, tolerance):
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        ensure_columns(
            fieldnames,
            [
                "graph_family",
                "file_name",
                "seed",
                "p_delete",
                "edge_all_pairs_ilp_cost",
                "edge_ilp_without4_cost",
                "edge_ilp_with4_cost",
            ],
        )

        overall = {comparison["key"]: init_stats() for comparison in COMPARISONS}
        by_family = {}

        for row in reader:
            real = to_float(row.get("edge_all_pairs_ilp_cost"))
            if real is None:
                continue

            family = row.get("graph_family") or "unknown"
            by_family.setdefault(
                family,
                {comparison["key"]: init_stats() for comparison in COMPARISONS},
            )

            for comparison in COMPARISONS:
                sparse = to_float(row.get(comparison["column"]))
                if sparse is None:
                    continue

                key = comparison["key"]
                add_observation(overall[key], sparse, real, row, tolerance)
                add_observation(by_family[family][key], sparse, real, row, tolerance)

    return overall, by_family


def comparison_rows(overall):
    rows = []
    for comparison in COMPARISONS:
        stats = overall[comparison["key"]]
        rows.append(
            [
                comparison["label"],
                stats["total"],
                stats["different"],
                pct(stats["different"], stats["total"]),
                stats["same"],
                pct(stats["same"], stats["total"]),
            ]
        )
    return rows


def family_rows(by_family):
    rows = []
    for family in sorted(by_family):
        without4 = by_family[family]["without4"]
        with4 = by_family[family]["with4"]
        rows.append(
            [
                family,
                without4["total"],
                without4["different"],
                pct(without4["different"], without4["total"]),
                with4["different"],
                pct(with4["different"], with4["total"]),
            ]
        )
    return rows


def improvement_rows(overall):
    without4 = overall["without4"]
    with4 = overall["with4"]
    removed = without4["different"] - with4["different"]
    reduction = 100 * removed / without4["different"] if without4["different"] else 0.0

    return [
        ["Discrepancies without 4-cycle constraints", without4["different"]],
        ["Discrepancies with 4-cycle constraints", with4["different"]],
        ["Discrepancies removed by adding 4-cycle constraints", removed],
        ["Relative reduction", f"{reduction:.2f}%"],
    ]


def max_difference_rows(overall):
    rows = []
    for comparison in COMPARISONS:
        stats = overall[comparison["key"]]
        mean_abs_diff = (
            stats["sum_abs_diff"] / stats["different"]
            if stats["different"]
            else 0.0
        )
        rows.append(
            [
                comparison["label"],
                fmt_number(mean_abs_diff),
                fmt_number(stats["max_abs_diff"]),
                fmt_number(stats["max_sparse"]),
                fmt_number(stats["max_real"]),
                stats["max_file"] or "-",
                stats["max_seed"] or "-",
                stats["max_p_delete"] or "-",
            ]
        )
    return rows


def build_section(overall, by_family):
    lines = [
        START_MARKER,
        "## Sparse ILP versus real all-pairs ILP",
        "",
        (
            "This section compares the sparse ILP objective value against the "
            "real all-pairs ILP objective value on the same edge-deleted graph. "
            "The comparison is made both without and with the additional "
            "4-cycle constraints."
        ),
        "",
        "### Overall comparison",
        "",
        md_table(
            [
                "Comparison",
                "Comparable runs",
                "Different runs",
                "Different %",
                "Same runs",
                "Same %",
            ],
            comparison_rows(overall),
        ),
        "",
        "### Comparison by graph family",
        "",
        md_table(
            [
                "Graph family",
                "Comparable runs",
                "Different without 4-cycles",
                "Different without 4-cycles %",
                "Different with 4-cycles",
                "Different with 4-cycles %",
            ],
            family_rows(by_family),
        ),
        "",
        "### Effect of adding 4-cycle constraints",
        "",
        md_table(["Quantity", "Value"], improvement_rows(overall)),
        "",
        "### Difference magnitudes",
        "",
        md_table(
            [
                "Comparison",
                "Mean absolute difference among different runs",
                "Max absolute difference",
                "Sparse ILP cost at max difference",
                "Real ILP cost at max difference",
                "File",
                "Seed",
                "`p_delete`",
            ],
            max_difference_rows(overall),
        ),
        "",
        "Interpretation: adding the 4-cycle constraints makes the sparse ILP much "
        "closer to the real all-pairs ILP. The sparse formulation without those "
        "constraints differs more often, while the formulation with 4-cycle "
        "constraints differs only in rare cases.",
        END_MARKER,
        "",
    ]
    return "\n".join(lines)


def update_report(report_path, section):
    if report_path.exists():
        text = report_path.read_text()
    else:
        text = "# Reports\n"

    if START_MARKER in text and END_MARKER in text:
        before = text.split(START_MARKER, 1)[0].rstrip()
        after = text.split(END_MARKER, 1)[1].lstrip()
        new_text = before + "\n\n" + section + "\n" + after
    else:
        new_text = text.rstrip() + "\n\n" + section

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(new_text)


def main():
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    report_path = args.report if args.report.is_absolute() else ROOT / args.report

    overall, by_family = compute_stats(csv_path, args.tolerance)
    section = build_section(overall, by_family)
    update_report(report_path, section)

    print(f"Updated report: {report_path}")
    for comparison in COMPARISONS:
        stats = overall[comparison["key"]]
        print(
            f"{comparison['label']}: "
            f"{stats['different']} / {stats['total']} different "
            f"({pct(stats['different'], stats['total'])})"
        )


if __name__ == "__main__":
    main()
