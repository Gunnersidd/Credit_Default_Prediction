#!/usr/bin/env python3
"""Stream-audit the Kaggle Home Credit Default Risk CSVs without modifying them.

The JSON output contains aggregate schema, blank/sentinel counts, target counts,
candidate-key checks, foreign-key coverage, and ranges for relative day/month
fields. It does not copy or emit data rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_FILES = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "previous_application.csv",
    "bureau_balance.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
)

UNIQUE_KEY = {
    "application_train.csv": "SK_ID_CURR",
    "application_test.csv": "SK_ID_CURR",
    "bureau.csv": "SK_ID_BUREAU",
    "previous_application.csv": "SK_ID_PREV",
}

PARENT_KEY = {
    "application_train.csv": ("application_ids", "SK_ID_CURR"),
    "application_test.csv": ("application_ids", "SK_ID_CURR"),
    "bureau.csv": ("bureau_ids", "SK_ID_BUREAU"),
    "previous_application.csv": ("previous_application_ids", "SK_ID_PREV"),
}

FOREIGN_KEYS = {
    "bureau.csv": {"SK_ID_CURR": "application_ids"},
    "bureau_balance.csv": {"SK_ID_BUREAU": "bureau_ids"},
    "previous_application.csv": {"SK_ID_CURR": "application_ids"},
    "POS_CASH_balance.csv": {
        "SK_ID_CURR": "application_ids",
        "SK_ID_PREV": "previous_application_ids",
    },
    "credit_card_balance.csv": {
        "SK_ID_CURR": "application_ids",
        "SK_ID_PREV": "previous_application_ids",
    },
    "installments_payments.csv": {
        "SK_ID_CURR": "application_ids",
        "SK_ID_PREV": "previous_application_ids",
    },
}

MISSING_MARKERS = ("XNA", "XAP", "Unknown", "NaN", "nan")


def as_integer(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_file(path: Path, parent_ids: dict[str, set[int]]) -> dict[str, Any]:
    name = path.name
    print(f"Scanning {name}", file=sys.stderr, flush=True)
    result: dict[str, Any] = {
        "file": name,
        "bytes": path.stat().st_size,
        "rows": 0,
        "columns": [],
        "duplicate_column_names": [],
        "blank_cells_by_column": {},
        "marker_cells_by_column": {},
        "key_check": None,
        "foreign_key_checks": {},
        "dictionary_check": None,
        "cross_application_id_overlap": 0,
        "target_counts": None,
        "relative_time_ranges": {},
    }

    blank_counts: Counter[str] = Counter()
    marker_counts: dict[str, Counter[str]] = {}
    time_min: dict[str, float] = {}
    time_max: dict[str, float] = {}
    foreign_stats: dict[str, Counter[str]] = {}
    target_counts: Counter[str] = Counter()
    key_values: set[int] | None = set() if name in UNIQUE_KEY else None
    duplicate_keys = 0

    encoding = "cp1252" if name == "HomeCredit_columns_description.csv" else "utf-8-sig"
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            result["error"] = "empty file"
            return result

        result["columns"] = header
        result["duplicate_column_names"] = sorted(
            col for col, count in Counter(header).items() if count > 1
        )
        index = {column: i for i, column in enumerate(header)}
        key_col = UNIQUE_KEY.get(name)
        key_idx = index.get(key_col) if key_col else None
        target_idx = index.get("TARGET")
        parent_spec = PARENT_KEY.get(name)
        parent_idx = index.get(parent_spec[1]) if parent_spec else None
        foreign_idxs = {
            column: index[column]
            for column in FOREIGN_KEYS.get(name, {})
            if column in index
        }
        for column in FOREIGN_KEYS.get(name, {}):
            if column in foreign_idxs:
                foreign_stats[column] = Counter()
        time_idxs = {
            column: i
            for i, column in enumerate(header)
            if column.startswith("DAYS_") or column == "MONTHS_BALANCE"
        }

        for row in reader:
            result["rows"] += 1
            if len(row) != len(header):
                result.setdefault("malformed_row_widths", Counter())[len(row)] += 1
            for i, value in enumerate(row[: len(header)]):
                column = header[i]
                if value == "":
                    blank_counts[column] += 1
                elif value in MISSING_MARKERS:
                    marker_counts.setdefault(column, Counter())[value] += 1
            if target_idx is not None and target_idx < len(row):
                target_counts[row[target_idx]] += 1
            if key_idx is not None and key_idx < len(row) and key_values is not None:
                key = as_integer(row[key_idx])
                if key is not None:
                    if key in key_values:
                        duplicate_keys += 1
                    else:
                        key_values.add(key)
                    if name == "application_test.csv" and key in parent_ids["application_ids"]:
                        result["cross_application_id_overlap"] += 1
            if parent_spec and parent_idx is not None and parent_idx < len(row):
                parent_value = as_integer(row[parent_idx])
                if parent_value is not None:
                    parent_ids[parent_spec[0]].add(parent_value)
            for column, i in foreign_idxs.items():
                value = row[i] if i < len(row) else ""
                stats = foreign_stats[column]
                if value == "":
                    stats["blank"] += 1
                else:
                    key = as_integer(value)
                    if key is None:
                        stats["invalid_integer"] += 1
                    elif key not in parent_ids[FOREIGN_KEYS[name][column]]:
                        stats["unmatched"] += 1
            for column, i in time_idxs.items():
                if i >= len(row) or row[i] == "":
                    continue
                try:
                    value = float(row[i])
                except ValueError:
                    continue
                time_min[column] = min(value, time_min.get(column, value))
                time_max[column] = max(value, time_max.get(column, value))

    result["blank_cells_by_column"] = dict(blank_counts)
    result["marker_cells_by_column"] = {
        column: dict(counts) for column, counts in marker_counts.items()
    }
    result["target_counts"] = dict(target_counts) if target_idx is not None else None
    if key_col:
        result["key_check"] = {
            "candidate_key": key_col,
            "non_null_distinct_values": len(key_values or ()),
            "duplicate_non_null_rows": duplicate_keys,
            "blank_key_rows": blank_counts.get(key_col, 0),
        }
    result["foreign_key_checks"] = {
        column: dict(stats) for column, stats in foreign_stats.items()
    }
    result["relative_time_ranges"] = {
        column: {"min": time_min[column], "max": time_max[column]}
        for column in time_min
    }
    if "malformed_row_widths" in result:
        result["malformed_row_widths"] = {
            str(width): count
            for width, count in result["malformed_row_widths"].items()
        }
    return result


def check_dictionary(data_dir: Path, files: list[dict[str, Any]]) -> None:
    dictionary_path = data_dir / "HomeCredit_columns_description.csv"
    if not dictionary_path.is_file():
        return
    descriptions: dict[str, set[str]] = {}
    raw_names: dict[str, dict[str, str]] = {}
    with dictionary_path.open("r", encoding="cp1252", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            table = row.get("Table", "")
            raw_column = row.get("Row", "")
            column = raw_column.strip()
            if table and column:
                descriptions.setdefault(table, set()).add(column)
                raw_names.setdefault(table, {})[column] = raw_column
    for result in files:
        name = result["file"]
        table_name = (
            "application_{train|test}.csv"
            if name in {"application_train.csv", "application_test.csv"}
            else name
        )
        expected = descriptions.get(table_name)
        if expected is None:
            continue
        expected = set(expected)
        aliases: list[dict[str, str]] = []
        bureau_alias = "SK_BUREAU_ID"
        if bureau_alias in expected and name in {"bureau.csv", "bureau_balance.csv"}:
            expected.remove(bureau_alias)
            expected.add("SK_ID_BUREAU")
            aliases.append({"dictionary_name": bureau_alias, "file_name": "SK_ID_BUREAU"})
        if name == "application_test.csv":
            expected.discard("TARGET")
        actual = set(result["columns"])
        result["dictionary_check"] = {
            "dictionary_table": table_name,
            "normalized_trailing_whitespace": sorted(
                column for column, raw in raw_names.get(table_name, {}).items()
                if raw != column
            ),
            "known_name_aliases_applied": aliases,
            "columns_missing_from_data": sorted(expected - actual),
            "columns_undocumented": sorted(actual - expected),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing the CSV files")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path; defaults to stdout")
    args = parser.parse_args()

    parent_ids = {
        "application_ids": set(),
        "bureau_ids": set(),
        "previous_application_ids": set(),
    }
    files: list[dict[str, Any]] = []
    missing_files = [name for name in EXPECTED_FILES if not (args.data_dir / name).is_file()]
    for name in EXPECTED_FILES:
        path = args.data_dir / name
        if path.is_file():
            files.append(audit_file(path, parent_ids))
    check_dictionary(args.data_dir, files)

    report = {
        "data_directory": args.data_dir.name,
        "expected_files": list(EXPECTED_FILES),
        "missing_files": missing_files,
        "parent_id_counts": {name: len(values) for name, values in parent_ids.items()},
        "files": files,
        "limitations": [
            "Blank cells are counted separately from categorical marker values; marker counts are not automatically treated as missing.",
            "Only documented candidate single-column keys and declared foreign keys are checked; monthly/payment row-grain duplicates need task-specific rules.",
            "Relative day/month ranges do not prove when a field was available; inspect the dictionary and event semantics before feature approval.",
            "This script does not decide whether use or publication is allowed by Kaggle competition rules.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 1 if missing_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
