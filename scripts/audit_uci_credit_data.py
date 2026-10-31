#!/usr/bin/env python3
"""Audit UCI's Default of Credit Card Clients Excel workbook.

Requires xlrd 2.x to read the original .xls file. Output is aggregate metadata;
no individual client rows are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import xlrd


CATEGORICAL_FIELDS = {
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
}

PUBLISHED_CODES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    "PAY_0": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_2": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_3": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_4": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_5": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_6": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def audit(path: Path) -> dict[str, Any]:
    book = xlrd.open_workbook(str(path), on_demand=True)
    report: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "sheet_names": book.sheet_names(),
        "sheets": [],
    }

    for sheet in book.sheets():
        sheet_result: dict[str, Any] = {
            "sheet": sheet.name,
            "worksheet_rows_including_headers": sheet.nrows,
            "worksheet_columns": sheet.ncols,
            "data_rows": 0,
            "column_codes": [],
            "column_names": [],
            "duplicate_column_names": [],
            "blank_cells_by_column": {},
            "target_counts": {},
            "id_check": {},
            "duplicate_feature_rows_excluding_id": 0,
            "predictor_profile_groups": {},
            "numeric_ranges": {},
            "categorical_counts": {},
            "codes_outside_published_description": {},
            "row_width_mismatches": 0,
        }

        if sheet.nrows < 3:
            sheet_result["error"] = "fewer than two header rows and one data row"
            report["sheets"].append(sheet_result)
            continue

        codes = [str(value).strip() for value in sheet.row_values(0)]
        names = [str(value).strip() for value in sheet.row_values(1)]
        sheet_result["column_codes"] = codes
        sheet_result["column_names"] = names
        sheet_result["duplicate_column_names"] = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        column_index = {name: index for index, name in enumerate(names)}
        blank_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        category_counts: dict[str, Counter[str]] = {
            name: Counter() for name in CATEGORICAL_FIELDS if name in column_index
        }
        numeric_min: dict[str, float] = {}
        numeric_max: dict[str, float] = {}
        ids: set[int] = set()
        duplicate_ids = 0
        non_integer_ids = 0
        duplicate_feature_rows = 0
        feature_rows: set[tuple[Any, ...]] = set()
        predictor_profiles: dict[tuple[Any, ...], Counter[str]] = {}
        width_mismatches = 0

        target_index = column_index.get("default payment next month")
        id_index = column_index.get("ID")
        for row_index in range(2, sheet.nrows):
            row = sheet.row_values(row_index)
            sheet_result["data_rows"] += 1
            if len(row) != len(names):
                width_mismatches += 1

            for index, name in enumerate(names):
                value = row[index] if index < len(row) else ""
                if value == "" or value is None:
                    blank_counts[name] += 1
                    continue
                number = as_number(value)
                if number is not None:
                    numeric_min[name] = min(number, numeric_min.get(name, number))
                    numeric_max[name] = max(number, numeric_max.get(name, number))
                if name in category_counts:
                    category_counts[name][str(value)] += 1

            if target_index is not None:
                target_counts[str(row[target_index])] += 1
            if id_index is not None:
                raw_id = as_number(row[id_index])
                if raw_id is None or not raw_id.is_integer():
                    non_integer_ids += 1
                else:
                    client_id = int(raw_id)
                    if client_id in ids:
                        duplicate_ids += 1
                    ids.add(client_id)

            feature_tuple = tuple(row[1:])
            if feature_tuple in feature_rows:
                duplicate_feature_rows += 1
            else:
                feature_rows.add(feature_tuple)
            if target_index is not None:
                predictors = tuple(row[1:target_index])
                predictor_profiles.setdefault(predictors, Counter())[str(row[target_index])] += 1

        sheet_result["blank_cells_by_column"] = dict(blank_counts)
        sheet_result["target_counts"] = dict(target_counts)
        sheet_result["id_check"] = {
            "non_null_distinct_ids": len(ids),
            "duplicate_ids": duplicate_ids,
            "non_integer_or_missing_ids": non_integer_ids,
            "min_id": min(ids) if ids else None,
            "max_id": max(ids) if ids else None,
        }
        sheet_result["duplicate_feature_rows_excluding_id"] = duplicate_feature_rows
        repeated_profiles = {
            profile: counts
            for profile, counts in predictor_profiles.items()
            if sum(counts.values()) > 1
        }
        sheet_result["predictor_profile_groups"] = {
            "unique_profiles": len(predictor_profiles),
            "repeated_profile_groups": len(repeated_profiles),
            "rows_in_repeated_profiles": sum(
                sum(counts.values()) for counts in repeated_profiles.values()
            ),
            "excess_rows_in_repeated_profiles": sum(
                sum(counts.values()) - 1 for counts in repeated_profiles.values()
            ),
            "profiles_with_both_target_classes": sum(
                len(counts) > 1 for counts in repeated_profiles.values()
            ),
        }
        sheet_result["numeric_ranges"] = {
            name: {"min": numeric_min[name], "max": numeric_max[name]}
            for name in numeric_min
        }
        sheet_result["categorical_counts"] = {
            name: dict(sorted(counts.items(), key=lambda item: item[0]))
            for name, counts in category_counts.items()
        }
        sheet_result["codes_outside_published_description"] = {
            name: {
                code: count
                for code, count in sorted(
                    category_counts[name].items(), key=lambda item: item[0]
                )
                if _code_is_undocumented(code, PUBLISHED_CODES[name])
            }
            for name in category_counts
        }
        sheet_result["row_width_mismatches"] = width_mismatches
        report["sheets"].append(sheet_result)

    book.release_resources()
    report["limitations"] = [
        "The workbook has two header rows; the code labels and human-readable field names are preserved separately.",
        "Published code descriptions are compared literally; undocumented codes are reported, not silently recoded.",
        "A clean local file does not establish current-market representativeness or prove model feature timing beyond the source definition.",
    ]
    return report


def _code_is_undocumented(code: str, published: set[int]) -> bool:
    try:
        number = float(code)
    except ValueError:
        return True
    return not number.is_integer() or int(number) not in published


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Path to the original UCI .xls workbook")
    parser.add_argument("--output", type=Path, help="Write aggregate JSON to this path")
    args = parser.parse_args()
    result = audit(args.workbook)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
