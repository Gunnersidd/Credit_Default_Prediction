#!/usr/bin/env python3
"""Build aggregate SQL outputs for the UCI credit-card portfolio demo."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

TARGET = "default payment next month"
STATUS_MONTHS = {
    "Apr 2005": "PAY_6", "May 2005": "PAY_5", "Jun 2005": "PAY_4",
    "Jul 2005": "PAY_3", "Aug 2005": "PAY_2", "Sep 2005": "PAY_0",
}


def load_workbook(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=0, header=1)
    frame.columns = [str(c).strip() for c in frame.columns]
    if TARGET not in frame or "ID" not in frame:
        raise ValueError(f"Expected ID and {TARGET!r} columns; found {frame.columns.tolist()}")
    expected = set(STATUS_MONTHS.values())
    if not expected.issubset(frame.columns):
        raise ValueError(f"Missing repayment-status fields: {sorted(expected - set(frame.columns))}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Path to the locally held UCI .xls workbook")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/portfolio_mart"))
    args = parser.parse_args()
    data = load_workbook(args.workbook)
    data["LIMIT_BAND"] = pd.cut(data["LIMIT_BAL"], [-float("inf"), 50000, 150000, 300000, float("inf")],
                                labels=["≤50k", "50k–150k", "150k–300k", ">300k"])
    data["AGE_BAND"] = pd.cut(data["AGE"], [0, 29, 39, 49, 59, float("inf")],
                              labels=["≤29", "30–39", "40–49", "50–59", "60+"])
    status = data[["ID", *STATUS_MONTHS.values()]].melt(
        id_vars="ID", var_name="source_field", value_name="status_code")
    month_by_field = {field: month for month, field in STATUS_MONTHS.items()}
    status["month"] = status["source_field"].map(month_by_field)
    status["month_order"] = status["month"].map({m: i for i, m in enumerate(STATUS_MONTHS, 1)})

    con = duckdb.connect()
    con.register("source_clients", data)
    con.register("source_status", status)
    con.execute("CREATE TABLE clients AS SELECT * FROM source_clients")
    con.execute("CREATE TABLE monthly_status AS SELECT * FROM source_status")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        "portfolio_kpis.csv": """SELECT count(*) AS client_count, count(DISTINCT ID) AS unique_clients,
            avg(\"default payment next month\") AS next_month_default_share,
            avg(LIMIT_BAL) AS mean_credit_limit, median(LIMIT_BAL) AS median_credit_limit
            FROM clients""",
        "monthly_status_mix.csv": """SELECT month, month_order, status_code, count(*) AS client_months,
            avg(CASE WHEN status_code > 0 THEN 1.0 ELSE 0.0 END) AS positive_code_share
            FROM monthly_status GROUP BY month, month_order, status_code ORDER BY month_order, status_code""",
        "limit_band_default_rate.csv": """SELECT LIMIT_BAND AS segment, count(*) AS clients,
            avg(\"default payment next month\") AS next_month_default_share FROM clients
            GROUP BY 1 ORDER BY 1""",
        "age_band_default_rate.csv": """SELECT AGE_BAND AS segment, count(*) AS clients,
            avg(\"default payment next month\") AS next_month_default_share FROM clients
            GROUP BY 1 ORDER BY 1""",
    }
    for filename, sql in queries.items():
        con.execute(sql).df().to_csv(args.output_dir / filename, index=False)
    metadata = pd.DataFrame([{"source_rows": len(data), "unique_ids": data.ID.nunique(),
                              "status_periods": len(STATUS_MONTHS),
                              "status_fields": ",".join(STATUS_MONTHS.values()),
                              "scope": "aggregate demo outputs; source rows not exported"}])
    metadata.to_json(args.output_dir / "metadata.json", orient="records", indent=2)
    con.close()
    print(f"Wrote aggregate mart outputs to {args.output_dir}; raw rows were not exported.")


if __name__ == "__main__":
    main()
