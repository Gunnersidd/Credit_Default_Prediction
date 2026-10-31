"""Local Gradio + DuckDB portfolio monitoring dashboard."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import duckdb
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TARGET = "default payment next month"
STATUS_MONTHS = {
    "Apr 2005": "PAY_6", "May 2005": "PAY_5", "Jun 2005": "PAY_4",
    "Jul 2005": "PAY_3", "Aug 2005": "PAY_2", "Sep 2005": "PAY_0",
}
AGE_OPTIONS = ["All", "≤29", "30–39", "40–49", "50–59", "60+"]
LIMIT_OPTIONS = ["All", "≤50k", "50k–150k", "150k–300k", ">300k"]


@lru_cache(maxsize=2)
def load_mart(path_text: str):
    frame = pd.read_excel(path_text, sheet_name=0, header=1)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = {"ID", "LIMIT_BAL", "AGE", TARGET, *STATUS_MONTHS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Workbook is missing required columns: {sorted(missing)}")
    frame["LIMIT_BAND"] = pd.cut(
        frame.LIMIT_BAL, [-float("inf"), 50000, 150000, 300000, float("inf")],
        labels=LIMIT_OPTIONS[1:],
    ).astype(str)
    frame["AGE_BAND"] = pd.cut(
        frame.AGE, [0, 29, 39, 49, 59, float("inf")],
        labels=AGE_OPTIONS[1:], include_lowest=True,
    ).astype(str)
    status = frame[["ID", *STATUS_MONTHS.values()]].melt(
        id_vars="ID", var_name="source_field", value_name="status_code")
    status["month"] = status.source_field.map({field: month for month, field in STATUS_MONTHS.items()})
    status["month_order"] = status.month.map({month: i for i, month in enumerate(STATUS_MONTHS, 1)})
    con = duckdb.connect(database=":memory:")
    con.register("client_frame", frame)
    con.register("status_frame", status)
    con.execute("CREATE TABLE clients AS SELECT * FROM client_frame")
    con.execute("CREATE TABLE monthly_status AS SELECT * FROM status_frame")
    return con, len(frame), int(frame.ID.nunique())


def query(con, sql: str, params=None):
    return con.execute(sql, params or []).df()


def empty_outputs(message: str):
    return message, None, None, pd.DataFrame(), None, None


def refresh(path_text: str, age_band: str, limit_band: str):
    if not path_text:
        return empty_outputs("Enter the local .xls workbook path, then click **Load portfolio**.")
    if not Path(path_text).is_file():
        return empty_outputs("Workbook not found. Check the path and try again.")
    try:
        con, source_rows, unique_ids = load_mart(path_text)
    except Exception as exc:
        return empty_outputs(f"Could not load workbook: {exc}")

    conditions, params = [], []
    if age_band != "All":
        conditions.append("AGE_BAND = ?")
        params.append(age_band)
    if limit_band != "All":
        conditions.append("LIMIT_BAND = ?")
        params.append(limit_band)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    kpi = query(con, f"""SELECT count(*) AS n, avg(\"{TARGET}\") AS default_share,
        median(LIMIT_BAL) AS median_limit,
        avg(CASE WHEN PAY_0 > 0 THEN 1.0 ELSE 0.0 END) AS latest_positive_code_share
        FROM clients{where}""", params).iloc[0]
    kpis = f"""### Portfolio snapshot
**Accounts in selection:** {int(kpi.n):,} of {source_rows:,} source rows ({unique_ids:,} unique source IDs)  
**Historical next-month default share:** {kpi.default_share:.1%}  
**Median stated credit limit:** {kpi.median_limit:,.0f}  
**September positive status-code share:** {kpi.latest_positive_code_share:.1%}

Positive status-code share means `PAY_0 > 0` divided by selected clients. It is a raw code indicator, not a portfolio delinquency-rate definition.
"""
    selected_ids = f"SELECT ID FROM clients{where}"
    status_mix = query(con, f"""SELECT month, month_order,
        CASE WHEN status_code > 0 THEN 'Positive code' ELSE 'Non-positive code' END AS status_group,
        count(*) AS client_months
        FROM monthly_status WHERE ID IN ({selected_ids})
        GROUP BY 1,2,3 ORDER BY month_order, status_group""", params)
    fig_status, ax = plt.subplots(figsize=(8, 4))
    piv = status_mix.pivot(index="month", columns="status_group", values="client_months").fillna(0)
    piv = piv.reindex(list(STATUS_MONTHS))
    piv.plot(kind="bar", stacked=True, ax=ax, color=["#4285f4", "#e07a5f"])
    ax.set(title="Repayment status code mix", xlabel="Statement month", ylabel="Client-month records")
    ax.legend(title="Observed code group", frameon=False)
    fig_status.tight_layout()

    transitions = query(con, f"""SELECT
        CASE WHEN previous.status_code > 0 THEN CAST(previous.status_code AS VARCHAR) ELSE '≤0' END AS aug_code,
        CASE WHEN latest.status_code > 0 THEN CAST(latest.status_code AS VARCHAR) ELSE '≤0' END AS sep_code,
        count(*) AS clients
        FROM monthly_status previous JOIN monthly_status latest USING (ID)
        WHERE previous.source_field='PAY_2' AND latest.source_field='PAY_0'
          AND previous.ID IN ({selected_ids})
        GROUP BY 1,2 ORDER BY clients DESC""", params)

    limit_segments = query(con, f"""SELECT LIMIT_BAND AS segment, count(*) AS clients,
        avg(\"{TARGET}\") AS default_share FROM clients{where}
        GROUP BY 1 ORDER BY CASE segment WHEN '≤50k' THEN 1 WHEN '50k–150k' THEN 2 WHEN '150k–300k' THEN 3 ELSE 4 END""", params)
    fig_limit, ax = plt.subplots(figsize=(6, 4))
    ax.bar(limit_segments.segment, limit_segments.default_share, color="#2a9d8f")
    ax.set(title="Historical next-month default share by limit", xlabel="Credit limit band", ylabel="Share")
    ax.tick_params(axis="x", rotation=20)
    fig_limit.tight_layout()

    age_segments = query(con, f"""SELECT AGE_BAND AS segment, count(*) AS clients,
        avg(\"{TARGET}\") AS default_share FROM clients{where}
        GROUP BY 1 ORDER BY CASE segment WHEN '≤29' THEN 1 WHEN '30–39' THEN 2 WHEN '40–49' THEN 3 WHEN '50–59' THEN 4 ELSE 5 END""", params)
    fig_age, ax = plt.subplots(figsize=(6, 4))
    ax.bar(age_segments.segment, age_segments.default_share, color="#8064a2")
    ax.set(title="Historical next-month default share by age", xlabel="Age band", ylabel="Share")
    ax.tick_params(axis="x", rotation=20)
    fig_age.tight_layout()

    return kpis, fig_status, transitions, limit_segments, fig_limit, fig_age


with gr.Blocks(title="Credit Portfolio Monitor") as demo:
    gr.Markdown("""# Credit Portfolio Monitor
Analyst workflow demo · SQL-backed portfolio and payment behavior views

> **Historical UCI sample:** 30,000 Taiwan credit-card clients from 2005. The target is default payment next month. Use this to demonstrate analyst workflows, not to infer current portfolio performance or make customer decisions.
""")
    with gr.Row():
        workbook_path = gr.Textbox(
            label="Local UCI workbook (.xls)", value=os.environ.get("CREDIT_DATA_PATH", ""),
            placeholder="/path/to/default of credit card clients.xls", scale=5,
        )
        load_button = gr.Button("Load portfolio", variant="primary", scale=1)
    gr.Markdown("The workbook stays on this machine; the app prepares an in-memory DuckDB mart.")
    with gr.Row():
        age_filter = gr.Dropdown(AGE_OPTIONS, value="All", label="Age band")
        limit_filter = gr.Dropdown(LIMIT_OPTIONS, value="All", label="Credit limit band")
    kpis_out = gr.Markdown("Enter the local workbook path, then click **Load portfolio**.")
    status_plot = gr.Plot(label="Six-month status code mix")
    with gr.Row():
        transition_table = gr.Dataframe(label="August-to-September code transitions", interactive=False)
        limit_table = gr.Dataframe(label="Credit limit segment rates", interactive=False)
    with gr.Row():
        limit_plot = gr.Plot(label="Default share by limit")
        age_plot = gr.Plot(label="Default share by age")
    gr.Markdown("""**Definitions:** `PAY_6`, `PAY_5`, `PAY_4`, `PAY_3`, `PAY_2`, `PAY_0` map to April through September 2005. Positive status-code share is the proportion with latest `PAY_0 > 0`. Non-positive codes are grouped without interpreting their meanings. Segment outcome shares describe this historical sample only; small filtered groups may be unstable.""")
    outputs = [kpis_out, status_plot, transition_table, limit_table, limit_plot, age_plot]
    inputs = [workbook_path, age_filter, limit_filter]
    load_button.click(refresh, inputs=inputs, outputs=outputs)
    age_filter.change(refresh, inputs=inputs, outputs=outputs)
    limit_filter.change(refresh, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
