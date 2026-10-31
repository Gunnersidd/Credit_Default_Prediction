# LaTeX deliverables

This directory contains the editable LaTeX sources and two compiled PDFs:

- `credit_portfolio_technical_report.pdf` — project purpose, evidence, model comparison, dashboard workflow, limits, and a Q&A section at the end.
- `project_source_documentation.pdf` — code map, environment setup, reproducible commands, artifacts, and source/data notes.

## Build

From this directory, run:

```sh
sh build_pdfs.sh
```

The script uses `pdflatex` and runs each document twice so the table of contents and links settle. The source figures are read from `../` (the parent `reports/` directory). The PDFs are written beside the `.tex` sources.

Sources are `credit_portfolio_technical_report.tex` and `project_source_documentation.tex`. Edit those files and rebuild; do not hand-edit the generated PDFs.

## Evidence boundary

The report's performance values come from the saved development-only OOF tables under `../t06_*` and `../t07_*`. The 5,996-row locked outer test was not used or scored. The dataset is the historical 2005 Taiwan cohort; the documents do not claim current-market or production performance. See `../../data_audit.md`, `../../validation_design.md`, and `../../execution.md` for detailed evidence and task history.
