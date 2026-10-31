# Modeling results index

All predictive metrics here are development-only out-of-fold estimates from the same five-fold `StratifiedGroupKFold` design. The locked outer test set contains 5,996 records and was not fitted or scored. Split manifests record IDs and fold assignments; no row-level probabilities are stored.

## T04 — LightGBM null importance

- `t04_null_importance_summary.csv`: actual mean gain share, null mean/95th percentile/maximum, empirical upper-tail p-value, and BH q-value per feature.
- `t04_actual_importances_by_fold.csv`, `t04_null_importances.csv`: full importance samples.
- `t04_null_importance_metadata.json`, `t04_split_manifest.csv`: run settings, versions, checksums, and fold assignments.
- `t04_null_importance.png`: actual importance compared with shuffled-target null mean and maximum.

## T05 — Logistic baseline

- `t05_logistic_metrics.json`: pooled OOF metrics and protocol.
- `t05_logistic_fold_metrics.csv`: fold-level metrics.
- `t05_logistic_calibration.csv`: 10-bin equal-frequency calibration summary.
- `t05_split_manifest.csv`: outer and development fold assignments.

## T06 — Logistic vs. LightGBM vs. XGBoost

- `t06_model_comparison.json`, `t06_model_comparison_metrics.csv`: pooled OOF comparison for all three fixed models.
- `t06_model_comparison_folds.csv`: paired fold-level results and metric differences.
- `t06_model_comparison_calibration.csv`: calibration bins for both models.
- `t06_split_manifest.csv`: split assignments.
- `t06_roc_pr_curves.png`, `t06_calibration_curve.png`, `t06_metric_comparison.png`, `t06_fold_metric_differences.png`: OOF comparison plots.

## T07 — Feature-family ablation

- `t07_ablation_summary.csv`: pooled OOF metrics and differences from the full feature set for each model/condition.
- `t07_ablation_fold_metrics.csv`: fold-level metrics for each ablation condition.
- `t07_ablation_calibration.csv`: calibration bins by model/condition.
- `t07_ablation_metadata.json`, `t07_split_manifest.csv`: family definitions, run settings, and assignments.
- `t07_ablation_ranking.png`, `t07_ablation_deltas.png`: ranking metrics and direction-adjusted changes against the full model.

Reproduction environment is pinned in `../requirements-modeling.txt`. Scripts and the source workbook location/attribution are documented in the project execution log and data audit.
# T08 — Analyst workflow

- Local interactive portfolio monitor: [`../app/portfolio_dashboard.py`](../app/portfolio_dashboard.py)
- DuckDB aggregate mart builder: [`../scripts/build_portfolio_mart.py`](../scripts/build_portfolio_mart.py)
- SQL workflow: [`../sql/portfolio_mart.sql`](../sql/portfolio_mart.sql)
- Dashboard dependencies: [`../requirements-dashboard.txt`](../requirements-dashboard.txt)
- Run the app: `CREDIT_DATA_PATH="/path/to/default of credit card clients.xls" python app/portfolio_dashboard.py`
- Generate aggregate tables: `python scripts/build_portfolio_mart.py "/path/to/default of credit card clients.xls"`
- T08 local data mart generated tables are aggregate-only. The raw workbook remains outside this repository.

# T11 — LaTeX deliverables

- [Technical report PDF](latex_source/credit_portfolio_technical_report.pdf): validation results, dashboard workflow, caveats, and Q&A at the end.
- [Project source documentation PDF](latex_source/project_source_documentation.pdf): code map, setup, data notes, launch/reproduction commands, and artifact guide.
- Editable LaTeX sources, [build script](latex_source/build_pdfs.sh), and build notes are in [latex_source/](latex_source/).
- Rebuild from that directory with `sh build_pdfs.sh`.
