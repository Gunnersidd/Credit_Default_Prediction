# Credit Default Prediction

An educational credit-risk project that estimates the probability of next-month payment default for clients in the UCI Default of Credit Card Clients dataset. The analysis is retrospective and is not a lending decision tool.

## Dataset

The project uses the historical Taiwan credit-card-client dataset published by the UCI Machine Learning Repository: 30,000 records, 23 explanatory features, and the binary `default payment next month` outcome. The source workbook is not included in this repository. Obtain it from the [UCI dataset page](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients) and retain the required [CC BY 4.0 attribution](https://creativecommons.org/licenses/by/4.0/). Citation: I-Cheng Yeh (2009), [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H).

## Project contents

- `scripts/`: dataset audits, logistic baseline, model comparison, feature ablation, and portfolio-mart builder.
- `sql/`: SQL transformations for the aggregate portfolio mart.
- `app/portfolio_dashboard.py`: local interactive portfolio dashboard.
- `reports/`: aggregate metrics, plots, audit summaries, and final reports.
- `requirements-modeling.txt` and `requirements-dashboard.txt`: separate environments for modeling and dashboard work.

## Reproduction

Create an environment and install the dependencies needed for the workflow you plan to run:

```sh
python -m pip install -r requirements-modeling.txt
python -m pip install -r requirements-dashboard.txt
```

The dashboard and mart builder require the local UCI workbook path. For example:

```sh
CREDIT_DATA_PATH="/path/to/default of credit card clients.xls" python app/portfolio_dashboard.py
python scripts/build_portfolio_mart.py "/path/to/default of credit card clients.xls"
```

The repository documents the validation protocol, data audit, and interpretation limits in its reports and source documentation. Results describe this historical dataset only; they do not establish current-market performance or deployment utility.
