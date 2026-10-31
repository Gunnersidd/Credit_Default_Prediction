#!/usr/bin/env python3
"""Grouped-CV logistic regression baseline for UCI credit-card default data."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default payment next month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]:
    book = xlrd.open_workbook(str(path), on_demand=True)
    sheet = book.sheet_by_name("Data")
    names = [str(value).strip() for value in sheet.row_values(1)]
    data = pd.DataFrame(sheet.row_values(i) for i in range(2, sheet.nrows))
    data.columns = names
    data = data[pd.to_numeric(data["ID"], errors="coerce").notna()].copy()
    data["ID"] = data["ID"].astype(int)
    y = data.pop(TARGET).astype(int).to_numpy()
    ids = data.pop("ID").to_numpy()
    groups = pd.factorize(pd.MultiIndex.from_frame(data), sort=False)[0]
    book.release_resources()
    if len(data) != 30_000 or data.shape[1] != 23 or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Unexpected source shape/target: {data.shape}, {np.unique(y)}")
    return data, y, groups, ids, sha256(path)


def make_pipeline(numeric: list[str], categorical: list[str] = CATEGORICAL) -> Pipeline:
    transformers = []
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                      class_weight=None, random_state=43)),
    ])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = p >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "predicted_positive_rate_at_0_5": float(pred.mean()),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    frame = pd.DataFrame({"y": y, "probability": p})
    frame["bin"] = pd.qcut(frame.probability, q=bins, duplicates="drop")
    result = frame.groupby("bin", observed=True).agg(
        rows=("y", "size"), mean_predicted=("probability", "mean"),
        observed_rate=("y", "mean"),
    ).reset_index(drop=True)
    ece = float(np.average(np.abs(result.mean_predicted - result.observed_rate),
                           weights=result.rows))
    return result, ece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    X, y, groups, ids, digest = load(args.workbook)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(outer.split(X, y, groups))
    # Fold 0's training indices are exactly the other four outer folds.
    dev_idx = outer_splits[0][0]
    test_idx = outer_splits[0][1]
    Xd, yd, gd = X.iloc[dev_idx].reset_index(drop=True), y[dev_idx], groups[dev_idx]
    numeric = [column for column in X.columns if column not in CATEGORICAL]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    oof = np.full(len(yd), np.nan)
    inner_fold = np.full(len(yd), -1, dtype=int)
    fold_metrics: list[dict] = []
    for fold, (train, valid) in enumerate(cv.split(Xd, yd, gd)):
        overlap = set(gd[train]).intersection(gd[valid])
        if overlap:
            raise RuntimeError(f"development fold {fold} has {len(overlap)} groups crossing")
        model = make_pipeline(numeric)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(Xd.iloc[train], yd[train])
        p = model.predict_proba(Xd.iloc[valid])[:, 1]
        oof[valid] = p
        inner_fold[valid] = fold
        item = {"fold": fold, "rows": int(len(valid)), "defaults": int(yd[valid].sum()),
                "default_rate": float(yd[valid].mean()),
                "groups": int(len(np.unique(gd[valid]))),
                "convergence_warnings": int(sum(isinstance(w.message, ConvergenceWarning) for w in caught))}
        item.update(metrics(yd[valid], p))
        fold_metrics.append(item)
        print(f"Completed development fold {fold + 1}/5")
    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions missing for development rows")

    pooled = metrics(yd, oof)
    prevalence = float(yd.mean())
    pooled.update({
        "development_rows": int(len(yd)),
        "development_default_rate": prevalence,
        "prevalence_only_brier": float(brier_score_loss(yd, np.full(len(yd), prevalence))),
        "prevalence_only_log_loss": float(log_loss(yd, np.full(len(yd), prevalence), labels=[0, 1])),
    })
    pooled["brier_skill_vs_prevalence"] = float(
        1 - pooled["brier_score"] / pooled["prevalence_only_brier"]
    )
    cal, ece = calibration_table(yd, oof)
    pooled["ece_10_equal_frequency_bins"] = ece

    args.output.mkdir(parents=True, exist_ok=True)
    outer_fold = np.full(len(X), -1, dtype=int)
    for fold, (_, valid) in enumerate(outer_splits):
        outer_fold[valid] = fold
    dev_cv_fold = np.full(len(X), -1, dtype=int)
    dev_cv_fold[dev_idx] = inner_fold
    pd.DataFrame({"ID": ids, "outer_fold": outer_fold,
                  "outer_role": np.where(outer_fold == 0, "locked_test", "development"),
                  "development_cv_fold": dev_cv_fold}).to_csv(
                      args.output / "t05_split_manifest.csv", index=False)
    cal.to_csv(args.output / "t05_logistic_calibration.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(args.output / "t05_logistic_fold_metrics.csv", index=False)
    result = {
        "model": "unweighted LogisticRegression(C=1.0, solver=lbfgs, max_iter=2000)",
        "metrics": pooled,
        "data": {"source_file": args.workbook.name, "sha256": digest,
                 "rows": int(len(X)), "predictors": list(X.columns), "target": TARGET,
                 "development_rows": int(len(dev_idx)), "locked_test_rows": int(len(test_idx)),
                 "outer_lock_fold": 0,
                 "split_manifest": "t05_split_manifest.csv"},
        "split": {"outer": "StratifiedGroupKFold(5, shuffle=True, random_state=42); fold 0 locked",
                  "development": "StratifiedGroupKFold(5, shuffle=True, random_state=43)",
                  "group": "exact tuple of 23 predictors; excludes ID and target",
                  "fold_metrics": fold_metrics},
        "preprocessing": {"categorical": CATEGORICAL,
                          "categorical_transform": "fold-fitted OneHotEncoder(handle_unknown=ignore)",
                          "numeric_transform": "fold-fitted StandardScaler",
                          "numeric": numeric, "class_weight": None},
        "calibration": {"method": "10 equal-frequency bins of development out-of-fold probabilities",
                        "ece": ece, "table": "t05_logistic_calibration.csv"},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "xlrd": xlrd.__version__},
        "limitations": ["Development grouped-CV estimates are for the historical 2005 Taiwan sample, not current-market performance.",
                        "Threshold 0.5 is a fixed reporting convention, not a decision recommendation.",
                        "No row-level predictions are saved; outer locked test is not fitted or scored."],
    }
    (args.output / "t05_logistic_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
