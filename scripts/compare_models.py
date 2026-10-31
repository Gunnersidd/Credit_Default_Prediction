#!/usr/bin/env python3
"""Compare the fixed T05 logistic and T04 LightGBM models on shared OOF folds."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from logistic_baseline import CATEGORICAL, calibration_table, load, make_pipeline, metrics
from null_importance_lightgbm import categorical_train_valid, make_model


def make_xgb_pipeline(numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    transformers = []
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    classifier = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_estimators=600, max_depth=4, learning_rate=0.03,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, n_jobs=4, random_state=seed, verbosity=0,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    X, y, groups, ids, digest = load(args.workbook)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(outer.split(X, y, groups))
    dev_idx, test_idx = outer_splits[0][0], outer_splits[0][1]
    Xd, yd, gd = X.iloc[dev_idx].reset_index(drop=True), y[dev_idx], groups[dev_idx]
    numeric = [col for col in X.columns if col not in CATEGORICAL]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    predictions = {name: np.full(len(yd), np.nan)
                   for name in ("logistic", "lightgbm", "xgboost")}
    fold_rows: list[dict] = []
    fold_assign = np.full(len(yd), -1, dtype=int)
    fit_seconds = {"logistic": [], "lightgbm": [], "xgboost": []}
    for fold, (tr, va) in enumerate(cv.split(Xd, yd, gd)):
        if set(gd[tr]).intersection(gd[va]):
            raise RuntimeError(f"development fold {fold} has duplicate-profile leakage")
        fold_assign[va] = fold

        lr = make_pipeline(numeric)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            lr.fit(Xd.iloc[tr], yd[tr])
        fit_seconds["logistic"].append(time.perf_counter() - start)
        p_lr = lr.predict_proba(Xd.iloc[va])[:, 1]
        predictions["logistic"][va] = p_lr

        Xtr_tree, Xva_tree = categorical_train_valid(Xd.iloc[tr], Xd.iloc[va])
        tree = make_model(1000 + fold)
        start = time.perf_counter()
        tree.fit(Xtr_tree, yd[tr], categorical_feature=CATEGORICAL)
        fit_seconds["lightgbm"].append(time.perf_counter() - start)
        p_tree = tree.predict_proba(Xva_tree)[:, 1]
        predictions["lightgbm"][va] = p_tree

        booster = make_xgb_pipeline(numeric, CATEGORICAL, 3000 + fold)
        start = time.perf_counter()
        booster.fit(Xd.iloc[tr], yd[tr])
        fit_seconds["xgboost"].append(time.perf_counter() - start)
        p_xgb = booster.predict_proba(Xd.iloc[va])[:, 1]
        predictions["xgboost"][va] = p_xgb

        row = {"fold": fold, "rows": int(len(va)), "defaults": int(yd[va].sum()),
               "default_rate": float(yd[va].mean()),
               "logistic_convergence_warnings": int(sum(isinstance(w.message, ConvergenceWarning) for w in caught))}
        fold_probabilities = {"logistic": p_lr, "lightgbm": p_tree, "xgboost": p_xgb}
        for name, prob in fold_probabilities.items():
            row.update({f"{name}_{key}": value for key, value in metrics(yd[va], prob).items()})
        for left, right in (("lightgbm", "logistic"), ("xgboost", "logistic"),
                            ("xgboost", "lightgbm")):
            for key in ("roc_auc", "average_precision", "brier_score", "log_loss",
                        "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"):
                row[f"{left}_minus_{right}_{key}"] = row[f"{left}_{key}"] - row[f"{right}_{key}"]
        fold_rows.append(row)
        print(f"Completed shared development fold {fold + 1}/5")

    if any(np.isnan(p).any() for p in predictions.values()) or np.any(fold_assign < 0):
        raise RuntimeError("Missing shared OOF predictions/fold assignments")

    prevalence = float(yd.mean())
    summary: dict[str, dict] = {}
    calibration_rows = []
    for name, prob in predictions.items():
        result = metrics(yd, prob)
        result["brier_skill_vs_prevalence"] = float(
            1 - result["brier_score"] / metrics(yd, np.full(len(yd), prevalence))["brier_score"]
        )
        cal, ece = calibration_table(yd, prob)
        result["ece_10_equal_frequency_bins"] = ece
        result["mean_fit_seconds_per_fold"] = float(np.mean(fit_seconds[name]))
        result["total_fit_seconds"] = float(np.sum(fit_seconds[name]))
        result["convergence_warnings"] = int(sum(r["logistic_convergence_warnings"] for r in fold_rows)) if name == "logistic" else 0
        summary[name] = result
        cal.insert(0, "model", name)
        calibration_rows.append(cal)
    summary["prevalence_only"] = {
        "development_default_rate": prevalence,
        "brier_score": float(metrics(yd, np.full(len(yd), prevalence))["brier_score"]),
        "log_loss": float(metrics(yd, np.full(len(yd), prevalence))["log_loss"]),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    outer_fold = np.full(len(X), -1, dtype=int)
    for fold, (_, valid) in enumerate(outer_splits):
        outer_fold[valid] = fold
    dev_cv = np.full(len(X), -1, dtype=int)
    dev_cv[dev_idx] = fold_assign
    pd.DataFrame({"ID": ids, "outer_fold": outer_fold,
                  "outer_role": np.where(outer_fold == 0, "locked_test", "development"),
                  "development_cv_fold": dev_cv}).to_csv(args.output / "t06_split_manifest.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output / "t06_model_comparison_folds.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(args.output / "t06_model_comparison_calibration.csv", index=False)
    pd.DataFrame([{"model": key, **value} for key, value in summary.items()]).to_csv(
        args.output / "t06_model_comparison_metrics.csv", index=False)

    colors = {"logistic": "#2878B5", "lightgbm": "#D95319", "xgboost": "#2E8B57"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob in predictions.items():
        fpr, tpr, _ = roc_curve(yd, prob)
        auc = metrics(yd, prob)["roc_auc"]
        axes[0].plot(fpr, tpr, color=colors[name], label=f"{name} (AUC {auc:.3f})")
        precision, recall, _ = precision_recall_curve(yd, prob)
        ap = average_precision_score(yd, prob)
        axes[1].plot(recall, precision, color=colors[name], label=f"{name} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(title="ROC, development out-of-fold", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(prevalence, linestyle="--", color="gray", linewidth=1, label=f"Prevalence {prevalence:.3f}")
    axes[1].set(title="Precision-recall, development out-of-fold", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "t06_roc_pr_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, prob in predictions.items():
        cal, _ = calibration_table(yd, prob)
        ax.plot(cal.mean_predicted, cal.observed_rate, marker="o", color=colors[name], label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(title="Calibration, development out-of-fold", xlabel="Mean predicted probability",
           ylabel="Observed default rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "t06_calibration_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_groups = [
        ("Ranking", ["roc_auc", "average_precision"]),
        ("Probability loss (lower is better)", ["brier_score", "log_loss"]),
        ("Threshold 0.5", ["precision_at_0_5", "recall_at_0_5", "f1_at_0_5"]),
    ]
    for ax, (title, keys) in zip(axes, metric_groups):
        positions = np.arange(len(keys))
        width = .24
        for offset, name in ((-width, "logistic"), (0, "lightgbm"), (width, "xgboost")):
            ax.bar(positions + offset, [summary[name][key] for key in keys], width,
                   color=colors[name], label=name)
        ax.set(title=title, xticks=positions, xticklabels=[key.replace("_at_0_5", "") for key in keys])
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.suptitle("Fixed-model comparison, development OOF")
    fig.tight_layout()
    fig.savefig(args.output / "t06_metric_comparison.png", dpi=180)
    plt.close(fig)

    fold_frame = pd.DataFrame(fold_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ("roc_auc", "average_precision", "brier_score", "log_loss")):
        for model_name in ("lightgbm", "xgboost"):
            ax.plot(fold_frame.fold, fold_frame[f"{model_name}_minus_logistic_{key}"],
                    marker="o", color=colors[model_name], label=f"{model_name} - logistic")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=key, xlabel="Development fold", ylabel="Difference vs logistic")
        ax.set_xticks(fold_frame.fold)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Fold metric differences vs logistic; lower losses are better")
    fig.tight_layout()
    fig.savefig(args.output / "t06_fold_metric_differences.png", dpi=180)
    plt.close(fig)

    importance = pd.read_csv(args.output / "t04_null_importance_summary.csv")
    importance = importance.nlargest(15, "actual_mean_share").sort_values("actual_mean_share")
    fig, ax = plt.subplots(figsize=(9, 7))
    ypos = np.arange(len(importance))
    height = .24
    ax.barh(ypos - height, importance.actual_mean_share, height, label="Actual mean", color="#2878B5")
    ax.barh(ypos, importance.null_mean_share, height, label="Null mean", color="#888888")
    ax.barh(ypos + height, importance.null_max, height, label="Null max", color="#E6A23C")
    ax.set(yticks=ypos, yticklabels=importance.feature, xlabel="Normalized gain share",
           title="T04 LightGBM importance vs shuffled-target null")
    ax.legend()
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "t04_null_importance.png", dpi=180)
    plt.close(fig)
    result = {
        "models": {
            "logistic": "T05: unweighted LogisticRegression(C=1.0, solver=lbfgs, max_iter=2000); fold-fitted one-hot + StandardScaler",
            "lightgbm": "T04: unweighted LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=40, colsample_bytree=0.9, subsample=0.9, reg_lambda=1.0, importance_type=gain)",
            "xgboost": "unweighted XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03, min_child_weight=10, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method=hist); fold-fitted one-hot for categorical features",
        },
        "pooled_development_oof": summary,
        "protocol": {"source_file": args.workbook.name, "sha256": digest,
                     "rows": int(len(X)), "predictors": list(X.columns), "target": "default payment next month",
                     "development_rows": int(len(dev_idx)), "locked_test_rows": int(len(test_idx)),
                     "outer": "StratifiedGroupKFold(5, shuffle=True, random_state=42); fold 0 locked",
                     "inner": "StratifiedGroupKFold(5, shuffle=True, random_state=43)",
                     "groups": "exact tuple of predictors, excluding ID and target",
                     "categorical": CATEGORICAL,
                     "test_fold_fitted_or_scored": False,
                     "row_level_predictions_saved": False,
                     "plots": ["t04_null_importance.png", "t06_roc_pr_curves.png",
                               "t06_calibration_curve.png", "t06_metric_comparison.png",
                               "t06_fold_metric_differences.png"],
                     "fit_seconds": fit_seconds,
                     "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                  "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
                                  "xgboost": xgb.__version__}},
        "limitations": ["This is one fixed-configuration comparison; it is not a hyperparameter search.",
                        "Five grouped folds on one 2005 Taiwan vintage do not establish current-market or temporal generalization.",
                        "The 0.5 threshold is a reporting convention, not a decision policy.",
                        "Fold-level differences are descriptive and do not replace a larger-sample uncertainty analysis."],
    }
    (args.output / "t06_model_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
