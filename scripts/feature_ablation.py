#!/usr/bin/env python3
"""Fixed grouped-CV feature-family ablation for logistic and LightGBM."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

from logistic_baseline import CATEGORICAL, calibration_table, load, make_pipeline, metrics
from null_importance_lightgbm import categorical_train_valid, make_model

FAMILIES = {
    "repayment_status": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    "client_profile": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
    "billing_amounts": [f"BILL_AMT{i}" for i in range(1, 7)],
    "payment_amounts": [f"PAY_AMT{i}" for i in range(1, 7)],
    "credit_limit": ["LIMIT_BAL"],
}
METRICS = ["roc_auc", "average_precision", "brier_score", "log_loss",
           "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"]


def conditions(all_features: list[str]) -> dict[str, list[str]]:
    out = {"full": all_features}
    for family, features in FAMILIES.items():
        out[f"only_{family}"] = features
        out[f"without_{family}"] = [feature for feature in all_features if feature not in features]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    X, y, groups, ids, digest = load(args.workbook)
    feature_sets = conditions(list(X.columns))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(outer.split(X, y, groups))
    dev_idx, test_idx = outer_splits[0][0], outer_splits[0][1]
    Xd, yd, gd = X.iloc[dev_idx].reset_index(drop=True), y[dev_idx], groups[dev_idx]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    folds = list(cv.split(Xd, yd, gd))
    fold_assign = np.full(len(yd), -1, dtype=int)
    for fold, (_, valid) in enumerate(folds):
        fold_assign[valid] = fold

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    calibration_rows: list[pd.DataFrame] = []
    all_fit_seconds = {"logistic": 0.0, "lightgbm": 0.0}
    prevalence = float(yd.mean())
    prevalence_metrics = metrics(yd, np.full(len(yd), prevalence))

    for condition, selected in feature_sets.items():
        categories = [name for name in CATEGORICAL if name in selected]
        numeric = [name for name in selected if name not in categories]
        outputs = {"logistic": np.full(len(yd), np.nan),
                   "lightgbm": np.full(len(yd), np.nan)}
        fit_seconds = {"logistic": [], "lightgbm": []}
        for fold, (train, valid) in enumerate(folds):
            if set(gd[train]).intersection(gd[valid]):
                raise RuntimeError(f"group leakage in fold {fold}")
            # Logistic transformations are learned from the training rows only.
            lr = make_pipeline(numeric, categorical=categories)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                lr.fit(Xd.iloc[train][selected], yd[train])
            elapsed_lr = time.perf_counter() - start
            fit_seconds["logistic"].append(elapsed_lr)
            p_lr = lr.predict_proba(Xd.iloc[valid][selected])[:, 1]
            outputs["logistic"][valid] = p_lr

            # Native category vocabularies are derived from each training fold.
            Xtr_tree, Xva_tree = categorical_train_valid(
                Xd.iloc[train][selected], Xd.iloc[valid][selected], categorical=categories)
            tree = make_model(1000 + fold)
            start = time.perf_counter()
            tree.fit(Xtr_tree, yd[train], categorical_feature=categories)
            elapsed_tree = time.perf_counter() - start
            fit_seconds["lightgbm"].append(elapsed_tree)
            p_tree = tree.predict_proba(Xva_tree)[:, 1]
            outputs["lightgbm"][valid] = p_tree

            fold_row = {"condition": condition, "fold": fold, "rows": int(len(valid)),
                        "features": int(len(selected)),
                        "logistic_convergence_warnings": int(sum(isinstance(w.message, ConvergenceWarning) for w in caught))}
            for model_name, probability in (("logistic", p_lr), ("lightgbm", p_tree)):
                fold_row.update({f"{model_name}_{key}": value
                                 for key, value in metrics(yd[valid], probability).items()})
            fold_rows.append(fold_row)

        if any(np.isnan(prob).any() for prob in outputs.values()):
            raise RuntimeError(f"incomplete OOF predictions for {condition}")
        for model_name, probability in outputs.items():
            values = metrics(yd, probability)
            cal, ece = calibration_table(yd, probability)
            values["ece_10_equal_frequency_bins"] = ece
            values["brier_skill_vs_prevalence"] = float(1 - values["brier_score"] / prevalence_metrics["brier_score"])
            mean_time = float(np.mean(fit_seconds[model_name]))
            values["mean_fit_seconds_per_fold"] = mean_time
            values["total_fit_seconds"] = float(sum(fit_seconds[model_name]))
            all_fit_seconds[model_name] += values["total_fit_seconds"]
            summary_rows.append({"condition": condition, "model": model_name,
                                 "feature_count": len(selected),
                                 "features": ";".join(selected), **values})
            cal.insert(0, "model", model_name)
            cal.insert(0, "condition", condition)
            calibration_rows.append(cal)
        print(f"Completed ablation {condition} ({len(selected)} features)")

    summary = pd.DataFrame(summary_rows)
    for model_name in ("logistic", "lightgbm"):
        full = summary[(summary.condition == "full") & (summary.model == model_name)].iloc[0]
        mask = summary.model == model_name
        for metric in METRICS:
            summary.loc[mask, f"delta_{metric}_vs_full"] = summary.loc[mask, metric] - full[metric]

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "t07_ablation_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output / "t07_ablation_fold_metrics.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(args.output / "t07_ablation_calibration.csv", index=False)
    outer_fold = np.full(len(X), -1, dtype=int)
    for fold, (_, valid) in enumerate(outer_splits):
        outer_fold[valid] = fold
    dev_cv = np.full(len(X), -1, dtype=int)
    dev_cv[dev_idx] = fold_assign
    pd.DataFrame({"ID": ids, "outer_fold": outer_fold,
                  "outer_role": np.where(outer_fold == 0, "locked_test", "development"),
                  "development_cv_fold": dev_cv}).to_csv(args.output / "t07_split_manifest.csv", index=False)

    order = list(feature_sets)
    display_labels = {"full": "Full"}
    for family in FAMILIES:
        display_labels[f"only_{family}"] = f"Only {family.replace('_', ' ')}"
        display_labels[f"without_{family}"] = f"Without {family.replace('_', ' ')}"
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    x = np.arange(len(order))
    width = .36
    for ax, model_name in zip(axes, ("logistic", "lightgbm")):
        subset = summary[summary.model == model_name].set_index("condition").loc[order]
        ax.bar(x - width / 2, subset.roc_auc, width, label="ROC AUC")
        ax.bar(x + width / 2, subset.average_precision, width, label="Average precision")
        ax.set(title=model_name, ylabel="Score")
        ax.axhline(.5, linestyle="--", linewidth=.8, color="#2878B5", label="ROC AUC chance baseline (0.5)")
        ax.axhline(prevalence, linestyle=":", linewidth=1, color="#D95319",
                   label=f"AP prevalence baseline ({prevalence:.3f})")
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    axes[-1].set_xticks(x, [display_labels[name] for name in order], rotation=30, ha="right")
    fig.suptitle("Feature-family ablation: ranking metrics (development OOF)")
    fig.savefig(args.output / "t07_ablation_ranking.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    delta_metrics = ["roc_auc", "average_precision", "brier_score", "log_loss"]
    for ax, model_name in zip(axes, ("logistic", "lightgbm")):
        subset = summary[summary.model == model_name].set_index("condition").loc[order]
        matrix = np.column_stack([
            subset[f"delta_{metric}_vs_full"].to_numpy() * (-1 if metric in ("brier_score", "log_loss") else 1)
            for metric in delta_metrics
        ])
        limit = max(float(np.max(np.abs(matrix))), 1e-8)
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        ax.set(title=f"{model_name}: metric change vs full (positive is better)",
               yticks=np.arange(len(order)), yticklabels=[display_labels[name] for name in order],
               xticks=np.arange(len(delta_metrics)), xticklabels=delta_metrics)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=.75, label="Change vs full model (positive is better)")
    fig.suptitle("Feature-family ablation: metric changes (development OOF)")
    fig.savefig(args.output / "t07_ablation_deltas.png", dpi=180)
    plt.close(fig)

    result = {
        "protocol": {"source_file": args.workbook.name, "sha256": digest,
                     "development_rows": int(len(dev_idx)), "locked_test_rows": int(len(test_idx)),
                     "test_fold_fitted_or_scored": False,
                     "outer": "StratifiedGroupKFold(5, shuffle=True, random_state=42); fold 0 locked",
                     "inner": "StratifiedGroupKFold(5, shuffle=True, random_state=43)",
                     "group": "exact tuple of all 23 predictors; ID and target excluded",
                     "feature_families": FAMILIES,
                     "conditions": {key: value for key, value in feature_sets.items()},
                     "models": {"logistic": "unweighted LogisticRegression(C=1.0)",
                                "lightgbm": "fixed T04 configuration; 400 trees; unweighted"},
                     "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                  "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__}},
        "total_fit_seconds": all_fit_seconds,
        "limitations": ["Ablation measures conditional predictive contribution for these fixed models and folds, not causal contribution.",
                        "Group-only models are diagnostic; they are not expected to be deployable alternatives.",
                        "No hyperparameter tuning or outer-test scoring was performed."],
    }
    (args.output / "t07_ablation_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary[["condition", "model", "feature_count", "roc_auc", "average_precision",
                   "brier_score", "log_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
