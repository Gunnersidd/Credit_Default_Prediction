#!/usr/bin/env python3
"""Grouped-CV LightGBM target-permutation importance for UCI credit data.

Only development folds are used. The locked outer test fold is never read by
the modeling loop. Requires pandas, numpy, scikit-learn, LightGBM, and xlrd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.model_selection import StratifiedGroupKFold

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TARGET = "default payment next month"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]:
    book = xlrd.open_workbook(str(path), on_demand=True)
    sheet = book.sheet_by_name("Data")
    names = [str(v).strip() for v in sheet.row_values(1)]
    frame = pd.DataFrame(sheet.row_values(i) for i in range(2, sheet.nrows))
    frame.columns = names
    frame = frame[pd.to_numeric(frame["ID"], errors="coerce").notna()].copy()
    frame["ID"] = frame["ID"].astype(int)
    y = frame.pop(TARGET).astype(int).to_numpy()
    ids = frame.pop("ID").to_numpy()
    groups = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)[0]
    book.release_resources()
    return frame, y, groups, ids, sha256(path)


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=400, learning_rate=0.04,
        num_leaves=15, max_depth=-1, min_child_samples=40,
        colsample_bytree=0.9, subsample=0.9, subsample_freq=1,
        reg_lambda=1.0, importance_type="gain", n_jobs=4,
        random_state=seed, verbosity=-1,
    )


def categorical_training_frame(frame: pd.DataFrame, categorical: list[str] = CATEGORICAL) -> pd.DataFrame:
    """Fit category vocabularies on this fold's training rows only."""
    result = frame.copy()
    for col in categorical:
        result[col] = result[col].astype("category")
    return result


def categorical_train_valid(train: pd.DataFrame, valid: pd.DataFrame,
                            categorical: list[str] = CATEGORICAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build category levels from train rows and apply that schema to validation."""
    train_result = train.copy()
    valid_result = valid.copy()
    for col in categorical:
        train_result[col] = train_result[col].astype("category")
        valid_result[col] = valid_result[col].astype(train_result[col].dtype)
    return train_result, valid_result


def bh_adjust(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q_sorted = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")

    X, y, groups, ids, digest = load(args.workbook)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(outer.split(X, y, groups))
    # Fold 0's training indices are exactly the other four outer folds.
    dev_idx = outer_splits[0][0]
    test_idx = outer_splits[0][1]
    Xd, yd, gd = X.iloc[dev_idx].reset_index(drop=True), y[dev_idx], groups[dev_idx]

    # Verify all outer partitions are group-disjoint and log the untouched test
    # fold's aggregate metadata without using its rows in fitting or selection.
    split_rows = []
    for fold, (tr, va) in enumerate(outer_splits):
        overlap = set(groups[tr]).intersection(groups[va])
        if overlap:
            raise RuntimeError(f"outer fold {fold} has {len(overlap)} profile groups crossing")
        split_rows.append({"fold": fold, "rows": int(len(va)), "defaults": int(y[va].sum()),
                           "rate": float(y[va].mean()), "groups": int(len(np.unique(groups[va]))),
                           "role": "locked_test" if fold == 0 else "development"})

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    actual_rows: list[dict] = []
    null_rows: list[dict] = []
    inner_fold = np.full(len(Xd), -1, dtype=int)
    for fold, (tr, va) in enumerate(cv.split(Xd, yd, gd)):
        if set(gd[tr]).intersection(gd[va]):
            raise RuntimeError(f"development fold {fold} has profile overlap")
        inner_fold[va] = fold
        X_train = categorical_training_frame(Xd.iloc[tr])
        model = make_model(1000 + fold)
        model.fit(X_train, yd[tr], categorical_feature=CATEGORICAL)
        imp = model.booster_.feature_importance(importance_type="gain").astype(float)
        imp = imp / imp.sum() if imp.sum() else imp
        actual_rows += [{"feature": f, "fold": fold, "importance_share": float(v)}
                        for f, v in zip(X.columns, imp)]
        del model
        for rep in range(args.permutations):
            rng = np.random.default_rng(20261030 + fold * 10000 + rep)
            y_perm = yd[tr].copy()
            rng.shuffle(y_perm)
            null_model = make_model(20000 + fold * args.permutations + rep)
            null_model.fit(X_train, y_perm, categorical_feature=CATEGORICAL)
            null_imp = null_model.booster_.feature_importance(importance_type="gain").astype(float)
            null_imp = null_imp / null_imp.sum() if null_imp.sum() else null_imp
            null_rows += [{"feature": f, "fold": fold, "permutation": rep,
                           "importance_share": float(v)} for f, v in zip(X.columns, null_imp)]
            del null_model
        print(f"Completed development fold {fold + 1}/5")

    actual = pd.DataFrame(actual_rows)
    null = pd.DataFrame(null_rows)
    summaries = []
    for feature in X.columns:
        av = actual.loc[actual.feature == feature, "importance_share"].to_numpy()
        nv = null.loc[null.feature == feature, "importance_share"].to_numpy()
        a_mean = float(av.mean())
        p = float((1 + np.count_nonzero(nv >= a_mean)) / (len(nv) + 1))
        summaries.append({"feature": feature, "actual_mean_share": a_mean,
                          "actual_sd_across_folds": float(av.std(ddof=1)),
                          "null_mean_share": float(nv.mean()),
                          "null_95th_percentile": float(np.quantile(nv, .95)),
                          "null_max": float(nv.max()),
                          "empirical_upper_tail_p": p})
    summary = pd.DataFrame(summaries)
    summary["bh_q_value"] = bh_adjust(summary["empirical_upper_tail_p"].to_numpy())
    summary["actual_minus_null_mean"] = summary.actual_mean_share - summary.null_mean_share
    summary = summary.sort_values(["bh_q_value", "actual_mean_share"], ascending=[True, False])

    args.output.mkdir(parents=True, exist_ok=True)
    outer_fold = np.full(len(X), -1, dtype=int)
    for fold, (_, valid) in enumerate(outer_splits):
        outer_fold[valid] = fold
    dev_cv_fold = np.full(len(X), -1, dtype=int)
    dev_cv_fold[dev_idx] = inner_fold
    pd.DataFrame({"ID": ids, "outer_fold": outer_fold,
                  "outer_role": np.where(outer_fold == 0, "locked_test", "development"),
                  "development_cv_fold": dev_cv_fold}).to_csv(
                      args.output / "t04_split_manifest.csv", index=False)
    summary.to_csv(args.output / "t04_null_importance_summary.csv", index=False)
    actual.to_csv(args.output / "t04_actual_importances_by_fold.csv", index=False)
    null.to_csv(args.output / "t04_null_importances.csv", index=False)
    meta = {"source_file": args.workbook.name, "sha256": digest,
            "rows": int(len(X)), "features": list(X.columns), "target": TARGET,
            "outer_split": split_rows, "development_rows": int(len(dev_idx)),
            "locked_test_rows": int(len(test_idx)), "null_permutations_per_fold": args.permutations,
            "split_manifest": "t04_split_manifest.csv",
            "null_model_fits": int(args.permutations * 5), "actual_model_fits": 5,
            "permutation": "shuffle labels among training rows separately within each development fold",
            "importance": "LightGBM gain importance normalized to sum to one within each fitted model",
            "categorical_handling": "pandas categorical vocabularies formed from each training fold only; negative source codes retained as category levels",
            "p_value": "(1 + number of null shares >= mean actual share across folds)/(1 + null fits)",
            "split_group": "exact tuple of 23 predictor values; IDs and target excluded",
            "seeds": {"outer": 42, "development": 43, "null_base": 20261030},
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
                         "xlrd": xlrd.__version__},
            "caveats": ["Importance and permutation p-values are model-dependent screening evidence, not causal effects.",
                        "Only 5 actual fold fits and the configured null fit count are used.",
                        "The locked outer test fold was not fit, scored, or used for feature selection.",
                        "This null comparison is approximate because null labels are shuffled within each fold's training rows."]}
    (args.output / "t04_null_importance_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote T04 outputs under {args.output}")


if __name__ == "__main__":
    main()
