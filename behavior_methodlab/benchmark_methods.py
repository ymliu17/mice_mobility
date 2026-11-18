#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from behavior_methodlab.feature_spaces import get_feature_space, load_unified_session_table


OUTDIR = Path("outputs/methodlab/benchmarks")
SEED = 42
FEATURE_SPACES = ["baseline_current", "graph_core5", "usage_graph_hybrid"]
MODEL_FAMILIES = ["elasticnet_logreg", "linear_svm", "gbdt"]


def _make_model(name: str) -> Pipeline:
    if name == "elasticnet_logreg":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])
    if name == "linear_svm":
        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=10000,
            random_state=SEED,
        )
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])
    if name == "gbdt":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            random_state=SEED,
        )
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])
    raise ValueError(f"Unknown model family: {name}")


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="binary")),
    }
    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc"] = np.nan
    return out


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        out["macro_ovr_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["macro_ovr_auc"] = np.nan
    return out


def _binary_score(model: Pipeline, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        score = model.predict_proba(X_te)[:, 1]
    elif hasattr(clf, "decision_function"):
        score = model.decision_function(X_te)
    else:
        score = model.predict(X_te).astype(float)
    pred = (score >= 0.5).astype(int) if score.ndim == 1 and score.dtype.kind == "f" and score.min() >= 0 and score.max() <= 1 else model.predict(X_te)
    return np.asarray(pred), np.asarray(score)


def _multiclass_score(model: Pipeline, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = model.named_steps["clf"]
    pred = model.predict(X_te)
    if hasattr(clf, "predict_proba"):
        prob = model.predict_proba(X_te)
    elif hasattr(clf, "decision_function"):
        score = model.decision_function(X_te)
        score = np.asarray(score, dtype=float)
        if score.ndim == 1:
            score = np.c_[1.0 - score, score]
        score = score - score.max(axis=1, keepdims=True)
        exp = np.exp(score)
        prob = exp / exp.sum(axis=1, keepdims=True)
    else:
        n_classes = len(np.unique(pred))
        prob = np.zeros((len(pred), n_classes), dtype=float)
        prob[np.arange(len(pred)), pred] = 1.0
    return np.asarray(pred), np.asarray(prob)


def _run_binary_task(df: pd.DataFrame, feature_space: str, model_family: str) -> dict[str, float]:
    X = get_feature_space(df, feature_space).to_numpy(dtype=float)
    y = df["binary_abnormal_label"].to_numpy(dtype=int)
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred_all, score_all, true_all = [], [], []
    for tr, te in cv.split(X, y):
        model = _make_model(model_family).fit(X[tr], y[tr])
        pred, score = _binary_score(model, X[te])
        pred_all.append(pred)
        score_all.append(score)
        true_all.append(y[te])
    y_true = np.concatenate(true_all)
    y_pred = np.concatenate(pred_all)
    y_score = np.concatenate(score_all)
    metrics = _binary_metrics(y_true, y_pred, y_score)
    metrics.update({
        "task": "binary_abnormal",
        "feature_space": feature_space,
        "model_family": model_family,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_splits": int(n_splits),
    })
    return metrics


def _run_multiclass_task(df: pd.DataFrame, feature_space: str, model_family: str) -> dict[str, float]:
    X = get_feature_space(df, feature_space).to_numpy(dtype=float)
    labels = df["multiclass_condition_name"].astype(str).to_numpy()
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred_all, prob_all, true_all = [], [], []
    for tr, te in cv.split(X, y):
        model = _make_model(model_family).fit(X[tr], y[tr])
        pred, prob = _multiclass_score(model, X[te])
        pred_all.append(pred)
        prob_all.append(prob)
        true_all.append(y[te])
    y_true = np.concatenate(true_all)
    y_pred = np.concatenate(pred_all)
    y_prob = np.concatenate(prob_all, axis=0)
    metrics = _multiclass_metrics(y_true, y_pred, y_prob)
    metrics.update({
        "task": "multiclass_condition",
        "feature_space": feature_space,
        "model_family": model_family,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(enc.classes_)),
        "n_splits": int(n_splits),
    })
    return metrics


def run(input_csv: str = "outputs/05_stage5/unified_model_session_table.csv") -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_unified_session_table(input_csv)
    rows = []
    for feature_space in FEATURE_SPACES:
        for model_family in MODEL_FAMILIES:
            rows.append(_run_binary_task(df, feature_space, model_family))
            rows.append(_run_multiclass_task(df, feature_space, model_family))
    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "benchmark_summary.csv", index=False)

    binary = results[results["task"] == "binary_abnormal"].copy()
    multiclass = results[results["task"] == "multiclass_condition"].copy()
    binary.sort_values(["auc", "balanced_accuracy"], ascending=[False, False]).to_csv(
        OUTDIR / "benchmark_binary_ranked.csv", index=False
    )
    multiclass.sort_values(["macro_ovr_auc", "balanced_accuracy"], ascending=[False, False]).to_csv(
        OUTDIR / "benchmark_multiclass_ranked.csv", index=False
    )

    overview = {
        "input_csv": input_csv,
        "feature_spaces": FEATURE_SPACES,
        "model_families": MODEL_FAMILIES,
        "n_samples": int(len(df)),
        "best_binary_auc": float(binary["auc"].max()),
        "best_multiclass_macro_ovr_auc": float(multiclass["macro_ovr_auc"].max()),
        "best_binary_row": binary.sort_values(["auc", "balanced_accuracy"], ascending=[False, False]).iloc[0].to_dict(),
        "best_multiclass_row": multiclass.sort_values(["macro_ovr_auc", "balanced_accuracy"], ascending=[False, False]).iloc[0].to_dict(),
    }
    (OUTDIR / "benchmark_overview.json").write_text(json.dumps(overview, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark feature spaces and model families in the behavior method lab.")
    parser.add_argument(
        "--input",
        default="outputs/05_stage5/unified_model_session_table.csv",
        help="Session-level unified table used as the benchmark input.",
    )
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
