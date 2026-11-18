"""
Unified multiclass behavioral model on session-level features.

Uses the Stage 5 unified session table:
`outputs/05_stage5/unified_model_session_table.csv`

Primary target:
`multiclass_condition_name` with four classes:
  - control
  - social_isolation
  - surgery
  - recovery_timepoint
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages.s4_classification import BlockScaler


ROOT = Path("outputs/05_stage5")
INPUT = ROOT / "unified_model_session_table.csv"
OUTDIR = ROOT / "unified_multiclass_model"
SEED = 42
TOP_K = 12


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("freq50_")
        or c.startswith("trans_")
        or c.startswith("transsummary_")
        or c.startswith("metastate_usage_")
    ]


def _feature_group(name: str) -> str:
    if name.startswith("freq50_"):
        return "usage"
    if name.startswith("transsummary_"):
        return "transsummary"
    if name.startswith("trans_"):
        return "transition"
    if name.startswith("metastate_usage_"):
        return "metastate"
    return "other"


def _apply_filters(
    df: pd.DataFrame,
    env_contexts: list[str] | None = None,
    study_cohorts: list[str] | None = None,
    study_contexts: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if env_contexts:
        wanted = {x.strip() for x in env_contexts if x.strip()}
        out = out[out["env_context"].isin(wanted)].copy()
    if study_cohorts:
        wanted = {x.strip() for x in study_cohorts if x.strip()}
        out = out[out["study_cohort"].isin(wanted)].copy()
    if study_contexts:
        wanted = {x.strip() for x in study_contexts if x.strip()}
        out = out[out["study_context"].isin(wanted)].copy()
    return out.reset_index(drop=True)


def _make_multiclass_model(block_sizes: list[int]) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("block_scale", BlockScaler(block_sizes)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )),
    ])


def _evaluate_pooled_cv(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    feature_names: list[str],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = _make_multiclass_model([X.shape[1]])
    pred_rows: list[dict] = []
    coef_rows: list[dict] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
        fold_model = clone(model).fit(X[tr], y[tr])
        probs = fold_model.predict_proba(X[te])
        preds = fold_model.predict(X[te])
        coef = fold_model.named_steps["clf"].coef_

        for local_idx, row_idx in enumerate(te):
            row = {
                "split_type": "pooled",
                "fold": fold_idx,
                "row_index": int(row_idx),
                "y_true": int(y[row_idx]),
                "y_true_name": str(classes[y[row_idx]]),
                "y_pred": int(preds[local_idx]),
                "y_pred_name": str(classes[preds[local_idx]]),
            }
            for class_idx, class_name in enumerate(classes):
                row[f"prob_{class_name}"] = float(probs[local_idx, class_idx])
            pred_rows.append(row)

        for class_idx, class_name in enumerate(classes):
            for feat, value in zip(feature_names, coef[class_idx]):
                coef_rows.append({
                    "split_type": "pooled",
                    "fold": fold_idx,
                    "class_name": str(class_name),
                    "feature": feat,
                    "feature_group": _feature_group(feat),
                    "coefficient": float(value),
                    "abs_coefficient": float(abs(value)),
                })

    return pd.DataFrame(pred_rows), pd.DataFrame(coef_rows)


def _summary_metrics(pred_df: pd.DataFrame, classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = pred_df["y_true_name"].to_numpy()
    y_pred = pred_df["y_pred_name"].to_numpy()
    prob_cols = [f"prob_{c}" for c in classes]
    y_prob = pred_df[prob_cols].to_numpy(dtype=float)

    overall = {
        "n_samples": int(len(pred_df)),
        "n_classes": int(len(classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    lb = LabelBinarizer().fit(classes)
    y_true_bin = lb.transform(y_true)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin[:, None]
    try:
        overall["macro_ovr_auc"] = float(
            roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        overall["macro_ovr_auc"] = np.nan

    per_class_rows = []
    for class_name in classes:
        yt = (y_true == class_name).astype(int)
        yp = (y_pred == class_name).astype(int)
        col = pred_df[f"prob_{class_name}"].to_numpy(dtype=float)
        auc = np.nan
        if yt.sum() > 0 and yt.sum() < len(yt):
            auc = float(roc_auc_score(yt, col))
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        per_class_rows.append({
            "class_name": class_name,
            "n_true": int(yt.sum()),
            "n_pred": int(yp.sum()),
            "precision": precision,
            "recall": recall,
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "ovr_auc": auc,
        })

    return pd.DataFrame([overall]), pd.DataFrame(per_class_rows)


def _stable_features(coef_df: pd.DataFrame, feature_names: list[str], classes: list[str]) -> pd.DataFrame:
    rows = []
    for class_name in classes:
        sub = coef_df[coef_df["class_name"] == class_name].copy()
        n_folds = max(sub["fold"].nunique(), 1)
        inclusion = defaultdict(int)
        signed_sum = defaultdict(float)
        for _, fold_df in sub.groupby("fold"):
            top = fold_df.sort_values("abs_coefficient", ascending=False).head(TOP_K)
            for _, row in top.iterrows():
                inclusion[row["feature"]] += 1
                signed_sum[row["feature"]] += row["coefficient"]
        mean_coef = sub.groupby("feature")["coefficient"].mean()
        for feat in feature_names:
            rows.append({
                "class_name": class_name,
                "feature": feat,
                "feature_group": _feature_group(feat),
                "mean_coefficient": float(mean_coef.get(feat, np.nan)),
                "topk_inclusion": inclusion[feat] / float(n_folds),
                "mean_signed_topk_coefficient": signed_sum[feat] / float(max(inclusion[feat], 1)),
            })
    return pd.DataFrame(rows).sort_values(
        ["class_name", "topk_inclusion", "mean_coefficient"],
        ascending=[True, False, False],
    )


def _plot_outputs(
    pred_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    stable_df: pd.DataFrame,
    classes: list[str],
    outdir: Path,
    title_suffix: str = "",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    row = summary_df.iloc[0]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc"]
    vals = [row[m] for m in metrics]
    ax.bar(metrics, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Pooled multiclass performance{title_suffix}")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[0, 1]
    cm = confusion_matrix(
        pred_df["y_true_name"],
        pred_df["y_pred_name"],
        labels=classes,
        normalize="true",
    )
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_yticklabels(classes)
    ax.set_title(f"Recall-normalized confusion{title_suffix}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    plot_df = per_class_df.sort_values("ovr_auc", ascending=False)
    ax.barh(plot_df["class_name"], plot_df["ovr_auc"], color="#64B5CD")
    ax.set_xlim(0, 1)
    ax.set_xlabel("One-vs-rest AUC")
    ax.set_title(f"Per-class discrimination{title_suffix}")

    ax = axes[1, 1]
    stable_top = stable_df.groupby("class_name").head(3).copy()
    y = np.arange(len(stable_top))
    labels = [f"{r.class_name}: {r.feature}" for r in stable_top.itertuples()]
    ax.barh(y, stable_top["topk_inclusion"], color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-k inclusion")
    ax.set_title(f"Top stable features by class{title_suffix}")

    plt.tight_layout()
    fig.savefig(outdir / "multiclass_model_overview.png", dpi=170)
    plt.close(fig)


def run(
    input_csv: str = str(INPUT),
    outdir: str | Path = OUTDIR,
    env_contexts: list[str] | None = None,
    study_cohorts: list[str] | None = None,
    study_contexts: list[str] | None = None,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = _apply_filters(df, env_contexts=env_contexts, study_cohorts=study_cohorts, study_contexts=study_contexts)
    if df.empty:
        raise ValueError("No rows remain after applying filters.")

    feature_names = _feature_columns(df)
    X = df[feature_names].to_numpy(dtype=float)
    y_name = df["multiclass_condition_name"].astype(str).to_numpy()
    classes = np.array(sorted(pd.unique(y_name)))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y = np.array([class_to_idx[name] for name in y_name], dtype=int)

    rng = np.random.default_rng(SEED)
    order = np.arange(len(df))
    rng.shuffle(order)
    X = X[order]
    y = y[order]
    df = df.iloc[order].reset_index(drop=True)

    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    pred_df, coef_df = _evaluate_pooled_cv(X, y, classes, feature_names, n_splits)
    pred_df["sample_id"] = df.loc[pred_df["row_index"], "sample_id"].to_numpy()
    pred_df["study_cohort"] = df.loc[pred_df["row_index"], "study_cohort"].to_numpy()
    pred_df["study_context"] = df.loc[pred_df["row_index"], "study_context"].to_numpy()
    pred_df["env_context"] = df.loc[pred_df["row_index"], "env_context"].to_numpy()

    summary_df, per_class_df = _summary_metrics(pred_df, classes.tolist())
    summary_df["n_folds"] = n_splits
    stable_df = _stable_features(coef_df, feature_names, classes.tolist())

    cm_counts = pd.DataFrame(
        confusion_matrix(pred_df["y_true_name"], pred_df["y_pred_name"], labels=classes),
        index=classes,
        columns=classes,
    )
    cm_recall = pd.DataFrame(
        confusion_matrix(pred_df["y_true_name"], pred_df["y_pred_name"], labels=classes, normalize="true"),
        index=classes,
        columns=classes,
    )

    context_summary = (
        pred_df.groupby(["study_context", "y_true_name", "y_pred_name"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["study_context", "y_true_name", "n_samples"], ascending=[True, True, False])
    )

    pred_df.to_csv(outdir / "prediction_table.csv", index=False)
    coef_df.to_csv(outdir / "coefficient_table.csv", index=False)
    summary_df.to_csv(outdir / "pooled_summary.csv", index=False)
    per_class_df.to_csv(outdir / "per_class_summary.csv", index=False)
    stable_df.to_csv(outdir / "pooled_stable_features.csv", index=False)
    cm_counts.to_csv(outdir / "confusion_matrix_counts.csv")
    cm_recall.to_csv(outdir / "confusion_matrix_recall_normalized.csv")
    context_summary.to_csv(outdir / "study_context_prediction_summary.csv", index=False)

    overview = {
        "input_csv": input_csv,
        "outdir": str(outdir),
        "n_samples": int(len(df)),
        "n_features": int(len(feature_names)),
        "classes": classes.tolist(),
        "class_counts": df["multiclass_condition_name"].value_counts().to_dict(),
        "study_cohorts": sorted(df["study_cohort"].dropna().unique().tolist()),
        "study_contexts": sorted(df["study_context"].dropna().unique().tolist()),
        "env_contexts": sorted(df["env_context"].dropna().unique().tolist()),
        "pooled_accuracy": float(summary_df["accuracy"].iloc[0]),
        "pooled_balanced_accuracy": float(summary_df["balanced_accuracy"].iloc[0]),
        "pooled_macro_f1": float(summary_df["macro_f1"].iloc[0]),
        "pooled_macro_ovr_auc": float(summary_df["macro_ovr_auc"].iloc[0]),
        "notes": [
            "This pooled multiclass model is the primary analysis.",
            "Leave-one-cohort-out is intentionally not reported because classes are strongly confounded with cohorts.",
            "Use per-class coefficients and confusion patterns for interpretation rather than a single shared abnormal axis.",
        ],
    }
    (outdir / "overview.json").write_text(json.dumps(overview, indent=2))

    title_suffix = ""
    if env_contexts:
        title_suffix = f" ({', '.join(env_contexts)})"
    _plot_outputs(pred_df, summary_df, per_class_df, stable_df, classes.tolist(), outdir, title_suffix=title_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified multiclass condition model analysis.")
    parser.add_argument("--input", default=str(INPUT), help="Path to unified model session table CSV")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Output directory")
    parser.add_argument(
        "--env-context",
        action="append",
        default=[],
        help="Filter to one or more env_context values; repeat flag to pass multiple times",
    )
    parser.add_argument(
        "--study-cohort",
        action="append",
        default=[],
        help="Filter to one or more study_cohort values; repeat flag to pass multiple times",
    )
    parser.add_argument(
        "--study-context",
        action="append",
        default=[],
        help="Filter to one or more study_context values; repeat flag to pass multiple times",
    )
    args = parser.parse_args()
    run(
        args.input,
        outdir=args.outdir,
        env_contexts=args.env_context or None,
        study_cohorts=args.study_cohort or None,
        study_contexts=args.study_context or None,
    )


if __name__ == "__main__":
    main()
