"""
Unified normal-vs-abnormal behavioral model on session-level features.

Uses the session-level unified table exported by Stage 5:
`outputs/05_stage5/unified_model_session_table.csv`

Outputs are written to:
`outputs/05_stage5/unified_abnormal_model/`
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages.s4_classification import make_model


ROOT = Path("outputs/05_stage5")
INPUT = ROOT / "unified_model_session_table.csv"
OUTDIR = ROOT / "unified_abnormal_model"
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


def _evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    split_name: str,
    split_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = make_model([X.shape[1]])
    pred_rows: list[dict] = []
    coef_rows: list[dict] = []

    for fold_idx, (tr, te) in enumerate(splits):
        fold_model = clone(model).fit(X[tr], y[tr])
        prob = fold_model.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        coef = fold_model.named_steps["clf"].coef_[0]
        split_label = split_labels[fold_idx] if split_labels is not None else f"{split_name}_{fold_idx}"

        for ix, prob_i, pred_i in zip(te, prob, pred):
            pred_rows.append({
                "split_type": split_name,
                "fold": fold_idx,
                "held_out": split_label,
                "row_index": int(ix),
                "y_true": int(y[ix]),
                "y_prob": float(prob_i),
                "y_pred": int(pred_i),
            })

        for name, value in zip(feature_names, coef):
            coef_rows.append({
                "split_type": split_name,
                "fold": fold_idx,
                "held_out": split_label,
                "feature": name,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
                "feature_group": _feature_group(name),
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(coef_rows)


def _summarize_predictions(
    pred_df: pd.DataFrame,
    by_col: str,
) -> pd.DataFrame:
    rows = []
    for key, sub in pred_df.groupby(by_col):
        auc = np.nan
        if sub["y_true"].nunique() >= 2:
            auc = float(roc_auc_score(sub["y_true"], sub["y_prob"]))
        rows.append({
            by_col: key,
            "n_samples": int(len(sub)),
            "n_normal": int((sub["y_true"] == 0).sum()),
            "n_abnormal": int((sub["y_true"] == 1).sum()),
            "auc": auc,
            "accuracy": float(accuracy_score(sub["y_true"], sub["y_pred"])),
        })
    return pd.DataFrame(rows)


def _stable_features(
    coef_df: pd.DataFrame,
    feature_names: list[str],
    split_type: str,
) -> pd.DataFrame:
    sub = coef_df[coef_df["split_type"] == split_type].copy()
    n_folds = max(sub["fold"].nunique(), 1)
    inclusion = defaultdict(int)
    signed_sum = defaultdict(float)

    for _, fold_df in sub.groupby("fold"):
        top = fold_df.sort_values("abs_coefficient", ascending=False).head(TOP_K)
        for _, row in top.iterrows():
            inclusion[row["feature"]] += 1
            signed_sum[row["feature"]] += row["coefficient"]

    rows = []
    mean_coef = sub.groupby("feature")["coefficient"].mean()
    for feat in feature_names:
        rows.append({
            "split_type": split_type,
            "feature": feat,
            "feature_group": _feature_group(feat),
            "mean_coefficient": float(mean_coef.get(feat, np.nan)),
            "topk_inclusion": inclusion[feat] / float(n_folds),
            "mean_signed_topk_coefficient": signed_sum[feat] / float(max(inclusion[feat], 1)),
        })
    return pd.DataFrame(rows).sort_values(
        ["topk_inclusion", "mean_coefficient"],
        ascending=[False, False],
    )


def _plot_overview(
    pooled_summary: pd.DataFrame,
    losg_summary: pd.DataFrame,
    loco_summary: pd.DataFrame,
    stable_df: pd.DataFrame,
    outdir: Path,
    title_suffix: str = "",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if not pooled_summary.empty:
        row = pooled_summary.iloc[0]
        ax.bar(["pooled"], [row["auc"]], color="#4C72B0")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC")
        ax.set_title(f"Pooled 5-fold performance{title_suffix}")
        ax.text(0, row["auc"] + 0.03, f"{row['auc']:.2f}", ha="center")
    else:
        ax.set_axis_off()

    ax = axes[0, 1]
    if not losg_summary.empty:
        plot_df = losg_summary.sort_values("auc", ascending=False)
        ax.barh(plot_df["held_out"], plot_df["auc"], color="#55A868")
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC")
        ax.set_title(f"Leave-one-study-group-out{title_suffix}")
    else:
        ax.set_axis_off()

    ax = axes[1, 0]
    if not loco_summary.empty:
        plot_df = loco_summary.sort_values("auc", ascending=False)
        ax.barh(plot_df["held_out"], plot_df["auc"], color="#C44E52")
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC")
        ax.set_title(f"Leave-one-cohort-out{title_suffix}")
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    stable_top = stable_df.head(10).copy()
    if not stable_top.empty:
        y = np.arange(len(stable_top))
        ax.barh(y, stable_top["topk_inclusion"], color="#8172B3")
        ax.set_yticks(y)
        ax.set_yticklabels(stable_top["feature"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Top-k inclusion")
        ax.set_title(f"Most stable LOSG features{title_suffix}")
    else:
        ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(outdir / "unified_model_overview.png", dpi=170)
    plt.close(fig)


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
    y = df["binary_abnormal_label"].to_numpy(dtype=int)

    rng = np.random.default_rng(SEED)
    order = np.arange(len(df))
    rng.shuffle(order)
    X = X[order]
    y = y[order]
    df = df.iloc[order].reset_index(drop=True)

    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    pooled_cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X, y))
    pooled_pred, pooled_coef = _evaluate_cv(X, y, pooled_cv, feature_names, "pooled")
    pooled_summary = _summarize_predictions(pooled_pred, "split_type")
    pooled_summary["n_folds"] = n_splits

    losg_splits = []
    losg_labels = []
    for held_out, sub in df.groupby("leave_one_study_group"):
        te = sub.index.to_numpy()
        tr = np.setdiff1d(np.arange(len(df)), te)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        losg_splits.append((tr, te))
        losg_labels.append(str(held_out))
    losg_pred, losg_coef = _evaluate_cv(X, y, losg_splits, feature_names, "leave_one_study_group", losg_labels)
    losg_summary = _summarize_predictions(losg_pred, "held_out")

    loco_splits = []
    loco_labels = []
    for held_out, sub in df.groupby("leave_one_cohort"):
        te = sub.index.to_numpy()
        tr = np.setdiff1d(np.arange(len(df)), te)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        loco_splits.append((tr, te))
        loco_labels.append(str(held_out))
    loco_pred, loco_coef = _evaluate_cv(X, y, loco_splits, feature_names, "leave_one_cohort", loco_labels)
    loco_summary = _summarize_predictions(loco_pred, "held_out")

    pred_df = pd.concat([pooled_pred, losg_pred, loco_pred], ignore_index=True)
    coef_df = pd.concat([pooled_coef, losg_coef, loco_coef], ignore_index=True)
    pred_df["sample_id"] = df.loc[pred_df["row_index"], "sample_id"].to_numpy()
    pred_df["study_cohort"] = df.loc[pred_df["row_index"], "study_cohort"].to_numpy()
    pred_df["study_context"] = df.loc[pred_df["row_index"], "study_context"].to_numpy()
    pred_df["abnormal_condition"] = df.loc[pred_df["row_index"], "abnormal_condition"].to_numpy()

    losg_stable = _stable_features(coef_df, feature_names, "leave_one_study_group")
    loco_stable = _stable_features(coef_df, feature_names, "leave_one_cohort")
    pooled_stable = _stable_features(coef_df, feature_names, "pooled")

    pred_df.to_csv(outdir / "prediction_table.csv", index=False)
    coef_df.to_csv(outdir / "coefficient_table.csv", index=False)
    pooled_summary.to_csv(outdir / "pooled_summary.csv", index=False)
    losg_summary.to_csv(outdir / "leave_one_study_group_summary.csv", index=False)
    loco_summary.to_csv(outdir / "leave_one_cohort_summary.csv", index=False)
    pooled_stable.to_csv(outdir / "pooled_stable_features.csv", index=False)
    losg_stable.to_csv(outdir / "leave_one_study_group_stable_features.csv", index=False)
    loco_stable.to_csv(outdir / "leave_one_cohort_stable_features.csv", index=False)

    overview = {
        "input_csv": input_csv,
        "outdir": str(outdir),
        "n_samples": int(len(df)),
        "n_features": int(len(feature_names)),
        "binary_counts": df["binary_abnormal_name"].value_counts().to_dict(),
        "env_contexts": sorted(df["env_context"].dropna().unique().tolist()),
        "study_cohorts": sorted(df["study_cohort"].dropna().unique().tolist()),
        "study_contexts": sorted(df["study_context"].dropna().unique().tolist()),
        "pooled_auc": float(pooled_summary["auc"].iloc[0]) if not pooled_summary.empty else None,
        "mean_losg_auc": float(losg_summary["auc"].mean()) if not losg_summary.empty else None,
        "mean_loco_auc": float(loco_summary["auc"].mean()) if not loco_summary.empty else None,
    }
    (outdir / "overview.json").write_text(json.dumps(overview, indent=2))
    title_suffix = ""
    if env_contexts:
        title_suffix = f" ({', '.join(env_contexts)})"
    _plot_overview(pooled_summary, losg_summary, loco_summary, losg_stable, outdir, title_suffix=title_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified abnormal-vs-normal model analysis.")
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
