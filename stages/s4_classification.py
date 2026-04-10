"""
Stage 4 – Syllable-based behaviour classification.
===================================================
Compares all feature engineering setups across three analyses:

  A. HC vs PS classification (joint surgery model syllables)
     – 8 feature setups × 4 cohort/test combinations
     – 5-fold stratified CV, LogReg + SVM-linear

  B. Fiber photometry: Control vs Social Isolation
     – joint-model syllables if results_fiber.h5 exists, else fiber-own
     – LOO-CV (n=16), balanced class weights

  C. Reverse: HC vs PS using fiber-model syllables
     – surgery sessions inferred through fiber model (results_surgery.h5)
     – 5-fold CV

Outputs saved to: outputs/04_classification/

Usage:
    python pipeline.py run --stages 4
    python stages/s4_classification.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

from utils.io_utils import load_config, resolve_path, load_all_preprocessed
from stages.s3_moseq_fitting import _filter_to_use_bodyparts

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

JOINT_PROJECT   = "outputs/03_moseq/kpms_project_surgery_joint"
JOINT_MODEL     = "model_surgery_joint"
JOINT_H5        = f"{JOINT_PROJECT}/{JOINT_MODEL}/results.h5"
FIBER_PROJECT   = "outputs/03_moseq/kpms_project_fiber_photometry"
FIBER_MODEL     = "model_v01"
TAKAYA_META_CSV = "Data/sugery_behavior test (Takaya Ishii)_Gen/video_metadata.csv"
PREPROCESSED    = "outputs/02_preprocessed"
NUM_STATES      = 50
OUT_DIR         = Path("outputs/04_classification")
OUT_DIR.mkdir(parents=True, exist_ok=True)
USE_BODYPARTS   = ["nose", "left_ear", "right_ear", "neck",
                   "mid_back", "mouse_center", "tail_base"]

LAPLACE_PSEUDOCOUNT = 0.1


# ── Model application (prerequisites for classification) ─────────────────────

def _apply_joint_to_fiber() -> None:
    """Apply joint surgery model to fiber sessions → results_fiber.h5."""
    results_path = f"{JOINT_PROJECT}/{JOINT_MODEL}/results_fiber.h5"
    if Path(results_path).exists():
        logger.info("results_fiber.h5 already exists – skipping apply_joint_to_fiber.")
        return

    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    from jax_moseq.utils.debugging import convert_data_precision

    logger.info("Applying joint surgery model to fiber sessions ...")
    coords, confs, bodyparts = load_all_preprocessed(PREPROCESSED)
    fiber_keys = [k for k in coords if k.startswith("fiberphotometry")]
    logger.info("  Fiber sessions: %d", len(fiber_keys))

    fiber_coords, fiber_confs = _filter_to_use_bodyparts(
        {k: coords[k] for k in fiber_keys},
        {k: confs[k]  for k in fiber_keys},
        bodyparts, USE_BODYPARTS,
    )
    data, metadata = kpms.format_data(fiber_coords, confidences=fiber_confs,
                                      project_dir=JOINT_PROJECT)
    data = convert_data_precision(data)
    ckpt = kpms.load_checkpoint(JOINT_PROJECT, JOINT_MODEL)
    model = ckpt[0] if isinstance(ckpt, tuple) else ckpt

    kpms.apply_model(
        model=model, data=data, metadata=metadata,
        project_dir=JOINT_PROJECT, model_name=JOINT_MODEL,
        num_iters=10, results_path=results_path, overwrite=True,
        parallel_message_passing=False, verbose=True,
        anterior_idxs=[USE_BODYPARTS.index("nose")],
        posterior_idxs=[USE_BODYPARTS.index("tail_base")],
        error_estimator={"slope": -0.5, "intercept": 0.25},
    )
    logger.info("  Saved to %s", results_path)


def _apply_fiber_to_surgery() -> None:
    """Apply fiber model to surgery sessions → results_surgery.h5."""
    results_path = f"{FIBER_PROJECT}/{FIBER_MODEL}/results_surgery.h5"
    if Path(results_path).exists():
        logger.info("results_surgery.h5 already exists – skipping apply_fiber_to_surgery.")
        return

    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    from jax_moseq.utils.debugging import convert_data_precision

    logger.info("Applying fiber model to surgery sessions ...")
    coords, confs, bodyparts = load_all_preprocessed(PREPROCESSED)
    surgery_keys = [k for k in coords if not k.startswith("fiberphotometry")]
    logger.info("  Surgery sessions: %d", len(surgery_keys))

    surg_coords, surg_confs = _filter_to_use_bodyparts(
        {k: coords[k] for k in surgery_keys},
        {k: confs[k]  for k in surgery_keys},
        bodyparts, USE_BODYPARTS,
    )
    data, metadata = kpms.format_data(surg_coords, confidences=surg_confs,
                                      project_dir=FIBER_PROJECT)
    data = convert_data_precision(data)
    ckpt = kpms.load_checkpoint(FIBER_PROJECT, FIBER_MODEL)
    model = ckpt[0] if isinstance(ckpt, tuple) else ckpt

    kpms.apply_model(
        model=model, data=data, metadata=metadata,
        project_dir=FIBER_PROJECT, model_name=FIBER_MODEL,
        num_iters=10, results_path=results_path, overwrite=True,
        parallel_message_passing=False, verbose=True,
        anterior_idxs=[USE_BODYPARTS.index("nose")],
        posterior_idxs=[USE_BODYPARTS.index("tail_base")],
        error_estimator={"slope": -0.5, "intercept": 0.25},
    )
    logger.info("  Saved to %s", results_path)


# ── Setup definitions ─────────────────────────────────────────────────────────
# Each setup is a dict with:
#   name        – short label
#   usage       – "all50" | "top5" | "top10" | "freq50"
#   trans       – None | "top5" | "top10" | "freq50"  (same syllables as usage if matching)
SETUPS = [
    {"name": "all50_usage",        "usage": "all50",  "trans": None},
    {"name": "top5_usage",         "usage": "top5",   "trans": None},
    {"name": "top10_usage",        "usage": "top10",  "trans": None},
    {"name": "top5_usage+trans",   "usage": "top5",   "trans": "top5"},
    {"name": "top10_usage+trans",  "usage": "top10",  "trans": "top10"},
    {"name": "all50+top5_trans",   "usage": "all50",  "trans": "top5"},
    {"name": "all50+top10_trans",  "usage": "all50",  "trans": "top10"},
    {"name": "freq50_usage+trans", "usage": "freq50", "trans": "freq50"},
]


# ── Metadata builders ─────────────────────────────────────────────────────────

def _parse_surgery_15mice_meta(sessions: list[str]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        timepoint = "pre" if s.startswith("POD-1") else "post"
        parts = s.split()
        test_type = parts[1] if len(parts) > 1 else "unknown"
        if len(parts) > 2 and parts[1] == "Y" and parts[2].lower().startswith("maze"):
            test_type = "Y maze"
        m = re.search(r"_mouse(\d+)$", s)
        mouse_id = int(m.group(1)) if m else None
        rows.append({"session": s, "timepoint": timepoint,
                     "test_type": test_type, "mouse_id": mouse_id,
                     "cohort": "surgery_15mice"})
    return pd.DataFrame(rows)


def _parse_takaya_meta(sessions: list[str]) -> pd.DataFrame:
    meta_csv = pd.read_csv(TAKAYA_META_CSV)
    lookup = meta_csv.set_index("video_stem")
    rows = []
    for s in sessions:
        stem = s.split("_superanimal")[0]
        if stem in lookup.index:
            row = lookup.loc[stem]
            rows.append({"session": s, "timepoint": row["timepoint"],
                         "test_type": row["test_type"], "mouse_id": row["mouse_id"],
                         "group": row.get("group", None), "cohort": "takaya_surgery"})
        else:
            rows.append({"session": s, "timepoint": None, "test_type": None,
                         "mouse_id": None, "cohort": "takaya_surgery"})
    return pd.DataFrame(rows)


def build_metadata() -> pd.DataFrame:
    with h5py.File(JOINT_H5, "r") as f:
        all_sessions = list(f.keys())
    s15_sessions = [s for s in all_sessions if "_mouse" in s]
    tak_sessions = [s for s in all_sessions if "_mouse" not in s]
    df = pd.concat([_parse_surgery_15mice_meta(s15_sessions),
                    _parse_takaya_meta(tak_sessions)], ignore_index=True)
    df = df[df["test_type"].notna()].copy()
    print(f"\nMetadata built: {len(df)} sessions")
    print(df.groupby(["cohort", "test_type", "timepoint"]).size().to_string())
    return df


# ── Syllable selectors ────────────────────────────────────────────────────────

def load_usage_full(sessions: list[str]) -> np.ndarray:
    rows = []
    with h5py.File(JOINT_H5, "r") as f:
        for s in sessions:
            seq    = f[s]["syllable"][:].astype(int)
            counts = np.bincount(seq, minlength=NUM_STATES)[:NUM_STATES]
            rows.append(counts / counts.sum())
    return np.array(rows)


def select_top_n(X_full: np.ndarray, n: int) -> np.ndarray:
    return np.sort(np.argsort(X_full.mean(axis=0))[::-1][:n])


def select_freq(X_full: np.ndarray, min_freq: float = 0.50) -> np.ndarray:
    present = (X_full > 0).mean(axis=0)
    return np.where(present >= min_freq)[0]


def resolve_sylls(spec: str | None, X_full: np.ndarray) -> np.ndarray | None:
    if spec is None:
        return None
    if spec == "all50":
        return np.arange(NUM_STATES)
    if spec == "top5":
        return select_top_n(X_full, 5)
    if spec == "top10":
        return select_top_n(X_full, 10)
    if spec == "freq50":
        return select_freq(X_full, 0.50)
    raise ValueError(f"Unknown spec: {spec!r}")


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(sessions: list[str],
                   usage_sylls: np.ndarray,
                   trans_sylls: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    """
    usage_sylls : syllable indices for usage features
    trans_sylls : syllable indices for transition submatrix (None = no transitions)
    Returns (X, feature_names).
    """
    n_u = len(usage_sylls)
    n_t = len(trans_sylls) if trans_sylls is not None else 0
    rows = []

    with h5py.File(JOINT_H5, "r") as f:
        for sess in sessions:
            seq = f[sess]["syllable"][:].astype(int)
            T   = len(seq)

            counts = np.bincount(seq, minlength=NUM_STATES)[:NUM_STATES]
            u_vec  = (counts / T)[usage_sylls]

            if trans_sylls is not None:
                trans = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
                for t in range(T - 1):
                    trans[seq[t], seq[t + 1]] += 1
                t_vec = np.zeros(n_t * n_t)
                for i, si in enumerate(trans_sylls):
                    row_counts = trans[si, trans_sylls] + LAPLACE_PSEUDOCOUNT
                    row_total  = trans[si].sum() + LAPLACE_PSEUDOCOUNT * n_t
                    t_vec[i * n_t:(i + 1) * n_t] = row_counts / row_total
                rows.append(np.concatenate([u_vec, t_vec]))
            else:
                rows.append(u_vec)

    feat_names = [f"usage_{s}" for s in usage_sylls]
    if trans_sylls is not None:
        feat_names += [f"P({si}→{sj})" for si in trans_sylls for sj in trans_sylls]

    return np.array(rows), feat_names


# ── CV helper ─────────────────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for clf_name, clf in [
        ("LogReg",     LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        ("SVM-linear", SVC(kernel="linear", probability=True, random_state=42)),
    ]:
        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        results[clf_name] = {"auc_mean": auc.mean(), "auc_std": auc.std(),
                             "acc_mean": acc.mean(), "acc_std": acc.std()}
    return results


# ── Per-setup within-cohort run ───────────────────────────────────────────────

def run_setup(sessions: list[str], y: np.ndarray, setup: dict) -> dict | None:
    X_full = load_usage_full(sessions)

    usage_sylls = resolve_sylls(setup["usage"], X_full)
    trans_sylls = resolve_sylls(setup["trans"], X_full)

    if usage_sylls is None or len(usage_sylls) == 0:
        return None

    X_raw, feat_names = build_features(sessions, usage_sylls, trans_sylls)
    n_u = len(usage_sylls)
    n_t = len(trans_sylls) if trans_sylls is not None else 0
    print(f"      [{setup['name']}]  feats={X_raw.shape[1]}  "
          f"({n_u} usage + {n_t**2 if n_t else 0} trans)")

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    cv_res = run_cv(X, y)

    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(X, y)

    coef_df = pd.DataFrame({"feature": feat_names, "coef": lr.coef_[0],
                             "mean_HC": X_raw[y == 0].mean(0),
                             "mean_PS": X_raw[y == 1].mean(0)})
    coef_df["delta"] = coef_df["mean_PS"] - coef_df["mean_HC"]

    lr_auc = cv_res["LogReg"]["auc_mean"]
    svm_auc = cv_res["SVM-linear"]["auc_mean"]
    print(f"        LogReg AUC={lr_auc:.3f}  SVM AUC={svm_auc:.3f}")

    return {"cv": cv_res, "coef_df": coef_df,
            "usage_sylls": usage_sylls, "trans_sylls": trans_sylls,
            "n_feat": X_raw.shape[1]}


# ── Main cohort loop ──────────────────────────────────────────────────────────

COMBOS = [
    ("takaya_surgery", "OFT"),
    ("takaya_surgery", "Y maze"),
    ("surgery_15mice", "OFT"),
    ("surgery_15mice", "Y maze"),
]

COMBO_LABELS = {
    ("takaya_surgery", "OFT"):    "Takaya OFT",
    ("takaya_surgery", "Y maze"): "Takaya Y maze",
    ("surgery_15mice", "OFT"):    "S15 OFT",
    ("surgery_15mice", "Y maze"): "S15 Y maze",
}


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for cohort, test_type in COMBOS:
        label = COMBO_LABELS[(cohort, test_type)]
        sub = df[(df["cohort"] == cohort) & (df["test_type"] == test_type)].copy()
        if sub["timepoint"].nunique() < 2 or len(sub) < 6:
            continue
        sessions = sub["session"].tolist()
        y = (sub["timepoint"] == "post").astype(int).values
        print(f"\n{'='*60}")
        print(f"  {label}  (n={len(y)}, HC={(y==0).sum()}, PS={(y==1).sum()})")
        print(f"{'='*60}")

        for setup in SETUPS:
            res = run_setup(sessions, y, setup)
            if res is None:
                continue
            records.append({
                "cohort_test": label,
                "setup":       setup["name"],
                "n_feat":      res["n_feat"],
                "LogReg_AUC":  res["cv"]["LogReg"]["auc_mean"],
                "LogReg_std":  res["cv"]["LogReg"]["auc_std"],
                "SVM_AUC":     res["cv"]["SVM-linear"]["auc_mean"],
                "SVM_std":     res["cv"]["SVM-linear"]["auc_std"],
            })

    return pd.DataFrame(records)


# ── Summary table + heatmap ───────────────────────────────────────────────────

def plot_summary(df_res: pd.DataFrame) -> None:
    # ── Pivot table ──────────────────────────────────────────────────────────
    pivot = df_res.pivot(index="setup", columns="cohort_test", values="LogReg_AUC")
    # Keep setup order
    setup_order = [s["name"] for s in SETUPS]
    col_order   = [COMBO_LABELS[c] for c in COMBOS]
    pivot = pivot.reindex(index=setup_order, columns=col_order)

    pivot_svm = df_res.pivot(index="setup", columns="cohort_test", values="SVM_AUC")
    pivot_svm = pivot_svm.reindex(index=setup_order, columns=col_order)

    print("\n\n=== LogReg AUC summary ===")
    print(pivot.round(3).to_string())
    print("\n\n=== SVM AUC summary ===")
    print(pivot_svm.round(3).to_string())

    # Mean across cohorts
    pivot["Mean"] = pivot.mean(axis=1)
    pivot_svm["Mean"] = pivot_svm.mean(axis=1)
    print("\n\n=== Mean AUC (LogReg | SVM) ===")
    combined = pd.DataFrame({
        "LogReg": pivot["Mean"],
        "SVM":    pivot_svm["Mean"],
        "Avg":    (pivot["Mean"] + pivot_svm["Mean"]) / 2,
    }).reindex(setup_order)
    print(combined.round(3).to_string())

    df_res.to_csv(OUT_DIR / "all_setups_results.csv", index=False)
    print(f"\n  → {OUT_DIR / 'all_setups_results.csv'}")

    # ── Side-by-side heatmaps (LogReg | SVM) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, piv, title in zip(axes,
                               [pivot.drop(columns="Mean"), pivot_svm.drop(columns="Mean")],
                               ["LogReg AUC", "SVM AUC"]):
        data = piv.values.astype(float)
        im = ax.imshow(data, vmin=0.4, vmax=0.95, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="AUC (5-fold CV)")
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, fontsize=11)
        ax.set_yticks(range(len(setup_order)))
        ax.set_yticklabels(setup_order, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black" if 0.45 < v < 0.85 else "white")
        ax.set_title(f"HC vs PS — {title}\n(Joint surgery model, 5-fold CV)", fontsize=12)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "all_setups_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'all_setups_heatmap.png'}")

    # ── Bar chart (LogReg + SVM side-by-side per setup, one panel per cohort) ──
    fig, axes = plt.subplots(1, len(col_order), figsize=(16, 5), sharey=True)
    colors_lr  = "#4C72B0"
    colors_svm = "#C44E52"

    for ax, col in zip(axes, col_order):
        sub = df_res[df_res["cohort_test"] == col].set_index("setup").reindex(setup_order)
        x = np.arange(len(setup_order))
        w = 0.35
        bars1 = ax.bar(x - w/2, sub["LogReg_AUC"].fillna(0), w,
                       yerr=sub["LogReg_std"].fillna(0), capsize=2,
                       color=colors_lr, alpha=0.85, label="LogReg")
        bars2 = ax.bar(x + w/2, sub["SVM_AUC"].fillna(0), w,
                       yerr=sub["SVM_std"].fillna(0), capsize=2,
                       color=colors_svm, alpha=0.85, label="SVM")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_title(col, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(setup_order, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.2, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("AUC (5-fold CV)")
            ax.legend(fontsize=8)

    fig.suptitle("HC vs PS classification — all feature setups", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "all_setups_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'all_setups_bars.png'}")


# ── Fiber photometry classification ──────────────────────────────────────────

FIBER_H5_JOINT   = "outputs/03_moseq/kpms_project_surgery_joint/model_surgery_joint/results_fiber.h5"
FIBER_H5_OWN     = "outputs/03_moseq/kpms_project_fiber_photometry/model_v01/results.h5"
SURGERY_H5_FIBER = "outputs/03_moseq/kpms_project_fiber_photometry/model_v01/results_surgery.h5"
FIBER_DIR        = "Data/Fiber photometry"
FIBER_NUM_STATES = 50

# Use joint-model inference if available (same syllable space as surgery),
# otherwise fall back to fiber-specific model.
FIBER_H5 = FIBER_H5_JOINT if Path(FIBER_H5_JOINT).exists() else FIBER_H5_OWN


def build_fiber_metadata() -> pd.DataFrame:
    """Parse experiment/condition from folder structure."""
    import os
    path_map = {}
    for root, dirs, files in os.walk(FIBER_DIR):
        parts = root.replace(FIBER_DIR + "/", "").split("/")
        if len(parts) >= 4:
            experiment = parts[0]
            condition  = parts[1]
            session_folder = parts[3]
            for sub in os.listdir(root):
                key = f"{session_folder}_{sub}"
                path_map[key] = (experiment, condition)

    with h5py.File(FIBER_H5, "r") as f:
        sessions = list(f.keys())

    rows = []
    for s in sessions:
        key  = "_".join(s.split("_")[:2])
        meta = path_map.get(key, (None, None))
        rows.append({"session": s, "experiment": meta[0], "condition": meta[1]})

    df = pd.DataFrame(rows)
    print(f"\nFiber metadata: {len(df)} sessions")
    print(df.groupby(["experiment", "condition"]).size().to_string())
    return df


def load_fiber_usage_full(sessions: list[str]) -> np.ndarray:
    rows = []
    with h5py.File(FIBER_H5, "r") as f:
        for s in sessions:
            seq    = f[s]["syllable"][:].astype(int)
            counts = np.bincount(seq, minlength=FIBER_NUM_STATES)[:FIBER_NUM_STATES]
            rows.append(counts / counts.sum())
    return np.array(rows)


def build_fiber_features(sessions: list[str],
                         usage_sylls: np.ndarray,
                         trans_sylls: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    n_t = len(trans_sylls) if trans_sylls is not None else 0
    rows = []
    with h5py.File(FIBER_H5, "r") as f:
        for sess in sessions:
            seq    = f[sess]["syllable"][:].astype(int)
            T      = len(seq)
            counts = np.bincount(seq, minlength=FIBER_NUM_STATES)[:FIBER_NUM_STATES]
            u_vec  = (counts / T)[usage_sylls]

            if trans_sylls is not None:
                trans = np.zeros((FIBER_NUM_STATES, FIBER_NUM_STATES), dtype=np.float64)
                for t in range(T - 1):
                    trans[seq[t], seq[t + 1]] += 1
                t_vec = np.zeros(n_t * n_t)
                for i, si in enumerate(trans_sylls):
                    row_counts = trans[si, trans_sylls] + LAPLACE_PSEUDOCOUNT
                    row_total  = trans[si].sum() + LAPLACE_PSEUDOCOUNT * n_t
                    t_vec[i * n_t:(i + 1) * n_t] = row_counts / row_total
                rows.append(np.concatenate([u_vec, t_vec]))
            else:
                rows.append(u_vec)

    feat_names = [f"usage_{s}" for s in usage_sylls]
    if trans_sylls is not None:
        feat_names += [f"P({si}→{sj})" for si in trans_sylls for sj in trans_sylls]
    return np.array(rows), feat_names


def run_fiber_setup(sessions: list[str], y: np.ndarray, setup: dict) -> dict | None:
    X_full = load_fiber_usage_full(sessions)

    # resolve syllable selectors using fiber usage matrix
    usage_sylls = resolve_sylls(setup["usage"], X_full)
    trans_sylls = resolve_sylls(setup["trans"], X_full)

    if usage_sylls is None or len(usage_sylls) == 0:
        return None

    X_raw, feat_names = build_fiber_features(sessions, usage_sylls, trans_sylls)
    n_u = len(usage_sylls)
    n_t = len(trans_sylls) if trans_sylls is not None else 0
    print(f"      [{setup['name']}]  feats={X_raw.shape[1]}  "
          f"({n_u} usage + {n_t**2 if n_t else 0} trans)")

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # LOO CV (n=16 is too small for 5-fold; balanced weights for 12:4 imbalance)
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
    loo = LeaveOneOut()

    results = {}
    for clf_name, clf in [
        ("LogReg",     LogisticRegression(C=1.0, max_iter=2000, random_state=42,
                                          class_weight="balanced")),
        ("SVM-linear", SVC(kernel="linear", probability=True, random_state=42,
                           class_weight="balanced")),
    ]:
        y_prob = np.zeros(len(y))
        for train_idx, test_idx in loo.split(X):
            clf.fit(X[train_idx], y[train_idx])
            y_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y, y_prob)
        acc = (y_prob.round() == y).mean()
        results[clf_name] = {"auc_mean": auc, "auc_std": 0.0,
                             "acc_mean": acc, "acc_std": 0.0}

    lr_auc  = results["LogReg"]["auc_mean"]
    svm_auc = results["SVM-linear"]["auc_mean"]
    print(f"        LogReg AUC={lr_auc:.3f}  SVM AUC={svm_auc:.3f}")
    return {"cv": results, "n_feat": X_raw.shape[1]}


def run_fiber_all(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for experiment in ["AST", "Food intake during habituation"]:
        sub = df[df["experiment"] == experiment].copy()
        if sub["condition"].nunique() < 2 or len(sub) < 4:
            continue
        sessions = sub["session"].tolist()
        y = (sub["condition"] == "Social isolation").astype(int).values
        n_ctrl = (y == 0).sum(); n_si = (y == 1).sum()
        print(f"\n{'='*60}")
        print(f"  Fiber {experiment}  (n={len(y)}, Control={n_ctrl}, SocialIso={n_si})")
        print(f"{'='*60}")

        for setup in SETUPS:
            res = run_fiber_setup(sessions, y, setup)
            if res is None:
                continue
            records.append({
                "experiment":  experiment,
                "setup":       setup["name"],
                "n_feat":      res["n_feat"],
                "LogReg_AUC":  res["cv"]["LogReg"]["auc_mean"],
                "SVM_AUC":     res["cv"]["SVM-linear"]["auc_mean"],
            })

    return pd.DataFrame(records)


def build_reverse_features(sessions: list[str],
                           usage_sylls: np.ndarray,
                           trans_sylls: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    """Same as build_features but reads from SURGERY_H5_FIBER (fiber-space syllables)."""
    n_t = len(trans_sylls) if trans_sylls is not None else 0
    rows = []
    with h5py.File(SURGERY_H5_FIBER, "r") as f:
        for sess in sessions:
            seq    = f[sess]["syllable"][:].astype(int)
            T      = len(seq)
            counts = np.bincount(seq, minlength=FIBER_NUM_STATES)[:FIBER_NUM_STATES]
            u_vec  = (counts / T)[usage_sylls]
            if trans_sylls is not None:
                trans = np.zeros((FIBER_NUM_STATES, FIBER_NUM_STATES), dtype=np.float64)
                for t in range(T - 1):
                    trans[seq[t], seq[t + 1]] += 1
                t_vec = np.zeros(n_t * n_t)
                for i, si in enumerate(trans_sylls):
                    row_counts = trans[si, trans_sylls] + LAPLACE_PSEUDOCOUNT
                    row_total  = trans[si].sum() + LAPLACE_PSEUDOCOUNT * n_t
                    t_vec[i * n_t:(i + 1) * n_t] = row_counts / row_total
                rows.append(np.concatenate([u_vec, t_vec]))
            else:
                rows.append(u_vec)
    feat_names = [f"usage_{s}" for s in usage_sylls]
    if trans_sylls is not None:
        feat_names += [f"P({si}→{sj})" for si in trans_sylls for sj in trans_sylls]
    return np.array(rows), feat_names


def load_reverse_usage_full(sessions: list[str]) -> np.ndarray:
    rows = []
    with h5py.File(SURGERY_H5_FIBER, "r") as f:
        for s in sessions:
            seq    = f[s]["syllable"][:].astype(int)
            counts = np.bincount(seq, minlength=FIBER_NUM_STATES)[:FIBER_NUM_STATES]
            rows.append(counts / counts.sum())
    return np.array(rows)


def run_reverse_setup(sessions: list[str], y: np.ndarray, setup: dict) -> dict | None:
    X_full = load_reverse_usage_full(sessions)
    usage_sylls = resolve_sylls(setup["usage"], X_full)
    trans_sylls = resolve_sylls(setup["trans"], X_full)
    if usage_sylls is None or len(usage_sylls) == 0:
        return None
    X_raw, feat_names = build_reverse_features(sessions, usage_sylls, trans_sylls)
    n_u = len(usage_sylls)
    n_t = len(trans_sylls) if trans_sylls is not None else 0
    print(f"      [{setup['name']}]  feats={X_raw.shape[1]}  "
          f"({n_u} usage + {n_t**2 if n_t else 0} trans)")
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    cv_res = run_cv(X, y)
    lr_auc  = cv_res["LogReg"]["auc_mean"]
    svm_auc = cv_res["SVM-linear"]["auc_mean"]
    print(f"        LogReg AUC={lr_auc:.3f}  SVM AUC={svm_auc:.3f}")
    return {"cv": cv_res, "n_feat": X_raw.shape[1]}


def run_reverse_all(df: pd.DataFrame) -> pd.DataFrame:
    """HC vs PS classification using fiber-model syllables (5-fold CV, same as surgery analysis)."""
    records = []
    # Check which surgery sessions are in the fiber-inferred h5
    with h5py.File(SURGERY_H5_FIBER, "r") as f:
        available = set(f.keys())

    for cohort, test_type in COMBOS:
        label = COMBO_LABELS[(cohort, test_type)]
        sub = df[(df["cohort"] == cohort) & (df["test_type"] == test_type)].copy()
        sub = sub[sub["session"].isin(available)]
        if sub["timepoint"].nunique() < 2 or len(sub) < 6:
            print(f"\n  [{label}] skipped — not enough sessions in fiber h5")
            continue
        sessions = sub["session"].tolist()
        y = (sub["timepoint"] == "post").astype(int).values
        print(f"\n{'='*60}")
        print(f"  {label} [fiber-space]  (n={len(y)}, HC={(y==0).sum()}, PS={(y==1).sum()})")
        print(f"{'='*60}")
        for setup in SETUPS:
            res = run_reverse_setup(sessions, y, setup)
            if res is None:
                continue
            records.append({
                "cohort_test": label,
                "setup":       setup["name"],
                "n_feat":      res["n_feat"],
                "LogReg_AUC":  res["cv"]["LogReg"]["auc_mean"],
                "LogReg_std":  res["cv"]["LogReg"]["auc_std"],
                "SVM_AUC":     res["cv"]["SVM-linear"]["auc_mean"],
                "SVM_std":     res["cv"]["SVM-linear"]["auc_std"],
            })
    return pd.DataFrame(records)


def plot_reverse_summary(df_res: pd.DataFrame) -> None:
    setup_order = [s["name"] for s in SETUPS]
    col_order   = [COMBO_LABELS[c] for c in COMBOS]
    pivot_lr  = df_res.pivot(index="setup", columns="cohort_test",
                              values="LogReg_AUC").reindex(index=setup_order, columns=col_order)
    pivot_svm = df_res.pivot(index="setup", columns="cohort_test",
                              values="SVM_AUC").reindex(index=setup_order, columns=col_order)

    print("\n=== Reverse LogReg AUC (fiber-space syllables, 5-fold CV) ===")
    print(pivot_lr.round(3).to_string())
    print("\n=== Reverse SVM AUC ===")
    print(pivot_svm.round(3).to_string())

    df_res.to_csv(OUT_DIR / "reverse_setups_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, piv, title in zip(axes, [pivot_lr, pivot_svm], ["LogReg AUC", "SVM AUC"]):
        data = piv.values.astype(float)
        im = ax.imshow(data, vmin=0.4, vmax=0.95, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="AUC (5-fold CV)")
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, fontsize=11)
        ax.set_yticks(range(len(setup_order)))
        ax.set_yticklabels(setup_order, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black" if 0.45 < v < 0.85 else "white")
        ax.set_title(f"HC vs PS — {title}\n(Fiber model syllables, 5-fold CV)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "reverse_setups_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'reverse_setups_heatmap.png'}")


def plot_fiber_summary(df_res: pd.DataFrame) -> None:
    setup_order = [s["name"] for s in SETUPS]
    experiments = df_res["experiment"].unique().tolist()

    pivot_lr  = df_res.pivot(index="setup", columns="experiment", values="LogReg_AUC").reindex(setup_order)
    pivot_svm = df_res.pivot(index="setup", columns="experiment", values="SVM_AUC").reindex(setup_order)

    print("\n\n=== Fiber LogReg AUC (LOO-CV) ===")
    print(pivot_lr.round(3).to_string())
    print("\n=== Fiber SVM AUC (LOO-CV) ===")
    print(pivot_svm.round(3).to_string())

    df_res.to_csv(OUT_DIR / "fiber_setups_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, piv, title in zip(axes, [pivot_lr, pivot_svm], ["LogReg AUC", "SVM AUC"]):
        data = piv.values.astype(float)
        im = ax.imshow(data, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="AUC (LOO-CV)")
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, fontsize=10)
        ax.set_yticks(range(len(setup_order)))
        ax.set_yticklabels(setup_order, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black" if 0.45 < v < 0.92 else "white")
        ax.set_title(f"Fiber: Control vs Social Isolation\n{title} (LOO-CV, balanced)", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fiber_setups_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'fiber_setups_heatmap.png'}")


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    """Called by pipeline.py run --stages 4."""
    np.random.seed(42)

    # ── Prerequisites: apply models to cross-cohort sessions ─────────────────
    logger.info("Applying models to cross-cohort sessions (if not done) ...")
    _apply_joint_to_fiber()
    _apply_fiber_to_surgery()

    logger.info("=" * 60)
    logger.info("  STAGE 4A: HC vs PS — joint surgery model syllables")
    logger.info("=" * 60)
    df = build_metadata()
    df_res = run_all(df)
    plot_summary(df_res)

    logger.info("=" * 60)
    logger.info("  STAGE 4B: Fiber photometry — Control vs Social Isolation")
    logger.info("=" * 60)
    df_fiber = build_fiber_metadata()
    df_fiber_res = run_fiber_all(df_fiber)
    plot_fiber_summary(df_fiber_res)

    logger.info("=" * 60)
    logger.info("  STAGE 4C: Reverse — HC vs PS using fiber-model syllables")
    logger.info("=" * 60)
    if Path(SURGERY_H5_FIBER).exists():
        df_rev_res = run_reverse_all(df)
        plot_reverse_summary(df_rev_res)
    else:
        logger.warning("results_surgery.h5 not found — skipping reverse analysis. "
                       "Run apply_fiber_to_surgery.py first.")

    logger.info("Stage 4 complete. Outputs in %s", OUT_DIR)


# ── CLI (standalone) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)

    print(f"\nAll outputs saved to {OUT_DIR}/")
