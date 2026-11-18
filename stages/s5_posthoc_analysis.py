"""
Stage 5 – Post-hoc behavioural atlas, robustness, signatures, and affordance.

This stage builds a figure-first analysis package on top of the shared
`model_all` keypoint-MoSeq syllable space. Outputs are cached under
`outputs/05_stage5/` and grouped into four sections:

  1. syllable_atlas      – qualitative atlas for the 5 freq50 syllables
  2. robustness          – stability of usage, durations, and grammar
  3. shared_signatures   – recovery and abnormal-condition signatures
  4. affordance          – within-cohort context modulation summaries

Phase 5 brain-signal validation is intentionally deferred until those data are
organized and ready for behavioural alignment.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
KPMS_DIR = ROOT_DIR / "keypoint-moseq"
if str(KPMS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_DIR))

from stages.s4_classification import (
    ALL_H5,
    GI_META_CSV,
    GROUP_DATASET,
    GROUP_REPRESENTATION,
    NUM_STATES,
    STATE_MOSEQ_H5,
    STATE_MOSEQ_K,
    TAKAYA_META_CSV,
    _apply_representation,
    _assemble_feature_matrix,
    _build_group_metadata,
    _parse_surgery_15mice_meta,
    build_five_group_data,
    build_fiber_metadata,
    build_gi_metadata,
    build_session_feature_bank,
    evaluate_model,
    make_model,
)
from stages.s3_moseq_fitting import _build_skeleton
from utils.io_utils import (
    collect_all_videos,
    load_all_preprocessed,
    load_config,
    load_fps_per_session,
    save_dataframe,
    save_json,
)

logger = logging.getLogger(__name__)

OUT_ROOT = Path("outputs/05_stage5")
ATLAS_DIR = OUT_ROOT / "syllable_atlas"
ROBUST_DIR = OUT_ROOT / "robustness"
SIG_DIR = OUT_ROOT / "shared_signatures"
AFFORD_DIR = OUT_ROOT / "affordance"
MANUSCRIPT_DIR = OUT_ROOT / "manuscript"
META_DIR = OUT_ROOT / "metastate"
NATIVE_KPMS_DIR = OUT_ROOT / "_kpms_native_project"

FEATURE_KIND = "freq50+trans"
SIGNATURE_FEATURE_CANDIDATES = (
    "freq50+trans",
    "freq50+transsummary",
    "freq50+transsummary+seqstats",
    "freq50+transsummary+metastate",
    "freq50+trans+transsummary+seqstats",
)
BOOTSTRAP_N = 150
TOP_K = 10
SEED = 42
METASTATE_MIN_GLOBAL_USAGE = 0.01
EXCLUDED_STAGE5_SESSIONS = {
    "S2280008 021726Tue  Gi-04 pre_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
}


@dataclass
class DatasetSpec:
    name: str
    label: str
    sessions: list[str]
    y: np.ndarray
    cv_type: str
    mouse_ids: np.ndarray | None
    representation: str
    section: str
    notes: str = ""


def _ensure_dirs() -> None:
    for path in (OUT_ROOT, ATLAS_DIR, ROBUST_DIR, SIG_DIR, AFFORD_DIR, MANUSCRIPT_DIR, META_DIR, NATIVE_KPMS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _prepare_third_party_runtime() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _bout_lengths(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(seq) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    starts = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1]) + 1]
    ends = np.r_[starts[1:], len(seq)]
    return seq[starts], starts, ends - starts


def _load_sequences() -> dict[str, np.ndarray]:
    with h5py.File(ALL_H5, "r") as f:
        all_sessions = set(f.keys())
        seqs = {
            sess: f[sess]["syllable"][:].astype(np.int32)
            for sess in f.keys()
            if sess not in EXCLUDED_STAGE5_SESSIONS
        }
    excluded_present = EXCLUDED_STAGE5_SESSIONS.intersection(all_sessions)
    if excluded_present:
        logger.info("Excluded %d Stage 5 session(s): %s", len(excluded_present), ", ".join(sorted(excluded_present)))
    return seqs


def _load_state_sequences(path: str | Path = STATE_MOSEQ_H5) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    with h5py.File(path, "r") as f:
        return {
            sess: f[sess][:].astype(np.int32)
            for sess in f.keys()
            if isinstance(f[sess], h5py.Dataset)
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _save_fig(fig, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_image_panel(ax, path: Path, title: str, bg: str = "white") -> None:
    ax.set_facecolor(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, pad=8)
    if path.exists():
        try:
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_aspect("auto")
            return
        except Exception:
            pass
    ax.text(0.5, 0.5, f"Missing:\n{path.name}", ha="center", va="center", fontsize=10)


def _safe_spearman(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, float]:
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)
    mask = np.isfinite(vec_a) & np.isfinite(vec_b)
    vec_a = vec_a[mask]
    vec_b = vec_b[mask]
    if len(vec_a) < 3:
        return np.nan, np.nan
    if np.nanstd(vec_a) < 1e-12 or np.nanstd(vec_b) < 1e-12:
        return 0.0, 1.0
    res = spearmanr(vec_a, vec_b)
    return float(res.correlation), float(res.pvalue)


def _compose_native_kpms_summary() -> None:
    native_dir = ATLAS_DIR / "native_kpms"
    if not native_dir.exists():
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (native_dir / "syllable_frequency_distribution.png", "Frequency distribution"),
        (native_dir / "duration_distribution.png", "Duration distribution"),
        (native_dir / "similarity_dendrogram.png", "Trajectory similarity"),
        (native_dir / "cohort_frequency_stats" / "frequency_stat_stats.png", "Cohort freq50 usage"),
        (native_dir / "cohort_velocity_stats" / "velocity_px_s_mean_stat_stats.png", "Cohort speed by syllable"),
        (ATLAS_DIR / "freq50_mean_kinematics_heatmap.png", "Mean kinematics across freq50"),
    ]
    for ax, (path, title) in zip(axes.ravel(), panels):
        _draw_image_panel(ax, path, title)
    fig.suptitle("Native keypoint-moseq post-hoc overview", fontsize=16, y=0.98)
    plt.tight_layout()
    fig.savefig(native_dir / "native_kpms_overview.png", dpi=180)
    plt.close(fig)


def _compose_metastate_summary() -> None:
    if not META_DIR.exists():
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        (META_DIR / "metastate_usage_heatmap.png", "Metastate usage by cohort/arena"),
        (META_DIR / "metastate_signature_summary.png", "Metastate-augmented signatures"),
        (META_DIR / "metastate_model_comparison_ari.png", "Model comparison ARI"),
        (ROBUST_DIR / "lagged_mutual_information.png", "Lagged MI versus baselines"),
    ]
    for ax, (path, title) in zip(axes.ravel(), panels):
        _draw_image_panel(ax, path, title)
    fig.suptitle("Metastate and sequence-structure overview", fontsize=16, y=0.98)
    plt.tight_layout()
    fig.savefig(META_DIR / "metastate_overview.png", dpi=180)
    plt.close(fig)


def _compose_grammar_summary() -> None:
    graph_dir = ROBUST_DIR / "native_transition_graphs"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        (ROBUST_DIR / "robustness_overview.png", "Usage-rank robustness"),
        (ROBUST_DIR / "transition_entropy_baselines.png", "Entropy deltas"),
        (ROBUST_DIR / "lagged_mutual_information.png", "Lagged mutual information"),
        (graph_dir / "s15_oft_vs_y_maze" / "bigram" / "transition_matrices.png", "s15 OFT vs Y-maze"),
        (graph_dir / "takaya_oft_vs_y_maze" / "bigram" / "transition_matrices.png", "Takaya OFT vs Y-maze"),
        (graph_dir / "fiber_ast_vs_food_intake_during_habituation" / "bigram" / "transition_matrices.png", "Fiber AST vs Food"),
    ]
    for ax, (path, title) in zip(axes.ravel(), panels):
        _draw_image_panel(ax, path, title)
    fig.suptitle("Grammar robustness overview", fontsize=16, y=0.98)
    plt.tight_layout()
    fig.savefig(ROBUST_DIR / "grammar_overview.png", dpi=180)
    plt.close(fig)


def _compose_transition_graph_group_summary() -> None:
    graph_dir = ROBUST_DIR / "native_transition_graphs"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panels = [
        (graph_dir / "s15_oft_vs_y_maze" / "graph_group" / "transition_graphs.png", "s15 group graphs"),
        (graph_dir / "takaya_oft_vs_y_maze" / "graph_group" / "transition_graphs.png", "Takaya group graphs"),
        (graph_dir / "fiber_ast_vs_food_intake_during_habituation" / "graph_group" / "transition_graphs.png", "Fiber group graphs"),
    ]
    for ax, (path, title) in zip(np.atleast_1d(axes).ravel(), panels):
        _draw_image_panel(ax, path, title)
    fig.suptitle("Native transition graph-group views across affordances", fontsize=16, y=0.98)
    plt.tight_layout()
    fig.savefig(ROBUST_DIR / "transition_graph_group_overview.png", dpi=180)
    plt.close(fig)


def _prepare_native_kpms_project(meta_df: pd.DataFrame) -> tuple[Path, str]:
    project_dir = NATIVE_KPMS_DIR
    model_name = "model_all"
    model_dir = project_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results_link = model_dir / "results.h5"
    if results_link.exists() or results_link.is_symlink():
        results_link.unlink()
    os.symlink(Path(ALL_H5).resolve(), results_link)

    index_df = (
        meta_df[["session", "cohort", "arena_simple", "arena", "timepoint"]]
        .copy()
        .rename(columns={"session": "name"})
    )
    index_df["group"] = index_df["cohort"].fillna("unknown")
    index_df.to_csv(project_dir / "index.csv", index=False)
    return project_dir, model_name


def _load_session_metadata(sessions: list[str]) -> pd.DataFrame:
    tak_csv = pd.read_csv(TAKAYA_META_CSV)
    tak_lookup = tak_csv.set_index("video_stem")
    gi_meta = build_gi_metadata().set_index("session")
    fiber_meta = build_fiber_metadata().set_index("session")

    rows: list[dict] = []
    s15_sessions = [s for s in sessions if "_mouse" in s]
    s15_meta = _parse_surgery_15mice_meta(s15_sessions).set_index("session") if s15_sessions else None

    for sess in sessions:
        row = {
            "session": sess,
            "cohort": "unknown",
            "arena": "unknown",
            "timepoint": None,
            "mouse_id": None,
            "condition_group": None,
            "experiment": None,
            "condition": None,
        }

        if s15_meta is not None and sess in s15_meta.index:
            meta = s15_meta.loc[sess]
            row.update({
                "cohort": "s15",
                "arena": meta["test_type"],
                "timepoint": meta["timepoint"],
                "mouse_id": str(meta["mouse_id"]),
                "condition_group": "surgery_recovery",
            })
        elif sess in fiber_meta.index:
            meta = fiber_meta.loc[sess]
            row.update({
                "cohort": "fiber",
                "arena": meta["experiment"],
                "experiment": meta["experiment"],
                "condition": meta["condition"],
                "condition_group": meta["condition"],
            })
        elif sess in gi_meta.index:
            meta = gi_meta.loc[sess]
            row.update({
                "cohort": "gi",
                "arena": "OFT",
                "timepoint": meta["timepoint"],
                "mouse_id": str(meta["mouse_id"]),
                "condition_group": meta["group"],
                "condition": meta["group"],
            })
        else:
            stem = sess.split("_superanimal")[0]
            if stem in tak_lookup.index:
                meta = tak_lookup.loc[stem]
                row.update({
                    "cohort": "takaya",
                    "arena": meta["test_type"],
                    "timepoint": meta["timepoint"],
                    "mouse_id": str(meta["mouse_id"]),
                    "condition_group": meta["group"],
                    "condition": meta["group"],
                })

        rows.append(row)

    meta_df = pd.DataFrame(rows)
    meta_df["arena_simple"] = meta_df["arena"].replace({
        "Food intake during habituation": "Food",
        "AST": "AST",
        "Y maze": "Y-maze",
        "OFT": "OFT",
        "BFT": "BFT",
    })
    return meta_df


def _build_video_lookup(cfg: dict) -> dict[str, str]:
    all_videos = collect_all_videos(cfg, str(Path(".").resolve()))
    lookup: dict[str, str] = {}
    for videos in all_videos.values():
        for path in videos:
            lookup[Path(path).stem] = path
    return lookup


def _frame_kinematics(coords: np.ndarray, bodyparts: list[str]) -> dict[str, np.ndarray]:
    bp_to_idx = {bp: i for i, bp in enumerate(bodyparts)}
    center_idx = bp_to_idx.get("mouse_center", 0)
    nose_idx = bp_to_idx.get("nose", 0)
    tail_idx = bp_to_idx.get("tail_base", center_idx)

    center = coords[:, center_idx, :].astype(float)
    nose = coords[:, nose_idx, :].astype(float)
    tail = coords[:, tail_idx, :].astype(float)

    if np.isnan(center).all():
        center = np.nanmean(coords, axis=1)

    d_center = np.diff(center, axis=0, prepend=center[[0]])
    speed = np.linalg.norm(d_center, axis=1)

    heading_vec = nose - tail
    body_len = np.linalg.norm(heading_vec, axis=1)
    heading_ang = np.unwrap(np.arctan2(heading_vec[:, 1], heading_vec[:, 0]))
    ang_vel = np.abs(np.diff(heading_ang, prepend=heading_ang[:1]))

    move_norm = np.linalg.norm(d_center, axis=1)
    head_alignment = np.full(len(speed), np.nan, dtype=float)
    valid = (move_norm > 1e-6) & (body_len > 1e-6)
    head_alignment[valid] = np.sum(d_center[valid] * heading_vec[valid], axis=1) / (
        move_norm[valid] * body_len[valid]
    )

    center_ref = np.nanmedian(center, axis=0)
    radial = np.linalg.norm(center - center_ref, axis=1)
    scale = np.nanpercentile(radial, 90) if np.isfinite(radial).any() else np.nan
    radial_norm = radial / scale if np.isfinite(scale) and scale > 0 else np.full_like(radial, np.nan)

    return {
        "speed": speed,
        "ang_vel": ang_vel,
        "body_len": body_len,
        "head_alignment": head_alignment,
        "radial_norm": radial_norm,
        "center_x": center[:, 0],
        "center_y": center[:, 1],
    }


def _prototype_scores(row: pd.Series) -> dict[str, float]:
    speed = float(np.nan_to_num(row["speed_z"], nan=0.0))
    turn = float(np.nan_to_num(row["turn_z"], nan=0.0))
    bout = float(np.nan_to_num(row["bout_z"], nan=0.0))
    pause = float(np.nan_to_num(row["pause_z"], nan=0.0))
    align = float(np.nan_to_num(row["align_z"], nan=0.0))
    entropy = float(np.nan_to_num(row["entropy_z"], nan=0.0))
    compact = -float(np.nan_to_num(row["body_len_z"], nan=0.0))

    return {
        "locomotion": 1.4 * speed + 0.8 * align - 1.1 * pause,
        "turning": 1.6 * turn + 0.6 * speed + 0.2 * entropy,
        "resting": 1.5 * pause + 1.0 * bout - 1.3 * speed - 0.6 * entropy,
        "sniffing/investigation": 0.8 * speed + 1.1 * entropy + 0.8 * align - 0.5 * bout,
        "grooming-like": 1.0 * compact + 0.8 * pause + 0.7 * entropy - 0.8 * speed,
    }


def _soft_prototype_annotation(scores: dict[str, float]) -> tuple[str, str, float]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else best_score
    margin = float(best_score - second_score)
    label_map = {
        "locomotion": "locomotion-like",
        "turning": "turning-like",
        "resting": "pause/rest-like",
        "sniffing/investigation": "investigation-like",
        "grooming-like": "grooming-like",
    }
    if margin < 0.2:
        return "mixed/ambiguous", "low", margin
    if margin < 0.6:
        return label_map.get(best, best), "moderate", margin
    return label_map.get(best, best), "high", margin


def _zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mean = s.mean(skipna=True)
    std = s.std(ddof=0, skipna=True)
    if np.isnan(mean):
        return pd.Series(np.zeros(len(s)), index=s.index)
    if std == 0 or np.isnan(std):
        out = s.copy()
        out[:] = 0.0
        return out.fillna(0.0)
    return ((s - mean) / std).fillna(0.0)


def _safe_plot_limits(Xs: np.ndarray, pctl: float = 25.0) -> np.ndarray:
    flat = np.asarray(Xs, dtype=float).reshape(-1, 2)
    finite = np.isfinite(flat).all(axis=1)
    flat = flat[finite]
    if len(flat) == 0:
        return np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float)

    lo = np.nanpercentile(flat, pctl, axis=0)
    hi = np.nanpercentile(flat, 100 - pctl, axis=0)
    center = (lo + hi) / 2.0
    span = np.maximum(hi - lo, 1e-2)
    span = np.maximum(span, np.array([0.5, 0.5]))
    pad = span * 0.2
    return np.vstack([center - span / 2.0 - pad, center + span / 2.0 + pad])


def _expand_limits(lims: np.ndarray, frac: float = 0.12, min_pad: float = 0.15) -> np.ndarray:
    lims = np.asarray(lims, dtype=float)
    span = lims[1] - lims[0]
    pad = np.maximum(span * frac, min_pad)
    return np.vstack([lims[0] - pad, lims[1] + pad])


def _center_trajectories(Xs: np.ndarray) -> np.ndarray:
    centered = np.array(Xs, copy=True, dtype=float)
    for i in range(len(centered)):
        flat = centered[i].reshape(-1, centered[i].shape[-1])
        finite = np.isfinite(flat).all(axis=1)
        if finite.any():
            center = np.nanmedian(flat[finite], axis=0)
            centered[i] = centered[i] - center
    return centered


def _rotate_trajectory_to_target_heading(
    X: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    target_angle: float = np.pi / 2.0,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[-1] != 2:
        return X

    ref = X[-1]
    nose = ref[nose_idx]
    tail = ref[tail_idx]
    if not (np.isfinite(nose).all() and np.isfinite(tail).all()):
        return X

    vec = nose - tail
    if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-8:
        return X

    angle = float(np.arctan2(vec[1], vec[0]))
    delta = target_angle - angle
    c, s = np.cos(delta), np.sin(delta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return np.einsum("ij,tkj->tki", R, X)


def _align_head_direction(
    Xs: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    target_angle: float = np.pi / 2.0,
) -> np.ndarray:
    aligned = np.array(Xs, copy=True, dtype=float)
    for i in range(len(aligned)):
        aligned[i] = _rotate_trajectory_to_target_heading(
            aligned[i],
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            target_angle=target_angle,
        )
    return aligned


def _plot_compact_trajectory_panel(
    ax,
    X: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    lims: np.ndarray,
    shrink: float = 0.72,
    keypoint_colormap: str = "autumn",
) -> None:
    cmap = plt.get_cmap(keypoint_colormap)
    colors = cmap(np.linspace(0, 1, X.shape[1]))

    X = np.asarray(X, dtype=float) * shrink
    n_steps = X.shape[0]
    alpha_floor = 0.12

    for i in range(n_steps):
        frac = i / max(n_steps - 1, 1)
        alpha = alpha_floor + (1.0 - alpha_floor) * frac
        lw = 1.8 if i == n_steps - 1 else 1.2
        node_size = 28 if i == n_steps - 1 else 18
        for ii, jj in edges:
            ax.plot(
                X[i, (ii, jj), 0],
                X[i, (ii, jj), 1],
                color=colors[ii],
                alpha=alpha,
                linewidth=lw,
                solid_capstyle="round",
                clip_on=True,
            )
        ax.scatter(
            X[i, :, 0],
            X[i, :, 1],
            c=colors,
            s=node_size,
            alpha=alpha,
            edgecolors="0.25",
            linewidths=0.35,
            clip_on=True,
            zorder=3,
        )

    ax.set_xlim(lims[0, 0], lims[1, 0])
    ax.set_ylim(lims[0, 1], lims[1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, pad=10)


def _build_feature_caches(
    seqs: dict[str, np.ndarray],
    feature_bank: dict[str, dict],
    meta_df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
    video_lookup = _build_video_lookup(cfg)
    freq50_idx = feature_bank[next(iter(feature_bank))]["freq50_idx"]

    session_rows: list[dict] = []
    syllable_rows: list[dict] = []
    clip_rows: list[dict] = []

    kinematics_cache: dict[str, dict[str, np.ndarray]] = {}
    used_sessions = [s for s in seqs if s in coords_all]
    global_speed_vals: list[np.ndarray] = []

    for sess in used_sessions:
        kin = _frame_kinematics(coords_all[sess], bodyparts)
        kinematics_cache[sess] = kin
        global_speed_vals.append(kin["speed"])

    all_speed = np.concatenate(global_speed_vals) if global_speed_vals else np.array([0.0])
    pause_thr = float(np.nanpercentile(all_speed[np.isfinite(all_speed)], 25)) if np.isfinite(all_speed).any() else 0.0

    meta_lookup = meta_df.set_index("session")

    for sess, seq in seqs.items():
        feat = feature_bank[sess]
        usage = feat["usage"]
        transsummary = feat["transsummary"]
        seqstats = feat["seqstats"]
        fps = float(fps_map.get(sess, 10.0))
        top_rank = np.argsort(usage)[::-1]
        session_meta = meta_lookup.loc[sess].to_dict() if sess in meta_lookup.index else {}

        row = {
            "session": sess,
            "n_frames": int(len(seq)),
            "fps": fps,
            "switch_rate": float(seqstats[1]),
            "usage_entropy": float(seqstats[2]),
            "top5_usage_sum": float(seqstats[3]),
            "max_usage": float(seqstats[4]),
            "mean_bout_frames": float(seqstats[5]),
            "median_bout_frames": float(seqstats[6]),
            "std_bout_frames": float(seqstats[7]),
            "long_bout_fraction": float(seqstats[8]),
            "active_syllable_fraction": float(seqstats[9]),
            "mean_transition_entropy": float(seqstats[10]),
            "std_transition_entropy": float(seqstats[11]),
            "top_syllable": int(top_rank[0]),
            "top_syllable_usage": float(usage[top_rank[0]]),
            "second_syllable": int(top_rank[1]),
            "second_syllable_usage": float(usage[top_rank[1]]),
        }
        for s_idx in range(NUM_STATES):
            row[f"usage_s{s_idx:02d}"] = float(usage[s_idx])
            row[f"self_trans_s{s_idx:02d}"] = float(transsummary[s_idx])
            row[f"row_entropy_s{s_idx:02d}"] = float(transsummary[NUM_STATES + s_idx])
        row.update(session_meta)
        session_rows.append(row)

        if sess not in kinematics_cache:
            continue

        seq_trim = seq[:len(kinematics_cache[sess]["speed"])]
        bout_syll, bout_starts, bout_lens = _bout_lengths(seq_trim)
        video_path = video_lookup.get(sess.split("_superanimal")[0])

        for state in np.unique(seq_trim):
            mask = seq_trim == state
            if not np.any(mask):
                continue
            kin = kinematics_cache[sess]
            speed_vals = kin["speed"][mask]
            turn_vals = kin["ang_vel"][mask]
            body_vals = kin["body_len"][mask]
            align_vals = kin["head_alignment"][mask]
            syllable_rows.append({
                "session": sess,
                "syllable": int(state),
                "n_frames": int(mask.sum()),
                "usage": float(mask.mean()),
                "speed_mean": float(np.nanmean(speed_vals)) if np.isfinite(speed_vals).any() else np.nan,
                "ang_vel_mean": float(np.nanmean(turn_vals)) if np.isfinite(turn_vals).any() else np.nan,
                "body_len_mean": float(np.nanmean(body_vals)) if np.isfinite(body_vals).any() else np.nan,
                "head_alignment_mean": float(np.nanmean(align_vals)) if np.isfinite(align_vals).any() else np.nan,
                "pause_fraction": float(np.nanmean(speed_vals <= pause_thr)),
                "center_fraction": float(np.nanmean(kin["radial_norm"][mask] <= 0.5)),
                "wall_fraction": float(np.nanmean(kin["radial_norm"][mask] >= 0.8)),
                "mean_transition_entropy": float(transsummary[NUM_STATES + int(state)]),
                "self_transition": float(transsummary[int(state)]),
                "cohort": session_meta.get("cohort"),
                "arena": session_meta.get("arena"),
                "timepoint": session_meta.get("timepoint"),
                "condition_group": session_meta.get("condition_group"),
            })

        syll_mask = np.isin(bout_syll, freq50_idx)
        for state, start, dur in zip(bout_syll[syll_mask], bout_starts[syll_mask], bout_lens[syll_mask]):
            clip_rows.append({
                "syllable": int(state),
                "session": sess,
                "video_path": video_path,
                "start_frame": int(start),
                "end_frame": int(start + dur),
                "duration_frames": int(dur),
                "duration_sec": float(dur / fps if fps > 0 else np.nan),
                "cohort": session_meta.get("cohort"),
                "arena": session_meta.get("arena"),
                "timepoint": session_meta.get("timepoint"),
                "condition_group": session_meta.get("condition_group"),
            })

    session_df = pd.DataFrame(session_rows)
    syllable_df = pd.DataFrame(syllable_rows)
    clip_df = pd.DataFrame(clip_rows)

    # Aggregate per-syllable kinematics and prototype labels.
    agg = syllable_df.groupby("syllable", as_index=False).agg(
        usage_mean=("usage", "mean"),
        usage_median=("usage", "median"),
        session_prevalence=("session", "nunique"),
        speed_mean=("speed_mean", "mean"),
        ang_vel_mean=("ang_vel_mean", "mean"),
        body_len_mean=("body_len_mean", "mean"),
        head_alignment_mean=("head_alignment_mean", "mean"),
        pause_fraction=("pause_fraction", "mean"),
        center_fraction=("center_fraction", "mean"),
        wall_fraction=("wall_fraction", "mean"),
        mean_transition_entropy=("mean_transition_entropy", "mean"),
        self_transition=("self_transition", "mean"),
    )

    bout_df = clip_df.groupby("syllable", as_index=False).agg(
        mean_bout_frames=("duration_frames", "mean"),
        median_bout_frames=("duration_frames", "median"),
        max_bout_frames=("duration_frames", "max"),
    )
    agg = agg.merge(bout_df, on="syllable", how="left")
    agg["mean_bout_sec"] = agg["mean_bout_frames"] / np.nanmean(session_df["fps"])

    for col, zcol in [
        ("speed_mean", "speed_z"),
        ("ang_vel_mean", "turn_z"),
        ("mean_bout_frames", "bout_z"),
        ("pause_fraction", "pause_z"),
        ("head_alignment_mean", "align_z"),
        ("mean_transition_entropy", "entropy_z"),
        ("body_len_mean", "body_len_z"),
    ]:
        agg[zcol] = _zscore_series(agg[col])

    prototype_labels = []
    prototype_scores_all = []
    prototype_conf = []
    prototype_conf_label = []
    qc_notes = []
    for _, row in agg.iterrows():
        scores = _prototype_scores(row)
        label, conf_label, conf = _soft_prototype_annotation(scores)
        prototype_labels.append(label)
        prototype_scores_all.append(scores)
        prototype_conf.append(float(conf))
        prototype_conf_label.append(conf_label)
        qc_notes.append("manual clip review recommended" if conf_label != "high" else "supported by kinematics + clips")

    agg["prototype_label"] = prototype_labels
    agg["prototype_confidence"] = prototype_conf
    agg["prototype_confidence_label"] = prototype_conf_label
    agg["manual_qc_note"] = qc_notes
    for proto_name in ("locomotion", "turning", "resting", "sniffing/investigation", "grooming-like"):
        agg[f"score_{proto_name}"] = [scores[proto_name] for scores in prototype_scores_all]
    agg["is_freq50"] = agg["syllable"].isin(freq50_idx)

    if not clip_df.empty:
        top_clips = (
            clip_df.sort_values(["syllable", "duration_frames"], ascending=[True, False])
            .groupby("syllable")
            .head(8)
            .reset_index(drop=True)
        )
        save_dataframe(top_clips, ATLAS_DIR / "freq50_clip_manifest.csv")

    save_dataframe(session_df, OUT_ROOT / "session_feature_cache.csv")
    save_dataframe(agg.sort_values("syllable"), OUT_ROOT / "syllable_feature_cache.csv")
    save_dataframe(syllable_df, OUT_ROOT / "syllable_session_feature_cache.csv")
    return session_df, agg.sort_values("syllable").reset_index(drop=True), freq50_idx


def _plot_mean_pose_card(
    syllable: int,
    seqs: dict[str, np.ndarray],
    coords_all: dict[str, np.ndarray],
    bodyparts: list[str],
    syllable_features: pd.DataFrame,
    meta_df: pd.DataFrame,
    freq50_idx: np.ndarray,
) -> None:
    frames: list[np.ndarray] = []
    usage_rows: list[dict] = []
    max_frames = 1500
    for sess, seq in seqs.items():
        if sess not in coords_all:
            continue
        coords = coords_all[sess]
        usable = min(len(seq), len(coords))
        mask = seq[:usable] == syllable
        if not np.any(mask):
            continue
        selected = coords[:usable][mask]
        if len(selected) > 0:
            frames.append(selected[: max(1, max_frames // max(len(frames) + 1, 1))])

    if frames:
        pose = np.concatenate(frames, axis=0)
        mean_pose = np.nanmean(pose, axis=0)
        center = np.nanmean(mean_pose, axis=0)
        mean_pose = mean_pose - center
    else:
        mean_pose = np.zeros((len(bodyparts), 2), dtype=float)

    feat_row = syllable_features.loc[syllable_features["syllable"] == syllable].iloc[0]
    usage_by_cohort = (
        meta_df[["session", "cohort", "arena"]]
        .merge(
            pd.read_csv(OUT_ROOT / "syllable_session_feature_cache.csv")
            .query("syllable == @syllable")[["session", "usage"]],
            on="session",
            how="left",
        )
        .fillna({"usage": 0.0})
        .query("cohort != 'unknown'")
        .groupby("cohort", as_index=False)["usage"]
        .mean()
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.scatter(mean_pose[:, 0], mean_pose[:, 1], c="#4C72B0", s=40)
    for i, bp in enumerate(bodyparts):
        ax.text(mean_pose[i, 0], mean_pose[i, 1], bp, fontsize=7)
    ax.set_title(f"Syllable {syllable}: mean egocentric pose")
    ax.axhline(0, color="0.85", lw=1)
    ax.axvline(0, color="0.85", lw=1)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.bar(usage_by_cohort["cohort"], usage_by_cohort["usage"], color="#55A868")
    ax.set_title("Mean usage by cohort")
    ax.set_ylabel("Usage fraction")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    metric_map = {
        "speed_z": feat_row["speed_z"],
        "turn_z": feat_row["turn_z"],
        "log_bout_z": math.copysign(math.log1p(abs(float(feat_row["mean_bout_frames"]))), float(feat_row["bout_z"])) if pd.notna(feat_row["mean_bout_frames"]) else 0.0,
        "pause_z": feat_row["pause_z"],
        "entropy_z": feat_row["entropy_z"],
    }
    metric_vals = [float(np.nan_to_num(v, nan=0.0)) for v in metric_map.values()]
    colors = ["#C44E52" if v >= 0 else "#64B5CD" for v in metric_vals]
    ax.bar(metric_map.keys(), metric_vals, color=colors)
    ax.axhline(0, color="0.25", lw=1)
    ax.set_title("Scaled kinematics and grammar")
    ax.set_ylabel("Relative deviation")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    score_cols = [c for c in syllable_features.columns if c.startswith("score_")]
    score_dict = {
        col.replace("score_", ""): float(feat_row[col])
        for col in score_cols
    }
    ax.barh(list(score_dict.keys()), list(score_dict.values()), color="#8172B3")
    ax.set_title(
        f"Prototype: {feat_row['prototype_label']} "
        f"({feat_row.get('prototype_confidence_label', 'n/a')})"
    )
    ax.axvline(0, color="0.25", lw=1)

    fig.suptitle(
        f"freq50 syllable {syllable} | prevalence={feat_row['session_prevalence']:.0f} "
        f"| note={feat_row['manual_qc_note']}"
    )
    plt.tight_layout()
    fig.savefig(ATLAS_DIR / f"syllable_{syllable:02d}_card.png", dpi=160)
    plt.close(fig)


def _plot_freq50_kinematics_summary(syllable_features: pd.DataFrame, freq50_idx: np.ndarray) -> None:
    atlas_df = syllable_features[syllable_features["syllable"].isin(freq50_idx)].copy()
    if atlas_df.empty:
        return
    cols = [
        "speed_z",
        "turn_z",
        "bout_z",
        "pause_z",
        "align_z",
        "entropy_z",
        "body_len_z",
    ]
    label_map = {
        "speed_z": "speed",
        "turn_z": "turn",
        "bout_z": "bout",
        "pause_z": "pause",
        "align_z": "forward",
        "entropy_z": "entropy",
        "body_len_z": "body_len",
    }
    plot_df = atlas_df.set_index("syllable")[cols].rename(columns=label_map)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(plot_df.to_numpy().T, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(plot_df.index)))
    ax.set_xticklabels([str(int(s)) for s in plot_df.index])
    ax.set_yticks(range(len(plot_df.columns)))
    ax.set_yticklabels(plot_df.columns)
    ax.set_xlabel("freq50 syllable")
    ax.set_title("Mean kinematics per freq50 syllable (z-scored)")
    for i in range(plot_df.shape[1]):
        for j in range(plot_df.shape[0]):
            ax.text(j, i, f"{plot_df.iloc[j, i]:.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(ATLAS_DIR / "freq50_mean_kinematics_heatmap.png", dpi=170)
    plt.close(fig)


def run_syllable_atlas(
    cfg: dict,
    seqs: dict[str, np.ndarray],
    meta_df: pd.DataFrame,
    syllable_features: pd.DataFrame,
    freq50_idx: np.ndarray,
) -> None:
    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    atlas_df = syllable_features[syllable_features["syllable"].isin(freq50_idx)].copy()
    save_dataframe(atlas_df, ATLAS_DIR / "freq50_syllable_atlas.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(atlas_df)))
    ax.bar(
        [str(int(s)) for s in atlas_df["syllable"]],
        atlas_df["usage_mean"],
        color=colors,
    )
    ax.set_title("Five freq50 syllables in the shared model")
    ax.set_xlabel("Syllable")
    ax.set_ylabel("Mean usage")
    for i, row in atlas_df.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["usage_mean"] + 0.001,
            row["prototype_label"],
            rotation=45,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(ATLAS_DIR / "freq50_overview.png", dpi=160)
    plt.close(fig)

    for syllable in atlas_df["syllable"]:
        _plot_mean_pose_card(
            int(syllable),
            seqs,
            coords_all,
            bodyparts,
            syllable_features,
            meta_df,
            freq50_idx,
        )

    _plot_freq50_kinematics_summary(syllable_features, freq50_idx)
    _generate_kpms_style_trajectory_plots(cfg, freq50_idx)


def _run_native_kpms_atlas(
    cfg: dict,
    meta_df: pd.DataFrame,
    freq50_idx: np.ndarray,
) -> None:
    _prepare_third_party_runtime()

    import keypoint_moseq as kpms
    from keypoint_moseq.analysis import compute_moseq_df, compute_stats_df, plot_syll_stats_with_sem
    from keypoint_moseq.viz import (
        generate_grid_movies,
        plot_duration_distribution,
        plot_similarity_dendrogram,
        plot_syllable_frequencies,
    )

    native_dir = ATLAS_DIR / "native_kpms"
    native_dir.mkdir(parents=True, exist_ok=True)
    project_dir, model_name = _prepare_native_kpms_project(meta_df)

    results = kpms.load_results(path=ALL_H5)
    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
    median_fps = int(round(float(np.nanmedian(list(fps_map.values()))))) if fps_map else 10

    fig, _ = plot_syllable_frequencies(path=ALL_H5, min_frequency=0.0)
    _save_fig(fig, native_dir / "syllable_frequency_distribution.png")
    fig, _ = plot_duration_distribution(path=ALL_H5, fps=median_fps)
    _save_fig(fig, native_dir / "duration_distribution.png")

    moseq_df = compute_moseq_df(str(project_dir), model_name, results_dict=results, fps=median_fps)
    stats_df = compute_stats_df(
        str(project_dir),
        model_name,
        moseq_df,
        min_frequency=0.0,
        groupby=["group", "name"],
        fps=median_fps,
    )
    save_dataframe(moseq_df, native_dir / "moseq_frame_table.csv")
    save_dataframe(stats_df, native_dir / "moseq_stats_by_group.csv")
    save_dataframe(
        stats_df[stats_df["syllable"].isin(freq50_idx)].sort_values(["group", "syllable"]),
        native_dir / "freq50_native_stats.csv",
    )
    min_freq_for_movies = 0.02
    atlas_path = ATLAS_DIR / "freq50_syllable_atlas.csv"
    if atlas_path.exists():
        atlas_df = pd.read_csv(atlas_path)
        if not atlas_df.empty:
            min_freq_for_movies = float(max(0.001, atlas_df["usage_mean"].min() * 0.98))

    stats_freq50 = stats_df[stats_df["syllable"].isin(freq50_idx)].copy()
    if not stats_freq50.empty and stats_freq50["group"].nunique() > 1:
        fig, _ = plot_syll_stats_with_sem(
            stats_freq50,
            str(project_dir),
            model_name,
            save_dir=str(native_dir / "cohort_frequency_stats"),
            plot_sig=False,
            stat="frequency",
            order="stat",
            groups=sorted(stats_freq50["group"].dropna().unique()),
            figsize=(8, 4),
        )
        plt.close(fig)
        fig, _ = plot_syll_stats_with_sem(
            stats_freq50,
            str(project_dir),
            model_name,
            save_dir=str(native_dir / "cohort_velocity_stats"),
            plot_sig=False,
            stat="velocity_px_s_mean",
            order="stat",
            groups=sorted(stats_freq50["group"].dropna().unique()),
            figsize=(8, 4),
        )
        plt.close(fig)

    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    use_bodyparts = cfg["keypoints"]["use_bodyparts"]
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    coordinates = {k: v[:, use_idx, :] for k, v in coords_all.items() if k in results}
    centroids = {k: results[k]["centroid"] for k in coordinates if "centroid" in results[k]}
    headings = {k: results[k]["heading"] for k in coordinates if "heading" in results[k]}
    skeleton = _build_skeleton(use_bodyparts)

    grid_dir = native_dir / "grid_movies"
    grid_dir.mkdir(parents=True, exist_ok=True)
    try:
        generate_grid_movies(
            results=results,
            output_dir=str(grid_dir),
            coordinates=coordinates,
            centroids=centroids,
            headings=headings,
            bodyparts=use_bodyparts,
            use_bodyparts=use_bodyparts,
            skeleton=skeleton,
            keypoints_only=True,
            overlay_keypoints=False,
            rows=2,
            cols=3,
            pre=0.25,
            post=0.5,
            min_frequency=min_freq_for_movies,
            min_duration=3,
            fps=median_fps,
            quality=7,
            sampling_options={"n_neighbors": 50},
        )
        keep = {f"syllable{int(s)}.mp4" for s in freq50_idx}
        for movie_path in grid_dir.glob("*.mp4"):
            if movie_path.name not in keep:
                movie_path.unlink()
    except Exception as exc:
        logger.warning("Native kpms grid movie export failed: %s", exc)

    try:
        plot_similarity_dendrogram(
            coordinates=coordinates,
            results=results,
            save_path=str(native_dir / "similarity_dendrogram"),
            bodyparts=use_bodyparts,
            use_bodyparts=use_bodyparts,
            pre=0.2,
            post=0.6,
            min_frequency=min_freq_for_movies,
            fps=median_fps,
            density_sample=True,
            sampling_options={"n_neighbors": 50},
        )
    except Exception as exc:
        logger.warning("Similarity dendrogram export failed: %s", exc)

    _compose_native_kpms_summary()


def _generate_kpms_style_trajectory_plots(cfg: dict, freq50_idx: np.ndarray) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    import keypoint_moseq as kpms
    from keypoint_moseq.util import get_typical_trajectories
    from keypoint_moseq.viz import plot_trajectories

    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    use_bodyparts = cfg["keypoints"]["use_bodyparts"]
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    coordinates = {k: v[:, use_idx, :] for k, v in coords_all.items()}
    results = kpms.load_results(path=ALL_H5)

    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
    median_fps = int(round(float(np.nanmedian(list(fps_map.values()))))) if fps_map else 10
    pre = max(2, int(round(0.2 * median_fps)))
    post = max(5, int(round(0.6 * median_fps)))

    traj_dir = ATLAS_DIR / "kpms_style_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    trajectories = get_typical_trajectories(
        coordinates=coordinates,
        results=results,
        pre=pre,
        post=post,
        min_frequency=0.0,
        min_duration=3,
        bodyparts=use_bodyparts,
        use_bodyparts=use_bodyparts,
        density_sample=True,
        sampling_options={"n_neighbors": 50},
    )
    selected = {int(s): trajectories[int(s)] for s in freq50_idx if int(s) in trajectories}
    if not selected:
        logger.warning("No freq50 syllable trajectories available for keypoint-moseq style plotting.")
        return

    titles = [f"Syllable {s}" for s in selected.keys()]
    Xs = np.stack([selected[s] for s in selected.keys()])
    Xs_plot = Xs * np.array([1, -1])
    nose_idx = use_bodyparts.index("nose")
    tail_idx = use_bodyparts.index("tail_base")
    edges = [] if not use_bodyparts else kpms.get_edges(use_bodyparts, _build_skeleton(use_bodyparts))

    # Aligned/centered view for direct cross-syllable comparison at one shared scale.
    Xs_aligned = _center_trajectories(Xs_plot)
    Xs_aligned = _align_head_direction(
        Xs_aligned,
        nose_idx=nose_idx,
        tail_idx=tail_idx,
        target_angle=np.pi / 2.0,
    )
    aligned_lims = _expand_limits(_safe_plot_limits(Xs_aligned, pctl=30), frac=0.16, min_pad=0.18)

    # Canonical comparison strip: each syllable gets its own clipped subplot.
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, title, traj in zip(axes, titles, Xs_aligned):
        _plot_compact_trajectory_panel(
            ax,
            traj,
            edges=edges,
            title=title,
            lims=aligned_lims,
            shrink=0.62,
            keypoint_colormap="autumn",
        )
    fig.suptitle("Aligned freq50 trajectories at matched scale", fontsize=15, y=0.99)
    plt.tight_layout()
    fig.savefig(traj_dir / "freq50_trajectory_strip.pdf")
    fig.savefig(traj_dir / "freq50_trajectory_strip.png", dpi=180)
    plt.close(fig)

    for title, syllable, traj in zip(titles, selected.keys(), Xs_aligned):
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        _plot_compact_trajectory_panel(
            ax,
            traj,
            edges=edges,
            title=title,
            lims=aligned_lims,
            shrink=0.62,
            keypoint_colormap="autumn",
        )
        plt.tight_layout()
        fig.savefig(traj_dir / f"syllable_{int(syllable):02d}_trajectory.pdf")
        fig.savefig(traj_dir / f"syllable_{int(syllable):02d}_trajectory.png", dpi=180)
        plt.close(fig)


def _pairwise_bootstrap_spearman(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_col: str,
    group_col: str,
    n_boot: int = BOOTSTRAP_N,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    left_vals = left.groupby(group_col)[value_col].mean().reindex(range(NUM_STATES), fill_value=0.0).to_numpy()
    right_vals = right.groupby(group_col)[value_col].mean().reindex(range(NUM_STATES), fill_value=0.0).to_numpy()
    point = _safe_spearman(left_vals, right_vals)[0]

    boots: list[float] = []
    left_sessions = left["session"].unique()
    right_sessions = right["session"].unique()
    for _ in range(n_boot):
        l_draw = rng.choice(left_sessions, size=len(left_sessions), replace=True)
        r_draw = rng.choice(right_sessions, size=len(right_sessions), replace=True)
        l_boot = (
            left[left["session"].isin(l_draw)]
            .groupby(group_col)[value_col]
            .mean()
            .reindex(range(NUM_STATES), fill_value=0.0)
            .to_numpy()
        )
        r_boot = (
            right[right["session"].isin(r_draw)]
            .groupby(group_col)[value_col]
            .mean()
            .reindex(range(NUM_STATES), fill_value=0.0)
            .to_numpy()
        )
        boots.append(_safe_spearman(l_boot, r_boot)[0])

    return point, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def _transition_baseline_metrics(seqs: dict[str, np.ndarray]) -> pd.DataFrame:
    _prepare_third_party_runtime()
    rng = np.random.default_rng(SEED)
    try:
        import jax.numpy as jnp
        import jax.random as jr
        from state_moseq.util import simulate_markov_chain
    except Exception as exc:
        logger.warning("state-moseq Markov baseline unavailable: %s", exc)
        jnp = None
        jr = None
        simulate_markov_chain = None

    rows = []
    for idx, (sess, seq) in enumerate(seqs.items()):
        if len(seq) < 5:
            continue
        trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a, b] += 1
        row_entropy = []
        for i in range(NUM_STATES):
            if trans[i].sum() > 0:
                row_entropy.append(_entropy(trans[i] / trans[i].sum()))
        observed = float(np.mean(row_entropy)) if row_entropy else np.nan

        shuffled = np.array(seq, copy=True)
        rng.shuffle(shuffled)
        sh_trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
        for a, b in zip(shuffled[:-1], shuffled[1:]):
            sh_trans[a, b] += 1
        sh_entropy = []
        for i in range(NUM_STATES):
            if sh_trans[i].sum() > 0:
                sh_entropy.append(_entropy(sh_trans[i] / sh_trans[i].sum()))
        shuffled_mean = float(np.mean(sh_entropy)) if sh_entropy else np.nan

        usage = np.bincount(seq, minlength=NUM_STATES).astype(float)
        usage = usage / usage.sum()
        iid_seq = rng.choice(np.arange(NUM_STATES), size=len(seq), replace=True, p=usage)
        iid_trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
        for a, b in zip(iid_seq[:-1], iid_seq[1:]):
            iid_trans[a, b] += 1
        iid_entropy = []
        for i in range(NUM_STATES):
            if iid_trans[i].sum() > 0:
                iid_entropy.append(_entropy(iid_trans[i] / iid_trans[i].sum()))
        iid_mean = float(np.mean(iid_entropy)) if iid_entropy else np.nan

        markov_mean = np.nan
        if simulate_markov_chain is not None:
            trans_probs = trans + 1e-3
            trans_probs = trans_probs / trans_probs.sum(axis=1, keepdims=True)
            init_probs = np.bincount(seq, minlength=NUM_STATES).astype(float) + 1e-3
            init_probs = init_probs / init_probs.sum()
            try:
                sim = np.array(
                    simulate_markov_chain(
                        jr.PRNGKey(SEED + idx),
                        jnp.asarray(trans_probs),
                        len(seq),
                        jnp.asarray(init_probs),
                    )
                )
                sim_trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
                for a, b in zip(sim[:-1], sim[1:]):
                    sim_trans[a, b] += 1
                sim_entropy = []
                for i in range(NUM_STATES):
                    if sim_trans[i].sum() > 0:
                        sim_entropy.append(_entropy(sim_trans[i] / sim_trans[i].sum()))
                markov_mean = float(np.mean(sim_entropy)) if sim_entropy else np.nan
            except Exception as exc:
                logger.warning("Markov simulation failed for %s: %s", sess, exc)

        rows.append({
            "session": sess,
            "observed_mean_row_entropy": observed,
            "shuffled_mean_row_entropy": shuffled_mean,
            "iid_mean_row_entropy": iid_mean,
            "markov_mean_row_entropy": markov_mean,
            "delta_vs_shuffled": observed - shuffled_mean,
            "delta_vs_iid": observed - iid_mean,
            "delta_vs_markov": observed - markov_mean if np.isfinite(markov_mean) else np.nan,
        })
    return pd.DataFrame(rows)


def _combined_usage(sequences: list[np.ndarray]) -> np.ndarray:
    if not sequences:
        return np.zeros(NUM_STATES, dtype=float)
    counts = np.zeros(NUM_STATES, dtype=float)
    for seq in sequences:
        counts += np.bincount(seq, minlength=NUM_STATES)[:NUM_STATES]
    return counts / max(counts.sum(), 1.0)


def _plot_native_transition_contrasts(
    seqs: dict[str, np.ndarray],
    meta_df: pd.DataFrame,
    freq50_idx: np.ndarray,
) -> None:
    _prepare_third_party_runtime()
    from keypoint_moseq.analysis import (
        get_transition_matrix,
        plot_transition_graph_difference,
        plot_transition_graph_group,
        visualize_transition_bigram,
    )

    contrasts = [
        ("s15", "OFT", "Y maze"),
        ("takaya", "OFT", "Y maze"),
        ("fiber", "AST", "Food intake during habituation"),
    ]

    project_dir, model_name = _prepare_native_kpms_project(meta_df)
    for cohort, arena_a, arena_b in contrasts:
        left_names = meta_df[(meta_df["cohort"] == cohort) & (meta_df["arena"] == arena_a)]["session"].tolist()
        right_names = meta_df[(meta_df["cohort"] == cohort) & (meta_df["arena"] == arena_b)]["session"].tolist()
        left_seqs = [seqs[s] for s in left_names if s in seqs]
        right_seqs = [seqs[s] for s in right_names if s in seqs]
        if not left_seqs or not right_seqs:
            continue

        left_mat = get_transition_matrix(left_seqs, max_syllable=NUM_STATES, normalize="bigram", combine=True)
        right_mat = get_transition_matrix(right_seqs, max_syllable=NUM_STATES, normalize="bigram", combine=True)
        left_use = _combined_usage(left_seqs)
        right_use = _combined_usage(right_seqs)

        contrast_dir = ROBUST_DIR / "native_transition_graphs" / f"{_slug(cohort)}_{_slug(arena_a)}_vs_{_slug(arena_b)}"
        contrast_dir.mkdir(parents=True, exist_ok=True)
        visualize_transition_bigram(
            str(project_dir),
            model_name,
            [arena_a, arena_b],
            [left_mat[np.ix_(freq50_idx, freq50_idx)], right_mat[np.ix_(freq50_idx, freq50_idx)]],
            freq50_idx,
            save_dir=str(contrast_dir / "bigram"),
            normalize="bigram",
            figsize=(11, 5),
            show_syllable_names=False,
        )
        plot_transition_graph_group(
            str(project_dir),
            model_name,
            [arena_a, arena_b],
            [left_mat[np.ix_(freq50_idx, freq50_idx)], right_mat[np.ix_(freq50_idx, freq50_idx)]],
            [left_use[freq50_idx], right_use[freq50_idx]],
            freq50_idx,
            save_dir=str(contrast_dir / "graph_group"),
            layout="circular",
            show_syllable_names=False,
        )
        plot_transition_graph_difference(
            str(project_dir),
            model_name,
            [arena_a, arena_b],
            [left_mat[np.ix_(freq50_idx, freq50_idx)], right_mat[np.ix_(freq50_idx, freq50_idx)]],
            [left_use[freq50_idx], right_use[freq50_idx]],
            freq50_idx,
            save_dir=str(contrast_dir / "graph_difference"),
            layout="circular",
            show_syllable_names=False,
        )


def _plot_lagged_mi_summary(seqs: dict[str, np.ndarray]) -> None:
    _prepare_third_party_runtime()
    try:
        import jax.numpy as jnp
        from state_moseq.util import cross_sequence_mutual_information, lagged_mutual_information
    except Exception as exc:
        logger.warning("Lagged MI analysis unavailable: %s", exc)
        return

    names = sorted(seqs.keys())
    if not names:
        return
    if len(names) > 24:
        keep_idx = np.unique(np.linspace(0, len(names) - 1, 24).astype(int))
        names = [names[i] for i in keep_idx]

    max_len = max(len(seqs[name]) for name in names)
    padded = np.zeros((len(names), max_len), dtype=np.int32)
    mask = np.zeros((len(names), max_len), dtype=bool)
    for i, name in enumerate(names):
        seq = np.asarray(seqs[name], dtype=np.int32)
        padded[i, : len(seq)] = seq
        mask[i, : len(seq)] = True

    lags = np.array([1, 2, 4], dtype=np.int32)
    real_mi, markov_mi, shuff_mi = lagged_mutual_information(
        jnp.asarray(padded),
        jnp.asarray(mask),
        jnp.asarray(lags),
    )
    real_mi = np.asarray(real_mi)
    markov_mi = np.asarray(markov_mi)
    shuff_mi = np.asarray(shuff_mi)
    lagged_df = pd.DataFrame({
        "lag": np.tile(lags, len(names)),
        "session": np.repeat(names, len(lags)),
        "real_mi": real_mi.reshape(-1),
        "markov_mi": markov_mi.reshape(-1),
        "shuffled_mi": shuff_mi.reshape(-1),
    })
    save_dataframe(lagged_df, ROBUST_DIR / "lagged_mutual_information.csv")

    n_categories = int(max(np.max(v) for v in seqs.values()) + 1)
    lag1_rows = []
    for i, name in enumerate(names):
        seq = padded[i]
        m = mask[i].astype(float)
        if m.sum() < 4:
            continue
        seq_shift = np.roll(seq, 1)
        m_shift = mask[i].copy()
        m_shift[0] = False
        lag1 = float(cross_sequence_mutual_information(jnp.asarray(seq), jnp.asarray(seq_shift), jnp.asarray(m_shift), n_categories))
        lag1_rows.append({"session": name, "lag1_mi": lag1})
    save_dataframe(pd.DataFrame(lag1_rows), ROBUST_DIR / "lag1_mutual_information.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = lagged_df.groupby("lag", as_index=False)[["real_mi", "markov_mi", "shuffled_mi"]].mean()
    ax.plot(summary["lag"], summary["real_mi"], marker="o", color="#4C72B0", label="Observed")
    ax.plot(summary["lag"], summary["markov_mi"], marker="o", color="#55A868", label="Markov")
    ax.plot(summary["lag"], summary["shuffled_mi"], marker="o", color="#C44E52", label="Shuffled")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Mutual information")
    ax.set_title("Lagged syllable MI versus Markov and shuffled baselines")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(ROBUST_DIR / "lagged_mutual_information.png", dpi=170)
    plt.close(fig)


def run_robustness(
    meta_df: pd.DataFrame,
    syllable_session_df: pd.DataFrame,
    session_df: pd.DataFrame,
    seqs: dict[str, np.ndarray],
    freq50_idx: np.ndarray,
) -> None:
    contrasts = [
        ("s15", "OFT", "Y maze"),
        ("takaya", "OFT", "Y maze"),
        ("fiber", "AST", "Food intake during habituation"),
    ]

    rows: list[dict] = []
    for cohort, arena_a, arena_b in contrasts:
        left_sessions = meta_df[(meta_df["cohort"] == cohort) & (meta_df["arena"] == arena_a)]["session"]
        right_sessions = meta_df[(meta_df["cohort"] == cohort) & (meta_df["arena"] == arena_b)]["session"]
        left = syllable_session_df[syllable_session_df["session"].isin(left_sessions)]
        right = syllable_session_df[syllable_session_df["session"].isin(right_sessions)]
        if left.empty or right.empty:
            continue

        usage_point, usage_lo, usage_hi = _pairwise_bootstrap_spearman(left, right, "usage", "syllable")
        rows.append({
            "contrast": f"{cohort}:{arena_a} vs {arena_b}",
            "usage_rank_spearman": usage_point,
            "usage_rank_ci_lo": usage_lo,
            "usage_rank_ci_hi": usage_hi,
        })

    robustness_df = pd.DataFrame(rows)
    save_dataframe(robustness_df, ROBUST_DIR / "usage_rank_stability.csv")
    stale_gi = ROBUST_DIR / "gi_group_usage_stability_matrix.csv"
    if stale_gi.exists():
        stale_gi.unlink()

    entropy_df = _transition_baseline_metrics(seqs).merge(
        meta_df[["session", "cohort", "arena", "condition_group", "timepoint"]],
        on="session",
        how="left",
    )
    save_dataframe(entropy_df, ROBUST_DIR / "transition_entropy_vs_baseline.csv")
    _plot_native_transition_contrasts(seqs, meta_df, freq50_idx)
    _plot_lagged_mi_summary(seqs)
    _compose_grammar_summary()
    _compose_transition_graph_group_summary()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    if not robustness_df.empty:
        x = np.arange(len(robustness_df))
        width = 0.55
        ax.bar(x, robustness_df["usage_rank_spearman"], width=width, color="#4C72B0")
        ax.errorbar(
            x,
            robustness_df["usage_rank_spearman"],
            yerr=[
                robustness_df["usage_rank_spearman"] - robustness_df["usage_rank_ci_lo"],
                robustness_df["usage_rank_ci_hi"] - robustness_df["usage_rank_spearman"],
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(robustness_df["contrast"], rotation=25, ha="right")
        ax.set_title("Usage-rank stability across affordances")
        ax.set_ylabel("Spearman correlation")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(ROBUST_DIR / "robustness_overview.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropy_df["delta_vs_shuffled"].dropna(), bins=24, alpha=0.7, color="#55A868", label="Observed - shuffled")
    ax.hist(entropy_df["delta_vs_iid"].dropna(), bins=24, alpha=0.7, color="#C44E52", label="Observed - IID")
    if "delta_vs_markov" in entropy_df:
        ax.hist(entropy_df["delta_vs_markov"].dropna(), bins=24, alpha=0.6, color="#4C72B0", label="Observed - Markov")
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Transition entropy relative to grammar-free baselines")
    ax.set_xlabel("Mean row-entropy delta")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ROBUST_DIR / "transition_entropy_baselines.png", dpi=160)
    plt.close(fig)


def _feature_names(feature_bank: dict[str, dict], feature_kind: str) -> list[str]:
    sample = next(iter(feature_bank.values()))
    freq50 = [int(x) for x in sample["freq50_idx"]]
    names = [f"usage_s{s:02d}" for s in freq50]
    mods = set(feature_kind.split("+")) - {"freq50"}
    if "trans" in mods:
        names.extend([f"trans_s{a:02d}_to_s{b:02d}" for a in freq50 for b in freq50])
    if "transsummary" in mods:
        names.extend([f"self_trans_s{s:02d}" for s in freq50])
        names.extend([f"row_entropy_s{s:02d}" for s in freq50])
    if "seqstats" in mods:
        names.extend([f"seqstat_{i:02d}" for i in range(12)])
    if "metastate" in mods:
        meta_len = len(sample.get("metastate_usage", []))
        names.extend([f"metastate_{i:02d}" for i in range(meta_len)])
    return names


def _feature_group_name(name: str) -> str:
    if name.startswith("usage_"):
        return "usage"
    if name.startswith("trans_"):
        return "transition"
    if name.startswith("self_trans_") or name.startswith("row_entropy_") or name.startswith("seqstat_"):
        return "summary"
    if name.startswith("metastate_"):
        return "metastate"
    return "other"


def _signature_score(summary: dict) -> float:
    point_auc = summary.get("point_auc")
    if point_auc is not None and not pd.isna(point_auc):
        return float(point_auc)
    acc = summary.get("acc")
    if acc is not None and not pd.isna(acc):
        return float(acc)
    return float("-inf")


def _select_informative_metastates(summary_df: pd.DataFrame) -> list[int]:
    if summary_df.empty or "global_usage" not in summary_df:
        return []
    informative = (
        summary_df.loc[summary_df["global_usage"].fillna(0) >= METASTATE_MIN_GLOBAL_USAGE, "metastate"]
        .astype(int)
        .tolist()
    )
    if informative:
        return informative
    ranked = (
        summary_df.sort_values("global_usage", ascending=False)["metastate"]
        .dropna()
        .astype(int)
        .tolist()
    )
    return ranked[:2]


def run_metastate_posthoc(
    meta_df: pd.DataFrame,
    feature_bank: dict[str, dict],
    seqs: dict[str, np.ndarray],
) -> None:
    _prepare_third_party_runtime()
    from sklearn.metrics import adjusted_rand_score
    try:
        from state_moseq.util import get_adjusted_rand, get_durations, get_frequencies
        from state_moseq.viz import plot_sankey
    except Exception as exc:
        logger.warning("Metastate posthoc unavailable: %s", exc)
        return

    state_seqs = _load_state_sequences(STATE_MOSEQ_H5)
    if not state_seqs:
        logger.warning("No canonical state-moseq cache found at %s", STATE_MOSEQ_H5)
        return

    sessions = [s for s in meta_df["session"] if s in state_seqs]
    if not sessions:
        return

    rows = []
    for sess in sessions:
        z = state_seqs[sess]
        usage = np.bincount(z, minlength=STATE_MOSEQ_K)[:STATE_MOSEQ_K].astype(float)
        usage = usage / max(len(z), 1)
        meta = meta_df.set_index("session").loc[sess]
        for state_ix, value in enumerate(usage):
            rows.append({
                "session": sess,
                "metastate": state_ix,
                "usage": float(value),
                "cohort": meta["cohort"],
                "arena": meta["arena"],
                "timepoint": meta["timepoint"],
                "condition_group": meta["condition_group"],
            })
    usage_df = pd.DataFrame(rows)

    freq = np.asarray(get_frequencies(state_seqs))
    dur_rows = []
    for state_ix in range(STATE_MOSEQ_K):
        state_durs: list[int] = []
        for z in state_seqs.values():
            z = np.asarray(z, dtype=int)
            if len(z) == 0:
                continue
            bouts, starts, lens = _bout_lengths(z)
            state_durs.extend(lens[bouts == state_ix].tolist())
        dur_rows.append({
            "metastate": state_ix,
            "global_usage": float(freq[state_ix]) if state_ix < len(freq) else np.nan,
            "mean_duration_frames": float(np.nanmean(state_durs)) if state_durs else np.nan,
        })
    summary_df = pd.DataFrame(dur_rows)
    informative_states = _select_informative_metastates(summary_df)
    if informative_states:
        usage_df = usage_df.loc[usage_df["metastate"].isin(informative_states)].copy()
        summary_df = summary_df.loc[summary_df["metastate"].isin(informative_states)].copy()
    save_dataframe(usage_df, META_DIR / "metastate_session_usage.csv")
    save_dataframe(summary_df, META_DIR / "metastate_global_summary.csv")

    context_df = (
        usage_df.groupby(["cohort", "arena", "metastate"], as_index=False)["usage"]
        .mean()
        .pivot_table(index=["cohort", "arena"], columns="metastate", values="usage", fill_value=0.0)
    )
    if informative_states:
        context_df = context_df.reindex(columns=informative_states, fill_value=0.0)
    context_df.to_csv(META_DIR / "metastate_usage_by_context.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(context_df.to_numpy(), aspect="auto", cmap="magma")
    ax.set_yticks(range(len(context_df.index)))
    ax.set_yticklabels([f"{c} | {a}" for c, a in context_df.index])
    ax.set_xticks(range(context_df.shape[1]))
    ax.set_xticklabels([f"M{i}" for i in context_df.columns])
    title = "Canonical metastate usage across cohorts and arenas"
    if informative_states:
        title = f"Informative metastate usage across cohorts and arenas (n={len(informative_states)})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(META_DIR / "metastate_usage_heatmap.png", dpi=170)
    plt.close(fig)

    alt_candidates = [
        Path("outputs/03_moseq/kpms_project_all/model_all/state_moseq_K10_kappa1.0_ebs0.5.h5"),
        Path("outputs/03_moseq/kpms_project_all/model_all/state_moseq_K10_kappa0.1_ebs1.0.h5"),
    ]
    alt_path = next((p for p in alt_candidates if p.exists()), None)
    if alt_path is not None:
        alt_state_seqs = _load_state_sequences(alt_path)
        common = {k: state_seqs[k] for k in sessions if k in alt_state_seqs and len(state_seqs[k]) == len(alt_state_seqs[k])}
        alt_common = {k: alt_state_seqs[k] for k in common}
        if common:
            try:
                fig = plot_sankey(common, alt_common)
                fig.write_html(str(META_DIR / "metastate_model_comparison_sankey.html"))
            except Exception as exc:
                logger.warning("Sankey export failed: %s", exc)

            ari = float(get_adjusted_rand(common, alt_common))
            rows = []
            for sess in common:
                rows.append({"session": sess, "adjusted_rand": float(adjusted_rand_score(common[sess], alt_common[sess]))})
            ari_df = pd.DataFrame(rows)
            save_dataframe(ari_df, META_DIR / "metastate_model_comparison_ari.csv")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ari_df["adjusted_rand"], bins=20, color="#4C72B0", alpha=0.8)
            ax.axvline(ari, color="#C44E52", linestyle="--", label=f"pooled ARI={ari:.2f}")
            ax.set_title("Canonical vs alternate state-moseq agreement")
            ax.set_xlabel("Adjusted Rand index")
            ax.legend(frameon=False)
            plt.tight_layout()
            fig.savefig(META_DIR / "metastate_model_comparison_ari.png", dpi=170)
            plt.close(fig)

    sig_specs = []
    recovery_specs, abnormal_specs, _ = _build_stage5_specs(meta_df)
    sig_specs.extend(recovery_specs)
    sig_specs.extend([s for s in abnormal_specs if s.name.startswith("fiber_")])
    rows = []
    for spec in sig_specs:
        try:
            _, _, summary = _fit_binary_signature(spec, feature_bank, feature_kind="freq50+transsummary+metastate", n_boot=80)
        except Exception as exc:
            logger.warning("Skipping metastate signature summary for %s: %s", spec.name, exc)
            continue
        rows.append({"dataset": spec.name, "label": spec.label, **summary})
    if rows:
        summary_df = pd.DataFrame(rows)
        save_dataframe(summary_df, META_DIR / "metastate_signature_summary.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_df = summary_df.dropna(subset=["point_auc"]).copy()
        ax.bar(plot_df["dataset"], plot_df["point_auc"], color="#8172B3")
        ax.errorbar(
            plot_df["dataset"],
            plot_df["point_auc"],
            yerr=[plot_df["point_auc"] - plot_df["auc_ci_lo"], plot_df["auc_ci_hi"] - plot_df["point_auc"]],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_ylabel("AUC")
        ax.set_title("Metastate-augmented signatures")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(META_DIR / "metastate_signature_summary.png", dpi=170)
        plt.close(fig)

    _compose_metastate_summary()


def _fit_binary_signature(
    spec: DatasetSpec,
    feature_bank: dict[str, dict],
    feature_kind: str = FEATURE_KIND,
    n_boot: int = BOOTSTRAP_N,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result = _assemble_feature_matrix(spec.sessions, feature_bank, feature_kind)
    if result is None:
        raise ValueError(f"Could not build feature matrix for {spec.name}")
    X_raw, block_sizes = result
    meta = _build_group_metadata(spec.name, spec.sessions, spec.y, spec.mouse_ids)
    transformed = _apply_representation(spec.name, meta, X_raw, spec.cv_type, spec.representation)
    if transformed is None:
        raise ValueError(f"Representation incompatible for {spec.name}")
    X_use, y_use, cv_use, groups_use = transformed

    model = make_model(block_sizes)
    fitted = clone(model).fit(X_use, y_use)
    coef = fitted.named_steps["clf"].coef_[0]
    names = _feature_names(feature_bank, feature_kind)

    coef_df = pd.DataFrame({
        "dataset": spec.name,
        "label": spec.label,
        "feature": names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
        "feature_group": [_feature_group_name(name) for name in names],
    }).sort_values("abs_coefficient", ascending=False)

    rng = np.random.default_rng(SEED)
    inclusion = defaultdict(int)
    unique_y = np.unique(y_use)
    class_idx = {cls: np.where(y_use == cls)[0] for cls in unique_y}
    for _ in range(n_boot):
        draw = np.concatenate([
            rng.choice(class_idx[cls], size=len(class_idx[cls]), replace=True)
            for cls in unique_y
        ])
        boot_model = clone(model).fit(X_use[draw], y_use[draw])
        boot_coef = np.abs(boot_model.named_steps["clf"].coef_[0])
        top_idx = np.argsort(boot_coef)[::-1][:TOP_K]
        for idx in top_idx:
            inclusion[names[idx]] += 1
    coef_df["bootstrap_topk_inclusion"] = coef_df["feature"].map(
        lambda x: inclusion[x] / float(n_boot)
    )

    metrics = evaluate_model(X_use, y_use, cv_use, groups_use, block_sizes) or {}
    top_df = coef_df.head(25).copy()
    top_df["direction"] = np.where(top_df["coefficient"] >= 0, "positive", "negative")
    return coef_df, top_df, {
        "n_samples": int(len(y_use)),
        "n_features": int(X_use.shape[1]),
        "cv_type": cv_use,
        "representation": spec.representation,
        "point_auc": metrics.get("point_auc"),
        "auc_ci_lo": metrics.get("auc_ci_lo"),
        "auc_ci_hi": metrics.get("auc_ci_hi"),
        "acc": metrics.get("acc"),
        "notes": spec.notes,
    }


def _select_signature_feature_set(
    spec: DatasetSpec,
    feature_bank: dict[str, dict],
    candidates: tuple[str, ...] = SIGNATURE_FEATURE_CANDIDATES,
) -> tuple[str, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    trial_rows: list[dict] = []
    best: tuple[str, pd.DataFrame, pd.DataFrame, dict] | None = None
    best_key: tuple[float, float, float] | None = None

    for feature_kind in candidates:
        try:
            coef_df, top_df, summary = _fit_binary_signature(
                spec,
                feature_bank,
                feature_kind=feature_kind,
                n_boot=80,
            )
        except Exception as exc:
            trial_rows.append({
                "dataset": spec.name,
                "label": spec.label,
                "feature_kind": feature_kind,
                "selection_score": np.nan,
                "point_auc": np.nan,
                "acc": np.nan,
                "n_features": np.nan,
                "status": f"failed: {exc}",
            })
            continue

        score = _signature_score(summary)
        point_auc = summary.get("point_auc")
        point_auc = float(point_auc) if point_auc is not None and not pd.isna(point_auc) else float("-inf")
        n_features = float(summary.get("n_features", np.inf))
        trial_rows.append({
            "dataset": spec.name,
            "label": spec.label,
            "feature_kind": feature_kind,
            "selection_score": score,
            "point_auc": summary.get("point_auc"),
            "acc": summary.get("acc"),
            "n_features": summary.get("n_features"),
            "status": "ok",
        })
        rank_key = (score, point_auc, -n_features)
        if best is None or rank_key > best_key:
            best = (feature_kind, coef_df, top_df, summary)
            best_key = rank_key

    if best is None:
        raise ValueError(f"No shared-signature feature set succeeded for {spec.name}")
    return best[0], best[1], best[2], best[3], pd.DataFrame(trial_rows)


def _fit_multiclass_gi_signatures(
    sessions: list[str],
    labels: np.ndarray,
    mouse_ids: np.ndarray,
    feature_bank: dict[str, dict],
    feature_kind: str = FEATURE_KIND,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = _assemble_feature_matrix(sessions, feature_bank, feature_kind)
    if result is None:
        raise ValueError("Could not build GI four-group feature matrix")
    X_raw, _ = result
    names = _feature_names(feature_bank, feature_kind)

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )),
    ])
    pipeline.fit(X_raw, labels)
    clf = pipeline.named_steps["clf"]

    rows = []
    for class_idx, class_name in enumerate(clf.classes_):
        coef = clf.coef_[class_idx]
        for name, value in zip(names, coef):
            rows.append({
                "group": class_name,
                "feature": name,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
            })
    multi_df = pd.DataFrame(rows).sort_values(["group", "abs_coefficient"], ascending=[True, False])

    # One-vs-rest agreement at the coarse top-feature level.
    agree_rows = []
    for group in np.unique(labels):
        y = (labels == group).astype(int)
        spec = DatasetSpec(
            name=f"GI_group_{group}",
            label=f"GI {group} vs rest",
            sessions=sessions,
            y=y,
            cv_type="stratified",
            mouse_ids=mouse_ids,
            representation="raw",
            section="abnormal",
            notes="one-vs-rest agreement check",
        )
        coef_df, _, _ = _fit_binary_signature(spec, feature_bank, feature_kind=feature_kind, n_boot=80)
        top_binary = set(coef_df.head(TOP_K)["feature"])
        top_multi = set(multi_df[multi_df["group"] == group].head(TOP_K)["feature"])
        agree_rows.append({
            "group": group,
            "topk_overlap": len(top_binary & top_multi),
            "topk_union": len(top_binary | top_multi),
            "jaccard": len(top_binary & top_multi) / max(len(top_binary | top_multi), 1),
        })
    return multi_df, pd.DataFrame(agree_rows)


def _top_signed_features_from_coef(coef_df: pd.DataFrame, top_k: int = TOP_K) -> set[str]:
    top = coef_df.head(top_k).copy()
    return {
        f"{row['direction']}::{row['feature']}"
        for _, row in top.iterrows()
    }


def _plot_overlap_heatmap(
    labels: list[str],
    sets: list[set[str]],
    out_path: Path,
    title: str,
) -> None:
    if not labels:
        return
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            mat[i, j] = len(a & b) / max(len(a | b), 1)
    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 3, 0.9 * len(labels) + 3))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            color = "white" if val < 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_signed_feature_overlap(
    overlap_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if overlap_df.empty:
        return
    labels = overlap_df["dataset"].tolist()
    sets = overlap_df["top_signed_features"].tolist()
    _plot_overlap_heatmap(labels, sets, out_path, title)


def _format_signature_feature_label(signed_feature: str) -> str:
    direction, feature = signed_feature.split("::", 1)
    arrow = "+" if direction == "positive" else "-"
    if feature.startswith("trans_"):
        body = feature.replace("trans_", "").replace("_to_", " -> ")
    elif feature.startswith("usage_"):
        body = feature.replace("usage_", "usage ")
    elif feature.startswith("row_entropy_"):
        body = feature.replace("row_entropy_", "entropy ")
    elif feature.startswith("self_trans_"):
        body = feature.replace("self_trans_", "self ")
    elif feature.startswith("metastate_"):
        body = feature.replace("metastate_", "metastate ")
    elif feature.startswith("seqstat_"):
        body = feature.replace("seqstat_", "seqstat ")
    else:
        body = feature
    return f"{arrow} {body}"


def _plot_signature_identity_panel(
    top_tables: list[pd.DataFrame],
    section: str,
    out_path: Path,
    csv_path: Path,
    title: str,
) -> None:
    section_tables = [df.copy() for df in top_tables if not df.empty and df["section"].iloc[0] == section]
    if not section_tables:
        return

    feature_counts: dict[str, int] = defaultdict(int)
    feature_strength: dict[str, float] = defaultdict(float)
    feature_group: dict[str, str] = {}
    for df in section_tables:
        seen: set[str] = set()
        for _, row in df.head(TOP_K).iterrows():
            signed = f"{row['direction']}::{row['feature']}"
            feature_strength[signed] = max(feature_strength[signed], abs(float(row["coefficient"])))
            feature_group[signed] = row.get("feature_group", "")
            if signed not in seen:
                feature_counts[signed] += 1
                seen.add(signed)

    ordered = sorted(
        feature_counts,
        key=lambda x: (feature_counts[x], feature_strength[x]),
        reverse=True,
    )
    keep = [feat for feat in ordered if feature_counts[feat] >= 2]
    if not keep:
        keep = ordered[:min(10, len(ordered))]
    else:
        keep = keep[:12]

    records = []
    dataset_order = []
    for df in section_tables:
        dataset = df["dataset"].iloc[0]
        dataset_order.append(dataset)
        lookup = {
            f"{row['direction']}::{row['feature']}": float(row["coefficient"])
            for _, row in df.head(TOP_K).iterrows()
        }
        for feat in keep:
            records.append({
                "dataset": dataset,
                "signed_feature": feat,
                "display_label": _format_signature_feature_label(feat),
                "feature_group": feature_group.get(feat, ""),
                "shared_count": feature_counts.get(feat, 0),
                "coefficient": lookup.get(feat, 0.0),
                "present_in_topk": int(feat in lookup),
            })

    matrix_df = pd.DataFrame(records)
    save_dataframe(matrix_df, csv_path)
    if matrix_df.empty:
        return

    heat = (
        matrix_df.pivot(index="dataset", columns="display_label", values="coefficient")
        .reindex(dataset_order)
        .fillna(0.0)
    )
    if heat.empty:
        return

    fig_w = max(8.5, 0.8 * heat.shape[1] + 3.5)
    fig_h = max(4.8, 0.65 * heat.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = np.nanmax(np.abs(heat.to_numpy()))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(heat.index)
    ax.set_title(title)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = float(heat.iat[i, j])
            color = "white" if abs(val) > (0.55 * vmax) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signed coefficient")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _build_stage5_specs(meta_df: pd.DataFrame) -> tuple[list[DatasetSpec], list[DatasetSpec], list[DatasetSpec]]:
    recovery_specs: list[DatasetSpec] = []
    abnormal_specs: list[DatasetSpec] = []
    afford_specs: list[DatasetSpec] = []

    # Recovery: s15 pre/post.
    for arena in ("OFT", "Y maze"):
        sub = meta_df[(meta_df["cohort"] == "s15") & (meta_df["arena"] == arena)]
        if not sub.empty:
            recovery_specs.append(DatasetSpec(
                name=f"s15_{arena.replace(' ', '_')}_recovery",
                label=f"s15 {arena} pre vs post",
                sessions=sub["session"].tolist(),
                y=(sub["timepoint"] == "post").astype(int).to_numpy(),
                cv_type="group",
                mouse_ids=sub["mouse_id"].astype(str).to_numpy(),
                representation="raw",
                section="recovery",
            ))
            afford_specs.append(DatasetSpec(
                name=f"s15_{arena.replace(' ', '_')}_afford",
                label=f"s15 {arena} pre vs post",
                sessions=sub["session"].tolist(),
                y=(sub["timepoint"] == "post").astype(int).to_numpy(),
                cv_type="group",
                mouse_ids=sub["mouse_id"].astype(str).to_numpy(),
                representation="mouse_centered",
                section="affordance",
            ))

    tak_lookup = pd.read_csv(TAKAYA_META_CSV)
    surgery_ids = tak_lookup[tak_lookup["group"] == "surgery"]["mouse_id"].astype(str).unique()
    for arena in ("OFT", "Y maze"):
        sub = meta_df[
            (meta_df["cohort"] == "takaya")
            & (meta_df["arena"] == arena)
            & (meta_df["mouse_id"].isin(surgery_ids))
        ]
        if not sub.empty:
            recovery_specs.append(DatasetSpec(
                name=f"takaya_{arena.replace(' ', '_')}_recovery",
                label=f"Takaya surgery-only {arena} pre vs post",
                sessions=sub["session"].tolist(),
                y=(sub["timepoint"] == "post").astype(int).to_numpy(),
                cv_type="group",
                mouse_ids=sub["mouse_id"].astype(str).to_numpy(),
                representation="raw",
                section="recovery",
                notes="restricted to Takaya surgery mice for recovery signatures",
            ))

        aff_sub = meta_df[(meta_df["cohort"] == "takaya") & (meta_df["arena"] == arena)]
        if not aff_sub.empty:
            afford_specs.append(DatasetSpec(
                name="OFT_Takaya" if arena == "OFT" else "Ymaze_Takaya",
                label=f"Takaya {arena} control vs surgery (delta pre-post)",
                sessions=aff_sub["session"].tolist(),
                y=(aff_sub["condition_group"] == "surgery").astype(int).to_numpy(),
                cv_type="group",
                mouse_ids=aff_sub["mouse_id"].astype(str).to_numpy(),
                representation="delta_prepost",
                section="affordance",
            ))

    # GI recovery.
    gi = meta_df[meta_df["cohort"] == "gi"].copy()
    for target_tp, label in [("post1", "pre vs post1"), ("post3", "pre vs post3")]:
        sub = gi[gi["timepoint"].isin(["pre", target_tp])]
        if not sub.empty:
            recovery_specs.append(DatasetSpec(
                name=f"gi_pre_vs_{target_tp}",
                label=f"GI OFT {label}",
                sessions=sub["session"].tolist(),
                y=(sub["timepoint"] == target_tp).astype(int).to_numpy(),
                cv_type="group",
                mouse_ids=sub["mouse_id"].astype(str).to_numpy(),
                representation="mouse_mean",
                section="recovery",
            ))
    sub = gi[gi["timepoint"].isin(["pre", "post1", "post3"])]
    if not sub.empty:
        recovery_specs.append(DatasetSpec(
            name="gi_pre_vs_pooled_post",
            label="GI OFT pre vs pooled post",
            sessions=sub["session"].tolist(),
            y=(sub["timepoint"] != "pre").astype(int).to_numpy(),
            cv_type="group",
            mouse_ids=sub["mouse_id"].astype(str).to_numpy(),
            representation="mouse_mean",
            section="recovery",
        ))

    # Abnormal conditions: fiber isolation only. GI remains in recovery analyses.
    for exp in ("AST", "Food intake during habituation"):
        sub = meta_df[(meta_df["cohort"] == "fiber") & (meta_df["arena"] == exp)]
        if not sub.empty:
            abnormal_specs.append(DatasetSpec(
                name=f"fiber_{exp.replace(' ', '_')}",
                label=f"Fiber {exp} social isolation vs control",
                sessions=sub["session"].tolist(),
                y=(sub["condition"] == "Social isolation").astype(int).to_numpy(),
                cv_type="loo",
                mouse_ids=None,
                representation="raw",
                section="abnormal",
            ))
            afford_specs.append(DatasetSpec(
                name=f"fiber_{exp.replace(' ', '_')}_afford",
                label=f"Fiber {exp} social isolation vs control",
                sessions=sub["session"].tolist(),
                y=(sub["condition"] == "Social isolation").astype(int).to_numpy(),
                cv_type="loo",
                mouse_ids=None,
                representation="raw",
                section="affordance",
            ))

    return recovery_specs, abnormal_specs, afford_specs


def run_shared_signatures(
    meta_df: pd.DataFrame,
    feature_bank: dict[str, dict],
    syllable_features: pd.DataFrame,
) -> None:
    recovery_specs, abnormal_specs, _ = _build_stage5_specs(meta_df)
    coef_tables: list[pd.DataFrame] = []
    top_tables: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    overlap_rows: list[dict] = []
    trial_tables: list[pd.DataFrame] = []

    prototype_lookup = syllable_features.set_index("syllable")["prototype_label"].to_dict()

    for spec in recovery_specs:
        feature_kind, coef_df, top_df, summary, trial_df = _select_signature_feature_set(spec, feature_bank)
        trial_tables.append(trial_df)
        coef_df["section"] = "recovery"
        coef_df["feature_kind"] = feature_kind
        top_df["prototype_label"] = top_df["feature"].map(
            lambda feat: prototype_lookup.get(int(re.search(r"s(\d+)", feat).group(1)))
            if re.search(r"s(\d+)", feat) else None
        )
        top_df["feature_kind"] = feature_kind
        out_prefix = SIG_DIR / f"{spec.name}"
        save_dataframe(coef_df, f"{out_prefix}_coefficients.csv")
        save_dataframe(top_df, f"{out_prefix}_top_features.csv")
        coef_tables.append(coef_df)
        top_tables.append(top_df.assign(section="recovery"))
        summary_rows.append({
            "dataset": spec.name,
            "label": spec.label,
            "section": "recovery",
            "feature_kind": feature_kind,
            "selection_score": _signature_score(summary),
            **summary,
        })
        overlap_rows.append({
            "dataset": spec.name,
            "top_signed_features": _top_signed_features_from_coef(top_df),
        })

    for spec in abnormal_specs:
        feature_kind, coef_df, top_df, summary, trial_df = _select_signature_feature_set(spec, feature_bank)
        trial_tables.append(trial_df)
        coef_df["section"] = "abnormal"
        coef_df["feature_kind"] = feature_kind
        top_df["prototype_label"] = top_df["feature"].map(
            lambda feat: prototype_lookup.get(int(re.search(r"s(\d+)", feat).group(1)))
            if re.search(r"s(\d+)", feat) else None
        )
        top_df["feature_kind"] = feature_kind
        out_prefix = SIG_DIR / f"{spec.name}"
        save_dataframe(coef_df, f"{out_prefix}_coefficients.csv")
        save_dataframe(top_df, f"{out_prefix}_top_features.csv")
        coef_tables.append(coef_df)
        top_tables.append(top_df.assign(section="abnormal"))
        summary_rows.append({
            "dataset": spec.name,
            "label": spec.label,
            "section": "abnormal",
            "feature_kind": feature_kind,
            "selection_score": _signature_score(summary),
            **summary,
        })
        overlap_rows.append({
            "dataset": spec.name,
            "top_signed_features": _top_signed_features_from_coef(top_df),
        })

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, SIG_DIR / "signature_model_summary.csv")
    if coef_tables:
        all_coef = pd.concat(coef_tables, ignore_index=True)
        save_dataframe(all_coef, SIG_DIR / "all_binary_signature_coefficients.csv")
    if trial_tables:
        save_dataframe(pd.concat(trial_tables, ignore_index=True), SIG_DIR / "signature_feature_set_benchmark.csv")

    overlap_df = pd.DataFrame(overlap_rows)
    if not overlap_df.empty:
        recovery_names = set(summary_df[summary_df["section"] == "recovery"]["dataset"])
        abnormal_names = set(summary_df[summary_df["section"] == "abnormal"]["dataset"])
        _plot_signed_feature_overlap(
            overlap_df[overlap_df["dataset"].isin(recovery_names)],
            SIG_DIR / "recovery_syllable_overlap.png",
            "Recovery signature overlap (signed top features)",
        )
        _plot_signed_feature_overlap(
            overlap_df[overlap_df["dataset"].isin(abnormal_names)],
            SIG_DIR / "abnormal_syllable_overlap.png",
            "Abnormal-condition overlap (signed top features)",
        )
    if top_tables:
        _plot_signature_identity_panel(
            top_tables,
            section="recovery",
            out_path=SIG_DIR / "recovery_signature_identity.png",
            csv_path=SIG_DIR / "recovery_signature_identity.csv",
            title="Recovery signatures: recurrent specific features",
        )
        _plot_signature_identity_panel(
            top_tables,
            section="abnormal",
            out_path=SIG_DIR / "abnormal_signature_identity.png",
            csv_path=SIG_DIR / "abnormal_signature_identity.csv",
            title="Abnormal-condition signatures: recurrent specific features",
        )

    # Plot manuscript-ready summary panels instead of relying on CSV-only panels.
    rec_df = summary_df[summary_df["section"] == "recovery"].dropna(subset=["point_auc"]).copy()
    if not rec_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(rec_df))
        ax.bar(x, rec_df["point_auc"], color="#4C72B0")
        ax.errorbar(
            x,
            rec_df["point_auc"],
            yerr=[rec_df["point_auc"] - rec_df["auc_ci_lo"], rec_df["auc_ci_hi"] - rec_df["point_auc"]],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rec_df["dataset"], rotation=25, ha="right")
        ax.set_ylabel("AUC")
        ax.set_title("Recovery signatures across cohorts and arenas")
        plt.tight_layout()
        fig.savefig(SIG_DIR / "recovery_signature_summary.png", dpi=160)
        plt.close(fig)

    abn_df = summary_df[(summary_df["section"] == "abnormal") & summary_df["point_auc"].notna()].copy()
    if not abn_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(abn_df))
        ax.bar(x, abn_df["point_auc"], color="#DD8452")
        ax.errorbar(
            x,
            abn_df["point_auc"],
            yerr=[abn_df["point_auc"] - abn_df["auc_ci_lo"], abn_df["auc_ci_hi"] - abn_df["point_auc"]],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(abn_df["dataset"], rotation=25, ha="right")
        ax.set_ylabel("AUC")
        ax.set_title("Binary abnormal-condition signatures")
        plt.tight_layout()
        fig.savefig(SIG_DIR / "abnormal_signature_summary.png", dpi=160)
        plt.close(fig)


def run_affordance(
    meta_df: pd.DataFrame,
    feature_bank: dict[str, dict],
    syllable_features: pd.DataFrame,
) -> None:
    syll_sess = pd.read_csv(OUT_ROOT / "syllable_session_feature_cache.csv")
    freq50 = feature_bank[next(iter(feature_bank))]["freq50_idx"]
    rows: list[dict] = []
    effect_rows: list[dict] = []
    proto_lookup = syllable_features.set_index("syllable")["prototype_label"].to_dict()

    def _usage_effect(session_subset: pd.DataFrame, y_col: str) -> pd.Series:
        merged = session_subset[["session", y_col]].merge(
            syll_sess[["session", "syllable", "usage"]],
            on="session",
            how="left",
        )
        grp = (
            merged.groupby([y_col, "syllable"])["usage"]
            .mean()
            .unstack(0)
            .reindex(range(NUM_STATES))
            .fillna(0.0)
        )
        if grp.shape[1] < 2:
            return pd.Series(np.zeros(NUM_STATES), index=range(NUM_STATES))
        cols = sorted(grp.columns)
        return grp[cols[-1]] - grp[cols[0]]

    def _grammar_effect(session_names: list[str], y: np.ndarray) -> np.ndarray:
        left = [feature_bank[s]["trans_norm"][np.ix_(freq50, freq50)] for s, yy in zip(session_names, y) if yy == 0]
        right = [feature_bank[s]["trans_norm"][np.ix_(freq50, freq50)] for s, yy in zip(session_names, y) if yy == 1]
        if not left or not right:
            return np.zeros(len(freq50) * len(freq50), dtype=float)
        return (np.nanmean(np.stack(right), axis=0) - np.nanmean(np.stack(left), axis=0)).reshape(-1)

    def _bootstrap_corr(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, float]:
        boots = []
        rng = np.random.default_rng(SEED)
        n = len(vec_a)
        if n < 3:
            return np.nan, np.nan
        for _ in range(200):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            boots.append(_safe_spearman(vec_a[idx], vec_b[idx])[0])
        return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))

    comparisons = [
        {
            "dataset": "s15_affordance_modulation",
            "cohort": "s15",
            "contexts": ("OFT", "Y maze"),
            "label_col": "timepoint",
            "positive": "post",
            "description": "s15 post-vs-pre effect consistency between OFT and Y-maze",
        },
        {
            "dataset": "takaya_affordance_modulation",
            "cohort": "takaya",
            "contexts": ("OFT", "Y maze"),
            "label_col": "condition_group",
            "positive": "surgery",
            "description": "Takaya surgery-vs-control effect consistency between OFT and Y-maze",
        },
        {
            "dataset": "fiber_affordance_modulation",
            "cohort": "fiber",
            "contexts": ("AST", "Food intake during habituation"),
            "label_col": "condition",
            "positive": "Social isolation",
            "description": "Fiber isolation-vs-control effect consistency between AST and food context",
        },
    ]

    for comp in comparisons:
        cohort_df = meta_df[meta_df["cohort"] == comp["cohort"]].copy()
        ctx_a, ctx_b = comp["contexts"]
        sub_a = cohort_df[cohort_df["arena"] == ctx_a].copy()
        sub_b = cohort_df[cohort_df["arena"] == ctx_b].copy()
        if sub_a.empty or sub_b.empty:
            continue
        for sub in (sub_a, sub_b):
            sub["condition_binary"] = (sub[comp["label_col"]] == comp["positive"]).astype(int)

        usage_a = _usage_effect(sub_a, "condition_binary")
        usage_b = _usage_effect(sub_b, "condition_binary")
        grammar_a = _grammar_effect(sub_a["session"].tolist(), sub_a["condition_binary"].to_numpy())
        grammar_b = _grammar_effect(sub_b["session"].tolist(), sub_b["condition_binary"].to_numpy())

        usage_corr, usage_p = _safe_spearman(usage_a.to_numpy(), usage_b.to_numpy())
        grammar_corr, grammar_p = _safe_spearman(grammar_a, grammar_b)
        usage_ci_lo, usage_ci_hi = _bootstrap_corr(usage_a.to_numpy(), usage_b.to_numpy())
        grammar_ci_lo, grammar_ci_hi = _bootstrap_corr(grammar_a, grammar_b)

        top_idx = np.argsort(np.abs((usage_a + usage_b) / 2.0).to_numpy())[::-1][:8]
        top_sylls = [int(usage_a.index[i]) for i in top_idx]
        top_labels = [proto_lookup.get(s) for s in top_sylls]
        rows.append({
            "dataset": comp["dataset"],
            "label": comp["description"],
            "context_a": ctx_a,
            "context_b": ctx_b,
            "usage_effect_corr": usage_corr,
            "usage_ci_lo": usage_ci_lo,
            "usage_ci_hi": usage_ci_hi,
            "usage_pvalue": usage_p,
            "grammar_effect_corr": grammar_corr,
            "grammar_ci_lo": grammar_ci_lo,
            "grammar_ci_hi": grammar_ci_hi,
            "grammar_pvalue": grammar_p,
            "top_syllables": ",".join(map(str, top_sylls)),
            "top_prototypes": ",".join([x for x in top_labels if x]),
            "notes": "high correlation means condition effects are preserved across contexts",
        })

        for syll, eff_a, eff_b in zip(usage_a.index, usage_a.to_numpy(), usage_b.to_numpy()):
            effect_rows.append({
                "dataset": comp["dataset"],
                "context_a": ctx_a,
                "context_b": ctx_b,
                "syllable": int(syll),
                "effect_a": float(eff_a),
                "effect_b": float(eff_b),
            })

        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        xa = usage_a.to_numpy()
        xb = usage_b.to_numpy()
        if np.nanmax(np.abs(np.r_[xa, xb])) < 1e-9:
            jitter = np.linspace(-0.0025, 0.0025, len(xa))
            xa = xa + jitter
            xb = xb - jitter
        ax.scatter(xa, xb, s=56, alpha=0.9, color="#4C72B0", edgecolors="white", linewidths=0.5)
        ax.axhline(0, color="0.7", lw=1)
        ax.axvline(0, color="0.7", lw=1)
        lim = np.nanmax(np.abs(np.r_[xa, xb])) * 1.1
        lim = max(lim, 0.02)
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="#C44E52", lw=1)
        for syll in top_sylls[:5]:
            idx = int(np.where(usage_a.index.to_numpy() == syll)[0][0])
            ax.text(xa[idx], xb[idx], str(syll), fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Condition effect in {ctx_a}")
        ax.set_ylabel(f"Condition effect in {ctx_b}")
        ax.set_title(f"{comp['cohort']} affordance modulation\nusage rho={usage_corr:.2f}, grammar rho={grammar_corr:.2f}")
        ax.text(
            0.03,
            0.97,
            f"n={len(xa)} syllables\nusage p={usage_p:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
        )
        plt.tight_layout()
        fig.savefig(AFFORD_DIR / f"{comp['dataset']}_effect_correlation.png", dpi=170)
        plt.close(fig)

    afford_df = pd.DataFrame(rows)
    effect_df = pd.DataFrame(effect_rows)
    save_dataframe(afford_df, AFFORD_DIR / "affordance_summary.csv")
    save_dataframe(effect_df, AFFORD_DIR / "affordance_effect_vectors.csv")
    if afford_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(afford_df))
    axes[0].bar(x, afford_df["usage_effect_corr"], color="#4C72B0")
    axes[0].errorbar(
        x,
        afford_df["usage_effect_corr"],
        yerr=[afford_df["usage_effect_corr"] - afford_df["usage_ci_lo"], afford_df["usage_ci_hi"] - afford_df["usage_effect_corr"]],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axes[0].set_title("Usage-effect correlation across contexts")
    axes[0].set_ylabel("Spearman rho")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(afford_df["dataset"], rotation=25, ha="right")

    axes[1].bar(x, afford_df["grammar_effect_corr"], color="#55A868")
    axes[1].errorbar(
        x,
        afford_df["grammar_effect_corr"],
        yerr=[afford_df["grammar_effect_corr"] - afford_df["grammar_ci_lo"], afford_df["grammar_ci_hi"] - afford_df["grammar_effect_corr"]],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axes[1].set_title("Grammar-effect correlation across contexts")
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(afford_df["dataset"], rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(AFFORD_DIR / "affordance_modulation_overview.png", dpi=170)
    plt.close(fig)


def write_summary_report(
    freq50_idx: np.ndarray,
    meta_df: pd.DataFrame,
) -> None:
    report = {
        "stage": 5,
        "summary": "Behavioural atlas, robustness, shared signatures, and affordance analyses generated from the shared model_all syllable space.",
        "freq50_syllables": [int(x) for x in freq50_idx],
        "session_counts_by_cohort": meta_df["cohort"].value_counts(dropna=False).to_dict(),
        "brain_validation_status": "deferred_until_data_ready",
        "outputs": {
            "atlas": str(ATLAS_DIR),
            "robustness": str(ROBUST_DIR),
            "shared_signatures": str(SIG_DIR),
            "affordance": str(AFFORD_DIR),
            "metastate": str(META_DIR),
        },
    }
    save_json(report, OUT_ROOT / "stage5_summary.json")

    md = [
        "# Stage 5 Summary",
        "",
        "Stage 5 generated four figure-first behavioural analysis packages from the shared `model_all` syllable space.",
        "",
        f"- `freq50` syllables: {', '.join(map(str, report['freq50_syllables']))}",
        f"- Cohorts covered: {', '.join(sorted(str(k) for k in report['session_counts_by_cohort'].keys()))}",
        "- Brain-signal validation: deferred until data are organized and linked to behaviour sessions.",
        "",
        "## Output folders",
        f"- `syllable_atlas`: `{ATLAS_DIR}`",
        f"- `robustness`: `{ROBUST_DIR}`",
        f"- `shared_signatures`: `{SIG_DIR}`",
        f"- `affordance`: `{AFFORD_DIR}`",
        f"- `metastate`: `{META_DIR}`",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(md))


def _extract_fiber_mouse_id(session: str) -> str:
    stem = session.split("_superanimal")[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[1].split("-")[0]
    return stem


def _enrich_export_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    meta = meta_df.copy()
    fiber_mask = (meta["cohort"] == "fiber") & (meta["mouse_id"].isna() | (meta["mouse_id"].astype(str) == "None"))
    if fiber_mask.any():
        meta.loc[fiber_mask, "mouse_id"] = meta.loc[fiber_mask, "session"].map(_extract_fiber_mouse_id)
    return meta


def _top_metastates_for_export(
    feature_bank: dict[str, dict],
    sessions: list[str],
    top_n: int = 2,
) -> list[int]:
    rows = []
    for sess in sessions:
        feat = feature_bank.get(sess)
        if feat is None:
            continue
        rows.append(np.asarray(feat["metastate_usage"], dtype=float))
    if not rows:
        return list(range(top_n))
    mean_usage = np.nanmean(np.vstack(rows), axis=0)
    order = np.argsort(mean_usage)[::-1]
    return [int(x) for x in order[:top_n]]


def _presence_ranked_syllables_for_export(
    feature_bank: dict[str, dict],
    sessions: list[str],
    min_presence: float = 0.10,
) -> list[int]:
    rows = []
    for sess in sessions:
        feat = feature_bank.get(sess)
        if feat is None:
            continue
        rows.append(np.asarray(feat["usage"], dtype=float))
    if not rows:
        return []
    usage_mat = np.vstack(rows)
    presence = (usage_mat > 0).mean(axis=0)
    mean_usage = usage_mat.mean(axis=0)
    idx = np.where(presence > float(min_presence))[0]
    order = idx[np.argsort(mean_usage[idx])[::-1]]
    return [int(x) for x in order.tolist()]


def _condition_name_for_row(
    group_name: str,
    cohort: str,
    label: int,
    timepoint: str | None = None,
) -> str:
    if group_name in {"AST", "Food_intake"}:
        return "social_isolation" if int(label) == 1 else "control"
    if group_name in {"OFT_s15", "Ymaze_s15"}:
        return "post_surgery" if int(label) == 1 else "pre_surgery"
    if group_name in {"OFT_Takaya", "Ymaze_Takaya"}:
        return "surgery" if int(label) == 1 else "control"
    if group_name == "OFT_GI":
        if timepoint in {"pre", "post1", "post3"}:
            return str(timepoint)
        return "pooled_post" if int(label) == 1 else "pre"
    if group_name == "OFT_single":
        if cohort == "gi":
            if timepoint in {"pre", "post1", "post3"}:
                return str(timepoint)
            return "pooled_post" if int(label) == 1 else "pre"
        if cohort == "takaya":
            return "surgery" if int(label) == 1 else "control"
        return "abnormal" if int(label) == 1 else "reference"
    if group_name == "Y_maze":
        if cohort == "s15":
            return "post_surgery" if int(label) == 1 else "pre_surgery"
        if cohort == "takaya":
            return "surgery" if int(label) == 1 else "control"
        return "abnormal" if int(label) == 1 else "reference"
    return "abnormal" if int(label) == 1 else "reference"


def _context_name_for_row(environment_context: str) -> str:
    mapping = {
        "AST": "ast",
        "Food intake during habituation": "food_intake",
        "OFT": "oft",
        "Y maze": "y_maze",
    }
    return mapping.get(str(environment_context), str(environment_context).lower().replace(" ", "_"))


def _condition_family(condition_detail: str) -> str:
    detail = str(condition_detail)
    if detail == "social_isolation":
        return "social_isolation"
    if detail == "surgery":
        return "surgery"
    if detail == "control":
        return "control"
    if detail in {"pre", "post1", "post3", "pre_surgery", "post_surgery"}:
        return "recovery_timepoint"
    return detail


def _env_context_for_export(group_name: str, arena: str) -> str:
    if group_name == "OFT_s15":
        return "OFT_DUO"
    return arena


def _build_export_feature_columns(
    feature_bank: dict[str, dict],
    sessions: list[str],
    syllable_ids: list[int] | None = None,
    metastate_ids: list[int] | None = None,
) -> tuple[list[int], list[int], list[str]]:
    if syllable_ids is None:
        sample_sess = next(sess for sess in sessions if sess in feature_bank)
        freq50_idx = [int(x) for x in np.asarray(feature_bank[sample_sess]["freq50_idx"]).tolist()]
    else:
        freq50_idx = [int(x) for x in syllable_ids]
    if metastate_ids is None:
        metastate_ids = _top_metastates_for_export(feature_bank, sessions, top_n=2)
    else:
        metastate_ids = [int(x) for x in metastate_ids]

    feature_cols = [f"freq50_s{s:02d}" for s in freq50_idx]
    feature_cols.extend([
        f"trans_s{a:02d}_to_s{b:02d}"
        for a in freq50_idx
        for b in freq50_idx
    ])
    feature_cols.extend([f"transsummary_self_trans_s{s:02d}" for s in freq50_idx])
    feature_cols.extend([f"transsummary_row_entropy_s{s:02d}" for s in freq50_idx])
    feature_cols.extend([f"metastate_usage_{m:02d}" for m in metastate_ids])
    return freq50_idx, metastate_ids, feature_cols


def _build_session_level_export_table(
    feature_bank: dict[str, dict],
    meta_df: pd.DataFrame,
    freq50_idx: list[int],
    metastate_ids: list[int],
) -> pd.DataFrame:
    meta_lookup = meta_df.set_index("session")
    rows: list[dict] = []
    for sess, feat in feature_bank.items():
        if sess in EXCLUDED_STAGE5_SESSIONS or sess not in meta_lookup.index:
            continue
        usage = np.asarray(feat["usage"], dtype=float)
        trans = np.asarray(feat["trans_norm"], dtype=float)
        transsummary = np.asarray(feat["transsummary"], dtype=float)
        metastate = np.asarray(feat["metastate_usage"], dtype=float)
        meta = meta_lookup.loc[sess]
        row = {
            "session": sess,
            "mouse_id": str(meta["mouse_id"]) if pd.notna(meta["mouse_id"]) else None,
            "cohort": meta["cohort"],
            "arena": meta["arena"],
            "timepoint": meta["timepoint"],
            "condition_group": meta["condition_group"],
            "condition": meta["condition"],
        }
        for s in freq50_idx:
            row[f"freq50_s{s:02d}"] = float(usage[s])
        for a in freq50_idx:
            for b in freq50_idx:
                row[f"trans_s{a:02d}_to_s{b:02d}"] = float(trans[a, b])
        for s in freq50_idx:
            row[f"transsummary_self_trans_s{s:02d}"] = float(transsummary[s])
            row[f"transsummary_row_entropy_s{s:02d}"] = float(transsummary[NUM_STATES + s])
        for m in metastate_ids:
            row[f"metastate_usage_{m:02d}"] = float(metastate[m])
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_constant_or_join(values: pd.Series) -> str:
    vals = [str(v) for v in values.dropna().astype(str).tolist() if str(v) != "None"]
    uniq = list(dict.fromkeys(vals))
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]
    preferred = {"pre": 0, "post1": 1, "post3": 2, "post": 3}
    uniq = sorted(uniq, key=lambda x: (preferred.get(x, 99), x))
    return "|".join(uniq)


def _mouse_mean_group_keys(
    group_name: str,
    sub_df: pd.DataFrame,
) -> list[str]:
    keys = ["mouse_id", "label_binary"]
    if group_name == "OFT_GI":
        keys = ["mouse_id", "timepoint"]
    elif group_name == "OFT_single":
        keys = ["mouse_id", "cohort", "group_split_key"]
    return keys


def _zscore_within_study(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for study_cohort, sub_idx in out.groupby("study_cohort").groups.items():
        block = out.loc[sub_idx, feature_cols].astype(float)
        mean = block.mean(axis=0)
        std = block.std(axis=0, ddof=0).replace(0, np.nan)
        z = ((block - mean) / std).fillna(0.0)
        out.loc[sub_idx, feature_cols] = z
    return out


def export_group_feature_master_csv(
    meta_df: pd.DataFrame,
    feature_bank: dict[str, dict],
) -> Path:
    return export_group_feature_master_variant_csv(
        meta_df=meta_df,
        feature_bank=feature_bank,
        syllable_ids=None,
        metastate_ids=None,
        out_csv_name="study_group_feature_master_zscored.csv",
        out_meta_name="study_group_feature_master_zscored_metadata.json",
        syllable_block_name="freq50",
        syllable_note="All feature columns are z-scored within study cohort after group-specific representation. Metastate columns are metastate usage proportions.",
        syllable_selection_note="Canonical freq50 syllables (present in >=50% of sessions).",
    )


def export_group_feature_master_variant_csv(
    meta_df: pd.DataFrame,
    feature_bank: dict[str, dict],
    syllable_ids: list[int] | None,
    metastate_ids: list[int] | None,
    out_csv_name: str,
    out_meta_name: str,
    syllable_block_name: str,
    syllable_note: str,
    syllable_selection_note: str,
) -> Path:
    meta_df = _enrich_export_metadata(meta_df)
    cleaned_sessions = [s for s in meta_df["session"].tolist() if s not in EXCLUDED_STAGE5_SESSIONS]
    freq50_idx, metastate_ids, feature_cols = _build_export_feature_columns(
        feature_bank,
        cleaned_sessions,
        syllable_ids=syllable_ids,
        metastate_ids=metastate_ids,
    )
    raw_df = _build_session_level_export_table(feature_bank, meta_df, freq50_idx, metastate_ids)
    raw_df = raw_df[raw_df["session"].isin(cleaned_sessions)].copy()
    raw_df = raw_df.set_index("session", drop=False)

    groups_data = build_five_group_data()
    export_rows: list[dict] = []

    for group_name, (sessions, y, cv_type, mouse_ids) in groups_data.items():
        keep_mask = np.array([s not in EXCLUDED_STAGE5_SESSIONS and s in raw_df.index for s in sessions], dtype=bool)
        if not keep_mask.any():
            continue
        sess_use = [s for s, keep in zip(sessions, keep_mask) if keep]
        y_use = np.asarray(y)[keep_mask]
        mids_use = None if mouse_ids is None else np.asarray(mouse_ids)[keep_mask].astype(str)

        group_df = raw_df.loc[sess_use, ["session", "mouse_id", "cohort", "arena", "timepoint", "condition_group", "condition", *feature_cols]].copy()
        group_df["label_binary"] = y_use
        if mids_use is not None:
            group_df["mouse_id"] = mids_use

        representation = GROUP_REPRESENTATION[group_name]
        dataset_name = GROUP_DATASET.get(group_name, group_name)

        if representation == "raw":
            for _, row in group_df.iterrows():
                export = {
                    "sample_id": f"{group_name}::{row['session']}",
                    "session_usage": row["session"],
                    "n_sessions": 1,
                    "mouse_id": row["mouse_id"],
                    "study_cohort": row["cohort"],
                    "study_context": group_name,
                    "abnormal_condition": _condition_name_for_row(group_name, row["cohort"], row["label_binary"]),
                    "time_points": row["timepoint"] if pd.notna(row["timepoint"]) else "",
                    "env_context": _env_context_for_export(group_name, row["arena"]),
                    "repr_version": representation,
                    "cv_type": cv_type,
                    "label_binary": int(row["label_binary"]),
                }
                for col in feature_cols:
                    export[col] = float(row[col])
                export_rows.append(export)

        elif representation == "mouse_mean":
            if group_name == "OFT_single":
                group_df = group_df.copy()
                group_df["group_split_key"] = np.where(
                    group_df["cohort"].astype(str) == "gi",
                    group_df["timepoint"].fillna("unknown").astype(str),
                    ("label" + group_df["label_binary"].astype(int).astype(str)),
                )
            group_keys = _mouse_mean_group_keys(group_name, group_df)
            for group_vals, sub in group_df.groupby(group_keys, sort=False):
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)
                key_map = dict(zip(group_keys, group_vals))
                mouse_id = str(key_map["mouse_id"])
                label = int(sub["label_binary"].iloc[0])
                timepoint_ctx = key_map.get("timepoint", _aggregate_constant_or_join(sub["timepoint"]))
                if group_name == "OFT_single" and str(key_map.get("cohort", "")) == "gi":
                    timepoint_ctx = str(key_map.get("group_split_key", timepoint_ctx))
                export = {
                    "sample_id": f"{group_name}::{mouse_id}::{timepoint_ctx or ('label' + str(int(label)))}::mean",
                    "session_usage": "|".join(sub["session"].tolist()),
                    "n_sessions": int(len(sub)),
                    "mouse_id": mouse_id,
                    "study_cohort": _aggregate_constant_or_join(sub["cohort"]),
                    "study_context": group_name,
                    "abnormal_condition": _condition_name_for_row(
                        group_name,
                        _aggregate_constant_or_join(sub["cohort"]),
                        label,
                        timepoint=timepoint_ctx if timepoint_ctx else None,
                    ),
                    "time_points": timepoint_ctx,
                    "env_context": _env_context_for_export(group_name, _aggregate_constant_or_join(sub["arena"])),
                    "repr_version": representation,
                    "cv_type": cv_type,
                    "label_binary": int(label),
                }
                for col in feature_cols:
                    export[col] = float(sub[col].astype(float).mean())
                export_rows.append(export)

        elif representation == "mouse_centered":
            centered = group_df.copy()
            centered.loc[:, feature_cols] = centered[feature_cols].astype(float)
            for mouse_id, sub in centered.groupby("mouse_id"):
                idx = sub.index
                centered.loc[idx, feature_cols] = sub[feature_cols] - sub[feature_cols].mean(axis=0)
            for _, row in centered.iterrows():
                export = {
                    "sample_id": f"{group_name}::{row['session']}::centered",
                    "session_usage": row["session"],
                    "n_sessions": 1,
                    "mouse_id": row["mouse_id"],
                    "study_cohort": row["cohort"],
                    "study_context": group_name,
                    "abnormal_condition": _condition_name_for_row(group_name, row["cohort"], row["label_binary"]),
                    "time_points": row["timepoint"] if pd.notna(row["timepoint"]) else "",
                    "env_context": _env_context_for_export(group_name, row["arena"]),
                    "repr_version": representation,
                    "cv_type": cv_type,
                    "label_binary": int(row["label_binary"]),
                }
                for col in feature_cols:
                    export[col] = float(row[col])
                export_rows.append(export)

        elif representation == "delta_prepost":
            for mouse_id, sub in group_df.groupby("mouse_id", sort=False):
                tp = set(sub["timepoint"].dropna().astype(str))
                if not {"pre", "post"}.issubset(tp):
                    continue
                if sub["label_binary"].nunique() != 1:
                    continue
                label = int(sub["label_binary"].iloc[0])
                pre = sub[sub["timepoint"] == "pre"][feature_cols].astype(float).mean(axis=0)
                post = sub[sub["timepoint"] == "post"][feature_cols].astype(float).mean(axis=0)
                delta = post - pre
                export = {
                    "sample_id": f"{group_name}::{mouse_id}::delta_post_minus_pre",
                    "session_usage": "|".join(sub["session"].tolist()),
                    "n_sessions": int(len(sub)),
                    "mouse_id": mouse_id,
                    "study_cohort": _aggregate_constant_or_join(sub["cohort"]),
                    "study_context": group_name,
                    "abnormal_condition": _condition_name_for_row(group_name, _aggregate_constant_or_join(sub["cohort"]), label),
                    "time_points": "post_minus_pre",
                    "env_context": _env_context_for_export(group_name, _aggregate_constant_or_join(sub["arena"])),
                    "repr_version": representation,
                    "cv_type": "stratified",
                    "label_binary": label,
                }
                for col in feature_cols:
                    export[col] = float(delta[col])
                export_rows.append(export)

    export_df = pd.DataFrame(export_rows)
    export_df = _zscore_within_study(export_df, feature_cols)
    lead_cols = [
        "sample_id",
        "session_usage",
        "n_sessions",
        "mouse_id",
        "study_cohort",
        "study_context",
        "abnormal_condition",
        "time_points",
        "env_context",
        "repr_version",
        "cv_type",
        "label_binary",
    ]
    export_df = export_df[lead_cols + feature_cols]
    export_df = export_df.sort_values(["study_context", "study_cohort", "mouse_id", "sample_id"]).reset_index(drop=True)

    out_path = OUT_ROOT / out_csv_name
    save_dataframe(export_df, out_path)

    meta_out = OUT_ROOT / out_meta_name
    save_json({
        "syllable_ids": freq50_idx,
        "syllable_block_name": syllable_block_name,
        "metastate_ids": metastate_ids,
        "metastate_feature_type": "usage_proportion",
        "feature_blocks": {
            syllable_block_name: len(freq50_idx),
            "trans_matrix": len(freq50_idx) * len(freq50_idx),
            "transsummary": 2 * len(freq50_idx),
            "metastate": len(metastate_ids),
        },
        "note": syllable_note,
        "syllable_selection_note": syllable_selection_note,
    }, meta_out)
    return out_path


def export_unified_model_session_table(
    meta_df: pd.DataFrame,
    feature_bank: dict[str, dict],
) -> Path:
    meta_df = _enrich_export_metadata(meta_df)
    cleaned_sessions = [s for s in meta_df["session"].tolist() if s not in EXCLUDED_STAGE5_SESSIONS]
    freq50_idx, metastate_ids, feature_cols = _build_export_feature_columns(feature_bank, cleaned_sessions)
    raw_df = _build_session_level_export_table(feature_bank, meta_df, freq50_idx, metastate_ids)
    raw_df = raw_df[raw_df["session"].isin(cleaned_sessions)].copy()
    raw_df = raw_df.sort_values(["cohort", "arena", "mouse_id", "session"]).reset_index(drop=True)

    rows: list[dict] = []
    for _, row in raw_df.iterrows():
        cohort = str(row["cohort"])
        arena = str(row["arena"])
        condition_raw = row["condition"]
        timepoint = row["timepoint"] if pd.notna(row["timepoint"]) else ""

        if cohort == "fiber":
            abnormal_condition = "social_isolation" if str(condition_raw) == "Social isolation" else "control"
        elif cohort == "takaya":
            abnormal_condition = "surgery" if str(condition_raw) == "surgery" else "control"
        elif cohort == "s15":
            abnormal_condition = "post_surgery" if str(timepoint) == "post" else "pre_surgery"
        elif cohort == "gi":
            abnormal_condition = str(timepoint)
        else:
            abnormal_condition = str(condition_raw) if pd.notna(condition_raw) else "unknown"

        binary_name = "normal" if abnormal_condition in {"control", "pre", "pre_surgery"} else "abnormal"
        multiclass_name = _condition_family(abnormal_condition)

        export = {
            "sample_id": f"session::{row['session']}",
            "session_id": row["session"],
            "mouse_id": row["mouse_id"],
            "study_cohort": cohort,
            "study_context": f"{cohort}_{arena}".replace(" ", "_"),
            "env_context": arena,
            "abnormal_condition": abnormal_condition,
            "time_points": timepoint,
            "binary_abnormal_name": binary_name,
            "binary_abnormal_label": 0 if binary_name == "normal" else 1,
            "multiclass_condition_name": multiclass_name,
            "multiclass_condition_label": {
                "control": 0,
                "social_isolation": 1,
                "surgery": 2,
                "recovery_timepoint": 3,
            }.get(multiclass_name, -1),
            "leave_one_study_group": f"{cohort}_{arena}".replace(" ", "_"),
            "leave_one_cohort": cohort,
        }
        for col in feature_cols:
            export[col] = float(row[col])
        rows.append(export)

    export_df = pd.DataFrame(rows)
    lead_cols = [
        "sample_id",
        "session_id",
        "mouse_id",
        "study_cohort",
        "study_context",
        "env_context",
        "abnormal_condition",
        "time_points",
        "binary_abnormal_name",
        "binary_abnormal_label",
        "multiclass_condition_name",
        "multiclass_condition_label",
        "leave_one_study_group",
        "leave_one_cohort",
    ]
    export_df = export_df[lead_cols + feature_cols]

    out_path = OUT_ROOT / "unified_model_session_table.csv"
    save_dataframe(export_df, out_path)

    meta_out = OUT_ROOT / "unified_model_session_table_metadata.json"
    save_json({
        "freq50_syllables": freq50_idx,
        "metastate_ids": metastate_ids,
        "metastate_feature_type": "usage_proportion",
        "feature_blocks": {
            "freq50": len(freq50_idx),
            "trans_matrix": len(freq50_idx) * len(freq50_idx),
            "transsummary": 2 * len(freq50_idx),
            "metastate": len(metastate_ids),
        },
        "row_definition": "One raw video/session per row for unified modeling.",
        "normalization": "raw_features_only",
        "binary_mapping": {
            "normal": ["control", "pre", "pre_surgery"],
            "abnormal": ["social_isolation", "surgery", "post1", "post3", "post_surgery"],
        },
        "multiclass_mapping": {
            "control": ["control"],
            "social_isolation": ["social_isolation"],
            "surgery": ["surgery"],
            "recovery_timepoint": ["pre", "pre_surgery", "post1", "post3", "post_surgery"],
        },
        "note": "Session-level raw feature table intended for unified normal-vs-abnormal and subtype modeling. Recommended practice is to standardize features inside the model pipeline.",
    }, meta_out)
    return out_path


def write_manuscript_package() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(ATLAS_DIR / "freq50_syllable_atlas.csv")
    robustness = pd.read_csv(ROBUST_DIR / "usage_rank_stability.csv")
    signatures = pd.read_csv(SIG_DIR / "signature_model_summary.csv")
    affordance = pd.read_csv(AFFORD_DIR / "affordance_summary.csv")
    metastate_sig = pd.read_csv(META_DIR / "metastate_signature_summary.csv") if (META_DIR / "metastate_signature_summary.csv").exists() else pd.DataFrame()

    atlas_rows = []
    for _, row in atlas.sort_values("usage_mean", ascending=False).iterrows():
        atlas_rows.append({
            "section": "Figure 1",
            "result_type": "freq50 syllable atlas",
            "comparison": f"Syllable {int(row['syllable'])}",
            "headline_result": (
                f"Interpretation={row['prototype_label']}; mean usage={row['usage_mean']:.3f}; "
                f"mean bout={row['mean_bout_sec']:.2f}s; prevalence={int(row['session_prevalence'])} sessions"
            ),
            "strength_metric": row["prototype_confidence"],
            "strength_label": "interpretation margin",
            "source_file": f"outputs/05_stage5/syllable_atlas/syllable_{int(row['syllable']):02d}_card.png",
            "caveat": row["manual_qc_note"],
        })

    robustness_rows = []
    for _, row in robustness.iterrows():
        robustness_rows.append({
            "section": "Figure 2",
            "result_type": "usage-rank robustness",
            "comparison": row["contrast"],
            "headline_result": (
                f"Spearman rank stability={row['usage_rank_spearman']:.3f} "
                f"[{row['usage_rank_ci_lo']:.3f}, {row['usage_rank_ci_hi']:.3f}]"
            ),
            "strength_metric": row["usage_rank_spearman"],
            "strength_label": "rank stability",
            "source_file": "outputs/05_stage5/robustness/robustness_overview.png",
            "caveat": "usage-rank is shown here; grammar stability is summarized separately in the transition/entropy panels",
        })

    signature_rows = []
    for _, row in signatures.iterrows():
        metric = row["point_auc"]
        metric_txt = "descriptive multinomial" if pd.isna(metric) else f"AUC={metric:.3f} [{row['auc_ci_lo']:.3f}, {row['auc_ci_hi']:.3f}]"
        caveat = row["notes"] if pd.notna(row["notes"]) else ""
        signature_rows.append({
            "section": "Figure 3" if row["section"] == "recovery" else "Figure 4",
            "result_type": f"{row['section']} signature",
            "comparison": row["label"],
            "headline_result": metric_txt,
            "strength_metric": metric if pd.notna(metric) else np.nan,
            "strength_label": "AUC",
            "source_file": (
                "outputs/05_stage5/shared_signatures/recovery_signature_summary.png"
                if row["section"] == "recovery"
                else "outputs/05_stage5/shared_signatures/abnormal_signature_summary.png"
            ),
            "caveat": caveat,
        })

    afford_rows = []
    for _, row in affordance.sort_values("usage_effect_corr", ascending=False).iterrows():
        afford_rows.append({
            "section": "Figure 5",
            "result_type": "affordance modulation",
            "comparison": row["label"],
            "headline_result": (
                f"usage rho={row['usage_effect_corr']:.3f} [{row['usage_ci_lo']:.3f}, {row['usage_ci_hi']:.3f}] | "
                f"grammar rho={row['grammar_effect_corr']:.3f} [{row['grammar_ci_lo']:.3f}, {row['grammar_ci_hi']:.3f}] | "
                f"top prototypes={row['top_prototypes']}"
            ),
            "strength_metric": row["usage_effect_corr"],
            "strength_label": "usage-effect rho",
            "source_file": "outputs/05_stage5/affordance/affordance_modulation_overview.png",
            "caveat": row["notes"] if pd.notna(row["notes"]) else "",
        })

    summary_df = pd.DataFrame(atlas_rows + robustness_rows + signature_rows + afford_rows)
    save_dataframe(summary_df, MANUSCRIPT_DIR / "stage5_results_summary.csv")

    figure_map = pd.DataFrame([
        {
            "figure_id": "Figure 1",
            "title": "Shared freq50 syllables define a compact behavioural atlas",
            "panel": "A",
            "purpose": "Overview of the five shared freq50 syllables and their assigned prototypes",
            "artifact": "outputs/05_stage5/syllable_atlas/freq50_overview.png",
        },
        {
            "figure_id": "Figure 1",
            "title": "Shared freq50 syllables define a compact behavioural atlas",
            "panel": "B-F",
            "purpose": "MoSeq-style trajectory views plus one card per freq50 syllable showing mean pose, usage by cohort, kinematics, and prototype score",
            "artifact": "outputs/05_stage5/syllable_atlas/kpms_style_trajectories/freq50_trajectory_strip.png and outputs/05_stage5/syllable_atlas/syllable_11_card.png ... syllable_45_card.png",
        },
        {
            "figure_id": "Figure 1",
            "title": "Shared freq50 syllables define a compact behavioural atlas",
            "panel": "G-I",
            "purpose": "Native keypoint-moseq post-hoc summaries including frequency, duration, cohort stats, and grid movies",
            "artifact": "outputs/05_stage5/syllable_atlas/native_kpms/native_kpms_overview.png plus grid_movies/syllable11.mp4 ... syllable45.mp4",
        },
        {
            "figure_id": "Figure 2",
            "title": "Shared syllables are moderately stable across affordances while grammar departs from random baselines",
            "panel": "A",
            "purpose": "Usage-rank stability across s15, Takaya, and fiber arena contrasts",
            "artifact": "outputs/05_stage5/robustness/robustness_overview.png",
        },
        {
            "figure_id": "Figure 2",
            "title": "Shared syllables are moderately stable across affordances while grammar departs from random baselines",
            "panel": "B-D",
            "purpose": "Observed transition entropy relative to shuffled and IID baselines",
            "artifact": "outputs/05_stage5/robustness/grammar_overview.png and outputs/05_stage5/robustness/native_transition_graphs/",
        },
        {
            "figure_id": "Figure 2",
            "title": "Shared syllables are moderately stable across affordances while grammar departs from random baselines",
            "panel": "E",
            "purpose": "Native transition graph-group views for each cohort/context contrast",
            "artifact": "outputs/05_stage5/robustness/transition_graph_group_overview.png",
        },
        {
            "figure_id": "Figure 3",
            "title": "Recovery signatures differ across cohorts and arenas",
            "panel": "A",
            "purpose": "Recovery model summary across s15, Takaya surgery-only, and GI pre/post contrasts",
            "artifact": "outputs/05_stage5/shared_signatures/recovery_signature_summary.png",
        },
        {
            "figure_id": "Figure 3",
            "title": "Recovery signatures differ across cohorts and arenas",
            "panel": "B",
            "purpose": "Identity map of recurrent recovery-related specific features with signed coefficients",
            "artifact": "outputs/05_stage5/shared_signatures/recovery_signature_identity.png and recovery_signature_identity.csv",
        },
        {
            "figure_id": "Figure 4",
            "title": "Abnormal-condition signatures focus on social isolation effects",
            "panel": "A",
            "purpose": "Binary abnormal-condition model summary for fiber isolation datasets",
            "artifact": "outputs/05_stage5/shared_signatures/abnormal_signature_summary.png",
        },
        {
            "figure_id": "Figure 4",
            "title": "Abnormal-condition signatures focus on social isolation effects",
            "panel": "B",
            "purpose": "Identity map of recurrent abnormal-condition specific features with signed coefficients",
            "artifact": "outputs/05_stage5/shared_signatures/abnormal_signature_identity.png and abnormal_signature_identity.csv",
        },
        {
            "figure_id": "Figure 5",
            "title": "Environmental affordance modulates behavioural separability",
            "panel": "A",
            "purpose": "Condition-effect correlation across within-cohort context contrasts",
            "artifact": "outputs/05_stage5/affordance/affordance_modulation_overview.png",
        },
        {
            "figure_id": "Figure 5",
            "title": "Environmental affordance modulates behavioural separability",
            "panel": "B",
            "purpose": "Per-syllable effect vectors across paired contexts",
            "artifact": "outputs/05_stage5/affordance/affordance_effect_vectors.csv and *_effect_correlation.png",
        },
        {
            "figure_id": "Figure 5",
            "title": "Environmental affordance modulates behavioural separability",
            "panel": "C-D",
            "purpose": "Metastate usage and metastate-augmented signatures",
            "artifact": "outputs/05_stage5/metastate/metastate_overview.png",
        },
    ])
    save_dataframe(figure_map, MANUSCRIPT_DIR / "figure_panel_map.csv")

    md = [
        "# Stage 5 Manuscript Package",
        "",
        "This package converts the generated Stage 5 outputs into a manuscript-facing summary.",
        "",
        "## Figure Order",
        "1. Figure 1: Shared freq50 syllables define a compact behavioural atlas",
        "2. Figure 2: Shared syllables are moderately stable across affordances while grammar departs from random baselines",
        "3. Figure 3: Recovery signatures differ across cohorts and arenas",
        "4. Figure 4: Abnormal-condition signatures focus on social isolation effects",
        "5. Figure 5: Environmental context modulates condition effects",
        "",
        "## Panel Map",
    ]
    for row in figure_map.itertuples(index=False):
        md.append(f"- {row.figure_id}{row.panel}: {row.purpose}")
        md.append(f"  Artifact: `{row.artifact}`")
    md.extend([
        "",
        "## Important Caveats",
        "- GI drug-group comparisons were removed from the abnormal-condition section because those mice also underwent surgery; GI remains in the recovery-oriented analyses only.",
        "- Several kinematic stability cells remain `NaN` because some syllable-level kinematic means are not observed in both sides of a contrast.",
        "- Syllable interpretation labels are soft, clip-first behavioural descriptions rather than direct one-to-one mapping to published syllable IDs.",
    ])
    if not metastate_sig.empty:
        md.extend([
            "- Metastate summaries are exploratory add-ons built from the cached canonical `state_moseq_K10.h5` and should be interpreted as higher-level sequence structure, not replacements for the shared syllable atlas.",
        ])
    (MANUSCRIPT_DIR / "figure_panel_map.md").write_text("\n".join(md))


def run(config_path: str = "config.yaml") -> None:
    _ensure_dirs()
    cfg = load_config(config_path)
    logger.info("Loading Stage 5 inputs ...")
    seqs = _load_sequences()
    meta_df = _load_session_metadata(list(seqs.keys()))
    feature_bank = build_session_feature_bank()

    logger.info("Building cached session and syllable feature tables ...")
    session_df, syllable_features, freq50_idx = _build_feature_caches(seqs, feature_bank, meta_df, cfg)

    logger.info("Running syllable atlas ...")
    run_syllable_atlas(cfg, seqs, meta_df, syllable_features, freq50_idx)
    logger.info("Running native keypoint-moseq atlas summaries ...")
    _run_native_kpms_atlas(cfg, meta_df, freq50_idx)

    logger.info("Running robustness analyses ...")
    syllable_session_df = pd.read_csv(OUT_ROOT / "syllable_session_feature_cache.csv")
    run_robustness(meta_df, syllable_session_df, session_df, seqs, freq50_idx)

    logger.info("Running shared signature analyses ...")
    run_shared_signatures(meta_df, feature_bank, syllable_features)

    logger.info("Running affordance analyses ...")
    run_affordance(meta_df, feature_bank, syllable_features)

    logger.info("Running metastate posthoc analyses ...")
    run_metastate_posthoc(meta_df, feature_bank, seqs)

    logger.info("Writing Stage 5 summary ...")
    write_summary_report(freq50_idx, meta_df)
    write_manuscript_package()
    logger.info("Stage 5 complete. Outputs saved under %s", OUT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 post-hoc behavioural analysis.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run(config_path=args.config)


if __name__ == "__main__":
    main()
