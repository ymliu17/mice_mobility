#!/usr/bin/env python3
"""
Create short-window exemplar trajectory panels for syllables whose full
multi-frame clips smear the mouse body shape too much.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

TARGETS = (1, 4)
OUTDIR = Path("outputs/05_stage5/syllable_atlas_presence_gt10/kpms_style_trajectories")


def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    idx = np.arange(len(y))
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return y
    if finite.sum() == 1:
        y[~finite] = y[finite][0]
        return y
    y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    return y


def _fill_clip_nans(clip: np.ndarray) -> np.ndarray:
    filled = np.asarray(clip, dtype=float).copy()
    T, K, D = filled.shape
    for k in range(K):
        for d in range(D):
            filled[:, k, d] = _interp_nan_1d(filled[:, k, d])
    return filled


def _extract_candidates(seqs, coords_all, use_bodyparts, bodyparts, target, pre, post):
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    candidates = []
    for sess, seq in seqs.items():
        coords = coords_all.get(sess)
        if coords is None or len(coords) == 0:
            continue
        coords = coords[:, use_idx, :]
        usable = min(len(seq), len(coords))
        if usable < pre + post + 3:
            continue
        seq_use = seq[:usable]
        coords_use = coords[:usable]
        bout_states, bout_starts, bout_lengths = s5._bout_lengths(seq_use)
        for state, start, length in zip(bout_states, bout_starts, bout_lengths):
            if int(state) != int(target):
                continue
            center = int(start + max(length // 2, 0))
            left = center - pre
            right = center + post + 1
            if left < 0 or right > usable:
                continue
            clip = coords_use[left:right].astype(float, copy=True)
            finite_fraction = float(np.isfinite(clip).mean())
            if finite_fraction < 0.45:
                continue
            candidates.append((finite_fraction, int(length), clip))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [clip for _, _, clip in candidates[:120]]


def _normalize_clip(clip, nose_idx, tail_idx, center_idx):
    X = _fill_clip_nans(clip)
    if center_idx is not None and np.isfinite(X[:, center_idx, :]).all():
        X = X - X[:, center_idx:center_idx + 1, :]
    else:
        X = s5._center_trajectories(X[None, ...])[0]
    X = X * np.array([1.0, -1.0])
    X = s5._rotate_trajectory_to_target_heading(
        X,
        nose_idx=nose_idx,
        tail_idx=tail_idx,
        target_angle=np.pi / 2.0,
    )
    body_vec = X[:, nose_idx, :] - X[:, tail_idx, :]
    body_len = np.linalg.norm(body_vec, axis=1)
    finite = np.isfinite(body_len) & (body_len > 1e-6)
    if finite.sum() < 3:
        return None
    scale = float(np.nanmedian(body_len[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return None
    return X / scale


def _score_clip(clip, nose_idx, tail_idx):
    body_vec = clip[:, nose_idx, :] - clip[:, tail_idx, :]
    body_len = np.linalg.norm(body_vec, axis=1)
    finite = np.isfinite(body_len) & (body_len > 1e-6)
    if finite.sum() < 3:
        return -np.inf
    cv = float(np.nanstd(body_len[finite]) / max(np.nanmean(body_len[finite]), 1e-6))
    torso = clip[:, [0, 1, 2, 3, 4, 5, 6], :]
    spread = np.linalg.norm(torso - torso[:, [5], :], axis=-1)
    spread95 = float(np.nanpercentile(spread[np.isfinite(spread)], 90)) if np.isfinite(spread).any() else np.inf
    return float(np.isfinite(clip).mean()) - 0.9 * cv - 0.12 * spread95


def run():
    cfg = load_config("config.yaml")
    seqs = s5._load_sequences()
    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    use_bodyparts = cfg["keypoints"]["use_bodyparts"]
    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
    median_fps = int(round(float(np.nanmedian(list(fps_map.values()))))) if fps_map else 10
    pre = max(2, int(round(0.2 * median_fps)))
    post = max(5, int(round(0.6 * median_fps)))

    nose_idx = use_bodyparts.index("nose")
    tail_idx = use_bodyparts.index("tail_base")
    center_idx = use_bodyparts.index("mouse_center") if "mouse_center" in use_bodyparts else None
    skeleton = s5._build_skeleton(use_bodyparts)
    bp_index = {bp: i for i, bp in enumerate(use_bodyparts)}
    edges = [(bp_index[a], bp_index[b]) for a, b in skeleton if a in bp_index and b in bp_index]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        clips = _extract_candidates(seqs, coords_all, use_bodyparts, bodyparts, target, pre, post)
        norm = []
        for clip in clips:
            nclip = _normalize_clip(clip, nose_idx, tail_idx, center_idx)
            if nclip is not None:
                norm.append((_score_clip(nclip, nose_idx, tail_idx), nclip))
        if not norm:
            print(f"Skipping {target}")
            continue
        norm.sort(key=lambda x: x[0], reverse=True)
        best = norm[0][1]
        mid = best.shape[0] // 2
        left = max(0, mid - 1)
        right = min(best.shape[0], mid + 3)
        short = best[left:right]
        lims = s5._expand_limits(s5._safe_plot_limits(short[None, ...], pctl=10), frac=0.20, min_pad=0.20)
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        s5._plot_compact_trajectory_panel(
            ax,
            short,
            edges=edges,
            title=f"Syllable {target}",
            lims=lims,
            shrink=0.76,
            keypoint_colormap="autumn",
        )
        plt.tight_layout()
        fig.savefig(OUTDIR / f"syllable_{target:02d}_trajectory_shortwindow.pdf")
        fig.savefig(OUTDIR / f"syllable_{target:02d}_trajectory_shortwindow.png", dpi=180)
        plt.close(fig)
        print(f"Saved -> {OUTDIR / f'syllable_{target:02d}_trajectory_shortwindow.png'}")


if __name__ == "__main__":
    run()
