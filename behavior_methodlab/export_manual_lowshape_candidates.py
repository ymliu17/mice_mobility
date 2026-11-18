#!/usr/bin/env python3
"""
Generate stricter single-bout trajectory candidates for low-shape syllables.

The goal is not to average across variable bouts, but to pick one cleaner,
anatomically plausible clip per syllable for atlas replacement decisions.
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


def _extract_candidate_clips(
    seqs: dict[str, np.ndarray],
    coords_all: dict[str, np.ndarray],
    use_bodyparts: list[str],
    bodyparts: list[str],
    target: int,
    pre: int,
    post: int,
    max_clips: int = 160,
    min_finite_fraction: float = 0.45,
) -> list[np.ndarray]:
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    candidates: list[tuple[float, int, np.ndarray]] = []

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
            if finite_fraction < min_finite_fraction:
                continue
            candidates.append((finite_fraction, int(length), clip))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [clip for _, _, clip in candidates[:max_clips]]


def _normalize_clip(
    clip: np.ndarray,
    use_bodyparts: list[str],
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
) -> np.ndarray | None:
    X = _fill_clip_nans(clip)
    if center_idx is not None:
        center_trace = X[:, center_idx, :]
        if np.isfinite(center_trace).all():
            X = X - center_trace[:, None, :]
        else:
            X = s5._center_trajectories(X[None, ...])[0]
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
    if finite.sum() < max(3, len(body_len) // 2):
        return None
    scale = float(np.nanmedian(body_len[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return None
    return X / scale


def _quality_score(
    clip: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
) -> float:
    body_vec = clip[:, nose_idx, :] - clip[:, tail_idx, :]
    body_len = np.linalg.norm(body_vec, axis=1)
    finite = np.isfinite(body_len) & (body_len > 1e-6)
    if finite.sum() < 3:
        return -np.inf
    body_cv = float(np.nanstd(body_len[finite]) / max(np.nanmean(body_len[finite]), 1e-6))

    if center_idx is not None:
        center = clip[:, center_idx, :]
    else:
        center = np.nanmean(clip, axis=1)
    step = np.linalg.norm(np.diff(center, axis=0), axis=1)
    step_med = float(np.nanmedian(step[np.isfinite(step)])) if np.isfinite(step).any() else 0.0

    radius = np.linalg.norm(clip - center[:, None, :], axis=-1)
    radius_p95 = float(np.nanpercentile(radius[np.isfinite(radius)], 95)) if np.isfinite(radius).any() else np.inf

    finite_fraction = float(np.isfinite(clip).mean())
    return finite_fraction - 0.8 * body_cv - 0.15 * step_med - 0.08 * radius_p95


def run() -> None:
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
        raw = _extract_candidate_clips(
            seqs=seqs,
            coords_all=coords_all,
            use_bodyparts=use_bodyparts,
            bodyparts=bodyparts,
            target=target,
            pre=pre,
            post=post,
        )
        scored: list[tuple[float, np.ndarray]] = []
        for clip in raw:
            norm = _normalize_clip(
                clip,
                use_bodyparts=use_bodyparts,
                nose_idx=nose_idx,
                tail_idx=tail_idx,
                center_idx=center_idx,
            )
            if norm is None:
                continue
            scored.append((_quality_score(norm, nose_idx, tail_idx, center_idx), norm))
        if not scored:
            print(f"Skipping {target}: no scored clips")
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        lims = s5._expand_limits(
            s5._safe_plot_limits(best[None, ...], pctl=15),
            frac=0.20,
            min_pad=0.22,
        )
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        s5._plot_compact_trajectory_panel(
            ax,
            best,
            edges=edges,
            title=f"Syllable {target}",
            lims=lims,
            shrink=0.72,
            keypoint_colormap="autumn",
        )
        plt.tight_layout()
        pdf_path = OUTDIR / f"syllable_{target:02d}_trajectory_manualbest.pdf"
        png_path = OUTDIR / f"syllable_{target:02d}_trajectory_manualbest.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"Saved -> {png_path}")


if __name__ == "__main__":
    run()
