#!/usr/bin/env python3
"""
Rebuild selected low-shape syllable trajectory panels from real bout-centered
 keypoint clips rather than keypoint-moseq typical-trajectory summaries.

This is meant as a targeted fallback for syllables whose exported atlas panels
do not preserve an interpretable mouse body shape.
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

TARGETS = (0, 1, 4, 7)
OUTDIR = Path("outputs/05_stage5/syllable_atlas_presence_gt10/kpms_style_trajectories")


def _extract_bout_clips(
    seqs: dict[str, np.ndarray],
    coords_all: dict[str, np.ndarray],
    use_bodyparts: list[str],
    bodyparts: list[str],
    target: int,
    pre: int,
    post: int,
    max_clips: int = 120,
    min_finite_fraction: float = 0.85,
) -> list[np.ndarray]:
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    clips: list[tuple[float, int, np.ndarray]] = []

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
            clips.append((finite_fraction, int(length), clip))

    clips.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [clip for _, _, clip in clips[:max_clips]]


def _normalize_clip(
    clip: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
) -> np.ndarray | None:
    X = np.asarray(clip, dtype=float).copy()
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
    body_len = body_len[np.isfinite(body_len) & (body_len > 1e-6)]
    if len(body_len) == 0:
        return None
    scale = float(np.nanmedian(body_len))
    if not np.isfinite(scale) or scale <= 1e-6:
        return None
    return X / scale


def _aggregate_representative(clips: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(clips, axis=0)
    proto = np.nanmedian(arr, axis=0)

    # Keep the most self-consistent subset to reduce smearing for variable clips.
    diff = arr - proto[None, ...]
    dist = np.nanmean(np.linalg.norm(diff, axis=-1), axis=(1, 2))
    keep_n = max(12, int(round(len(arr) * 0.35)))
    keep = np.argsort(dist)[:keep_n]
    return np.nanmedian(arr[keep], axis=0)


def _select_exemplar(clips: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(clips, axis=0)
    proto = np.nanmedian(arr, axis=0)
    diff = arr - proto[None, ...]
    dist = np.nanmean(np.linalg.norm(diff, axis=-1), axis=(1, 2))
    return arr[int(np.nanargmin(dist))]


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
        raw_clips: list[np.ndarray] = []
        for threshold in (0.85, 0.70, 0.55):
            raw_clips = _extract_bout_clips(
                seqs=seqs,
                coords_all=coords_all,
                use_bodyparts=use_bodyparts,
                bodyparts=bodyparts,
                target=target,
                pre=pre,
                post=post,
                min_finite_fraction=threshold,
            )
            if len(raw_clips) >= 8:
                break
        norm_clips = [
            clip_n
            for clip_n in (
                _normalize_clip(clip, nose_idx=nose_idx, tail_idx=tail_idx, center_idx=center_idx)
                for clip in raw_clips
            )
            if clip_n is not None
        ]
        if not norm_clips:
            print(f"Skipping {target}: no valid clips")
            continue

        proto = _aggregate_representative(norm_clips)
        lims = s5._expand_limits(
            s5._safe_plot_limits(proto[None, ...], pctl=20),
            frac=0.18,
            min_pad=0.22,
        )

        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        s5._plot_compact_trajectory_panel(
            ax,
            proto,
            edges=edges,
            title=f"Syllable {target}",
            lims=lims,
            shrink=0.70,
            keypoint_colormap="autumn",
        )
        plt.tight_layout()
        pdf_path = OUTDIR / f"syllable_{target:02d}_trajectory_boutmedian.pdf"
        png_path = OUTDIR / f"syllable_{target:02d}_trajectory_boutmedian.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"Saved -> {png_path}")

        exemplar = _select_exemplar(norm_clips)
        ex_lims = s5._expand_limits(
            s5._safe_plot_limits(exemplar[None, ...], pctl=20),
            frac=0.18,
            min_pad=0.22,
        )
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        s5._plot_compact_trajectory_panel(
            ax,
            exemplar,
            edges=edges,
            title=f"Syllable {target}",
            lims=ex_lims,
            shrink=0.70,
            keypoint_colormap="autumn",
        )
        plt.tight_layout()
        pdf_path = OUTDIR / f"syllable_{target:02d}_trajectory_exemplar.pdf"
        png_path = OUTDIR / f"syllable_{target:02d}_trajectory_exemplar.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"Saved -> {png_path}")


if __name__ == "__main__":
    run()
