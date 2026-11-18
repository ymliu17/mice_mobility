#!/usr/bin/env python3
"""
Export alternate trajectory views for the 12 presence>10% syllables so they
can be compared directly against the original atlas figures.

Two alternate renderings are generated:
1. densityfree  : keypoint-moseq typical trajectories with density_sample=False
2. rawexemplar  : single representative bout reconstructed from raw keypoints

The script does not overwrite the canonical figures. It writes alternate PNG/PDF
files and two contact-sheet comparison figures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keypoint_moseq as kpms
from keypoint_moseq.util import get_typical_trajectories

from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

OUTDIR = Path("outputs/05_stage5/syllable_atlas_presence_gt10/kpms_style_trajectories")
ATLAS_CSV = Path("outputs/05_stage5/syllable_atlas_presence_gt10/presence_gt10_syllable_atlas.csv")


def _syllable_ids() -> list[int]:
    if ATLAS_CSV.exists():
        df = pd.read_csv(ATLAS_CSV)
        if "syllable" in df.columns and not df.empty:
            return [int(x) for x in df["syllable"].tolist()]
    return [32, 11, 26, 37, 45, 43, 46, 0, 7, 24, 4, 1]


def _extract_bout_clips(
    seqs: dict[str, np.ndarray],
    coords_all: dict[str, np.ndarray],
    use_bodyparts: list[str],
    bodyparts: list[str],
    target: int,
    pre: int,
    post: int,
    max_clips: int = 120,
    min_finite_fraction: float = 0.55,
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


def _normalize_clip(
    clip: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
) -> np.ndarray | None:
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


def _select_exemplar(clips: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(clips, axis=0)
    proto = np.nanmedian(arr, axis=0)
    diff = arr - proto[None, ...]
    dist = np.nanmean(np.linalg.norm(diff, axis=-1), axis=(1, 2))
    return arr[int(np.nanargmin(dist))]


def _plot_single(X: np.ndarray, edges: list[tuple[int, int]], title: str, out_pdf: Path, out_png: Path) -> None:
    lims = s5._expand_limits(
        s5._safe_plot_limits(X[None, ...], pctl=20),
        frac=0.18,
        min_pad=0.22,
    )
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    s5._plot_compact_trajectory_panel(
        ax,
        X,
        edges=edges,
        title=title,
        lims=lims,
        shrink=0.72,
        keypoint_colormap="autumn",
    )
    plt.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _compose_comparison_strip(syllables: list[int], suffix: str, out_name: str) -> None:
    fig, axes = plt.subplots(len(syllables), 2, figsize=(8.2, 3.1 * len(syllables)))
    axes = np.atleast_2d(axes)
    for row, syll in enumerate(syllables):
        left = OUTDIR / f"syllable_{syll:02d}_trajectory.png"
        right = OUTDIR / f"syllable_{syll:02d}_trajectory_{suffix}.png"
        for ax, path, title in (
            (axes[row, 0], left, f"Syllable {syll} original"),
            (axes[row, 1], right, f"Syllable {syll} {suffix}"),
        ):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(title, fontsize=10, pad=6)
            if path.exists():
                try:
                    ax.imshow(plt.imread(path))
                    ax.set_aspect("auto")
                except Exception:
                    ax.text(0.5, 0.5, path.name, ha="center", va="center")
            else:
                ax.text(0.5, 0.5, f"Missing\n{path.name}", ha="center", va="center")
    plt.tight_layout()
    fig.savefig(OUTDIR / out_name, dpi=180)
    plt.close(fig)


def run() -> None:
    cfg = load_config("config.yaml")
    syllables = _syllable_ids()
    seqs = s5._load_sequences()
    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    use_bodyparts = cfg["keypoints"]["use_bodyparts"]
    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
    median_fps = int(round(float(np.nanmedian(list(fps_map.values()))))) if fps_map else 10
    pre = max(2, int(round(0.2 * median_fps)))
    post = max(5, int(round(0.6 * median_fps)))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    nose_idx = use_bodyparts.index("nose")
    tail_idx = use_bodyparts.index("tail_base")
    center_idx = use_bodyparts.index("mouse_center") if "mouse_center" in use_bodyparts else None
    skeleton = s5._build_skeleton(use_bodyparts)
    bp_index = {bp: i for i, bp in enumerate(use_bodyparts)}
    edges = [(bp_index[a], bp_index[b]) for a, b in skeleton if a in bp_index and b in bp_index]

    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    coordinates = {k: v[:, use_idx, :] for k, v in coords_all.items()}
    results = kpms.load_results(path=s5.ALL_H5)

    trajectories = get_typical_trajectories(
        coordinates=coordinates,
        results=results,
        pre=pre,
        post=post,
        min_frequency=0.0,
        min_duration=3,
        bodyparts=use_bodyparts,
        use_bodyparts=use_bodyparts,
        density_sample=False,
    )

    for syll in syllables:
        if int(syll) in trajectories:
            traj = trajectories[int(syll)] * np.array([1.0, -1.0])
            traj = s5._center_trajectories(traj[None, ...])[0]
            traj = s5._rotate_trajectory_to_target_heading(
                traj,
                nose_idx=nose_idx,
                tail_idx=tail_idx,
                target_angle=np.pi / 2.0,
            )
            _plot_single(
                traj,
                edges=edges,
                title=f"Syllable {int(syll)}",
                out_pdf=OUTDIR / f"syllable_{int(syll):02d}_trajectory_densityfree.pdf",
                out_png=OUTDIR / f"syllable_{int(syll):02d}_trajectory_densityfree.png",
            )

        raw_clips = _extract_bout_clips(
            seqs=seqs,
            coords_all=coords_all,
            use_bodyparts=use_bodyparts,
            bodyparts=bodyparts,
            target=int(syll),
            pre=pre,
            post=post,
        )
        norm_clips = [
            clip_n
            for clip_n in (
                _normalize_clip(clip, nose_idx=nose_idx, tail_idx=tail_idx, center_idx=center_idx)
                for clip in raw_clips
            )
            if clip_n is not None
        ]
        if norm_clips:
            exemplar = _select_exemplar(norm_clips)
            _plot_single(
                exemplar,
                edges=edges,
                title=f"Syllable {int(syll)}",
                out_pdf=OUTDIR / f"syllable_{int(syll):02d}_trajectory_rawexemplar.pdf",
                out_png=OUTDIR / f"syllable_{int(syll):02d}_trajectory_rawexemplar.png",
            )

    _compose_comparison_strip(
        syllables,
        suffix="densityfree",
        out_name="presence_gt10_compare_original_vs_densityfree.png",
    )
    _compose_comparison_strip(
        syllables,
        suffix="rawexemplar",
        out_name="presence_gt10_compare_original_vs_rawexemplar.png",
    )
    print("Saved alternate trajectory sets and comparison sheets to", OUTDIR)


if __name__ == "__main__":
    run()
