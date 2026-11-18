#!/usr/bin/env python3
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

from behavior_methodlab.export_shortwindow_lowshape import (
    _extract_candidates,
    _normalize_clip,
    _score_clip,
)
from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

OUTDIR = Path("outputs/05_stage5/syllable_atlas_presence_gt10/kpms_style_trajectories")


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

    clips = _extract_candidates(seqs, coords_all, use_bodyparts, bodyparts, 1, pre, post)
    scored = []
    for clip in clips:
        nclip = _normalize_clip(clip, nose_idx, tail_idx, center_idx)
        if nclip is not None:
            score = _score_clip(nclip, nose_idx, tail_idx)
            scored.append((score, nclip))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:9]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if idx >= len(top):
            continue
        score, clip = top[idx]
        mid = clip.shape[0] // 2
        short = clip[max(0, mid - 1): min(clip.shape[0], mid + 2)]
        lims = s5._expand_limits(s5._safe_plot_limits(short[None, ...], pctl=5), frac=0.22, min_pad=0.18)
        s5._plot_compact_trajectory_panel(
            ax,
            short,
            edges=edges,
            title=f"#{idx+1} score={score:.2f}",
            lims=lims,
            shrink=0.84,
            keypoint_colormap="autumn",
        )
    plt.tight_layout()
    fig.savefig(OUTDIR / "syllable_01_candidate_gallery.png", dpi=180)
    plt.close(fig)
    print("Saved ->", OUTDIR / "syllable_01_candidate_gallery.png")


if __name__ == "__main__":
    run()
