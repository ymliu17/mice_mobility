#!/usr/bin/env python3
"""
Canonical-skeleton trajectory panels for the five key syllables.

All panels share the exact same skeleton geometry so segment lengths are
identical across syllables. Behavioral differences are shown by the movement
paths of the nose and mouse_center over a short aligned window.
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

from behavior_methodlab.export_key_syllable_slide_trajectories import (
    META,
    TARGETS,
    _extract_candidates,
    _normalize_clip,
    _score_clip,
)
from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

OUTDIR = Path("outputs/05_stage5/key_syllable_slide_panels")


def _select_short_window(
    target: int,
    seqs: dict[str, np.ndarray],
    coords_all: dict[str, np.ndarray],
    use_bodyparts: list[str],
    bodyparts: list[str],
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
    pre: int,
    post: int,
) -> np.ndarray:
    clips = _extract_candidates(seqs, coords_all, use_bodyparts, bodyparts, target, pre, post)
    scored: list[tuple[float, np.ndarray]] = []
    for clip in clips:
        nclip = _normalize_clip(clip, nose_idx, tail_idx, center_idx)
        if nclip is not None:
            scored.append((_score_clip(nclip, nose_idx, tail_idx, center_idx), nclip))
    if not scored:
        raise RuntimeError(f"No valid clip for syllable {target}")
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    mid = best.shape[0] // 2
    return best[max(0, mid - 1): min(best.shape[0], mid + 3)]


def _make_template(clips: dict[int, np.ndarray]) -> np.ndarray:
    refs = np.stack([clip[len(clip) // 2] for clip in clips.values()], axis=0)
    return np.nanmedian(refs, axis=0)


def _global_lims(template: np.ndarray, clips: dict[int, np.ndarray], center_idx: int) -> np.ndarray:
    pts = [template[:, None, :]]
    for clip in clips.values():
        center = clip[:, center_idx, :]
        nose = clip[:, 0, :]
        pts.append(center[:, None, :])
        pts.append(nose[:, None, :])
    arr = np.concatenate(pts, axis=0)
    lims = s5._safe_plot_limits(arr[None, ...], pctl=2)
    span = lims[1] - lims[0]
    max_span = float(np.max(span))
    center = np.nanmean(lims, axis=0)
    half = max(1.55, max_span * 0.62)
    return np.array([center - half, center + half], dtype=float)


def _plot_template_panel(
    ax,
    template: np.ndarray,
    clip: np.ndarray,
    lims: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    subtitle: str,
    accent: str,
    nose_idx: int,
    tail_idx: int,
    center_idx: int,
) -> None:
    ax.set_xlim(lims[0, 0], lims[1, 0])
    ax.set_ylim(lims[0, 1], lims[1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Canonical fixed-geometry skeleton.
    for ii, jj in edges:
        ax.plot(
            template[(ii, jj), 0],
            template[(ii, jj), 1],
            color="#222222",
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )
    ax.scatter(
        template[:, 0],
        template[:, 1],
        s=34,
        color="#F3F4F6",
        edgecolors="#50555A",
        linewidths=0.8,
        zorder=4,
    )
    ax.scatter(template[nose_idx, 0], template[nose_idx, 1], s=42, color="#D1495B", edgecolors="white", linewidths=0.5, zorder=5)
    ax.scatter(template[tail_idx, 0], template[tail_idx, 1], s=40, color="#EDAE49", edgecolors="white", linewidths=0.5, zorder=5)

    # Dynamic cues only.
    center = clip[:, center_idx, :]
    nose = clip[:, nose_idx, :]
    ax.plot(center[:, 0], center[:, 1], linestyle="--", color=accent, linewidth=1.9, alpha=0.95, zorder=1)
    ax.plot(nose[:, 0], nose[:, 1], linestyle="-", color=accent, linewidth=1.5, alpha=0.45, zorder=1)
    alphas = np.linspace(0.35, 1.0, len(center))
    sizes = np.linspace(28, 46, len(center))
    for i in range(len(center)):
        ax.scatter(center[i, 0], center[i, 1], s=sizes[i], color=accent, alpha=alphas[i], edgecolors="white", linewidths=0.4, zorder=2)
        ax.scatter(nose[i, 0], nose[i, 1], s=sizes[i] * 0.7, color="#D1495B", alpha=alphas[i] * 0.9, edgecolors="white", linewidths=0.35, zorder=2)

    if len(center) >= 2:
        ax.annotate("", xy=center[-1], xytext=center[-2], arrowprops=dict(arrowstyle="->", color=accent, lw=1.5), zorder=2)

    ax.set_title(title, fontsize=14, pad=10, color="#111111", weight="bold")
    ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=accent)

    x0 = lims[0, 0] + 0.18
    y0 = lims[0, 1] + 0.14
    ax.plot([x0, x0 + 1.0], [y0, y0], color="#5E646A", linewidth=2.0, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 - 0.08, "same skeleton scale", ha="center", va="top", fontsize=8, color="#5E646A")


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
    center_idx = use_bodyparts.index("mouse_center")
    skeleton = s5._build_skeleton(use_bodyparts)
    bp_index = {bp: i for i, bp in enumerate(use_bodyparts)}
    edges = [(bp_index[a], bp_index[b]) for a, b in skeleton if a in bp_index and b in bp_index]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    clips: dict[int, np.ndarray] = {}
    for target in TARGETS:
        clips[target] = _select_short_window(
            target=target,
            seqs=seqs,
            coords_all=coords_all,
            use_bodyparts=use_bodyparts,
            bodyparts=bodyparts,
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            center_idx=center_idx,
            pre=pre,
            post=post,
        )

    template = _make_template(clips)
    lims = _global_lims(template, clips, center_idx)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18.5, 4.8))
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        meta = META[target]
        group_text = "Enriched in abnormal" if meta["group"] == "abnormal" else "Enriched in normal"
        subtitle = f"{group_text} | canonical skeleton + paths"
        _plot_template_panel(
            ax=ax,
            template=template,
            clip=clips[target],
            lims=lims,
            edges=edges,
            title=f"s{target:02d}",
            subtitle=subtitle,
            accent=meta["accent"],
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            center_idx=center_idx,
        )
    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUTDIR / "key5_canonical_skeleton_paths_strip.pdf")
    fig.savefig(OUTDIR / "key5_canonical_skeleton_paths_strip.png", dpi=220)
    plt.close(fig)

    for target in TARGETS:
        meta = META[target]
        fig, ax = plt.subplots(figsize=(4.0, 4.8))
        group_text = "Enriched in abnormal" if meta["group"] == "abnormal" else "Enriched in normal"
        subtitle = f"{group_text} | canonical skeleton + paths"
        _plot_template_panel(
            ax=ax,
            template=template,
            clip=clips[target],
            lims=lims,
            edges=edges,
            title=f"s{target:02d}",
            subtitle=subtitle,
            accent=meta["accent"],
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            center_idx=center_idx,
        )
        plt.tight_layout()
        fig.savefig(OUTDIR / f"s{target:02d}_canonical_paths.pdf")
        fig.savefig(OUTDIR / f"s{target:02d}_canonical_paths.png", dpi=220)
        plt.close(fig)

    print(f"Saved canonical path panels to {OUTDIR}")


if __name__ == "__main__":
    run()
