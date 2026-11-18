#!/usr/bin/env python3
"""
Redraw slide-ready trajectories for the five key syllables used in the PCA
interpretation section: s26, s11, s00, s24, s07.

Design goals:
- identical body-length normalization across syllables
- one shared coordinate range across all five panels
- preserve both posture and motion cue
- cleaner slide-oriented visual encoding than the default keypoint rainbow
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

OUTDIR = Path("outputs/05_stage5/key_syllable_slide_panels")
TARGETS = [26, 11, 0, 24, 7]

META = {
    26: {
        "group": "abnormal",
        "label": "passive coping / freezing-like",
        "pose_hint": "compact still state",
        "accent": "#B5524A",
    },
    11: {
        "group": "normal",
        "label": "flexible exploratory",
        "pose_hint": "choice-point / scan",
        "accent": "#4C7A86",
    },
    0: {
        "group": "abnormal",
        "label": "secondary inactive stable",
        "pose_hint": "stable inactive state",
        "accent": "#B5524A",
    },
    24: {
        "group": "normal",
        "label": "structured navigation",
        "pose_hint": "focused deliberation",
        "accent": "#4C7A86",
    },
    7: {
        "group": "abnormal",
        "label": "agitated maladaptive",
        "pose_hint": "restless strategy",
        "accent": "#B5524A",
    },
}


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


def _extract_candidates(
    seqs: dict[str, np.ndarray],
    coords_all: dict[str, np.ndarray],
    use_bodyparts: list[str],
    bodyparts: list[str],
    target: int,
    pre: int,
    post: int,
    max_clips: int = 120,
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
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
) -> np.ndarray | None:
    X = _fill_clip_nans(clip)
    if center_idx is not None and np.isfinite(X[:, center_idx, :]).all():
        ref_t = len(X) // 2
        ref_center = X[ref_t, center_idx, :].copy()
        X = X - ref_center[None, None, :]
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


def _score_clip(clip: np.ndarray, nose_idx: int, tail_idx: int, center_idx: int | None) -> float:
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
    radius_p90 = float(np.nanpercentile(radius[np.isfinite(radius)], 90)) if np.isfinite(radius).any() else np.inf
    finite_fraction = float(np.isfinite(clip).mean())
    return finite_fraction - 0.85 * body_cv - 0.12 * step_med - 0.08 * radius_p90


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
    # Keep a fixed short window for all five so posture stays readable.
    left = max(0, mid - 1)
    right = min(best.shape[0], mid + 3)
    return best[left:right]


def _global_limits(clips: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(clips, axis=0)
    lims = s5._safe_plot_limits(stack, pctl=5)
    span = lims[1] - lims[0]
    max_span = float(np.max(span))
    center = np.nanmean(lims, axis=0)
    half = max(1.6, max_span * 0.65)
    return np.array([center - half, center + half], dtype=float)


def _plot_panel(
    ax,
    clip: np.ndarray,
    edges: list[tuple[int, int]],
    lims: np.ndarray,
    title: str,
    subtitle: str,
    accent: str,
    nose_idx: int,
    tail_idx: int,
    center_idx: int | None,
) -> None:
    ax.set_xlim(lims[0, 0], lims[1, 0])
    ax.set_ylim(lims[0, 1], lims[1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if center_idx is not None:
        center = clip[:, center_idx, :]
    else:
        center = np.nanmean(clip, axis=1)

    # Centroid path is shown separately from pose, so motion is easier to read.
    ax.plot(
        center[:, 0],
        center[:, 1],
        linestyle="--",
        color=accent,
        linewidth=1.8,
        alpha=0.7,
        zorder=1,
    )
    ax.scatter(center[0, 0], center[0, 1], s=36, facecolors="white", edgecolors=accent, linewidths=1.4, zorder=2)
    ax.scatter(center[-1, 0], center[-1, 1], s=40, color=accent, edgecolors="white", linewidths=0.6, zorder=3)
    if len(center) >= 2:
        ax.annotate(
            "",
            xy=(center[-1, 0], center[-1, 1]),
            xytext=(center[-2, 0], center[-2, 1]),
            arrowprops=dict(arrowstyle="->", color=accent, lw=1.4, shrinkA=0, shrinkB=0),
            zorder=2,
        )

    # Pose stack: older frames in light gray, final frame emphasized.
    for t in range(len(clip)):
        frac = t / max(len(clip) - 1, 1)
        line_color = "#BFC5CA" if t < len(clip) - 1 else "#1E1E1E"
        alpha = 0.28 + 0.20 * frac if t < len(clip) - 1 else 0.95
        lw = 1.25 if t < len(clip) - 1 else 2.2
        for ii, jj in edges:
            ax.plot(
                clip[t, (ii, jj), 0],
                clip[t, (ii, jj), 1],
                color=line_color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=4 + t,
            )
        body_color = "#D5D9DD" if t < len(clip) - 1 else "#F2F2F2"
        ax.scatter(
            clip[t, :, 0],
            clip[t, :, 1],
            s=20 if t < len(clip) - 1 else 26,
            color=body_color,
            edgecolors=line_color,
            linewidths=0.6,
            alpha=alpha,
            zorder=5 + t,
        )
        # Re-highlight head and tail so orientation is obvious.
        ax.scatter(
            clip[t, nose_idx, 0],
            clip[t, nose_idx, 1],
            s=26 if t < len(clip) - 1 else 34,
            color="#D1495B",
            edgecolors="white",
            linewidths=0.5,
            alpha=alpha if t < len(clip) - 1 else 1.0,
            zorder=7 + t,
        )
        ax.scatter(
            clip[t, tail_idx, 0],
            clip[t, tail_idx, 1],
            s=22 if t < len(clip) - 1 else 30,
            color="#EDAE49",
            edgecolors="white",
            linewidths=0.5,
            alpha=alpha if t < len(clip) - 1 else 1.0,
            zorder=7 + t,
        )

    ax.set_title(title, fontsize=14, pad=10, color="#111111", weight="bold")
    ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=accent)

    # 1 body-length scale bar in every panel.
    x0 = lims[0, 0] + 0.18
    y0 = lims[0, 1] + 0.14
    ax.plot([x0, x0 + 1.0], [y0, y0], color="#5E646A", linewidth=2.0, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 - 0.08, "1 body length", ha="center", va="top", fontsize=8, color="#5E646A")


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

    lims = _global_limits([clips[t] for t in TARGETS])

    # Single strip for slides.
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18.5, 4.6))
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        meta = META[target]
        group_text = "Enriched in abnormal" if meta["group"] == "abnormal" else "Enriched in normal"
        subtitle = f"{group_text} | {meta['pose_hint']}"
        _plot_panel(
            ax=ax,
            clip=clips[target],
            edges=edges,
            lims=lims,
            title=f"s{target:02d}",
            subtitle=subtitle,
            accent=meta["accent"],
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            center_idx=center_idx,
        )
    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUTDIR / "key5_trajectory_strip_same_scale.pdf")
    fig.savefig(OUTDIR / "key5_trajectory_strip_same_scale.png", dpi=220)
    plt.close(fig)

    # Individual cards.
    for target in TARGETS:
        meta = META[target]
        fig, ax = plt.subplots(figsize=(4.0, 4.6))
        group_text = "Enriched in abnormal" if meta["group"] == "abnormal" else "Enriched in normal"
        subtitle = f"{group_text} | {meta['label']}"
        _plot_panel(
            ax=ax,
            clip=clips[target],
            edges=edges,
            lims=lims,
            title=f"s{target:02d}",
            subtitle=subtitle,
            accent=meta["accent"],
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            center_idx=center_idx,
        )
        plt.tight_layout()
        fig.savefig(OUTDIR / f"s{target:02d}_slide_trajectory.pdf")
        fig.savefig(OUTDIR / f"s{target:02d}_slide_trajectory.png", dpi=220)
        plt.close(fig)

    print(f"Saved slide-ready trajectories to {OUTDIR}")


if __name__ == "__main__":
    run()
