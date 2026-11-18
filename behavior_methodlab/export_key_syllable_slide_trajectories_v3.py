#!/usr/bin/env python3
"""
Comparison version for key syllable trajectory panels.

Goal:
- preserve syllable-specific orientation/pattern better than canonical-template
- enforce stronger cross-syllable consistency than whole-skeleton-only scaling

Method:
- center on reference mouse_center frame
- align heading to a common direction
- normalize whole-skeleton scale
- additionally normalize each keypoint's radius relative to mouse_center using a
  template radius vector estimated from the selected clips
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

OUTDIR = Path("outputs/05_stage5/key_syllable_slide_panels_v3")
TARGETS = [26, 11, 0, 24, 7]

META = {
    26: {"group": "abnormal", "label": "passive coping / freezing-like", "accent": "#B5524A"},
    11: {"group": "normal", "label": "flexible exploratory", "accent": "#4C7A86"},
    0: {"group": "abnormal", "label": "secondary inactive stable", "accent": "#B5524A"},
    24: {"group": "normal", "label": "structured navigation", "accent": "#4C7A86"},
    7: {"group": "abnormal", "label": "agitated maladaptive", "accent": "#B5524A"},
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
    for k in range(filled.shape[1]):
        for d in range(filled.shape[2]):
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


def _edge_length_sum(frame: np.ndarray, edges: list[tuple[int, int]]) -> float:
    vals = []
    for i, j in edges:
        if np.isfinite(frame[i]).all() and np.isfinite(frame[j]).all():
            vals.append(np.linalg.norm(frame[i] - frame[j]))
    return float(np.sum(vals)) if vals else np.nan


def _pre_normalize_clip(clip: np.ndarray, nose_idx: int, tail_idx: int, center_idx: int, edges: list[tuple[int, int]]) -> np.ndarray | None:
    X = _fill_clip_nans(clip)
    ref_t = len(X) // 2
    ref_center = X[ref_t, center_idx].copy()
    X = X - ref_center[None, None, :]
    X = X * np.array([1.0, -1.0])
    X = s5._rotate_trajectory_to_target_heading(X, nose_idx=nose_idx, tail_idx=tail_idx, target_angle=np.pi / 2.0)

    scales = np.array([_edge_length_sum(frame, edges) for frame in X], dtype=float)
    finite = np.isfinite(scales) & (scales > 1e-6)
    if finite.sum() < 2:
        return None
    scale = float(np.nanmedian(scales[finite]))
    return X / scale


def _score_clip(clip: np.ndarray, nose_idx: int, tail_idx: int, center_idx: int) -> float:
    body_vec = clip[:, nose_idx, :] - clip[:, tail_idx, :]
    body_len = np.linalg.norm(body_vec, axis=1)
    finite = np.isfinite(body_len) & (body_len > 1e-6)
    if finite.sum() < 3:
        return -np.inf
    body_cv = float(np.nanstd(body_len[finite]) / max(np.nanmean(body_len[finite]), 1e-6))
    center = clip[:, center_idx, :]
    step = np.linalg.norm(np.diff(center, axis=0), axis=1)
    step_med = float(np.nanmedian(step[np.isfinite(step)])) if np.isfinite(step).any() else 0.0
    return float(np.isfinite(clip).mean()) - 0.9 * body_cv - 0.1 * step_med


def _make_radius_template(best_raw: dict[int, np.ndarray], center_idx: int) -> np.ndarray:
    mids = []
    for clip in best_raw.values():
        mid = clip[len(clip) // 2]
        center = mid[center_idx]
        mids.append(np.linalg.norm(mid - center[None, :], axis=1))
    return np.nanmedian(np.stack(mids, axis=0), axis=0)


def _apply_radius_template(clip: np.ndarray, center_idx: int, radius_template: np.ndarray, fixed_idx: set[int]) -> np.ndarray:
    X = np.array(clip, copy=True, dtype=float)
    for t in range(X.shape[0]):
        center = X[t, center_idx].copy()
        for k in range(X.shape[1]):
            if k == center_idx or k in fixed_idx:
                continue
            vec = X[t, k] - center
            r = float(np.linalg.norm(vec))
            target_r = float(radius_template[k])
            if not np.isfinite(r) or r < 1e-8 or not np.isfinite(target_r):
                continue
            X[t, k] = center + vec * (target_r / r)
    return X


def _short_window(clip: np.ndarray) -> np.ndarray:
    mid = clip.shape[0] // 2
    return clip[max(0, mid - 1): min(clip.shape[0], mid + 3)]


def _global_limits(clips: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(clips, axis=0)
    lims = s5._safe_plot_limits(stack, pctl=1)
    span = lims[1] - lims[0]
    max_span = float(np.max(span))
    center = np.nanmean(lims, axis=0)
    half = max(0.88, max_span * 0.34)
    return np.array([center - half, center + half], dtype=float)


def _plot_panel(ax, clip: np.ndarray, edges: list[tuple[int, int]], lims: np.ndarray, title: str, subtitle: str, accent: str, nose_idx: int, tail_idx: int, center_idx: int) -> None:
    ax.set_xlim(lims[0, 0], lims[1, 0])
    ax.set_ylim(lims[0, 1], lims[1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    center = clip[:, center_idx, :]
    ax.plot(center[:, 0], center[:, 1], linestyle="--", color=accent, linewidth=1.3, alpha=0.5, zorder=1)

    for t in range(len(clip)):
        is_final = (t == len(clip) - 1)
        line_color = "#BFC5CA" if not is_final else "#232323"
        alpha = 0.24 + 0.20 * (t / max(len(clip) - 1, 1)) if not is_final else 0.96
        lw = 1.2 if not is_final else 2.1
        for ii, jj in edges:
            ax.plot(clip[t, (ii, jj), 0], clip[t, (ii, jj), 1], color=line_color, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=4 + t)
        ax.scatter(clip[t, :, 0], clip[t, :, 1], s=18 if not is_final else 24, color="#ECEFF1" if not is_final else "#F5F5F5",
                   edgecolors=line_color, linewidths=0.55, alpha=alpha, zorder=5 + t)
        ax.scatter(clip[t, nose_idx, 0], clip[t, nose_idx, 1], s=22 if not is_final else 30, color="#D1495B",
                   edgecolors="white", linewidths=0.45, alpha=alpha if not is_final else 1.0, zorder=6 + t)
        ax.scatter(clip[t, tail_idx, 0], clip[t, tail_idx, 1], s=18 if not is_final else 26, color="#EDAE49",
                   edgecolors="white", linewidths=0.45, alpha=alpha if not is_final else 1.0, zorder=6 + t)

    ax.set_title(title, fontsize=14, pad=10, color="#111111", weight="bold")
    ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=accent)
    x0 = lims[0, 0] + 0.16
    y0 = lims[0, 1] + 0.13
    ax.plot([x0, x0 + 1.0], [y0, y0], color="#5E646A", linewidth=2.0, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 - 0.075, "radius-template normalized", ha="center", va="top", fontsize=8, color="#5E646A")


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

    best_raw: dict[int, np.ndarray] = {}
    for target in TARGETS:
        clips = _extract_candidates(seqs, coords_all, use_bodyparts, bodyparts, target, pre, post)
        scored = []
        for clip in clips:
            nclip = _pre_normalize_clip(clip, nose_idx, tail_idx, center_idx, edges)
            if nclip is not None:
                scored.append((_score_clip(nclip, nose_idx, tail_idx, center_idx), nclip))
        if not scored:
            raise RuntimeError(f"No valid clip for syllable {target}")
        scored.sort(key=lambda x: x[0], reverse=True)
        best_raw[target] = scored[0][1]

    radius_template = _make_radius_template(best_raw, center_idx)
    fixed_idx = {nose_idx, tail_idx}
    best_norm = {target: _apply_radius_template(clip, center_idx, radius_template, fixed_idx) for target, clip in best_raw.items()}
    clips = {target: _short_window(best_norm[target]) for target in TARGETS}
    lims = _global_limits([clips[t] for t in TARGETS])

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18.5, 4.8))
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        meta = META[target]
        group_text = "Enriched in abnormal" if meta["group"] == "abnormal" else "Enriched in normal"
        subtitle = f"{group_text} | radius-template normalized"
        _plot_panel(ax, clips[target], edges, lims, f"s{target:02d}", subtitle, meta["accent"], nose_idx, tail_idx, center_idx)
    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUTDIR / "key5_trajectory_strip_radius_template.pdf")
    fig.savefig(OUTDIR / "key5_trajectory_strip_radius_template.png", dpi=220)
    plt.close(fig)

    for target in TARGETS:
        meta = META[target]
        fig, ax = plt.subplots(figsize=(4.1, 4.8))
        group_text = "Enriched in abnormal" if meta["group"] == "abnormal" else "Enriched in normal"
        subtitle = f"{group_text} | {meta['label']}"
        _plot_panel(ax, clips[target], edges, lims, f"s{target:02d}", subtitle, meta["accent"], nose_idx, tail_idx, center_idx)
        plt.tight_layout()
        fig.savefig(OUTDIR / f"s{target:02d}_radius_template.png", dpi=220)
        fig.savefig(OUTDIR / f"s{target:02d}_radius_template.pdf")
        plt.close(fig)

    print(f"Saved radius-template comparison trajectories to {OUTDIR}")


if __name__ == "__main__":
    run()
