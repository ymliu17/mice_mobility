#!/usr/bin/env python3
"""
Final 3+2 presentation layout for the five key syllables.

Features:
- uses the v2 preserved-pose / unified-whole-skeleton-scale representation
- 3+2 card layout for slides
- one shared blurred background abstract center-symmetric template
- dotted connector lines from abstract template keypoints to the final syllable
  pose to emphasize temporal change rather than a fixed default posture
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

from behavior_methodlab.export_key_syllable_slide_trajectories_v2 import (
    META,
    TARGETS,
    _collect_raw_windows,
    _global_limits,
    _renormalize_best_clips,
    _short_window,
)
from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

OUTDIR = Path("outputs/05_stage5/key_syllable_slide_panels_final")

SIGNED_LOADING_SIGNS = {
    26: ("+", "+", "-"),
    11: ("-", "-", "+"),
    0: ("+", "+", "~"),
    24: ("-", "-", "-"),
    7: ("+", "+", "+"),
}

CARD_TEXT = {
    26: {
        "title": "s26",
        "interpretation": "Passive coping / freezing-like state",
        "relevance": "Enriched in abnormal",
    },
    11: {
        "title": "s11",
        "interpretation": "Flexible exploratory / choice-point behavior",
        "relevance": "Enriched in normal",
    },
    0: {
        "title": "s00",
        "interpretation": "Secondary inactive stable state",
        "relevance": "Enriched in abnormal",
    },
    24: {
        "title": "s24",
        "interpretation": "Focused deliberation / structured navigation",
        "relevance": "Enriched in normal",
    },
    7: {
        "title": "s07",
        "interpretation": "Agitated maladaptive strategy",
        "relevance": "Enriched in abnormal",
    },
}

def _make_template(clips: dict[int, np.ndarray], center_idx: int, nose_idx: int, tail_idx: int) -> np.ndarray:
    refs = np.stack([clip[len(clip) // 2] for clip in clips.values()], axis=0)

    # Start from the empirical median geometry, centered at mouse_center.
    med = np.nanmedian(refs, axis=0)
    med = med - med[center_idx:center_idx + 1]
    template = np.array(med, copy=True, dtype=float)

    # Build a symmetric, geometrically plausible scaffold from the real data.
    # For the spine-like points, keep the empirical median y-position but force x=0.
    axis_indices = [nose_idx, 3, 4, center_idx, tail_idx]
    for idx in axis_indices:
        if idx < template.shape[0]:
            template[idx, 0] = 0.0

    # Symmetrize ears using empirical left/right spread and common height.
    if template.shape[0] > 2:
        ear_x = float(np.nanmedian(np.abs(refs[:, [1, 2], 0])))
        ear_y = float(np.nanmedian(refs[:, [1, 2], 1] - refs[:, center_idx:center_idx + 1, 1]))
        ear_x = max(ear_x, 0.10)
        template[1] = np.array([-ear_x, ear_y], dtype=float)
        template[2] = np.array([ear_x, ear_y], dtype=float)

    # Re-center exactly at mouse_center.
    template = template - template[center_idx:center_idx + 1]

    # Mild regularization so the spine ordering is monotonic and distances remain sane.
    if template.shape[0] > tail_idx:
        nose_y = max(float(template[nose_idx, 1]), 0.18)
        neck_y = float(template[3, 1]) if template.shape[0] > 3 else nose_y * 0.55
        back_y = float(template[4, 1]) if template.shape[0] > 4 else -0.08
        tail_y = min(float(template[tail_idx, 1]), -0.18)
        neck_y = min(neck_y, nose_y - 0.06)
        neck_y = max(neck_y, 0.04)
        back_y = min(back_y, -0.02)
        back_y = max(back_y, tail_y + 0.06)
        template[nose_idx, 1] = nose_y
        if template.shape[0] > 3:
            template[3, 1] = neck_y
        if template.shape[0] > 4:
            template[4, 1] = back_y
        template[center_idx, 1] = 0.0
        template[tail_idx, 1] = tail_y

    return template


def _plot_blurred_template(
    ax,
    template: np.ndarray,
    nose_idx: int,
    tail_idx: int,
) -> None:
    # Soft halo around a neutral center-symmetric point scaffold.
    halo_specs = [
        (78, 0.08, "#E2E6EB"),
        (54, 0.12, "#D9DEE4"),
        (34, 0.17, "#D2D8DE"),
    ]
    for size, alpha, color in halo_specs:
        ax.scatter(
            template[:, 0],
            template[:, 1],
            s=size,
            color=color,
            edgecolors="none",
            alpha=alpha,
            zorder=0,
        )
    # A faint symmetric spine cue, but still no directional posture.
    ax.plot(
        [0.0, 0.0],
        [template[tail_idx, 1], template[nose_idx, 1]],
        color="#DCE1E6",
        linewidth=6.0,
        alpha=0.12,
        solid_capstyle="round",
        zorder=0,
    )
    ax.scatter(template[nose_idx, 0], template[nose_idx, 1], s=54, color="#E7A2AD", edgecolors="none", alpha=0.30, zorder=0)
    ax.scatter(template[tail_idx, 0], template[tail_idx, 1], s=50, color="#F4D29A", edgecolors="none", alpha=0.32, zorder=0)


def _plot_connectors(
    ax,
    template: np.ndarray,
    final_pose: np.ndarray,
    center_idx: int,
) -> None:
    for k in range(template.shape[0]):
        color = "#C7CDD4" if k != center_idx else "#B9C6D2"
        ax.plot(
            [template[k, 0], final_pose[k, 0]],
            [template[k, 1], final_pose[k, 1]],
            linestyle=(0, (2.0, 2.6)),
            color=color,
            linewidth=1.05,
            alpha=0.75,
            zorder=1,
        )


def _plot_pose_stack(
    ax,
    clip: np.ndarray,
    edges: list[tuple[int, int]],
    nose_idx: int,
    tail_idx: int,
    center_idx: int,
    accent: str,
) -> None:
    center = clip[:, center_idx, :]
    ax.plot(
        center[:, 0],
        center[:, 1],
        linestyle="--",
        color=accent,
        linewidth=1.35,
        alpha=0.50,
        zorder=2,
    )
    ax.scatter(
        center[0, 0],
        center[0, 1],
        s=18,
        facecolors="white",
        edgecolors=accent,
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        center[-1, 0],
        center[-1, 1],
        s=24,
        color=accent,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    for t in range(len(clip)):
        is_final = (t == len(clip) - 1)
        frac = t / max(len(clip) - 1, 1)
        line_color = "#C2C7CD" if not is_final else "#242424"
        alpha = 0.20 + 0.22 * frac if not is_final else 0.98
        lw = 1.05 if not is_final else 2.15
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
        ax.scatter(
            clip[t, :, 0],
            clip[t, :, 1],
            s=16 if not is_final else 24,
            color="#ECEFF1" if not is_final else "#F7F7F7",
            edgecolors=line_color,
            linewidths=0.5,
            alpha=alpha,
            zorder=5 + t,
        )
        ax.scatter(
            clip[t, nose_idx, 0],
            clip[t, nose_idx, 1],
            s=20 if not is_final else 30,
            color="#D1495B",
            edgecolors="white",
            linewidths=0.45,
            alpha=alpha if not is_final else 1.0,
            zorder=6 + t,
        )
        ax.scatter(
            clip[t, tail_idx, 0],
            clip[t, tail_idx, 1],
            s=18 if not is_final else 26,
            color="#EDAE49",
            edgecolors="white",
            linewidths=0.45,
            alpha=alpha if not is_final else 1.0,
            zorder=6 + t,
        )


def _style_panel(
    ax,
    lims: np.ndarray,
) -> None:
    ax.set_xlim(lims[0, 0], lims[1, 0])
    ax.set_ylim(lims[0, 1], lims[1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_signed_loading_strip(ax, target: int) -> None:
    labels = ("F", "S", "X")
    signs = SIGNED_LOADING_SIGNS[target]
    x_axis = 0.92
    half_len = 0.055
    ys = (0.62, 0.50, 0.38)
    axis_color = "#D7DCE1"
    pos_color = "#C46A6A"
    neg_color = "#5A7FA6"
    neu_color = "#9AA0A6"

    for y, label, sign in zip(ys, labels, signs):
        ax.text(
            x_axis - 0.075,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#6A727A",
        )
        ax.plot(
            [x_axis, x_axis],
            [y - 0.04, y + 0.04],
            transform=ax.transAxes,
            color=axis_color,
            linewidth=1.1,
            solid_capstyle="round",
            zorder=20,
        )
        if sign == "+":
            ax.plot(
                [x_axis, x_axis + half_len],
                [y, y],
                transform=ax.transAxes,
                color=pos_color,
                linewidth=2.3,
                solid_capstyle="round",
                zorder=21,
            )
        elif sign == "-":
            ax.plot(
                [x_axis - half_len, x_axis],
                [y, y],
                transform=ax.transAxes,
                color=neg_color,
                linewidth=2.3,
                solid_capstyle="round",
                zorder=21,
            )
        else:
            ax.scatter(
                [x_axis],
                [y],
                transform=ax.transAxes,
                s=12,
                color=neu_color,
                zorder=21,
            )


def _plot_card_text(ax, target: int) -> None:
    meta = CARD_TEXT[target]
    relevance_color = "#B85C5C" if "abnormal" in meta["relevance"].lower() else "#4F7DAF"
    ax.text(
        0.03,
        0.97,
        meta["title"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        color="#101010",
        weight="bold",
    )
    ax.text(
        0.03,
        0.09,
        meta["interpretation"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.9,
        color="#424A52",
    )
    ax.text(
        0.03,
        0.035,
        meta["relevance"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=relevance_color,
        weight="bold",
    )


def _expanded_limits(lims: np.ndarray, zoom: float = 0.80) -> np.ndarray:
    center = np.nanmean(lims, axis=0)
    half = (lims[1] - lims[0]) * 0.5 * zoom
    min_half = np.array([0.88, 0.88], dtype=float)
    half = np.maximum(half, min_half)
    return np.vstack([center - half, center + half])


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

    raw_best, target_scale = _collect_raw_windows(
        seqs,
        coords_all,
        use_bodyparts,
        bodyparts,
        nose_idx,
        tail_idx,
        center_idx,
        pre,
        post,
    )
    norm_best = _renormalize_best_clips(raw_best, target_scale, edges)
    clips = {target: _short_window(norm_best[target]) for target in TARGETS}
    template = _make_template(clips, center_idx=center_idx, nose_idx=nose_idx, tail_idx=tail_idx)
    lims = _expanded_limits(_global_limits([clips[t] for t in TARGETS]), zoom=0.78)

    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 6, wspace=0.55, hspace=0.70)
    panel_axes = {
        26: fig.add_subplot(gs[0, 0:2]),
        11: fig.add_subplot(gs[0, 2:4]),
        0: fig.add_subplot(gs[0, 4:6]),
        24: fig.add_subplot(gs[1, 1:3]),
        7: fig.add_subplot(gs[1, 3:5]),
    }

    for target, ax in panel_axes.items():
        meta = META[target]
        _style_panel(ax, lims)
        _plot_blurred_template(ax, template, nose_idx, tail_idx)
        _plot_connectors(ax, template, clips[target][-1], center_idx)
        _plot_pose_stack(ax, clips[target], edges, nose_idx, tail_idx, center_idx, meta["accent"])
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template.pdf")
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template.png", dpi=220)
    plt.close(fig)

    for target in TARGETS:
        meta = META[target]
        fig, ax = plt.subplots(figsize=(4.35, 4.95))
        _style_panel(ax, lims)
        _plot_blurred_template(ax, template, nose_idx, tail_idx)
        _plot_connectors(ax, template, clips[target][-1], center_idx)
        _plot_pose_stack(ax, clips[target], edges, nose_idx, tail_idx, center_idx, meta["accent"])
        plt.tight_layout()
        fig.savefig(OUTDIR / f"s{target:02d}_card_blurred_template.pdf")
        fig.savefig(OUTDIR / f"s{target:02d}_card_blurred_template.png", dpi=220)
        plt.close(fig)

    # Clean presentation version with short labels in filename for easy use.
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 6, wspace=0.48, hspace=0.62)
    panel_axes = {
        26: fig.add_subplot(gs[0, 0:2]),
        11: fig.add_subplot(gs[0, 2:4]),
        0: fig.add_subplot(gs[0, 4:6]),
        24: fig.add_subplot(gs[1, 1:3]),
        7: fig.add_subplot(gs[1, 3:5]),
    }
    for target, ax in panel_axes.items():
        meta = META[target]
        _style_panel(ax, lims)
        _plot_blurred_template(ax, template, nose_idx, tail_idx)
        _plot_connectors(ax, template, clips[target][-1], center_idx)
        _plot_pose_stack(ax, clips[target], edges, nose_idx, tail_idx, center_idx, meta["accent"])
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template_clean.pdf")
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template_clean.png", dpi=240)
    plt.close(fig)

    # Comparison version: same clean layout plus a minimal signed loading strip.
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 6, wspace=0.48, hspace=0.62)
    panel_axes = {
        26: fig.add_subplot(gs[0, 0:2]),
        11: fig.add_subplot(gs[0, 2:4]),
        0: fig.add_subplot(gs[0, 4:6]),
        24: fig.add_subplot(gs[1, 1:3]),
        7: fig.add_subplot(gs[1, 3:5]),
    }
    for target, ax in panel_axes.items():
        meta = META[target]
        _style_panel(ax, lims)
        _plot_blurred_template(ax, template, nose_idx, tail_idx)
        _plot_connectors(ax, template, clips[target][-1], center_idx)
        _plot_pose_stack(ax, clips[target], edges, nose_idx, tail_idx, center_idx, meta["accent"])
        _plot_signed_loading_strip(ax, target)
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template_signedloadings.pdf")
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template_signedloadings.png", dpi=240)
    plt.close(fig)

    # Presentation version with concise labels and group relevance.
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 6, wspace=0.48, hspace=0.62)
    panel_axes = {
        26: fig.add_subplot(gs[0, 0:2]),
        11: fig.add_subplot(gs[0, 2:4]),
        0: fig.add_subplot(gs[0, 4:6]),
        24: fig.add_subplot(gs[1, 1:3]),
        7: fig.add_subplot(gs[1, 3:5]),
    }
    for target, ax in panel_axes.items():
        meta = META[target]
        _style_panel(ax, lims)
        _plot_blurred_template(ax, template, nose_idx, tail_idx)
        _plot_connectors(ax, template, clips[target][-1], center_idx)
        _plot_pose_stack(ax, clips[target], edges, nose_idx, tail_idx, center_idx, meta["accent"])
        _plot_signed_loading_strip(ax, target)
        _plot_card_text(ax, target)
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template_annotated.pdf")
    fig.savefig(OUTDIR / "key5_cards_3plus2_blurred_template_annotated.png", dpi=240)
    plt.close(fig)

    print(f"Saved 3+2 blurred-template cards to {OUTDIR}")


if __name__ == "__main__":
    run()
