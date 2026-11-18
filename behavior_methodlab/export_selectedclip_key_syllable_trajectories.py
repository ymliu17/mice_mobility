#!/usr/bin/env python3
"""
Export key syllable trajectories from fixed representative 2-second clips.

This variant is intentionally clip-driven rather than score-driven:
- it reconstructs the representative clip table used to name the exported mp4s
- locks onto a user-selected clip for each of the five key syllables
- extracts the matching processed-keypoint segment
- chooses the most stable 10-frame window within that 2-second segment
- renders both per-syllable panels and a combined strip
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_config, load_preprocessed

OUTDIR = Path("outputs/05_stage5/key_syllable_selectedclip_trajectories")
TARGETS = [26, 11, 0, 24, 7]
WINDOW_FRAMES = 10
CLIP_SECONDS = 2.0
GIF_FRAME_DURATION_SEC = 0.18
HEADING_TARGET_ANGLE = 0.0

TARGET_SPECS = {
    0: {
        "clip_name": "s00_oft_pre_05.mp4",
        "session": "S2280005 021726Tue  Gi-01 pre_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "bout_start_frame_processed": 3265,
        "bout_duration_frames_processed": 529,
        "processed_frame_count": 20760,
        "effective_fps": 30.0,
    },
    7: {
        "clip_name": "s07_ymaze_post_05.mp4",
        # Replaced representative clip; content-matched back to
        # s07_ymaze_pre_02.mp4 while respecting the allowed source syllables
        # for display syllable 07: {7, 45, 43}.
        "session": "POD-1 Y maze 240218-1_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn_mouse1",
        "bout_start_frame_processed": 1745,
        "bout_duration_frames_processed": 17,
        "processed_frame_count": 3383,
        "effective_fps": 30.0,
    },
    11: {
        "clip_name": "s11_oft-duo_post_04.mp4",
        # In surgery paired videos, odd mouse IDs are the right-side mouse.
        # Use mouse5 here so the trajectory follows the right mouse in the clip.
        "session": "POD+1 OFT 240204-5 -6_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn_mouse5",
        "bout_start_frame_processed": 1105,
        "bout_duration_frames_processed": 368,
        "processed_frame_count": 3629,
        "effective_fps": 30.0,
    },
    24: {
        "clip_name": "s24_oft_post1_03.mp4",
        # Replaced representative clip; exact hash match to
        # s07_oft_post1_04.mp4 selected by export_key_syllable_video_clips.py.
        "session": "S2330005  022826Sat  Gi-11 post1_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "bout_start_frame_processed": 2372,
        "bout_duration_frames_processed": 4,
        "processed_frame_count": 20040,
        "effective_fps": 30.0,
    },
    26: {
        "clip_name": "s26_oft_post1_01.mp4",
        "session": "S2330001  022826Sat  Gi-07 post1_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "bout_start_frame_processed": 10603,
        "bout_duration_frames_processed": 408,
        "processed_frame_count": 20790,
        "effective_fps": 30.0,
    },
}

META = {
    26: {"accent": "#B5524A", "subtitle": "s26_oft_post1_01"},
    11: {"accent": "#4C7A86", "subtitle": "s11_oft-duo_post_04"},
    0: {"accent": "#B5524A", "subtitle": "s00_oft_pre_05"},
    24: {"accent": "#4C7A86", "subtitle": "s24_oft_post1_03"},
    7: {"accent": "#B5524A", "subtitle": "s07_ymaze_post_05"},
}

WINDOW_LEFT_OVERRIDES = {
    # Frame 0 shows a visible outlier in the exported trajectory, so skip it.
    7: 1,
}

USE_LAST_WINDOW = {
    24,
}

CLIP_START_OFFSETS = {
    # Shift the 2 s pose clip slightly later so the selected right mouse does not
    # include the unstable transition frame at the start of the last 10-pose run.
    11: 2,
}

UNFILTERED_EDGES = {
    frozenset(("mid_back", "mouse_center")),
    frozenset(("mouse_center", "tail_base")),
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


def _fill_window_with_context(
    raw_window: np.ndarray,
    raw_clip: np.ndarray,
    center_idx: int,
) -> np.ndarray:
    filled = np.asarray(raw_window, dtype=float).copy()
    clip_context = _fill_clip_nans(raw_clip)

    for k in range(filled.shape[1]):
        for d in range(filled.shape[2]):
            if np.isfinite(filled[:, k, d]).any():
                continue
            source = clip_context[:, k, d]
            finite = np.isfinite(source)
            if finite.any():
                filled[:, k, d] = float(np.nanmedian(source[finite]))

    filled = _fill_clip_nans(filled)

    for t in range(filled.shape[0]):
        if not np.isfinite(filled[t, center_idx]).all():
            frame = filled[t]
            finite_rows = np.isfinite(frame).all(axis=1)
            if finite_rows.any():
                filled[t, center_idx] = np.nanmedian(frame[finite_rows], axis=0)
            else:
                filled[t, center_idx] = np.zeros(2, dtype=float)

    for t in range(filled.shape[0]):
        center = filled[t, center_idx] if np.isfinite(filled[t, center_idx]).all() else np.zeros(2, dtype=float)
        for k in range(filled.shape[1]):
            if not np.isfinite(filled[t, k]).all():
                filled[t, k] = center

    return filled


def _edge_length_sum(frame: np.ndarray, edges: list[tuple[int, int]]) -> float:
    vals = []
    for i, j in edges:
        if np.isfinite(frame[i]).all() and np.isfinite(frame[j]).all():
            vals.append(np.linalg.norm(frame[i] - frame[j]))
    return float(np.sum(vals)) if vals else np.nan


def _build_selected_manifest() -> pd.DataFrame:
    rows = []
    for syllable, spec in TARGET_SPECS.items():
        rows.append({
            "syllable": int(syllable),
            **spec,
            "clip_stem": str(spec["clip_name"]).replace(".mp4", ""),
        })
    return pd.DataFrame(rows).sort_values("syllable").reset_index(drop=True)


def _load_selected_sessions(cfg: dict, sessions: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dir"])
    coords_all: dict[str, np.ndarray] = {}
    bodyparts: list[str] = []
    for session in sessions:
        path = preprocessed_dir / f"{session}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed file for session: {path}")
        coords, _, loaded_bodyparts, _ = load_preprocessed(str(path))
        coords_all[session] = coords
        if not bodyparts:
            bodyparts = loaded_bodyparts
    return coords_all, bodyparts


def _extract_processed_clip(
    row: pd.Series,
    coords_all: dict[str, np.ndarray],
    use_bodyparts: list[str],
    bodyparts: list[str],
    pose_fps: float,
    start_offset: int = 0,
) -> tuple[np.ndarray, int, int]:
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    session = str(row["session"])
    coords = coords_all.get(session)
    if coords is None:
        raise KeyError(f"Missing preprocessed coordinates for session: {session}")
    coords = coords[:, use_idx, :]

    usable = min(len(coords), int(row["processed_frame_count"]))
    midpoint = float(row["bout_start_frame_processed"]) + (float(row["bout_duration_frames_processed"]) / 2.0)
    clip_frames = max(WINDOW_FRAMES, int(round(float(pose_fps) * CLIP_SECONDS)))

    start = int(round(midpoint - (clip_frames / 2.0))) + int(start_offset)
    end = start + clip_frames
    if start < 0:
        end = min(usable, end - start)
        start = 0
    if end > usable:
        start = max(0, start - (end - usable))
        end = usable
    if end - start < WINDOW_FRAMES:
        raise RuntimeError(f"Clip window for {row['clip_name']} is shorter than {WINDOW_FRAMES} frames.")
    return coords[start:end].astype(float, copy=True), start, end


def _score_stability_window(
    raw_window: np.ndarray,
    raw_clip: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int,
) -> float:
    finite_fraction = float(np.isfinite(raw_window).mean())

    window = _fill_window_with_context(raw_window, raw_clip, center_idx)
    body_vec = window[:, nose_idx, :] - window[:, tail_idx, :]
    body_len = np.linalg.norm(body_vec, axis=1)
    finite_body = np.isfinite(body_len) & (body_len > 1e-6)
    if finite_body.sum() < 4:
        return finite_fraction - 5.0

    center = window[:, center_idx, :]
    step = np.linalg.norm(np.diff(center, axis=0), axis=1)
    heading = np.unwrap(np.arctan2(body_vec[:, 1], body_vec[:, 0]))
    pose_delta = np.linalg.norm(np.diff(window, axis=0), axis=-1)

    body_cv = float(np.nanstd(body_len[finite_body]) / max(np.nanmean(body_len[finite_body]), 1e-6))
    step_med = float(np.nanmedian(step[np.isfinite(step)])) if np.isfinite(step).any() else np.inf
    heading_std = float(np.nanstd(heading[np.isfinite(heading)])) if np.isfinite(heading).any() else np.inf
    pose_step = float(np.nanmedian(pose_delta[np.isfinite(pose_delta)])) if np.isfinite(pose_delta).any() else np.inf

    return finite_fraction - 1.10 * body_cv - 0.30 * step_med - 0.22 * heading_std - 0.12 * pose_step


def _select_stable_window(
    raw_clip: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int,
) -> tuple[np.ndarray, int, float]:
    best_score = -np.inf
    best_left = 0
    best_window = _fill_window_with_context(raw_clip[:WINDOW_FRAMES], raw_clip, center_idx)

    for left in range(0, raw_clip.shape[0] - WINDOW_FRAMES + 1):
        candidate_raw = raw_clip[left:left + WINDOW_FRAMES]
        score = _score_stability_window(candidate_raw, raw_clip, nose_idx, tail_idx, center_idx)
        if score > best_score:
            best_score = score
            best_left = left
            best_window = _fill_window_with_context(candidate_raw, raw_clip, center_idx)

    return best_window, best_left, float(best_score)


def _select_window_for_target(
    target: int,
    raw_clip: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int,
) -> tuple[np.ndarray, int, float]:
    if target in USE_LAST_WINDOW:
        left = max(0, raw_clip.shape[0] - WINDOW_FRAMES)
        window = _fill_window_with_context(raw_clip[left:left + WINDOW_FRAMES], raw_clip, center_idx)
        score = _score_stability_window(raw_clip[left:left + WINDOW_FRAMES], raw_clip, nose_idx, tail_idx, center_idx)
        return window, left, float(score)

    if target in WINDOW_LEFT_OVERRIDES:
        left = int(WINDOW_LEFT_OVERRIDES[target])
        left = min(max(0, left), max(0, raw_clip.shape[0] - WINDOW_FRAMES))
        window = _fill_window_with_context(raw_clip[left:left + WINDOW_FRAMES], raw_clip, center_idx)
        score = _score_stability_window(raw_clip[left:left + WINDOW_FRAMES], raw_clip, nose_idx, tail_idx, center_idx)
        return window, left, float(score)

    return _select_stable_window(raw_clip, nose_idx, tail_idx, center_idx)


def _normalize_window(
    window: np.ndarray,
    nose_idx: int,
    tail_idx: int,
    center_idx: int,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    X = np.asarray(window, dtype=float).copy()
    ref_t = len(X) // 2
    ref_center = X[ref_t, center_idx].copy()
    X = X - ref_center[None, None, :]
    X = X * np.array([1.0, -1.0])
    X = s5._rotate_trajectory_to_target_heading(
        X,
        nose_idx=nose_idx,
        tail_idx=tail_idx,
        target_angle=HEADING_TARGET_ANGLE,
    )

    scales = np.array([_edge_length_sum(frame, edges) for frame in X], dtype=float)
    finite = np.isfinite(scales) & (scales > 1e-6)
    if finite.sum() >= 3:
        scale = float(np.nanmedian(scales[finite]))
    else:
        body_vec = X[:, nose_idx, :] - X[:, tail_idx, :]
        body_len = np.linalg.norm(body_vec, axis=1)
        finite_body = np.isfinite(body_len) & (body_len > 1e-6)
        if finite_body.sum() >= 3:
            scale = float(np.nanmedian(body_len[finite_body]))
        else:
            center = X[:, center_idx, :]
            radius = np.linalg.norm(X - center[:, None, :], axis=-1)
            finite_radius = np.isfinite(radius) & (radius > 1e-6)
            if finite_radius.sum() < 6:
                raise RuntimeError("Window does not contain enough finite scale samples.")
            scale = float(np.nanmedian(radius[finite_radius]))
    return X / scale


def _global_limits(clips: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(clips, axis=0)
    lims = s5._safe_plot_limits(stack, pctl=2)
    span = lims[1] - lims[0]
    max_span = float(np.max(span))
    center = np.nanmean(lims, axis=0)
    half = max(0.95, max_span * 0.42)
    return np.array([center - half, center + half], dtype=float)


def _edge_outlier_threshold(clip: np.ndarray, edges: list[tuple[int, int]]) -> float:
    lengths = []
    for t in range(clip.shape[0]):
        for ii, jj in edges:
            p = clip[t, ii]
            q = clip[t, jj]
            if np.isfinite(p).all() and np.isfinite(q).all():
                lengths.append(float(np.linalg.norm(p - q)))
    if not lengths:
        return np.inf
    return float(np.nanmedian(np.asarray(lengths, dtype=float)) * 2.2)


def _plot_panel(
    ax,
    clip: np.ndarray,
    edges: list[tuple[int, int]],
    lims: np.ndarray,
    title: str,
    subtitle: str,
    accent: str,
    center_idx: int,
    nose_idx: int,
    tail_idx: int,
    bodyparts: list[str],
    frame_limit: int | None = None,
) -> None:
    if frame_limit is None:
        frame_limit = len(clip)
    frame_limit = max(1, min(int(frame_limit), len(clip)))
    clip_use = clip[:frame_limit]

    ax.set_xlim(lims[0, 0], lims[1, 0])
    ax.set_ylim(lims[0, 1], lims[1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    center = clip_use[:, center_idx, :]
    ax.plot(center[:, 0], center[:, 1], linestyle="--", color=accent, linewidth=1.35, alpha=0.55, zorder=1)
    edge_threshold = _edge_outlier_threshold(clip_use, edges)

    for t in range(len(clip_use)):
        frac = t / max(len(clip_use) - 1, 1)
        is_final = t == len(clip_use) - 1
        line_color = "#C7CDD2" if not is_final else "#202020"
        alpha = 0.16 + 0.70 * frac
        lw = 1.10 if not is_final else 2.0
        size = 15 if not is_final else 24
        for ii, jj in edges:
            edge_name = frozenset((bodyparts[ii], bodyparts[jj]))
            if edge_name not in UNFILTERED_EDGES:
                if np.isfinite(clip_use[t, ii]).all() and np.isfinite(clip_use[t, jj]).all():
                    if float(np.linalg.norm(clip_use[t, ii] - clip_use[t, jj])) > edge_threshold:
                        continue
            ax.plot(
                clip_use[t, (ii, jj), 0],
                clip_use[t, (ii, jj), 1],
                color=line_color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=4 + t,
            )
        ax.scatter(
            clip_use[t, :, 0],
            clip_use[t, :, 1],
            s=size,
            color="#F2F4F6",
            edgecolors=line_color,
            linewidths=0.50,
            alpha=alpha,
            zorder=5 + t,
        )

    ax.set_title(title, fontsize=14, pad=10, color="#111111", weight="bold")
    ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=accent)


def _fig_to_image(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    image = imageio.imread(buf)
    buf.close()
    return image


def _save_panel_gif(
    out_path: Path,
    clip: np.ndarray,
    edges: list[tuple[int, int]],
    lims: np.ndarray,
    title: str,
    subtitle: str,
    accent: str,
    center_idx: int,
    nose_idx: int,
    tail_idx: int,
    bodyparts: list[str],
) -> None:
    frames = []
    for frame_limit in range(1, len(clip) + 1):
        fig, ax = plt.subplots(figsize=(4.2, 4.9))
        _plot_panel(
            ax=ax,
            clip=clip,
            edges=edges,
            lims=lims,
            title=title,
            subtitle=subtitle,
            accent=accent,
            center_idx=center_idx,
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            bodyparts=bodyparts,
            frame_limit=frame_limit,
        )
        plt.tight_layout()
        frames.append(_fig_to_image(fig))
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION_SEC, loop=0)


def _save_strip_gif(
    out_path: Path,
    clips: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    lims: np.ndarray,
    center_idx: int,
    nose_idx: int,
    tail_idx: int,
    bodyparts: list[str],
) -> None:
    frames = []
    max_frames = max(len(clip) for clip in clips.values())
    for frame_limit in range(1, max_frames + 1):
        fig, axes = plt.subplots(1, len(TARGETS), figsize=(19.0, 5.0))
        for ax, target in zip(np.atleast_1d(axes), TARGETS):
            meta = META[target]
            _plot_panel(
                ax=ax,
                clip=clips[target],
                edges=edges,
                lims=lims,
                title=f"s{target:02d}",
                subtitle=f"{meta['subtitle']} | 10 stable poses",
                accent=meta["accent"],
                center_idx=center_idx,
                nose_idx=nose_idx,
                tail_idx=tail_idx,
                bodyparts=bodyparts,
                frame_limit=min(frame_limit, len(clips[target])),
            )
        plt.tight_layout(w_pad=1.0)
        frames.append(_fig_to_image(fig))
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION_SEC, loop=0)


def run() -> None:
    cfg = load_config("config.yaml")
    manifest = _build_selected_manifest()
    coords_all, bodyparts = _load_selected_sessions(cfg, manifest["session"].astype(str).tolist())
    use_bodyparts = cfg["keypoints"]["use_bodyparts"]
    manifest_by_name = manifest.set_index("clip_name")
    pose_fps = float(cfg.get("deeplabcut", {}).get("target_fps") or 10.0)

    nose_idx = use_bodyparts.index("nose")
    tail_idx = use_bodyparts.index("tail_base")
    center_idx = use_bodyparts.index("mouse_center")
    skeleton = s5._build_skeleton(use_bodyparts)
    bp_index = {bp: i for i, bp in enumerate(use_bodyparts)}
    edges = [(bp_index[a], bp_index[b]) for a, b in skeleton if a in bp_index and b in bp_index]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    norm_windows: dict[int, np.ndarray] = {}
    selected_rows: list[dict[str, object]] = []

    for target in TARGETS:
        clip_name = str(TARGET_SPECS[target]["clip_name"])
        row = manifest_by_name.loc[clip_name]
        raw_clip, clip_start, clip_end = _extract_processed_clip(
            row,
            coords_all,
            use_bodyparts,
            bodyparts,
            pose_fps,
            CLIP_START_OFFSETS.get(target, 0),
        )
        stable_window, stable_left, stability_score = _select_window_for_target(
            target,
            raw_clip,
            nose_idx,
            tail_idx,
            center_idx,
        )
        norm_window = _normalize_window(stable_window, nose_idx, tail_idx, center_idx, edges)

        norm_windows[target] = norm_window
        selected_rows.append({
            "syllable": int(target),
            "clip_name": clip_name,
            "session": str(row["session"]),
            "clip_start_frame_processed": int(clip_start),
            "clip_end_frame_processed": int(clip_end),
            "stable_window_start_frame_processed": int(clip_start + stable_left),
            "stable_window_end_frame_processed": int(clip_start + stable_left + WINDOW_FRAMES),
            "window_frames": int(WINDOW_FRAMES),
            "pose_fps_used": float(pose_fps),
            "source_clip_fps_metadata": float(row["effective_fps"]),
            "stability_score": float(stability_score),
        })

    manifest.sort_values(["syllable"], inplace=True)
    pd.DataFrame(selected_rows).sort_values("syllable").to_csv(OUTDIR / "selected_windows.csv", index=False)
    manifest.to_csv(OUTDIR / "representative_manifest_reconstructed.csv", index=False)

    lims = _global_limits([norm_windows[target] for target in TARGETS])

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(19.0, 5.0))
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        meta = META[target]
        _plot_panel(
            ax=ax,
            clip=norm_windows[target],
            edges=edges,
            lims=lims,
            title=f"s{target:02d}",
            subtitle=f"{meta['subtitle']} | 10 stable poses",
            accent=meta["accent"],
            center_idx=center_idx,
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            bodyparts=use_bodyparts,
        )
    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUTDIR / "key5_selectedclip_trajectory_strip.pdf")
    fig.savefig(OUTDIR / "key5_selectedclip_trajectory_strip.png", dpi=220)
    plt.close(fig)
    _save_strip_gif(
        out_path=OUTDIR / "key5_selectedclip_trajectory_strip.gif",
        clips=norm_windows,
        edges=edges,
        lims=lims,
        center_idx=center_idx,
        nose_idx=nose_idx,
        tail_idx=tail_idx,
        bodyparts=use_bodyparts,
    )

    for target in TARGETS:
        meta = META[target]
        fig, ax = plt.subplots(figsize=(4.2, 4.9))
        _plot_panel(
            ax=ax,
            clip=norm_windows[target],
            edges=edges,
            lims=lims,
            title=f"s{target:02d}",
            subtitle=f"{meta['subtitle']} | 10 stable poses",
            accent=meta["accent"],
            center_idx=center_idx,
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            bodyparts=use_bodyparts,
        )
        plt.tight_layout()
        fig.savefig(OUTDIR / f"s{target:02d}_selectedclip_trajectory.pdf")
        fig.savefig(OUTDIR / f"s{target:02d}_selectedclip_trajectory.png", dpi=220)
        plt.close(fig)
        _save_panel_gif(
            out_path=OUTDIR / f"s{target:02d}_selectedclip_trajectory.gif",
            clip=norm_windows[target],
            edges=edges,
            lims=lims,
            title=f"s{target:02d}",
            subtitle=f"{meta['subtitle']} | 10 stable poses",
            accent=meta["accent"],
            center_idx=center_idx,
            nose_idx=nose_idx,
            tail_idx=tail_idx,
            bodyparts=use_bodyparts,
        )

    print(f"Saved selected-clip trajectories to {OUTDIR}")


if __name__ == "__main__":
    run()
