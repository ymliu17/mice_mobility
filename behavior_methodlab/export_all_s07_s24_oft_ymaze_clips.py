#!/usr/bin/env python3
"""
Export all raw model-predicted s07 / s24 bout clips from OFT and Y-maze sessions.

This is intentionally permissive:
- no confidence ranking threshold
- no representative sampling
- no minimum bout duration beyond >0 frames
- no edge trimming beyond video bounds

Each exported clip spans the full predicted bout with a small temporal pad on both
sides to make manual review easier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from behavior_methodlab.export_key_syllable_video_clips import (
    SOURCE_SYLLABLE_BY_LABEL,
    _analysis_context_slug,
    _condition_slug,
    _load_video_info,
    _resolve_video_path,
)
from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

OUTDIR = Path("outputs/05_stage5/key_syllable_all_oft_ymaze_candidates")
TARGETS = [7, 24]
PRE_PAD_SEC = 0.5
POST_PAD_SEC = 0.5


def _export_native_range(video_path: Path, out_path: Path, start_frame: int, end_frame: int) -> tuple[int, int, float]:
    import cv2

    cap, fps, width, height, frame_count = _load_video_info(video_path)
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    start_frame = max(0, int(start_frame))
    end_frame = min(frame_count, int(end_frame))
    if end_frame <= start_frame:
        cap.release()
        raise RuntimeError(f"Empty clip interval for {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer: {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    written = 0
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1
        written += 1

    writer.release()
    cap.release()
    return start_frame, start_frame + written, fps


def run() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config("config.yaml")
    seqs = s5._load_sequences()
    _, confs_all, _ = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=30.0)
    meta_df = s5._load_session_metadata(list(seqs.keys())).set_index("session")
    video_lookup = s5._build_video_lookup(cfg)

    rows: list[dict] = []
    for sess, seq in seqs.items():
        confs = confs_all.get(sess)
        if confs is None or len(confs) == 0:
            continue
        usable = min(len(seq), len(confs))
        if usable < 1:
            continue

        session_meta = meta_df.loc[sess] if sess in meta_df.index else pd.Series(dtype=object)
        analysis_context = _analysis_context_slug(pd.Series({
            "cohort": session_meta.get("cohort"),
            "context": session_meta.get("arena"),
        }))
        if analysis_context not in {"oft", "ymaze", "oft-duo"}:
            continue

        video_path = _resolve_video_path(sess, video_lookup)
        if not video_path:
            continue

        effective_fps = float(fps_map.get(sess, 30.0) or 30.0)
        bout_syll, bout_starts, bout_lens = s5._bout_lengths(seq[:usable])
        for state, start, length in zip(bout_syll, bout_starts, bout_lens):
            state = int(state)
            display_targets = [label for label, source in SOURCE_SYLLABLE_BY_LABEL.items() if int(source) == state and label in TARGETS]
            if not display_targets:
                continue
            start = int(start)
            length = int(length)
            end = start + length
            bout_conf = confs[start:end]
            conf_score = float(np.nanmean(bout_conf)) if np.isfinite(bout_conf).any() else np.nan
            for display_state in display_targets:
                rows.append({
                    "syllable": int(display_state),
                    "source_syllable": int(state),
                    "session": sess,
                    "cohort": session_meta.get("cohort"),
                    "raw_context": session_meta.get("arena"),
                    "analysis_context": analysis_context,
                    "condition_group": session_meta.get("condition_group"),
                    "timepoint": session_meta.get("timepoint"),
                    "video_path": video_path,
                    "processed_frame_count": int(usable),
                    "effective_fps": effective_fps,
                    "bout_start_frame_processed": start,
                    "bout_end_frame_processed": end,
                    "bout_duration_frames_processed": length,
                    "confidence_mean": conf_score,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No s07/s24 OFT/Ymaze candidates found.")

    df.sort_values(
        ["syllable", "analysis_context", "session", "bout_start_frame_processed"],
        inplace=True,
    )
    df["clip_no"] = df.groupby("syllable").cumcount() + 1

    export_rows: list[dict] = []
    for row in df.itertuples(index=False):
        syll = int(row.syllable)
        context = str(row.analysis_context)
        condition = _condition_slug(pd.Series(row._asdict()))
        clip_name = f"s{syll:02d}_{context}_{condition}_all{int(row.clip_no):03d}.mp4"
        out_path = OUTDIR / f"s{syll:02d}" / clip_name

        cap, native_fps, _, _, frame_count = _load_video_info(Path(row.video_path))
        cap.release()
        native_start = int(round((row.bout_start_frame_processed) * (frame_count - 1) / max(row.processed_frame_count - 1, 1)))
        native_end = int(round((row.bout_end_frame_processed) * (frame_count - 1) / max(row.processed_frame_count - 1, 1)))
        native_start -= int(round(PRE_PAD_SEC * native_fps))
        native_end += int(round(POST_PAD_SEC * native_fps))

        start_frame_native, end_frame_native, fps = _export_native_range(
            Path(row.video_path),
            out_path,
            native_start,
            native_end,
        )
        export_rows.append({
            **row._asdict(),
            "clip_name": clip_name,
            "output_path": str(out_path),
            "native_start_frame": start_frame_native,
            "native_end_frame": end_frame_native,
            "native_fps": fps,
            "native_duration_sec": (end_frame_native - start_frame_native) / fps if fps > 0 else None,
        })

    out_df = pd.DataFrame(export_rows)
    out_df.to_csv(OUTDIR / "all_candidates_manifest.csv", index=False)
    print(f"Exported {len(out_df)} all-candidate clips to {OUTDIR}")
    print(f"Manifest: {OUTDIR / 'all_candidates_manifest.csv'}")


if __name__ == "__main__":
    run()
