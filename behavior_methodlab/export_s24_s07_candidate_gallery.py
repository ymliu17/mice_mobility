#!/usr/bin/env python3
"""
Export a looser manual-review candidate gallery for s24 and s07.

This does not overwrite the representative final set. It creates a separate
candidate pool from OFT / Y-maze sessions only, with looser filtering and
session-aware spacing so the user can manually choose clips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from behavior_methodlab.export_key_syllable_video_clips import (
    _analysis_context_slug,
    _condition_slug,
    _export_clip,
    _gather_candidates,
)

OUTDIR = Path("outputs/05_stage5/key_syllable_candidate_gallery")
TARGETS = [24, 7]
MAX_PER_SYLLABLE = 12
CLIP_SECONDS = 2.0
EDGE_OVERRIDE = {
    24: 2.0,
    7: 3.0,
}
MIN_SAME_SESSION_GAP = {
    24: 240,
    7: 300,
}


def _collect_loose_candidates() -> pd.DataFrame:
    cand = _gather_candidates().copy()
    cand["analysis_context"] = cand.apply(_analysis_context_slug, axis=1)
    cand = cand[cand["syllable"].isin(TARGETS)].copy()
    cand = cand[cand["analysis_context"].isin(["oft", "ymaze"])].copy()
    if cand.empty:
        return cand

    # Relax the edge filter slightly by allowing earlier starts than the final set.
    keep_rows = []
    for row in cand.itertuples(index=False):
        state = int(row.syllable)
        edge_sec = EDGE_OVERRIDE[state]
        edge_frames = edge_sec * float(row.effective_fps)
        midpoint_frame = float(row.bout_start_frame_processed + row.bout_duration_frames_processed / 2.0)
        if midpoint_frame < edge_frames:
            continue
        if midpoint_frame > max(0.0, row.processed_frame_count - edge_frames):
            continue
        keep_rows.append(row._asdict())
    cand = pd.DataFrame(keep_rows)
    if cand.empty:
        return cand

    cand.sort_values(
        ["syllable", "analysis_context", "confidence_mean", "bout_duration_frames_processed"],
        ascending=[True, True, False, False],
        inplace=True,
    )
    return cand


def _select_spaced_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for syll in TARGETS:
        sub = cand[cand["syllable"] == syll].copy()
        if sub.empty:
            continue

        picks = []
        starts_by_session: dict[str, list[int]] = {}
        min_gap = MIN_SAME_SESSION_GAP[syll]

        for _, row in sub.iterrows():
            sess = str(row["session"])
            start = int(row["bout_start_frame_processed"])
            prev = starts_by_session.get(sess, [])
            if any(abs(start - p) < min_gap for p in prev):
                continue
            picks.append(row)
            starts_by_session.setdefault(sess, []).append(start)
            if len(picks) >= MAX_PER_SYLLABLE:
                break

        if picks:
            out = pd.DataFrame(picks).copy()
            out["clip_no"] = range(1, len(out) + 1)
            selected.append(out)

    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def run() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cand = _collect_loose_candidates()
    if cand.empty:
        raise SystemExit("No candidate bouts found for s24/s07 gallery.")
    selected = _select_spaced_candidates(cand)
    if selected.empty:
        raise SystemExit("No spaced candidates selected for s24/s07 gallery.")

    export_rows = []
    for row in selected.itertuples(index=False):
        syll = int(row.syllable)
        context = str(row.analysis_context)
        condition = _condition_slug(pd.Series(row._asdict()))
        clip_name = f"s{syll:02d}_{context}_{condition}_cand{int(row.clip_no):02d}.mp4"
        out_path = OUTDIR / f"s{syll:02d}" / clip_name

        start_frame_native, end_frame_native, native_fps = _export_clip(
            Path(row.video_path),
            out_path,
            float(row.bout_start_frame_processed + (row.bout_duration_frames_processed / 2.0)),
            int(row.processed_frame_count),
            CLIP_SECONDS,
        )
        export_rows.append({
            **row._asdict(),
            "clip_name": clip_name,
            "output_path": str(out_path),
            "native_start_frame": start_frame_native,
            "native_end_frame": end_frame_native,
            "native_fps": native_fps,
            "native_duration_sec": (end_frame_native - start_frame_native) / native_fps if native_fps > 0 else None,
        })

    out_df = pd.DataFrame(export_rows)
    out_df.to_csv(OUTDIR / "candidate_manifest.csv", index=False)
    print(f"Exported {len(out_df)} candidate clips to {OUTDIR}")
    print(f"Manifest: {OUTDIR / 'candidate_manifest.csv'}")


if __name__ == "__main__":
    run()
