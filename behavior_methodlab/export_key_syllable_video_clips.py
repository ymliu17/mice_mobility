#!/usr/bin/env python3
"""
Export top-confidence video clips for the five key syllables used in the slide.

Outputs:
  outputs/05_stage5/key_syllable_video_clips/
    manifest.csv
    s26/
    s11/
    s00/
    s24/
    s07/

Each clip filename follows:
  sxx_cohort_context_condition_no.mp4
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

OUTDIR = Path("outputs/05_stage5/key_syllable_video_clips_representative")
TAKAYA_FALLBACK_DIR = Path("Data/sugery_Takaya_Gen")
TARGETS = [26, 11, 0, 24, 7]
SOURCE_SYLLABLES_BY_LABEL = {
    26: [26],
    11: [11],
    0: [0],
    24: [43],
    7: [7, 45, 43],
}
N_PER_SYLLABLE = 5
CLIP_SECONDS = 2.0
MIN_BOUT_FRAMES = 2
EDGE_BUFFER_SEC = 5.0
EDGE_BUFFER_SEC_BY_SYLLABLE = {
    24: 5.0,
    7: 5.0,
}

PRIMARY_CONTEXTS = {
    7: ["oft", "ymaze"],
    24: ["oft", "ymaze"],
    26: ["oft", "oft-duo", "ast-food"],
    11: ["oft", "oft-duo", "ast-food"],
    0: ["oft", "oft-duo", "ast-food"],
}
FALLBACK_CONTEXTS = {
    7: set(),
    24: set(),
    26: {"ast-food"},
    11: {"ast-food"},
    0: {"ast-food"},
}
MIN_CONTEXT_QUOTAS = {
    26: [("oft", 3), ("oft-duo", 1), ("ast-food", 1)],
    11: [("oft", 3), ("oft-duo", 1), ("ast-food", 1)],
    0: [("oft", 3), ("oft-duo", 1), ("ast-food", 1)],
    24: [("oft", 3), ("ymaze", 1)],
    7: [("ymaze", 2), ("oft", 1)],
}

# Explicitly drop a few reviewed-but-bad bouts so reruns keep the curated set stable.
EXCLUDED_BOUTS = {
    24: {
        ("S1350008_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn", 43, 386),
    },
    7: {
        ("S2310010 022626Thu  Gi-10 pre_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn", 45, 709),
    },
}

PINNED_OUTPUT_BOUTS = {
    (26, 2): {
        "session": "S1760030_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "source_syllable": 26,
        "bout_start_frame_processed": 982,
    },
    (24, 2): {
        "session": "S2310011 022626Thu  Gi-11 pre_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "source_syllable": 43,
        "bout_start_frame_processed": 11581,
    },
    (7, 3): {
        "session": "S2280006 021726Tue  Gi-02 pre_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "source_syllable": 43,
        "bout_start_frame_processed": 11894,
    },
    (7, 4): {
        "session": "S2330005  022826Sat  Gi-11 post1_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn",
        "source_syllable": 43,
        "bout_start_frame_processed": 2372,
    },
    (7, 5): {
        "session": "POD+1 Y maze 240204-2_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn_mouse2",
        "source_syllable": 45,
        "bout_start_frame_processed": 2985,
    },
}


def _slug(text: object) -> str:
    return s5._slug("unknown" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text))


def _condition_slug(row: pd.Series) -> str:
    context = _analysis_context_slug(row)
    cond = row.get("condition_group")
    tp = row.get("timepoint")

    if context == "ast-food":
        cond_text = "" if cond is None or (isinstance(cond, float) and math.isnan(cond)) else str(cond).strip().lower()
        if "social isolation" in cond_text:
            return "social-isolation"
        if "control" in cond_text:
            return "control"
        return _slug(cond_text or "unknown")

    if tp is not None and not (isinstance(tp, float) and math.isnan(tp)):
        tp_text = str(tp).strip().lower()
        if tp_text:
            return _slug(tp_text)

    cond_text = "" if cond is None or (isinstance(cond, float) and math.isnan(cond)) else str(cond).strip().lower()
    if cond_text:
        return _slug(cond_text)
    return "unknown"


def _analysis_context_slug(row: pd.Series) -> str:
    cohort = str(row.get("cohort") or "").strip().lower()
    context = str(row.get("context") or "").strip().lower()
    if context == "y maze":
        return "ymaze"
    if context == "oft":
        return "oft-duo" if cohort == "s15" else "oft"
    if context in {"ast", "food intake during habituation"}:
        return "ast-food"
    return _slug(context or "unknown")


def _load_video_info(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, width, height, frame_count


def _resolve_video_path(session: str, video_lookup: dict[str, str]) -> str | None:
    base = session.split("_superanimal")[0]
    candidates = [base]
    stripped_mouse = re.sub(r"_mouse\d+$", "", base)
    if stripped_mouse != base:
        candidates.append(stripped_mouse)
    stripped_trailing = re.sub(r"\s+mouse\d+$", "", base)
    if stripped_trailing not in candidates:
        candidates.append(stripped_trailing)
    for cand in candidates:
        if cand in video_lookup:
            return video_lookup[cand]
    # Config still points to an older Takaya folder name in some runs; fall back to
    # the actual on-disk directory if the stem exists there.
    if TAKAYA_FALLBACK_DIR.exists():
        for cand in candidates:
            matches = list(TAKAYA_FALLBACK_DIR.rglob(f"{cand}.MP4"))
            if not matches:
                matches = list(TAKAYA_FALLBACK_DIR.rglob(f"{cand}.mp4"))
            if matches:
                return str(matches[0])
    return None


def _export_clip(
    video_path: Path,
    out_path: Path,
    midpoint_frame_processed: float,
    processed_frame_count: int,
    clip_seconds: float,
) -> tuple[int, int, float]:
    import cv2

    cap, fps, width, height, frame_count = _load_video_info(video_path)
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    if processed_frame_count <= 1:
        cap.release()
        raise RuntimeError(f"Invalid processed frame count for {video_path}")

    native_midpoint = int(round(float(midpoint_frame_processed) * (frame_count - 1) / max(processed_frame_count - 1, 1)))
    clip_frames = max(1, int(round(clip_seconds * fps)))
    half_clip = clip_frames // 2
    start_frame = max(0, native_midpoint - half_clip)
    end_frame = min(frame_count, start_frame + clip_frames)
    if end_frame - start_frame < clip_frames:
        start_frame = max(0, end_frame - clip_frames)
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
    written = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        frame_idx += 1

    writer.release()
    cap.release()
    return start_frame, start_frame + written, fps


def _gather_candidates() -> pd.DataFrame:
    cfg = load_config("config.yaml")
    seqs = s5._load_sequences()
    _, confs_all, _ = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=30.0)
    meta_df = s5._load_session_metadata(list(seqs.keys()))
    meta_lookup = meta_df.set_index("session")
    video_lookup = s5._build_video_lookup(cfg)

    rows: list[dict] = []
    for sess, seq in seqs.items():
        confs = confs_all.get(sess)
        if confs is None or len(confs) == 0:
            continue
        usable = min(len(seq), len(confs))
        if usable < MIN_BOUT_FRAMES:
            continue

        bout_syll, bout_starts, bout_lens = s5._bout_lengths(seq[:usable])
        session_meta = meta_lookup.loc[sess] if sess in meta_lookup.index else pd.Series(dtype=object)
        video_path = _resolve_video_path(sess, video_lookup)
        effective_fps = float(fps_map.get(sess, 30.0))
        if effective_fps <= 0:
            effective_fps = 30.0

        for state, start, length in zip(bout_syll, bout_starts, bout_lens):
            state = int(state)
            if int(length) < MIN_BOUT_FRAMES:
                continue
            display_targets = [
                label for label, sources in SOURCE_SYLLABLES_BY_LABEL.items()
                if int(state) in [int(s) for s in sources]
            ]
            if not display_targets:
                continue
            left = int(start)
            right = int(start + length)
            bout_conf = confs[left:right]
            conf_score = float(np.nanmean(bout_conf)) if np.isfinite(bout_conf).any() else np.nan
            finite_fraction = float(np.isfinite(bout_conf).mean()) if bout_conf.size else 0.0
            midpoint_frame = float(start + (length / 2.0))
            edge_sec = float(max(EDGE_BUFFER_SEC_BY_SYLLABLE.get(label, EDGE_BUFFER_SEC) for label in display_targets))
            edge_buffer_frames = float(edge_sec * effective_fps)
            if midpoint_frame < edge_buffer_frames:
                continue
            if midpoint_frame > max(0.0, usable - edge_buffer_frames):
                continue
            midpoint_sec = midpoint_frame / effective_fps
            half_sec = CLIP_SECONDS / 2.0
            for display_state in display_targets:
                source_rank = SOURCE_SYLLABLES_BY_LABEL[int(display_state)].index(int(state))
                rows.append({
                    "syllable": int(display_state),
                    "source_syllable": int(state),
                    "source_rank": int(source_rank),
                    "session": sess,
                    "video_path": video_path,
                    "cohort": session_meta.get("cohort"),
                    "context": session_meta.get("arena"),
                    "timepoint": session_meta.get("timepoint"),
                    "condition_group": session_meta.get("condition_group"),
                    "bout_start_frame_processed": int(start),
                    "bout_end_frame_processed": int(start + length),
                    "bout_duration_frames_processed": int(length),
                    "processed_frame_count": int(usable),
                    "effective_fps": effective_fps,
                    "midpoint_sec": midpoint_sec,
                    "clip_start_sec": max(0.0, midpoint_sec - half_sec),
                    "clip_end_sec": midpoint_sec + half_sec,
                    "confidence_mean": conf_score,
                    "confidence_finite_fraction": finite_fraction,
                })

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    cand = cand.dropna(subset=["video_path", "confidence_mean"]).copy()
    cand["video_exists"] = cand["video_path"].map(lambda p: Path(p).exists())
    cand = cand[cand["video_exists"]].copy()
    cand.sort_values(
        ["syllable", "confidence_mean", "confidence_finite_fraction", "bout_duration_frames_processed"],
        ascending=[True, False, False, False],
        inplace=True,
    )
    return cand


def _select_top_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    def _pick_from_ordered(ordered_sub: pd.DataFrame, limit: int) -> pd.DataFrame:
        picks: list[pd.Series] = []
        used_sessions: set[str] = set()
        used_indices: set[int] = set()

        combo_best = (
            ordered_sub.drop_duplicates(subset=["analysis_context", "session"], keep="first")
            .drop_duplicates(subset=["cohort", "context", "session"], keep="first")
            .drop_duplicates(subset=["cohort", "context"], keep="first")
        )
        for idx, row in combo_best.iterrows():
            if len(picks) >= limit:
                break
            sess = str(row["session"])
            if sess in used_sessions:
                continue
            picks.append(row)
            used_sessions.add(sess)
            used_indices.add(idx)

        cond_best = (
            ordered_sub.loc[~ordered_sub.index.isin(used_indices)]
            .drop_duplicates(subset=["condition_group", "timepoint", "session"], keep="first")
            .drop_duplicates(subset=["condition_group", "timepoint"], keep="first")
        )
        for idx, row in cond_best.iterrows():
            if len(picks) >= limit:
                break
            sess = str(row["session"])
            if sess in used_sessions:
                continue
            picks.append(row)
            used_sessions.add(sess)
            used_indices.add(idx)

        fill_unique = ordered_sub.loc[~ordered_sub.index.isin(used_indices)]
        for idx, row in fill_unique.iterrows():
            if len(picks) >= limit:
                break
            sess = str(row["session"])
            if sess in used_sessions:
                continue
            picks.append(row)
            used_sessions.add(sess)
            used_indices.add(idx)

        fill_any = ordered_sub.loc[~ordered_sub.index.isin(used_indices)]
        for idx, row in fill_any.iterrows():
            if len(picks) >= limit:
                break
            picks.append(row)
            used_indices.add(idx)

        if not picks:
            return pd.DataFrame()
        return pd.DataFrame(picks).head(limit).copy()

    def _pick_quota(sub: pd.DataFrame, context_name: str, need: int, used_sessions: set[str]) -> pd.DataFrame:
        if need <= 0:
            return pd.DataFrame()
        ctx = sub[sub["analysis_context"] == context_name].copy()
        if ctx.empty:
            return pd.DataFrame()
        ctx = ctx.loc[~ctx["session"].astype(str).isin(used_sessions)].copy()
        if ctx.empty:
            return pd.DataFrame()
        return _pick_from_ordered(ctx, need)

    selected: list[pd.DataFrame] = []
    for syll in TARGETS:
        sub = cand[cand["syllable"] == syll].copy()
        if sub.empty:
            continue

        excluded = EXCLUDED_BOUTS.get(syll, set())
        if excluded:
            sub = sub.loc[
                ~sub.apply(
                    lambda row: (
                        str(row["session"]),
                        int(row["source_syllable"]),
                        int(row["bout_start_frame_processed"]),
                    ) in excluded,
                    axis=1,
                )
            ].copy()
            if sub.empty:
                continue

        sub["analysis_context"] = sub.apply(_analysis_context_slug, axis=1)
        sub["source_priority"] = 1
        if syll == 7:
            sub.loc[(sub["source_syllable"] == 7) & (sub["analysis_context"] == "ymaze"), "source_priority"] = 0
        sub.sort_values(
            ["source_priority", "confidence_mean", "confidence_finite_fraction", "bout_duration_frames_processed"],
            ascending=[True, False, False, False],
            inplace=True,
        )
        primary_order = PRIMARY_CONTEXTS.get(syll, [])
        primary = set(primary_order)
        fallback = FALLBACK_CONTEXTS.get(syll, set())

        primary_sub = sub[sub["analysis_context"].isin(primary)].copy()
        fallback_sub = sub[sub["analysis_context"].isin(fallback)].copy()
        remainder_sub = sub[~sub["analysis_context"].isin(primary | fallback)].copy()
        if syll in {7, 24}:
            remainder_sub = remainder_sub.iloc[0:0].copy()

        quota_parts: list[pd.DataFrame] = []
        used_sessions: set[str] = set()
        for context_name, quota in MIN_CONTEXT_QUOTAS.get(syll, []):
            part = _pick_quota(sub, context_name, quota, used_sessions)
            if not part.empty:
                quota_parts.append(part)
                used_sessions.update(part["session"].astype(str).tolist())

        combined = pd.concat(quota_parts, ignore_index=True) if quota_parts else pd.DataFrame()
        if len(combined) < N_PER_SYLLABLE:
            need = N_PER_SYLLABLE - len(combined)
            primary_pool = primary_sub.loc[~primary_sub["session"].astype(str).isin(used_sessions)].copy()
            primary_fill = _pick_from_ordered(primary_pool, need)
            if not primary_fill.empty:
                combined = pd.concat([combined, primary_fill], ignore_index=True)
                used_sessions.update(primary_fill["session"].astype(str).tolist())

        if len(combined) < N_PER_SYLLABLE:
            need = N_PER_SYLLABLE - len(combined)
            fallback_pool = fallback_sub.loc[~fallback_sub["session"].astype(str).isin(used_sessions)].copy()
            fallback_fill = _pick_from_ordered(fallback_pool, need)
            if not fallback_fill.empty:
                combined = pd.concat([combined, fallback_fill], ignore_index=True)
                used_sessions.update(fallback_fill["session"].astype(str).tolist())

        if len(combined) < N_PER_SYLLABLE:
            need = N_PER_SYLLABLE - len(combined)
            remainder_pool = remainder_sub.loc[~remainder_sub["session"].astype(str).isin(used_sessions)].copy()
            remainder_fill = _pick_from_ordered(remainder_pool, need)
            if not remainder_fill.empty:
                combined = pd.concat([combined, remainder_fill], ignore_index=True)

        if not combined.empty:
            selected.append(combined.head(N_PER_SYLLABLE).copy())

    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    out["clip_no"] = out.groupby("syllable").cumcount() + 1
    for (syll, clip_no), spec in PINNED_OUTPUT_BOUTS.items():
        mask = (out["syllable"] == int(syll)) & (out["clip_no"] == int(clip_no))
        if not mask.any():
            continue
        cand_mask = (
            (cand["syllable"] == int(syll))
            & (cand["session"].astype(str) == str(spec["session"]))
            & (cand["source_syllable"] == int(spec["source_syllable"]))
            & (cand["bout_start_frame_processed"] == int(spec["bout_start_frame_processed"]))
        )
        if not cand_mask.any():
            continue
        replacement = cand.loc[cand_mask].iloc[0].to_dict()
        replacement["clip_no"] = int(clip_no)
        out.loc[mask, :] = pd.DataFrame([replacement], index=out.index[mask])
    return out


def run() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cand = _gather_candidates()
    if cand.empty:
        raise SystemExit("No candidate bouts found for the target syllables.")

    selected = _select_top_candidates(cand)
    if selected.empty:
        raise SystemExit("No exportable candidates selected.")

    # Clean previously exported mp4s so reruns do not leave stale clips behind.
    for syll in TARGETS:
        syll_dir = OUTDIR / f"s{int(syll):02d}"
        if syll_dir.exists():
            for old_mp4 in syll_dir.glob("*.mp4"):
                old_mp4.unlink()

    export_rows: list[dict] = []
    for row in selected.itertuples(index=False):
        syll = int(row.syllable)
        context = _analysis_context_slug(pd.Series(row._asdict()))
        condition = _condition_slug(pd.Series(row._asdict()))
        clip_name = f"s{syll:02d}_{context}_{condition}_{int(row.clip_no):02d}.mp4"
        out_path = OUTDIR / f"s{syll:02d}" / clip_name

        try:
            start_frame_native, end_frame_native, native_fps = _export_clip(
                Path(row.video_path),
                out_path,
                float(row.bout_start_frame_processed + (row.bout_duration_frames_processed / 2.0)),
                int(row.processed_frame_count),
                CLIP_SECONDS,
            )
            export_rows.append({
                **row._asdict(),
                "output_path": str(out_path),
                "clip_name": clip_name,
                "native_start_frame": start_frame_native,
                "native_end_frame": end_frame_native,
                "native_fps": native_fps,
                "native_duration_sec": (end_frame_native - start_frame_native) / native_fps if native_fps > 0 else np.nan,
            })
        except Exception as exc:
            export_rows.append({
                **row._asdict(),
                "output_path": str(out_path),
                "clip_name": clip_name,
                "export_error": str(exc),
            })

    export_df = pd.DataFrame(export_rows)
    export_df.to_csv(OUTDIR / "manifest.csv", index=False)
    ok = export_df["output_path"].map(lambda p: Path(p).exists()).sum() if "output_path" in export_df.columns else 0
    print(f"Exported {ok} clips to {OUTDIR}")
    print(f"Manifest: {OUTDIR / 'manifest.csv'}")


if __name__ == "__main__":
    run()
