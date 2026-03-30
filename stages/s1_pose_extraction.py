"""
Stage 1 – DeepLabCut SuperAnimal TopViewMouse pose extraction.

Runs inference on all configured video files and writes:
  outputs/01_dlc_raw/<context>/<video_stem>DLC_*.h5   (keypoint coords + confidence)
  outputs/01_dlc_raw/<context>/<video_stem>DLC_*.csv  (human-readable, optional)

Usage (via pipeline.py):
  python pipeline.py run --stages 1

Or standalone:
  python stages/s1_pose_extraction.py --config config.yaml
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.io_utils import load_config, collect_all_videos, resolve_path

logger = logging.getLogger(__name__)


# ─── DLC version detection ────────────────────────────────────────────────────

def _get_dlc_version() -> tuple[int, int]:
    """Return (major, minor) of the installed deeplabcut package."""
    import deeplabcut
    parts = deeplabcut.__version__.split(".")
    return int(parts[0]), int(parts[1])


# ─── Inference helpers ────────────────────────────────────────────────────────

def run_superanimal_inference_pytorch(
    videos: List[str],
    superanimal_name: str,
    model_name: str,
    detector_name: str,
    scale_list: List[int],
    dest_folder: str,
    video_adapt: bool = False,
    adapt_iterations: int = 100,
    batch_size: int = 8,
    save_as_csv: bool = True,
) -> None:
    """
    Run SuperAnimal inference with DLC >= 3.0 (PyTorch backend).

    For each video this produces:
      <dest_folder>/<video_stem>DLC_<scorer>.h5
      <dest_folder>/<video_stem>DLC_<scorer>.csv   (if save_as_csv=True)
    """
    import deeplabcut

    os.makedirs(dest_folder, exist_ok=True)

    if video_adapt:
        logger.info(
            "Running video adaptation (%d iters) before inference ...", adapt_iterations
        )
        # Video adaptation refines weights on unlabelled target frames.
        # It improves accuracy at the cost of compute time.
        deeplabcut.video_inference_superanimal(
            videos=videos,
            superanimal_name=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
            scale_list=scale_list,
            videotype=_infer_videotype(videos),
            dest_folder=dest_folder,
            video_adapt=True,
            adapt_iterations=adapt_iterations,
            batch_size=batch_size,
            create_labeled_video=False,
        )
    else:
        deeplabcut.video_inference_superanimal(
            videos=videos,
            superanimal_name=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
            scale_list=scale_list,
            videotype=_infer_videotype(videos),
            dest_folder=dest_folder,
            batch_size=batch_size,
            create_labeled_video=False,
        )


def run_superanimal_inference_tf(
    videos: List[str],
    superanimal_name: str,
    dest_folder: str,
    save_as_csv: bool = True,
) -> None:
    """
    Run SuperAnimal inference with DLC < 3.0 (TensorFlow / DLC 2.x backend).

    This uses the legacy ``superanimal_topviewmouse_dlcrnet`` model.
    """
    import deeplabcut

    # DLC 2.x superanimal uses a different model identifier
    legacy_name = f"{superanimal_name}_dlcrnet"
    os.makedirs(dest_folder, exist_ok=True)

    deeplabcut.video_inference_superanimal(
        videos=videos,
        superanimal_name=legacy_name,
        videotype=_infer_videotype(videos),
        dest_folder=dest_folder,
        save_as_csv=save_as_csv,
    )


def _infer_videotype(videos: List[str]) -> str:
    """Return the lowercase extension of the first video (without the dot)."""
    if not videos:
        return "mp4"
    return Path(videos[0]).suffix.lstrip(".").lower()


# ─── FPS helpers ──────────────────────────────────────────────────────────────

def get_video_fps(video_path: str) -> float:
    """Return the native frame rate of a video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        logger.warning("Could not read fps from %s; assuming 30.", video_path)
        return 30.0
    return float(fps)


def downsample_video(src: str, dst: str, target_fps: float) -> None:
    """Create a copy of *src* resampled to *target_fps* using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",   # drop audio – not needed for pose tracking
        dst,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src}:\n{result.stderr.decode(errors='replace')}"
        )


def _prepare_videos(
    videos: List[str],
    target_fps: float,
    tmp_dir: str,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Downsample each video whose native fps exceeds *target_fps*.

    Returns (paths_to_use, {video_stem: effective_fps}).
    Downsampled copies are written to *tmp_dir* under their original filenames
    so the DLC scorer output retains the original session names.
    """
    out_paths: List[str] = []
    fps_manifest: Dict[str, float] = {}

    for v in videos:
        stem = Path(v).stem
        actual_fps = get_video_fps(v)
        if actual_fps > target_fps * 1.05:   # 5 % tolerance avoids rounding noise
            dst = os.path.join(tmp_dir, Path(v).name)
            logger.info(
                "  Resampling %s: %.1f → %.1f fps ...", Path(v).name, actual_fps, target_fps
            )
            downsample_video(v, dst, target_fps)
            out_paths.append(dst)
            fps_manifest[stem] = target_fps
        else:
            logger.info(
                "  %s is already %.1f fps – no resampling needed.", Path(v).name, actual_fps
            )
            out_paths.append(v)
            fps_manifest[stem] = actual_fps

    return out_paths, fps_manifest


# ─── Per-context processing ───────────────────────────────────────────────────

def process_context(
    context_name: str,
    videos: List[str],
    dlc_cfg: dict,
    output_root: str,
    base_dir: str,
    target_fps: Optional[float] = None,
) -> int:
    """
    Run DLC on all videos for a single context.

    If *target_fps* is set and a video's native fps exceeds it, a downsampled
    copy is created via ffmpeg, DLC runs on that copy, and the copy is removed
    afterwards.  An fps_manifest.json is written to *dest_folder* recording the
    effective fps for each session so downstream stages can convert frame counts
    to seconds.

    Returns the number of H5 files produced.
    """
    if not videos:
        logger.warning("No videos found for context '%s' – skipping.", context_name)
        return 0

    dest_folder = os.path.join(output_root, context_name)
    os.makedirs(dest_folder, exist_ok=True)

    logger.info(
        "Context '%s': %d video(s) → %s", context_name, len(videos), dest_folder
    )
    for v in videos:
        logger.info("  %s", v)

    # ── Optional fps downsampling ─────────────────────────────────────────────
    videos_to_run = videos
    fps_manifest: Dict[str, float] = {}
    tmp_dir: Optional[str] = None

    if target_fps is not None:
        tmp_dir = os.path.join(dest_folder, "_tmp_ds")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            videos_to_run, fps_manifest = _prepare_videos(videos, target_fps, tmp_dir)
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir = None
            logger.error(
                "Downsampling failed (%s) – running DLC on original frames.", exc
            )
            videos_to_run = videos
            fps_manifest = {Path(v).stem: get_video_fps(v) for v in videos}
    else:
        fps_manifest = {Path(v).stem: get_video_fps(v) for v in videos}

    # ── DLC inference ─────────────────────────────────────────────────────────
    try:
        major, minor = _get_dlc_version()
        if major >= 3:
            run_superanimal_inference_pytorch(
                videos=videos_to_run,
                superanimal_name=dlc_cfg["superanimal_name"],
                model_name=dlc_cfg["model_name"],
                detector_name=dlc_cfg["detector_name"],
                scale_list=dlc_cfg.get("scale_list", [200, 300, 400]),
                dest_folder=dest_folder,
                video_adapt=dlc_cfg.get("video_adapt", False),
                adapt_iterations=dlc_cfg.get("adapt_iterations", 100),
                batch_size=dlc_cfg.get("batch_size", 8),
                save_as_csv=dlc_cfg.get("save_as_csv", True),
            )
        else:
            logger.warning(
                "DLC version %d.%d detected – falling back to TensorFlow / DLC 2.x path.",
                major,
                minor,
            )
            run_superanimal_inference_tf(
                videos=videos_to_run,
                superanimal_name=dlc_cfg["superanimal_name"],
                dest_folder=dest_folder,
                save_as_csv=dlc_cfg.get("save_as_csv", True),
            )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Write fps manifest ────────────────────────────────────────────────────
    manifest_path = os.path.join(dest_folder, "fps_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(fps_manifest, fh, indent=2)
    logger.info("  fps manifest → %s", manifest_path)

    produced = list(Path(dest_folder).glob("*.h5"))
    logger.info("  → %d H5 result files written.", len(produced))
    return len(produced)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    dlc_output_dir = resolve_path(base_dir, cfg["paths"]["dlc_output_dir"])
    all_videos = collect_all_videos(cfg, base_dir)
    dlc_cfg = cfg["dlc"]
    target_fps: Optional[float] = dlc_cfg.get("target_fps") or None
    if target_fps is not None:
        logger.info("target_fps = %.1f – videos will be resampled before DLC.", target_fps)

    total_processed = 0
    for ctx_name, videos in all_videos.items():
        n = process_context(
            context_name=ctx_name,
            videos=videos,
            dlc_cfg=dlc_cfg,
            output_root=dlc_output_dir,
            base_dir=base_dir,
            target_fps=target_fps,
        )
        total_processed += n

    logger.info("Stage 1 complete. %d H5 files produced.", total_processed)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Stage 1: DLC SuperAnimal inference")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
