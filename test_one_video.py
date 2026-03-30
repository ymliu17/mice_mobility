"""
Quick end-to-end test on a single video.

Steps
-----
1. Download SuperAnimal TopViewMouse weights (pose model + detector) via dlclibrary
2. Run DLC inference on one video  →  outputs/test/01_dlc_raw/
3. Preprocess the resulting H5 file →  outputs/test/02_preprocessed/
4. Print a summary (keypoint confidence, NaN rates)

Usage
-----
# activate your environment first, then:
python test_one_video.py

# use a different video:
python test_one_video.py --video "Data/EPM videos/G1-1 EPM.MP4"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
DEFAULT_VIDEO = str(REPO_ROOT / "Data/David_Pain/FX_Th Cell_OF_5.mpg")
MODEL_DIR = REPO_ROOT / "superanimal_topviewmouse_model"
TEST_OUT = REPO_ROOT / "outputs/test"

# ── model names (correct names from dlclibrary) ────────────────────────────────
POSE_MODEL_NAME = "superanimal_topviewmouse_hrnet_w32"
DETECTOR_NAME   = "superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn"
LEGACY_TF_NAME  = "superanimal_topviewmouse"          # DLC 2.x single-archive fallback


# ─── Step 1: download model weights ───────────────────────────────────────────

def download_models(model_dir: Path) -> None:
    """Download pose model + detector weights if not already present."""
    sys.path.insert(0, str(REPO_ROOT / "DLClibrary"))
    from dlclibrary import download_huggingface_model

    model_dir.mkdir(parents=True, exist_ok=True)

    # Check whether weights are already downloaded
    pt_files = list(model_dir.glob("*.pt"))
    tar_files = list(model_dir.glob("*.tar.gz"))
    pkl_files = list(model_dir.glob("*.pkl"))

    if pt_files or tar_files or pkl_files:
        logger.info("Model weights already present in %s – skipping download.", model_dir)
        return

    logger.info("Detecting DLC version to choose model format ...")
    try:
        import deeplabcut
        major = int(deeplabcut.__version__.split(".")[0])
    except ImportError:
        logger.error(
            "deeplabcut is not installed.\n"
            "Install it with one of:\n"
            "  pip install 'deeplabcut[pytorch]'   # DLC 3.x – PyTorch\n"
            "  pip install deeplabcut               # DLC 2.x – TensorFlow\n"
            "\nOr create the conda environment:\n"
            "  conda env create -f environment.yml && conda activate mobility"
        )
        sys.exit(1)

    if major >= 3:
        # PyTorch backend: need pose model AND detector separately
        logger.info("Downloading PyTorch pose model: %s ...", POSE_MODEL_NAME)
        download_huggingface_model(POSE_MODEL_NAME, str(model_dir))

        logger.info("Downloading PyTorch detector:   %s ...", DETECTOR_NAME)
        download_huggingface_model(DETECTOR_NAME, str(model_dir))
    else:
        # TensorFlow / DLC 2.x: single archive
        logger.info("Downloading TF model: %s ...", LEGACY_TF_NAME)
        download_huggingface_model(LEGACY_TF_NAME, str(model_dir))

    logger.info("Download complete. Files in %s:", model_dir)
    for f in sorted(model_dir.iterdir()):
        logger.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)


# ─── Step 2: DLC inference ────────────────────────────────────────────────────

def run_inference(video_path: str, out_dir: Path) -> list[str]:
    """Run SuperAnimal inference on one video; return list of produced H5 paths."""
    import deeplabcut

    out_dir.mkdir(parents=True, exist_ok=True)
    major = int(deeplabcut.__version__.split(".")[0])

    logger.info("Running DLC %d.x inference on: %s", major, video_path)

    if major >= 3:
        deeplabcut.video_inference_superanimal(
            videos=[video_path],
            superanimal_name="superanimal_topviewmouse",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_mobilenet_v3_large_fpn",
            scale_list=[200, 300, 400],
            videotype=Path(video_path).suffix.lstrip(".").lower(),
            dest_folder=str(out_dir),
            batch_size=8,
            create_labeled_video=False,  # skip annotated video for speed
        )
    else:
        deeplabcut.video_inference_superanimal(
            videos=[video_path],
            superanimal_name=LEGACY_TF_NAME,
            videotype=Path(video_path).suffix.lstrip(".").lower(),
            dest_folder=str(out_dir),
            save_as_csv=True,
        )

    h5_files = sorted(out_dir.glob("*.h5"))
    logger.info("Inference done. H5 files produced: %d", len(h5_files))
    return [str(f) for f in h5_files]


# ─── Step 3: Preprocess + print summary ───────────────────────────────────────

def preprocess_and_summarise(h5_path: str, out_dir: Path) -> None:
    """Run Stage 2 preprocessing on one H5 file and print a confidence summary."""
    sys.path.insert(0, str(REPO_ROOT))
    from utils.io_utils import load_config, session_name_from_dlc_file, save_preprocessed
    from stages.s2_preprocessing import (
        load_dlc_h5,
        apply_confidence_mask,
        interpolate_gaps,
        smooth_savgol,
        normalize_body_length,
    )

    cfg = load_config(str(REPO_ROOT / "config.yaml"))
    prep = cfg["preprocessing"]
    kp   = cfg["keypoints"]

    logger.info("Loading DLC H5: %s", h5_path)
    coords, confs, bodyparts = load_dlc_h5(h5_path)
    T, K = confs.shape

    # ── confidence summary (raw) ───────────────────────────────────────────────
    import numpy as np
    print("\n── Raw confidence summary (per keypoint) ────────────────────────────")
    print(f"  {'Keypoint':<20}  {'Mean conf':>10}  {'<{:.2f} (%)'.format(prep['conf_threshold']):>14}")
    print("  " + "-" * 50)
    for ki, bp in enumerate(bodyparts):
        mean_c = float(np.nanmean(confs[:, ki]))
        low_frac = float(np.mean(confs[:, ki] < prep["conf_threshold"])) * 100
        flag = "  ⚠" if low_frac > 30 else ""
        print(f"  {bp:<20}  {mean_c:>10.3f}  {low_frac:>13.1f}%{flag}")

    # ── run preprocessing ──────────────────────────────────────────────────────
    masked = apply_confidence_mask(coords, confs, prep["conf_threshold"])
    interp = interpolate_gaps(masked, max_gap=prep.get("max_interp_gap", 10))
    smoothed = smooth_savgol(interp,
                             window=prep.get("smooth_window", 5),
                             polyorder=prep.get("smooth_polyorder", 2))
    normed, mean_bl = normalize_body_length(
        smoothed, bodyparts,
        body_length_parts=tuple(kp["body_length_parts"])
    )

    # ── NaN summary (after preprocessing) ─────────────────────────────────────
    nan_mask = np.isnan(normed).any(axis=-1)   # (T, K)
    print("\n── NaN fraction after preprocessing (per keypoint) ─────────────────")
    print(f"  {'Keypoint':<20}  {'NaN (%)':>8}")
    print("  " + "-" * 32)
    for ki, bp in enumerate(bodyparts):
        frac = float(np.mean(nan_mask[:, ki])) * 100
        flag = "  ⚠" if frac > 20 else ""
        print(f"  {bp:<20}  {frac:>7.1f}%{flag}")

    print(f"\n  Total frames : {T}")
    print(f"  Body length  : {mean_bl:.1f} px (median nose→tail_base)")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session_name_from_dlc_file(h5_path)
    saved = save_preprocessed(str(out_dir), session, normed, confs, bodyparts,
                               metadata={"mean_body_length_px": mean_bl})
    logger.info("Preprocessed data saved to: %s", saved)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="One-video pipeline test")
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="Path to the video file to test.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download (weights already present).")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    logger.info("Test video : %s", video_path)
    logger.info("Output dir : %s", TEST_OUT)

    if not args.skip_download:
        download_models(MODEL_DIR)

    h5_files = run_inference(video_path, TEST_OUT / "01_dlc_raw")

    if not h5_files:
        logger.error("No H5 output produced. Check DLC logs above.")
        sys.exit(1)

    preprocess_and_summarise(h5_files[0], TEST_OUT / "02_preprocessed")

    print("\n✓  Test complete.")
    print(f"   DLC output  : {TEST_OUT / '01_dlc_raw'}")
    print(f"   Preprocessed: {TEST_OUT / '02_preprocessed'}")


if __name__ == "__main__":
    main()
