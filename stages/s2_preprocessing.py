"""
Stage 2 – Keypoint preprocessing.

Reads DLC H5 files produced by Stage 1 and applies:
  1. Confidence masking    – set coords to NaN where confidence < threshold
  2. Gap interpolation     – linear spline for runs of NaN ≤ max_interp_gap
  3. Savitzky-Golay smooth – remove high-frequency jitter
  4. Body-length normalization – divide all coords by per-session mean body length
  5. (Optional) session-level z-scoring of each coordinate dimension

Writes per-session compressed NPZ files to outputs/02_preprocessed/<session>.npz.

Usage:
  python pipeline.py run --stages 2
  python stages/s2_preprocessing.py --config config.yaml
"""

from __future__ import annotations

import json
import os
import re
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from utils.io_utils import (
    load_config,
    resolve_path,
    find_dlc_result_files,
    session_name_from_dlc_file,
    save_preprocessed,
)
from utils.qc import (
    confidence_report,
    missing_data_report,
    body_length_report,
    flag_low_quality_sessions,
    plot_confidence_heatmap,
    plot_nan_heatmap,
)

logger = logging.getLogger(__name__)


# ─── DLC H5 loader ────────────────────────────────────────────────────────────

def load_dlc_h5(filepath: str, animal: str = "animal0") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a DLC result H5 file.

    Handles both:
    - DLC 2.x: (scorer, bodypart, coord)           – 3-level MultiIndex
    - DLC 3.x: (scorer, animal, bodypart, coord)   – 4-level MultiIndex

    For multi-animal outputs the ``animal`` argument selects which animal
    to extract (default: ``"animal0"``).

    Returns
    -------
    coordinates : (T, K, 2)  – x/y pixel positions
    confidences : (T, K)     – likelihood scores in [0, 1]
    bodyparts   : list of K keypoint names
    """
    df = pd.read_hdf(filepath)

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"Unexpected column structure in {filepath}: {df.columns[:5]}")

    nlevels = df.columns.nlevels
    scorer = df.columns.get_level_values(0)[0]

    if nlevels == 4:
        # DLC 3.x: (scorer, animal, bodypart, coord)
        available_animals = df[scorer].columns.get_level_values(0).unique().tolist()
        if animal not in available_animals:
            logger.warning(
                "Animal '%s' not found in %s. Available: %s. Using first.",
                animal, filepath, available_animals,
            )
            animal = available_animals[0]
        sub = df[scorer, animal]
    elif nlevels == 3:
        # DLC 2.x: (scorer, bodypart, coord)
        sub = df[scorer]
    else:
        raise ValueError(f"Unsupported MultiIndex depth {nlevels} in {filepath}")

    bodyparts = sub.columns.get_level_values(0).unique().tolist()
    T, K = len(df), len(bodyparts)
    coords = np.full((T, K, 2), np.nan, dtype=np.float32)
    confs  = np.full((T, K),    np.nan, dtype=np.float32)
    for ki, bp in enumerate(bodyparts):
        coords[:, ki, 0] = sub[bp, "x"].values
        coords[:, ki, 1] = sub[bp, "y"].values
        confs[:, ki] = sub[bp, "likelihood"].values if "likelihood" in sub[bp] else 1.0

    return coords, confs, bodyparts


def load_dlc_h5_all_animals(
    filepath: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load all animals from a DLC 3.x H5 file.

    Returns
    -------
    coords_by_animal : {animal_name: (T, K, 2)}
    confs_by_animal  : {animal_name: (T, K)}
    bodyparts        : list of K keypoint names
    """
    df = pd.read_hdf(filepath)

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"Unexpected column structure in {filepath}")

    nlevels = df.columns.nlevels
    scorer = df.columns.get_level_values(0)[0]

    if nlevels != 4:
        # Single-animal file — wrap as animal0
        coords, confs, bodyparts = load_dlc_h5(filepath)
        return {"animal0": coords}, {"animal0": confs}, bodyparts

    available_animals = df[scorer].columns.get_level_values(0).unique().tolist()
    bodyparts = df[scorer, available_animals[0]].columns.get_level_values(0).unique().tolist()
    T, K = len(df), len(bodyparts)

    coords_by_animal: Dict[str, np.ndarray] = {}
    confs_by_animal:  Dict[str, np.ndarray] = {}

    for animal in available_animals:
        sub = df[scorer, animal]
        coords = np.full((T, K, 2), np.nan, dtype=np.float32)
        confs  = np.full((T, K),    np.nan, dtype=np.float32)
        for ki, bp in enumerate(bodyparts):
            coords[:, ki, 0] = sub[bp, "x"].values
            coords[:, ki, 1] = sub[bp, "y"].values
            confs[:, ki] = sub[bp, "likelihood"].values if "likelihood" in sub[bp] else 1.0
        coords_by_animal[animal] = coords
        confs_by_animal[animal]  = confs

    return coords_by_animal, confs_by_animal, bodyparts


def _parse_surgery_mouse_pair(session_name: str) -> Optional[Tuple[int, int]]:
    """
    Extract the (odd_id, even_id) mouse pair from a surgery_15mice session name.

    Filenames encode the pair as '<date>-X -Y', e.g. '240204-1 -2', where X is
    the right (odd-numbered) mouse and Y is the left (even-numbered) mouse.

    Returns (odd_id, even_id) or None if no pair found (e.g. Y-maze single-animal).
    """
    # Match the " -X -Y" pattern (space-dash first, space-dash second)
    m = re.search(r'-(\d+)\s+-(\d+)', session_name)
    if m is None:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    # Determine which is odd (right) and which is even (left)
    if a % 2 == 1 and b % 2 == 0:
        return a, b   # (odd, even)
    if a % 2 == 0 and b % 2 == 1:
        return b, a   # reorder to (odd, even)
    logger.warning("Unexpected mouse pair '%s-%s' in '%s'; both odd or both even.", a, b, session_name)
    return a, b


def _parse_surgery_single_mouse_id(session_name: str) -> Optional[int]:
    """
    Extract the single mouse ID from a surgery_15mice Y-maze session name.

    Filenames like 'POD+1 Y maze 240204-3_superanimal...' encode a single mouse
    as '<date>-<id>', e.g. '240204-3' → mouse ID 3.

    Returns the integer mouse ID, or None if not found.
    """
    # Match date-singleID but NOT date-id space -id (pair format)
    m = re.search(r'\d{6}-(\d+)(?!\s*-\d)', session_name)
    if m:
        return int(m.group(1))
    return None


def _mean_x_of_animal(coords: np.ndarray, bodyparts: List[str], ref_part: str = "mouse_center") -> float:
    """Return the mean x-coordinate of ref_part over valid frames."""
    if ref_part in bodyparts:
        idx = bodyparts.index(ref_part)
    else:
        idx = len(bodyparts) // 2  # fallback: middle keypoint
    x = coords[:, idx, 0]
    valid = x[~np.isnan(x)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def load_dlc_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a DLC CSV result file (fallback when H5 is absent)."""
    # DLC CSVs have two header rows: bodypart names + coord names
    df = pd.read_csv(filepath, header=[1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(0).unique().tolist()
    T = len(df)
    K = len(bodyparts)
    coords = np.full((T, K, 2), np.nan, dtype=np.float32)
    confs = np.full((T, K), np.nan, dtype=np.float32)
    for ki, bp in enumerate(bodyparts):
        coords[:, ki, 0] = df[bp]["x"].values
        coords[:, ki, 1] = df[bp]["y"].values
        if "likelihood" in df[bp].columns:
            confs[:, ki] = df[bp]["likelihood"].values
        else:
            confs[:, ki] = 1.0
    return coords, confs, bodyparts


# ─── Preprocessing steps ──────────────────────────────────────────────────────

def apply_confidence_mask(
    coords: np.ndarray,
    confs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Set keypoint positions to NaN wherever confidence < threshold.

    Parameters
    ----------
    coords : (T, K, 2)
    confs  : (T, K)
    threshold : float

    Returns
    -------
    masked_coords : (T, K, 2)
    """
    masked = coords.copy()
    low = confs < threshold               # (T, K) boolean
    masked[low] = np.nan                  # broadcast over xy dimension
    return masked


def interpolate_gaps(
    coords: np.ndarray,
    max_gap: Optional[int] = 10,
) -> np.ndarray:
    """
    Linearly interpolate NaN runs of length ≤ max_gap per keypoint dimension.

    If max_gap is None, all gaps are interpolated regardless of length.

    Parameters
    ----------
    coords  : (T, K, 2)
    max_gap : int or None

    Returns
    -------
    interpolated : (T, K, 2)
    """
    out = coords.copy()
    T, K, _ = coords.shape
    t = np.arange(T, dtype=float)

    for ki in range(K):
        for di in range(2):
            series = out[:, ki, di]
            nan_mask = np.isnan(series)
            if not nan_mask.any():
                continue
            valid = ~nan_mask
            if valid.sum() < 2:
                continue  # not enough valid points to interpolate

            if max_gap is None:
                # Interpolate all gaps
                f = interp1d(t[valid], series[valid], kind="linear",
                             bounds_error=False, fill_value=np.nan)
                filled = f(t)
                # Keep extrapolated (boundary) regions as NaN
                interp_range = slice(t[valid].min().astype(int),
                                     t[valid].max().astype(int) + 1)
                out[:, ki, di][interp_range] = filled[interp_range]
            else:
                # Only interpolate gaps ≤ max_gap
                interp_series = series.copy()
                f = interp1d(t[valid], series[valid], kind="linear",
                             bounds_error=False, fill_value=np.nan)
                candidate = f(t)

                # Walk through NaN runs and fill only short ones
                i = 0
                while i < T:
                    if nan_mask[i]:
                        j = i
                        while j < T and nan_mask[j]:
                            j += 1
                        gap_len = j - i
                        if gap_len <= max_gap:
                            interp_series[i:j] = candidate[i:j]
                        i = j
                    else:
                        i += 1
                out[:, ki, di] = interp_series

    return out


def smooth_savgol(
    coords: np.ndarray,
    window: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to each keypoint dimension independently.

    NaN segments are skipped (filter applied only to valid contiguous blocks).

    Parameters
    ----------
    coords     : (T, K, 2)
    window     : filter window length (must be odd)
    polyorder  : polynomial order

    Returns
    -------
    smoothed : (T, K, 2)
    """
    if window < 3 or window % 2 == 0:
        logger.warning("smooth_window must be an odd integer ≥ 3; skipping smoothing.")
        return coords

    out = coords.copy()
    T, K, _ = coords.shape

    for ki in range(K):
        for di in range(2):
            series = out[:, ki, di]
            nan_mask = np.isnan(series)
            if nan_mask.all():
                continue

            # Apply filter to each contiguous valid block separately
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) == 0:
                continue

            # Find contiguous blocks
            breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
            blocks = np.split(valid_indices, breaks)

            for block in blocks:
                if len(block) < window:
                    continue  # block too short for the filter
                smooth_block = savgol_filter(
                    series[block], window_length=window, polyorder=polyorder
                )
                out[block, ki, di] = smooth_block

    return out


def normalize_body_length(
    coords: np.ndarray,
    bodyparts: List[str],
    body_length_parts: Tuple[str, str] = ("nose", "tail_base"),
) -> Tuple[np.ndarray, float]:
    """
    Divide all coordinates by the per-session mean body length.

    Body length = distance between body_length_parts[0] and body_length_parts[1].
    Only valid (non-NaN) frames contribute to the mean.

    Returns
    -------
    norm_coords : (T, K, 2)
    mean_body_length : float  – stored for inverse-transform if needed
    """
    idx0 = bodyparts.index(body_length_parts[0])
    idx1 = bodyparts.index(body_length_parts[1])
    diffs = coords[:, idx0] - coords[:, idx1]            # (T, 2)
    lengths = np.linalg.norm(diffs, axis=-1)             # (T,)
    valid_lengths = lengths[~np.isnan(lengths)]

    if len(valid_lengths) == 0:
        logger.warning("No valid frames for body-length estimation; skipping normalisation.")
        return coords, 1.0

    mean_bl = float(np.median(valid_lengths))   # median is more robust than mean
    if mean_bl == 0:
        logger.warning("Body length is zero; skipping normalisation.")
        return coords, 1.0

    return coords / mean_bl, mean_bl


def session_zscore(coords: np.ndarray) -> np.ndarray:
    """
    Z-score each keypoint×dimension channel within the session.

    Only valid frames contribute to mean/std computation.
    """
    out = coords.copy()
    T, K, D = coords.shape
    for ki in range(K):
        for di in range(D):
            series = out[:, ki, di]
            valid = series[~np.isnan(series)]
            if len(valid) < 2:
                continue
            mu = float(np.mean(valid))
            sigma = float(np.std(valid))
            if sigma > 0:
                out[:, ki, di] = (series - mu) / sigma
    return out


# ─── Full preprocessing pipeline ─────────────────────────────────────────────

def preprocess_session(
    coords: np.ndarray,
    confs: np.ndarray,
    bodyparts: List[str],
    prep_cfg: dict,
    kp_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply all preprocessing steps to a single session.

    Returns
    -------
    coords_out       : (T, K, 2) – preprocessed coordinates
    confs_out        : (T, K)    – original confidence scores (unchanged)
    mean_body_length : float
    """
    threshold = prep_cfg.get("conf_threshold", 0.6)
    max_gap = prep_cfg.get("max_interp_gap", 10)
    win = prep_cfg.get("smooth_window", 5)
    poly = prep_cfg.get("smooth_polyorder", 2)
    do_norm = prep_cfg.get("normalize_body_length", True)
    do_zscore = prep_cfg.get("session_standardize", False)
    bl_parts = kp_cfg.get("body_length_parts", ["nose", "tail_base"])

    # Step 1: confidence masking
    coords = apply_confidence_mask(coords, confs, threshold)

    # Step 2: gap interpolation
    if max_gap is not None and max_gap > 0:
        coords = interpolate_gaps(coords, max_gap=max_gap)

    # Step 3: smoothing
    if win and win >= 3:
        coords = smooth_savgol(coords, window=int(win), polyorder=int(poly))

    # Step 4: body-length normalization
    mean_bl = 1.0
    if do_norm:
        coords, mean_bl = normalize_body_length(coords, bodyparts, tuple(bl_parts))

    # Step 5: optional z-scoring
    if do_zscore:
        coords = session_zscore(coords)

    return coords, confs, mean_bl


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(config_path: str = "config.yaml", filter_contexts: Optional[List[str]] = None) -> None:
    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    dlc_dir = resolve_path(base_dir, cfg["paths"]["dlc_output_dir"])
    out_dir = resolve_path(base_dir, cfg["paths"]["preprocessed_dir"])
    qc_dir = os.path.join(out_dir, "qc")
    prep_cfg = cfg["preprocessing"]
    kp_cfg = cfg["keypoints"]

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    # Discover all DLC H5 result files
    dlc_files = find_dlc_result_files(dlc_dir, extensions=(".h5",))
    if not dlc_files:
        logger.warning(
            "No H5 files found in %s – trying CSV fallback ...", dlc_dir
        )
        dlc_files = find_dlc_result_files(dlc_dir, extensions=(".csv",))

    if not dlc_files:
        logger.error(
            "No DLC result files found under %s. Run Stage 1 first.", dlc_dir
        )
        return

    if filter_contexts:
        dlc_files = [f for f in dlc_files if Path(f).parent.name in filter_contexts]
        logger.info("Context filter applied – processing: %s", filter_contexts)

    logger.info("Found %d DLC result files.", len(dlc_files))

    # ── Load fps manifests written by Stage 1 (keyed by DLC output directory) ─
    _fps_cache: Dict[str, dict] = {}

    # Manifests are keyed by the bare video stem, but *session* still carries the
    # pose-model suffix (session_name_from_dlc_file only strips a "DLC" suffix,
    # which SuperAnimal filenames do not have). Match on the longest manifest key
    # that prefixes the session name.
    _fps_fallback = float(cfg.get("dlc", {}).get("target_fps") or 30.0)

    def _effective_fps(fpath: str, session: str) -> float:
        ctx_dir = str(Path(fpath).parent)
        if ctx_dir not in _fps_cache:
            mp = os.path.join(ctx_dir, "fps_manifest.json")
            if os.path.exists(mp):
                with open(mp) as mf:
                    _fps_cache[ctx_dir] = json.load(mf)
            else:
                logger.warning(
                    "No fps_manifest.json in %s - every session there falls back "
                    "to %.1f fps.", ctx_dir, _fps_fallback,
                )
                _fps_cache[ctx_dir] = {}

        manifest = _fps_cache[ctx_dir]
        if session in manifest:
            return float(manifest[session])

        prefixes = [k for k in manifest if session.startswith(k)]
        if prefixes:
            return float(manifest[max(prefixes, key=len)])

        logger.warning(
            "Session '%s' absent from %s/fps_manifest.json (%d keys) - falling "
            "back to %.1f fps. Per-second metrics will be wrong if that is not "
            "the true rate.", session, ctx_dir, len(manifest), _fps_fallback,
        )
        return _fps_fallback

    # ── Load, preprocess, save ────────────────────────────────────────────────
    raw_confidences: Dict[str, np.ndarray] = {}
    all_bodyparts: List[str] = []

    for fpath in dlc_files:
        session = session_name_from_dlc_file(fpath)
        logger.info("Processing session '%s' ...", session)

        # Detect surgery_15mice two-animal videos by context folder name
        is_surgery_15 = Path(fpath).parent.name == "surgery_15mice"

        try:
            if fpath.endswith(".h5") and is_surgery_15:
                coords_all, confs_all, bodyparts = load_dlc_h5_all_animals(fpath)
            elif fpath.endswith(".h5"):
                _c, _cf, bodyparts = load_dlc_h5(fpath)
                coords_all, confs_all = {"animal0": _c}, {"animal0": _cf}
            else:
                _c, _cf, bodyparts = load_dlc_csv(fpath)
                coords_all, confs_all = {"animal0": _c}, {"animal0": _cf}
        except Exception as exc:
            logger.error("  Failed to load %s: %s", fpath, exc)
            continue

        if not all_bodyparts:
            all_bodyparts = bodyparts

        # Only split two-animal videos when filename encodes a pair (e.g. "240204-1 -2").
        # Y-maze videos have a single mouse (e.g. "240204-1") — no pair → no split.
        pair = _parse_surgery_mouse_pair(session) if is_surgery_15 else None

        if is_surgery_15 and len(coords_all) >= 2 and pair is not None:
            # Assign animal0/animal1 to specific mouse IDs by x-position
            odd_id, even_id = pair
            animals = list(coords_all.keys())[:2]
            mean_x = {a: _mean_x_of_animal(coords_all[a], bodyparts) for a in animals}
            left_animal  = min(animals, key=lambda a: mean_x[a])   # lower x = left = even
            right_animal = max(animals, key=lambda a: mean_x[a])   # higher x = right = odd
            assignments = {left_animal: even_id, right_animal: odd_id}

            for animal, mouse_id in assignments.items():
                sub_session = f"{session}_mouse{mouse_id}"
                coords = coords_all[animal]
                confs  = confs_all[animal]

                raw_confidences[sub_session] = confs.copy()
                coords_out, confs_out, mean_bl = preprocess_session(
                    coords, confs, bodyparts, prep_cfg, kp_cfg
                )
                metadata = {
                    "source_file": fpath,
                    "animal_id": animal,
                    "mouse_id": mouse_id,
                    "mean_body_length_px": mean_bl,
                    "n_frames": int(coords_out.shape[0]),
                    "effective_fps": _effective_fps(fpath, session),
                }
                save_preprocessed(out_dir, sub_session, coords_out, confs_out, bodyparts, metadata)
                logger.info(
                    "  → mouse%s (%s, mean_x=%.1f): %d frames (body length: %.1f px)",
                    mouse_id, animal, mean_x[animal], coords_out.shape[0], mean_bl,
                )
        else:
            # Single-animal session (all other contexts, and surgery_15mice Y-maze)
            coords = list(coords_all.values())[0]
            confs  = list(confs_all.values())[0]

            # For surgery_15mice single-animal videos (Y-maze), append mouse ID from filename
            if is_surgery_15:
                mouse_id = _parse_surgery_single_mouse_id(session)
                save_session = f"{session}_mouse{mouse_id}" if mouse_id is not None else session
            else:
                save_session = session

            raw_confidences[save_session] = confs.copy()
            coords_out, confs_out, mean_bl = preprocess_session(
                coords, confs, bodyparts, prep_cfg, kp_cfg
            )
            metadata = {
                "source_file": fpath,
                "mean_body_length_px": mean_bl,
                "n_frames": int(coords_out.shape[0]),
                "effective_fps": _effective_fps(fpath, session),
            }
            save_preprocessed(out_dir, save_session, coords_out, confs_out, bodyparts, metadata)
            logger.info(
                "  → %d frames saved (body length: %.1f px, fps: %.1f)",
                coords_out.shape[0], mean_bl, metadata["effective_fps"],
            )

    # ── QC reports ────────────────────────────────────────────────────────────
    if all_bodyparts and raw_confidences:
        conf_df = confidence_report(
            raw_confidences, all_bodyparts, threshold=prep_cfg["conf_threshold"]
        )
        conf_df.to_csv(os.path.join(qc_dir, "qc_confidence.csv"), index=False)
        plot_confidence_heatmap(
            conf_df,
            os.path.join(qc_dir, "qc_confidence_heatmap.png"),
            threshold=prep_cfg["conf_threshold"],
        )

        flagged = flag_low_quality_sessions(
            conf_df,
            pd.DataFrame(columns=["session", "keypoint", "nan_fraction"]),
        )
        if flagged:
            logger.warning("LOW-QUALITY sessions (review QC report): %s", flagged)
        else:
            logger.info("All sessions passed QC thresholds.")

    logger.info("Stage 2 complete.  Preprocessed files: %s", out_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Stage 2: Keypoint preprocessing")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--contexts", nargs="+", metavar="CTX",
        help="Only process these context names (default: all).",
    )
    args = parser.parse_args()
    run(args.config, filter_contexts=args.contexts)
