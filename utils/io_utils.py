"""Shared I/O helpers used across pipeline stages."""

from __future__ import annotations

import os
import json
import glob
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Config loading ───────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(base_dir: str, rel_or_abs: str) -> str:
    """Return an absolute path; if rel_or_abs is already absolute, return it."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(base_dir, rel_or_abs)


# ─── Video discovery ──────────────────────────────────────────────────────────

def collect_videos(context_cfg: dict, base_dir: str) -> List[str]:
    """Return sorted list of video file paths for a context config entry."""
    folder = resolve_path(base_dir, context_cfg["folder"])
    exts = context_cfg.get("video_extensions", [".mp4", ".avi", ".mpg"])
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        videos.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(videos))


def collect_all_videos(cfg: dict, base_dir: str) -> Dict[str, List[str]]:
    """Return {context_name: [video_paths]} for all configured contexts."""
    result = {}
    for ctx in cfg["contexts"]:
        result[ctx["name"]] = collect_videos(ctx, base_dir)
    return result


# ─── DLC output discovery ─────────────────────────────────────────────────────

def find_dlc_result_files(
    dlc_dir: str,
    extensions: Tuple[str, ...] = (".h5", ".csv"),
) -> List[str]:
    """Return all DLC result files under dlc_dir."""
    files = []
    for root, _, fnames in os.walk(dlc_dir):
        for fn in fnames:
            if any(fn.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, fn))
    return sorted(files)


def session_name_from_dlc_file(filepath: str) -> str:
    """
    Derive a clean session name from a DLC result filename.

    DLC typically appends a scorer suffix, e.g.:
      ``FX_Th Cell_OF_1DLC_resnet50_superanimal_topviewmouseShuffle1_200000.h5``
    We strip everything after the first 'DLC' suffix.
    """
    stem = Path(filepath).stem
    # Remove DLC scorer suffix if present
    idx = stem.find("DLC")
    if idx > 0:
        stem = stem[:idx].rstrip("_")
    return stem


# ─── Preprocessed data I/O ────────────────────────────────────────────────────

def save_preprocessed(
    out_dir: str,
    session_name: str,
    coordinates: np.ndarray,
    confidences: np.ndarray,
    bodyparts: List[str],
    metadata: Optional[dict] = None,
) -> str:
    """Save preprocessed arrays to a compressed NPZ file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{session_name}.npz")
    save_dict: dict = dict(
        coordinates=coordinates.astype(np.float32),
        confidences=confidences.astype(np.float32),
        bodyparts=np.array(bodyparts),
    )
    if metadata:
        save_dict["metadata_json"] = np.bytes_(json.dumps(metadata))
    np.savez_compressed(path, **save_dict)
    return path


def load_preprocessed(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """Load a preprocessed NPZ file; return (coords, confs, bodyparts, metadata)."""
    npz = np.load(path, allow_pickle=True)
    coords = npz["coordinates"]
    confs = npz["confidences"]
    bodyparts = list(npz["bodyparts"])
    metadata: dict = {}
    if "metadata_json" in npz:
        metadata = json.loads(npz["metadata_json"].item())
    return coords, confs, bodyparts, metadata


def load_fps_per_session(
    preprocessed_dir: str,
    default_fps: float = 30.0,
) -> Dict[str, float]:
    """Return {session_name: effective_fps} loaded from each NPZ's metadata."""
    fps_map: Dict[str, float] = {}
    for path in sorted(glob.glob(os.path.join(preprocessed_dir, "*.npz"))):
        name = Path(path).stem
        _, _, _, metadata = load_preprocessed(path)
        fps_map[name] = float(metadata.get("effective_fps", default_fps))
    return fps_map


def load_all_preprocessed(
    preprocessed_dir: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load all NPZ files from a directory.

    Returns
    -------
    coordinates : {session_name: (T, K, 2)}
    confidences : {session_name: (T, K)}
    bodyparts   : list of K keypoint names (taken from the first file)
    """
    coordinates: Dict[str, np.ndarray] = {}
    confidences: Dict[str, np.ndarray] = {}
    bodyparts: List[str] = []

    for path in sorted(glob.glob(os.path.join(preprocessed_dir, "*.npz"))):
        name = Path(path).stem
        coords, confs, bp, _ = load_preprocessed(path)
        coordinates[name] = coords
        confidences[name] = confs
        if not bodyparts:
            bodyparts = bp

    return coordinates, confidences, bodyparts


# ─── JSON / CSV helpers ───────────────────────────────────────────────────────

def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
