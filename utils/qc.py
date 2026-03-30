"""Quality-control checks and reports for each pipeline stage."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# ─── DLC / confidence QC ─────────────────────────────────────────────────────

def confidence_report(
    confidences: Dict[str, np.ndarray],
    bodyparts: List[str],
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Return a per-session, per-keypoint table of low-confidence frame fractions.

    Parameters
    ----------
    confidences : {session: (T, K)}
    bodyparts   : keypoint names (length K)
    threshold   : confidence threshold used during preprocessing

    Returns
    -------
    DataFrame with columns [session, keypoint, low_conf_fraction]
    """
    rows = []
    for session, conf in confidences.items():
        for ki, kp in enumerate(bodyparts):
            frac = float(np.mean(conf[:, ki] < threshold))
            rows.append({"session": session, "keypoint": kp, "low_conf_fraction": frac})
    return pd.DataFrame(rows)


def missing_data_report(
    coordinates: Dict[str, np.ndarray],
    bodyparts: List[str],
) -> pd.DataFrame:
    """Fraction of NaN frames per session/keypoint after preprocessing."""
    rows = []
    for session, coords in coordinates.items():
        # coords shape: (T, K, 2)
        nan_mask = np.isnan(coords).any(axis=-1)  # (T, K)
        for ki, kp in enumerate(bodyparts):
            frac = float(np.mean(nan_mask[:, ki]))
            rows.append({"session": session, "keypoint": kp, "nan_fraction": frac})
    return pd.DataFrame(rows)


def body_length_report(
    coordinates: Dict[str, np.ndarray],
    bodyparts: List[str],
    body_length_parts: Tuple[str, str] = ("nose", "tail_base"),
) -> pd.DataFrame:
    """Per-session statistics on body-length estimate (proxy for tracking quality)."""
    idx0 = bodyparts.index(body_length_parts[0])
    idx1 = bodyparts.index(body_length_parts[1])
    rows = []
    for session, coords in coordinates.items():
        # coords shape: (T, K, 2) – may already be normalised (length ~1)
        d = np.linalg.norm(coords[:, idx0] - coords[:, idx1], axis=-1)
        d_valid = d[~np.isnan(d)]
        if len(d_valid) == 0:
            rows.append({"session": session, "mean": np.nan, "std": np.nan,
                         "cv": np.nan, "n_valid": 0})
        else:
            mean = float(np.mean(d_valid))
            std = float(np.std(d_valid))
            rows.append({
                "session": session,
                "mean_body_length": mean,
                "std_body_length": std,
                "cv_body_length": std / mean if mean > 0 else np.nan,
                "n_valid_frames": len(d_valid),
            })
    return pd.DataFrame(rows)


def flag_low_quality_sessions(
    conf_df: pd.DataFrame,
    nan_df: pd.DataFrame,
    max_low_conf: float = 0.30,
    max_nan: float = 0.20,
) -> List[str]:
    """
    Return session names that breach quality thresholds.

    A session is flagged if any keypoint has:
    - low_conf_fraction > max_low_conf, OR
    - nan_fraction > max_nan
    """
    flagged = set()
    for _, row in conf_df.iterrows():
        if row["low_conf_fraction"] > max_low_conf:
            flagged.add(row["session"])
    for _, row in nan_df.iterrows():
        if row["nan_fraction"] > max_nan:
            flagged.add(row["session"])
    return sorted(flagged)


# ─── MoSeq / syllable QC ─────────────────────────────────────────────────────

def syllable_coverage_report(
    results: Dict[str, dict],
    num_states: int = 100,
) -> pd.DataFrame:
    """
    Check how many syllable states were actually used across all sessions.

    Returns a DataFrame with one row per syllable showing total usage count.
    """
    counts = np.zeros(num_states, dtype=int)
    for session, res in results.items():
        syllables = np.asarray(res["syllable"])
        for s in syllables:
            if 0 <= s < num_states:
                counts[s] += 1
    total = counts.sum()
    df = pd.DataFrame({
        "syllable": np.arange(num_states),
        "count": counts,
        "fraction": counts / total if total > 0 else counts,
    })
    return df[df["count"] > 0].reset_index(drop=True)


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def plot_confidence_heatmap(
    conf_df: pd.DataFrame,
    out_path: str,
    threshold: float = 0.6,
) -> None:
    """Save a heatmap of low-confidence fractions per session × keypoint."""
    pivot = conf_df.pivot(index="session", columns="keypoint", values="low_conf_fraction")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.5),
                                    max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    plt.colorbar(im, ax=ax, label=f"Fraction < {threshold}")
    ax.set_title("Low-confidence keypoint fractions (red = problematic)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nan_heatmap(nan_df: pd.DataFrame, out_path: str) -> None:
    pivot = nan_df.pivot(index="session", columns="keypoint", values="nan_fraction")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.5),
                                    max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, vmin=0, vmax=0.5, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="NaN fraction after preprocessing")
    ax.set_title("Missing keypoint fractions after preprocessing")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
