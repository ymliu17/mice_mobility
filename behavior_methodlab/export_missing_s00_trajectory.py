#!/usr/bin/env python3
"""
Re-generate the syllable 00 trajectory plot using the same get_typical_trajectories
approach as _generate_kpms_style_trajectory_plots in s5_posthoc_analysis.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keypoint_moseq as kpms
from keypoint_moseq.util import get_typical_trajectories

from stages import s5_posthoc_analysis as s5
from utils.io_utils import load_all_preprocessed, load_config, load_fps_per_session

ALL_H5  = "outputs/03_moseq/kpms_project_all/model_all/results.h5"
OUTDIR  = Path("outputs/05_stage5/syllable_atlas_presence_gt10/kpms_style_trajectories")
TARGET  = 0


def run() -> None:
    cfg = load_config("config.yaml")
    use_bodyparts = cfg["keypoints"]["use_bodyparts"]

    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
    coordinates = {k: v[:, use_idx, :] for k, v in coords_all.items()}
    results = kpms.load_results(path=ALL_H5)

    fps_map = load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
    median_fps = int(round(float(np.nanmedian(list(fps_map.values()))))) if fps_map else 10
    pre  = max(2, int(round(0.2 * median_fps)))
    post = max(5, int(round(0.6 * median_fps)))

    # density_sample=False is required for syllable 0: its instances are too
    # spread in embedding space to pass the density filter used by other syllables.
    trajectories = get_typical_trajectories(
        coordinates=coordinates,
        results=results,
        pre=pre,
        post=post,
        min_frequency=0.0,
        min_duration=3,
        bodyparts=use_bodyparts,
        use_bodyparts=use_bodyparts,
        density_sample=False,
    )

    if TARGET not in trajectories:
        raise RuntimeError(f"Syllable {TARGET} not found in trajectories.")

    traj = trajectories[TARGET]           # (t, k, 2)
    traj_plot = traj * np.array([1, -1])  # flip y to match screen coords

    nose_idx = use_bodyparts.index("nose")
    tail_idx = use_bodyparts.index("tail_base")

    # center + align — same as main pipeline
    traj_aligned = s5._center_trajectories(traj_plot[None, ...])[0]
    traj_aligned = s5._rotate_trajectory_to_target_heading(
        traj_aligned, nose_idx=nose_idx, tail_idx=tail_idx, target_angle=np.pi / 2.0
    )

    skeleton = s5._build_skeleton(use_bodyparts)
    bp_index = {bp: i for i, bp in enumerate(use_bodyparts)}
    edges = [(bp_index[a], bp_index[b]) for a, b in skeleton if a in bp_index and b in bp_index]

    lims = s5._expand_limits(
        s5._safe_plot_limits(traj_aligned[None, ...], pctl=30),
        frac=0.16, min_pad=0.18,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    s5._plot_compact_trajectory_panel(
        ax,
        traj_aligned,
        edges=edges,
        title="Syllable 0",
        lims=lims,
        shrink=0.62,
        keypoint_colormap="autumn",
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "syllable_00_trajectory.pdf")
    fig.savefig(OUTDIR / "syllable_00_trajectory.png", dpi=180)
    plt.close(fig)
    print(f"Saved → {OUTDIR}/syllable_00_trajectory.png")


if __name__ == "__main__":
    run()
