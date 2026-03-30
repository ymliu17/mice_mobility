"""
Stage 3 – keypoint-MoSeq model fitting.

Pipeline:
  1. Load preprocessed keypoints from Stage 2.
  2. Initialise a keypoint-moseq project with the correct bodypart list.
  3. Fit PCA to reduce dimensionality (inspect scree plot to tune pca_components).
  4. Run noise calibration to set observation noise priors.
  5. Fit the keypoint-SLDS model (AR warm-up → full SLDS).
  6. Extract per-session syllable labels and save results.

Writes:
  outputs/03_moseq/<model_name>/
      config.yml               – keypoint-moseq project config
      pca.p                    – fitted PCA
      noise_calibration.p      – noise prior params
      checkpoint_<iter>.p      – model checkpoints
      results_<model>.h5       – syllable labels, latent states, centroids
      results_<model>.csv      – human-readable CSV

Usage:
  python pipeline.py run --stages 3
  python stages/s3_moseq_fitting.py --config config.yaml
"""

from __future__ import annotations

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.io_utils import load_config, resolve_path, load_all_preprocessed

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _filter_to_use_bodyparts(
    coordinates: Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    bodyparts: List[str],
    use_bodyparts: List[str],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Subset coordinates and confidences to only the keypoints in use_bodyparts.

    Parameters
    ----------
    coordinates : {session: (T, K_all, 2)}
    confidences : {session: (T, K_all)}
    bodyparts   : all K_all keypoint names
    use_bodyparts : subset of keypoint names to keep

    Returns
    -------
    coords_sub : {session: (T, K_use, 2)}
    confs_sub  : {session: (T, K_use)}
    """
    indices = [bodyparts.index(bp) for bp in use_bodyparts]
    coords_sub = {k: v[:, indices, :] for k, v in coordinates.items()}
    confs_sub = {k: v[:, indices] for k, v in confidences.items()}
    return coords_sub, confs_sub


def _build_skeleton(use_bodyparts: List[str]) -> List[List[str]]:
    """
    Return a sensible skeleton for the 13-keypoint TopViewMouse subset.

    Edges connect anatomically adjacent landmarks.
    Only edges where both endpoints are in use_bodyparts are included.
    """
    candidate_edges = [
        # Head
        ["nose", "left_ear"],
        ["nose", "right_ear"],
        ["nose", "neck"],
        ["left_ear", "neck"],
        ["right_ear", "neck"],
        # Spine
        ["neck", "mid_back"],
        ["mid_back", "mouse_center"],
        ["mouse_center", "tail_base"],
        # Left limb chain
        ["neck", "left_shoulder"],
        ["left_shoulder", "left_midside"],
        ["left_midside", "left_hip"],
        ["left_hip", "tail_base"],
        # Right limb chain
        ["neck", "right_shoulder"],
        ["right_shoulder", "right_midside"],
        ["right_midside", "right_hip"],
        ["right_hip", "tail_base"],
    ]
    bp_set = set(use_bodyparts)
    return [e for e in candidate_edges if e[0] in bp_set and e[1] in bp_set]


# ─── Main fitting logic ───────────────────────────────────────────────────────

def run(config_path: str = "config.yaml") -> None:
    import keypoint_moseq as kpms

    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    preprocessed_dir = resolve_path(base_dir, cfg["paths"]["preprocessed_dir"])
    moseq_dir = resolve_path(base_dir, cfg["paths"]["moseq_project_dir"])
    kp_cfg = cfg["keypoints"]
    moseq_cfg = cfg["moseq"]

    use_bodyparts: List[str] = kp_cfg["use_bodyparts"]
    model_name: str = moseq_cfg["model_name"]
    pca_components: int = moseq_cfg["pca_components"]

    os.makedirs(moseq_dir, exist_ok=True)

    # ── 1. Load preprocessed data ─────────────────────────────────────────────
    logger.info("Loading preprocessed keypoints from %s ...", preprocessed_dir)
    coordinates, confidences, bodyparts = load_all_preprocessed(preprocessed_dir)

    if not coordinates:
        logger.error("No preprocessed files found. Run Stage 2 first.")
        return

    logger.info("Loaded %d sessions.", len(coordinates))

    # Filter to the use_bodyparts subset
    coordinates, confidences = _filter_to_use_bodyparts(
        coordinates, confidences, bodyparts, use_bodyparts
    )

    # ── 2. Set up keypoint-moseq project ──────────────────────────────────────
    skeleton = _build_skeleton(use_bodyparts)

    project_dir = os.path.join(moseq_dir, "kpms_project")
    kpms_config_path = os.path.join(project_dir, "config.yml")

    if not os.path.exists(kpms_config_path):
        logger.info("Initialising keypoint-moseq project at %s ...", project_dir)
        os.makedirs(project_dir, exist_ok=True)
        kpms.generate_config(
            project_dir,
            bodyparts=use_bodyparts,
            use_bodyparts=use_bodyparts,
            skeleton=skeleton,
            video_dir=preprocessed_dir,
            # MoSeq hyper-parameters from our config
            error_estimator={"slope": -0.5, "intercept": 0.25},
            ar_hypparams={
                "latent_dim": pca_components,
                "nlags": moseq_cfg.get("nlags", 3),
                "S_0_scale": 0.01,
                "K_0_scale": 10.0,
            },
            trans_hypparams={
                "num_states": moseq_cfg.get("num_states", 100),
                "gamma": moseq_cfg.get("kappa", 1e6),  # HDP concentration
                "alpha": 5.7,
                "kappa": moseq_cfg.get("kappa", 1e6),
            },
        )
    else:
        logger.info("Found existing keypoint-moseq config at %s.", kpms_config_path)

    # ── 3. Format data ────────────────────────────────────────────────────────
    logger.info("Formatting data for keypoint-moseq ...")
    data, metadata = kpms.format_data(
        coordinates,
        confidences=confidences,
        project_dir=project_dir,
    )

    # ── 4. PCA ────────────────────────────────────────────────────────────────
    pca_path = os.path.join(project_dir, "pca.p")
    if os.path.exists(pca_path):
        logger.info("Loading existing PCA from %s ...", pca_path)
        pca = kpms.load_pca(project_dir)
    else:
        logger.info("Fitting PCA (%d components) ...", pca_components)
        pca = kpms.fit_pca(
            data,
            project_dir=project_dir,
            pca_path=pca_path,
        )
        # Print variance explanation for the chosen number of components
        from keypoint_moseq.util import print_dims_to_explain_variance
        print_dims_to_explain_variance(pca, 0.9)

    # Save a scree plot for manual inspection
    kpms.plot_scree(pca, project_dir=project_dir)

    # ── 5. Noise calibration ──────────────────────────────────────────────────
    noise_path = os.path.join(project_dir, "noise_calibration.p")
    if not os.path.exists(noise_path):
        logger.info("Running noise calibration ...")
        kpms.noise_calibration(
            project_dir=project_dir,
            coordinates=coordinates,
            confidences=confidences,
        )
    else:
        logger.info("Found existing noise calibration at %s.", noise_path)

    # ── 6. Initialise model ───────────────────────────────────────────────────
    logger.info("Initialising keypoint-SLDS model ...")
    model = kpms.init_model(data, pca=pca, project_dir=project_dir)

    # ── 7. AR warm-up ─────────────────────────────────────────────────────────
    num_ar_iters: int = moseq_cfg.get("num_ar_iters", 50)
    logger.info("AR warm-up: %d iterations ...", num_ar_iters)
    model, ar_history, _ = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=project_dir,
        model_name=model_name,
        ar_only=True,
        num_iters=num_ar_iters,
        save_every=moseq_cfg.get("save_every", 25),
    )

    # ── 8. Full SLDS fitting ──────────────────────────────────────────────────
    num_full_iters: int = moseq_cfg.get("num_full_iters", 200)
    logger.info("Full SLDS fitting: %d iterations ...", num_full_iters)
    model, full_history, _ = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=project_dir,
        model_name=model_name,
        ar_only=False,
        num_iters=num_full_iters,
        save_every=moseq_cfg.get("save_every", 25),
    )

    # ── 9. Extract and save results ───────────────────────────────────────────
    logger.info("Extracting results ...")
    results = kpms.extract_results(
        model=model,
        metadata=metadata,
        project_dir=project_dir,
        model_name=model_name,
    )

    kpms.save_results_as_csv(
        results=results,
        project_dir=project_dir,
        model_name=model_name,
    )

    logger.info(
        "Stage 3 complete. Results at: %s",
        os.path.join(project_dir, model_name),
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Stage 3: keypoint-MoSeq fitting")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
