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
import re
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


# ─── Core fitting helper ──────────────────────────────────────────────────────

def _fit_one_model(
    kpms,
    coordinates: Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    use_bodyparts: List[str],
    moseq_cfg: dict,
    project_dir: str,
    preprocessed_dir: str,
    label: str = "",
) -> None:
    """Fit a single keypoint-MoSeq model for the given set of sessions."""
    from jax_moseq.utils.debugging import convert_data_precision
    from keypoint_moseq.util import print_dims_to_explain_variance

    prefix = f"[{label}] " if label else ""
    pca_components: int = moseq_cfg["pca_components"]
    model_name: str = moseq_cfg["model_name"]
    skeleton = _build_skeleton(use_bodyparts)

    os.makedirs(project_dir, exist_ok=True)
    kpms_config_path = os.path.join(project_dir, "config.yml")

    if not os.path.exists(kpms_config_path):
        logger.info("%sInitialising keypoint-moseq project at %s ...", prefix, project_dir)
        kpms.generate_config(
            project_dir,
            bodyparts=use_bodyparts,
            use_bodyparts=use_bodyparts,
            skeleton=skeleton,
            video_dir=preprocessed_dir,
            anterior_bodyparts=["nose"],
            posterior_bodyparts=["tail_base"],
            error_estimator={"slope": -0.5, "intercept": 0.25},
            ar_hypparams={
                "latent_dim": pca_components,
                "nlags": moseq_cfg.get("nlags", 3),
                "S_0_scale": 0.01,
                "K_0_scale": 10.0,
            },
            trans_hypparams={
                "num_states": moseq_cfg.get("num_states", 100),
                "gamma": moseq_cfg.get("kappa", 1e6),
                "alpha": 5.7,
                "kappa": moseq_cfg.get("kappa", 1e6),
            },
        )
    else:
        logger.info("%sFound existing config at %s.", prefix, kpms_config_path)
        kpms.update_config(project_dir,
                           anterior_bodyparts=["nose"],
                           posterior_bodyparts=["tail_base"])

    logger.info("%sFormatting %d sessions ...", prefix, len(coordinates))
    data, metadata = kpms.format_data(coordinates, confidences=confidences,
                                      project_dir=project_dir)
    data = convert_data_precision(data)

    pca_path = os.path.join(project_dir, "pca.p")
    if os.path.exists(pca_path):
        logger.info("%sLoading existing PCA ...", prefix)
        pca = kpms.load_pca(project_dir)
    else:
        logger.info("%sFitting PCA (%d components) ...", prefix, pca_components)
        anterior_idxs  = [use_bodyparts.index("nose")]
        posterior_idxs = [use_bodyparts.index("tail_base")]
        pca = kpms.fit_pca(
            data["Y"], data["mask"],
            anterior_idxs=anterior_idxs, posterior_idxs=posterior_idxs,
            conf=data.get("conf"), project_dir=project_dir, pca_path=pca_path,
        )
        print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=project_dir)

    logger.info("%sSkipping noise calibration – using config defaults.", prefix)

    # Save PCA so downstream scripts (robustness checks, apply_model) can load it
    if not os.path.exists(pca_path):
        kpms.save_pca(pca, project_dir)
        logger.info("%sPCA saved to %s", prefix, pca_path)

    num_ar_iters: int  = moseq_cfg.get("num_ar_iters", 50)
    num_full_iters: int = moseq_cfg.get("num_full_iters", 200)
    save_every: int    = moseq_cfg.get("save_every", 25)

    model_dir         = os.path.join(project_dir, model_name)
    checkpoint_h5     = os.path.join(model_dir, "checkpoint.h5")
    total_iters_file  = os.path.join(model_dir, "total_full_iters.txt")

    # Read how many full SLDS iters have been completed so far
    done_iters = 0
    if os.path.exists(total_iters_file):
        done_iters = int(Path(total_iters_file).read_text().strip())

    remaining = num_full_iters - done_iters

    if remaining <= 0:
        logger.info("%sModel already trained to %d iters (target %d) – extracting results.",
                    prefix, done_iters, num_full_iters)
        if os.path.exists(checkpoint_h5):
            ckpt = kpms.load_checkpoint(project_dir, model_name)
            model = ckpt[0] if isinstance(ckpt, tuple) else ckpt
        else:
            logger.warning("%sNo checkpoint found; cannot extract results.", prefix)
            return
    elif done_iters > 0 and os.path.exists(checkpoint_h5):
        logger.info("%sResuming from checkpoint (%d done, %d remaining of %d target).",
                    prefix, done_iters, remaining, num_full_iters)
        ckpt = kpms.load_checkpoint(project_dir, model_name)
        model = ckpt[0] if isinstance(ckpt, tuple) else ckpt
        model, _ = kpms.fit_model(
            model=model, data=data, metadata=metadata,
            project_dir=project_dir, model_name=model_name,
            ar_only=False, num_iters=remaining, save_every=save_every,
            parallel_message_passing=False,
        )
        Path(total_iters_file).write_text(str(num_full_iters))
    else:
        logger.info("%sInitialising model ...", prefix)
        kpms_cfg = kpms.load_config(project_dir)
        anterior_idxs  = [use_bodyparts.index(bp) for bp in kpms_cfg["anterior_bodyparts"]]
        posterior_idxs = [use_bodyparts.index(bp) for bp in kpms_cfg["posterior_bodyparts"]]
        trans_hp = {k: float(v) if isinstance(v, str) else v
                    for k, v in kpms_cfg["trans_hypparams"].items()}
        model = kpms.init_model(
            data, pca=pca, project_dir=project_dir,
            anterior_idxs=anterior_idxs, posterior_idxs=posterior_idxs,
            trans_hypparams=trans_hp, ar_hypparams=kpms_cfg["ar_hypparams"],
            obs_hypparams=kpms_cfg["obs_hypparams"], cen_hypparams=kpms_cfg["cen_hypparams"],
            error_estimator=kpms_cfg["error_estimator"],
        )
        logger.info("%sAR warm-up: %d iters ...", prefix, num_ar_iters)
        model, _ = kpms.fit_model(
            model=model, data=data, metadata=metadata,
            project_dir=project_dir, model_name=model_name,
            ar_only=True, num_iters=num_ar_iters, save_every=save_every,
            parallel_message_passing=False,
        )
        logger.info("%sFull SLDS: %d iters ...", prefix, num_full_iters)
        model, _ = kpms.fit_model(
            model=model, data=data, metadata=metadata,
            project_dir=project_dir, model_name=model_name,
            ar_only=False, num_iters=num_full_iters, save_every=save_every,
            parallel_message_passing=False,
        )
        Path(total_iters_file).write_text(str(num_full_iters))

    logger.info("%sExtracting results ...", prefix)
    results = kpms.extract_results(model=model, metadata=metadata,
                                   project_dir=project_dir, model_name=model_name)
    kpms.save_results_as_csv(results=results, project_dir=project_dir,
                              model_name=model_name)
    logger.info("%sDone. Results at: %s", prefix, os.path.join(project_dir, model_name))


# ─── Test-type helpers for joint model BFT filtering ─────────────────────────

def _build_test_type_map(
    contexts: list,
    base_dir: str,
    ctx_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Return {session_name: test_type} for all sessions.

    For surgery_15mice: parse test type from session name (contains BFT/OFT/Y maze).
    For contexts with a metadata_csv: load CSV and look up by video_stem.
    """
    _TT_RE = re.compile(r"\b(BFT|OFT|Y\s+maze)\b", re.IGNORECASE)
    _SUPER_RE = re.compile(r"_superanimal.*$", re.IGNORECASE)

    # Load metadata CSVs for contexts that have one
    meta_lookups: Dict[str, Dict[str, str]] = {}  # ctx_name → {video_stem: test_type}
    for ctx_cfg in contexts:
        csv_path = ctx_cfg.get("metadata_csv")
        if csv_path:
            full_path = os.path.join(base_dir, csv_path)
            if os.path.exists(full_path):
                import pandas as _pd
                df = _pd.read_csv(full_path, dtype=str)
                if "test_type" in df.columns and "video_stem" in df.columns:
                    meta_lookups[ctx_cfg["name"]] = {
                        row["video_stem"]: row["test_type"]
                        for _, row in df.iterrows()
                    }

    tt_map: Dict[str, str] = {}
    for sess, ctx in ctx_map.items():
        clean = _SUPER_RE.sub("", sess)
        # Try name-based parsing first (works for surgery_15mice)
        m = _TT_RE.search(clean)
        if m:
            tt_map[sess] = re.sub(r"\s+", " ", m.group(1)).strip()
            continue
        # Try metadata CSV lookup
        if ctx in meta_lookups:
            video_stem = clean.split("_")[0]
            tt = meta_lookups[ctx].get(video_stem, "")
            if tt:
                tt_map[sess] = tt
                continue
        tt_map[sess] = ""
    return tt_map


# ─── Session → context mapping ────────────────────────────────────────────────

def _build_session_context_map(contexts: list, dlc_output_root: str) -> Dict[str, str]:
    """
    Build a mapping {session_stem -> context_name} by scanning each context's
    DLC output folder for H5 files.  Much more reliable than name-string matching.
    """
    mapping: Dict[str, str] = {}
    for ctx in contexts:
        ctx_name = ctx["name"]
        ctx_dir  = os.path.join(dlc_output_root, ctx_name)
        if not os.path.isdir(ctx_dir):
            continue
        for h5_path in Path(ctx_dir).glob("*.h5"):
            # Strip the DLC suffix to recover the original stem used in .npz files
            stem = h5_path.stem
            # Remove the DLC model suffix (everything from '_superanimal' onward)
            clean = re.sub(r"_superanimal.*$", "", stem, flags=re.IGNORECASE)
            mapping[stem]  = ctx_name   # full H5 stem key
            mapping[clean] = ctx_name   # clean stem key (matches .npz names)
    return mapping


def _session_context(session_name: str, contexts: list,
                     ctx_map: Dict[str, str] = None) -> str:
    """Return the context name for a session using the pre-built ctx_map."""
    if ctx_map:
        # Try exact match first, then clean match
        if session_name in ctx_map:
            return ctx_map[session_name]
        clean = re.sub(r"_superanimal.*$", "", session_name, flags=re.IGNORECASE)
        if clean in ctx_map:
            return ctx_map[clean]
    # Fallback: last context
    return contexts[-1]["name"] if contexts else "unknown"


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    config_path: str = "config.yaml",
    filter_contexts: Optional[List[str]] = None,
    joint_only: bool = False,
) -> None:
    """
    Parameters
    ----------
    filter_contexts : list of context names to fit (per-context mode only).
    joint_only      : if True, skip per-context fitting and only run the joint model.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms

    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    preprocessed_dir = resolve_path(base_dir, cfg["paths"]["preprocessed_dir"])
    moseq_dir        = resolve_path(base_dir, cfg["paths"]["moseq_project_dir"])
    kp_cfg           = cfg["keypoints"]
    moseq_cfg        = cfg["moseq"]
    contexts         = cfg.get("contexts", [])

    use_bodyparts: List[str] = kp_cfg["use_bodyparts"]
    model_per_context: bool  = bool(moseq_cfg.get("model_per_context", False))

    os.makedirs(moseq_dir, exist_ok=True)

    logger.info("Loading preprocessed keypoints from %s ...", preprocessed_dir)
    coordinates, confidences, bodyparts = load_all_preprocessed(preprocessed_dir)

    if not coordinates:
        logger.error("No preprocessed files found. Run Stage 2 first.")
        return
    logger.info("Loaded %d sessions.", len(coordinates))

    coordinates, confidences = _filter_to_use_bodyparts(
        coordinates, confidences, bodyparts, use_bodyparts
    )

    dlc_output_root = resolve_path(base_dir, cfg["paths"]["dlc_output_dir"])
    ctx_map = _build_session_context_map(contexts, dlc_output_root)
    logger.info("Built session→context map: %d session stems mapped.", len(ctx_map) // 2)

    # Assign each loaded session to its context
    ctx_sessions: Dict[str, List[str]] = {c["name"]: [] for c in contexts}
    unmatched = []
    for sess in coordinates:
        ctx = _session_context(sess, contexts, ctx_map)
        if ctx == (contexts[-1]["name"] if contexts else "unknown") and sess not in ctx_map:
            unmatched.append(sess)
        ctx_sessions.setdefault(ctx, []).append(sess)
    if unmatched:
        logger.warning("%d sessions fell back to last context: %s",
                       len(unmatched), unmatched[:3])

    # ── Joint model mode (only when explicitly requested or no context filter) ─
    joint_cfg = cfg.get("joint_surgery_model")
    run_joint = joint_only or (joint_cfg is not None and filter_contexts is None)
    if joint_cfg and run_joint:
        joint_contexts   = joint_cfg.get("contexts", [c["name"] for c in contexts])
        exclude_tt       = [t.upper() for t in joint_cfg.get("exclude_test_types", [])]
        joint_model_name = joint_cfg.get("model_name", "model_surgery_joint")
        project_dir      = os.path.join(moseq_dir, f"kpms_project_surgery_joint")
        results_h5       = os.path.join(project_dir, joint_model_name, "results.h5")

        if os.path.exists(results_h5):
            logger.info("Joint model results already exist – skipping.")
        else:
            # Build test-type map for filtering
            # Build a flat session→context map for all loaded sessions
            sess_ctx = {}
            for ctx_name, sess_list in ctx_sessions.items():
                for s in sess_list:
                    sess_ctx[s] = ctx_name

            tt_map = _build_test_type_map(contexts, base_dir, sess_ctx)

            joint_sessions: List[str] = []
            skipped_bft = 0
            for ctx_name in joint_contexts:
                for sess in ctx_sessions.get(ctx_name, []):
                    tt = tt_map.get(sess, "").upper()
                    if exclude_tt and any(ex in tt for ex in exclude_tt):
                        skipped_bft += 1
                        continue
                    joint_sessions.append(sess)

            logger.info(
                "Joint model: %d sessions from %s (excluded %d BFT sessions).",
                len(joint_sessions), joint_contexts, skipped_bft,
            )
            joint_coords = {s: coordinates[s] for s in joint_sessions}
            joint_confs  = {s: confidences[s]  for s in joint_sessions}
            joint_moseq_cfg = dict(moseq_cfg)
            joint_moseq_cfg["model_name"] = joint_model_name
            _fit_one_model(kpms, joint_coords, joint_confs, use_bodyparts,
                           joint_moseq_cfg, project_dir, preprocessed_dir,
                           label="surgery_joint")

    # ── All-data joint model ──────────────────────────────────────────────────
    all_cfg = cfg.get("joint_all_model")
    if all_cfg and run_joint:
        all_contexts   = all_cfg.get("contexts", [c["name"] for c in contexts])
        exclude_tt     = [t.upper() for t in all_cfg.get("exclude_test_types", [])]
        all_model_name = all_cfg.get("model_name", "model_all")
        project_dir    = os.path.join(moseq_dir, "kpms_project_all")
        results_h5     = os.path.join(project_dir, all_model_name, "results.h5")

        # Build session list for all-data model (used for both fresh fit and extend)
        sess_ctx = {s: ctx for ctx, sess_list in ctx_sessions.items() for s in sess_list}
        tt_map   = _build_test_type_map(contexts, base_dir, sess_ctx)

        all_sessions: List[str] = []
        skipped = 0
        for ctx_name in all_contexts:
            for sess in ctx_sessions.get(ctx_name, []):
                tt = tt_map.get(sess, "").upper()
                if exclude_tt and any(ex in tt for ex in exclude_tt):
                    skipped += 1
                    continue
                all_sessions.append(sess)
        logger.info("All-data model: %d sessions from %s (excluded %d BFT).",
                    len(all_sessions), all_contexts, skipped)

        all_coords    = {s: coordinates[s] for s in all_sessions}
        all_confs     = {s: confidences[s]  for s in all_sessions}
        all_moseq_cfg = dict(moseq_cfg)
        all_moseq_cfg["model_name"] = all_model_name

        # Check whether we need to (re-)fit or extend
        total_iters_file = os.path.join(project_dir, all_model_name, "total_full_iters.txt")
        num_full_target  = moseq_cfg.get("num_full_iters", 200)
        done_iters = 0
        if os.path.exists(total_iters_file):
            done_iters = int(Path(total_iters_file).read_text().strip())
        elif os.path.exists(results_h5):
            # Legacy: results exist but no sidecar → assume complete
            done_iters = num_full_target

        if done_iters >= num_full_target:
            logger.info("All-data model fully trained (%d / %d iters) – skipping.",
                        done_iters, num_full_target)
        else:
            _fit_one_model(kpms, all_coords, all_confs, use_bodyparts,
                           all_moseq_cfg, project_dir, preprocessed_dir,
                           label="all_data")

    # ── Per-context or single model mode ─────────────────────────────────────
    if joint_only:
        logger.info("--joint flag set: skipping per-context fitting.")
    elif model_per_context:
        logger.info("model_per_context=True: fitting one model per context.")

        for ctx_name, sess_list in ctx_sessions.items():
            if filter_contexts and ctx_name not in filter_contexts:
                logger.info("Skipping context '%s' (not in --contexts filter).", ctx_name)
                continue
            if not sess_list:
                logger.warning("No sessions for context '%s', skipping.", ctx_name)
                continue
            project_dir = os.path.join(moseq_dir, f"kpms_project_{ctx_name}")
            model_name  = moseq_cfg.get("model_name", "model_v01")
            results_h5  = os.path.join(project_dir, model_name, "results.h5")
            if os.path.exists(results_h5):
                logger.info("=== Context '%s': results already exist – skipping. ===", ctx_name)
                continue
            logger.info("=== Context '%s': %d sessions ===", ctx_name, len(sess_list))
            ctx_coords = {s: coordinates[s] for s in sess_list}
            ctx_confs  = {s: confidences[s]  for s in sess_list}
            _fit_one_model(kpms, ctx_coords, ctx_confs, use_bodyparts,
                           moseq_cfg, project_dir, preprocessed_dir, label=ctx_name)
    elif not joint_cfg:
        # ── Fit a single joint model (original behaviour) ─────────────────────
        project_dir = os.path.join(moseq_dir, "kpms_project")
        _fit_one_model(kpms, coordinates, confidences, use_bodyparts,
                       moseq_cfg, project_dir, preprocessed_dir)

    logger.info("Stage 3 complete.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Stage 3: keypoint-MoSeq fitting")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--contexts", nargs="+", default=None,
        help="Only fit models for these context names (per-context mode only).",
    )
    parser.add_argument(
        "--joint", action="store_true",
        help="Only run the joint_surgery_model (skip per-context fitting).",
    )
    args = parser.parse_args()
    run(args.config, filter_contexts=args.contexts, joint_only=args.joint)
