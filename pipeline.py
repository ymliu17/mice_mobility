#!/usr/bin/env python3
"""
Mobility Pipeline – main CLI entry point.

Commands
--------
run     Run one or more pipeline stages.
qc      Generate a QC report from preprocessed data.
check   Validate config and data paths without running anything.

Examples
--------
# Full pipeline
python pipeline.py run

# Individual stages
python pipeline.py run --stages 1
python pipeline.py run --stages 2 3
python pipeline.py run --stages 1 2 3 4

# QC after Stage 2
python pipeline.py qc

# Validate paths
python pipeline.py check

# Custom config
python pipeline.py run --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quieten noisy third-party loggers
    for name in ("matplotlib", "PIL", "jax", "absl"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ─── Stage dispatch ───────────────────────────────────────────────────────────

_STAGE_MAP = {
    1: ("stages.s1_pose_extraction",  "Stage 1: DLC SuperAnimal pose extraction"),
    2: ("stages.s2_preprocessing",    "Stage 2: Keypoint preprocessing"),
    3: ("stages.s3_moseq_fitting",    "Stage 3: keypoint-MoSeq model fitting"),
    4: ("stages.s4_classification",   "Stage 4: state-moseq fit + syllable-based classification"),
    5: ("stages.s5_posthoc_analysis", "Stage 5: post-hoc behavioural atlas and robustness"),
    6: ("stages.s6_fiber_joint",      "Stage 6: fiber photometry x kinematics joint analysis"),
}


def run_stages(stages: List[int], config_path: str, filter_contexts: List[str] | None = None) -> None:
    import importlib

    for stage_id in sorted(stages):
        if stage_id not in _STAGE_MAP:
            raise ValueError(f"Unknown stage id: {stage_id}. Must be 1–6.")

        module_path, description = _STAGE_MAP[stage_id]
        logger = logging.getLogger("pipeline")
        logger.info("=" * 60)
        logger.info("  %s", description)
        logger.info("=" * 60)

        module = importlib.import_module(module_path)
        if stage_id in (1, 2) and filter_contexts:
            module.run(config_path=config_path, filter_contexts=filter_contexts)
        else:
            module.run(config_path=config_path)

    logging.getLogger("pipeline").info("All requested stages complete.")


# ─── QC command ───────────────────────────────────────────────────────────────

def run_qc(config_path: str) -> None:
    """Generate QC figures from Stage 2 preprocessed data."""
    from utils.io_utils import load_config, resolve_path, load_all_preprocessed
    from utils.qc import (
        confidence_report,
        missing_data_report,
        body_length_report,
        flag_low_quality_sessions,
        plot_confidence_heatmap,
        plot_nan_heatmap,
    )

    logger = logging.getLogger("pipeline.qc")
    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    preprocessed_dir = resolve_path(base_dir, cfg["paths"]["preprocessed_dir"])
    analysis_dir = resolve_path(base_dir, cfg["paths"]["analysis_dir"])
    os.makedirs(analysis_dir, exist_ok=True)

    logger.info("Loading preprocessed data from %s ...", preprocessed_dir)
    coordinates, confidences, bodyparts = load_all_preprocessed(preprocessed_dir)

    if not coordinates:
        logger.error("No preprocessed files found. Run Stage 2 first.")
        return

    conf_df = confidence_report(
        confidences, bodyparts, threshold=cfg["preprocessing"]["conf_threshold"]
    )
    nan_df = missing_data_report(coordinates, bodyparts)
    bl_df = body_length_report(
        coordinates,
        bodyparts,
        body_length_parts=tuple(cfg["keypoints"]["body_length_parts"]),
    )

    conf_df.to_csv(os.path.join(analysis_dir, "qc_confidence.csv"), index=False)
    nan_df.to_csv(os.path.join(analysis_dir, "qc_nan.csv"), index=False)
    bl_df.to_csv(os.path.join(analysis_dir, "qc_body_length.csv"), index=False)

    plot_confidence_heatmap(
        conf_df,
        os.path.join(analysis_dir, "qc_confidence_heatmap.png"),
        threshold=cfg["preprocessing"]["conf_threshold"],
    )
    plot_nan_heatmap(nan_df, os.path.join(analysis_dir, "qc_nan_heatmap.png"))

    flagged = flag_low_quality_sessions(conf_df, nan_df)
    if flagged:
        logger.warning("LOW-QUALITY sessions (check QC heatmaps): %s", flagged)
        print("\nFlagged sessions:")
        for s in flagged:
            print(f"  {s}")
    else:
        logger.info("All sessions passed QC thresholds.")

    print(f"\nBody-length summary:\n{bl_df.to_string(index=False)}")
    logger.info("QC reports saved to %s", analysis_dir)


# ─── Check command ────────────────────────────────────────────────────────────

def run_check(config_path: str) -> None:
    """Validate that all configured video paths exist."""
    from utils.io_utils import load_config, collect_all_videos

    logger = logging.getLogger("pipeline.check")
    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    all_videos = collect_all_videos(cfg, base_dir)
    total = 0
    all_ok = True

    for ctx, videos in all_videos.items():
        if not videos:
            logger.warning("[%s] No videos found!", ctx)
            all_ok = False
        else:
            logger.info("[%s] %d video(s) found.", ctx, len(videos))
            total += len(videos)
            for v in videos:
                if not os.path.isfile(v):
                    logger.error("  Missing: %s", v)
                    all_ok = False

    print(f"\nTotal videos found: {total}")
    if all_ok:
        print("Config check PASSED.")
    else:
        print("Config check FAILED. Review warnings above.")
        sys.exit(1)


# ─── CLI parser ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Mobility Analysis Pipeline – DLC + keypoint-MoSeq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the pipeline config file (default: config.yaml).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    run_p = subparsers.add_parser("run", help="Run pipeline stages.")
    run_p.add_argument(
        "--stages", nargs="*", type=int, default=[1, 2, 3, 4],
        metavar="N",
        help="Stage numbers to run (1–6). Default: stages 1–4.",
    )
    run_p.add_argument(
        "--contexts", nargs="+", metavar="CTX",
        help="Limit stages 1 & 2 to these context names (default: all).",
    )

    # ── qc ──
    subparsers.add_parser("qc", help="Generate QC report from Stage 2 output.")

    # ── check ──
    subparsers.add_parser("check", help="Validate config and data paths.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    config_path = os.path.abspath(args.config)

    if args.command == "run":
        run_stages(args.stages, config_path, filter_contexts=getattr(args, "contexts", None))
    elif args.command == "qc":
        run_qc(config_path)
    elif args.command == "check":
        run_check(config_path)


if __name__ == "__main__":
    main()
