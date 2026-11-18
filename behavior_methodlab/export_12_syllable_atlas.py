#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages import s5_posthoc_analysis as s5
from stages.s4_classification import build_session_feature_bank
from utils.io_utils import load_config, load_all_preprocessed


OUTDIR = Path("outputs/05_stage5/syllable_atlas_presence_gt10")
SYLLABLES = np.array([32, 11, 26, 37, 45, 43, 46, 0, 7, 24, 4, 1], dtype=int)


def _save_overview(atlas_df: pd.DataFrame, outdir: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5.6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(atlas_df)))
    ax.bar([str(int(s)) for s in atlas_df["syllable"]], atlas_df["usage_mean"], color=colors)
    ax.set_title(f"{label} syllables in the shared model")
    ax.set_xlabel("Syllable")
    ax.set_ylabel("Mean usage")
    for i, row in atlas_df.reset_index(drop=True).iterrows():
        ax.text(
            i,
            float(row["usage_mean"]) + max(float(atlas_df["usage_mean"].max()) * 0.01, 0.001),
            str(row["prototype_label"]),
            rotation=45,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(outdir / "presence_gt10_overview.png", dpi=170)
    plt.close(fig)


def _plot_cards(
    cfg: dict,
    seqs: dict[str, np.ndarray],
    meta_df: pd.DataFrame,
    syllable_features: pd.DataFrame,
    syllable_ids: np.ndarray,
    outdir: Path,
) -> None:
    coords_all, _, bodyparts = load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
    old_atlas_dir = s5.ATLAS_DIR
    try:
        s5.ATLAS_DIR = outdir
        for syllable in syllable_ids:
            s5._plot_mean_pose_card(
                int(syllable),
                seqs,
                coords_all,
                bodyparts,
                syllable_features,
                meta_df,
                syllable_ids,
            )
    finally:
        s5.ATLAS_DIR = old_atlas_dir


def _plot_heatmap(
    syllable_features: pd.DataFrame,
    syllable_ids: np.ndarray,
    outdir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atlas_df = syllable_features[syllable_features["syllable"].isin(syllable_ids)].copy()
    if atlas_df.empty:
        return
    cols = ["speed_z", "turn_z", "bout_z", "pause_z", "align_z", "entropy_z", "body_len_z"]
    label_map = {
        "speed_z": "speed",
        "turn_z": "turn",
        "bout_z": "bout",
        "pause_z": "pause",
        "align_z": "forward",
        "entropy_z": "entropy",
        "body_len_z": "body_len",
    }
    plot_df = atlas_df.set_index("syllable")[cols].rename(columns=label_map)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    im = ax.imshow(plot_df.to_numpy().T, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(plot_df.index)))
    ax.set_xticklabels([str(int(s)) for s in plot_df.index])
    ax.set_yticks(range(len(plot_df.columns)))
    ax.set_yticklabels(plot_df.columns)
    ax.set_xlabel("Presence >10% syllable")
    ax.set_title("Mean kinematics per 12 selected syllables (z-scored)")
    for i in range(plot_df.shape[1]):
        for j in range(plot_df.shape[0]):
            ax.text(j, i, f"{plot_df.iloc[j, i]:.1f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outdir / "presence_gt10_mean_kinematics_heatmap.png", dpi=170)
    plt.close(fig)


def _generate_trajectories(cfg: dict, syllable_ids: np.ndarray, outdir: Path) -> None:
    old_atlas_dir = s5.ATLAS_DIR
    try:
        s5.ATLAS_DIR = outdir
        s5._generate_kpms_style_trajectory_plots(cfg, syllable_ids)
        traj_dir = outdir / "kpms_style_trajectories"
        renames = {
            "freq50_trajectory_strip.pdf": "presence_gt10_trajectory_strip.pdf",
            "freq50_trajectory_strip.png": "presence_gt10_trajectory_strip.png",
        }
        for src_name, dst_name in renames.items():
            src = traj_dir / src_name
            dst = traj_dir / dst_name
            if src.exists():
                src.replace(dst)
    finally:
        s5.ATLAS_DIR = old_atlas_dir


def _run_native(cfg: dict, meta_df: pd.DataFrame, syllable_ids: np.ndarray, outdir: Path) -> None:
    s5._prepare_third_party_runtime()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import keypoint_moseq as kpms
    from keypoint_moseq.analysis import compute_moseq_df, compute_stats_df, plot_syll_stats_with_sem
    from keypoint_moseq.viz import (
        generate_grid_movies,
        plot_duration_distribution,
        plot_similarity_dendrogram,
        plot_syllable_frequencies,
    )

    old_atlas_dir = s5.ATLAS_DIR
    try:
        s5.ATLAS_DIR = outdir
        native_dir = outdir / "native_kpms"
        native_dir.mkdir(parents=True, exist_ok=True)
        project_dir, model_name = s5._prepare_native_kpms_project(meta_df)

        allowed_sessions = set(meta_df["session"].astype(str).tolist())
        results_all = kpms.load_results(path=s5.ALL_H5)
        results = {k: v for k, v in results_all.items() if k in allowed_sessions}

        fps_map = s5.load_fps_per_session(cfg["paths"]["preprocessed_dir"], default_fps=10.0)
        median_fps = int(round(float(np.nanmedian(list(fps_map.values()))))) if fps_map else 10

        fig, _ = plot_syllable_frequencies(path=s5.ALL_H5, min_frequency=0.0)
        s5._save_fig(fig, native_dir / "syllable_frequency_distribution.png")
        fig, _ = plot_duration_distribution(path=s5.ALL_H5, fps=median_fps)
        s5._save_fig(fig, native_dir / "duration_distribution.png")

        moseq_df = compute_moseq_df(str(project_dir), model_name, results_dict=results, fps=median_fps)
        stats_df = compute_stats_df(
            str(project_dir),
            model_name,
            moseq_df,
            min_frequency=0.0,
            groupby=["group", "name"],
            fps=median_fps,
        )
        stats_df = stats_df[stats_df["syllable"].isin(syllable_ids)].copy()
        stats_df.to_csv(native_dir / "presence_gt10_native_stats.csv", index=False)
        moseq_df.to_csv(native_dir / "moseq_frame_table.csv", index=False)

        if not stats_df.empty and stats_df["group"].nunique() > 1:
            fig, _ = plot_syll_stats_with_sem(
                stats_df,
                str(project_dir),
                model_name,
                save_dir=str(native_dir / "cohort_frequency_stats"),
                plot_sig=False,
                stat="frequency",
                order="stat",
                groups=sorted(stats_df["group"].dropna().unique()),
                figsize=(10, 4),
            )
            plt.close(fig)
            fig, _ = plot_syll_stats_with_sem(
                stats_df,
                str(project_dir),
                model_name,
                save_dir=str(native_dir / "cohort_velocity_stats"),
                plot_sig=False,
                stat="velocity_px_s_mean",
                order="stat",
                groups=sorted(stats_df["group"].dropna().unique()),
                figsize=(10, 4),
            )
            plt.close(fig)

        coords_all, _, bodyparts = s5.load_all_preprocessed(cfg["paths"]["preprocessed_dir"])
        use_bodyparts = cfg["keypoints"]["use_bodyparts"]
        use_idx = [bodyparts.index(bp) for bp in use_bodyparts]
        coordinates = {k: v[:, use_idx, :] for k, v in coords_all.items() if k in results}
        centroids = {k: results[k]["centroid"] for k in coordinates if "centroid" in results[k]}
        headings = {k: results[k]["heading"] for k in coordinates if "heading" in results[k]}
        skeleton = s5._build_skeleton(use_bodyparts)

        min_freq_for_movies = 0.001
        try:
            atlas_df = pd.read_csv(outdir / "presence_gt10_syllable_atlas.csv")
            if not atlas_df.empty:
                min_freq_for_movies = float(max(0.001, atlas_df["usage_mean"].min() * 0.98))
        except Exception:
            pass

        grid_dir = native_dir / "grid_movies"
        grid_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_grid_movies(
                results=results,
                output_dir=str(grid_dir),
                coordinates=coordinates,
                centroids=centroids,
                headings=headings,
                bodyparts=use_bodyparts,
                use_bodyparts=use_bodyparts,
                skeleton=skeleton,
                keypoints_only=True,
                overlay_keypoints=False,
                rows=2,
                cols=3,
                pre=0.25,
                post=0.5,
                min_frequency=min_freq_for_movies,
                min_duration=3,
                fps=median_fps,
                quality=7,
                sampling_options={"n_neighbors": 50},
            )
            keep = {f"syllable{int(s)}.mp4" for s in syllable_ids}
            for movie_path in grid_dir.glob("*.mp4"):
                if movie_path.name not in keep:
                    movie_path.unlink()
        except Exception as exc:
            s5.logger.warning("12-syllable native kpms grid movie export failed: %s", exc)

        try:
            plot_similarity_dendrogram(
                coordinates=coordinates,
                results=results,
                save_path=str(native_dir / "similarity_dendrogram"),
                bodyparts=use_bodyparts,
                use_bodyparts=use_bodyparts,
                pre=0.2,
                post=0.6,
                min_frequency=min_freq_for_movies,
                fps=median_fps,
                density_sample=True,
                sampling_options={"n_neighbors": 50},
            )
        except Exception as exc:
            s5.logger.warning("12-syllable similarity dendrogram export failed: %s", exc)
    finally:
        s5.ATLAS_DIR = old_atlas_dir


def run(config_path: str = "config.yaml") -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    seqs = s5._load_sequences()
    meta_df = s5._load_session_metadata(list(seqs.keys()))
    feature_bank = build_session_feature_bank()
    _, syllable_features, _ = s5._build_feature_caches(seqs, feature_bank, meta_df, cfg)

    atlas_df = (
        syllable_features[syllable_features["syllable"].isin(SYLLABLES)]
        .copy()
        .sort_values("usage_mean", ascending=False)
        .reset_index(drop=True)
    )
    atlas_df.to_csv(OUTDIR / "presence_gt10_syllable_atlas.csv", index=False)
    # compatibility for native helper threshold logic
    atlas_df.to_csv(OUTDIR / "freq50_syllable_atlas.csv", index=False)

    _save_overview(atlas_df, OUTDIR, label="Twelve presence>10%")
    _plot_cards(cfg, seqs, meta_df, syllable_features, SYLLABLES, OUTDIR)
    _plot_heatmap(syllable_features, SYLLABLES, OUTDIR)
    _generate_trajectories(cfg, SYLLABLES, OUTDIR)
    try:
        _run_native(cfg, meta_df, SYLLABLES, OUTDIR)
    except Exception as exc:
        print(f"[warn] _run_native failed (non-critical): {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete 12-syllable atlas parallel to the Stage 5 5-syllable atlas.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
