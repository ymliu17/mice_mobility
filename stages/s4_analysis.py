"""
Stage 4 – Syllable analysis and context-aware mobility signatures.

Reads keypoint-moseq results and computes:

  Per session
  -----------
  syllable_labels.csv     – frame-by-frame syllable index
  usage_stats.csv         – fraction of time in each syllable
  duration_stats.csv      – dwell-time statistics per syllable
  transition_matrix.csv   – P(j | i) transition probabilities
  sequence_entropy.csv    – rolling-window entropy + mean dwell time per frame

  Across contexts
  ---------------
  context_usage_comparison.csv  – syllable usage per context (mean ± SEM)
  context_duration_comparison.csv
  context_entropy_summary.csv

  Figures (saved to outputs/04_analysis/figures/)
  -------
  transition_heatmap_<context>.png
  usage_barplot.png

Usage:
  python pipeline.py run --stages 4
  python stages/s4_analysis.py --config config.yaml
"""

from __future__ import annotations

import os
import json
import logging
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.io_utils import load_config, resolve_path, save_json, save_dataframe, load_fps_per_session

logger = logging.getLogger(__name__)


# ─── Result loading ───────────────────────────────────────────────────────────

def load_moseq_results(project_dir: str, model_name: str) -> Dict[str, dict]:
    """
    Load per-session results from the keypoint-moseq output CSVs.

    Each CSV has columns: frame_idx, syllable[, centroid_x, centroid_y, heading, ...]

    Returns
    -------
    results : {session_name: {syllable: np.ndarray(T,), ...}}
    """
    results_dir = os.path.join(project_dir, model_name)
    results: Dict[str, dict] = {}

    csv_files = sorted(Path(results_dir).glob("*.csv"))
    if not csv_files:
        # Fall back to the combined H5 if individual CSVs are absent
        h5_path = os.path.join(results_dir, f"results_{model_name}.h5")
        if os.path.exists(h5_path):
            import h5py
            with h5py.File(h5_path, "r") as f:
                for session in f.keys():
                    results[session] = {
                        k: f[session][k][:] for k in f[session].keys()
                    }
            return results
        raise FileNotFoundError(
            f"No results found in {results_dir}. Run Stage 3 first."
        )

    for csv_path in csv_files:
        session = csv_path.stem
        df = pd.read_csv(csv_path)
        results[session] = {col: df[col].values for col in df.columns}

    return results


# ─── Per-session syllable statistics ─────────────────────────────────────────

def compute_usage_stats(
    syllable_seq: np.ndarray,
    num_states: int = 100,
    min_usage: float = 0.005,
) -> pd.DataFrame:
    """
    Fraction of frames spent in each syllable.

    Returns DataFrame with columns [syllable, count, usage_fraction].
    Only syllables with usage ≥ min_usage are retained.
    """
    syllable_seq = syllable_seq.astype(int)
    counts = np.bincount(syllable_seq[syllable_seq >= 0], minlength=num_states)
    total = counts.sum()
    df = pd.DataFrame({
        "syllable": np.arange(num_states),
        "count": counts,
        "usage_fraction": counts / total if total > 0 else counts.astype(float),
    })
    df = df[df["usage_fraction"] >= min_usage].reset_index(drop=True)
    return df.sort_values("usage_fraction", ascending=False).reset_index(drop=True)


def compute_duration_stats(syllable_seq: np.ndarray) -> pd.DataFrame:
    """
    Per-syllable dwell-time statistics (in frames).

    Returns DataFrame with columns:
      [syllable, n_bouts, mean_dur, std_dur, median_dur, min_dur, max_dur]
    """
    syllable_seq = syllable_seq.astype(int)
    bouts: Dict[int, List[int]] = defaultdict(list)

    i = 0
    while i < len(syllable_seq):
        s = syllable_seq[i]
        j = i + 1
        while j < len(syllable_seq) and syllable_seq[j] == s:
            j += 1
        bouts[s].append(j - i)
        i = j

    rows = []
    for syllable, durations in sorted(bouts.items()):
        d = np.array(durations)
        rows.append({
            "syllable": syllable,
            "n_bouts": len(d),
            "mean_dur": float(np.mean(d)),
            "std_dur": float(np.std(d)),
            "median_dur": float(np.median(d)),
            "min_dur": int(np.min(d)),
            "max_dur": int(np.max(d)),
        })
    return pd.DataFrame(rows)


def compute_transition_matrix(
    syllable_seq: np.ndarray,
    num_states: int = 100,
    smooth: float = 1e-4,
) -> np.ndarray:
    """
    Compute the (num_states × num_states) row-normalised transition matrix.

    T[i, j] = P(j | current = i).
    ``smooth`` is added to each cell before normalisation (Laplace smoothing).
    """
    syllable_seq = syllable_seq.astype(int)
    T = np.full((num_states, num_states), smooth)

    for t in range(len(syllable_seq) - 1):
        s_cur = syllable_seq[t]
        s_nxt = syllable_seq[t + 1]
        if 0 <= s_cur < num_states and 0 <= s_nxt < num_states:
            T[s_cur, s_nxt] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T /= row_sums
    return T


def compute_sequence_entropy(
    syllable_seq: np.ndarray,
    window: int = 50,
    num_states: int = 100,
) -> np.ndarray:
    """
    Rolling Shannon entropy over a window of frames.

    H_t = -Σ p_s * log2(p_s)  where p_s is the empirical syllable distribution
    in the window ending at frame t.

    Returns array of shape (T,) with NaN for frames where the window is partial.
    """
    syllable_seq = syllable_seq.astype(int)
    T = len(syllable_seq)
    entropy = np.full(T, np.nan)

    for t in range(window - 1, T):
        window_slice = syllable_seq[t - window + 1 : t + 1]
        counts = np.bincount(window_slice[window_slice >= 0], minlength=num_states)
        probs = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
        # Avoid log(0)
        probs_nonzero = probs[probs > 0]
        entropy[t] = -float(np.sum(probs_nonzero * np.log2(probs_nonzero)))

    return entropy


def compute_mean_dwell_per_frame(syllable_seq: np.ndarray) -> np.ndarray:
    """
    For each frame, look up the mean dwell time of the syllable at that frame.

    This allows per-frame feature extraction without re-running duration stats.
    """
    dur_df = compute_duration_stats(syllable_seq)
    mean_dwell_lookup = dict(zip(dur_df["syllable"], dur_df["mean_dur"]))
    return np.array([mean_dwell_lookup.get(int(s), np.nan) for s in syllable_seq])


# ─── Context-level comparisons ────────────────────────────────────────────────

def assign_context(
    session_name: str,
    context_names: List[str],
) -> str:
    """
    Heuristically assign a context label from the session name.

    Matches the first context name whose string appears (case-insensitive) in
    the session name.  Falls back to 'unknown'.
    """
    lower = session_name.lower()
    for ctx in context_names:
        if ctx.lower() in lower:
            return ctx
    return "unknown"


def aggregate_by_context(
    per_session_usage: Dict[str, pd.DataFrame],
    context_names: List[str],
) -> pd.DataFrame:
    """
    Return mean ± SEM of syllable usage fraction, grouped by context.

    Returns DataFrame with columns:
      [context, syllable, mean_usage, sem_usage, n_sessions]
    """
    rows = []
    # Collect all (context, syllable, usage) triples
    context_data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for session, usage_df in per_session_usage.items():
        ctx = assign_context(session, context_names)
        for _, row in usage_df.iterrows():
            context_data[ctx][int(row["syllable"])].append(float(row["usage_fraction"]))

    for ctx, syll_dict in context_data.items():
        for syllable, values in syll_dict.items():
            arr = np.array(values)
            rows.append({
                "context": ctx,
                "syllable": syllable,
                "mean_usage": float(np.mean(arr)),
                "sem_usage": float(np.std(arr) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "n_sessions": len(arr),
            })

    return pd.DataFrame(rows).sort_values(["context", "mean_usage"], ascending=[True, False])


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_transition_heatmap(
    T_mat: np.ndarray,
    active_syllables: List[int],
    title: str,
    out_path: str,
) -> None:
    """Plot a heatmap of the transition probability sub-matrix for active syllables."""
    sub = T_mat[np.ix_(active_syllables, active_syllables)]
    labels = [str(s) for s in active_syllables]

    fig, ax = plt.subplots(figsize=(
        min(20, max(6, len(active_syllables) * 0.45)),
        min(18, max(5, len(active_syllables) * 0.4)),
    ))
    sns.heatmap(
        sub,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        vmin=0,
        vmax=sub.max(),
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_xlabel("Next syllable")
    ax.set_ylabel("Current syllable")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_usage_barplot(
    context_usage_df: pd.DataFrame,
    top_n: int = 20,
    out_path: str = "usage_barplot.png",
) -> None:
    """Grouped bar chart of top-N syllable usages across contexts."""
    # Find globally top-N syllables by mean usage
    top_sylls = (
        context_usage_df.groupby("syllable")["mean_usage"]
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )
    plot_df = context_usage_df[context_usage_df["syllable"].isin(top_sylls)].copy()
    plot_df["syllable"] = plot_df["syllable"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    contexts = plot_df["context"].unique()
    x = np.arange(len(top_sylls))
    width = 0.8 / max(len(contexts), 1)

    for i, ctx in enumerate(sorted(contexts)):
        ctx_df = plot_df[plot_df["context"] == ctx].set_index("syllable")
        heights = [ctx_df.loc[str(s), "mean_usage"] if str(s) in ctx_df.index else 0
                   for s in top_sylls]
        yerrs = [ctx_df.loc[str(s), "sem_usage"] if str(s) in ctx_df.index else 0
                 for s in top_sylls]
        ax.bar(x + i * width, heights, width, label=ctx, yerr=yerrs, capsize=3)

    ax.set_xticks(x + width * (len(contexts) - 1) / 2)
    ax.set_xticklabels([str(s) for s in top_sylls], rotation=45)
    ax.set_ylabel("Mean usage fraction")
    ax.set_xlabel("Syllable index")
    ax.set_title(f"Top-{top_n} syllable usage by context")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    base_dir = str(Path(config_path).parent.resolve())

    moseq_dir = resolve_path(base_dir, cfg["paths"]["moseq_project_dir"])
    analysis_dir = resolve_path(base_dir, cfg["paths"]["analysis_dir"])
    model_name: str = cfg["moseq"]["model_name"]
    num_states: int = cfg["moseq"].get("num_states", 100)
    min_usage: float = cfg["analysis"]["min_usage_fraction"]
    entropy_window: int = cfg["analysis"]["entropy_window"]
    top_n: int = cfg["analysis"]["top_n_syllables"]
    context_names: List[str] = [c["name"] for c in cfg["contexts"]]

    kpms_project_dir = os.path.join(moseq_dir, "kpms_project")
    figures_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────────
    logger.info("Loading MoSeq results from %s ...", kpms_project_dir)
    results = load_moseq_results(kpms_project_dir, model_name)
    logger.info("Loaded results for %d sessions.", len(results))

    preprocessed_dir = resolve_path(base_dir, cfg["paths"]["preprocessed_dir"])
    fps_per_session = load_fps_per_session(preprocessed_dir)

    per_session_usage: Dict[str, pd.DataFrame] = {}
    per_session_duration: Dict[str, pd.DataFrame] = {}
    per_session_trans: Dict[str, np.ndarray] = {}
    per_session_entropy: Dict[str, np.ndarray] = {}

    session_summary_rows = []

    for session, res in results.items():
        syllable_seq = np.asarray(res["syllable"]).astype(int)
        fps = fps_per_session.get(session, 30.0)
        logger.info("  Session '%s': %d frames @ %.1f fps.", session, len(syllable_seq), fps)

        # --- syllable labels per frame ---
        labels_df = pd.DataFrame({"frame": np.arange(len(syllable_seq)),
                                   "syllable": syllable_seq})
        save_dataframe(labels_df, os.path.join(analysis_dir, "syllable_labels",
                                                f"{session}.csv"))

        # --- usage stats ---
        usage_df = compute_usage_stats(syllable_seq, num_states, min_usage)
        save_dataframe(usage_df, os.path.join(analysis_dir, "usage_stats",
                                               f"{session}.csv"))
        per_session_usage[session] = usage_df

        # --- duration stats (frames + seconds) ---
        dur_df = compute_duration_stats(syllable_seq)
        for col in ("mean_dur", "std_dur", "median_dur", "min_dur", "max_dur"):
            dur_df[col.replace("dur", "dur_s")] = dur_df[col] / fps
        save_dataframe(dur_df, os.path.join(analysis_dir, "duration_stats",
                                             f"{session}.csv"))
        per_session_duration[session] = dur_df

        # --- transition matrix ---
        trans_mat = compute_transition_matrix(syllable_seq, num_states)
        per_session_trans[session] = trans_mat
        # Save as CSV (flat: from_syllable, to_syllable, probability)
        active = usage_df["syllable"].astype(int).tolist()
        trans_rows = []
        for si in active:
            for sj in active:
                trans_rows.append({
                    "from_syllable": si,
                    "to_syllable": sj,
                    "probability": float(trans_mat[si, sj]),
                })
        save_dataframe(
            pd.DataFrame(trans_rows),
            os.path.join(analysis_dir, "transition_matrices", f"{session}.csv"),
        )

        # Plot heatmap for this session
        plot_transition_heatmap(
            trans_mat,
            active_syllables=active[:top_n],
            title=f"Transition matrix – {session}",
            out_path=os.path.join(figures_dir, f"transition_{session}.png"),
        )

        # --- sequence entropy ---
        entropy_arr = compute_sequence_entropy(syllable_seq, window=entropy_window,
                                               num_states=num_states)
        mean_dwell_arr = compute_mean_dwell_per_frame(syllable_seq)
        ent_df = pd.DataFrame({
            "frame": np.arange(len(syllable_seq)),
            "syllable": syllable_seq,
            "sequence_entropy": entropy_arr,
            "mean_dwell_of_current_syllable": mean_dwell_arr,
        })
        save_dataframe(ent_df, os.path.join(analysis_dir, "sequence_entropy",
                                             f"{session}.csv"))
        per_session_entropy[session] = entropy_arr

        # Summary row for this session
        valid_ent = entropy_arr[~np.isnan(entropy_arr)]
        mean_bout_frames = float(dur_df["mean_dur"].mean())
        session_summary_rows.append({
            "session": session,
            "context": assign_context(session, context_names),
            "effective_fps": fps,
            "n_frames": len(syllable_seq),
            "duration_s": len(syllable_seq) / fps,
            "n_active_syllables": len(usage_df),
            "mean_entropy": float(np.mean(valid_ent)) if len(valid_ent) else np.nan,
            "std_entropy": float(np.std(valid_ent)) if len(valid_ent) else np.nan,
            "mean_bout_duration": mean_bout_frames,
            "mean_bout_duration_s": mean_bout_frames / fps,
        })

    # ── Context-level aggregation ─────────────────────────────────────────────
    logger.info("Aggregating across contexts ...")

    # Session summary
    session_summary_df = pd.DataFrame(session_summary_rows)
    save_dataframe(session_summary_df, os.path.join(analysis_dir, "session_summary.csv"))

    # Context usage comparison
    ctx_usage_df = aggregate_by_context(per_session_usage, context_names)
    save_dataframe(ctx_usage_df, os.path.join(analysis_dir, "context_usage_comparison.csv"))

    # Grouped usage bar plot
    if not ctx_usage_df.empty:
        plot_usage_barplot(
            ctx_usage_df,
            top_n=top_n,
            out_path=os.path.join(figures_dir, "usage_barplot.png"),
        )

    # Context duration comparison
    dur_rows = []
    for session, dur_df in per_session_duration.items():
        ctx = assign_context(session, context_names)
        fps = fps_per_session.get(session, 30.0)
        for _, row in dur_df.iterrows():
            dur_rows.append({"context": ctx, "session": session, "effective_fps": fps, **row.to_dict()})
    ctx_dur_df = pd.DataFrame(dur_rows)
    save_dataframe(ctx_dur_df, os.path.join(analysis_dir, "context_duration_comparison.csv"))

    # Entropy summary per context
    ent_rows = []
    for session, ent_arr in per_session_entropy.items():
        ctx = assign_context(session, context_names)
        valid = ent_arr[~np.isnan(ent_arr)]
        ent_rows.append({
            "context": ctx,
            "session": session,
            "mean_entropy": float(np.mean(valid)) if len(valid) else np.nan,
            "std_entropy": float(np.std(valid)) if len(valid) else np.nan,
        })
    ctx_ent_df = pd.DataFrame(ent_rows)
    save_dataframe(ctx_ent_df, os.path.join(analysis_dir, "context_entropy_summary.csv"))

    # Persist a JSON summary of active syllable counts per context
    context_syllable_counts = (
        ctx_usage_df.groupby("context")["syllable"].nunique().to_dict()
    )
    save_json(
        {
            "n_sessions": len(results),
            "n_active_syllables_global": int(
                ctx_usage_df["syllable"].nunique()
            ),
            "active_syllables_per_context": context_syllable_counts,
        },
        os.path.join(analysis_dir, "pipeline_summary.json"),
    )

    logger.info("Stage 4 complete.  Outputs: %s", analysis_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Stage 4: Syllable analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
