"""
Stage 4 – Syllable-based behaviour classification.
===================================================
Prerequisites (run automatically if cache is absent):
  - state-moseq HHMM fit on model_all syllable sequences

Two canonical feature sets (selected via benchmark in analyses/benchmark_improvements.py):

  freq50+trans                  – usage + 5×5 transition submatrix (p=30)
                                  wins for open-field / Y-maze behaviour groups

  freq50+transsummary+metastate – usage + per-syllable self-trans/entropy
                                  + meta-state usage (p=25)
                                  wins for fiber-photometry and surgery recovery groups

Each group uses a fixed representation (see GROUP_REPRESENTATION).

Outputs saved to: outputs/04_classification/

Usage:
    python pipeline.py run --stages 4
    python stages/s4_classification.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"Features .* are constant\.", category=UserWarning)
warnings.filterwarnings("ignore", message=r"invalid value encountered in divide",
                        category=RuntimeWarning)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TAKAYA_META_CSV = "Data/sugery_Takaya_Gen/video_metadata.csv"
GI_META_CSV     = "Data/Gi mice OFT/video_metadata.csv"
ALL_H5          = "outputs/03_moseq/kpms_project_all/model_all/results.h5"
STATE_MOSEQ_K            = 10
STATE_MOSEQ_TRANS_KAPPA  = 10.0
STATE_MOSEQ_EBS          = 0.5
STATE_MOSEQ_NUM_ITERS    = 200
STATE_MOSEQ_H5           = (
    f"outputs/03_moseq/kpms_project_all/model_all/"
    f"state_moseq_K{STATE_MOSEQ_K}.h5"
)
NUM_STATES          = 50
LAPLACE_PSEUDOCOUNT = 0.1
OUT_DIR = Path("outputs/04_classification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Canonical feature sets ────────────────────────────────────────────────────
# Selected by benchmark in analyses/benchmark_improvements.py.
# Format: '+'-joined components with freq50 always as the base.
#   freq50        – usage of the 5 syllables present in ≥50 % of sessions (p=5)
#   trans         – 5×5 Laplace-smoothed transition submatrix              (p=25)
#   transsummary  – per-syllable self-trans prob + row entropy               (p=10)
#   metastate     – state-moseq meta-state usage vector                     (p=10)

FEATURE_SETS = [
    "freq50+trans",                   # p=30; behaviour groups
    "freq50+transsummary+metastate",  # p=25; fiber / surgery-recovery groups
]

# Fixed representation per group (same dataset → same representation).
# raw           – each session is one sample
# mouse_mean    – average all sessions per mouse
# mouse_centered– subtract each mouse's own mean (deviation from baseline)
# delta_prepost – post − pre per mouse (surgery-induced change)
GROUP_REPRESENTATION: dict[str, str] = {
    "AST":          "raw",
    "Food_intake":  "raw",
    "OFT_GI":       "mouse_mean",
    "OFT_single":   "mouse_mean",
    "Y_maze":       "mouse_mean",
    "OFT_s15":      "mouse_centered",
    "Ymaze_s15":    "mouse_centered",
    "OFT_Takaya":   "delta_prepost",
    "Ymaze_Takaya": "delta_prepost",
}

GROUP_DATASET: dict[str, str] = {
    "AST":          "Fiber photometry",
    "Food_intake":  "Fiber photometry",
    "OFT_GI":       "GI-DREADD",
    "OFT_single":   "OFT pooled",
    "Y_maze":       "Y-maze pooled",
    "OFT_s15":      "Surgery 15mice",
    "Ymaze_s15":    "Surgery 15mice",
    "OFT_Takaya":   "Surgery Takaya",
    "Ymaze_Takaya": "Surgery Takaya",
}

# ── Group definitions ─────────────────────────────────────────────────────────

FIVE_GROUPS = [
    {"name": "AST",          "label": "Fiber AST",              "cv": "loo"},
    {"name": "Food_intake",  "label": "Fiber Food",             "cv": "loo"},
    {"name": "OFT_single",   "label": "OFT\n(GI+Takaya pool)", "cv": "group"},
    {"name": "OFT_s15",      "label": "OFT\n(s15mice)",        "cv": "group"},
    {"name": "Y_maze",       "label": "Y maze\n(s15+Takaya)",  "cv": "group"},
    {"name": "OFT_GI",       "label": "OFT\n(GI-DREADD)",      "cv": "group"},
    {"name": "OFT_Takaya",   "label": "OFT\n(Takaya)",         "cv": "group"},
    {"name": "Ymaze_s15",    "label": "Y maze\n(s15mice)",     "cv": "group"},
    {"name": "Ymaze_Takaya", "label": "Y maze\n(Takaya)",      "cv": "group"},
]


# ── BlockScaler ───────────────────────────────────────────────────────────────

class BlockScaler(BaseEstimator, TransformerMixin):
    """Scale each feature block by 1/sqrt(block_size) so every block
    contributes equally to the L2 norm regardless of dimensionality."""
    def __init__(self, block_sizes: list[int]):
        self.block_sizes = block_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy().astype(float)
        start = 0
        for size in self.block_sizes:
            out[:, start:start + size] /= np.sqrt(size)
            start += size
        return out


def make_model(block_sizes: list[int]) -> Pipeline:
    """L2 LogReg pipeline with block-proportional scaling."""
    return Pipeline([
        ("impute",      SimpleImputer(strategy="median")),
        ("block_scale", BlockScaler(block_sizes)),
        ("scale",       StandardScaler()),
        ("clf",         LogisticRegression(
            penalty="l2", solver="liblinear", C=1.0,
            class_weight="balanced", max_iter=5000, random_state=42,
        )),
    ])


# ── Metadata parsers ──────────────────────────────────────────────────────────

def _parse_surgery_15mice_meta(sessions: list[str]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        timepoint = "pre" if s.startswith("POD-1") else "post"
        parts = s.split()
        test_type = parts[1] if len(parts) > 1 else "unknown"
        if len(parts) > 2 and parts[1] == "Y" and parts[2].lower().startswith("maze"):
            test_type = "Y maze"
        m = re.search(r"_mouse(\d+)$", s)
        mouse_id = int(m.group(1)) if m else None
        rows.append({"session": s, "timepoint": timepoint,
                     "test_type": test_type, "mouse_id": mouse_id,
                     "cohort": "surgery_15mice"})
    return pd.DataFrame(rows)



FIBER_H5_JOINT = "outputs/03_moseq/kpms_project_surgery_joint/model_surgery_joint/results_fiber.h5"
FIBER_H5_OWN   = "outputs/03_moseq/kpms_project_fiber_photometry/model_v01/results.h5"
FIBER_DIR      = "Data/Fiber photometry"
FIBER_H5 = FIBER_H5_JOINT if Path(FIBER_H5_JOINT).exists() else FIBER_H5_OWN


def build_fiber_metadata() -> pd.DataFrame:
    """Parse experiment/condition from folder structure."""
    import os
    path_map = {}
    for root, dirs, files in os.walk(FIBER_DIR):
        parts = root.replace(FIBER_DIR + "/", "").split("/")
        if len(parts) >= 4:
            experiment  = parts[0]
            condition   = parts[1]
            session_folder = parts[3]
            for sub in os.listdir(root):
                key = f"{session_folder}_{sub}"
                path_map[key] = (experiment, condition)

    with h5py.File(FIBER_H5, "r") as f:
        sessions = list(f.keys())

    rows = []
    for s in sessions:
        key  = "_".join(s.split("_")[:2])
        meta = path_map.get(key, (None, None))
        rows.append({"session": s, "experiment": meta[0], "condition": meta[1]})

    df = pd.DataFrame(rows)
    print(f"\nFiber metadata: {len(df)} sessions")
    print(df.groupby(["experiment", "condition"]).size().to_string())
    return df


def _parse_gi_meta(sessions: list[str]) -> pd.DataFrame:
    meta_csv = pd.read_csv(GI_META_CSV)
    lookup = meta_csv.set_index("video_stem")
    rows = []
    for s in sessions:
        stem = s.split("_superanimal")[0].strip()
        if stem in lookup.index:
            row = lookup.loc[stem]
            rows.append({"session": s, "timepoint": row["timepoint"],
                         "mouse_id": row["mouse_id"], "dreadd": row["dreadd"],
                         "lps": row["lps"], "group": row["group"],
                         "cohort": "gi_mice_oft"})
        else:
            rows.append({"session": s, "timepoint": None, "mouse_id": None,
                         "dreadd": None, "lps": None, "group": None,
                         "cohort": "gi_mice_oft"})
    return pd.DataFrame(rows)


def build_gi_metadata() -> pd.DataFrame:
    if not Path(ALL_H5).exists():
        logger.warning("model_all results.h5 not found — skipping Gi analysis.")
        return pd.DataFrame()
    with h5py.File(ALL_H5, "r") as f:
        gi_sessions = [s for s in f.keys() if "Gi-" in s]
    df = _parse_gi_meta(gi_sessions)
    df = df[df["timepoint"].notna()].copy()
    logger.info("Gi metadata: %d sessions", len(df))
    logger.info("\n%s", df.groupby(["timepoint", "group"]).size().to_string())
    return df


# ── state-moseq prerequisite ──────────────────────────────────────────────────

def _fit_state_moseq() -> None:
    """Fit a hierarchical HMM (state-moseq) on model_all syllable sequences.
    Output saved to STATE_MOSEQ_H5.  Skips silently if already exists."""
    if Path(STATE_MOSEQ_H5).exists():
        logger.info("state-moseq cache exists — skipping fit: %s", STATE_MOSEQ_H5)
        return

    if not Path(ALL_H5).exists():
        logger.warning("model_all results.h5 not found — cannot fit state-moseq.")
        return

    logger.info(
        "Fitting state-moseq  K=%d  kappa=%.1f  ebs=%.2f  iters=%d ...",
        STATE_MOSEQ_K, STATE_MOSEQ_TRANS_KAPPA, STATE_MOSEQ_EBS, STATE_MOSEQ_NUM_ITERS,
    )

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.random as jr
    import state_moseq as sm
    import tqdm

    seqs: dict[str, np.ndarray] = {}
    with h5py.File(ALL_H5, "r") as f:
        for s in f.keys():
            seqs[s] = f[s]["syllable"][:].astype(np.int32)
    logger.info("  %d sessions loaded", len(seqs))

    syllables_batched, mask, metadata = sm.batch(seqs, seg_length=3000)
    keys_arr, bounds_arr = metadata
    logger.info("  Batched shape: %s  mask sum: %.0f",
                syllables_batched.shape, mask.sum())

    data = {
        "syllables": syllables_batched.astype(np.int32),
        "mask":      mask.astype(np.float32),
    }
    hypparams = {
        "n_syllables":           NUM_STATES,
        "n_states":              STATE_MOSEQ_K,
        "emission_base_sigma":   1.0,
        "emission_biases_sigma": STATE_MOSEQ_EBS,
        "trans_kappa":           STATE_MOSEQ_TRANS_KAPPA,
        "trans_beta":            1.0,
        "emission_gd_iters":     50,
        "emission_gd_lr":        0.01,
    }

    rng = jr.PRNGKey(42)
    rng, init_rng = jr.split(rng)
    logger.info("  Initialising parameters ...")
    params = sm.hhmm_efficient.initialize_params(data, hypparams, seed=init_rng)

    rng, s0 = jr.split(rng)
    states, _ = sm.hhmm_efficient.resample_states(s0, data, params)
    for _ in tqdm.trange(STATE_MOSEQ_NUM_ITERS, desc="state-moseq Gibbs"):
        rng, s1, s2 = jr.split(rng, 3)
        params, _ = sm.hhmm_efficient.resample_params(s1, data, states, hypparams, params)
        states, _ = sm.hhmm_efficient.resample_states(s2, data, params)

    states_np = np.array(states)
    per_session = sm.unbatch(states_np, keys_arr, bounds_arr)

    all_z = np.concatenate(list(per_session.values()))
    counts = np.bincount(all_z, minlength=STATE_MOSEQ_K)
    logger.info("Meta-state usage:")
    for k in range(STATE_MOSEQ_K):
        logger.info("  state %2d: %.1f%%", k, 100 * counts[k] / counts.sum())

    Path(STATE_MOSEQ_H5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(STATE_MOSEQ_H5, "w") as f:
        f.attrs["K"]                     = STATE_MOSEQ_K
        f.attrs["n_syllables"]           = NUM_STATES
        f.attrs["num_iters"]             = STATE_MOSEQ_NUM_ITERS
        f.attrs["trans_kappa"]           = STATE_MOSEQ_TRANS_KAPPA
        f.attrs["emission_biases_sigma"] = STATE_MOSEQ_EBS
        for sess, z in per_session.items():
            f.create_dataset(sess, data=z.astype(np.int16), compression="gzip")
    logger.info("Saved to %s", STATE_MOSEQ_H5)


def _build_metastate_features(sessions: list[str],
                               include_trans: bool = False) -> np.ndarray | None:
    """Build meta-state usage (and optionally transition) features from
    the state-moseq cache.  Returns None if the cache doesn't exist."""
    if not Path(STATE_MOSEQ_H5).exists():
        return None
    K = STATE_MOSEQ_K
    rows = []
    with h5py.File(STATE_MOSEQ_H5, "r") as f:
        for sess in sessions:
            if sess not in f:
                return None
            z = f[sess][:].astype(int)
            T = len(z)
            u = np.bincount(z, minlength=K)[:K] / T
            if include_trans:
                trans = np.zeros((K, K), dtype=np.float64)
                for t in range(T - 1):
                    trans[z[t], z[t + 1]] += 1
                t_vec = np.zeros(K * K)
                for i in range(K):
                    row = trans[i] + LAPLACE_PSEUDOCOUNT
                    t_vec[i * K:(i + 1) * K] = row / row.sum()
                rows.append(np.concatenate([u, t_vec]))
            else:
                rows.append(u)
    return np.array(rows)


# ── Feature bank ──────────────────────────────────────────────────────────────

def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) else 0.0


def _bout_lengths(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(seq) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1]) + 1]
    ends   = np.r_[starts[1:], len(seq)]
    return seq[starts], (ends - starts)


def build_session_feature_bank() -> dict[str, dict]:
    """Compute all per-session features once and return a feature bank dict.

    Each entry is a dict with keys:
      usage       – (50,) normalised syllable usage
      trans_norm  – (50, 50) Laplace-smoothed row-normalised transition matrix
      transsummary– (100,) = [self_trans(50) | row_entropy(50)]
      seqstats    – (12,) session-level sequence statistics
      freq50_idx  – indices of syllables present in ≥50 % of sessions
      metastate_usage       – (K,) if state-moseq cache available
      metastate_usage_trans – (K+K²,) if state-moseq cache available
    """
    seqs: dict[str, np.ndarray] = {}
    with h5py.File(ALL_H5, "r") as f:
        for sess in f.keys():
            seqs[sess] = f[sess]["syllable"][:].astype(np.int32)

    features: dict[str, dict] = {}
    usage_rows = []

    for sess, seq in seqs.items():
        T = max(len(seq), 1)
        counts = np.bincount(seq, minlength=NUM_STATES)[:NUM_STATES].astype(float)
        usage  = counts / T

        trans = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
        if len(seq) > 1:
            for t in range(len(seq) - 1):
                trans[seq[t], seq[t + 1]] += 1

        row_sums = trans.sum(axis=1)
        self_trans = np.divide(
            np.diag(trans), row_sums,
            out=np.zeros(NUM_STATES, dtype=float), where=row_sums > 0,
        )
        row_entropy = np.array([
            _entropy(trans[i] / row_sums[i]) if row_sums[i] > 0 else 0.0
            for i in range(NUM_STATES)
        ])

        _, bout_lens = _bout_lengths(seq)
        switch_rate       = float(np.mean(seq[1:] != seq[:-1])) if len(seq) > 1 else 0.0
        n_bouts           = len(bout_lens)
        seqstats = np.array([
            len(seq),
            switch_rate,
            _entropy(usage),
            float(np.sort(usage)[::-1][:5].sum()),
            float(np.max(usage)) if len(usage) else 0.0,
            float(np.mean(bout_lens)) if n_bouts else 0.0,
            float(np.median(bout_lens)) if n_bouts else 0.0,
            float(np.std(bout_lens)) if n_bouts else 0.0,
            float(np.mean(bout_lens >= 3)) if n_bouts else 0.0,
            float(np.mean(usage > 0)),
            float(np.mean(row_entropy[row_sums > 0])) if np.any(row_sums > 0) else 0.0,
            float(np.std(row_entropy[row_sums > 0])) if np.any(row_sums > 0) else 0.0,
        ], dtype=float)

        trans_smooth = trans + 1.0
        trans_norm   = trans_smooth / trans_smooth.sum(axis=1, keepdims=True)

        features[sess] = {
            "usage":        usage,
            "trans_norm":   trans_norm,
            "transsummary": np.concatenate([self_trans, row_entropy]),
            "seqstats":     seqstats,
        }
        usage_rows.append(usage)

    # Global syllable indices (consistent across groups)
    U = np.vstack(usage_rows)
    freq50_idx = np.where((U > 0).mean(axis=0) >= 0.50)[0]
    for sess in seqs:
        features[sess]["freq50_idx"] = freq50_idx

    # Meta-state features (optional)
    sessions_list = list(seqs.keys())
    meta_u = _build_metastate_features(sessions_list, include_trans=False)
    meta_t = _build_metastate_features(sessions_list, include_trans=True)
    for i, sess in enumerate(sessions_list):
        if meta_u is not None:
            features[sess]["metastate_usage"] = meta_u[i]
        if meta_t is not None:
            features[sess]["metastate_usage_trans"] = meta_t[i]

    return features


def _assemble_feature_matrix(
    sessions: list[str],
    feature_bank: dict[str, dict],
    feature_kind: str,
) -> tuple[np.ndarray, list[int]] | None:
    """Return (X, block_sizes) for the given feature_kind string.

    feature_kind is a '+'-joined set of components, always with freq50 as base:
      freq50        – usage of freq50 syllables                              (p=5)
      trans         – freq50×freq50 Laplace-smoothed transition submatrix   (p=25)
      transsummary  – self-trans + row-entropy for freq50 syllables         (p=10)
      seqstats      – 12 session-level sequence statistics
      metastate     – meta-state usage vector from state-moseq             (p=K)
    """
    mods = set(feature_kind.split("+")) - {"freq50"}
    rows: list[np.ndarray] = []
    block_sizes: list[int] | None = None

    for sess in sessions:
        f   = feature_bank[sess]
        idx = f["freq50_idx"]
        if len(idx) == 0:
            return None

        pieces = [f["usage"][idx]]

        if "trans" in mods:
            pieces.append(f["trans_norm"][np.ix_(idx, idx)].ravel())

        if "transsummary" in mods:
            ts = f["transsummary"]               # [self_trans(50) | row_entropy(50)]
            pieces.append(ts[:NUM_STATES][idx])
            pieces.append(ts[NUM_STATES:][idx])

        if "seqstats" in mods:
            pieces.append(f["seqstats"])

        if "metastate" in mods:
            m = f.get("metastate_usage")
            if m is None:
                return None
            pieces.append(m)

        if block_sizes is None:
            block_sizes = [len(p) for p in pieces]

        rows.append(np.concatenate(pieces).astype(float))

    if not rows:
        return None
    return np.vstack(rows), block_sizes  # type: ignore[return-value]


# ── Group metadata & representation ──────────────────────────────────────────

def _build_group_metadata(
    group_name: str,
    sessions: list[str],
    y: np.ndarray,
    mouse_ids: np.ndarray | None,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "session":  sessions,
        "label":    y.astype(int),
        "mouse_id": (mouse_ids if mouse_ids is not None
                     else [f"sess_{i}" for i in range(len(sessions))]),
    })

    if group_name in {"OFT_s15", "Ymaze_s15"}:
        meta = _parse_surgery_15mice_meta(sessions)[
            ["session", "timepoint", "test_type", "mouse_id"]]
        meta["mouse_id"] = meta["mouse_id"].astype(str)
        df = df.merge(meta, on=["session", "mouse_id"], how="left")

    elif group_name in {"OFT_Takaya", "Ymaze_Takaya"}:
        tak = pd.read_csv(TAKAYA_META_CSV)
        lookup = tak.set_index("video_stem")
        rows = []
        for sess in sessions:
            stem = sess.split("_superanimal")[0]
            row  = lookup.loc[stem]
            rows.append({
                "session":    sess,
                "timepoint":  row["timepoint"],
                "test_type":  row["test_type"],
                "mouse_id":   str(row["mouse_id"]),
                "group_name": row["group"],
            })
        meta = pd.DataFrame(rows)
        df["mouse_id"] = df["mouse_id"].astype(str)
        df = df.merge(meta, on=["session", "mouse_id"], how="left")

    elif group_name == "OFT_GI":
        meta = _parse_gi_meta(sessions)[["session", "timepoint", "mouse_id", "group"]]
        meta["mouse_id"] = meta["mouse_id"].astype(str)
        df["mouse_id"]   = df["mouse_id"].astype(str)
        df = df.merge(meta, on=["session", "mouse_id"], how="left")

    df["group_name"] = group_name
    return df


def _apply_representation(
    group_name: str,
    meta: pd.DataFrame,
    X: np.ndarray,
    outer_cv_type: str,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None] | None:
    """Return (X_use, y_use, cv_use, groups_use) for the given representation,
    or None if the representation is incompatible with the group."""
    work = meta.copy()
    work["row_ix"] = np.arange(len(work))

    if len(np.unique(work["label"])) < 2:
        return None

    # Fiber groups are always raw / LOO
    if group_name.startswith("Fiber") or group_name in {"AST", "Food_intake"}:
        if representation != "raw":
            return None
        return X, work["label"].to_numpy(), outer_cv_type, work["mouse_id"].to_numpy(dtype=str)

    if outer_cv_type != "group":
        return X, work["label"].to_numpy(), outer_cv_type, work["mouse_id"].to_numpy(dtype=str)

    if representation == "raw":
        return X, work["label"].to_numpy(), "group", work["mouse_id"].to_numpy(dtype=str)

    if representation == "mouse_mean":
        rows, labels, mice = [], [], []
        for (mouse_id, label), sub in work.groupby(["mouse_id", "label"]):
            rows.append(X[sub["row_ix"].to_numpy()].mean(axis=0))
            labels.append(int(label))
            mice.append(str(mouse_id))
        y_new = np.array(labels, dtype=int)
        if len(np.unique(y_new)) < 2:
            return None
        return np.vstack(rows), y_new, "stratified", np.array(mice, dtype=str)

    if representation == "mouse_centered":
        repeated = work.groupby("mouse_id")["label"].nunique().max() >= 2
        if not repeated:
            return None
        X_new = X.copy()
        for mouse_id, sub in work.groupby("mouse_id"):
            idx = sub["row_ix"].to_numpy()
            X_new[idx] -= X_new[idx].mean(axis=0, keepdims=True)
        return X_new, work["label"].to_numpy(), "group", work["mouse_id"].to_numpy(dtype=str)

    if representation == "delta_prepost":
        if group_name not in {"OFT_Takaya", "Ymaze_Takaya"}:
            return None
        if "timepoint" not in work.columns:
            return None
        rows, labels, mice = [], [], []
        for mouse_id, sub in work.groupby("mouse_id"):
            if (set(sub["timepoint"].dropna()) >= {"pre", "post"}
                    and sub["label"].nunique() == 1):
                pre  = X[sub[sub["timepoint"] == "pre"]["row_ix"].to_numpy()].mean(axis=0)
                post = X[sub[sub["timepoint"] == "post"]["row_ix"].to_numpy()].mean(axis=0)
                rows.append(post - pre)
                labels.append(int(sub["label"].iloc[0]))
                mice.append(str(mouse_id))
        if len(rows) < 4 or len(np.unique(labels)) < 2:
            return None
        return np.vstack(rows), np.array(labels, dtype=int), "stratified", np.array(mice, dtype=str)

    raise ValueError(f"Unknown representation: {representation!r}")


# ── CV helpers ────────────────────────────────────────────────────────────────

def _make_splits(y: np.ndarray, cv_type: str, groups: np.ndarray | None):
    if cv_type == "loo":
        return list(LeaveOneOut().split(np.zeros(len(y), dtype=int)))
    if cv_type == "group":
        if groups is None:
            return None
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return None
        n_splits = min(5, n_groups)
        if n_splits < 2:
            return None
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups=groups))
    if cv_type == "stratified":
        min_class = int(np.min(np.bincount(y)))
        if min_class < 2:
            return None
        return list(StratifiedKFold(n_splits=min(5, min_class),
                                    shuffle=True, random_state=42).split(
                                        np.zeros(len(y)), y))
    raise ValueError(f"Unknown cv_type: {cv_type!r}")


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_type: str,
    groups: np.ndarray | None,
    block_sizes: list[int] | None = None,
) -> dict | None:
    from sklearn.base import clone
    splits = _make_splits(y, cv_type, groups)
    if splits is None or len(splits) < 2:
        return None

    model = make_model(block_sizes or [X.shape[1]])
    y_prob_all = np.zeros(len(y), dtype=float)
    auc_scores, acc_scores, f1_scores = [], [], []

    for tr, te in splits:
        fitted   = clone(model).fit(X[tr], y[tr])
        y_prob   = fitted.predict_proba(X[te])[:, 1]
        y_pred   = (y_prob >= 0.5).astype(int)
        y_prob_all[te] = y_prob
        if len(np.unique(y[te])) == 2:
            auc_scores.append(roc_auc_score(y[te], y_prob))
        acc_scores.append(accuracy_score(y[te], y_pred))
        f1_scores.append(f1_score(y[te], y_pred, average="macro"))

    point_auc = float(roc_auc_score(y, y_prob_all)) if len(np.unique(y)) == 2 else np.nan

    # Bootstrap 95 % CI on the pooled held-out predictions (works for LOO and k-fold)
    ci_lo, ci_hi = np.nan, np.nan
    if not np.isnan(point_auc):
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(1000):
            idx = rng.integers(0, len(y), size=len(y))
            if len(np.unique(y[idx])) == 2:
                boot.append(roc_auc_score(y[idx], y_prob_all[idx]))
        if boot:
            ci_lo = float(np.percentile(boot, 2.5))
            ci_hi = float(np.percentile(boot, 97.5))

    return {
        "point_auc": point_auc,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "auc_mean":  float(np.mean(auc_scores)) if auc_scores else np.nan,
        "auc_std":   float(np.std(auc_scores))  if auc_scores else np.nan,
        "acc":       float(np.mean(acc_scores)),
        "f1":        float(np.mean(f1_scores)),
    }


# ── Group data builder (exported for benchmark_improvements.py) ───────────────


def build_five_group_data() -> dict[str, tuple]:
    """Return {group_name: (sessions, y, cv_type, mouse_ids)} for all 9 groups."""
    with h5py.File(ALL_H5, "r") as f:
        all_in_h5 = set(f.keys())

    groups: dict[str, tuple] = {}

    # Groups 1 & 2: Fiber
    fib_meta = build_fiber_metadata()
    fib_meta = fib_meta[fib_meta["session"].isin(all_in_h5)].copy()
    for exp_key, grp_name in [("AST", "AST"),
                               ("Food intake during habituation", "Food_intake")]:
        sub = fib_meta[(fib_meta["experiment"] == exp_key) &
                       fib_meta["condition"].notna()].copy()
        if sub["condition"].nunique() >= 2 and len(sub) >= 4:
            y = (sub["condition"] == "Social isolation").astype(int).values
            groups[grp_name] = (sub["session"].tolist(), y, "loo", None)

    # Takaya metadata
    tak_csv    = pd.read_csv(TAKAYA_META_CSV)
    tak_lookup = tak_csv.set_index("video_stem")
    tak_rows   = []
    for s in all_in_h5:
        if not s.startswith("S1"):
            continue
        stem = s.split("_superanimal")[0]
        if stem not in tak_lookup.index:
            continue
        r = tak_lookup.loc[stem]
        tak_rows.append({"session": s, "test_type": r["test_type"],
                         "group": r["group"], "mouse_id": r["mouse_id"]})
    tak_df = pd.DataFrame(tak_rows) if tak_rows else pd.DataFrame(
        columns=["session", "test_type", "group", "mouse_id"])

    # s15mice metadata
    s15_sess = [s for s in all_in_h5 if "_mouse" in s]
    s15_df   = _parse_surgery_15mice_meta(s15_sess)

    # GI-DREADD metadata
    gi_df = build_gi_metadata()
    gi_df = gi_df[gi_df["session"].isin(all_in_h5) &
                  gi_df["timepoint"].isin(["pre", "post1", "post3"])].copy()
    gi_df["label"] = (gi_df["timepoint"] != "pre").astype(int)

    # OFT_GI: pre=0, post1+post3=1; GroupKFold by mouse
    if len(gi_df) >= 6:
        groups["OFT_GI"] = (gi_df["session"].tolist(),
                            gi_df["label"].values, "group",
                            gi_df["mouse_id"].astype(str).values)

    # OFT_Takaya: control=0, surgery=1
    tak_oft = tak_df[tak_df["test_type"] == "OFT"].copy()
    tak_oft["label"] = (tak_oft["group"] == "surgery").astype(int)
    if len(tak_oft) >= 6:
        groups["OFT_Takaya"] = (tak_oft["session"].tolist(),
                                tak_oft["label"].values, "group",
                                tak_oft["mouse_id"].astype(str).values)

    # OFT_s15: pre=0, post=1
    s15_oft = s15_df[s15_df["test_type"] == "OFT"].copy()
    s15_oft["label"] = (s15_oft["timepoint"] == "post").astype(int)
    if len(s15_oft) >= 6:
        groups["OFT_s15"] = (s15_oft["session"].tolist(),
                             s15_oft["label"].values, "group",
                             s15_oft["mouse_id"].astype(str).values)

    # Ymaze_s15: pre=0, post=1
    s15_ymaze = s15_df[s15_df["test_type"] == "Y maze"].copy()
    s15_ymaze["label"] = (s15_ymaze["timepoint"] == "post").astype(int)
    if len(s15_ymaze) >= 6:
        groups["Ymaze_s15"] = (s15_ymaze["session"].tolist(),
                               s15_ymaze["label"].values, "group",
                               s15_ymaze["mouse_id"].astype(str).values)

    # Ymaze_Takaya: control=0, surgery=1
    tak_ymaze = (tak_df[tak_df["test_type"] == "Y maze"].copy()
                 if not tak_df.empty
                 else pd.DataFrame(columns=["session", "test_type", "group", "mouse_id"]))
    tak_ymaze["label"] = (tak_ymaze["group"] == "surgery").astype(int)
    if len(tak_ymaze) >= 6:
        groups["Ymaze_Takaya"] = (tak_ymaze["session"].tolist(),
                                  tak_ymaze["label"].values, "group",
                                  tak_ymaze["mouse_id"].astype(str).values)

    # OFT_single: pooled GI + Takaya OFT
    if len(gi_df) >= 4 and len(tak_oft) >= 4:
        gi_pool  = gi_df[["session", "label", "mouse_id"]].copy()
        gi_pool["mouse_id"] = "gi_" + gi_pool["mouse_id"].astype(str)
        tak_pool = tak_oft[["session", "label", "mouse_id"]].copy()
        tak_pool["mouse_id"] = "tak_" + tak_pool["mouse_id"].astype(str)
        pool_oft = pd.concat([gi_pool, tak_pool], ignore_index=True)
        groups["OFT_single"] = (pool_oft["session"].tolist(),
                                pool_oft["label"].values, "group",
                                pool_oft["mouse_id"].values)

    # Y_maze: pooled s15 + Takaya Y-maze
    if len(s15_ymaze) >= 4 and len(tak_ymaze) >= 4:
        s15_pool    = s15_ymaze[["session", "label", "mouse_id"]].copy()
        s15_pool["mouse_id"] = "s15_" + s15_pool["mouse_id"].astype(str)
        tak_ym_pool = tak_ymaze[["session", "label", "mouse_id"]].copy()
        tak_ym_pool["mouse_id"] = "tak_" + tak_ym_pool["mouse_id"].astype(str)
        pool_ymaze = pd.concat([s15_pool, tak_ym_pool], ignore_index=True)
        groups["Y_maze"] = (pool_ymaze["session"].tolist(),
                            pool_ymaze["label"].values, "group",
                            pool_ymaze["mouse_id"].values)

    for g in FIVE_GROUPS:
        if g["name"] in groups:
            sess, y, cv, mids = groups[g["name"]]
            n_mice = len(np.unique(mids)) if mids is not None else len(sess)
            print(f"  {g['label']:35s} n={len(sess):3d}  "
                  f"class0={(y==0).sum():3d}  class1={(y==1).sum():3d}  "
                  f"cv={cv}  mice={n_mice}")
        else:
            print(f"  {g['label']:35s} SKIPPED (insufficient data)")

    return groups


# ── Main classification run ───────────────────────────────────────────────────

def run_five_groups() -> pd.DataFrame:
    if not Path(ALL_H5).exists():
        logger.warning("model_all results.h5 not found — skipping classification.")
        return pd.DataFrame()

    print("\n" + "="*60)
    print("  STAGE 4: Nine-group classification (model_all syllables)")
    print("="*60)

    print("\nBuilding session feature bank ...")
    feature_bank = build_session_feature_bank()

    groups_data = build_five_group_data()
    records = []

    for grp in FIVE_GROUPS:
        gname = grp["name"]
        if gname not in groups_data:
            continue
        sessions, y, cv_type, mouse_ids = groups_data[gname]
        group_meta = _build_group_metadata(gname, sessions, y, mouse_ids)
        fixed_repr = GROUP_REPRESENTATION[gname]

        print(f"\n  {grp['label'].replace(chr(10), ' ')}  "
              f"(n={len(y)}, class0={(y==0).sum()}, class1={(y==1).sum()}, "
              f"cv={cv_type}, repr={fixed_repr})")

        for feat_name in FEATURE_SETS:
            result = _assemble_feature_matrix(sessions, feature_bank, feat_name)
            if result is None:
                print(f"    [{feat_name}]  skipped (features unavailable)")
                continue
            X_raw, block_sizes = result

            transformed = _apply_representation(
                gname, group_meta, X_raw, cv_type, fixed_repr)
            if transformed is None:
                print(f"    [{feat_name}]  skipped (representation incompatible)")
                continue
            X_use, y_use, cv_use, groups_use = transformed

            if len(y_use) < 6 or len(np.unique(y_use)) < 2:
                print(f"    [{feat_name}]  skipped (too few samples after representation)")
                continue

            # p/n filter: skip if p > 2 * n_train (unreliable L2 LogReg)
            n_folds = len(y_use) if cv_use == "loo" else 5
            n_train = int(len(y_use) * (n_folds - 1) / n_folds)
            if X_use.shape[1] > 2 * n_train:
                print(f"    [{feat_name}]  skipped (p={X_use.shape[1]} > 2*n_train={2*n_train})")
                continue

            res = evaluate_model(X_use, y_use, cv_use, groups_use, block_sizes)
            if res is None:
                continue

            auc_str = (f"point_AUC={res['point_auc']:.3f} "
                       f"95%CI=[{res['auc_ci_lo']:.3f},{res['auc_ci_hi']:.3f}]")
            print(f"    [{feat_name:35s}]  p={X_use.shape[1]:3d}  "
                  f"{auc_str}  acc={res['acc']:.3f}  f1={res['f1']:.3f}")

            records.append({
                "dataset":       GROUP_DATASET.get(gname, gname),
                "group_name":    gname,
                "group_label":   grp["label"].replace("\n", " "),
                "feature_set":   feat_name,
                "representation": fixed_repr,
                "n_samples":     len(y_use),
                "n_features":    int(X_use.shape[1]),
                "cv_type":       cv_use,
                "point_auc":     round(res["point_auc"],  4),
                "auc_ci_lo":     round(res["auc_ci_lo"],  4),
                "auc_ci_hi":     round(res["auc_ci_hi"],  4),
                "auc_mean":      round(res["auc_mean"],   4),
                "auc_std":       round(res["auc_std"],    4),
                "acc":           round(res["acc"],         4),
                "f1":            round(res["f1"],          4),
            })

    return pd.DataFrame(records)


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame) -> None:
    if df.empty:
        return

    # ── Raw results (all numeric columns) ────────────────────────────────────
    out_csv = OUT_DIR / "classification_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  → {out_csv}")

    # ── Formatted summary table ───────────────────────────────────────────────
    out = df.copy()
    out["AUC (point)"] = out.apply(
        lambda r: f"{r['point_auc']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]",
        axis=1,
    )
    out["AUC (CV mean\u00b1std)"] = out.apply(
        lambda r: (f"{r['auc_mean']:.3f}\u00b1{r['auc_std']:.3f}"
                   if not np.isnan(r["auc_mean"]) and r["cv_type"] != "loo"
                   else "\u2014"),
        axis=1,
    )

    summary = out[["group_name", "feature_set", "representation",
                   "AUC (point)", "AUC (CV mean\u00b1std)"]].rename(columns={
        "group_name":     "Group",
        "feature_set":    "Feature Set",
        "representation": "Representation",
    })

    summary_csv = OUT_DIR / "classification_summary.csv"
    summary.to_csv(summary_csv, index=False, sep="\t")
    print(f"  → {summary_csv}\n")

    # Print with dataset as grouping rows
    cv_col  = "AUC (CV mean\u00b1std)"
    dataset_col = out["dataset"].tolist()
    header = ("Group           Feature Set                           Representation    "
              "AUC (point)                AUC (CV mean\u00b1std)")
    print(header)
    print("-" * len(header))
    prev_dataset = None
    for (_, row), dataset in zip(summary.iterrows(), dataset_col):
        if dataset != prev_dataset:
            print(f"\n  [{dataset}]")
            prev_dataset = dataset
        print(f"  {row['Group']:<12}  {row['Feature Set']:<35}  "
              f"{row['Representation']:<16}  {row['AUC (point)']:<26}  "
              f"{row[cv_col]}")


def plot_results(df: pd.DataFrame) -> None:
    if df.empty:
        return

    df = df.copy()
    group_order  = [g["name"]  for g in FIVE_GROUPS if g["name"] in df["group_name"].values]
    group_labels = {g["name"]: g["label"].replace("\n", " ") for g in FIVE_GROUPS}
    colors       = ["#4C72B0", "#DD8452"]

    x      = np.arange(len(group_order))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, feat in enumerate(FEATURE_SETS):
        sub = df[df["feature_set"] == feat].set_index("group_name")
        aucs   = [sub.loc[g, "point_auc"] if g in sub.index else np.nan for g in group_order]
        ci_lo  = [sub.loc[g, "auc_ci_lo"] if g in sub.index else np.nan for g in group_order]
        ci_hi  = [sub.loc[g, "auc_ci_hi"] if g in sub.index else np.nan for g in group_order]
        # asymmetric error bars: [below, above]
        errs = [
            [max(0.0, a - lo) for a, lo in zip(aucs, ci_lo)],
            [max(0.0, hi - a) for a, hi in zip(aucs, ci_hi)],
        ]
        bars = ax.bar(x + (i - 0.5) * width, aucs, width,
                      yerr=errs, capsize=3,
                      color=colors[i], alpha=0.85, label=feat)
        for rect, val in zip(bars, aucs):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([group_labels[g] for g in group_order], fontsize=9)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_title("Stage 4 classification — two canonical feature sets", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    out_path = OUT_DIR / "classification_auc_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    """Called by pipeline.py run --stages 4."""
    np.random.seed(42)
    _fit_state_moseq()
    df = run_five_groups()
    save_results(df)
    plot_results(df)
    logger.info("Stage 4 complete. Outputs in %s", OUT_DIR)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(config_path=args.config)
    print(f"\nAll outputs saved to {OUT_DIR}/")
