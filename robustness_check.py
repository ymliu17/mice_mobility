"""
MoSeq Model Robustness Checks
==============================
Two complementary tests for model_all (270 sessions, no BFT):

  1. Checkpoint consistency  — ARI between syllable assignments at consecutive
                               saved iterations (0,25,50,...,400).
                               Converged = ARI plateau near 1.0.

  2. Stratified bootstrap    — 10× resample 80% of sessions stratified by
                               cohort+condition; refit each; align to model_all;
                               report mean stability score (fraction of frames
                               with consistent syllable after alignment).

Outputs saved to: outputs/robustness/

Run:
    # Checkpoint consistency (after model_all finishes):
    PYTHONPATH=. /home/yangliu/miniconda3/envs/mice/bin/python robustness_check.py

    # Bootstrap subsampling (expensive, ~4 hrs):
    PYTHONPATH=. nohup .../python robustness_check.py --bootstrap --n-boot 10 > logs/bootstrap.log 2>&1 &
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ALL   = "outputs/03_moseq/kpms_project_all"
MODEL_ALL     = "model_all"
CHECKPOINT    = f"{PROJECT_ALL}/{MODEL_ALL}/checkpoint.h5"
RESULTS_ALL   = f"{PROJECT_ALL}/{MODEL_ALL}/results.h5"
PREPROCESSED  = "outputs/02_preprocessed"
USE_BODYPARTS = ["nose", "left_ear", "right_ear", "neck",
                 "mid_back", "mouse_center", "tail_base"]
NUM_STATES    = 50
OUT_DIR       = Path("outputs/robustness")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hungarian_align(ref_usage: np.ndarray, cmp_usage: np.ndarray) -> np.ndarray:
    """
    Return a mapping array such that cmp_usage[mapping[i]] ≈ ref_usage[i].
    Uses cosine similarity between usage profiles.
    ref_usage / cmp_usage: (num_states,) mean usage vectors.
    """
    # Cost = negative cosine similarity between every pair of syllable profiles
    # (here usage is a scalar per syllable, so just use absolute difference)
    cost = np.abs(ref_usage[:, None] - cmp_usage[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = np.zeros(NUM_STATES, dtype=int)
    mapping[col_ind] = row_ind   # cmp syllable col_ind[i] → ref syllable row_ind[i]
    # Invert: for each ref syllable index, which cmp syllable maps to it?
    inv = np.zeros(NUM_STATES, dtype=int)
    inv[row_ind] = col_ind
    return inv   # apply as: aligned_z = inv[z_cmp]


def z_from_checkpoint(snap_key: str) -> np.ndarray:
    """Load flattened syllable assignments from a checkpoint snapshot."""
    with h5py.File(CHECKPOINT, "r") as f:
        z   = f[f"model_snapshots/{snap_key}/states/z"][:]       # (sessions, T-nlags)
        msk = f["data/mask"][:]                                   # (sessions, T)
    # z is shorter than mask by nlags; trim mask
    T_z = z.shape[1]
    msk = msk[:, :T_z].astype(bool)
    return z[msk].astype(int)


def z_from_results(results_h5: str, session_order: list[str] | None = None) -> np.ndarray:
    """Load flattened syllable assignments from results.h5."""
    with h5py.File(results_h5, "r") as f:
        sessions = session_order or list(f.keys())
        return np.concatenate([f[s]["syllable"][:].astype(int) for s in sessions])


def usage_from_z(z: np.ndarray) -> np.ndarray:
    counts = np.bincount(z, minlength=NUM_STATES)[:NUM_STATES].astype(float)
    return counts / counts.sum()


# ── Check 1: Checkpoint consistency ──────────────────────────────────────────

def check_checkpoint_consistency() -> None:
    print("\n" + "="*60)
    print("  CHECK 1: Checkpoint consistency")
    print("="*60)

    with h5py.File(CHECKPOINT, "r") as f:
        snap_keys = sorted(f["model_snapshots"].keys(), key=lambda x: int(x))
    print(f"  Snapshots: {snap_keys}")

    # Also include final results if available
    has_results = Path(RESULTS_ALL).exists()

    z_snaps = {}
    for k in snap_keys:
        z_snaps[int(k)] = z_from_checkpoint(k)
        print(f"  iter {k:>4}: {len(z_snaps[int(k)])} frames, "
              f"{len(np.unique(z_snaps[int(k)]))} unique syllables")

    if has_results and 200 not in z_snaps:
        z_snaps[200] = z_from_results(RESULTS_ALL)
        print(f"  iter  200 (results.h5): {len(z_snaps[200])} frames")

    iters = sorted(z_snaps.keys())
    aris = []
    for i in range(1, len(iters)):
        a, b = iters[i-1], iters[i]
        # Subsample for speed (ARI on 1M+ frames is slow)
        n = min(len(z_snaps[a]), len(z_snaps[b]), 50_000)
        idx = np.random.choice(min(len(z_snaps[a]), len(z_snaps[b])), n, replace=False)
        ari = adjusted_rand_score(z_snaps[a][idx], z_snaps[b][idx])
        aris.append((a, b, ari))
        print(f"  ARI({a}→{b}): {ari:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [b for a, b, _ in aris]
    ys = [ari for _, _, ari in aris]
    ax.plot(xs, ys, "o-", color="#4C72B0", lw=2, ms=6)
    ax.axhline(0.9, color="green", ls="--", lw=1, label="ARI=0.9 (good)")
    ax.axhline(0.7, color="orange", ls="--", lw=1, label="ARI=0.7 (marginal)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("ARI vs previous checkpoint")
    ax.set_title("Checkpoint consistency — syllable assignment stability")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "checkpoint_consistency.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'checkpoint_consistency.png'}")

    np.save(OUT_DIR / "checkpoint_aris.npy",
            np.array([(a, b, ari) for a, b, ari in aris]))




# ── Check 3: Stratified bootstrap ────────────────────────────────────────────

def check_bootstrap(n_boot: int = 10, frac: float = 0.8) -> None:
    print("\n" + "="*60)
    print(f"  CHECK 3: Stratified bootstrap (n={n_boot}, frac={frac})")
    print("="*60)

    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    from jax_moseq.utils.debugging import convert_data_precision
    from utils.io_utils import load_all_preprocessed
    from stages.s3_moseq_fitting import _filter_to_use_bodyparts
    import re as _re

    # Build session → stratum mapping
    coords, confs, bodyparts = load_all_preprocessed(PREPROCESSED)
    keep = [s for s in coords if "BFT" not in s]
    coords = {s: coords[s] for s in keep}
    confs  = {s: confs[s]  for s in keep}
    coords, confs = _filter_to_use_bodyparts(coords, confs, bodyparts, USE_BODYPARTS)

    def stratum(s):
        if s.startswith("fiberphotometry"):
            return "fiber"
        elif "_mouse" in s:
            return "s15_pre" if s.startswith("POD-1") else "s15_post"
        else:
            return "takaya"

    strata: dict[str, list[str]] = {}
    for s in keep:
        st = stratum(s)
        strata.setdefault(st, []).append(s)
    print(f"  Strata: { {k: len(v) for k, v in strata.items()} }")

    # Reference usage
    if not Path(RESULTS_ALL).exists():
        print("  model_all results.h5 not found — run training first.")
        return
    z_ref_flat = z_from_results(RESULTS_ALL)
    usage_ref  = usage_from_z(z_ref_flat)

    # Reference per-session z arrays (in order)
    with h5py.File(RESULTS_ALL, "r") as f:
        ref_sessions = list(f.keys())
        z_ref_per = {s: f[s]["syllable"][:].astype(int) for s in ref_sessions}

    stabilities = []
    pca_path = Path(PROJECT_ALL) / "pca.p"
    if pca_path.exists():
        pca = kpms.load_pca(PROJECT_ALL)
    else:
        print("  pca.p not found — fitting PCA from model_all data ...")
        all_data, all_meta = kpms.format_data(coords, confidences=confs,
                                              project_dir=PROJECT_ALL)
        from jax_moseq.utils.debugging import convert_data_precision as _cdp
        all_data = _cdp(all_data)
        kpms_cfg_tmp = kpms.load_config(PROJECT_ALL)
        ant_tmp  = [USE_BODYPARTS.index(bp) for bp in kpms_cfg_tmp["anterior_bodyparts"]]
        post_tmp = [USE_BODYPARTS.index(bp) for bp in kpms_cfg_tmp["posterior_bodyparts"]]
        pca = kpms.fit_pca(all_data["Y"], all_data["mask"],
                           anterior_idxs=ant_tmp, posterior_idxs=post_tmp,
                           conf=all_data.get("conf"))
        kpms.save_pca(pca, PROJECT_ALL)
        print(f"  PCA saved to {pca_path}")
    kpms_cfg = kpms.load_config(PROJECT_ALL)
    ant  = [USE_BODYPARTS.index(bp) for bp in kpms_cfg["anterior_bodyparts"]]
    post = [USE_BODYPARTS.index(bp) for bp in kpms_cfg["posterior_bodyparts"]]

    for b in range(n_boot):
        np.random.seed(b)
        boot_sessions = []
        for st, sess_list in strata.items():
            n_keep = max(1, int(len(sess_list) * frac))
            boot_sessions.extend(np.random.choice(sess_list, n_keep, replace=False))

        boot_coords = {s: coords[s] for s in boot_sessions}
        boot_confs  = {s: confs[s]  for s in boot_sessions}

        model_name = f"model_all_boot{b:02d}"
        results_h5 = f"{PROJECT_ALL}/{model_name}/results.h5"

        if not Path(results_h5).exists():
            print(f"  Bootstrap {b}: fitting {len(boot_sessions)} sessions ...")
            data, metadata = kpms.format_data(
                boot_coords, confidences=boot_confs, project_dir=PROJECT_ALL)
            data = convert_data_precision(data)
            model = kpms.init_model(
                data, pca=pca, project_dir=PROJECT_ALL,
                anterior_idxs=ant, posterior_idxs=post,
                trans_hypparams={k: float(v) if isinstance(v, str) else v
                                 for k, v in kpms_cfg["trans_hypparams"].items()},
                ar_hypparams=kpms_cfg["ar_hypparams"],
                obs_hypparams=kpms_cfg["obs_hypparams"],
                cen_hypparams=kpms_cfg["cen_hypparams"],
                error_estimator=kpms_cfg["error_estimator"],
                seed=1000 + b,
            )
            model, _ = kpms.fit_model(
                model=model, data=data, metadata=metadata,
                project_dir=PROJECT_ALL, model_name=model_name,
                ar_only=True, num_iters=50, save_every=50,
                parallel_message_passing=False)
            model, _ = kpms.fit_model(
                model=model, data=data, metadata=metadata,
                project_dir=PROJECT_ALL, model_name=model_name,
                ar_only=False, num_iters=200, save_every=200,
                parallel_message_passing=False)
            kpms.extract_results(model=model, metadata=metadata,
                                  project_dir=PROJECT_ALL, model_name=model_name)

        # Load boot results and compare to reference on overlapping sessions
        with h5py.File(results_h5, "r") as f:
            boot_sess_in_results = list(f.keys())
            z_boot_per = {s: f[s]["syllable"][:].astype(int)
                          for s in boot_sess_in_results if s in ref_sessions}

        if not z_boot_per:
            continue

        # ARI between boot and ref on overlapping sessions (no alignment needed)
        z_b_all = np.concatenate([z_boot_per[s] for s in z_boot_per])
        z_r_all = np.concatenate([z_ref_per[s][:len(z_boot_per[s])] for s in z_boot_per])
        # Subsample for speed
        n = min(len(z_b_all), 50_000)
        idx = np.random.choice(len(z_b_all), n, replace=False)
        stab = adjusted_rand_score(z_r_all[idx], z_b_all[idx])
        stabilities.append(stab)
        print(f"  Bootstrap {b:2d}: {len(z_boot_per)} sessions, "
              f"ARI vs model_all = {stab:.4f}")

    if stabilities:
        print(f"\n  Mean ARI vs model_all: {np.mean(stabilities):.4f} ± {np.std(stabilities):.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(stabilities)), stabilities, color="#4C72B0", alpha=0.8)
        ax.axhline(np.mean(stabilities), color="red", ls="--", lw=1.5,
                   label=f"mean={np.mean(stabilities):.3f}")
        ax.set_xlabel("Bootstrap run"); ax.set_ylabel("ARI vs model_all")
        ax.set_title("Bootstrap subsampling — ARI vs model_all (overlapping sessions)")
        ax.set_ylim(0, 1); ax.legend()
        plt.tight_layout()
        fig.savefig(OUT_DIR / "bootstrap_stability.png", dpi=150)
        plt.close(fig)
        print(f"  → {OUT_DIR / 'bootstrap_stability.png'}")
        np.save(OUT_DIR / "bootstrap_stabilities.npy", np.array(stabilities))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run bootstrap subsampling (expensive).")
    parser.add_argument("--n-boot", type=int, default=10)
    args = parser.parse_args()

    if args.bootstrap:
        check_bootstrap(n_boot=args.n_boot)
    else:
        check_checkpoint_consistency()
