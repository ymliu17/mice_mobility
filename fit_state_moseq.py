"""
Fit a hierarchical HMM (state-moseq) on top of model_all syllable sequences.

The HHMM models K meta-states, each characterised by a distinct syllable
transition matrix.  Per-frame meta-state assignments are saved alongside the
syllable sequences so Stage 4E can use them as features.

Output: outputs/03_moseq/kpms_project_all/model_all/state_moseq_K{K}.h5
        keys = session names, values = int array of meta-state indices (T,)

Run:
    PYTHONPATH=. /home/yangliu/miniconda3/envs/mice/bin/python fit_state_moseq.py
    PYTHONPATH=. /home/yangliu/miniconda3/envs/mice/bin/python fit_state_moseq.py --K 5
"""

from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import jax
import jax.random as jr
import state_moseq as sm

jax.config.update("jax_enable_x64", True)

SYLLABLE_H5  = "outputs/03_moseq/kpms_project_all/model_all/results.h5"
OUT_DIR      = Path("outputs/03_moseq/kpms_project_all/model_all")
NUM_SYLLABLES = 50
NUM_ITERS     = 200
SEG_LENGTH    = 3000   # batch into segments of this length to avoid OOM


def load_syllable_sequences(h5_path: str) -> dict[str, np.ndarray]:
    seqs = {}
    with h5py.File(h5_path, "r") as f:
        for s in f.keys():
            seqs[s] = f[s]["syllable"][:].astype(np.int32)
    return seqs


def fit_state_moseq(K: int = 10, seed: int = 42,
                    trans_kappa: float = 1.0,
                    emission_biases_sigma: float = 0.5) -> None:
    tag = f"K{K}_kappa{trans_kappa}_ebs{emission_biases_sigma}"
    out_path = OUT_DIR / f"state_moseq_{tag}.h5"
    if out_path.exists():
        print(f"Cache already exists: {out_path}  (delete to refit)")
        return

    print(f"\nFitting state-moseq  {tag}  iters={NUM_ITERS}")
    print(f"Loading syllable sequences from {SYLLABLE_H5} ...")
    seqs = load_syllable_sequences(SYLLABLE_H5)
    print(f"  {len(seqs)} sessions loaded")

    # Batch sequences into padded array
    syllables_batched, mask, metadata = sm.batch(seqs, seg_length=SEG_LENGTH)
    keys_arr, bounds_arr = metadata
    print(f"  Batched shape: {syllables_batched.shape}  mask sum: {mask.sum():.0f}")

    data = {
        "syllables": syllables_batched.astype(np.int32),
        "mask":      mask.astype(np.float32),
    }

    hypparams = {
        "n_syllables":           NUM_SYLLABLES,
        "n_states":              K,
        "emission_base_sigma":   1.0,
        "emission_biases_sigma": emission_biases_sigma,
        "trans_kappa":           trans_kappa,
        "trans_beta":            1.0,
        "emission_gd_iters":     50,
        "emission_gd_lr":        0.01,
    }

    rng = jr.PRNGKey(seed)
    rng, init_rng = jr.split(rng)
    print("Initialising parameters ...")
    init_params = sm.hhmm_efficient.initialize_params(data, hypparams, seed=init_rng)

    # fit_gibbs in state-moseq 0.0.0 has an arg-order bug in its resample_params
    # call — run the Gibbs loop manually with the correct argument order.
    print(f"Running Gibbs sampling ({NUM_ITERS} iters) ...")
    import tqdm
    params = init_params
    rng, s0 = jr.split(rng)
    states, _ = sm.hhmm_efficient.resample_states(s0, data, params)
    log_joints = []
    for _ in tqdm.trange(NUM_ITERS):
        rng, s1, s2 = jr.split(rng, 3)
        # correct order: (seed, data, states, hypparams, params)
        params, _ = sm.hhmm_efficient.resample_params(s1, data, states, hypparams, params)
        states, mll = sm.hhmm_efficient.resample_states(s2, data, params)
        log_joints.append(float(mll))

    # states: (N_segs, T_seg) int array — unbatch to per-session sequences
    states_np = np.array(states)
    per_session = sm.unbatch(states_np, keys_arr, bounds_arr)

    print(f"\nMeta-state usage (global):")
    all_z = np.concatenate(list(per_session.values()))
    counts = np.bincount(all_z, minlength=K)
    for k in range(K):
        print(f"  state {k:2d}: {counts[k]/counts.sum():.1%}")

    print(f"\nTag: {tag}")
    print(f"Saving to {out_path} ...")
    with h5py.File(out_path, "w") as f:
        f.attrs["K"]          = K
        f.attrs["n_syllables"] = NUM_SYLLABLES
        f.attrs["num_iters"]  = NUM_ITERS
        f.attrs["log_joint_final"] = log_joints[-1] if log_joints else float("nan")
        for sess, z in per_session.items():
            f.create_dataset(sess, data=z.astype(np.int16), compression="gzip")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",    type=int,   default=10,  help="Number of meta-states")
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument("--trans-kappa",          type=float, default=1.0,
                        help="Meta-state stickiness (lower = more switching)")
    parser.add_argument("--emission-biases-sigma", type=float, default=0.5,
                        help="How distinct meta-state emission matrices can be")
    args = parser.parse_args()
    fit_state_moseq(K=args.K, seed=args.seed,
                    trans_kappa=args.trans_kappa,
                    emission_biases_sigma=args.emission_biases_sigma)
