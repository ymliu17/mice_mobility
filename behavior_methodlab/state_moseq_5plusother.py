#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from stages.s4_classification import ALL_H5
from stages.s5_posthoc_analysis import EXCLUDED_STAGE5_SESSIONS


LOGGER = logging.getLogger("methodlab.state_moseq_5plusother")
OUTDIR = Path("outputs/methodlab/state_moseq_5plusother")
DEFAULT_KS = [5, 8, 10]
DEFAULT_ITERS = 200
DEFAULT_TRANS_KAPPA = 10.0
DEFAULT_EBS = 0.5
DEFAULT_SEED = 42
CANONICAL_FIVE = [11, 32, 37, 43, 45]


def load_sequences(all_h5: str | Path = ALL_H5) -> dict[str, np.ndarray]:
    seqs: dict[str, np.ndarray] = {}
    with h5py.File(all_h5, "r") as f:
        for sess in f.keys():
            if sess in EXCLUDED_STAGE5_SESSIONS:
                continue
            seqs[sess] = f[sess]["syllable"][:].astype(np.int32)
    return seqs


def recode_sequences(
    seqs: dict[str, np.ndarray],
    syllable_ids: list[int],
) -> tuple[dict[str, np.ndarray], dict[int, int], int]:
    mapping = {int(s): i for i, s in enumerate(syllable_ids)}
    other_state = len(syllable_ids)
    recoded = {}
    for sess, seq in seqs.items():
        z = np.full(len(seq), other_state, dtype=np.int32)
        for orig, new in mapping.items():
            z[seq == orig] = new
        recoded[sess] = z
    return recoded, mapping, other_state


def save_recoded_sequences(
    recoded: dict[str, np.ndarray],
    syllable_ids: list[int],
    mapping: dict[int, int],
    other_state: int,
    outdir: Path = OUTDIR,
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    seq_h5 = outdir / "recoded_sequences_freq50_5plusother.h5"
    meta_json = outdir / "recoded_sequences_freq50_5plusother_metadata.json"

    with h5py.File(seq_h5, "w") as f:
        f.attrs["alphabet_size"] = len(syllable_ids) + 1
        f.attrs["other_state_id"] = other_state
        for sess, seq in recoded.items():
            f.create_dataset(sess, data=seq.astype(np.int16), compression="gzip")

    payload = {
        "selection_rule": "canonical 5 shared syllables + other",
        "selected_original_syllables": syllable_ids,
        "recoded_state_mapping": {str(k): int(v) for k, v in mapping.items()},
        "other_state_id": int(other_state),
        "alphabet_labels": [f"s{sid:02d}" for sid in syllable_ids] + ["other"],
    }
    meta_json.write_text(json.dumps(payload, indent=2))
    return seq_h5, meta_json


def fit_state_moseq_model(
    recoded: dict[str, np.ndarray],
    k: int,
    outdir: Path,
    num_iters: int = DEFAULT_ITERS,
    trans_kappa: float = DEFAULT_TRANS_KAPPA,
    ebs: float = DEFAULT_EBS,
    seed: int = DEFAULT_SEED,
) -> Path:
    model_h5 = outdir / f"state_moseq_recoded_K{k}.h5"
    summary_json = outdir / f"state_moseq_recoded_K{k}_summary.json"
    if model_h5.exists():
        LOGGER.info("state-moseq cache exists, skipping K=%d: %s", k, model_h5)
        return model_h5

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.random as jr
    import state_moseq as sm
    import tqdm

    outdir.mkdir(parents=True, exist_ok=True)
    syllables_batched, mask, metadata = sm.batch(recoded, seg_length=3000)
    keys_arr, bounds_arr = metadata
    data = {
        "syllables": syllables_batched.astype(np.int32),
        "mask": mask.astype(np.float32),
    }
    hypparams = {
        "n_syllables": int(max(np.max(x) for x in recoded.values()) + 1),
        "n_states": int(k),
        "emission_base_sigma": 1.0,
        "emission_biases_sigma": float(ebs),
        "trans_kappa": float(trans_kappa),
        "trans_beta": 1.0,
        "emission_gd_iters": 50,
        "emission_gd_lr": 0.01,
    }

    rng = jr.PRNGKey(seed)
    rng, init_rng = jr.split(rng)
    LOGGER.info("Initialising state-moseq for K=%d", k)
    params = sm.hhmm_efficient.initialize_params(data, hypparams, seed=init_rng)
    rng, s0 = jr.split(rng)
    states, _ = sm.hhmm_efficient.resample_states(s0, data, params)
    for _ in tqdm.trange(num_iters, desc=f"state-moseq Gibbs K={k}"):
        rng, s1, s2 = jr.split(rng, 3)
        params, _ = sm.hhmm_efficient.resample_params(s1, data, states, hypparams, params)
        states, _ = sm.hhmm_efficient.resample_states(s2, data, params)

    states_np = np.array(states)
    per_session = sm.unbatch(states_np, keys_arr, bounds_arr)
    all_z = np.concatenate(list(per_session.values()))
    counts = np.bincount(all_z, minlength=k)
    usage = (counts / counts.sum()).astype(float)

    with h5py.File(model_h5, "w") as f:
        f.attrs["K"] = int(k)
        f.attrs["n_syllables"] = int(hypparams["n_syllables"])
        f.attrs["num_iters"] = int(num_iters)
        f.attrs["trans_kappa"] = float(trans_kappa)
        f.attrs["emission_biases_sigma"] = float(ebs)
        for sess, z in per_session.items():
            f.create_dataset(sess, data=z.astype(np.int16), compression="gzip")

    summary_json.write_text(json.dumps({
        "K": int(k),
        "n_syllables": int(hypparams["n_syllables"]),
        "num_iters": int(num_iters),
        "trans_kappa": float(trans_kappa),
        "emission_biases_sigma": float(ebs),
        "global_state_usage": [float(x) for x in usage.tolist()],
    }, indent=2))
    LOGGER.info("Saved K=%d model to %s", k, model_h5)
    return model_h5


def run(
    ks: list[int] | None = None,
    num_iters: int = DEFAULT_ITERS,
) -> None:
    ks = ks or DEFAULT_KS
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seqs = load_sequences()
    recoded, mapping, other_state = recode_sequences(seqs, CANONICAL_FIVE)
    seq_h5, meta_json = save_recoded_sequences(recoded, CANONICAL_FIVE, mapping, other_state, OUTDIR)

    selection_summary = {
        "selected_original_syllables": CANONICAL_FIVE,
        "recoded_sequence_h5": str(seq_h5),
        "recoded_metadata_json": str(meta_json),
        "requested_Ks": ks,
    }
    (OUTDIR / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2))

    for k in ks:
        fit_state_moseq_model(recoded, k=k, outdir=OUTDIR, num_iters=num_iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit state-moseq on canonical 5-syllable + other recoded sequences.")
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS, help="List of K values to fit")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Number of Gibbs iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable info logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(ks=args.ks, num_iters=args.iters)


if __name__ == "__main__":
    main()
