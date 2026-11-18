#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages.s4_classification import build_session_feature_bank
from stages.s5_posthoc_analysis import (
    EXCLUDED_STAGE5_SESSIONS,
    _load_sequences,
    _load_session_metadata,
    export_group_feature_master_variant_csv,
)


def _load_custom_metastate_usage(h5_path: str | Path) -> tuple[dict[str, np.ndarray], int]:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Custom metastate model not found: {h5_path}")
    usage = {}
    with h5py.File(h5_path, "r") as f:
        k = int(f.attrs["K"])
        for sess in f.keys():
            z = f[sess][:].astype(int)
            u = np.bincount(z, minlength=k)[:k].astype(float) / max(len(z), 1)
            usage[sess] = u
    return usage, k


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a group master table using a custom state-moseq metastate model.")
    parser.add_argument("--state-h5", required=True, help="Path to custom state-moseq .h5")
    parser.add_argument("--syllables", nargs="+", type=int, required=True, help="Syllable ids used for usage/trans/transsummary blocks")
    parser.add_argument("--out-csv", required=True, help="Output CSV filename under outputs/05_stage5/")
    parser.add_argument("--out-meta", required=True, help="Output metadata JSON filename under outputs/05_stage5/")
    parser.add_argument("--syllable-block-name", required=True, help="Name for the syllable feature block")
    parser.add_argument("--selection-note", required=True, help="Description of syllable selection")
    args = parser.parse_args()

    seqs = _load_sequences()
    sessions = [s for s in seqs.keys() if s not in EXCLUDED_STAGE5_SESSIONS]
    meta_df = _load_session_metadata(sessions)
    feature_bank = build_session_feature_bank()

    custom_usage, k = _load_custom_metastate_usage(args.state_h5)
    for sess, u in custom_usage.items():
        if sess in feature_bank:
            feature_bank[sess]["metastate_usage"] = np.asarray(u, dtype=float)

    out_path = export_group_feature_master_variant_csv(
        meta_df=meta_df,
        feature_bank=feature_bank,
        syllable_ids=[int(x) for x in args.syllables],
        metastate_ids=list(range(k)),
        out_csv_name=args.out_csv,
        out_meta_name=args.out_meta,
        syllable_block_name=args.syllable_block_name,
        syllable_note=(
            "All feature columns are z-scored within study cohort after group-specific representation. "
            f"Metastate columns are usage proportions from custom state-moseq model `{args.state_h5}`."
        ),
        syllable_selection_note=args.selection_note,
    )

    meta_path = Path("outputs/05_stage5") / args.out_meta
    payload = json.loads(meta_path.read_text())
    payload["custom_state_moseq_h5"] = str(args.state_h5)
    payload["n_selected_syllables"] = len(args.syllables)
    payload["n_selected_metastates"] = int(k)
    meta_path.write_text(json.dumps(payload, indent=2))
    print(out_path)
    print(meta_path)


if __name__ == "__main__":
    main()
