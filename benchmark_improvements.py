"""
Benchmark — broader feature matrix over the Stage 4 canonical feature sets.

Sweeps all freq50-based feature combinations (±trans, ±transsummary, ±seqstats,
±metastate) across all 9 groups and writes ranked results to CSV.  Use this to
validate or challenge the two canonical sets chosen in Stage 4
(freq50+trans and freq50+transsummary+metastate).

All building blocks (feature bank, representations, CV, model) are imported
from stages/s4_classification.py so logic stays in one place.

Outputs:
  outputs/04_classification/benchmark_improvements/

Run:
  PYTHONPATH=. python analyses/benchmark_improvements.py
  PYTHONPATH=. python analyses/benchmark_improvements.py --groups OFT_s15,Ymaze_s15
  PYTHONPATH=. python analyses/benchmark_improvements.py --out-prefix my_run
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from stages.s4_classification import (
    GROUP_REPRESENTATION,
    _apply_representation,
    _assemble_feature_matrix,
    _build_group_metadata,
    build_five_group_data,
    build_session_feature_bank,
    evaluate_model,
)

OUT_DIR = Path("outputs/04_classification/benchmark_improvements")
OUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"Features .* are constant\.", category=UserWarning)
warnings.filterwarnings("ignore", message=r"invalid value encountered in divide",
                        category=RuntimeWarning)

# ── Benchmark feature matrix ──────────────────────────────────────────────────
# Full cross of freq50 base × add-on combinations × ± metastate.
# Superset of the two canonical Stage 4 sets; used to confirm the selection.

FEATURE_SPECS: list[dict[str, str]] = []
for _meta in ("", "+metastate", "+metatrans"):
    for _kind in (
        "freq50",
        "freq50+seqstats",
        "freq50+transsummary",
        "freq50+transsummary+seqstats",
        "freq50+trans",
        "freq50+trans+seqstats",
    ):
        name = _kind + _meta
        FEATURE_SPECS.append({"name": name, "kind": name})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark broader feature matrix against Stage 4 canonical sets.")
    parser.add_argument(
        "--groups", default=None,
        help="Comma-separated group names to run (default: all).")
    parser.add_argument(
        "--out-prefix", default="benchmark",
        help="Output CSV prefix (default: benchmark).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    print("Loading session feature bank ...")
    feature_bank = build_session_feature_bank()

    groups_data = build_five_group_data()
    records: list[dict] = []
    out_csv  = OUT_DIR / f"{args.out_prefix}_results.csv"
    best_csv = OUT_DIR / f"{args.out_prefix}_best_per_group.csv"

    group_filter = (
        {g.strip() for g in args.groups.split(",") if g.strip()}
        if args.groups else None
    )

    for group_name, group_tuple in groups_data.items():
        if group_filter and group_name not in group_filter:
            continue
        sessions, y, cv_type, mouse_ids = group_tuple
        group_meta = _build_group_metadata(group_name, sessions, y, mouse_ids)
        fixed_repr = GROUP_REPRESENTATION[group_name]

        for feature_spec in FEATURE_SPECS:
            result = _assemble_feature_matrix(sessions, feature_bank, feature_spec["kind"])
            if result is None:
                continue
            X_raw, block_sizes = result

            transformed = _apply_representation(
                group_name, group_meta, X_raw, cv_type, fixed_repr)
            if transformed is None:
                continue
            X_use, y_use, cv_use, groups_use = transformed

            if len(y_use) < 6 or len(np.unique(y_use)) < 2:
                continue

            # p/n filter: skip if p > 2 * n_train
            n_folds = len(y_use) if cv_use == "loo" else 5
            n_train = int(len(y_use) * (n_folds - 1) / n_folds)
            if X_use.shape[1] > 2 * n_train:
                continue

            res = evaluate_model(X_use, y_use, cv_use, groups_use, block_sizes)
            if res is None:
                continue

            records.append({
                "group_name":    group_name,
                "feature_set":   feature_spec["name"],
                "representation": fixed_repr,
                "n_samples":     len(y_use),
                "n_features":    int(X_use.shape[1]),
                "cv_type":       cv_use,
                "point_auc":     round(res["point_auc"], 4),
                "auc_mean":      round(res["auc_mean"],  4),
                "auc_std":       round(res["auc_std"],   4),
                "acc":           round(res["acc"],        4),
                "f1":            round(res["f1"],         4),
            })
            print(
                f"{group_name:12s} | {feature_spec['name']:40s} | "
                f"{fixed_repr:14s} | AUC={res['point_auc']:.3f}",
                flush=True,
            )
            pd.DataFrame(records).to_csv(out_csv, index=False)

    df = pd.DataFrame(records)
    df = df.sort_values(["group_name", "point_auc"], ascending=[True, False])
    df.to_csv(out_csv, index=False)

    if not df.empty:
        best = df.groupby("group_name", as_index=False).first()
        best.to_csv(best_csv, index=False)
        print("\nBest per group:")
        print(best[[
            "group_name", "feature_set", "representation",
            "point_auc", "auc_mean", "auc_std", "acc", "f1",
        ]].to_string(index=False))

    print("""
─────────────────────────────────────────────────────────────────
Feature set legend  (base: 5 freq50 syllables present in ≥50% sessions)
  freq50           usage frequency of the 5 freq50 syllables
  +trans           5×5 Laplace-smoothed transition submatrix (25 features)
  +transsummary    per-syllable self-transition prob + row entropy (10 features)
  +seqstats        12 session-level stats: bout length, switch rate, entropy, etc.
  +metastate       higher-order state usage from state-MoSeq (10 features)

Representation legend  (fixed per group via GROUP_REPRESENTATION in s4_classification)
  raw              each session is one sample; labels = group
  mouse_mean       average all sessions per mouse into one vector
  mouse_centered   subtract each mouse's own mean — encodes deviation from baseline
  delta_prepost    post − pre per mouse — encodes surgery-induced change

Stage 4 canonical sets (see stages/s4_classification.py):
  freq50+trans                  p=30  behaviour groups
  freq50+transsummary+metastate p=25  fiber / surgery-recovery groups
─────────────────────────────────────────────────────────────────""")

    print(f"\nSaved benchmark outputs to {OUT_DIR}/")
