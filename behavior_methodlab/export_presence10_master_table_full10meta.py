#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stages.s4_classification import STATE_MOSEQ_K, build_session_feature_bank
from stages.s5_posthoc_analysis import (
    EXCLUDED_STAGE5_SESSIONS,
    _load_sequences,
    _load_session_metadata,
    _presence_ranked_syllables_for_export,
    export_group_feature_master_variant_csv,
)


def main() -> None:
    seqs = _load_sequences()
    sessions = [s for s in seqs.keys() if s not in EXCLUDED_STAGE5_SESSIONS]
    meta_df = _load_session_metadata(sessions)
    feature_bank = build_session_feature_bank()

    syllable_ids = _presence_ranked_syllables_for_export(
        feature_bank=feature_bank,
        sessions=sessions,
        min_presence=0.10,
    )
    metastate_ids = list(range(int(STATE_MOSEQ_K)))

    out_path = export_group_feature_master_variant_csv(
        meta_df=meta_df,
        feature_bank=feature_bank,
        syllable_ids=syllable_ids,
        metastate_ids=metastate_ids,
        out_csv_name="study_group_feature_master_presence_gt10_full10meta_zscored.csv",
        out_meta_name="study_group_feature_master_presence_gt10_full10meta_zscored_metadata.json",
        syllable_block_name="presence_gt10",
        syllable_note=(
            "All feature columns are z-scored within study cohort after group-specific representation. "
            "Metastate columns are full 10-state usage proportions from the cached state-moseq fit. "
            "Syllable block uses all syllables with session presence >10%, ranked by global mean usage."
        ),
        syllable_selection_note="Presence >10% across sessions; sorted by global mean usage.",
    )

    meta_path = out_path.with_name("study_group_feature_master_presence_gt10_full10meta_zscored_metadata.json")
    payload = json.loads(meta_path.read_text())
    payload["n_selected_syllables"] = len(syllable_ids)
    payload["n_selected_metastates"] = len(metastate_ids)
    meta_path.write_text(json.dumps(payload, indent=2))
    print(out_path)
    print(meta_path)
    print("selected_syllables", syllable_ids)
    print("selected_metastates", metastate_ids)


if __name__ == "__main__":
    main()
