#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from behavior_methodlab.feature_spaces import (
    build_graph_core5_features,
    build_usage_graph_hybrid,
    feature_registry,
    load_unified_session_table,
    metadata_columns,
)
from behavior_methodlab.model_zoo import model_registry


OUTDIR = Path("outputs/methodlab")


def run(input_csv: str = "outputs/05_stage5/unified_model_session_table.csv") -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_unified_session_table(input_csv)
    meta_cols = metadata_columns(df)

    graph_df = build_graph_core5_features(df)
    hybrid_df = build_usage_graph_hybrid(df)
    registry_df = feature_registry(df)
    model_df = model_registry()

    graph_export = df[meta_cols].copy()
    graph_export = graph_export.join(graph_df)
    graph_export.to_csv(OUTDIR / "graph_core5_feature_table.csv", index=False)

    hybrid_export = df[meta_cols].copy()
    hybrid_export = hybrid_export.join(hybrid_df)
    hybrid_export.to_csv(OUTDIR / "usage_graph_hybrid_feature_table.csv", index=False)

    registry_df.to_csv(OUTDIR / "feature_space_registry.csv", index=False)
    model_df.to_csv(OUTDIR / "model_registry.csv", index=False)

    manifest = {
        "input_csv": input_csv,
        "n_samples": int(len(df)),
        "n_metadata_columns": int(len(meta_cols)),
        "feature_spaces_exported": {
            "graph_core5": int(graph_df.shape[1]),
            "usage_graph_hybrid": int(hybrid_df.shape[1]),
        },
        "next_priorities": [
            "Benchmark graph_core5 and usage_graph_hybrid against the current Stage 5 baseline.",
            "Add motif_core5 from raw session sequences.",
            "Compare elasticnet_logreg, linear_svm, and xgboost_or_gbdt.",
        ],
    }
    (OUTDIR / "methodlab_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the standalone behavioral method lab.")
    parser.add_argument(
        "--input",
        default="outputs/05_stage5/unified_model_session_table.csv",
        help="Session-level unified table used as the method-lab input.",
    )
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
