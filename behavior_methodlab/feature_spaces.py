from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/05_stage5")
INPUT = ROOT / "unified_model_session_table.csv"


def load_unified_session_table(path: str | Path = INPUT) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Unified session table not found: {path}")
    return pd.read_csv(path)


def metadata_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if not (
            c.startswith("freq50_")
            or c.startswith("trans_")
            or c.startswith("transsummary_")
            or c.startswith("metastate_usage_")
        )
    ]


def baseline_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("freq50_")
        or c.startswith("trans_")
        or c.startswith("transsummary_")
        or c.startswith("metastate_usage_")
    ]


def _syllable_ids(df: pd.DataFrame) -> list[str]:
    ids = []
    for col in df.columns:
        if col.startswith("freq50_s"):
            ids.append(col.split("_s", 1)[1])
    return sorted(ids, key=lambda x: int(x))


def build_graph_core5_features(df: pd.DataFrame) -> pd.DataFrame:
    syllables = _syllable_ids(df)
    out = pd.DataFrame(index=df.index)

    weights = {}
    for src in syllables:
        for dst in syllables:
            col = f"trans_s{src}_to_s{dst}"
            if col in df.columns:
                weights[(src, dst)] = df[col].astype(float).to_numpy()

    if not weights:
        raise ValueError("No 5-syllable transition matrix columns found.")

    # Node-level directed strength summaries.
    for sid in syllables:
        outgoing = np.zeros(len(df), dtype=float)
        incoming = np.zeros(len(df), dtype=float)
        for other in syllables:
            outgoing += weights[(sid, other)]
            incoming += weights[(other, sid)]
        out[f"graph_out_strength_s{sid}"] = outgoing
        out[f"graph_in_strength_s{sid}"] = incoming
        out[f"graph_strength_delta_s{sid}"] = outgoing - incoming
        self_col = f"trans_s{sid}_to_s{sid}"
        if self_col in df.columns:
            out[f"graph_self_loop_s{sid}"] = df[self_col].astype(float)
        ent_col = f"transsummary_row_entropy_s{sid}"
        if ent_col in df.columns:
            out[f"graph_row_entropy_s{sid}"] = df[ent_col].astype(float)

    # Global summaries from the 5x5 directed graph.
    offdiag_sum = np.zeros(len(df), dtype=float)
    reciprocity_num = np.zeros(len(df), dtype=float)
    cycle_2_mass = np.zeros(len(df), dtype=float)
    feedforward_mass = np.zeros(len(df), dtype=float)

    for i, src in enumerate(syllables):
        for j, dst in enumerate(syllables):
            w_ij = weights[(src, dst)]
            if src != dst:
                offdiag_sum += w_ij
                reciprocity_num += np.minimum(w_ij, weights[(dst, src)])
                cycle_2_mass += w_ij * weights[(dst, src)]
            for k, mid in enumerate(syllables):
                if len({src, dst, mid}) == 3:
                    feedforward_mass += weights[(src, dst)] * weights[(dst, mid)]

    self_loop_mass = np.zeros(len(df), dtype=float)
    total_mass = np.zeros(len(df), dtype=float)
    for src in syllables:
        for dst in syllables:
            total_mass += weights[(src, dst)]
            if src == dst:
                self_loop_mass += weights[(src, dst)]

    out["graph_total_mass"] = total_mass
    out["graph_self_loop_mass"] = self_loop_mass
    out["graph_offdiag_mass"] = offdiag_sum
    out["graph_reciprocity"] = np.divide(
        reciprocity_num,
        offdiag_sum,
        out=np.zeros(len(df), dtype=float),
        where=offdiag_sum > 0,
    )
    out["graph_cycle2_mass"] = cycle_2_mass
    out["graph_feedforward_mass"] = feedforward_mass
    out["graph_max_usage"] = df[[f"freq50_s{sid}" for sid in syllables]].max(axis=1)
    out["graph_usage_entropy_proxy"] = df[
        [f"freq50_s{sid}" for sid in syllables]
    ].apply(_safe_entropy, axis=1)

    return out


def build_usage_graph_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    usage_cols = [c for c in df.columns if c.startswith("freq50_")]
    return pd.concat(
        [
            df[usage_cols].copy(),
            build_graph_core5_features(df),
        ],
        axis=1,
    )


def feature_registry(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_cols = baseline_feature_columns(df)
    graph_df = build_graph_core5_features(df)
    hybrid_df = build_usage_graph_hybrid(df)

    rows.append({
        "feature_space": "baseline_current",
        "n_features": len(baseline_cols),
        "source": "stage5_unified_session_table",
        "notes": "Raw Stage 5 session features: usage + transitions + transsummary + metastate.",
    })
    rows.append({
        "feature_space": "graph_core5",
        "n_features": graph_df.shape[1],
        "source": "derived_from_5_syllable_transition_graph",
        "notes": "Directed graph summaries from the 5x5 transition matrix.",
    })
    rows.append({
        "feature_space": "usage_graph_hybrid",
        "n_features": hybrid_df.shape[1],
        "source": "usage_plus_graph_core5",
        "notes": "Core5 usage plus graph summaries.",
    })

    rows.extend([
        {
            "feature_space": "motif_core5",
            "n_features": np.nan,
            "source": "planned_from_raw_syllable_sequences",
            "notes": "Reserved for 2-step / 3-step motifs and dwell-time grammar features.",
        },
        {
            "feature_space": "prototype_program",
            "n_features": np.nan,
            "source": "planned_from_prototype_mapping",
            "notes": "Reserved for locomotion/turning/resting/investigation/grooming-like program summaries.",
        },
        {
            "feature_space": "kinematic_conditioned",
            "n_features": np.nan,
            "source": "planned_from_syllable_conditioned_kinematics",
            "notes": "Reserved for within-syllable kinematic style features.",
        },
    ])

    return pd.DataFrame(rows)


def get_feature_space(df: pd.DataFrame, feature_space: str) -> pd.DataFrame:
    if feature_space == "baseline_current":
        return df[baseline_feature_columns(df)].copy()
    if feature_space == "graph_core5":
        return build_graph_core5_features(df)
    if feature_space == "usage_graph_hybrid":
        return build_usage_graph_hybrid(df)
    raise ValueError(f"Unknown or not-yet-implemented feature space: {feature_space}")


def _safe_entropy(row: pd.Series) -> float:
    vals = row.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    return float(-(p * np.log2(p)).sum())
