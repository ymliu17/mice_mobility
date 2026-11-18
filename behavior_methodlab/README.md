# Behavior Method Lab

This folder is a standalone sandbox for behavioral feature-engineering and
classification experiments that should stay separate from the formal
`stages/` pipeline.

It is designed for methods exploration around:

- graph features from the 5-syllable transition graph
- higher-order grammar and motif features
- alternative classifiers beyond the current Stage 4/5 logistic models
- domain-robust comparisons that are easier to iterate without touching the
  stage framework

## Principles

- Reuse cached Stage 5 tables as inputs whenever possible.
- Write all new outputs to `outputs/methodlab/`.
- Keep experimental code here until it proves useful enough to upstream into a
  formal stage.
- Prefer interpretable feature spaces before deep models.

## Current input table

Primary source table:

- `outputs/05_stage5/unified_model_session_table.csv`

This gives one session per row with:

- session metadata
- `freq50_*` usage features
- `trans_*` 5x5 transition features
- `transsummary_*` self-transition and row-entropy features
- `metastate_usage_*` features

## First-pass feature spaces

The initial framework registers these feature spaces:

- `baseline_current`
  Stage-5-style raw session features.
- `graph_core5`
  Graph-theory summary features derived from the 5-syllable transition graph.
- `usage_graph_hybrid`
  `freq50_*` usage + `graph_core5`.

Planned next additions:

- `motif_core5`
  Higher-order sequence motifs from raw syllable sequences.
- `prototype_program`
  Prototype-level usage/transition features.
- `kinematic_conditioned`
  Session-level deviations in syllable-conditioned kinematics.

## First-pass model families

The method lab tracks these benchmark model families:

- `elasticnet_logreg`
- `linear_svm`
- `xgboost_or_gbdt`
- `graph_kernel_svm` (planned)
- `gnn` (deferred; only if graph features clearly help first)

## Quick start

Generate the feature-space manifest and the first graph feature table:

```bash
python behavior_methodlab/run_methodlab.py
```

Run the first-pass benchmark across feature spaces and model families:

```bash
python behavior_methodlab/benchmark_methods.py
```

Fit new state-moseq models on the `12-syllable + other` recoded alphabet:

```bash
python behavior_methodlab/state_moseq_12plusother.py --ks 5 8 10 --iters 200 --verbose
```

Outputs:

- `outputs/methodlab/methodlab_manifest.json`
- `outputs/methodlab/feature_space_registry.csv`
- `outputs/methodlab/graph_core5_feature_table.csv`
- `outputs/methodlab/usage_graph_hybrid_feature_table.csv`
- `outputs/methodlab/benchmarks/benchmark_summary.csv`
- `outputs/methodlab/benchmarks/benchmark_overview.json`
- `outputs/methodlab/state_moseq_12plusother/recoded_sequences_presence_gt10_12plusother.h5`
- `outputs/methodlab/state_moseq_12plusother/state_moseq_recoded_K5.h5`
- `outputs/methodlab/state_moseq_12plusother/state_moseq_recoded_K8.h5`
- `outputs/methodlab/state_moseq_12plusother/state_moseq_recoded_K10.h5`

## Suggested development order

1. Benchmark `baseline_current` vs `graph_core5` vs `usage_graph_hybrid`.
2. Add `motif_core5` from raw sequences.
3. Compare `elasticnet_logreg`, `linear_svm`, and tree-based models.
4. Only consider graph kernels or GNN if graph summaries clearly improve signal.
