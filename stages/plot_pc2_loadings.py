"""
PC2 Top Loadings & Profile Plots
=================================
Recomputes stable loadings (Parts 1+2 of s4_5_pca_analysis.py) and generates:
  1. Top-30 loading bar chart for PC2 (coloured by feature type)
  2. PC2 score distribution by env_context + condition (violin + strip)
  3. PC2 score by study_cohort (box)
  4. Syllable-level loading heatmap: freq vs self_trans vs entropy contributions per syllable

Usage:
    PYTHONPATH=. /home/yangliu/miniconda3/envs/mice/bin/python stages/plot_pc2_loadings.py
"""

from __future__ import annotations
import re, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_PATH = Path("outputs/05_stage5/"
                 "study_group_feature_master_presence_gt10_full10meta_zscored.csv")
OUT_DIR   = Path("outputs/05_stage5/pca_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ITERATIONS = 5000
RANDOM_SEED  = 123
TOP_N        = 30   # top loadings to show

# ── Part 1: data prep & residualisation ────────────────────────────────────────
print("Part 1: loading & residualising ...")
full_data = pd.read_csv(DATA_PATH)
all_cols  = full_data.columns.tolist()

freq_features       = [c for c in all_cols if re.match(r'^freq50_s',                   c)]
self_trans_features = [c for c in all_cols if re.match(r'^transsummary_self_trans_s',  c)]
entropy_features    = [c for c in all_cols if re.match(r'^transsummary_row_entropy_s', c)]
core_features       = freq_features + self_trans_features + entropy_features
print(f"  freq50:{len(freq_features)}  self_trans:{len(self_trans_features)}  entropy:{len(entropy_features)}")

analysis_data     = full_data.dropna(subset=["label_binary"]).copy()
residualized_data = analysis_data.copy()
for feature in core_features:
    m = smf.ols(f"Q('{feature}') ~ C(study_cohort) + C(env_context)",
                data=analysis_data, missing="drop").fit()
    residualized_data[feature] = m.resid

final_clean_data = residualized_data.dropna(subset=core_features).copy()
pca_input_matrix = final_clean_data[core_features].values
print(f"  {len(final_clean_data)} clean rows")

# ── Part 2: robust PCA ─────────────────────────────────────────────────────────
print(f"Part 2: robust PCA ({N_ITERATIONS} iters) ...")
rng = np.random.default_rng(RANDOM_SEED)
n_features = pca_input_matrix.shape[1]
n_strata   = final_clean_data.groupby(["env_context", "study_cohort"]).ngroups
n_pcs_keep = min(n_strata, n_features)
print(f"  n_strata={n_strata}, keeping {n_pcs_keep} PCs")

scaler_ref = StandardScaler()
ref_scaled = scaler_ref.fit_transform(pca_input_matrix)
ref_pca    = PCA(n_components=n_pcs_keep)
ref_pca.fit(ref_scaled)
reference_loadings = ref_pca.components_.T

aligned_loadings = np.zeros((n_features, n_pcs_keep))
for i in tqdm(range(N_ITERATIONS), desc="  Bootstrap PCA"):
    subsample = (final_clean_data
                 .groupby(["env_context", "study_cohort"], group_keys=False)
                 .apply(lambda g: g.sample(n=1, random_state=rng.integers(1_000_000_000))))
    X_sub = subsample[core_features].values
    X_sub_scaled = StandardScaler().fit_transform(X_sub)
    iter_pca = PCA(n_components=n_pcs_keep)
    iter_pca.fit(X_sub_scaled)
    iter_loadings = iter_pca.components_.T
    for pc_idx in range(n_pcs_keep):
        if np.dot(reference_loadings[:, pc_idx], iter_loadings[:, pc_idx]) < 0:
            iter_loadings[:, pc_idx] *= -1
    aligned_loadings += iter_loadings

stable_loadings = aligned_loadings / N_ITERATIONS
np.save(OUT_DIR / "stable_loadings.npy", stable_loadings)
np.save(OUT_DIR / "core_features.npy",   np.array(core_features))
print("  Stable loadings saved.")

# ── PC scores ──────────────────────────────────────────────────────────────────
scaler_full   = StandardScaler()
scaled_matrix = scaler_full.fit_transform(pca_input_matrix)
pc_scores     = scaled_matrix @ stable_loadings

modeling_data = final_clean_data[["label_binary", "study_cohort", "env_context",
                                   "abnormal_condition"]].copy()
for i in range(min(4, n_pcs_keep)):
    modeling_data[f"PC{i+1}"] = pc_scores[:, i]
modeling_data["label_binary"] = modeling_data["label_binary"].astype(float)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Top-30 PC2 loadings bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: PC2 top loadings ...")
pc2_load = pd.Series(stable_loadings[:, 1], index=core_features, name="loading")
pc2_top  = pc2_load.reindex(pc2_load.abs().nlargest(TOP_N).index)

def feat_type(f):
    if "row_entropy" in f: return "Entropy"
    if "self_trans"  in f: return "Self-transition"
    return "Frequency"

def feat_syll(f):
    m = re.search(r"s(\d+)$", f)
    return int(m.group(1)) if m else -1

colors = {"Frequency": "#E07B39", "Self-transition": "#4472C4", "Entropy": "#70AD47"}
feat_colors = [colors[feat_type(f)] for f in pc2_top.index]
short_labels = [f"{feat_type(f)[0]}s{feat_syll(f)}" for f in pc2_top.index]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(len(pc2_top)), pc2_top.values, color=feat_colors, edgecolor="white", lw=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(pc2_top)))
ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=8)
ax.set_ylabel("PC2 Loading (stable)", fontsize=11)
ax.set_title(f"PC2 — Top {TOP_N} loadings by absolute magnitude", fontsize=12)
patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
ax.legend(handles=patches, fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "pc2_top_loadings.png", dpi=160)
plt.close(fig)
print(f"  → {OUT_DIR / 'pc2_top_loadings.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Per-syllable loading profile (freq / self_trans / entropy) for PC2
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: PC2 per-syllable loading profile ...")

# Extract syllable numbers present in all three feature types
syll_ids = sorted(set(feat_syll(f) for f in core_features if feat_syll(f) >= 0))

load_by_syll = []
for s in syll_ids:
    row = {"syllable": s}
    for ftype, prefix in [("Frequency",    "freq50_s"),
                           ("Self-trans",   "transsummary_self_trans_s"),
                           ("Entropy",      "transsummary_row_entropy_s")]:
        fname = f"{prefix}{s}"
        row[ftype] = pc2_load.get(fname, 0.0)
    load_by_syll.append(row)

sdf = pd.DataFrame(load_by_syll).set_index("syllable")

# Sort syllables by total absolute loading on PC2
sdf["total_abs"] = sdf.abs().sum(axis=1)
sdf = sdf.sort_values("total_abs", ascending=False).head(25)
sdf = sdf.drop(columns="total_abs")

fig, ax = plt.subplots(figsize=(14, 4))
x = np.arange(len(sdf))
w = 0.28
ax.bar(x - w,   sdf["Frequency"],   w, color="#E07B39", label="Frequency",       alpha=0.9)
ax.bar(x,       sdf["Self-trans"],  w, color="#4472C4", label="Self-transition",  alpha=0.9)
ax.bar(x + w,   sdf["Entropy"],     w, color="#70AD47", label="Entropy",          alpha=0.9)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"s{i}" for i in sdf.index], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("PC2 Loading", fontsize=10)
ax.set_title("PC2 Loading Profile — top 25 syllables (by |freq|+|ST|+|entropy|)", fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "pc2_syllable_profile.png", dpi=160)
plt.close(fig)
print(f"  → {OUT_DIR / 'pc2_syllable_profile.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: PC2 score distribution by env_context × label_binary
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: PC2 score distribution by context × condition ...")

plot_df = modeling_data.copy()
plot_df["Condition"] = plot_df["label_binary"].map({1.0: "Abnormal", 0.0: "Control"})
plot_df["env_context"] = plot_df["env_context"].str.replace(
    "Food intake during habituation", "Food/AST", regex=False)

fig, ax = plt.subplots(figsize=(10, 4.5))
palette = {"Control": "#5B9BD5", "Abnormal": "#ED7D31"}
order = sorted(plot_df["env_context"].unique())

sns.violinplot(data=plot_df, x="env_context", y="PC2", hue="Condition",
               order=order, palette=palette, split=False,
               inner=None, linewidth=0.8, alpha=0.55, ax=ax)
sns.stripplot(data=plot_df, x="env_context", y="PC2", hue="Condition",
              order=order, palette=palette, dodge=True,
              size=3, alpha=0.6, jitter=True, ax=ax, legend=False)

ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("Behavioral Context", fontsize=10)
ax.set_ylabel("PC2 Score", fontsize=10)
ax.set_title("PC2 Score by Context and Condition", fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Condition", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
fig.savefig(OUT_DIR / "pc2_score_by_context.png", dpi=160)
plt.close(fig)
print(f"  → {OUT_DIR / 'pc2_score_by_context.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: PC2 score by study_cohort
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 4: PC2 by study_cohort × condition ...")

fig, ax = plt.subplots(figsize=(8, 4))
cohort_order = sorted(plot_df["study_cohort"].unique())
sns.boxplot(data=plot_df, x="study_cohort", y="PC2", hue="Condition",
            order=cohort_order, palette=palette, linewidth=1.0,
            flierprops=dict(marker=".", ms=3), ax=ax)
sns.stripplot(data=plot_df, x="study_cohort", y="PC2", hue="Condition",
              order=cohort_order, palette=palette, dodge=True,
              size=3, alpha=0.5, jitter=True, ax=ax, legend=False)
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("Study Cohort"); ax.set_ylabel("PC2 Score")
ax.set_title("PC2 Score by Study Cohort and Condition", fontsize=11)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Condition", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(OUT_DIR / "pc2_score_by_cohort.png", dpi=160)
plt.close(fig)
print(f"  → {OUT_DIR / 'pc2_score_by_cohort.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Print top PC2 loadings summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Top 20 PC2 loadings ──")
top20 = pc2_load.reindex(pc2_load.abs().nlargest(20).index)
for feat, val in top20.items():
    ftype = feat_type(feat)
    syll  = feat_syll(feat)
    print(f"  {val:+.4f}  s{syll:>2}  {ftype:<18}  ({feat})")

print(f"\nAll PC2 plots saved to {OUT_DIR}")
