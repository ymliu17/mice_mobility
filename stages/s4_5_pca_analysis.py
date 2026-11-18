"""
FINAL ANALYSIS PIPELINE: OFT-ANCHORED + FEATURE LOADING ANALYSIS
=================================================================
Python translation of stages/s4-5_pca_analysis.R

Parts:
  1. Data preparation & residualization
  2. Function for robust, stratified PCA
  3. OFT-anchored CV loop (per-fold PCA + GLMM + permutation test)
  4. Final significance robustness summary
  5. PC loading stability analysis
  6. Aggregated scree plot
  7. Analyzing feature loadings for key behavioral signatures

GLMM: uses statsmodels BinomialBayesMixedGLM (variational Bayes MAP) for a
true random-intercept logistic model. Z-scores are close but not identical to
R's glmer (which uses Laplace/REML). Falls back to plain logit on failure.

Usage:
    PYTHONPATH=. /home/yangliu/miniconda3/envs/mice/bin/python stages/s4_5_pca_analysis.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path("outputs/05_stage5/"
                 "study_group_feature_master_presence_gt10_full10meta_zscored.csv")
OUT_DIR   = Path("outputs/05_stage5/pca_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ITERATIONS    = 1000
N_PERMUTATIONS  = 999
RANDOM_SEED     = 123
MAX_PCS_TO_TEST = 4


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: DATA PREPARATION & RESIDUALISATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 1: DATA PREPARATION & RESIDUALISATION ---")

full_data = pd.read_csv(DATA_PATH)
all_cols  = full_data.columns.tolist()
freq_features       = [c for c in all_cols if re.match(r'^freq50_s',                   c)]
self_trans_features = [c for c in all_cols if re.match(r'^transsummary_self_trans_s',  c)]
entropy_features    = [c for c in all_cols if re.match(r'^transsummary_row_entropy_s', c)]
core_features       = freq_features + self_trans_features + entropy_features

print(f"  Features — freq50: {len(freq_features)}, "
      f"self_trans: {len(self_trans_features)}, entropy: {len(entropy_features)}")

analysis_data = full_data.dropna(subset=["label_binary"]).copy()
analysis_data["loco_context"] = analysis_data["env_context"].apply(
    lambda ctx: "Food_AST" if re.search(r"food intake|ast", ctx, re.IGNORECASE) else ctx
)
print(f"  Rows with label_binary: {len(analysis_data)}")

# Residualise each core feature by study_cohort + env_context
residualized_data = analysis_data.copy()
for feature in core_features:
    model = smf.ols(f"Q('{feature}') ~ C(study_cohort) + C(env_context)",
                    data=analysis_data, missing="drop").fit()
    residualized_data.loc[model.resid.index, feature] = model.resid

final_clean_data = residualized_data.dropna(subset=core_features).copy()
print(f"  Clean rows after residualisation: {len(final_clean_data)}")
print("Data preparation and residualization complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: FUNCTION FOR ROBUST, STRATIFIED PCA
# ══════════════════════════════════════════════════════════════════════════════

def get_stable_pc_loadings(data: pd.DataFrame, features: list[str],
                            n_iterations: int = 1000,
                            rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Robust PCA via stratified bootstrapping (one sample per env_context x study_cohort).
    Returns DataFrame of shape (n_features, n_pcs) with sign-aligned, averaged loadings.
    """
    print(f"    - Running robust PCA with {n_iterations} iterations...")
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_features = len(features)
    n_strata   = data.groupby(["env_context", "study_cohort"]).ngroups
    n_pcs      = min(n_strata, n_features)

    pca_input  = data[features].values
    ref_scaled = StandardScaler().fit_transform(pca_input)
    ref_pca    = PCA(n_components=n_pcs)
    ref_pca.fit(ref_scaled)
    reference_loadings = ref_pca.components_.T  # (n_features, n_pcs)

    aligned_loadings_list = []

    for _ in range(n_iterations):
        subsample = (data
                     .groupby(["env_context", "study_cohort"], group_keys=False)
                     .apply(lambda g: g.sample(n=1, random_state=rng.integers(int(1e9)))))
        X_sub = subsample[features].values

        variances  = X_sub.var(axis=0)
        valid_mask = variances > 1e-10
        if valid_mask.sum() < 2:
            continue

        X_valid   = X_sub[:, valid_mask]
        X_scaled  = StandardScaler().fit_transform(X_valid)
        n_comp    = min(n_pcs, X_valid.shape[0], X_valid.shape[1])

        iter_pca = PCA(n_components=n_comp)
        iter_pca.fit(X_scaled)

        # Expand to full feature space (zero for excluded features)
        iter_full = np.zeros((n_features, n_pcs))
        valid_idx = np.where(valid_mask)[0]
        iter_full[valid_idx, :n_comp] = iter_pca.components_.T

        # Sign alignment to reference
        for pc_idx in range(n_pcs):
            if np.dot(reference_loadings[:, pc_idx], iter_full[:, pc_idx]) < 0:
                iter_full[:, pc_idx] *= -1

        aligned_loadings_list.append(iter_full)

    if not aligned_loadings_list:
        return pd.DataFrame(reference_loadings,
                            index=features,
                            columns=[f"PC{i+1}" for i in range(n_pcs)])

    stable = np.mean(aligned_loadings_list, axis=0)
    pc_names = [f"PC{i+1}" for i in range(n_pcs)]
    return pd.DataFrame(stable, index=features, columns=pc_names)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: OFT-ANCHORED ROBUSTNESS TESTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 3: BEGINNING OFT-ANCHORED ROBUSTNESS TESTING ---")

all_contexts   = final_clean_data["loco_context"].unique().tolist()
other_contexts = [c for c in all_contexts if c != "OFT"]

summary_table      = []
all_fold_loadings  = {}   # fold_label -> DataFrame (features x PCs)
all_fold_variances = {}   # fold_label -> DataFrame

rng_main = np.random.default_rng(RANDOM_SEED)


def fit_glmm_fold(df: pd.DataFrame, pcs: list[str]) -> dict[str, float]:
    """
    GLMM logistic via BinomialBayesMixedGLM (variational Bayes MAP):
        label_binary ~ PC1 + ... + env_context, random intercept for study_cohort
    Falls back to plain logit on failure. Returns {pc: z_score}.
    """
    fixed_formula = f"label_binary ~ {' + '.join(pcs)} + env_context"
    try:
        mod = BinomialBayesMixedGLM.from_formula(
            fixed_formula,
            {"study_cohort": "0 + C(study_cohort)"},
            data=df,
        )
        res = mod.fit_map()
        fe_names = list(mod.fep_names)
        z_scores = res.fe_mean / res.fe_sd
        return {pc: float(z_scores[fe_names.index(pc)])
                for pc in pcs if pc in fe_names}
    except Exception:
        try:
            mod = smf.logit(fixed_formula, data=df).fit(disp=False, maxiter=200)
            return {pc: float(mod.tvalues[pc]) for pc in pcs if pc in mod.tvalues}
        except Exception:
            return {}


for partner_context in other_contexts:
    training_pair_label = f"OFT + {partner_context}"
    print(f"\n--- FOLD: Training on '{training_pair_label}' ---")

    train_data = final_clean_data[
        final_clean_data["loco_context"].isin(["OFT", partner_context])
    ].copy()
    print(f"  - Using {len(train_data)} observations for this fold.")

    stable_loadings_df = get_stable_pc_loadings(train_data, core_features, N_ITERATIONS, rng=rng_main)
    all_fold_loadings[training_pair_label] = stable_loadings_df

    # Project training data onto stable loadings
    train_scaled    = StandardScaler().fit_transform(train_data[core_features].values)
    stable_matrix   = stable_loadings_df.values   # (n_features, n_pcs)
    train_pc_scores = train_scaled @ stable_matrix # (n_train, n_pcs)
    pc_names_all    = stable_loadings_df.columns.tolist()
    n_pcs_total     = len(pc_names_all)

    # Variance per PC
    pc_vars   = train_pc_scores.var(axis=0)
    total_var = pc_vars.sum()
    all_fold_variances[training_pair_label] = pd.DataFrame({
        "PC":            pc_names_all,
        "fold":          training_pair_label,
        "variance":      pc_vars,
        "prop_variance": pc_vars / total_var,
    })

    # Select PCs to test (up to MAX_PCS_TO_TEST)
    pcs_to_test = pc_names_all[:min(MAX_PCS_TO_TEST, n_pcs_total)]
    if not pcs_to_test:
        print("  - WARNING: PCA failed to generate any PCs for this fold. Skipping.")
        continue
    print(f"  - PCA generated {n_pcs_total} PCs. Using: {', '.join(pcs_to_test)}")

    # Build modeling dataframe
    modeling_df = train_data[["label_binary", "study_cohort", "env_context"]].copy()
    for j, pc in enumerate(pc_names_all):
        modeling_df[pc] = train_pc_scores[:, j]
    modeling_df["label_binary"] = modeling_df["label_binary"].astype(float)

    # Fit real model
    print("    - Fitting real model and getting observed z-scores...")
    observed_stats = fit_glmm_fold(modeling_df, pcs_to_test)
    pcs_to_test = [pc for pc in pcs_to_test if pc in observed_stats]
    if not pcs_to_test:
        print("  - WARNING: No PC coefficients were estimated. Skipping fold.")
        continue

    # Permutations (within-cohort label shuffle)
    print(f"    - Running {N_PERMUTATIONS} permutations...")
    null_stats = {pc: np.full(N_PERMUTATIONS, np.nan) for pc in pcs_to_test}
    for i in tqdm(range(N_PERMUTATIONS), desc="    Permutations"):
        perm_df = modeling_df.copy()
        perm_df["label_binary"] = (
            perm_df.groupby("study_cohort")["label_binary"]
            .transform(lambda x: rng_main.permutation(x.values))
        )
        null_z = fit_glmm_fold(perm_df, pcs_to_test)
        for pc in pcs_to_test:
            if pc in null_z:
                null_stats[pc][i] = null_z[pc]

    # Permutation p-values
    for pc in pcs_to_test:
        valid    = null_stats[pc][~np.isnan(null_stats[pc])]
        n_valid  = len(valid)
        p_val    = (np.sum(np.abs(valid) >= abs(observed_stats[pc])) + 1) / (n_valid + 1)
        summary_table.append({
            "training_pair":       training_pair_label,
            "term":                pc,
            "observed_z_score":    round(observed_stats[pc], 2),
            "permutation_p_value": round(p_val, 4),
        })

    print("    - Fold complete.")

print("\n\n--- ANALYSIS COMPLETE ---")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: FINAL SIGNIFICANCE ROBUSTNESS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 4: FINAL SIGNIFICANCE ROBUSTNESS SUMMARY ---")

if summary_table:
    final_summary = (pd.DataFrame(summary_table)
                     .sort_values(["term", "training_pair"])
                     .reset_index(drop=True))
    final_summary["significance"] = final_summary["permutation_p_value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    print("\n--- Summary of PC Significance Across OFT-Anchored Folds ---")
    print(final_summary.to_string(index=False, float_format="%.4f"))
    final_summary.to_csv(OUT_DIR / "oft_anchored_significance.csv", index=False)

    # Z-score bar plot, one facet per PC
    terms = sorted(final_summary["term"].unique())
    n_terms = len(terms)
    n_cols  = 2
    n_rows  = (n_terms + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).flatten()
    for ax, term in zip(axes, terms):
        sub = final_summary[final_summary["term"] == term]
        ax.bar(range(len(sub)), sub["observed_z_score"],
               color=sns.color_palette("tab10", len(sub)))
        ax.axhline( 1.96, ls="--", color="red", lw=1)
        ax.axhline(-1.96, ls="--", color="red", lw=1)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["training_pair"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Observed Z-Score from GLMM")
        ax.set_title(term)
        ax.legend([], [], frameon=False)
    for ax in axes[n_terms:]:
        ax.set_visible(False)
    fig.suptitle("Stability of PC Z-Scores in OFT-Anchored Folds\n"
                 "PCs are derived from 'OFT + partner context'")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "oft_anchored_zscores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'oft_anchored_zscores.png'}")
else:
    print("No results to summarize.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: PC LOADING STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 5: PC LOADING STABILITY ANALYSIS ---")

for pc_to_check in ["PC1", "PC2", "PC3", "PC4"]:
    valid_folds = [f for f, df in all_fold_loadings.items()
                   if pc_to_check in df.columns]
    if len(valid_folds) < 2:
        print(f"\nSkipping stability analysis for {pc_to_check}: "
              "Not present in enough folds.")
        continue

    loadings_matrix = np.column_stack(
        [all_fold_loadings[f][pc_to_check].values for f in valid_folds]
    )
    corr_matrix = pd.DataFrame(
        np.corrcoef(loadings_matrix.T),
        index=valid_folds, columns=valid_folds,
    )
    print(f"\n\n--- Correlation of {pc_to_check} Loadings Across OFT-Anchored Folds ---")
    print(corr_matrix.round(3).to_string())

    fig, ax = plt.subplots(figsize=(max(4, len(valid_folds)),
                                    max(3, len(valid_folds) - 1)))
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".3f",
                cmap=sns.diverging_palette(10, 220, as_cmap=True),
                vmin=-1, vmax=1)
    ax.set_title(f"Stability of {pc_to_check} Definition Across OFT-Anchored Folds")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"loading_stability_{pc_to_check}.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / f'loading_stability_{pc_to_check}.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: AGGREGATED SCREE PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 6: AGGREGATED SCREE PLOT ---")

if all_fold_variances:
    variance_df = pd.concat(all_fold_variances.values(), ignore_index=True)
    aggregated  = (variance_df
                   .groupby("PC")["prop_variance"]
                   .agg(avg_prop_variance="mean", sd_prop_variance="std")
                   .reset_index())
    aggregated["PC_num"] = aggregated["PC"].str.replace("PC", "").astype(int)
    aggregated = aggregated.sort_values("PC_num")
    aggregated["cumulative_variance"] = aggregated["avg_prop_variance"].cumsum()
    top10 = aggregated.head(10)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(top10["PC"], top10["avg_prop_variance"],
            color="steelblue", alpha=0.8)
    ax1.errorbar(top10["PC"], top10["avg_prop_variance"],
                 yerr=top10["sd_prop_variance"].fillna(0),
                 fmt="none", color="gray", capsize=3, elinewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(top10["PC"], top10["cumulative_variance"], "o-",
             color="firebrick", lw=2, ms=5)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Avg. Proportion of Variance Explained")
    ax2.set_ylabel("Cumulative Proportion")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_title("Aggregated Scree Plot Across OFT-Anchored Folds\n"
                  "Bars show average variance (±SD); red line shows cumulative average.")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aggregated_scree_plot.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'aggregated_scree_plot.png'}")

    print("\n--- Aggregated Variance Explained Table ---")
    print(top10.to_string(index=False, float_format="%.3f"))
else:
    print("No variance information to plot.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7: ANALYZING FEATURE LOADINGS FOR KEY BEHAVIORAL SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 7: INTERPRETING THE KEY BEHAVIORAL SIGNATURES ---")

signature_A_folds = ["OFT + Food_AST", "OFT + OFT_DUO"]
signature_B_fold  = "OFT + Y maze"

# ── Signature A: Stable PC1 ────────────────────────────────────────────────────
print("\n\n--- Analysis of Signature A: Stable PC1 "
      "(from OFT+Food_AST & OFT+OFT_DUO) ---")

valid_A_folds = [f for f in signature_A_folds if f in all_fold_loadings]
if valid_A_folds:
    pc1_df = pd.DataFrame(
        {fold: all_fold_loadings[fold]["PC1"] for fold in valid_A_folds}
    )
    pc1_df["mean_loading"] = pc1_df[valid_A_folds].mean(axis=1)
    pc1_df["sd_loading"]   = pc1_df[valid_A_folds].std(axis=1)
    loadings_summary_A = (pc1_df[["mean_loading", "sd_loading"]]
                          .reset_index()
                          .rename(columns={"index": "feature"})
                          .assign(abs_mean=lambda d: d["mean_loading"].abs())
                          .sort_values("abs_mean", ascending=False)
                          .drop(columns="abs_mean")
                          .reset_index(drop=True))

    print("\nTop 10 most important AND stable features for Signature A (PC1):")
    print(loadings_summary_A.head(10).to_string(index=False, float_format="%.3f"))

    top20_A = loadings_summary_A.head(20)
    fig, ax = plt.subplots(figsize=(6, 7))
    colors = ["steelblue" if v > 0 else "coral"
              for v in top20_A["mean_loading"]]
    ax.barh(top20_A["feature"], top20_A["mean_loading"],
            color=colors, xerr=top20_A["sd_loading"],
            error_kw={"capsize": 3})
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Average Loading")
    ax.set_title("Signature A: Feature Loadings for Stable PC1\n"
                 "Average of 'OFT+Food_AST' & 'OFT+OFT_DUO' folds. Error bars show SD.")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "signature_A_pc1_loadings.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'signature_A_pc1_loadings.png'}")
else:
    print("Could not find loading data for Signature A folds.")


# ── Signature B: Predictive PC2 ───────────────────────────────────────────────
print("\n\n--- Analysis of Signature B: Predictive PC2 (from OFT + Y maze) ---")

if (signature_B_fold in all_fold_loadings
        and "PC2" in all_fold_loadings[signature_B_fold].columns):
    pc2_series = all_fold_loadings[signature_B_fold]["PC2"]
    loadings_summary_B = (pd.DataFrame({"feature": pc2_series.index,
                                         "loading": pc2_series.values})
                           .assign(abs_loading=lambda d: d["loading"].abs())
                           .sort_values("abs_loading", ascending=False)
                           .drop(columns="abs_loading")
                           .reset_index(drop=True))

    print("\nTop 10 most important features for Signature B (PC2):")
    print(loadings_summary_B.head(10).to_string(index=False, float_format="%.3f"))

    top20_B = loadings_summary_B.head(20)
    fig, ax = plt.subplots(figsize=(6, 7))
    colors = ["steelblue" if v > 0 else "coral" for v in top20_B["loading"]]
    ax.barh(top20_B["feature"], top20_B["loading"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Loading")
    ax.set_title("Signature B: Feature Loadings for Predictive PC2\n"
                 "Derived from 'OFT + Y maze' fold only.")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "signature_B_pc2_loadings.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'signature_B_pc2_loadings.png'}")
else:
    print("Could not find PC2 loading data for the 'OFT + Y maze' fold.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 8: SYLLABLE SIGNATURE SCATTER PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n--- PART 8: SYLLABLE SIGNATURE SCATTER PLOTS ---")


def make_syllable_scatter(summary_df: pd.DataFrame, loading_col: str,
                           title: str, out_path: Path) -> None:
    df = summary_df[["feature", loading_col]].rename(columns={loading_col: "loading_val"}).copy()
    df["node"] = df["feature"].str.extract(r"(s\d+)$")
    df["type"] = df["feature"].apply(
        lambda f: "freq"       if "freq50"      in f else
                  "self_trans" if "self_trans"   in f else
                  "entropy"    if "row_entropy"  in f else None
    )
    df = df.dropna(subset=["type", "node"])

    wide = df.pivot_table(index="node", columns="type", values="loading_val").reset_index()
    wide.columns.name = None
    wide["usage"]      = (wide["freq"].abs() + wide["self_trans"].abs() + wide["entropy"].abs()) / 3
    wide["node_label"] = wide["node"].apply(
        lambda n: "0" if re.fullmatch(r"s0+", n) else re.sub(r"^s0*", "", n))

    vmax = wide["self_trans"].abs().max()
    sizes = (wide["entropy"].abs() / wide["entropy"].abs().max()) * 400 + 30

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(wide["freq"], wide["usage"],
                    c=wide["self_trans"], s=sizes,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    alpha=0.85, edgecolors="0.3", linewidths=0.5)
    from adjustText import adjust_text
    texts = [ax.text(row["freq"], row["usage"], row["node_label"], fontsize=9)
             for _, row in wide.iterrows()]
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5),
                expand=(1.2, 1.4), force_text=(0.3, 0.5))

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Self-transition loading", fontsize=9)
    ax.axhline(0, ls="--", color="gray", lw=0.7)
    ax.axvline(0, ls="--", color="gray", lw=0.7)
    ax.set_xlabel("Occurrence loading (freq50)")
    ax.set_ylabel("Mean absolute loading (usage)")
    ax.set_title(title)

    # Size legend
    for frac, lbl in [(0.2, "low entropy"), (1.0, "high entropy")]:
        ax.scatter([], [], s=frac * 400 + 30, color="0.6",
                   edgecolors="0.3", label=lbl)
    ax.legend(title="|Row-entropy loading|", fontsize=8, title_fontsize=8,
              loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


if "loadings_summary_A" in dir() and len(loadings_summary_A) > 0:
    make_syllable_scatter(loadings_summary_A, "mean_loading",
                          "Signature A: Syllable Profile (Stable PC1)",
                          OUT_DIR / "signature_A_syllable_scatter.png")

if "loadings_summary_B" in dir() and len(loadings_summary_B) > 0:
    make_syllable_scatter(loadings_summary_B, "loading",
                          "Signature B: Syllable Profile (Predictive PC2)",
                          OUT_DIR / "signature_B_syllable_scatter.png")


print("\n\nAll outputs saved to:", OUT_DIR)
