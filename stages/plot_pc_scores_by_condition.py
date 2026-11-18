"""Plot PC scores by condition, using fold-specific PCs matching the GLMM.

For each fold ("OFT + partner"), the GLMM fits
  label_binary ~ PC1 + PC2 + PC3 + PC4 + env_context + (1 | study_cohort)
on residualized features (residualized on study_cohort + env_context).

These plots visualize the samples the GLMM saw: PC score from the *fold-specific*
stable loading, grouped by condition (derived from label_binary + abnormal_condition).
Each subplot is annotated with the GLMM observed z-score and permutation p-value
pulled from outputs/05_stage5/pca_analysis/oft_anchored_significance.csv.

Outputs (PNG and PDF) under outputs/05_stage5/pca_analysis_R/:
  pc1_food_ast_oft_duo_by_condition.{png,pdf}
  pc2_ymaze_by_condition.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

INPUT_CSV = Path("outputs/05_stage5/study_group_feature_master_presence_gt10_full10meta_zscored.csv")
GLMM_CSV = Path("outputs/05_stage5/pca_analysis/oft_anchored_significance.csv")
OUT_DIR = Path("outputs/05_stage5/pca_analysis_R")
N_ITERATIONS = 1000
SEED = 123

FEATURE_PREFIXES = ("freq50_s", "transsummary_self_trans_s", "transsummary_row_entropy_s")

# Context-aware condition labels. Food-AST is the social-isolation comparison;
# OFT-Duo and Y-maze are pre/post-surgery comparisons (takaya's control/surgery
# is the same pre/post surgical contrast as s15's pre_surgery/post_surgery).
CONDITION_MAP_FOOD = {
    "control":          "control",
    "social_isolation": "social isolation",
}
CONDITION_MAP_SURGERY = {
    "control":       "pre-surgery",
    "pre":           "pre-surgery",
    "pre_surgery":   "pre-surgery",
    "surgery":       "post-surgery",
    "post_surgery":  "post-surgery",
    "post1":         "post-surgery",
    "post3":         "post-surgery",
}
CONDITION_ORDER = ["control", "social isolation", "pre-surgery", "post-surgery"]
CONDITION_COLORS = {
    "control":          "#9e9e9e",
    "social isolation": "#1f77b4",
    "pre-surgery":      "#9e9e9e",
    "post-surgery":     "#d62728",
}


def load_and_residualize() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(INPUT_CSV)
    core = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    df = df[df["label_binary"].notna()].copy()
    df["loco_context"] = np.where(
        df["env_context"].str.lower().str.contains("food intake|ast", regex=True),
        "Food_AST", df["env_context"])

    meta = pd.get_dummies(df[["study_cohort", "env_context"]], drop_first=True).astype(float)
    X = sm.add_constant(meta, has_constant="add")
    for feat in core:
        y = df[feat]
        mask = y.notna()
        if mask.sum() < 3:
            continue
        res = sm.OLS(y[mask].astype(float), X.loc[mask]).fit()
        df.loc[mask, feat] = y[mask].astype(float) - res.predict(X.loc[mask])

    df = df.dropna(subset=core).reset_index(drop=True)
    return df, core


def stable_loadings(fold_df: pd.DataFrame, features: list[str],
                    rng: np.random.Generator) -> pd.DataFrame:
    X = fold_df[features].to_numpy(dtype=float)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    _, _, Vt_ref = np.linalg.svd(Xs, full_matrices=False)
    ref = Vt_ref.T

    groups = fold_df.groupby(["env_context", "study_cohort"]).indices
    accum = np.zeros_like(ref)
    n_valid = 0
    for _ in range(N_ITERATIONS):
        idx = np.concatenate([rng.choice(v, size=1, replace=True)
                              for v in groups.values()])
        Xi = fold_df.iloc[idx][features].to_numpy(dtype=float)
        sds = Xi.std(0, ddof=1)
        use = sds > 1e-10
        if use.sum() < 2:
            continue
        Xi_scaled = (Xi[:, use] - Xi[:, use].mean(0)) / sds[use]
        _, _, Vt = np.linalg.svd(Xi_scaled, full_matrices=False)
        loadings = np.zeros_like(ref)
        loadings[np.where(use)[0], :Vt.shape[0]] = Vt.T
        for pc in range(min(loadings.shape[1], ref.shape[1])):
            if float(ref[:, pc] @ loadings[:, pc]) < 0:
                loadings[:, pc] *= -1
        accum += loadings
        n_valid += 1

    return pd.DataFrame(accum / n_valid, index=features,
                        columns=[f"PC{i+1}" for i in range(accum.shape[1])])


def fold_pc_scores(fold_df: pd.DataFrame, features: list[str],
                   loadings: pd.DataFrame) -> pd.DataFrame:
    """Project fold samples onto their stable loadings, matching the R script.

    The R code scales the fold's feature matrix then multiplies by the stable
    loadings: scores = scale(X) @ loadings.
    """
    X = fold_df[features].to_numpy(dtype=float)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    scores = Xs @ loadings.values
    out = fold_df.copy()
    for i, pc in enumerate(loadings.columns):
        out[pc] = scores[:, i]
    return out


def sig_marker(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def lookup_glmm(glmm: pd.DataFrame, pair: str, term: str) -> tuple[float, float]:
    row = glmm[(glmm["training_pair"] == pair) & (glmm["term"] == term)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row["observed_z_score"].iloc[0]), float(row["permutation_p_value"].iloc[0])


def box_strip_panel(ax, sub: pd.DataFrame, score_col: str,
                    z: float, p: float, rng: np.random.Generator) -> None:
    conds = [c for c in CONDITION_ORDER if c in sub["condition"].unique()]
    box_data = [sub.loc[sub["condition"] == c, score_col].values for c in conds]
    bp = ax.boxplot(box_data, positions=range(len(conds)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor(CONDITION_COLORS[c])
        patch.set_alpha(0.35)
        patch.set_edgecolor(CONDITION_COLORS[c])

    # Shape encodes env_context so the GLMM's env_context term is visible.
    env_markers = {"OFT": "o", "Food_AST": "s", "OFT_DUO": "s", "Y maze": "s"}
    for i, c in enumerate(conds):
        g = sub[sub["condition"] == c]
        for env, mk in env_markers.items():
            sub_env = g[g["loco_context"] == env]
            if sub_env.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, len(sub_env))
            ax.scatter(np.full(len(sub_env), i, dtype=float) + jitter,
                       sub_env[score_col].values,
                       color=CONDITION_COLORS[c], edgecolor="black",
                       linewidth=0.4, s=32, alpha=0.85, marker=mk, zorder=3)

    # Significance line spanning Control <-> Abnormal (GLMM result).
    # Extend the y-axis upward to make room for the annotation so it doesn't
    # overlap the axis border.
    if len(conds) == 2:
        y_lo, y_hi = sub[score_col].min(), sub[score_col].max()
        y_span = y_hi - y_lo
        pad = 0.08 * y_span
        y_line = y_hi + pad
        ax.plot([0, 1], [y_line, y_line], color="black", linewidth=1)
        label = f"{sig_marker(p)}  (z={z:.2f}, p={p:.3f})"
        ax.text(0.5, y_line + pad * 0.4, label,
                ha="center", va="bottom", fontsize=10)
        ax.set_ylim(y_lo - pad, y_hi + pad * 3.0)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, features = load_and_residualize()
    glmm = pd.read_csv(GLMM_CSV)

    rng = np.random.default_rng(SEED)
    folds = {
        "OFT + Food_AST": df[df["loco_context"].isin(["OFT", "Food_AST"])].reset_index(drop=True),
        "OFT + OFT_DUO":  df[df["loco_context"].isin(["OFT", "OFT_DUO"])].reset_index(drop=True),
        "OFT + Y maze":   df[df["loco_context"].isin(["OFT", "Y maze"])].reset_index(drop=True),
    }
    loadings = {name: stable_loadings(fd, features, rng) for name, fd in folds.items()}
    scored = {name: fold_pc_scores(fd, features, loadings[name])
              for name, fd in folds.items()}

    # --- PC1 plot: two panels, one per Signature A fold ---
    jrng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=False)
    for ax, (pair, partner_env, ctx_label, cond_map) in zip(axes, [
        ("OFT + Food_AST", "Food_AST", "Food-AST", CONDITION_MAP_FOOD),
        ("OFT + OFT_DUO",  "OFT_DUO",  "OFT-Duo",  CONDITION_MAP_SURGERY),
    ]):
        sub = scored[pair].copy()
        sub = sub[sub["loco_context"] == partner_env]
        sub["condition"] = sub["abnormal_condition"].map(cond_map)
        sub = sub.dropna(subset=["condition"])
        z, p = lookup_glmm(glmm, pair, "PC1")
        print(f"{pair} PC1 samples: n={len(sub)}, z={z}, p={p}")
        print(sub.groupby("condition").size())
        box_strip_panel(ax, sub, "PC1", z, p, jrng)
        ax.set_xlabel(ctx_label, fontsize=12)
    axes[0].set_ylabel("PC1 score")
    axes[1].set_ylabel("PC1 score")
    fig.tight_layout()
    stem = OUT_DIR / "pc1_food_ast_oft_duo_by_condition"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {stem}.png / .pdf")

    # --- PC2 plot: Y-maze fold ---
    sub = scored["OFT + Y maze"].copy()
    sub = sub[sub["loco_context"] == "Y maze"]
    sub["condition"] = sub["abnormal_condition"].map(CONDITION_MAP_SURGERY)
    sub = sub.dropna(subset=["condition"])
    z, p = lookup_glmm(glmm, "OFT + Y maze", "PC2")
    print(f"OFT + Y maze PC2 samples: n={len(sub)}, z={z}, p={p}")
    print(sub.groupby("condition").size())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    box_strip_panel(ax, sub, "PC2", z, p, jrng)
    ax.set_ylabel("PC2 score")
    ax.set_xlabel("Y-maze", fontsize=12)
    fig.tight_layout()
    stem = OUT_DIR / "pc2_ymaze_by_condition"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {stem}.png / .pdf")


if __name__ == "__main__":
    main()
