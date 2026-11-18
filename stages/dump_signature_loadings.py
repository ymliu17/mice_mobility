"""Dump full PC loading tables for Signatures A and B.

Reproduces the stable-PCA step from s4-5_pca_analysis.R (1000-iter
stratified bootstrap, sign-aligned to a reference PCA) for the folds
that define the two signatures, then writes full loading CSVs.

- Signature A = mean PC1 loading across the OFT+Food_AST and
  OFT+OFT_DUO folds.
- Signature B = PC2 loading from the OFT+Y_maze fold.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT_CSV = Path("outputs/05_stage5/study_group_feature_master_presence_gt10_full10meta_zscored.csv")
OUT_DIR = Path("outputs/05_stage5/pca_analysis_R")
N_ITERATIONS = 1000
SEED = 123

FEATURE_PREFIXES = (
    "freq50_s",
    "transsummary_self_trans_s",
    "transsummary_row_entropy_s",
)


def load_and_residualize() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(INPUT_CSV)
    core = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    df = df[df["label_binary"].notna()].copy()
    df["loco_context"] = np.where(
        df["env_context"].str.lower().str.contains("food intake|ast", regex=True),
        "Food_AST", df["env_context"])

    # Residualize each feature on study_cohort + env_context via OLS.
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
    ref = Vt_ref.T  # features x PCs

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
        # Sign-align to reference.
        for pc in range(min(loadings.shape[1], ref.shape[1])):
            if float(ref[:, pc] @ loadings[:, pc]) < 0:
                loadings[:, pc] *= -1
        accum += loadings
        n_valid += 1

    mean_loadings = accum / n_valid
    return pd.DataFrame(mean_loadings,
                        index=features,
                        columns=[f"PC{i+1}" for i in range(mean_loadings.shape[1])])


def tidy(feature: str) -> tuple[str, str]:
    for prefix, kind in [
        ("freq50_s", "freq50"),
        ("transsummary_self_trans_s", "self_trans"),
        ("transsummary_row_entropy_s", "row_entropy"),
    ]:
        if feature.startswith(prefix):
            node = "s" + feature.split("_s")[-1]
            return kind, node
    return "other", feature


def save_signature_A(fold_loadings: dict[str, pd.DataFrame]) -> None:
    folds = ["OFT + Food_AST", "OFT + OFT_DUO"]
    pc1 = pd.concat({f: fold_loadings[f]["PC1"] for f in folds}, axis=1)
    out = pc1.rename(columns=lambda c: f"PC1_{c}")
    out["mean_loading"] = pc1.mean(axis=1)
    out["sd_loading"] = pc1.std(axis=1, ddof=1)
    out.insert(0, "feature", out.index)
    kinds = out["feature"].map(lambda f: tidy(f)[0])
    nodes = out["feature"].map(lambda f: tidy(f)[1])
    out.insert(1, "feature_type", kinds)
    out.insert(2, "node", nodes)
    out = out.sort_values("mean_loading", key=lambda s: s.abs(), ascending=False)
    path = OUT_DIR / "signature_A_pc1_loadings.csv"
    out.to_csv(path, index=False)
    print(f"  -> {path}  ({len(out)} rows)")


def save_signature_B(fold_loadings: dict[str, pd.DataFrame]) -> None:
    fold = "OFT + Y maze"
    pc2 = fold_loadings[fold]["PC2"]
    out = pd.DataFrame({"feature": pc2.index, "loading": pc2.values})
    out["feature_type"] = out["feature"].map(lambda f: tidy(f)[0])
    out["node"] = out["feature"].map(lambda f: tidy(f)[1])
    out = out[["feature", "feature_type", "node", "loading"]]
    out = out.sort_values("loading", key=lambda s: s.abs(), ascending=False)
    path = OUT_DIR / "signature_B_pc2_loadings.csv"
    out.to_csv(path, index=False)
    print(f"  -> {path}  ({len(out)} rows)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, features = load_and_residualize()
    print(f"data: {len(df)} rows, {len(features)} features")

    folds_needed = {"Food_AST", "OFT_DUO", "Y maze"}
    fold_loadings: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(SEED)
    for partner in folds_needed:
        fold_df = df[df["loco_context"].isin(["OFT", partner])].reset_index(drop=True)
        if fold_df.empty:
            print(f"  skipping {partner}: no rows")
            continue
        print(f"  fold 'OFT + {partner}': {len(fold_df)} rows, running PCA...")
        fold_loadings[f"OFT + {partner}"] = stable_loadings(fold_df, features, rng)

    save_signature_A(fold_loadings)
    save_signature_B(fold_loadings)


if __name__ == "__main__":
    main()
