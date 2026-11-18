from __future__ import annotations

import pandas as pd


def model_registry() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model_family": "elasticnet_logreg",
            "status": "recommended_next",
            "why": "Sparse linear baseline with interpretable coefficients and feature selection.",
        },
        {
            "model_family": "linear_svm",
            "status": "recommended_next",
            "why": "Strong small-sample classifier for high-dimensional behavioral features.",
        },
        {
            "model_family": "xgboost_or_gbdt",
            "status": "recommended_next",
            "why": "Tests whether nonlinear feature combinations improve beyond linear models.",
        },
        {
            "model_family": "graph_kernel_svm",
            "status": "later",
            "why": "Useful only if graph summaries show clear extra signal first.",
        },
        {
            "model_family": "gnn",
            "status": "deferred",
            "why": "Not first priority for a 5-node graph and modest sample size.",
        },
    ])

