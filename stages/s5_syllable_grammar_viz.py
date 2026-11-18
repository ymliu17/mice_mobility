"""
Syllable Grammar Visualisation
================================
Figure 1 — Grammar map: all 12 freq50 syllables in self-transition × row-entropy space
Figure 2 — Condition effect: Δ self-transition & Δ row-entropy (pre→post) for the 5 key syllables

Usage:
    PYTHONPATH=. /home/yangliu/miniconda3/envs/mice/bin/python stages/s5_syllable_grammar_viz.py
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats
from adjustText import adjust_text

# ── Paths ──────────────────────────────────────────────────────────────────────
SYL_CACHE   = "outputs/05_stage5/syllable_feature_cache.csv"
SES_CACHE   = "outputs/05_stage5/syllable_session_feature_cache.csv"
OUT_DIR     = "outputs/05_stage5/figures"

import os; os.makedirs(OUT_DIR, exist_ok=True)

KEY_SYLLS  = [0, 4, 7, 24, 26]
KEY_LABELS = {s: f"s{s:02d}" for s in KEY_SYLLS}
FREQ50_IDS = [0, 1, 4, 7, 11, 24, 26, 32, 37, 43, 45, 46]   # the 12 freq50 syllables

KEY_COLOR  = "#E63946"   # red for highlighted syllables
BG_COLOR   = "#ADB5BD"   # grey for background syllables

COHORT_COLORS = {
    "gi":     "#4C72B0",
    "s15":    "#DD8452",
    "takaya": "#55A868",
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Grammar Map
# ══════════════════════════════════════════════════════════════════════════════
syl = pd.read_csv(SYL_CACHE)
syl12 = syl[syl["syllable"].isin(FREQ50_IDS)].copy()

# Conditional entropy: H_cond = H_row / (1 - p_self)
# Captures exit-diversity independent of self-transition magnitude
syl12["cond_entropy"] = (
    syl12["mean_transition_entropy"] /
    (1 - syl12["self_transition"]).clip(lower=1e-6)
)

# Normalise usage to point area (min 40, max 400)
umin, umax = syl12["usage_mean"].min(), syl12["usage_mean"].max()
syl12["area"] = 40 + 360 * (syl12["usage_mean"] - umin) / (umax - umin + 1e-9)

fig, ax = plt.subplots(figsize=(7, 6))

# Background syllables
bg = syl12[~syl12["syllable"].isin(KEY_SYLLS)]
ax.scatter(bg["self_transition"], bg["cond_entropy"],
           s=bg["area"], color=BG_COLOR, alpha=0.7, linewidths=0.5,
           edgecolors="white", zorder=2)

# Key syllables
kw = syl12[syl12["syllable"].isin(KEY_SYLLS)]
ax.scatter(kw["self_transition"], kw["cond_entropy"],
           s=kw["area"], color=KEY_COLOR, alpha=0.9, linewidths=0.8,
           edgecolors="white", zorder=3)

# Label all syllables — adjustText prevents overlaps
texts = []
for _, row in syl12.iterrows():
    is_key = int(row["syllable"]) in KEY_SYLLS
    t = ax.text(
        row["self_transition"], row["cond_entropy"],
        f"s{int(row['syllable']):02d}",
        fontsize=8.5, fontweight="bold" if is_key else "normal",
        color=KEY_COLOR if is_key else "#555555",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=5,
    )
    texts.append(t)

adjust_text(
    texts, ax=ax,
    expand=(1.5, 1.8),
    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5),
    force_text=(0.6, 1.0),
    force_points=(0.3, 0.5),
)


ax.set_xlabel("Self-transition probability", fontsize=11)
ax.set_ylabel("Conditional exit entropy\nH(row) / (1 − p_self)", fontsize=10)
ax.set_title("Behavioral grammar map — freq50 syllables", fontsize=12, pad=10)
ax.set_xlim(syl12["self_transition"].min() - 0.005,
            syl12["self_transition"].max() + 0.005)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_grammar_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_DIR}/fig1_grammar_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Two-section layout:
#   Top 2 rows:    Surgery cohorts pooled (gi + s15 + takaya), HC vs PS,
#                  Welch t-test
#   Bottom 2 rows: Fiber photometry pooled (AST + Food intake), Control vs SI,
#                  Mann-Whitney U (small n)
# Columns = 12 freq50 syllables
# ══════════════════════════════════════════════════════════════════════════════
sg_all = pd.read_csv("outputs/05_stage5/"
                     "study_group_feature_master_presence_gt10_full10meta_zscored.csv")

surgery_df = sg_all[sg_all["study_cohort"].isin(["gi", "s15", "takaya"])].copy()
fiber_df   = sg_all[sg_all["study_cohort"] == "fiber"].copy()

COND_COLORS  = {"HC": "#4C9BE8", "PS": "#E63946"}
FIBER_COLORS = {"Control": "#4C9BE8", "SI": "#E6A817"}
syll_order   = [f"s{s:02d}" for s in FREQ50_IDS]

feat_pairs = [
    ("self-transition", [f"transsummary_self_trans_s{s:02d}"  for s in FREQ50_IDS]),
    ("row entropy",     [f"transsummary_row_entropy_s{s:02d}" for s in FREQ50_IDS]),
]

rng = np.random.default_rng(42)


def draw_strip_box(ax, vals_a, vals_b, labels, colors, feat_col, ylim, ypad, test="t"):
    for xi, (vals, lbl) in enumerate(zip([vals_a, vals_b], labels)):
        color = colors[lbl]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.bar(xi, q3 - q1, bottom=q1, width=0.45,
               color=color, alpha=0.35, zorder=2)
        ax.hlines(med, xi - 0.225, xi + 0.225, color=color, lw=2, zorder=3)
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, s=12, color=color,
                   alpha=0.6, linewidths=0, zorder=4)

    if test == "t":
        _, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    else:  # Mann-Whitney
        _, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")

    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    y_top = ylim[1] - ypad * 0.3
    ax.plot([0, 1], [y_top, y_top], color="black", lw=0.8)
    ax.text(0.5, y_top + ypad * 0.15, sig, ha="center", va="bottom",
            fontsize=9, color="black" if sig != "ns" else "grey")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)


# ── Fig 2a: Surgery cohorts ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 12, figsize=(26, 5.5),
                         gridspec_kw={"hspace": 0.5, "wspace": 0.25})
fig.suptitle("Surgery cohorts (gi + s15 + takaya) — HC vs PS", fontsize=13, y=1.01)

for fi, (feat_name, feat_cols) in enumerate(feat_pairs):
    all_vals = pd.concat([surgery_df[fc] for fc in feat_cols]).dropna()
    ypad = (all_vals.max() - all_vals.min()) * 0.12
    ylim = (all_vals.min() - ypad, all_vals.max() + ypad)

    for ci, (s, syll, feat_col) in enumerate(zip(FREQ50_IDS, syll_order, feat_cols)):
        ax = axes[fi, ci]
        hc_v = surgery_df.loc[surgery_df["label_binary"] == 0, feat_col].dropna().values
        ps_v = surgery_df.loc[surgery_df["label_binary"] == 1, feat_col].dropna().values
        draw_strip_box(ax, hc_v, ps_v, ["HC", "PS"], COND_COLORS,
                       feat_col, ylim, ypad, test="t")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "PS"], fontsize=8)
        if ci == 0:
            ax.set_ylabel(feat_name, fontsize=9)
        if fi == 0:
            tc = KEY_COLOR if s in KEY_SYLLS else "black"
            ax.set_title(syll, fontsize=9, fontweight="bold", color=tc)

fig.savefig(f"{OUT_DIR}/fig2a_surgery.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_DIR}/fig2a_surgery.png")

# ── Fig 2b: Fiber photometry ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 12, figsize=(26, 5.5),
                         gridspec_kw={"hspace": 0.5, "wspace": 0.25})
fig.suptitle("Fiber photometry (AST + Food intake) — Control vs SI  "
             "[Mann-Whitney U, n=24/8]", fontsize=13, y=1.01)

for fi, (feat_name, feat_cols) in enumerate(feat_pairs):
    all_vals = pd.concat([fiber_df[fc] for fc in feat_cols]).dropna()
    ypad = (all_vals.max() - all_vals.min()) * 0.12
    ylim = (all_vals.min() - ypad, all_vals.max() + ypad)

    for ci, (s, syll, feat_col) in enumerate(zip(FREQ50_IDS, syll_order, feat_cols)):
        ax = axes[fi, ci]
        ctrl_v = fiber_df.loc[fiber_df["label_binary"] == 0, feat_col].dropna().values
        si_v   = fiber_df.loc[fiber_df["label_binary"] == 1, feat_col].dropna().values
        draw_strip_box(ax, ctrl_v, si_v, ["Control", "SI"], FIBER_COLORS,
                       feat_col, ylim, ypad, test="mw")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ctrl", "SI"], fontsize=8)
        if ci == 0:
            ax.set_ylabel(feat_name, fontsize=9)
        if fi == 0:
            tc = KEY_COLOR if s in KEY_SYLLS else "black"
            ax.set_title(syll, fontsize=9, fontweight="bold", color=tc)

fig.savefig(f"{OUT_DIR}/fig2b_fiber.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_DIR}/fig2b_fiber.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Focused: s00 & s26 self-transition, surgery cohorts, HC vs PS
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import ttest_ind
from matplotlib.lines import Line2D

FOCUS_SYLLS = [0, 26]
FOCUS_FEAT  = [f"transsummary_self_trans_s{s:02d}" for s in FOCUS_SYLLS]
FOCUS_LABELS = [f"s{s:02d}" for s in FOCUS_SYLLS]

# Cohen's d helper
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * np.std(a, ddof=1)**2 +
                         (n2 - 1) * np.std(b, ddof=1)**2) / (n1 + n2 - 2))
    return (np.mean(b) - np.mean(a)) / (pooled_sd + 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(5.5, 5.0),
                         gridspec_kw={"wspace": 0.5})

# Global y-limits — add extra headroom at top for stat annotations
all_vals_focus = pd.concat([surgery_df[fc] for fc in FOCUS_FEAT]).dropna()
data_range = all_vals_focus.max() - all_vals_focus.min()
ypad_f  = data_range * 0.10          # padding below/above data
yhead_f = data_range * 0.38          # extra space at top for bar + sig + d
ylim_f  = (all_vals_focus.min() - ypad_f,
           all_vals_focus.max() + yhead_f)

rng2 = np.random.default_rng(0)

for ci, (syll, feat_col) in enumerate(zip(FOCUS_LABELS, FOCUS_FEAT)):
    ax = axes[ci]
    hc_v = surgery_df.loc[surgery_df["label_binary"] == 0, feat_col].dropna().values
    ps_v = surgery_df.loc[surgery_df["label_binary"] == 1, feat_col].dropna().values

    for xi, (vals, color) in enumerate(zip(
            [hc_v, ps_v], [COND_COLORS["HC"], COND_COLORS["PS"]])):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.bar(xi, q3 - q1, bottom=q1, width=0.5,
               color=color, alpha=0.35, zorder=2)
        ax.hlines(med, xi - 0.25, xi + 0.25, color=color, lw=2.2, zorder=3)
        jit = rng2.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jit, vals, s=18, color=color,
                   alpha=0.65, linewidths=0, zorder=4)

    # stats — three separate non-overlapping rows
    _, p = ttest_ind(hc_v, ps_v, equal_var=False)
    d    = cohens_d(hc_v, ps_v)
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    y_top   = all_vals_focus.max() + ypad_f * 0.5   # bracket line
    y_sig   = y_top + data_range * 0.05              # *** / ns
    y_d     = y_sig + data_range * 0.09              # Cohen's d
    ax.plot([0, 1], [y_top, y_top], color="black", lw=0.9)
    ax.plot([0, 0], [y_top - data_range * 0.02, y_top], color="black", lw=0.9)
    ax.plot([1, 1], [y_top - data_range * 0.02, y_top], color="black", lw=0.9)
    ax.text(0.5, y_sig, sig, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(0.5, y_d, f"d = {d:.2f}", ha="center", va="bottom",
            fontsize=8.5, color="#444444")

    # n counts below x-axis labels
    ax.text(0, ylim_f[0] + ypad_f * 0.1,
            f"n={len(hc_v)}", ha="center", va="bottom",
            fontsize=7.5, color=COND_COLORS["HC"])
    ax.text(1, ylim_f[0] + ypad_f * 0.1,
            f"n={len(ps_v)}", ha="center", va="bottom",
            fontsize=7.5, color=COND_COLORS["PS"])

    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(ylim_f)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HC", "PS"], fontsize=10)
    ax.set_title(syll, fontsize=13, fontweight="bold", color=KEY_COLOR, pad=6)
    if ci == 0:
        ax.set_ylabel("Self-transition probability (z)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Surgery cohorts (gi + s15 + takaya)\nSelf-transition: HC vs PS",
             fontsize=11, y=1.02)

# Shared legend
legend_els = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COND_COLORS["HC"],
           markersize=8, label='HC'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COND_COLORS["PS"],
           markersize=8, label='PS'),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_s00_s26_self_trans.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_DIR}/fig3_s00_s26_self_trans.png")
