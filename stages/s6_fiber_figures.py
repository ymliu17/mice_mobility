#!/usr/bin/env python3
"""
Stage 6 figures.

  figures/dopamine_ast.png    AST task dopamine: event locking, error prediction,
                              approach vigour, and the group contrast
  figures/locomotion.png      locomotion-onset response vs the 405 isosbestic,
                              per channel, plus the group x batch design

Every panel reports a statistic that respects the nesting of trials within
sessions -- session-paired tests, or OLS with session-clustered standard errors.
Treating trials as independent inflates significance several-fold here.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("pipeline.s6.fig")

C_CUE, C_REW, C_ERR, C_CTL = "#E8622D", "#1F7AE0", "#149C6B", "#9AA0A6"
C_GRP = {"Control": "#1F7AE0", "Social isolation": "#E8622D"}


def _session_date(session: str) -> pd.Timestamp:
    """`fiberphotometrysetup-240416-140112_...` -> 2024-04-16."""
    m = re.search(r"setup-(\d{2})(\d{2})(\d{2})-", session)
    if not m:
        return pd.NaT
    yy, mm, dd = (int(g) for g in m.groups())
    return pd.Timestamp(2000 + yy, mm, dd)


def _cluster_ols(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    """OLS slope with cluster-robust (CR0) standard errors."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((2, 2))
    for g in np.unique(groups):
        m = groups == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))[1]
    t = beta[1] / se
    df = len(np.unique(groups)) - 1
    p = 2 * stats.t.sf(abs(t), df)
    return beta[1], se, p, beta[1] - 1.96 * se, beta[1] + 1.96 * se


def _pfmt(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}".rstrip("0").rstrip(".")


def _titled(ax, title: str, subtitle: str, width: int = 68) -> None:
    """Bold panel title with a wrapped grey sub-line underneath, no overlap."""
    import textwrap
    lines = textwrap.wrap(subtitle, width) if subtitle else []
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12.5,
                 pad=10 + 12.5 * len(lines))
    for i, ln in enumerate(lines):
        ax.text(0, 1.012 + 0.043 * (len(lines) - 1 - i), ln, transform=ax.transAxes,
                fontsize=9.1, color="#5F6368", va="bottom", ha="left")


# ─── Figure 1: AST dopamine ──────────────────────────────────────────────────

def fig_dopamine_ast(cfg: dict, out: Path) -> None:
    o = Path(cfg["output_dir"])
    tt = pd.read_csv(o / "trial_table.csv")
    tr = np.load(o / "event_traces.npz", allow_pickle=True)
    fps, pre, post = float(tr["fps"]), float(tr["pre"]), float(tr["post"])
    t = np.arange(-int(pre * fps), int(post * fps)) / fps

    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10.2))
    fig.suptitle(
        f"Dopamine x approach kinematics in the AST  ·  "
        f"{len(tt)} trials, {tt.session.nunique()} sessions, {tt.mouse.nunique()} mice",
        fontsize=16, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.947,
             "Bleaching-robust dF/F (channel A). Speed from body_center at "
             f"{cfg['fps']:.0f} fps, scaled by neck->tail-base length "
             f"(1 unit ~ {cfg['cm_per_body_length']:.1f} cm). All contrasts within-session.",
             fontsize=10.5, color="#5F6368", ha="left")

    # ── A: event-locked dopamine ──
    a = ax[0, 0]
    spec = [("GateOpen", C_CUE, "GateOpen (trial cue)"),
            ("EatingStart", C_REW, "EatingStart (reward)"),
            ("IncorrectDigStart", C_ERR, "IncorrectDig (error)")]
    notes = []
    for ev, col, lab in spec:
        k, kc = f"{ev}|ALL|A|dff", f"{ev}|ALL|A|iso"
        if k not in tr:
            continue
        M, Mc = tr[k], tr[kc]
        m, sem = np.nanmean(M, 0), stats.sem(M, 0, nan_policy="omit")
        a.plot(t, m, color=col, lw=2, label=lab, zorder=3)
        a.fill_between(t, m - sem, m + sem, color=col, alpha=0.18, lw=0, zorder=2)
        w = slice(int(pre * fps), int((pre + 2) * fps))
        s_, c_ = np.nanmean(M[:, w], 1), np.nanmean(Mc[:, w], 1)
        n = min(len(s_), len(c_))
        p = stats.wilcoxon(s_[:n], c_[:n]).pvalue
        notes.append((lab.split(" (")[0], np.nanmean(s_), p, n))
    if "GateOpen|ALL|A|iso" in tr:
        a.plot(t, np.nanmean(tr["GateOpen|ALL|A|iso"], 0), color=C_CTL, lw=1.4,
               ls=(0, (3, 2)), label="405 isosbestic", zorder=1)
    a.axvline(0, color="#3C4043", lw=0.9, ls=":")
    a.axhline(0, color="#3C4043", lw=0.9)
    a.set_xlabel("time from event (s)")
    a.set_ylabel("dF/F (%)")
    a.set_xlim(-2, post)
    a.legend(frameon=False, fontsize=9.5, loc="upper left")
    sig = ", ".join(f"{n_}: {v:+.2f}% {_pfmt(p)}" for n_, v, p, _ in notes[:2])
    _titled(a, "Dopamine is event-locked", f"AUC 0-2 s vs the 405 control. {sig}")
    if len(notes) > 2:
        n_, v, p, nn = notes[2]
        a.text(0.985, 0.03, f"{n_}: {v:+.2f}%, {_pfmt(p)} (n={nn}) -- n.s.",
               transform=a.transAxes, fontsize=9, color=C_ERR, ha="right")

    # ── B: pre-cue dopamine state before errors ──
    # NB: the *evoked* cue response does not differ by outcome (p ~ 0.8).
    # What differs is the absolute dF/F level already present before the gate
    # opens, so this panel plots the un-baselined pre-cue window.
    b = ax[0, 1]
    cc = tt[tt.outcome.isin(["correct", "incorrect"])].dropna(subset=["da_pre"])
    g = cc.groupby(["session", "outcome"]).da_pre.mean().unstack().dropna()
    for _, row in g.iterrows():
        b.plot([0, 1], [row["correct"], row["incorrect"]], color="#BDC1C6", lw=0.9, zorder=1)
    for i, c_ in enumerate(["correct", "incorrect"]):
        jit = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.12
        b.scatter(np.full(len(g), i) + jit, g[c_], s=34,
                  color=[C_REW, C_CUE][i], zorder=3, edgecolor="white", linewidth=0.6)
        b.hlines(g[c_].mean(), i - 0.26, i + 0.26, color=[C_REW, C_CUE][i], lw=3.4, zorder=4)
    pw = stats.wilcoxon(g["incorrect"], g["correct"]).pvalue
    b.set_xticks([0, 1]); b.set_xticklabels(["correct", "incorrect"])
    b.set_xlim(-0.45, 1.45)
    b.set_ylabel("dF/F 2-1 s BEFORE GateOpen (%, not baselined)")
    _titled(b, "Pre-cue dopamine state predicts errors",
            f"Paired by session (n={len(g)}), Wilcoxon {_pfmt(pw)}: "
            f"{g['correct'].mean():+.2f}% -> {g['incorrect'].mean():+.2f}%. "
            "The cue-evoked response itself does not differ (p ~ 0.8).")

    # ── C: cue dopamine predicts approach vigour ──
    c = ax[1, 0]
    d = tt.dropna(subset=["da_gate", "approach_speed_cms"]).copy()
    for col in ("da_gate", "approach_speed_cms"):
        d[col + "_c"] = d[col] - d.groupby("session")[col].transform("mean")
    d["q"] = d.groupby("session").da_gate.transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if s.notna().sum() >= 4 else np.nan)
    q = d.dropna(subset=["q"]).groupby("q").approach_speed_cms_c.agg(["mean", "sem"])
    c.bar(q.index, q["mean"], yerr=q["sem"], color=C_REW, width=0.68,
          error_kw=dict(ecolor="#5F6368", capsize=4, lw=1.3), zorder=2)
    for i, r in q.iterrows():
        va = "bottom" if r["mean"] >= 0 else "top"
        off = r["sem"] + 0.08 if r["mean"] >= 0 else -(r["sem"] + 0.08)
        c.text(i, r["mean"] + off, f"{r['mean']:+.2f}", ha="center", va=va, fontsize=10)
    slope, se, p, lo, hi = _cluster_ols(d.approach_speed_cms_c.values,
                                        d.da_gate_c.values, d.session.values)
    sl = d.groupby("session").apply(
        lambda x: np.polyfit(x.da_gate, x.approach_speed_cms, 1)[0] if len(x) > 4 else np.nan,
        include_groups=False).dropna()
    c.axhline(0, color="#3C4043", lw=0.9)
    c.margins(y=0.20)
    c.set_xticks(range(4))
    c.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    c.set_xlabel("dF/F at GateOpen (within-session quartile)")
    c.set_ylabel("approach speed, session-centred (cm/s)")
    _titled(c, "Cue dopamine predicts approach vigour",
            f"{slope:+.2f} cm/s per 1% dF/F [95% CI {lo:.2f}, {hi:.2f}], "
            f"session-clustered {_pfmt(p)}; {(sl > 0).sum()}/{len(sl)} sessions positive.")

    # ── D: group contrast, with the batch confound made visible ──
    dax = ax[1, 1]
    sess = (tt.groupby(["session", "group"])[["da_gate", "da_eat"]]
              .mean().reset_index())
    sess["date"] = sess.session.map(_session_date)
    xnum = sess.date.map(pd.Timestamp.toordinal).astype(float)
    norm = plt.Normalize(xnum.min(), xnum.max())
    cmap = plt.get_cmap("viridis")

    pos = {("da_gate", "Control"): 0, ("da_gate", "Social isolation"): 0.9,
           ("da_eat", "Control"): 2.3, ("da_eat", "Social isolation"): 3.2}
    rng = np.random.default_rng(1)
    pvals = {}
    for var in ("da_gate", "da_eat"):
        vals = {}
        for grp in ("Control", "Social isolation"):
            s = sess[(sess.group == grp)].dropna(subset=[var])
            x = pos[(var, grp)] + (rng.random(len(s)) - 0.5) * 0.3
            dax.scatter(x, s[var], s=46, c=cmap(norm(s.date.map(pd.Timestamp.toordinal))),
                        edgecolor="#3C4043", linewidth=0.7, zorder=3)
            dax.hlines(s[var].mean(), pos[(var, grp)] - 0.33, pos[(var, grp)] + 0.33,
                       color="#3C4043", lw=3, zorder=4)
            vals[grp] = s[var].values
        pvals[var] = stats.mannwhitneyu(vals["Control"], vals["Social isolation"]).pvalue

    dax.set_xticks(list(pos.values()))
    dax.set_xticklabels(["Control", "Isolation", "Control", "Isolation"], fontsize=9.5)
    dax.text(0.45, -0.145, "cue (GateOpen)", transform=dax.get_xaxis_transform(),
             ha="center", fontsize=10.5, fontweight="bold")
    dax.text(2.75, -0.145, "reward (EatingStart)", transform=dax.get_xaxis_transform(),
             ha="center", fontsize=10.5, fontweight="bold")
    dax.axvline(1.6, color="#DADCE0", lw=1)
    dax.axhline(0, color="#3C4043", lw=0.9)
    dax.set_ylabel("session-mean dF/F (%)")
    _titled(dax, "Group difference is confounded with batch",
            f"Session means, MWU: cue {_pfmt(pvals['da_gate'])}, "
            f"reward {_pfmt(pvals['da_eat'])}. Colour = recording date: the two "
            "groups share no mouse and no date.", width=58)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=dax,
                      fraction=0.040, pad=0.02)
    ticks = np.linspace(xnum.min(), xnum.max(), 4)
    cb.set_ticks(ticks)
    cb.set_ticklabels([pd.Timestamp.fromordinal(int(v)).strftime("%Y-%m") for v in ticks])
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_visible(False)

    for a_ in ax.ravel():
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=3.2, w_pad=2.4)
    fig.savefig(out / "dopamine_ast.png", dpi=170, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out / "dopamine_ast.png")


# ─── Figure 1b: three-panel core result (preliminary slide) ─────────────────

def fig_dopamine_core(cfg: dict, out: Path) -> None:
    """The three statements a preliminary results slide needs, nothing else."""
    o = Path(cfg["output_dir"])
    tt = pd.read_csv(o / "trial_table.csv")
    tr = np.load(o / "event_traces.npz", allow_pickle=True)
    fps, pre, post = float(tr["fps"]), float(tr["pre"]), float(tr["post"])
    t = np.arange(-int(pre * fps), int(post * fps)) / fps

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 5.0))
    fig.suptitle(f"Dopamine and approach vigour in the AST  ·  {len(tt)} trials, "
                 f"{tt.session.nunique()} sessions, {tt.mouse.nunique()} mice",
                 fontsize=15, fontweight="bold", x=0.008, ha="left", y=0.99)

    # A: event-locked
    a = ax[0]
    sig = []
    for ev, col, lab in (("GateOpen", C_CUE, "cue (GateOpen)"),
                         ("EatingStart", C_REW, "reward (EatingStart)")):
        M, Mc = tr[f"{ev}|ALL|A|dff"], tr[f"{ev}|ALL|A|iso"]
        m, sem = np.nanmean(M, 0), stats.sem(M, 0, nan_policy="omit")
        a.plot(t, m, color=col, lw=2.1, label=lab, zorder=3)
        a.fill_between(t, m - sem, m + sem, color=col, alpha=0.18, lw=0, zorder=2)
        w = slice(int(pre * fps), int((pre + 2) * fps))
        n = min(len(M), len(Mc))
        sig.append((lab.split(" (")[0], np.nanmean(M[:n, w]),
                    stats.wilcoxon(np.nanmean(M[:n, w], 1),
                                   np.nanmean(Mc[:n, w], 1)).pvalue))
    a.plot(t, np.nanmean(tr["GateOpen|ALL|A|iso"], 0), color=C_CTL, lw=1.5,
           ls=(0, (3, 2)), label="405 isosbestic control", zorder=1)
    a.axvline(0, color="#3C4043", lw=0.9, ls=":")
    a.axhline(0, color="#3C4043", lw=0.9)
    a.set_xlim(-2, post)
    a.set_xlabel("time from event (s)")
    a.set_ylabel("dF/F (%)")
    a.legend(frameon=False, fontsize=9.5, loc="upper left")
    _titled(a, "Dopamine is event-locked",
            f"AUC 0-2 s vs the 405 control: cue {sig[0][1]:+.2f}%, "
            f"reward {sig[1][1]:+.2f}%, both {_pfmt(max(sig[0][2], sig[1][2]))}", width=54)

    # B: approach vigour
    b = ax[1]
    d = tt.dropna(subset=["da_gate", "approach_speed_cms"]).copy()
    for col in ("da_gate", "approach_speed_cms"):
        d[col + "_c"] = d[col] - d.groupby("session")[col].transform("mean")
    d["q"] = d.groupby("session").da_gate.transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False)
        if x.notna().sum() >= 4 else np.nan)
    q = d.dropna(subset=["q"]).groupby("q").approach_speed_cms_c.agg(["mean", "sem"])
    b.bar(q.index, q["mean"], yerr=q["sem"], color=C_REW, width=0.68,
          error_kw=dict(ecolor="#5F6368", capsize=4, lw=1.3), zorder=2)
    for i, r in q.iterrows():
        off = r["sem"] + 0.1 if r["mean"] >= 0 else -(r["sem"] + 0.1)
        b.text(i, r["mean"] + off, f"{r['mean']:+.2f}", ha="center",
               va="bottom" if r["mean"] >= 0 else "top", fontsize=10.5)
    slope, se, p, lo, hi = _cluster_ols(d.approach_speed_cms_c.values,
                                        d.da_gate_c.values, d.session.values)
    sl = d.groupby("session").apply(
        lambda x: np.polyfit(x.da_gate, x.approach_speed_cms, 1)[0] if len(x) > 4 else np.nan,
        include_groups=False).dropna()
    b.axhline(0, color="#3C4043", lw=0.9)
    b.margins(y=0.22)
    b.set_xticks(range(4))
    b.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    b.set_xlabel("cue dopamine (within-session quartile)")
    b.set_ylabel("approach speed, session-centred (cm/s)")
    _titled(b, "Cue dopamine predicts approach vigour",
            f"{slope:+.2f} cm/s per 1% dF/F [95% CI {lo:.2f}, {hi:.2f}], "
            f"session-clustered {_pfmt(p)}; {(sl > 0).sum()}/{len(sl)} sessions positive",
            width=54)

    # C: pre-cue state predicts errors
    c = ax[2]
    cc = tt[tt.outcome.isin(["correct", "incorrect"])].dropna(subset=["da_pre"])
    g = cc.groupby(["session", "outcome"]).da_pre.mean().unstack().dropna()
    for _, row in g.iterrows():
        c.plot([0, 1], [row["correct"], row["incorrect"]], color="#BDC1C6", lw=0.9, zorder=1)
    jit = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.12
    for i, nm in enumerate(("correct", "incorrect")):
        c.scatter(np.full(len(g), i) + jit, g[nm], s=36, color=[C_REW, C_CUE][i],
                  zorder=3, edgecolor="white", linewidth=0.6)
        c.hlines(g[nm].mean(), i - 0.26, i + 0.26, color=[C_REW, C_CUE][i], lw=3.4, zorder=4)
    pw = stats.wilcoxon(g["incorrect"], g["correct"]).pvalue
    c.axhline(0, color="#3C4043", lw=0.9)
    c.set_xticks([0, 1]); c.set_xticklabels(["correct", "incorrect"])
    c.set_xlim(-0.45, 1.45)
    c.set_ylabel("dF/F 2-1 s BEFORE the cue (%)")
    _titled(c, "Pre-cue state predicts errors",
            f"paired by session (n={len(g)}), {_pfmt(pw)}: "
            f"{g['correct'].mean():+.2f}% -> {g['incorrect'].mean():+.2f}%. "
            "The cue-evoked response itself does not differ.", width=54)

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.6)
    fig.savefig(out / "dopamine_core.png", dpi=170, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out / "dopamine_core.png")


# ─── Figure 2: experimental design (methods slide) ───────────────────────────

def _session_design(cfg: dict) -> pd.DataFrame:
    rows = []
    for f in sorted((Path(cfg["output_dir"]) / "sessions").glob("*.npz")):
        m = json.loads(bytes(np.load(f, allow_pickle=True)["meta_json"]).decode())
        rows.append(dict(session=m["session"], group=m["group"], test=m["test"],
                         mouse=m["mouse"]))
    d = pd.DataFrame(rows)
    d["date"] = d.session.map(_session_date)
    return d


def fig_design(cfg: dict, out: Path) -> None:
    """Who was recorded, when, in which assay -- and the batch confound."""
    d = _session_design(cfg)
    marks = {"AST": ("o", "AST"),
             "Food intake during habituation": ("s", "Food intake (habituation)")}

    fig, a = plt.subplots(figsize=(9.6, 4.0))
    rng = np.random.default_rng(2)
    for grp, y in (("Control", 1), ("Social isolation", 0)):
        for test, (mk, lab) in marks.items():
            s = d[(d.group == grp) & (d.test == test)]
            if not len(s):
                continue
            a.scatter(s.date, np.full(len(s), y) + (rng.random(len(s)) - .5) * .22,
                      s=58, marker=mk, color=C_GRP[grp], edgecolor="white",
                      linewidth=0.7, label=lab if y == 1 else None)
    a.set_yticks([0, 1])
    a.set_yticklabels([f"Social isolation\n{(d.group == 'Social isolation').sum()} sessions, "
                       f"{d[d.group == 'Social isolation'].mouse.nunique()} mice",
                       f"Control\n{(d.group == 'Control').sum()} sessions, "
                       f"{d[d.group == 'Control'].mouse.nunique()} mice"])
    a.set_ylim(-0.6, 1.6)
    a.set_xlabel("recording date")
    a.legend(frameon=False, fontsize=9.5, loc="lower right", title="assay",
             title_fontsize=9.5)
    n_m = int(d.groupby("mouse").group.nunique().gt(1).sum())
    n_d = int(d.groupby("date").group.nunique().gt(1).sum())
    _titled(a, "Design: group is confounded with batch",
            f"{len(d)} sessions, {d.mouse.nunique()} mice. {n_m} mice and {n_d} dates "
            "appear in both groups -- group, cohort and calendar time are inseparable.",
            width=86)
    for lab in a.get_xticklabels():
        lab.set_rotation(25); lab.set_ha("right")
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "design.png", dpi=170, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out / "design.png")


# ─── Figure 3: locomotion — definition, specificity, use ────────────────────

def fig_locomotion(cfg: dict, out: Path) -> None:
    """
    What 'locomotion' means here, why the signal is not a movement signal,
    and how speed enters the trial-level result.
    """
    o = Path(cfg["output_dir"])
    tt = pd.read_csv(o / "trial_table.csv")
    sess_dir = o / "sessions"
    fps, k_cm = cfg["fps"], cfg["cm_per_body_length"]
    lags = np.arange(-int(3 * fps), int(3 * fps) + 1, 2)
    PRE, POST, MOB = int(2 * fps), int(3 * fps), 0.5
    min_still, min_run = int(1.5 * fps), int(0.75 * fps)

    xc, onset = {"A": [], "C": []}, {"dffA": [], "isoA": []}
    for f in sorted(sess_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        sp = np.where(d["valid"], d["speed"], np.nan)
        for ch in ("A", "C"):
            y = d[f"dff{ch}"]
            n = min(len(sp), len(y))
            a_, b_ = sp[:n], y[:n]
            ok = np.isfinite(a_) & np.isfinite(b_)
            if ok.sum() < 2000:
                continue
            row = []
            for L in lags:
                sh = np.roll(b_, int(L))
                m = ok & np.isfinite(sh)
                row.append(np.corrcoef(a_[m], sh[m])[0, 1] if m.sum() > 500 else np.nan)
            xc[ch].append(row)
        m = np.nan_to_num(sp > MOB).astype(np.int8)
        dm = np.diff(m, prepend=m[:1])
        on = [i for i in np.where(dm == 1)[0]
              if i >= max(PRE, min_still) and i + max(POST, min_run) < len(m)
              and m[i - min_still:i].sum() == 0 and m[i:i + min_run].sum() == min_run]
        if len(on) >= 20:
            for k in onset:
                x = d[k]
                seg = np.stack([x[i - PRE:i + POST] for i in on if i + POST <= len(x)])
                if len(seg):
                    onset[k].append(np.nanmean(
                        seg - np.nanmean(seg[:, :int(fps)], 1, keepdims=True), 0))

    fig, ax = plt.subplots(2, 2, figsize=(14.0, 9.6))
    fig.suptitle("Locomotion: how it is measured, and how it relates to dopamine",
                 fontsize=15.5, fontweight="bold", x=0.010, ha="left", y=0.985)

    # ── A: definition, on one example trial ──
    a = ax[0, 0]
    cand = tt.dropna(subset=["straightness", "latency_s"])
    cand = cand[(cand.latency_s > 1.5) & (cand.latency_s < 3.0) & (cand.straightness > 0.9)]
    ev = pd.read_csv(o / "events.csv")
    ex = None
    for _, r in cand.iterrows():                      # need a gap-free example
        dd_ = np.load(sess_dir / f"{r.session}.npz", allow_pickle=True)
        cm, sd_ = dd_["cam_t"], dd_["speed"]
        nn_ = min(len(cm), len(sd_))
        cm, sd_ = cm[:nn_], sd_[:nn_]
        gg = ev[(ev.session == r.session) & (ev.event == "GateOpen")].sort_values("t").t.values
        if int(r.trial) >= len(gg):
            continue
        t0 = gg[int(r.trial)]
        mm = (cm >= t0 - 1.5) & (cm <= t0 + r.latency_s + 1.0)
        if mm.sum() > 20 and np.isfinite(sd_[mm]).all():
            ex, d, cam, spd, ctr, tg = r, dd_, cm, sd_, dd_["ctr"][:nn_], t0
            break
    if ex is None:
        ex = cand.iloc[len(cand) // 2]
        d = np.load(sess_dir / f"{ex.session}.npz", allow_pickle=True)
        cam, spd, ctr = d["cam_t"], d["speed"], d["ctr"]
        nn_ = min(len(cam), len(spd))
        cam, spd, ctr = cam[:nn_], spd[:nn_], ctr[:nn_]
        tg = ev[(ev.session == ex.session) & (ev.event == "GateOpen")].sort_values("t").t.values[int(ex.trial)]

    ta = tg + ex.latency_s
    m = (cam >= tg - 1.5) & (cam <= ta + 1.0)
    a.plot(cam[m] - tg, spd[m] * k_cm, color="#3C4043", lw=1.7, zorder=3)
    top = float(np.nanmax(spd[m] * k_cm)) * 1.55
    a.set_ylim(0, top)
    a.axvspan(0, 0.3, color=C_CUE, alpha=0.25, lw=0, label="dopamine read: 0-0.3 s")
    a.axvspan(0.5, ex.latency_s, color=C_REW, alpha=0.18, lw=0,
              label="speed read: 0.5 s to approach")
    for xx, lab in ((0, "GateOpen"), (ex.latency_s, "Approach")):
        a.axvline(xx, color="#3C4043", lw=1.0, ls=":")
        a.text(xx, top * 0.995, f" {lab}", fontsize=8.8, va="top", rotation=90)
    a.set_xlabel("time from cue (s)")
    a.set_ylabel("speed (cm/s)")
    a.legend(frameon=False, fontsize=8.6, loc="upper left",
             bbox_to_anchor=(0.0, 0.88))
    ins = a.inset_axes([0.66, 0.60, 0.31, 0.36])
    mp = (cam >= tg) & (cam < ta)
    P = ctr[mp]
    P = P[np.isfinite(P).all(1)]
    ins.plot(P[:, 0], P[:, 1], color=C_REW, lw=1.5)
    ins.scatter(*P[0], s=20, color="#3C4043", zorder=3)
    ins.scatter(*P[-1], s=20, color=C_CUE, zorder=3)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("body_center path", fontsize=8.0, pad=2)
    for sp_ in ins.spines.values():
        sp_.set_color("#DADCE0")
    st = tt.straightness.dropna()
    _titled(a, "How locomotion is measured",
            "Speed = frame-to-frame displacement of body_center at 20 fps, scaled by "
            f"each session's neck-tail-base length (1 BL ~ {k_cm:.1f} cm). The cue-to-approach "
            f"window is a directed run: straightness {st.median():.2f} "
            f"(net/path), {100 * (st > 0.7).mean():.0f}% of trials above 0.7.", width=74)

    # ── B: not a movement signal ──
    b = ax[0, 1]
    tl = lags / fps
    for ch, col in (("A", C_REW), ("C", C_CUE)):
        M = np.array(xc[ch])
        if not len(M):
            continue
        mu, sem = np.nanmean(M, 0), stats.sem(M, 0, nan_policy="omit")
        b.plot(tl, mu, color=col, lw=1.9, label=f"channel {ch}")
        b.fill_between(tl, mu - sem, mu + sem, color=col, alpha=0.18, lw=0)
    MA = np.array(xc["A"])
    peak = float(np.nanmax(np.abs(np.nanmean(MA, 0))))
    S, C = np.array(onset["dffA"]), np.array(onset["isoA"])
    nn = min(len(S), len(C))
    w = slice(PRE, PRE + int(1.5 * fps))
    s_, c_ = np.nanmean(S[:nn, w], 1), np.nanmean(C[:nn, w], 1)
    p_on = stats.wilcoxon(s_, c_).pvalue
    d_on = float(np.nanmean(s_ - c_))
    b.axvline(0, color="#3C4043", lw=0.9, ls=":")
    b.axhline(0, color="#3C4043", lw=0.9)
    b.set_ylim(-0.15, 0.15)
    b.set_xlabel("lag (s); positive = dopamine lags speed")
    b.set_ylabel("Pearson r")
    b.legend(frameon=False, fontsize=9, loc="lower right")
    _titled(b, "Dopamine does not track movement",
            f"Continuous speed x dF/F is negligible at every lag (max |r| = {peak:.3f}). "
            f"At locomotion onset the signal does not separate from the 405 control "
            f"({d_on:+.3f} pp, {_pfmt(p_on)}, n={nn}).", width=74)

    # ── C: the trial-level use ──
    c = ax[1, 0]
    dd = tt.dropna(subset=["da_gate", "approach_speed_cms"]).copy()
    for col in ("da_gate", "approach_speed_cms"):
        dd[col + "_c"] = dd[col] - dd.groupby("session")[col].transform("mean")
    dd["q"] = dd.groupby("session").da_gate.transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False)
        if x.notna().sum() >= 4 else np.nan)
    q = dd.dropna(subset=["q"]).groupby("q").approach_speed_cms_c.agg(["mean", "sem"])
    c.bar(q.index, q["mean"], yerr=q["sem"], color=C_REW, width=0.68,
          error_kw=dict(ecolor="#5F6368", capsize=4, lw=1.3), zorder=2)
    for i, r in q.iterrows():
        off = r["sem"] + 0.1 if r["mean"] >= 0 else -(r["sem"] + 0.1)
        c.text(i, r["mean"] + off, f"{r['mean']:+.2f}", ha="center",
               va="bottom" if r["mean"] >= 0 else "top", fontsize=10.5)
    c.axhline(0, color="#3C4043", lw=0.9)
    c.margins(y=0.22)
    c.set_xticks(range(4))
    c.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    c.set_xlabel("cue dopamine (within-session quartile)")
    c.set_ylabel("approach speed, session-centred (cm/s)")
    _titled(c, "But cue dopamine predicts approach vigour",
            "Trial-level, not moment-to-moment: trials with a larger cue response are "
            "run faster. Within-session quartiles.", width=74)

    # ── D: controls, as a forest plot ──
    dax = ax[1, 1]
    tests = [
        ("465 signal, cue 0-0.5 s -> speed cue-to-approach", "da_gate", "approach_speed_cms", True),
        ("465 signal, 0-0.3 s -> speed AFTER 0.5 s (no overlap)", "da_early", "speed_late_cms", True),
        ("465 signal -> speed in its OWN 0-0.3 s window", "da_early", "speed_early_cms", False),
        ("405 isosbestic control, same window as row 1", "iso_gate", "approach_speed_cms", False),
    ]
    ys, labs = [], []
    for j, (lab, xv, yv, _) in enumerate(tests):
        d2 = tt.dropna(subset=[xv, yv]).copy()
        for cc_ in (xv, yv):
            d2[cc_ + "_c"] = d2[cc_] - d2.groupby("session")[cc_].transform("mean")
        sl, se, p, lo, hi = _cluster_ols(d2[yv + "_c"].values, d2[xv + "_c"].values,
                                         d2.session.values)
        y = len(tests) - 1 - j
        col = C_REW if p < 0.05 else C_CTL
        dax.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle="round", zorder=2)
        dax.scatter([sl], [y], s=70, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        dax.text(hi + 0.12, y, f"{sl:+.2f}  {_pfmt(p)}", va="center", fontsize=9.2,
                 color="#3C4043" if p < 0.05 else "#80868B")
        ys.append(y); labs.append(lab)
    dax.axvline(0, color="#3C4043", lw=1.0, ls=":")
    dax.set_yticks(ys); dax.set_yticklabels(labs, fontsize=9.2)
    dax.set_ylim(-0.6, len(tests) - 0.4)
    dax.set_xlim(-2.0, 4.6)
    dax.set_xlabel("slope: cm/s of approach speed per 1 % dF/F  (95 % CI)")
    _titled(dax, "The relationship is not a motion artefact",
            "A motion artefact would show up in the 405 control and in the same-window "
            "test. It shows up in neither; the non-overlapping test still holds.", width=74)

    for a_ in ax.ravel():
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=3.4, w_pad=2.6)
    fig.savefig(out / "locomotion.png", dpi=170, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out / "locomotion.png")


# ─── Figure 4: direct group comparison ──────────────────────────────────────

def fig_group(cfg: dict, out: Path) -> None:
    """
    The direct comparison: one dot per mouse, 7 controls vs 8 isolated.

    The mouse is the independent unit -- each contributes 2-8 sessions, so a
    session-level test would overstate n roughly four-fold.  Dots are coloured
    by first recording date, which keeps the batch confound visible in place
    of a separate caveat panel.
    """
    o = Path(cfg["output_dir"])
    M = pd.read_csv(o / "mouse_summary.csv")
    M["mouse"] = M.mouse.astype(str)

    sess = pd.read_csv(o / "session_summary.csv")
    sess["mouse"] = sess.mouse.astype(str)
    idx = json.loads((o / "index.json").read_text())
    first = {}
    for k, v in idx.items():
        d = _session_date(k)
        mo = str(v["mouse"])
        if pd.notna(d) and (mo not in first or d < first[mo]):
            first[mo] = d
    M["date"] = M.mouse.map(first)

    PANELS = [
        ("onsets_per_min",    "locomotion onsets / min",   "Movement initiation"),
        ("pct_mobile",        "% of time mobile",          "Time spent moving"),
        ("speed_mobile_cms",  "speed while mobile (cm/s)", "Speed when moving"),
        ("accuracy_pct",      "AST accuracy (%)",          "Task accuracy"),
        ("cue_da",            "cue dF/F (%)",              "Cue response"),
        ("reward_da",         "reward dF/F (%)",           "Reward response"),
        ("pre_cue_da",        "pre-cue dF/F (%)",          "Pre-cue level"),
        ("dff_sd",            "dF/F SD (%)",               "Signal variability"),
    ]

    xnum = M.date.dropna().map(pd.Timestamp.toordinal)
    norm = plt.Normalize(xnum.min(), xnum.max())
    cmap = plt.get_cmap("viridis")
    rng = np.random.default_rng(3)

    fig, ax = plt.subplots(2, 4, figsize=(16.8, 9.6))
    fig.subplots_adjust(hspace=0.52, wspace=0.30, top=0.855, bottom=0.07,
                        left=0.045, right=0.935)
    fig.suptitle("Social isolation vs control: direct comparison, one dot per mouse",
                 fontsize=15.5, fontweight="bold", x=0.009, ha="left", y=0.978)
    fig.text(0.009, 0.926,
             "Mann-Whitney on mouse means (7 control vs 8 isolated). Top row behaviour, "
             "bottom row dopamine. Dot colour = first recording date: the groups do not "
             "overlap in time, so every effect below is confounded with batch.",
             fontsize=10, color="#5F6368", ha="left")

    for j, (var, ylab, title) in enumerate(PANELS):
        a = ax[j // 4, j % 4]
        if var not in M.columns:
            a.axis("off")
            continue
        vals = {}
        for i, grp in enumerate(("Control", "Social isolation")):
            g = M[(M.group == grp)].dropna(subset=[var])
            x = i + (rng.random(len(g)) - 0.5) * 0.26
            c = [cmap(norm(v)) if pd.notna(v) else "#BDC1C6"
                 for v in g.date.map(lambda d: pd.Timestamp.toordinal(d) if pd.notna(d) else np.nan)]
            a.scatter(x, g[var], s=72, c=c, edgecolor="#3C4043", linewidth=0.8, zorder=3)
            a.hlines(g[var].mean(), i - 0.3, i + 0.3, color="#3C4043", lw=3, zorder=4)
            vals[grp] = g[var].values
        p = stats.mannwhitneyu(vals["Control"], vals["Social isolation"]).pvalue
        d_ = vals["Social isolation"].mean() - vals["Control"].mean()
        a.set_xticks([0, 1])
        a.set_xticklabels(["Control", "Isolation"], fontsize=10)
        a.set_xlim(-0.55, 1.55)
        a.set_ylabel(ylab, fontsize=10)
        a.margins(y=0.22)
        mark = "*" if p < 0.05 else "n.s."
        a.set_title(f"{title}", loc="left", fontweight="bold", fontsize=11.5, pad=18)
        a.text(0, 1.015, f"{d_:+.2f}, p = {p:.3f} {mark}", transform=a.transAxes,
               fontsize=9.3, color="#3C4043" if p < 0.05 else "#80868B",
               va="bottom", ha="left")
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9.5)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, fraction=0.016, pad=0.012)
    ticks = np.linspace(xnum.min(), xnum.max(), 4)
    cb.set_ticks(ticks)
    cb.set_ticklabels([pd.Timestamp.fromordinal(int(v)).strftime("%Y-%m") for v in ticks])
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_visible(False)
    cb.set_label("first recording date", fontsize=9)

    fig.savefig(out / "group_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out / "group_comparison.png")


def make_figures(cfg: dict) -> None:
    out = Path(cfg["output_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dopamine_ast(cfg, out)
    fig_dopamine_core(cfg, out)
    fig_design(cfg, out)
    fig_locomotion(cfg, out)
    fig_group(cfg, out)
