#!/usr/bin/env python3
"""
Stage 6 – Fiber photometry x markerless kinematics (joint analysis).

Pairs each TDT photometry block with the matching DeepLabCut prediction from the
custom 5-keypoint AST project, computes a bleaching-robust dF/F, frame-aligns it
to the video, and derives peri-event dopamine and per-trial AST measures.

Steps (run in order; each is skippable once its outputs exist):

  index    pair DLC .h5 predictions with TDT block directories -> index.json
  traces   per session: kinematics + bleaching-robust dF/F, frame-aligned -> sessions/*.npz
  events   parse .tnt / .nxt behavioural annotations            -> events.csv
  peri     peri-event averages and per-session metrics          -> event_traces.npz, event_metrics.csv
  trials   per-trial AST table (outcome, kinematics, dopamine)  -> trial_table.csv
  figures  summary figures                                      -> figures/*.png

The dF/F is deliberately *not* the naive "regress raw 465 on raw 405" estimate.
Both channels are detrended against their own rolling-percentile baseline first,
so the isosbestic regression removes shared motion artefact rather than shared
bleaching.  See `dff_robust` for the rationale.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from scipy.signal import butter, filtfilt, savgol_filter

logger = logging.getLogger("pipeline.s6")

# ─── Defaults (overridable from config.yaml: fiber_joint:) ───────────────────

DEFAULTS = dict(
    data_dir="Data/Fiber photometry",
    dlc_project="Data/Fiber photometry/AST20260806-Justin Zielinski-2026-08-06",
    output_dir="outputs/06_fiber_joint",
    keypoints=["left_ear", "right_ear", "neck", "body_center", "tail_base"],
    conf_threshold=0.6,
    max_interp_gap=10,
    smooth_window=9,          # ~0.45 s at 20 fps
    smooth_polyorder=2,
    fps=20.0,
    cm_per_body_length=6.28,  # neck->tail-base length, measured on the rig
    lowpass_hz=10.0,
    baseline_window_s=120.0,
    baseline_pct=10,
    settle_s=30.0,
    peri_pre_s=2.0,
    peri_post_s=4.0,
    peri_baseline=(-2.0, -1.0),
    min_events=5,
    # trial-level dopamine windows, all baselined to [-2, -1] s before the cue
    gate_window=(0.0, 0.5),
    approach_window=(0.0, 0.5),
    eat_window=(0.0, 2.0),
)

EVENTS = [
    "EatingStart", "GateOpen", "ApproachCorrect", "ApproachIncorrect",
    "CorrectDigStart", "IncorrectDigStart", "CR", "Miss",
]

_STAGE_SUFFIX = re.compile(r"-(SD|CD|RL|IDS|EDS)$")
_NXT_LINE = re.compile(r"^(\S+)\s+@\s+([\d.]+)\s+->\s+(\d+)")


# ─── dF/F ────────────────────────────────────────────────────────────────────

def _led_onset(x: np.ndarray, fs: float, frac: float = 0.5) -> int:
    """First index where the LED is on and stays on for >= 1 s."""
    thr = frac * np.median(x[x > np.percentile(x, 50)])
    on = (x > thr).astype(float)
    w = max(1, int(fs))
    c = np.convolve(on, np.ones(w) / w, "same")
    idx = int(np.argmax(c > 0.95))
    return idx if c[idx] > 0.95 else 0


def _rolling_baseline(x: np.ndarray, fs: float, win_s: float, pct: int,
                      dec: int = 100) -> np.ndarray:
    """Rolling low-percentile baseline, computed on a decimated copy for speed."""
    xd = x[::dec]
    w = max(3, int(win_s * fs / dec) | 1)
    bd = percentile_filter(xd, pct, size=w, mode="nearest")
    return np.interp(np.arange(len(x)), np.arange(len(xd)) * dec, bd)


def dff_robust(sig465, ref405, fs: float, cfg: dict) -> dict:
    """
    Bleaching-robust dF/F.

    1. discard the LED turn-on transient plus `settle_s`
    2. low-pass both channels
    3. give *each* channel its own rolling-percentile F0, which removes that
       channel's bleaching before the two are compared
    4. regress detrended 405 onto detrended 465 -> shared (motion) component
    5. dF/F = (detrended 465 - motion) / F0_465 * 100

    Regressing the *raw* traces instead inflates R^2 to ~0.8 because both
    channels bleach together; that shared decay is then subtracted out of the
    signal along with the artefact.  After detrending, R^2 reflects genuine
    shared artefact only (typically ~0.1-0.2 here).

    Returns full-length arrays, NaN before the settle point.
    """
    a = np.asarray(sig465, float)
    r = np.asarray(ref405, float)
    n = min(len(a), len(r))          # some blocks drop a few samples on one channel
    a, r = a[:n], r[:n]

    i0 = max(_led_onset(a, fs), _led_onset(r, fs)) + int(cfg["settle_s"] * fs)
    i0 = min(i0, n - int(60 * fs)) if n > int(120 * fs) else 0

    b, aa = butter(2, cfg["lowpass_hz"] / (fs / 2), "low")
    fa = filtfilt(b, aa, a[i0:])
    fr = filtfilt(b, aa, r[i0:])

    F0a = _rolling_baseline(fa, fs, cfg["baseline_window_s"], cfg["baseline_pct"])
    F0r = _rolling_baseline(fr, fs, cfg["baseline_window_s"], cfg["baseline_pct"])
    F0a = np.maximum(F0a, 0.2 * np.median(F0a))
    F0r = np.maximum(F0r, 0.2 * np.median(F0r))

    da, dr = fa - F0a, fr - F0r
    A = np.vstack([dr, np.ones_like(dr)]).T
    coef, *_ = np.linalg.lstsq(A, da, rcond=None)
    motion = A @ coef
    r2 = 1 - ((da - motion) ** 2).sum() / ((da - da.mean()) ** 2).sum()

    dff = np.full(n, np.nan)
    iso = np.full(n, np.nan)
    dff[i0:] = (da - motion) / F0a * 100.0
    iso[i0:] = dr / F0r * 100.0
    return dict(dff=dff, iso=iso, r2=float(r2), i0=i0, slope=float(coef[0]))


# ─── Kinematics ──────────────────────────────────────────────────────────────

def _interp_gaps(a: np.ndarray, maxgap: int) -> np.ndarray:
    """Linear interpolation over NaN runs no longer than `maxgap` frames."""
    out = a.copy()
    for j in range(a.shape[1]):
        v = a[:, j]
        nan = np.isnan(v)
        if nan.all() or not nan.any():
            continue
        i = np.arange(len(v))
        out[:, j] = np.interp(i, i[~nan], v[~nan])
        d = np.diff(np.concatenate([[0], nan.astype(np.int8), [0]]))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s > maxgap:
                out[s:e, j] = np.nan
    return out


def kinematics(h5path: str, cfg: dict) -> dict:
    """Speed (body lengths/s), angular velocity and body bend from the DLC h5."""
    kp = cfg["keypoints"]
    df = pd.read_hdf(h5path)
    scorer = df.columns.get_level_values(0)[0]
    X = np.stack([np.column_stack([df[(scorer, b, "x")].values,
                                   df[(scorer, b, "y")].values]) for b in kp], 1)
    L = np.column_stack([df[(scorer, b, "likelihood")].values for b in kp])
    X[L < cfg["conf_threshold"]] = np.nan
    ok5 = (L >= cfg["conf_threshold"]).all(1)

    Xi = _interp_gaps(X.reshape(len(X), -1), cfg["max_interp_gap"]).reshape(X.shape)

    w = cfg["smooth_window"]
    flat = Xi.reshape(len(Xi), -1).copy()
    for j in range(flat.shape[1]):
        v = flat[:, j]
        m = np.isfinite(v)
        if m.sum() > w:
            v[m] = savgol_filter(v[m], w, cfg["smooth_polyorder"])
            flat[:, j] = v
    Xs = flat.reshape(Xi.shape)

    g = {b: i for i, b in enumerate(kp)}
    ctr = Xs[:, g["body_center"], :]
    neck = Xs[:, g["neck"], :]
    tb = Xs[:, g["tail_base"], :]
    earm = (Xs[:, g["left_ear"], :] + Xs[:, g["right_ear"], :]) / 2

    BL = float(np.nanmedian(np.linalg.norm(neck - tb, axis=1)))   # px
    fps = cfg["fps"]
    d = np.diff(ctr, axis=0, prepend=ctr[[0]])
    speed = np.linalg.norm(d, axis=1) * fps / BL                  # body lengths / s

    ax = neck - tb
    head = np.arctan2(ax[:, 1], ax[:, 0])
    angvel = np.abs(np.angle(np.exp(1j * np.diff(head, prepend=head[:1])))) * fps

    v1, v2 = earm - ctr, tb - ctr
    cosb = np.sum(v1 * v2, 1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    bend = np.degrees(np.arccos(np.clip(cosb, -1, 1)))

    return dict(speed=speed, angvel=angvel, bend=bend, ctr=ctr, ok5=ok5,
                BL=BL, valid=np.isfinite(ctr).all(1))


# ─── Event annotations ───────────────────────────────────────────────────────

def parse_events(tdt_dir: str) -> List[Tuple[float, str, str]]:
    """
    Read Synapse note annotations: `.tnt` holds the note names (1-indexed),
    `*SavedNotes.nxt` the timestamped references into that list.

    Returns (time_s, event, stage) sorted by time, where `stage` is IDS / EDS /
    BASE depending on the attentional-set-shift phase encoded in the name.
    """
    tnt = glob.glob(f"{tdt_dir}/*.tnt")
    nxt = glob.glob(f"{tdt_dir}/*SavedNotes.nxt")
    if not tnt or not nxt:
        return []
    names = [l.strip() for l in
             open(tnt[0], encoding="utf-8", errors="replace").read().splitlines()
             if l.strip()][1:]
    out = []
    for line in open(nxt[0], encoding="utf-8", errors="replace"):
        m = _NXT_LINE.match(line.strip())
        if not m:
            continue
        k = int(m.group(3))
        if not 0 < k <= len(names):
            continue
        nm = names[k - 1]
        stage = "IDS" if nm.endswith("-IDS") else ("EDS" if nm.endswith("-EDS") else "BASE")
        out.append((float(m.group(2)), _STAGE_SUFFIX.sub("", nm), stage))
    return sorted(out)


# ─── Step 1: index ───────────────────────────────────────────────────────────

def _session_key(h5name: str) -> str:
    """`...-240416-140112_1756456-1-240416-143809_Cam1DLC_...h5` -> session key."""
    base = os.path.basename(h5name)
    m = re.match(r"(.+?)_Cam\d+DLC_", base)
    return m.group(1) if m else base.split("DLC_")[0]


def step_index(cfg: dict) -> dict:
    """Pair each DLC prediction with its TDT block directory."""
    proj, data = cfg["dlc_project"], cfg["data_dir"]

    tdt_dirs = {}
    for d in glob.glob(f"{data}/**/", recursive=True):
        d = d.rstrip("/")
        if not glob.glob(f"{d}/*.tsq"):           # a TDT block has a .tsq
            continue
        parent, sub = os.path.basename(os.path.dirname(d)), os.path.basename(d)
        tdt_dirs[f"{parent}_{sub}"] = d

    index = {}
    for h5 in glob.glob(f"{proj}/**/*.h5", recursive=True):
        if "labeled-data" in h5:
            continue
        key = _session_key(h5)
        if key not in tdt_dirs:
            logger.debug("no TDT block for %s", key)
            continue
        tdt = tdt_dirs[key]
        parts = Path(tdt).parts
        # .../Fiber photometry/<test>/<group>/<date>/<setup>/<block>
        try:
            i = parts.index("Fiber photometry")
            test, group = parts[i + 1], parts[i + 2]
        except (ValueError, IndexError):
            logger.warning("cannot infer test/group from %s", tdt)
            continue
        cam = re.search(r"_(Cam\d+)DLC_", os.path.basename(h5))
        index[key] = dict(h5=h5, tdt=tdt, cam=cam.group(1) if cam else "Cam1",
                          test=test, group=group, mouse=key.split("_")[1].split("-")[0])

    out = Path(cfg["output_dir"]) / "index.json"
    out.write_text(json.dumps(index, indent=1))
    logger.info("indexed %d sessions -> %s", len(index), out)
    return index


# ─── Step 2: traces ──────────────────────────────────────────────────────────

def step_traces(cfg: dict, index: dict, force: bool = False) -> None:
    """Per session: kinematics + frame-aligned bleaching-robust dF/F."""
    import tdt

    sess_dir = Path(cfg["output_dir"]) / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    fps = cfg["fps"]

    for n, (s, v) in enumerate(sorted(index.items()), 1):
        op = sess_dir / f"{s}.npz"
        if op.exists() and not force:
            continue
        try:
            K = kinematics(v["h5"], cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                blk = tdt.read_block(v["tdt"],
                                     store=["465A", "405A", "465C", "405C", v["cam"]])
            fs = float(blk.streams["_465A"].fs)
            cam_t = np.asarray(blk.epocs[v["cam"]].onset)
            nf = len(K["speed"])

            rec, meta = {}, {}
            for ch in ("A", "C"):
                r = dff_robust(blk.streams[f"_465{ch}"].data,
                               blk.streams[f"_405{ch}"].data, fs, cfg)
                rec[f"dff{ch}"] = _frame_align(r["dff"], fs, cam_t, nf, fps)
                rec[f"iso{ch}"] = _frame_align(r["iso"], fs, cam_t, nf, fps)
                meta[f"r2{ch}"] = r["r2"]
                meta[f"settle_s_{ch}"] = r["i0"] / fs

            np.savez_compressed(
                op, speed=K["speed"], angvel=K["angvel"], bend=K["bend"],
                ctr=K["ctr"], valid=K["valid"], ok5=K["ok5"], BL=K["BL"],
                cam_t=cam_t[:nf], fs=fs, **rec,
                meta_json=np.bytes_(json.dumps(dict(
                    session=s, test=v["test"], group=v["group"], mouse=v["mouse"],
                    cam=v["cam"], n_video_frames=nf, n_cam_epocs=len(cam_t),
                    fps=fps, **meta))))
            logger.info("[%d/%d] %s R2=%.3f/%.3f yield=%.0f%%", n, len(index),
                        s[-26:], meta["r2A"], meta["r2C"], 100 * K["ok5"].mean())
        except Exception as e:                                   # noqa: BLE001
            logger.error("[%d/%d] FAILED %s: %s: %s", n, len(index), s[-26:],
                         type(e).__name__, e)


def _frame_align(x: np.ndarray, fs: float, cam_t: np.ndarray, nframes: int,
                 fps: float) -> np.ndarray:
    """Mean of `x` within each video frame's 1/fps exposure window."""
    out = np.full(nframes, np.nan)
    n = min(nframes, len(cam_t))
    i0 = (cam_t[:n] * fs).astype(int)
    i1 = ((cam_t[:n] + 1.0 / fps) * fs).astype(int)
    for k in range(n):
        a, b = i0[k], max(i1[k], i0[k] + 1)
        if 0 <= a < len(x):
            out[k] = np.nanmean(x[a:min(b, len(x))])
    return out


# ─── Step 3: events ──────────────────────────────────────────────────────────

def step_events(cfg: dict, index: dict) -> pd.DataFrame:
    rows = []
    for s, v in sorted(index.items()):
        for t, ev, stage in parse_events(v["tdt"]):
            rows.append(dict(session=s, test=v["test"], group=v["group"],
                             mouse=v["mouse"], t=t, event=ev, stage=stage))
    df = pd.DataFrame(rows)
    out = Path(cfg["output_dir"]) / "events.csv"
    df.to_csv(out, index=False)
    logger.info("%d events across %d sessions -> %s",
                len(df), df.session.nunique() if len(df) else 0, out)
    return df


# ─── Step 4: peri-event ──────────────────────────────────────────────────────

def step_peri(cfg: dict, index: dict, events: pd.DataFrame) -> pd.DataFrame:
    """Session-mean peri-event traces, for dF/F and for the 405 isosbestic."""
    sess_dir = Path(cfg["output_dir"]) / "sessions"
    pre, post = cfg["peri_pre_s"], cfg["peri_post_s"]
    b0, b1 = cfg["peri_baseline"]
    fps = cfg["fps"]
    nb, na = int(pre * fps), int(post * fps)

    traces: Dict[str, list] = {}
    rows = []
    for s, v in sorted(index.items()):
        f = sess_dir / f"{s}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        cam = d["cam_t"]
        meta = json.loads(bytes(d["meta_json"]).decode())
        e = events[events.session == s]
        if not len(e):
            continue
        for name in EVENTS:
            for stage in ("ALL", "IDS", "EDS", "BASE"):
                sel = e[(e.event == name) & ((e.stage == stage) if stage != "ALL" else True)]
                ts = sel.t.values
                if len(ts) < cfg["min_events"]:
                    continue
                for ch in ("A", "C"):
                    for kind in ("dff", "iso"):
                        x = d[f"{kind}{ch}"]
                        segs = []
                        for t in ts:
                            i = int(np.searchsorted(cam, t))
                            if i - nb < 0 or i + na >= len(x):
                                continue
                            seg = x[i - nb:i + na]
                            bl = np.nanmean(seg[int((b0 + pre) * fps):int((b1 + pre) * fps)])
                            if np.isfinite(bl):
                                segs.append(seg - bl)
                        if len(segs) < cfg["min_events"]:
                            continue
                        M = np.nanmean(np.stack(segs), 0)
                        traces.setdefault(f"{name}|{stage}|{ch}|{kind}", []).append(M)
                        if kind == "dff":
                            rows.append(dict(
                                session=s, test=v["test"], group=v["group"],
                                mouse=v["mouse"], event=name, stage=stage, ch=ch,
                                n_ev=len(segs), r2=meta[f"r2{ch}"],
                                peak=float(np.nanmax(M[int(pre * fps):int((pre + 2) * fps)])),
                                auc0_2=float(np.nanmean(M[int(pre * fps):int((pre + 2) * fps)]))))

    out = Path(cfg["output_dir"])
    df = pd.DataFrame(rows)
    df.to_csv(out / "event_metrics.csv", index=False)
    np.savez_compressed(out / "event_traces.npz",
                        **{k: np.array(v) for k, v in traces.items()},
                        fps=fps, pre=pre, post=post)
    logger.info("%d peri-event rows -> %s", len(df), out / "event_metrics.csv")
    return df


# ─── Step 5: AST trial table ─────────────────────────────────────────────────

def step_trials(cfg: dict, index: dict, events: pd.DataFrame) -> pd.DataFrame:
    """
    One row per AST trial, delimited by GateOpen.

    outcome is taken from the dig event inside the trial (CorrectDigStart ->
    correct, IncorrectDigStart -> incorrect, neither -> miss).  All dopamine
    measures use channel A and are baselined to [-2, -1] s before the cue, so
    they are comparable within a trial regardless of event latency.
    """
    sess_dir = Path(cfg["output_dir"]) / "sessions"
    fps, k_cm = cfg["fps"], cfg["cm_per_body_length"]
    gw, aw, ew = cfg["gate_window"], cfg["approach_window"], cfg["eat_window"]

    rows = []
    for s, v in sorted(index.items()):
        if v["test"] != "AST":
            continue
        f = sess_dir / f"{s}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        cam, dff, iso, speed, ctr = (d["cam_t"], d["dffA"], d["isoA"],
                                     d["speed"], d["ctr"])
        n = min(len(cam), len(dff), len(speed))
        cam, dff, iso, speed, ctr = cam[:n], dff[:n], iso[:n], speed[:n], ctr[:n]

        e = events[events.session == s].sort_values("t")
        gates = e[e.event == "GateOpen"].t.values
        if not len(gates):
            continue

        def mean_in(a: float, b: float, y: np.ndarray) -> float:
            m = (cam >= a) & (cam < b)
            return float(np.nanmean(y[m])) if m.any() else np.nan

        for i, tg in enumerate(gates):
            t_end = gates[i + 1] if i + 1 < len(gates) else cam[-1]
            w = e[(e.t >= tg) & (e.t < t_end)]
            base = mean_in(tg - 2.0, tg - 1.0, dff)
            base = 0.0 if not np.isfinite(base) else base

            app = w[w.event.isin(["ApproachCorrect", "ApproachIncorrect"])]
            if not len(app):
                continue
            t_app, app_name = app.t.values[0], app.event.values[0]

            if (w.event == "CorrectDigStart").any():
                outcome = "correct"
            elif (w.event == "IncorrectDigStart").any():
                outcome = "incorrect"
            else:
                outcome = "miss"

            m_app = (cam >= tg) & (cam < t_app)
            sp = speed[m_app]
            sp = sp[np.isfinite(sp)]

            # Non-overlapping variant: dopamine read before the animal has
            # really moved, speed read only after that window has closed.
            # This is the version that can claim prediction, not correlation.
            sp_late = speed[(cam >= tg + 0.5) & (cam < t_app)]
            sp_late = sp_late[np.isfinite(sp_late)]
            sp_early = speed[(cam >= tg) & (cam < tg + 0.3)]
            sp_early = sp_early[np.isfinite(sp_early)]

            # Approach geometry: is this window a directed approach at all?
            P = ctr[m_app]
            P = P[np.isfinite(P).all(1)]
            if len(P) > 1:
                path = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
                straight = (float(np.linalg.norm(P[-1] - P[0]) / path)
                            if path > 0 else np.nan)
            else:
                path, straight = np.nan, np.nan

            base_i = mean_in(tg - 2.0, tg - 1.0, iso)
            base_i = 0.0 if not np.isfinite(base_i) else base_i

            eat = w[w.event == "EatingStart"]
            da_eat = (mean_in(eat.t.values[0] + ew[0], eat.t.values[0] + ew[1], dff) - base
                      if len(eat) else np.nan)

            rows.append(dict(
                session=s, test=v["test"], group=v["group"], mouse=v["mouse"],
                trial=i, stage=w.stage.iloc[0] if len(w) else "BASE",
                outcome=outcome, approach=app_name, latency_s=float(t_app - tg),
                approach_speed=float(sp.mean()) if len(sp) else np.nan,
                approach_peakspeed=float(sp.max()) if len(sp) else np.nan,
                # absolute pre-cue level: a trial-state measure, NOT baselined.
                # The other da_* columns are cue-baselined evoked responses.
                da_pre=mean_in(tg - 2.0, tg - 1.0, dff),
                da_gate=mean_in(tg + gw[0], tg + gw[1], dff) - base,
                da_approach=mean_in(t_app + aw[0], t_app + aw[1], dff) - base,
                da_eat=da_eat,
                # controls for the vigour analysis
                da_early=mean_in(tg, tg + 0.3, dff) - base,
                iso_gate=mean_in(tg + gw[0], tg + gw[1], iso) - base_i,
                speed_late=float(sp_late.mean()) if len(sp_late) else np.nan,
                speed_early=float(sp_early.mean()) if len(sp_early) else np.nan,
                path_px=path, straightness=straight))

    df = pd.DataFrame(rows)
    if len(df):
        for col in ("approach_speed", "approach_peakspeed", "speed_late", "speed_early"):
            df[col + "_cms"] = df[col] * k_cm
    out = Path(cfg["output_dir"]) / "trial_table.csv"
    df.to_csv(out, index=False)
    logger.info("%d AST trials across %d sessions -> %s",
                len(df), df.session.nunique() if len(df) else 0, out)
    return df


# ─── Entry point ─────────────────────────────────────────────────────────────

ALL_STEPS = ("index", "traces", "events", "peri", "trials", "summary", "figures")


def step_summary(cfg: dict) -> pd.DataFrame:
    """
    One row per mouse: locomotion, AST behaviour and dopamine side by side.

    The mouse is the independent unit for any group comparison -- each animal
    contributes 2-8 sessions, so session-level tests overstate n by 3-4x.
    """
    out = Path(cfg["output_dir"])
    fps, k_cm = cfg["fps"], cfg["cm_per_body_length"]

    rows = []
    for f in sorted((out / "sessions").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        m = json.loads(bytes(d["meta_json"]).decode())
        sp = np.where(d["valid"], d["speed"], np.nan)
        y = d["dffA"]
        n = min(len(sp), len(y))
        sp, y = sp[:n], y[:n]
        mob = np.nan_to_num(sp > 0.5)
        starts = np.diff(mob.astype(np.int8), prepend=0) == 1
        rows.append(dict(
            mouse=m["mouse"], group=m["group"], test=m["test"],
            speed_mean_cms=np.nanmean(sp) * k_cm,
            speed_mobile_cms=np.nanmean(sp[mob]) * k_cm if mob.any() else np.nan,
            pct_mobile=100 * np.nanmean(mob),
            distance_m=np.nansum(sp) / fps * k_cm / 100,
            onsets_per_min=starts.sum() / (n / fps / 60),
            dff_sd=np.nanstd(y)))
    sess = pd.DataFrame(rows)
    sess.to_csv(out / "session_summary.csv", index=False)

    per_mouse = sess.groupby(["mouse", "group"]).mean(numeric_only=True).reset_index()
    per_mouse["mouse"] = per_mouse.mouse.astype(str)

    tp = out / "trial_table.csv"
    if tp.exists():
        tt = pd.read_csv(tp)
        ast = tt.groupby("mouse").agg(
            approach_speed_cms=("approach_speed_cms", "mean"),
            approach_latency_s=("latency_s", "median"),
            accuracy_pct=("outcome", lambda x: (x == "correct").mean() * 100),
            n_trials=("trial", "size"),
            cue_da=("da_gate", "mean"),
            reward_da=("da_eat", "mean"),
            pre_cue_da=("da_pre", "mean"))
        ast.index = ast.index.astype(str)
        per_mouse = per_mouse.merge(ast, on="mouse", how="left")

    per_mouse.to_csv(out / "mouse_summary.csv", index=False)
    logger.info("%d mice -> %s", len(per_mouse), out / "mouse_summary.csv")
    return per_mouse


def run(config_path: str, steps: List[str] | None = None, force: bool = False) -> None:
    warnings.filterwarnings("ignore")
    from utils.io_utils import load_config, resolve_path

    raw = load_config(config_path)
    base = str(Path(config_path).parent.resolve())
    cfg = dict(DEFAULTS)
    cfg.update(raw.get("fiber_joint", {}) or {})
    for k in ("data_dir", "dlc_project", "output_dir"):
        cfg[k] = resolve_path(base, cfg[k])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    steps = list(steps or ALL_STEPS)
    out = Path(cfg["output_dir"])

    index_path = out / "index.json"
    if "index" in steps or not index_path.exists():
        index = step_index(cfg)
    else:
        index = json.loads(index_path.read_text())

    if "traces" in steps:
        step_traces(cfg, index, force=force)

    events_path = out / "events.csv"
    if "events" in steps or not events_path.exists():
        events = step_events(cfg, index)
    else:
        events = pd.read_csv(events_path)

    if "peri" in steps:
        step_peri(cfg, index, events)
    if "trials" in steps:
        step_trials(cfg, index, events)
    if "summary" in steps:
        step_summary(cfg)
    if "figures" in steps:
        from stages.s6_fiber_figures import make_figures
        make_figures(cfg)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--steps", nargs="*", choices=ALL_STEPS, default=None)
    p.add_argument("--force", action="store_true", help="recompute existing session traces")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run(os.path.abspath(a.config), steps=a.steps, force=a.force)
