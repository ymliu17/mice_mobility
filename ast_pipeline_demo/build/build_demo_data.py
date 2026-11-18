#!/usr/bin/env python3
"""
Build the data bundle for the offline AST pipeline demo.

Reads existing pipeline outputs for one AST session (S1 SuperAnimal pose,
S2 preprocessed arrays, S3 model_all syllables, S6 fiber-joint session NPZ,
custom-DLC predictions, raw TDT block, provisional pilot events/trials) and writes:

  ast_pipeline_demo/data/demo_data.js   (window.DEMO = {...})
  ast_pipeline_demo/frames/f_XXXX.jpg   (video clip, 10 fps)

Run from the project root with the `mice` environment:
  /home/yangliu/miniconda3/envs/mice/bin/python ast_pipeline_demo/build/build_demo_data.py

Nothing in the project outputs is modified.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import cv2
import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "ast_pipeline_demo"
sys.path.insert(0, str(ROOT))

from stages.s2_preprocessing import (apply_confidence_mask, interpolate_gaps,  # noqa: E402
                                     smooth_savgol, normalize_body_length)
from stages.s6_fiber_joint import _led_onset, _rolling_baseline  # noqa: E402

# ─── Session selection ───────────────────────────────────────────────────────

BLOCK = "fiberphotometrysetup-250621-135128_1864968-1-250621-152216"
PILOT_ID = "1864968-1-250621-152216_Cam1"
SA_TAG = "_Cam1_superanimal_topviewmouse_hrnet_w32_fasterrcnn_mobilenet_v3_large_fpn"
SA_KEY = BLOCK + SA_TAG
TDT_DIR = ROOT / "Data/Fiber photometry/AST/Social isolation/20250621" / BLOCK.split("_")[0] / BLOCK.split("_")[1]
AVI = TDT_DIR / f"{BLOCK}_Cam1.avi"
CU_H5 = TDT_DIR / f"{BLOCK}_Cam1DLC_Resnet50_AST20260806Aug6shuffle1_snapshot_best-190.h5"
SA_H5 = ROOT / "outputs/01_dlc_raw/fiber_photometry" / f"{SA_KEY}.h5"
S2_NPZ = ROOT / "outputs/02_preprocessed" / f"{SA_KEY}.npz"
RESULTS = ROOT / "outputs/03_moseq/kpms_project_all/model_all/results.h5"
PCA_P = ROOT / "outputs/03_moseq/kpms_project_all/pca.p"
S6_DIR = ROOT / "outputs/06_fiber_joint"
S6_NPZ = S6_DIR / "sessions" / f"{BLOCK}.npz"
PILOT = ROOT / "outputs/ast_reprocess_lawrence"
QC_DIR = ROOT / "outputs/methodlab/ast_tracking_qc_20250621"

CLIP_T = (674.0, 712.0)      # TDT seconds: IDS trial 4 (GateOpen 677.7 -> EatingStart 705.5)
CLIP_W, CLIP_H = 480, 360
N_TOP = 7                    # syllables >= 1 % global usage (model_all)

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
fj = cfg["fiber_joint"]


def r(a, nd=2):
    """Round an array to a JSON-friendly nested list, NaN -> None."""
    a = np.asarray(a, dtype=float)
    out = np.round(a, nd).astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def load_dlc(path, bodyparts=None):
    df = pd.read_hdf(path)
    sc = df.columns.get_level_values(0)[0]
    if df.columns.nlevels == 4:
        an = df.columns.get_level_values(1)[0]
        get = lambda b, c: df[(sc, an, b, c)].values          # noqa: E731
        bps = list(dict.fromkeys(df.columns.get_level_values(2)))
    else:
        get = lambda b, c: df[(sc, b, c)].values               # noqa: E731
        bps = list(dict.fromkeys(df.columns.get_level_values(1)))
    bps = bodyparts or bps
    xy = np.stack([np.column_stack([get(b, "x"), get(b, "y")]) for b in bps], 1)
    cf = np.column_stack([get(b, "likelihood") for b in bps])
    return xy.astype(float), cf.astype(float), bps


# ─── Load core arrays ────────────────────────────────────────────────────────

print("loading arrays ...")
sa_xy, sa_cf_raw, sa_bps = load_dlc(SA_H5)
# SuperAnimal writes likelihood -1 (and x, y = -1) when the detector finds no mouse
no_det = sa_cf_raw < 0
sa_cf = np.clip(sa_cf_raw, 0, 1)
sa_xy[no_det] = np.nan
CU_BPS = ["nose", "left_ear", "right_ear", "neck", "body_center", "tail_base", "tail_tip"]
cu_xy, cu_cf, _ = load_dlc(CU_H5, CU_BPS)
s6 = np.load(S6_NPZ, allow_pickle=True)
s6_meta = json.loads(bytes(s6["meta_json"]).decode())
cam_t = np.asarray(s6["cam_t"], float)                 # one fewer than video frames
n_native = len(s6["speed"])
s2 = np.load(S2_NPZ, allow_pickle=True)
s2_meta = json.loads(bytes(s2["metadata_json"]).decode())
res = h5py.File(RESULTS, "r")
syl = res[SA_KEY]["syllable"][:].astype(int)
T10 = len(syl)
assert T10 == len(sa_xy) == len(s2["coordinates"])

# native-frame time base (TDT s); extrapolate the final frame that lacks an epoc
t_native = np.concatenate([cam_t, [cam_t[-1] + 1 / fj["fps"]]])[:n_native]
t10 = t_native[np.minimum(np.arange(T10) * 2, n_native - 1)]   # S1 row j ~ native frame 2j

# ─── Global syllable ordering (model_all, all 246 sessions) ──────────────────

cnt = np.zeros(100)
for k in res:
    cnt += np.bincount(res[k]["syllable"][:], minlength=100)
usage_global = cnt / cnt.sum()
top = [int(i) for i in np.argsort(-usage_global)[:N_TOP]]
code_of = {s: i for i, s in enumerate(top)}            # 0..6 top, 7 = other


def code(s):
    return code_of.get(int(s), N_TOP)


codes10 = np.array([code(s) for s in syl])

# ─── Clip frames ─────────────────────────────────────────────────────────────

print("extracting clip frames ...")
nat = np.where((t_native >= CLIP_T[0]) & (t_native < CLIP_T[1]))[0]
nat = nat[nat % 2 == 0]
rows = nat // 2
fdir = DEMO / "frames"
fdir.mkdir(parents=True, exist_ok=True)
for old in fdir.glob("f_*.jpg"):
    old.unlink()
cap = cv2.VideoCapture(str(AVI))
src_w, src_h = int(cap.get(3)), int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(nat[0]))
pos = int(nat[0])
files = []
want = set(int(i) for i in nat)
while pos <= nat[-1]:
    ok, fr = cap.read()
    if not ok:
        break
    if pos in want:
        fn = f"f_{len(files):04d}.jpg"
        cv2.imwrite(str(fdir / fn), cv2.resize(fr, (CLIP_W, CLIP_H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 74])
        files.append(fn)
        if len(files) == 1:
            cv2.imwrite(str(DEMO / "frames" / "arena.jpg"),
                        cv2.resize(fr, (CLIP_W, CLIP_H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
    pos += 1
cap.release()
assert len(files) == len(nat), (len(files), len(nat))

# ─── S2: recompute each step on the full session (SuperAnimal, 10 fps) ───────

print("S2 stepwise preprocessing ...")
pp = cfg["preprocessing"]
use_bps = cfg["keypoints"]["use_bodyparts"]
use_idx = [sa_bps.index(b) for b in use_bps]
masked = apply_confidence_mask(sa_xy, sa_cf, pp["conf_threshold"])
interp = interpolate_gaps(masked, pp["max_interp_gap"])
smooth = smooth_savgol(interp, pp["smooth_window"], pp["smooth_polyorder"])
norm, bl = normalize_body_length(smooth, sa_bps, tuple(cfg["keypoints"]["body_length_parts"]))
stored = s2["coordinates"].astype(float)
both = np.isfinite(norm) & np.isfinite(stored)
verify = float(np.nanmax(np.abs(norm[both] - stored[both]))) if both.any() else None
print(f"  recomputed vs stored S2: max |diff| = {verify}, body length {bl:.2f}px vs {s2_meta['mean_body_length_px']:.2f}")

lowconf = (sa_cf < pp["conf_threshold"]).mean(0)
nan_after = np.isnan(interp[..., 0]).mean(0)
per_kp = [dict(name=b, use=b in use_bps, lowconf=round(float(lowconf[i]) * 100, 1),
               nan_after=round(float(nan_after[i]) * 100, 1),
               median_conf=round(float(np.median(sa_cf[:, i])), 3))
          for i, b in enumerate(sa_bps)]

blen = np.linalg.norm(smooth[:, sa_bps.index("nose")] - smooth[:, sa_bps.index("tail_base")], axis=1)
blen_1hz = np.array([np.nanmedian(blen[i:i + 10]) if np.isfinite(blen[i:i + 10]).any() else np.nan
                     for i in range(0, T10, 10)])

ctr_px = smooth[:, sa_bps.index("mouse_center")]
ok = np.isfinite(ctr_px).all(1)
H, xe, ye = np.histogram2d(ctr_px[ok, 0], ctr_px[ok, 1], bins=[64, 48], range=[[0, src_w], [0, src_h]])
occupancy = (H.T / ok.sum()).tolist()          # rows = y bins

# 'any use_bodypart missing' fraction after interpolation (what MoSeq sees as masked)
use_missing = np.isnan(interp[:, use_idx, 0]).any(1).mean()

# ─── S1: confidence summaries ────────────────────────────────────────────────

print("S1 summaries ...")


def heat(cf, nb=360):
    edges = np.linspace(0, len(cf), nb + 1).astype(int)
    return [[round(float(cf[a:b, k].mean()), 3) for a, b in zip(edges[:-1], edges[1:])]
            for k in range(cf.shape[1])]


qc = pd.read_csv(QC_DIR / "keypoint_error_summary.csv")
qc_rows = []
for kp in ["nose", "left_ear", "right_ear", "neck", "body_center", "tail_base", "tail_tip"]:
    q = qc[qc.keypoint == kp].set_index("model")
    qc_rows.append(dict(kp=kp, definition=q.loc["new", "definition"], n=int(q.loc["new", "n_labeled"]),
                        new_med=round(float(q.loc["new", "median_error_px"]), 2),
                        old_med=round(float(q.loc["old", "median_error_px"]), 2),
                        new_rmse=round(float(q.loc["new", "rmse_px"]), 2),
                        old_rmse=round(float(q.loc["old", "rmse_px"]), 2),
                        new_cov=round(float(q.loc["new", "coverage_likelihood_ge06"]) * 100, 1),
                        old_cov=round(float(q.loc["old", "coverage_likelihood_ge06"]) * 100, 1)))

s1 = dict(
    sa_bodyparts=sa_bps, cu_bodyparts=CU_BPS,
    sa_frac=[round(float(x) * 100, 1) for x in (sa_cf >= pp["conf_threshold"]).mean(0)],
    cu_frac=[round(float(x) * 100, 1) for x in (cu_cf >= fj["conf_threshold"]).mean(0)],
    sa_heat=heat(sa_cf), cu_heat=heat(cu_cf), heat_t1=float(t_native[-1]),
    qc=qc_rows,
    n_sa=int(len(sa_xy)), n_cu=int(len(cu_xy)),
    sa_nodet=round(float(no_det.all(1).mean()) * 100, 1),
)

# ─── S3: model, usage, transitions, durations ───────────────────────────────

print("S3 summaries ...")
pca = joblib.load(PCA_P)
idx = json.load(open(S6_DIR / "index.json"))
ast_sessions = []
for k in res:
    blk = k.split("_Cam")[0]
    if blk in idx and idx[blk]["test"] == "AST":
        ast_sessions.append((k, idx[blk]["group"]))


def rle(c):
    b = np.flatnonzero(np.diff(c)) + 1
    starts = np.concatenate([[0], b])
    lens = np.diff(np.concatenate([starts, [len(c)]]))
    return starts, lens, c[starts]


def usage8(s):
    c = np.array([code(x) for x in s])
    return [round(float((c == i).mean()), 4) for i in range(N_TOP + 1)]


cohort = []
for k, g in ast_sessions:
    s = res[k]["syllable"][:]
    lab = k.split("_")[1].split("_Cam")[0]
    cohort.append(dict(session=lab, group=g, usage=usage8(s), current=(k == SA_KEY)))

st, ln, cd = rle(syl)
bout_codes = np.array([code(x) for x in cd])
trans = np.zeros((N_TOP + 1, N_TOP + 1))
for a, b in zip(bout_codes[:-1], bout_codes[1:]):
    if a != b:
        trans[a, b] += 1
rowsum = trans.sum(1, keepdims=True)
trans_p = np.divide(trans, rowsum, out=np.zeros_like(trans), where=rowsum > 0)

durations = {}
for i, s in enumerate(top):
    d = ln[cd == s] / 10.0
    if len(d):
        q = np.percentile(d, [10, 25, 50, 75, 90])
        h, _ = np.histogram(np.clip(d, 0, 5), bins=np.arange(0, 5.01, 0.2))
        durations[str(s)] = dict(n=int(len(d)), q=[round(float(x), 2) for x in q],
                                 mean=round(float(d.mean()), 2), hist=h.tolist())

# ─── S3 example outputs: key syllables (S4-5 signatures) + F/S/X features ────
# F = usage, S = frame-level self-transition P(i->i), X = transition-row entropy
# (bits), exactly as in stages/s4_classification.py::build_session_feature_bank.

print("key syllables ...")
import shutil  # noqa: E402

KEY = [26, 11, 0, 24, 7]
KEY_TEXT = {
    26: ("Passive coping / freezing-like state", "abnormal", "A"),
    11: ("Flexible exploratory / choice-point behavior", "normal", "A"),
    0: ("Secondary inactive stable state", "abnormal", "A"),
    24: ("Focused deliberation / structured navigation", "normal", "B"),
    7: ("Agitated maladaptive strategy", "abnormal", "B"),
}
GIF_DIR = ROOT / "outputs/05_stage5/key_syllable_selectedclip_trajectories"
gdir = DEMO / "assets" / "syllables"
gdir.mkdir(parents=True, exist_ok=True)
for s in KEY:
    shutil.copy(GIF_DIR / f"s{s:02d}_selectedclip_trajectory.gif", gdir / f"s{s:02d}.gif")
shutil.copy(GIF_DIR / "key5_selectedclip_trajectory_strip.gif", gdir / "key5_strip.gif")

sigA = pd.read_csv(ROOT / "outputs/05_stage5/pca_analysis_R/signature_A_pc1_loadings.csv").set_index("feature")
sigB = pd.read_csv(ROOT / "outputs/05_stage5/pca_analysis_R/signature_B_pc2_loadings.csv").set_index("feature")
lcolA = "mean_loading"
lcolB = [c for c in sigB.columns if c not in ("feature_type", "node")][0]


def fsx(seq):
    seq = np.asarray(seq, int)
    T = len(seq)
    tr = np.zeros((100, 100))
    np.add.at(tr, (seq[:-1], seq[1:]), 1)
    rs = tr.sum(1)
    out = {}
    for s in KEY:
        F = float((seq == s).sum() / T)
        S = float(tr[s, s] / rs[s]) if rs[s] > 0 else 0.0
        if rs[s] > 0:
            p = tr[s] / rs[s]
            p = p[p > 0]
            X = float(-(p * np.log2(p)).sum())
        else:
            X = 0.0
        out[s] = (F, S, X)
    return out


man = pd.read_csv(GIF_DIR / "representative_manifest_reconstructed.csv")
win = pd.read_csv(GIF_DIR / "selected_windows.csv")
cohort_fsx = [dict(session=c["session"], group=c["group"], current=c["current"],
                   v={str(s): [round(x, 4) for x in vals] for s, vals in fsx(res[k]["syllable"][:]).items()})
              for c, (k, _) in zip(cohort, ast_sessions)]
key_syl = []
for s in KEY:
    sig = sigA if KEY_TEXT[s][2] == "A" else sigB
    col = lcolA if KEY_TEXT[s][2] == "A" else lcolB
    load = [round(float(sig.loc[f"{p}_s{s:02d}", col]), 3) for p in
            ("freq50", "transsummary_self_trans", "transsummary_row_entropy")]
    loadA = [round(float(sigA.loc[f"{p}_s{s:02d}", lcolA]), 3) for p in
             ("freq50", "transsummary_self_trans", "transsummary_row_entropy")]
    loadB = [round(float(sigB.loc[f"{p}_s{s:02d}", lcolB]), 3) for p in
             ("freq50", "transsummary_self_trans", "transsummary_row_entropy")]
    m = man[man.syllable == s].iloc[0]
    wq = win[win.syllable == s].iloc[0]
    lab = res[m.session]["syllable"][:]
    plotted = lab[int(wq.stable_window_start_frame_processed):int(wq.stable_window_end_frame_processed)]
    vals, cnts = np.unique(plotted, return_counts=True)
    key_syl.append(dict(
        syl=s, text=KEY_TEXT[s][0], enriched=KEY_TEXT[s][1], signature=KEY_TEXT[s][2],
        loadings=load, loadings_A=loadA, loadings_B=loadB,
        gif=f"assets/syllables/s{s:02d}.gif",
        source_session=m.session.split("_superanimal")[0], clip=m.clip_name,
        window=[int(wq.stable_window_start_frame_processed), int(wq.stable_window_end_frame_processed)],
        model_labels={str(int(v)): int(c) for v, c in zip(vals, cnts)},
        label_verified=bool(len(vals) == 1 and int(vals[0]) == s),
        session_usage=round(float(np.mean(syl == s)), 4),
        session_bouts=int((cd == s).sum()),
    ))

s3_examples = dict(key=key_syl, cohort_fsx=cohort_fsx, this_fsx={str(s): [round(x, 4) for x in v]
                                                                for s, v in fsx(syl).items()})

s3 = dict(
    examples=s3_examples,
    top=top, n_top=N_TOP,
    usage_global=[round(float(usage_global[s]), 4) for s in top] + [round(float(1 - usage_global[top].sum()), 4)],
    usage_session=usage8(syl),
    n_states_used=int((np.bincount(syl, minlength=100) > 0).sum()),
    n_states_global=int((cnt > 0).sum()),
    cohort=cohort,
    trans=r(trans_p, 3), trans_n=trans.astype(int).tolist(),
    durations=durations,
    rle=np.column_stack([st, ln, bout_codes]).astype(int).tolist(),
    rle_syl=cd.astype(int).tolist(),
    n_frames=T10, t0=float(t10[0]), dt=0.1,
    pca_evr=[round(float(x), 4) for x in pca.explained_variance_ratio_],
    n_sessions_model=len(res.keys()),
)

# ─── S6: dF/F method steps from the raw TDT block ────────────────────────────

print("S6 dF/F steps from raw TDT ...")
import tdt  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    blk = tdt.read_block(str(TDT_DIR), store=["465A", "405A"])
fs = float(blk.streams["_465A"].fs)
a = np.asarray(blk.streams["_465A"].data, float)
rr = np.asarray(blk.streams["_405A"].data, float)
n = min(len(a), len(rr))
a, rr = a[:n], rr[:n]
i0 = max(_led_onset(a, fs), _led_onset(rr, fs)) + int(fj["settle_s"] * fs)
i0 = min(i0, n - int(60 * fs)) if n > int(120 * fs) else 0
b_, aa_ = butter(2, fj["lowpass_hz"] / (fs / 2), "low")
fa = filtfilt(b_, aa_, a[i0:])
fr = filtfilt(b_, aa_, rr[i0:])
F0a = _rolling_baseline(fa, fs, fj["baseline_window_s"], fj["baseline_pct"])
F0r = _rolling_baseline(fr, fs, fj["baseline_window_s"], fj["baseline_pct"])
F0a = np.maximum(F0a, 0.2 * np.median(F0a))
F0r = np.maximum(F0r, 0.2 * np.median(F0r))
da, dr = fa - F0a, fr - F0r
A = np.vstack([dr, np.ones_like(dr)]).T
coef, *_ = np.linalg.lstsq(A, da, rcond=None)
motion = A @ coef
r2 = 1 - ((da - motion) ** 2).sum() / ((da - da.mean()) ** 2).sum()
An = np.vstack([fr, np.ones_like(fr)]).T
cn, *_ = np.linalg.lstsq(An, fa, rcond=None)
r2_naive = 1 - ((fa - An @ cn) ** 2).sum() / ((fa - fa.mean()) ** 2).sum()
dff_raw = (da - motion) / F0a * 100
t_tdt = (np.arange(n - i0) + i0) / fs


def block_mean(x, step):
    m = len(x) // step
    return x[:m * step].reshape(m, step).mean(1)


step1 = int(round(fs))              # ~1 Hz
raw_t0 = np.arange(0, i0) / fs
pre = dict(t=r(block_mean(raw_t0, step1), 1), a=r(block_mean(a[:i0], step1), 1),
           r=r(block_mean(rr[:i0], step1), 1)) if i0 > step1 else None
W0, W1 = 664.0, 724.0
w = (t_tdt >= W0) & (t_tdt < W1)
stepw = int(round(fs / 20))
dff_steps = dict(
    fs=round(fs, 3), settle_s=round(i0 / fs, 2), r2=round(float(r2), 3), r2_naive=round(float(r2_naive), 3),
    slope=round(float(coef[0]), 3), slope_naive=round(float(cn[0]), 3),
    full=dict(t=r(block_mean(t_tdt, step1), 1), a=r(block_mean(fa, step1), 1), r=r(block_mean(fr, step1), 1),
              F0a=r(block_mean(F0a, step1), 1), F0r=r(block_mean(F0r, step1), 1),
              da=r(block_mean(da, step1), 2), dr=r(block_mean(dr, step1), 2)),
    pre=pre,
    win=dict(t=r(block_mean(t_tdt[w], stepw), 2), da=r(block_mean(da[w], stepw), 2),
             dr=r(block_mean(dr[w], stepw), 2), motion=r(block_mean(motion[w], stepw), 2),
             dff=r(block_mean(dff_raw[w], stepw), 3)),
)
del blk, a, rr, fa, fr, F0a, F0r, da, dr, A, motion, dff_raw, t_tdt

# ─── S6: events, overview, peri-event, trials, cohort ────────────────────────

print("S6 overview / peri-event / trials ...")
ev = pd.read_csv(PILOT / "pilot_events_v1/pilot_events.csv")
ev = ev[ev.session == PILOT_ID].sort_values("tdt_time_s")
events = [dict(t=round(float(t), 3), e=e, s=s) for t, e, s in zip(ev.tdt_time_s, ev.event, ev.stage)]

stage_bands = []
for sname, g in ev.groupby("stage", sort=False):
    stage_bands.append(dict(stage=sname, t0=float(g.tdt_time_s.min()), t1=float(g.tdt_time_s.max())))
stage_bands.sort(key=lambda d: d["t0"])

dffA, isoA = np.asarray(s6["dffA"], float), np.asarray(s6["isoA"], float)
speed_cm = np.asarray(s6["speed"], float) * fj["cm_per_body_length"]
valid6 = np.asarray(s6["ok5"], bool)


def ds(x, k):
    m = len(x) // k
    with np.errstate(all="ignore"):
        return np.nanmean(x[:m * k].reshape(m, k), 1)


K5 = 4                                     # 20 Hz -> 5 Hz
syl5 = codes10[np.minimum(np.arange(len(t_native) // K5) * 2, T10 - 1)]
overview = dict(
    fps=5, t=r(ds(t_native, K5), 2), dff=r(ds(dffA, K5), 2), iso=r(ds(isoA, K5), 2),
    speed=r(ds(speed_cm, K5), 1), valid=r(ds(valid6.astype(float), K5), 2), syl=syl5.tolist(),
)

PRE, POST, FPS = fj["peri_pre_s"], fj["peri_post_s"], fj["fps"]
nb, na = int(PRE * FPS), int(POST * FPS)
b0, b1 = fj["peri_baseline"]
peri = dict(t=r(np.arange(-nb, na) / FPS, 2), events={})
for name in ["GateOpen", "ApproachCorrect", "ApproachIncorrect", "CorrectDigStart", "IncorrectDigStart", "EatingStart"]:
    sel = ev[ev.event == name]
    tr, tri, stg, tt = [], [], [], []
    for t, sname in zip(sel.tdt_time_s, sel.stage):
        i = int(np.searchsorted(cam_t, t))
        if i - nb < 0 or i + na >= len(dffA):
            continue
        seg, segi = dffA[i - nb:i + na], isoA[i - nb:i + na]
        bl_ = np.nanmean(seg[int((b0 + PRE) * FPS):int((b1 + PRE) * FPS)])
        bli = np.nanmean(segi[int((b0 + PRE) * FPS):int((b1 + PRE) * FPS)])
        if not np.isfinite(bl_):
            continue
        tr.append(seg - bl_)
        tri.append(segi - bli)
        stg.append(sname)
        tt.append(round(float(t), 2))
    peri["events"][name] = dict(dff=r(np.array(tr), 3), iso=r(np.array(tri), 3), stage=stg, t=tt)

tt_ = pd.read_csv(PILOT / "pilot_integrated_v1/pilot_unified_trial_table_PROVISIONAL.csv")
tt_ = tt_[tt_.session == PILOT_ID]
keep = {
    "trial_number": "trial", "stage": "stage", "stage_trial_number": "stage_trial",
    "gate_open_tdt_s": "gate_t", "first_dig_event": "first_dig", "first_dig_tdt_s": "dig_t",
    "choice_latency_s": "latency", "outcome_from_annotations": "outcome",
    "approach_correct_count": "app_c", "approach_incorrect_count": "app_i",
    "pre_choice_median_strict_speed_cm_s_PROVISIONAL": "speed_med",
    "pre_choice_raw_all5_pct": "track_pct",
    "first_dig_dffA_0to1s_minus_prebaseline_pct": "dig_dff",
    "gate_dffA_early": "gate_dff_early", "gate_dffA_late": "gate_dff_late",
    "combined_qc_flags": "qc",
}
trials = []
for _, row in tt_.iterrows():
    d = {}
    for src, dst in keep.items():
        v = row[src]
        if isinstance(v, (float, np.floating)):
            d[dst] = None if not np.isfinite(v) else round(float(v), 3)
        elif isinstance(v, (int, np.integer)):
            d[dst] = int(v)
        else:
            d[dst] = None if pd.isna(v) else str(v)
    trials.append(d)

# cohort peri-event from historical Stage 6 outputs (sessions, not verified animals)
em = pd.read_csv(S6_DIR / "event_metrics.csv")
et = np.load(S6_DIR / "event_traces.npz", allow_pickle=True)
cohort_peri = {}
for name in ["GateOpen", "ApproachCorrect", "CorrectDigStart", "EatingStart"]:
    key = f"{name}|ALL|A|dff"
    if key not in et.files:
        continue
    rows_ = em[(em.event == name) & (em.stage == "ALL") & (em.ch == "A")].reset_index(drop=True)
    M = et[key]
    assert len(rows_) == len(M), (name, len(rows_), len(M))
    out = {}
    for g in ["Control", "Social isolation"]:
        sel = (rows_.test == "AST") & (rows_.group == g)
        X = M[sel.values]
        if len(X):
            out[g] = dict(mean=r(np.nanmean(X, 0), 3),
                          sem=r(np.nanstd(X, 0, ddof=1) / math.sqrt(len(X)) if len(X) > 1 else np.zeros(X.shape[1]), 3),
                          n=int(len(X)), n_mice_prefix=int(rows_[sel].mouse.nunique()))
    cohort_peri[name] = out

fig_src = ROOT / "outputs/06_fiber_joint/ast_timecourses/per_video" / f"{BLOCK}.png"
(DEMO / "assets" / "figures").mkdir(parents=True, exist_ok=True)
shutil.copy(fig_src, DEMO / "assets" / "figures" / "s6_per_video_timecourse.png")
qc6 = pd.read_csv(ROOT / "outputs/06_fiber_joint/ast_timecourses/session_QC.csv")
qc6_row = qc6[qc6.apply(lambda r_: BLOCK in " ".join(map(str, r_.values)), axis=1)]
qc6_row = {k: (None if pd.isna(v) else (round(float(v), 3) if isinstance(v, (float, np.floating)) else str(v)))
           for k, v in (qc6_row.iloc[0].to_dict().items() if len(qc6_row) else {})}

s6_out = dict(
    example_fig=dict(src="assets/figures/s6_per_video_timecourse.png",
                     origin=str(fig_src.relative_to(ROOT)),
                     script="behavior_methodlab/plot_ast_timecourses.py", session_qc=qc6_row),
    meta=dict(r2A=round(s6_meta["r2A"], 3), settle_s=round(s6_meta["settle_s_A"], 1),
              n_frames=s6_meta["n_video_frames"], n_epocs=s6_meta["n_cam_epocs"],
              BL_px=round(float(s6["BL"]), 2), yield5=round(float(valid6.mean()) * 100, 1)),
    dff_steps=dff_steps, overview=overview, events=events, stage_bands=stage_bands,
    peri=peri, trials=trials, cohort_peri=cohort_peri,
)

# ─── Clip payload ────────────────────────────────────────────────────────────

clip = dict(
    files=files, fps=10, n=len(files), w=CLIP_W, h=CLIP_H, src_w=src_w, src_h=src_h,
    native=nat.astype(int).tolist(), rows=rows.astype(int).tolist(),
    t=r(t_native[nat], 3),
    sa_xy=r(sa_xy[rows], 1), sa_cf=r(sa_cf_raw[rows], 3),
    s2_interp=r(interp[rows], 1), s2_smooth=r(smooth[rows], 1),
    cu_xy=r(cu_xy[nat], 1), cu_cf=r(cu_cf[nat], 3),
    syl=codes10[rows].tolist(), syl_raw=syl[rows].astype(int).tolist(),
    dff=r(dffA[nat], 3), speed=r(speed_cm[nat], 1),
)

# ─── Meta / config ───────────────────────────────────────────────────────────

meta = dict(
    block=BLOCK, mouse="1864968-1", date="2025-06-21", group="Social isolation", cam="Cam1",
    context="AST (attentional set-shifting task)",
    video=dict(fps=20, n_frames=n_native, w=src_w, h=src_h, duration_s=round(n_native / 20, 1)),
    tdt_fs=round(fs, 3), n_events=len(events), n_trials=len(trials),
    clip_t=list(CLIP_T), sa_rows=T10, s2_verify_max_diff=verify,
    s2_body_length_px=round(bl, 2), s2_use_missing_pct=round(float(use_missing) * 100, 1),
    s2_nan_pct=round(float(np.isnan(stored).mean()) * 100, 1),
    built_from=dict(
        s1_superanimal=str(SA_H5.relative_to(ROOT)), s1_custom=str(CU_H5.relative_to(ROOT)),
        s2=str(S2_NPZ.relative_to(ROOT)), s3=str(RESULTS.relative_to(ROOT)),
        s6=str(S6_NPZ.relative_to(ROOT)),
        events=str((PILOT / "pilot_events_v1/pilot_events.csv").relative_to(ROOT)),
        trials=str((PILOT / "pilot_integrated_v1/pilot_unified_trial_table_PROVISIONAL.csv").relative_to(ROOT)),
        tdt=str(TDT_DIR.relative_to(ROOT)), video=str(AVI.relative_to(ROOT)),
    ),
)
config = dict(
    dlc=cfg["dlc"], keypoints=dict(use_bodyparts=use_bps, anchor_part=cfg["keypoints"]["anchor_part"],
                                   body_length_parts=cfg["keypoints"]["body_length_parts"]),
    preprocessing=pp, moseq=cfg["moseq"], joint_all_model=cfg["joint_all_model"], fiber_joint=fj,
)

bundle = dict(meta=meta, config=config, clip=clip, s1=s1,
              s2=dict(per_kp=per_kp, blen=r(blen_1hz, 1), blen_t0=float(t10[0]), occupancy=occupancy,
                      occ_max=float(np.max(H) / ok.sum())),
              s3=s3, s6=s6_out)

out = DEMO / "data" / "demo_data.js"
out.parent.mkdir(parents=True, exist_ok=True)
txt = json.dumps(bundle, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
out.write_text("window.DEMO=" + txt + ";\n")
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(files)} frames")
