/* AST Pipeline Explorer — pages and routing. */
(function () {
  'use strict';
  const D = window.DEMO, C = window.Charts, P = window.Player;
  const main = document.getElementById('main');
  const clip = D.clip, M = D.meta;
  const pad2 = (s) => String(s).padStart(2, '0');
  const sName = (s) => 's' + pad2(s);
  const pct = (v, d = 1) => (v == null ? '–' : v.toFixed(d) + ' %');
  const nf = (v) => (v == null ? '–' : v.toLocaleString('en-US'));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  let players = [];

  // ── shared derived data ──────────────────────────────────────────
  const OV = D.s6.overview;
  const tMin = OV.t.map((t) => (t == null ? null : t / 60));
  const sessionDomain = [0, D.s1.heat_t1 / 60];
  const stageBands = D.s6.stage_bands.map((b) => ({ x0: b.t0 / 60, x1: b.t1 / 60, label: b.stage }));
  const clipWin = { x0: clip.t[0] / 60, x1: clip.t[clip.n - 1] / 60 };
  const sylSegs = D.s3.rle.map(([st, ln, code]) => [(D.s3.t0 + st * D.s3.dt) / 60, (D.s3.t0 + (st + ln) * D.s3.dt) / 60, code, st, ln]);
  const sylStarts = sylSegs.map((s) => s[0]);
  const RASTER_EVENTS = ['GateOpen', 'ApproachCorrect', 'ApproachIncorrect', 'CorrectDigStart', 'IncorrectDigStart', 'EatingStart', 'Miss'];
  const rasterRows = (unit = 60) => RASTER_EVENTS.map((e) => ({
    key: e, label: P.EVENT_META[e].label, color: P.EVENT_META[e].color,
    times: D.s6.events.filter((x) => x.e === e).map((x) => x.t / unit),
  }));
  const minFmt = (v) => `${Math.round(v)}′`;
  const minFmtLong = (v) => `${C.fmtClock(v * 60)} (TDT ${(v * 60).toFixed(1)} s)`;
  const sylLabelFull = (code, raw) => (raw != null ? sName(raw) : P.sylName(code));

  // ── building blocks ──────────────────────────────────────────────
  function head(o) {
    const chips = (o.chips || []).map((c) => `<span class="chip ${c.code ? 'code' : ''}">${c.k ? `<span class="k">${c.k}</span>` : ''}${esc(c.v)}</span>`).join('');
    return `<header class="page-head">
      <div class="eyebrow">${o.tag ? `<span class="tag">${o.tag}</span>` : ''}${o.eyebrow}</div>
      <h1>${o.title}</h1><p class="lede">${o.lede}</p><div class="chips">${chips}</div></header>`;
  }
  const io = (a, b, c) => `<div class="io">
      <div class="box"><h4>Inputs</h4><ul>${a.map((x) => `<li>${x}</li>`).join('')}</ul></div><div class="arrow">→</div>
      <div class="box"><h4>What this stage does</h4><ul>${b.map((x) => `<li>${x}</li>`).join('')}</ul></div><div class="arrow">→</div>
      <div class="box"><h4>Outputs</h4><ul>${c.map((x) => `<li>${x}</li>`).join('')}</ul></div></div>`;
  const tiles = (arr) => `<div class="tiles">${arr.map((t) => `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}${t.unit ? ` <small>${t.unit}</small>` : ''}</div>${t.note ? `<div class="note">${t.note}</div>` : ''}</div>`).join('')}</div>`;
  const section = (title, sub, body, id) => `<section class="section" ${id ? `id="${id}"` : ''}><h2>${title}</h2>${sub ? `<p class="sub">${sub}</p>` : ''}${body}</section>`;
  const card = (title, desc, body, cls = '') => `<div class="card ${cls}">${title ? `<h3>${title}</h3>` : ''}${desc ? `<p class="desc">${desc}</p>` : ''}${body}</div>`;
  const caveats = (items) => `<ul class="caveats">${items.map((i) => `<li><span>${i}</span></li>`).join('')}</ul>`;
  const legend = (items) => `<div class="legend">${items.map(([c, l, kind]) => `<span><i class="${kind || ''}" style="${kind === 'ring' ? `border-color:${c}` : `background:${c}`}"></i>${l}</span>`).join('')}</div>`;
  function params(obj, notes = {}, keys) {
    const ks = keys || Object.keys(obj);
    return `<div class="table-wrap"><table class="params"><thead><tr><th>parameter</th><th>value</th><th>meaning</th></tr></thead><tbody>${ks.map((k) => `<tr><td>${k}</td><td>${esc(JSON.stringify(obj[k]))}</td><td>${notes[k] || ''}</td></tr>`).join('')}</tbody></table></div>`;
  }
  function seg(name, options, on) {
    return `<div class="seg" data-seg="${name}">${options.map(([v, l]) => `<button data-v="${v}" class="${v === on ? 'on' : ''}">${l}</button>`).join('')}</div>`;
  }
  function bindSeg(name, fn, root = main) {
    const el = root.querySelector(`[data-seg="${name}"]`);
    el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      el.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
      fn(b.dataset.v);
    }));
  }
  const $ = (id) => document.getElementById(id);
  const tipHover = (el, html) => { el.addEventListener('pointermove', (e) => C.showTip(html, e.clientX, e.clientY)); el.addEventListener('pointerleave', C.hideTip); };

  function meanSem(rows) {
    const n = rows[0] ? rows[0].length : 0, mean = [], lo = [], hi = [];
    for (let j = 0; j < n; j++) {
      const v = rows.map((r) => r[j]).filter((x) => x != null && isFinite(x));
      const m = v.reduce((a, b) => a + b, 0) / (v.length || 1);
      const sd = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) : 0;
      const se = sd / Math.sqrt(v.length || 1);
      mean.push(v.length ? m : null); lo.push(v.length ? m - se : null); hi.push(v.length ? m + se : null);
    }
    return { mean, lo, hi };
  }

  // SVG dumbbell (manual-label error: SuperAnimal vs custom)
  function dumbbell(el, rows, o) {
    const render = () => {
      const w = el.clientWidth || 500, L = 96, R = 18, rowH = 26, T = 4, h = T + rows.length * rowH + 22;
      const xmax = o.xmax, sx = (v) => L + v / xmax * (w - L - R);
      let s = `<svg width="${w}" height="${h}" style="display:block">`;
      C.ticks(0, xmax, Math.max(2, Math.floor((w - L - R) / 70))).forEach((v) => {
        s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${h - 22}" stroke="${C.col('--grid')}"/><text x="${sx(v)}" y="${h - 6}" font-size="11" text-anchor="middle" fill="${C.col('--muted')}">${v}</text>`;
      });
      rows.forEach((r, k) => {
        const y = T + k * rowH + rowH / 2;
        s += `<text x="${L - 8}" y="${y + 4}" font-size="12" text-anchor="end" font-family="${C.cssVar('--mono')}" fill="${C.col('--ink-2')}">${r.kp}</text>`;
        s += `<line x1="${sx(r.a)}" x2="${sx(r.b)}" y1="${y}" y2="${y}" stroke="${C.col('--axis')}" stroke-width="2"/>`;
        s += `<circle cx="${sx(r.b)}" cy="${y}" r="5" fill="${C.col(o.colorB)}" stroke="${C.col('--surface')}" stroke-width="2" data-k="${k}"/>`;
        s += `<circle cx="${sx(r.a)}" cy="${y}" r="5" fill="${C.col(o.colorA)}" stroke="${C.col('--surface')}" stroke-width="2" data-k="${k}"/>`;
      });
      el.innerHTML = s + '</svg>';
      el.querySelectorAll('circle').forEach((c) => tipHover(c, rows[+c.dataset.k].tip));
    };
    new ResizeObserver(render).observe(el);
    C.registry.add({ el, render });
    render();
  }

  // SVG horizontal box plot on a log axis (bout durations)
  function boxPlot(el, rows, o) {
    const render = () => {
      const w = el.clientWidth || 500, L = 60, R = 18, rowH = 28, T = 4, h = T + rows.length * rowH + 22;
      const lo = Math.log10(o.xmin), hi = Math.log10(o.xmax);
      const sx = (v) => L + (Math.log10(Math.min(Math.max(v, o.xmin), o.xmax)) - lo) / (hi - lo) * (w - L - R);
      let s = `<svg width="${w}" height="${h}" style="display:block">`;
      o.ticks.forEach((v) => {
        s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${h - 22}" stroke="${C.col('--grid')}"/><text x="${sx(v)}" y="${h - 6}" font-size="11" text-anchor="middle" fill="${C.col('--muted')}">${v} s</text>`;
      });
      rows.forEach((r, k) => {
        const y = T + k * rowH + rowH / 2, [q10, q25, q50, q75, q90] = r.q, c = r.color;
        s += `<g data-k="${k}"><rect x="${L}" y="${y - rowH / 2}" width="${w - L - R}" height="${rowH}" fill="transparent"/>`;
        s += `<text x="${L - 8}" y="${y + 4}" font-size="12" text-anchor="end" font-family="${C.cssVar('--mono')}" fill="${C.col('--ink-2')}">${r.label}</text>`;
        s += `<line x1="${sx(q10)}" x2="${sx(q90)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="2" stroke-linecap="round"/>`;
        s += `<rect x="${sx(q25)}" y="${y - 7}" width="${Math.max(2, sx(q75) - sx(q25))}" height="14" rx="4" fill="${C.withAlpha(c, 0.25)}" stroke="${c}" stroke-width="1.5"/>`;
        s += `<line x1="${sx(q50)}" x2="${sx(q50)}" y1="${y - 8}" y2="${y + 8}" stroke="${C.col('--ink')}" stroke-width="2"/></g>`;
      });
      el.innerHTML = s + '</svg>';
      el.querySelectorAll('g[data-k]').forEach((g) => tipHover(g, rows[+g.dataset.k].tip));
    };
    new ResizeObserver(render).observe(el);
    C.registry.add({ el, render });
    render();
  }

  // ═══ Overview ════════════════════════════════════════════════════
  function pageOverview() {
    const nIDS = D.s6.trials.filter((t) => t.stage === 'IDS').length, nEDS = D.s6.trials.length - nIDS;
    const arrow = '<div class="arr"><svg viewBox="0 0 22 14"><path d="M0 7h18M13 2l6 5-6 5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></div>';
    const node = (o) => `<${o.href ? `a href="${o.href}"` : 'div'} class="node ${o.cls || ''}"><div class="n-tag">${o.tag}</div><div class="n-title">${o.title}</div><div class="n-body">${o.body}</div>${o.out ? `<div class="n-out">${o.out}</div>` : ''}</${o.href ? 'a' : 'div'}>`;
    main.innerHTML = `
      ${head({ eyebrow: 'Example AST recording', title: 'From raw video to syllables and dopamine-aligned trials', lede: `This page follows <b>one attentional set-shifting (AST) session</b> through the mice mobility pipeline: mouse <b>${M.mouse}</b> (${M.group}), recorded ${M.date} on ${M.cam}. Every chart uses that session's real pipeline outputs. Use the stage pages to see what goes into each stage and what comes out.`,
        chips: [{ k: 'block', v: M.block, code: true }, { k: 'video', v: `${M.video.w}×${M.video.h} · ${M.video.fps} fps` }, { k: 'TDT', v: `${M.tdt_fs} Hz` }] })}
      ${tiles([
        { label: 'Recording length', value: (M.video.duration_s / 60).toFixed(1), unit: 'min', note: `${nf(M.video.n_frames)} native frames` },
        { label: 'AST trials (GateOpen)', value: D.s6.trials.length, note: `IDS ${nIDS} · EDS ${nEDS}` },
        { label: 'Annotated events', value: M.n_events, note: 'Synapse notes (.tnt / .nxt)' },
        { label: 'Syllables used', value: D.s3.n_states_used, note: `of ${D.s3.n_states_global} states used in model_all` },
        { label: 'dF/F motion fit R²', value: D.s6.dff_steps.r2.toFixed(2), note: `naive raw regression: ${D.s6.dff_steps.r2_naive.toFixed(2)}` },
      ])}
      ${section('Pipeline map', 'Two lanes share the same raw recording. The behavior lane turns pose into syllables; the photometry lane links pose and dopamine to task events. Click a stage to open it.', `
        <div class="lanes">
          <div class="lane"><div class="lane-title"><span>Behavior lane · zero-shot pose → syllables</span><span>10 fps</span></div>
            <div class="flow">
              ${node({ tag: 'RAW', title: 'Video', body: `AVI from the arena camera, ${M.video.w}×${M.video.h}, ${M.video.fps} fps`, out: `${nf(M.video.n_frames)} frames`, cls: 'raw' })}${arrow}
              ${node({ tag: 'S1', title: 'Pose extraction', body: 'DeepLabCut SuperAnimal TopViewMouse, 27 keypoints, video downsampled to 10 fps', out: `${nf(D.s1.n_sa)} rows × 27 kp`, href: '#s1' })}${arrow}
              ${node({ tag: 'S2', title: 'Preprocessing', body: 'Confidence mask, gap interpolation, Savitzky–Golay smoothing, body-length scaling', out: `${nf(D.s1.n_sa)} × 7 kp → S3`, href: '#s2' })}${arrow}
              ${node({ tag: 'S3', title: 'Syllable model', body: 'keypoint-MoSeq <code>model_all</code>, shared across 246 sessions (κ = 1e5)', out: `${D.s3.n_states_used} syllables in this session`, href: '#s3' })}${arrow}
              ${node({ tag: 'S4–S5', title: 'Features & signatures', body: 'F / S / X features, PCA signatures A and B. Shown here only as S3 example outputs.', out: 'not covered in this demo', cls: 'dim' })}
            </div></div>
          <div class="lane"><div class="lane-title"><span>Photometry lane · custom pose + TDT → trials</span><span>20 fps</span></div>
            <div class="flow">
              ${node({ tag: 'RAW', title: 'Video + TDT block', body: '465/405 nm channels A & C, camera epocs, Synapse event notes', out: `${M.tdt_fs} Hz photometry`, cls: 'raw' })}${arrow}
              ${node({ tag: 'S1b', title: 'Custom AST pose', body: 'ResNet-50 DLC project trained on 151 labelled AST frames, 7 keypoints, native 20 fps', out: `${nf(D.s1.n_cu)} rows × 7 kp`, href: '#s1' })}${arrow}
              ${node({ tag: 'S6', title: 'Photometry × kinematics', body: 'Bleaching-robust dF/F, frame alignment, events, peri-event averages, trial table', out: `${D.s6.trials.length} trials · R² ${D.s6.dff_steps.r2.toFixed(2)}`, href: '#s6' })}
            </div></div>
          <div class="joint-link"><span><b>Joint view:</b> S3 syllables and S6 dopamine share the TDT clock through camera epocs. See the session timeline on the <a href="#s6">S6 page</a>.</span></div>
        </div>`)}
      ${section('How to read each stage page', null, `<div class="grid g3">
        ${card('1 · Inputs → outputs', null, '<p class="desc">What files go in, what the code does, what it writes, and the exact script and command.</p>')}
        ${card('2 · Interactive outputs', null, '<p class="desc">Real frames and arrays from this session. Drag across a time chart to zoom; double-click to reset. Hover any mark for values.</p>')}
        ${card('3 · Parameters', null, '<p class="desc">The exact <code>config.yaml</code> values each stage ran with, plus the script and command that produced the outputs.</p>')}
      </div>`)}
      <div class="footer-note">Built from existing project outputs by <code>build/build_demo_data.py</code>. Nothing in <code>outputs/</code> was modified. Open the <a href="#provenance">Provenance</a> page for every source file.</div>`;
  }

  // ═══ S1 ══════════════════════════════════════════════════════════
  function pageS1() {
    const q = D.s1.qc;
    main.innerHTML = `
      ${head({ tag: 'S1', eyebrow: 'Behavior lane · stage 1', title: 'Pose extraction', lede: 'DeepLabCut finds body keypoints in every video frame. Each keypoint gets x, y and a likelihood (confidence). Downstream stages decide which points to trust from that confidence.',
        chips: [{ v: 'stages/s1_pose_extraction.py', code: true }, { v: 'python pipeline.py run --stages 1', code: true }, { k: 'writes', v: 'outputs/01_dlc_raw/fiber_photometry/*.h5', code: true }] })}
      ${io([`Raw AVI, ${M.video.w}×${M.video.h} px, ${M.video.fps} fps, ${nf(M.video.n_frames)} frames`, '<code>config.yaml → dlc</code> and <code>contexts</code>'],
        ['Downsample to 10 fps (<code>target_fps</code>)', 'Faster R-CNN detector finds the mouse; HRNet-w32 places 27 TopViewMouse keypoints', 'Zero-shot: no AST-specific training'],
        [`H5 table: ${nf(D.s1.n_sa)} rows × 27 keypoints × (x, y, likelihood)`, 'CSV copy for inspection'])}
      <div class="callout" style="margin-top:12px"><b>Two trackers on the same video.</b> The behavior lane (S2 → S3) uses SuperAnimal's 27-point zero-shot model at 10 fps. Stage 6 instead uses a <b>custom AST ResNet-50 model</b> (7 keypoints, native 20 fps, <code>snapshot-best-190</code>), trained on 151 hand-labelled AST frames. Toggle between them below.</div>
      ${section('Look at the tracking', 'Clip: IDS trial 4 (gate opens at 677.7 s; incorrect approach, correct approach, correct dig, then eating). Hollow markers = likelihood below 0.6.', card(null, null, '<div id="p1"></div>'))}
      ${section('Tracking yield per keypoint', 'Percentage of frames in the whole session with likelihood ≥ 0.6. Bold names are the keypoints each stage actually uses.', `<div class="grid g2">
        ${card('SuperAnimal TopViewMouse · 27 keypoints', `${nf(D.s1.n_sa)} rows at 10 fps. Bold = the 7 points sent to keypoint-MoSeq.`, '<div id="b1a"></div>')}
        ${card('Custom AST model · 7 keypoints', `${nf(D.s1.n_cu)} rows at 20 fps. Bold = the 5 points Stage 6 uses for kinematics.`, '<div id="b1b"></div>')}
      </div>`)}
      ${section('Confidence across the session', 'Mean likelihood in about 12-s bins (no detection counted as 0). Pale bands show where the tracker loses the mouse (handling, occlusion by pots, digging).', card(null, null, `<div class="controls">${seg('heat', [['sa', 'SuperAnimal · 27'], ['cu', 'Custom · 7']], 'sa')}</div><div id="h1"></div>
        <div class="legend"><span>likelihood</span>${[0, 0.25, 0.5, 0.75, 1].map((v) => `<span><i style="background:${C.seq(v)}"></i>${v}</span>`).join('')}</div>`))}
      ${section('Accuracy against manual labels', `${q[0].n}–${Math.max(...q.map((r) => r.n))} labelled points per keypoint from 50 hand-annotated frames of this same video. Median pixel error per keypoint.`, `<div class="grid g-wide">
        ${card('Median error (px): SuperAnimal vs custom', null, `${legend([[C.col('--s2'), 'SuperAnimal (S1)'], [C.col('--s3'), 'Custom AST (S6)']])}<div id="d1"></div>
          <p class="chart-hint">body_center and tail_tip use the SuperAnimal proxies mouse_center and tail_end.</p>`)}
        ${card('Custom model evaluation (DLC)', 'From CombinedEvaluation-results.csv, 95 % train split, shuffle 1.', tiles([
          { label: 'Train RMSE', value: '0.87', unit: 'px' }, { label: 'Test RMSE', value: '4.29', unit: 'px' }, { label: 'Test RMSE, p ≥ 0.6', value: '2.10', unit: 'px' },
        ]))}
      </div>`)}
      ${section('Parameters', '<code>config.yaml → dlc</code>', card(null, null, params(D.config.dlc, {
        superanimal_name: 'model-zoo project', model_name: 'pose backbone', detector_name: 'mouse detector', scale_list: 'test-time scales (px)',
        video_adapt: 'self-training on the video (off)', target_fps: 'videos resampled before inference', conf_threshold: '', batch_size: 'pose batch', detector_batch_size: 'detector batch',
      })))}`;

    players.push(new P.ClipPlayer($('p1'), { mode: 's1' }));
    const useSet = new Set(D.config.keypoints.use_bodyparts), s6Set = new Set(D.config.fiber_joint.keypoints);
    C.barList($('b1a'), D.s1.sa_bodyparts.map((b, k) => ({ name: b, a: D.s1.sa_frac[k], strong: useSet.has(b), tip: `<div class="tt-h">${b}</div>${C.ttRow(null, 'frames ≥ 0.6', D.s1.sa_frac[k] + ' %')}` })), { max: 100, colorA: '--s2', fmt: (v) => v.toFixed(0) + '%' });
    C.barList($('b1b'), D.s1.cu_bodyparts.map((b, k) => ({ name: b, a: D.s1.cu_frac[k], strong: s6Set.has(b), tip: `<div class="tt-h">${b}</div>${C.ttRow(null, 'frames ≥ 0.6', D.s1.cu_frac[k] + ' %')}` })), { max: 100, colorA: '--s3', fmt: (v) => v.toFixed(0) + '%' });
    const bins = D.s1.sa_heat[0].length, binMin = D.s1.heat_t1 / 60 / bins;
    const mkHeat = (which) => {
      const names = which === 'sa' ? D.s1.sa_bodyparts : D.s1.cu_bodyparts, data = which === 'sa' ? D.s1.sa_heat : D.s1.cu_heat;
      const strong = which === 'sa' ? useSet : s6Set;
      return {
        rows: names, data, cellH: which === 'sa' ? 12 : 26, rowLabelW: 104, mono: true, gap: 0,
        rowStrong: (r) => strong.has(names[r]), color: (v) => C.seq(v), xAxis: { d0: 0, d1: D.s1.heat_t1 / 60, fmt: minFmt },
        tip: (r, c, v) => `<div class="tt-h">${names[r]}</div>${C.ttRow(null, 'time', `${(c * binMin).toFixed(1)}–${((c + 1) * binMin).toFixed(1)} min`)}${C.ttRow(null, 'mean likelihood', v.toFixed(2))}`,
      };
    };
    const heat = new C.Heatmap($('h1'), mkHeat('sa'));
    bindSeg('heat', (v) => { const o = mkHeat(v); heat.o.cellH = o.cellH; heat.update(o.data, o); });
    dumbbell($('d1'), q.map((r) => ({ kp: r.kp, a: r.old_med, b: r.new_med, tip: `<div class="tt-h">${r.kp} (${r.n} labels)</div>${C.ttRow(C.col('--s2'), 'SuperAnimal median', r.old_med + ' px')}${C.ttRow(C.col('--s3'), 'Custom median', r.new_med + ' px')}${C.ttRow(null, 'SA RMSE / custom RMSE', `${r.old_rmse} / ${r.new_rmse} px`)}${C.ttRow(null, 'coverage ≥ 0.6 (SA / custom)', `${r.old_cov}% / ${r.new_cov}%`)}` })), { xmax: Math.ceil(Math.max(...q.map((r) => r.old_med)) + 1), colorA: '--s2', colorB: '--s3' });
  }

  // ═══ S2 ══════════════════════════════════════════════════════════
  const S2_STEPS = [
    ['Raw', 'S1 output as written: every keypoint, including low-confidence guesses (hollow).'],
    ['Mask', 'Coordinates with likelihood < 0.6 are removed (<code>apply_confidence_mask</code>).'],
    ['Interpolate', 'Gaps of ≤ 10 frames (1 s at 10 fps) are filled linearly (<code>interpolate_gaps</code>). Ringed points are interpolated.'],
    ['Smooth', 'Savitzky–Golay filter (window 5, order 2) inside each contiguous valid block (<code>smooth_savgol</code>).'],
    ['Normalize', 'All coordinates are divided by the session median nose → tail_base distance (<code>normalize_body_length</code>). This is what S3 receives.'],
  ];
  function pageS2() {
    const use = D.config.keypoints.use_bodyparts;
    main.innerHTML = `
      ${head({ tag: 'S2', eyebrow: 'Behavior lane · stage 2', title: 'Keypoint preprocessing', lede: 'Turns noisy S1 detections into clean, scale-free coordinates: low-confidence points are removed, short gaps filled, jitter smoothed, and everything expressed in body lengths.',
        chips: [{ v: 'stages/s2_preprocessing.py', code: true }, { v: 'python pipeline.py run --stages 2', code: true }, { k: 'writes', v: 'outputs/02_preprocessed/*.npz', code: true }] })}
      ${io(['S1 H5 (27 kp × x, y, likelihood at 10 fps)', '<code>config.yaml → preprocessing</code>'],
        ['Mask → interpolate → smooth → normalize, per session', 'QC reports: confidence, missing data, body length'],
        ['NPZ: <code>coordinates</code> (T × 27 × 2, body lengths), <code>confidences</code>, <code>bodyparts</code>, metadata (body length px, fps)'])}
      ${tiles([
        { label: 'Recomputed vs stored S2', value: M.s2_verify_max_diff < 1e-4 ? 'match' : 'differs', note: `max |Δ| = ${M.s2_verify_max_diff.toExponential(1)} body lengths` },
        { label: 'Body length (median)', value: M.s2_body_length_px.toFixed(1), unit: 'px', note: 'nose → tail_base' },
      ])}
      ${section('Step through the cleaning', 'The overlay and the trace below both show the selected step. Click the trace to jump the video to that moment.', card(null, null, `
        <div class="controls">${seg('step', S2_STEPS.map(([n], k) => [String(k), `${k} · ${n}`]), '4')}</div>
        <p class="desc" id="stepText"></p>
        <div id="p2"></div>
        <div class="controls" style="margin-top:16px"><span class="ctl-label">Keypoint</span><select id="kpSel">${use.map((b) => `<option ${b === 'nose' ? 'selected' : ''}>${b}</option>`).join('')}</select>
          ${seg('coord', [['0', 'x'], ['1', 'y']], '0')}</div>
        <div id="legend2"></div><div id="tr2"></div><p class="chart-hint">Pixel coordinates over the 38-s clip. Drag to zoom, double-click to reset.</p>`))}
      <div class="grid g2 section">
        ${card('Body length over the session', '1-s medians of the smoothed nose → tail_base distance.', '<div id="bl2"></div>')}
        ${card('Arena occupancy', 'mouse_center position over the whole session (log scale), drawn on a frame from the clip.', '<div id="occ" class="chart"><canvas></canvas></div>')}
      </div>
      ${section('Parameters', '<code>config.yaml → preprocessing</code> and <code>keypoints</code>', card(null, null, params(Object.assign({}, D.config.preprocessing, D.config.keypoints), {
        conf_threshold: 'mask below this likelihood', max_interp_gap: 'longest gap filled (frames)', smooth_window: 'Savitzky–Golay window (frames)', smooth_polyorder: 'polynomial order',
        normalize_body_length: 'divide by median body length', session_standardize: 'per-session z-score (off)', use_bodyparts: 'subset passed to S3', anchor_part: 'egocentric anchor', body_length_parts: 'body-length definition',
      })))}`;

    const player = new P.ClipPlayer($('p2'), { mode: 's2' });
    players.push(player);
    let step = 4, kp = 'nose', coord = 0;
    const kIdx = () => D.s1.sa_bodyparts.indexOf(kp);
    const xs = clip.t;
    const traceSeries = () => {
      const k = kIdx(), thr = D.config.preprocessing.conf_threshold;
      const raw = clip.sa_xy.map((f) => (f[k] && f[k][coord] != null ? f[k][coord] : null));
      const cf = clip.sa_cf.map((f) => f[k]);
      const it = clip.s2_interp.map((f) => (f[k] && f[k][coord] != null ? f[k][coord] : null));
      const sm = clip.s2_smooth.map((f) => (f[k] && f[k][coord] != null ? f[k][coord] : null));
      const ok = raw.map((v, i) => (cf[i] >= thr ? v : null)), low = raw.map((v, i) => (cf[i] >= thr ? null : v));
      const filled = it.map((v, i) => (cf[i] >= thr ? null : v));
      const ser = [];
      if (step === 0) ser.push({ x: xs, y: low, name: 'raw, conf < 0.6', color: '--muted', points: true, hollow: true, r: 3 });
      ser.push({ x: xs, y: ok, name: 'raw, conf ≥ 0.6', color: '--ink-2', points: true, r: 2.2 });
      if (step >= 2) ser.push({ x: xs, y: filled, name: 'interpolated', color: '--s2', points: true, r: 3 });
      if (step >= 3) ser.push({ x: xs, y: sm, name: 'smoothed', color: '--s1', width: 2 });
      return ser;
    };
    const legendFor = () => {
      const items = [];
      if (step === 0) items.push([C.col('--muted'), 'raw, conf < 0.6', 'ring']);
      items.push([C.col('--ink-2'), step === 0 ? 'raw, conf ≥ 0.6' : 'kept after mask']);
      if (step >= 2) items.push([C.col('--s2'), 'interpolated']);
      if (step >= 3) items.push([C.col('--s1'), 'smoothed', 'line']);
      $('legend2').innerHTML = legend(items);
    };
    const tr = new C.LineChart($('tr2'), {
      series: traceSeries(), height: 200, yLabel: 'pixels', xFmt: (v) => v.toFixed(0) + ' s', vFmt: (v) => v.toFixed(1) + ' px',
      onClick: (x) => { P.setPlaying(false); P.setFrame(P.frameAtT(x)); },
      vlines: P.clipEvents.filter((e) => ['GateOpen', 'CorrectDigStart', 'EatingStart'].includes(e.e)).map((e) => ({ x: e.t, label: P.EVENT_META[e.e].short })),
    });
    const sync = (i) => tr.setMarker(clip.t[i]);
    P.state.listeners.add(sync); cleanups.push(() => P.state.listeners.delete(sync)); sync(P.state.i);
    const upd = () => { $('stepText').innerHTML = S2_STEPS[step][1]; player.setStep(step); tr.setSeries(traceSeries()); legendFor(); };
    bindSeg('step', (v) => { step = +v; upd(); });
    bindSeg('coord', (v) => { coord = +v; upd(); });
    $('kpSel').onchange = (e) => { kp = e.target.value; upd(); };
    upd();


    const blx = D.s2.blen.map((_, i) => (D.s2.blen_t0 + i) / 60);
    new C.LineChart($('bl2'), {
      series: [{ x: blx, y: D.s2.blen, name: 'body length', color: '--s1', width: 1.2 },
        { x: [blx[0], blx[blx.length - 1]], y: [M.s2_body_length_px, M.s2_body_length_px], name: 'session median', color: '--ink', width: 1.5, noScale: true }],
      height: 190, yLabel: 'px', robust: true, xFmt: minFmt, xFmtLong: minFmtLong, vFmt: (v) => v.toFixed(1) + ' px', bands: stageBands,
    });
    drawOccupancy($('occ'));
  }

  function drawOccupancy(el) {
    const img = new Image(); img.src = 'frames/arena.jpg';
    const render = () => {
      const w = el.clientWidth || 400, h = w * 3 / 4, cv = el.querySelector('canvas');
      const ctx = C.sizeCanvas(cv, w, h);
      if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, w, h);
      ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, w, h);
      const occ = D.s2.occupancy, ny = occ.length, nx = occ[0].length, mx = Math.log10(D.s2.occ_max * 1e5 + 1);
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const v = occ[y][x]; if (!v) continue;
        const t = Math.log10(v * 1e5 + 1) / mx;
        ctx.fillStyle = C.withAlpha(C.mix('#9ec5f4', '#ffd54f', t), 0.25 + 0.7 * t);
        ctx.fillRect(x / nx * w, y / ny * h, w / nx + 0.5, h / ny + 0.5);
      }
    };
    img.onload = render;
    new ResizeObserver(render).observe(el);
    C.registry.add({ el, render });
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect(), occ = D.s2.occupancy;
      const x = Math.floor((e.clientX - r.left) / r.width * occ[0].length), y = Math.floor((e.clientY - r.top) / r.height * occ.length);
      const v = occ[y] && occ[y][x];
      if (v != null) C.showTip(`<div class="tt-h">10 × 10 px cell</div>${C.ttRow(null, 'time in cell', (v * 100).toFixed(2) + ' %')}`, e.clientX, e.clientY);
    });
    el.addEventListener('pointerleave', C.hideTip);
  }

  // ═══ S3 ══════════════════════════════════════════════════════════
  function pageS3() {
    const ms = D.config.moseq, ex = D.s3.examples;
    const cum8 = D.s3.pca_evr.slice(0, ms.pca_components).reduce((a, b) => a + b, 0);
    main.innerHTML = `
      ${head({ tag: 'S3', eyebrow: 'Behavior lane · stage 3', title: 'Behavioral syllables (keypoint-MoSeq)', lede: 'keypoint-MoSeq splits continuous pose into reused motifs called syllables. A single shared model, <code>model_all</code>, labels every frame of every session, so syllable IDs are comparable across cohorts.',
        chips: [{ v: 'stages/s3_moseq_fitting.py', code: true }, { v: 'python pipeline.py run --stages 3', code: true }, { k: 'model', v: 'outputs/03_moseq/kpms_project_all/model_all/results.h5', code: true }] })}
      ${io([`S2 NPZ for ${D.s3.n_sessions_model} sessions (surgery_15mice, takaya_surgery, fiber_photometry, gi_mice_oft; BFT excluded)`, `7 keypoints: ${D.config.keypoints.use_bodyparts.join(', ')}`],
        [`Egocentric alignment + PCA → ${ms.pca_components} latent dims (${(cum8 * 100).toFixed(0)} % variance)`, `AR-HMM warm-up (${ms.num_ar_iters} iters), then full keypoint-SLDS (${ms.num_full_iters} iters)`, `Sticky HDP: κ = ${Number(ms.kappa).toExponential(0)}, γ = ${Number(ms.gamma).toExponential(0)}, ≤ ${ms.num_states} states, ${ms.nlags} AR lags`],
        ['Per frame: <code>syllable</code>, <code>centroid</code>, <code>heading</code>, <code>latent_state</code>', 'CSV export per session'])}
      <div class="callout" style="margin-top:12px"><b>Why this model.</b> κ = 1e5 was the most consistent setting across the five comparison groups (mean LogReg AUC 0.654). κ = 1e4 overfit the small fiber groups (AUC = 1.0 on leave-one-out); κ = 1e6 collapsed onto five dominant syllables. Bootstrap refits plateau at ARI ≈ 0.63. Seven syllables exceed 1 % global usage.</div>
      ${section('Syllables on the video', 'Smoothed S2 skeleton coloured by the model_all syllable at each frame. The strip under the video is the clip ethogram.', card(null, null, '<div id="p3"></div>'))}
      ${section('Full-session ethogram', 'Every frame of the 73-min session, with AST events below on the same clock. Drag to zoom into a trial; double-click to reset.', card(null, null, `
        <div class="legend">${D.s3.top.map((s, k) => `<span><i style="background:${P.sylColor(k)}"></i>${sName(s)}</span>`).join('')}<span><i style="background:${P.sylColor(7)}"></i>other (${D.s3.n_states_used - D.s3.top.filter((_, k) => D.s3.usage_session[k] > 0).length} rarer syllables)</span></div>
        <div id="eth"></div><div id="eth-r"></div>`))}
      <div class="grid g2 section">
        ${card('Syllable usage', 'Fraction of frames: this session vs all 246 model_all sessions.', `${legend([[C.col('--s1'), 'this session'], [C.col('--other'), 'model_all (global)']])}<div id="use3"></div>`)}
        ${card('PCA of egocentric pose', `Explained variance per component. The first ${ms.pca_components} (bold) form the latent space; together ${(cum8 * 100).toFixed(1)} %.`, '<div id="pca3"></div>')}
      </div>
      <div class="grid g2 section">
        ${card('Bout-to-bout transitions', 'Row-normalized probability of the next syllable, given the current one. Self-transitions are excluded by definition at bout level.', '<div id="tm3"></div>')}
        ${card('Bout duration', 'Per syllable in this session, log axis: median (bar), IQR (box), 10–90 % (whisker).', '<div id="du3"></div>')}
      </div>
      ${section('AST cohort usage', `Usage per session for the ${ex.cohort_fsx.length} AST recordings that are in model_all (12 Control, 4 Social isolation). The ringed dot is this session; ticks mark group means.`, card(null, null, `${legend([[C.col('--s1'), 'Control'], [C.col('--s2'), 'Social isolation'], [C.col('--ink'), 'this session', 'ring']])}<div id="co3"></div>`))}

      ${section('Example outputs: key syllables & F / S / X features', 'The five key syllables from the S4–S5 signature analysis, shown as what S3 output becomes downstream. Each GIF overlays 10 stable poses from one representative bout.', `
        <div class="grid g-wide" style="margin-bottom:14px">
          ${card('Three feature families per syllable', null, `<dl class="kv" style="grid-template-columns:auto 1fr;gap:6px 14px;font-size:13.5px">
            <dt><b>F</b> · frequency</dt><dd style="text-align:left"><code>freq50_s##</code>: fraction of frames assigned to the syllable</dd>
            <dt><b>S</b> · self-transition</dt><dd style="text-align:left"><code>transsummary_self_trans_s##</code>: frame-level P(s → s), i.e. persistence</dd>
            <dt><b>X</b> · exit entropy</dt><dd style="text-align:left"><code>transsummary_row_entropy_s##</code>: entropy (bits) of the outgoing transition row, i.e. how unpredictable the next state is</dd></dl>
            <p class="desc" style="margin-top:10px">Features are residualized on cohort and context, then PCA with a 1000-iteration stratified bootstrap. <b>Signature A</b> = PC1 (mean of the OFT + Food/AST and OFT + OFT-duo folds). <b>Signature B</b> = PC2 of the OFT + Y-maze fold. Bars show each syllable's signed loadings.</p>`)}
          ${card('This session, F / S / X across the AST cohort', 'Pick a feature family. Dots are the 16 AST sessions in model_all.', `<div class="controls">${seg('fsx', [['0', 'F · frequency'], ['1', 'S · self-transition'], ['2', 'X · exit entropy']], '0')}</div>
            ${legend([[C.col('--s1'), 'Control'], [C.col('--s2'), 'Social isolation'], [C.col('--ink'), 'this session', 'ring']])}<div id="fsx3"></div>`)}
        </div>
        <div class="keysyl" id="ks"></div>`)}
      ${section('Parameters', '<code>config.yaml → moseq</code> and <code>joint_all_model</code>', card(null, null, params(Object.assign({}, ms, { contexts: D.config.joint_all_model.contexts, exclude_test_types: D.config.joint_all_model.exclude_test_types }), {
        model_name: 'per-context default (model_all used here)', model_per_context: '', pca_components: 'latent dimensions', num_ar_iters: 'AR-only warm-up', num_full_iters: 'full SLDS iterations',
        num_states: 'HDP truncation', kappa: 'stickiness: higher = longer bouts', gamma: 'HDP concentration', nlags: 'AR order', save_every: 'checkpoint interval', contexts: 'cohorts in model_all', exclude_test_types: '',
      })))}`;

    players.push(new P.ClipPlayer($('p3'), { mode: 's3' }));
    // ethogram + raster, linked
    const g = new C.LinkGroup();
    new C.StripChart($('eth'), {
      segments: sylSegs, starts: sylStarts, xDomain: sessionDomain, group: g, height: 46, title: 'syllable', color: (c) => P.sylColor(c), label: (c) => P.sylName(c),
      xFmt: minFmt, highlight: clipWin, pad: { l: 120, t: 4, b: 18 },
      tip: (s) => `<div class="tt-h">${P.sylName(s[2])}${s[2] === 7 ? ` (${sName(D.s3.rle_syl[C.bisect(sylStarts, s[0])])})` : ''}</div>${C.ttRow(null, 'start', minFmtLong(s[0]))}${C.ttRow(null, 'duration', (s[4] / 10).toFixed(1) + ' s')}`,
    });
    new C.RasterChart($('eth-r'), { rows: rasterRows(), xDomain: sessionDomain, group: g, xFmt: minFmt, xFmtLong: minFmtLong, bands: stageBands, pad: { l: 120 } });
    // usage
    C.barList($('use3'), D.s3.top.map((s, k) => ({ name: sName(s), a: D.s3.usage_session[k] * 100, b: D.s3.usage_global[k] * 100, strong: true,
      tip: `<div class="tt-h">${sName(s)}</div>${C.ttRow(C.col('--s1'), 'this session', (D.s3.usage_session[k] * 100).toFixed(1) + ' %')}${C.ttRow(C.col('--other'), 'model_all', (D.s3.usage_global[k] * 100).toFixed(1) + ' %')}` }))
      .concat([{ name: 'other', a: D.s3.usage_session[7] * 100, b: D.s3.usage_global[7] * 100, tip: '<div class="tt-h">all other syllables</div>' }]), { colorA: '--s1', colorB: '--other', fmt: (v) => v.toFixed(1) + '%', both: true, valW: 96, nameW: 60 });
    let cum = 0;
    C.barList($('pca3'), D.s3.pca_evr.map((v, k) => { cum += v; return { name: 'PC' + (k + 1), a: v * 100, strong: k < ms.pca_components, colorA: k < ms.pca_components ? '--s1' : '--other', tip: `<div class="tt-h">PC${k + 1}</div>${C.ttRow(null, 'explained', (v * 100).toFixed(1) + ' %')}${C.ttRow(null, 'cumulative', (cum * 100).toFixed(1) + ' %')}` }; }), { fmt: (v) => v.toFixed(1) + '%', nameW: 60 });
    // transitions
    const labs = D.s3.top.map(sName).concat(['other']);
    const T = D.s3.trans, tmax = Math.max(...T.flat());
    new C.Heatmap($('tm3'), {
      rows: labs, cols: labs, data: T, square: true, rowLabelW: 52, colLabelH: 18, mono: true, gap: 2,
      color: (v, r, c) => (r === c ? C.col('--surface-2') : C.seq(v / tmax)), cellText: (v, r, c) => (r !== c && v >= 0.1 ? Math.round(v * 100) : ''),
      tip: (r, c, v) => `<div class="tt-h">${labs[r]} → ${labs[c]}</div>${r === c ? 'self (excluded)' : C.ttRow(null, 'P(next)', (v * 100).toFixed(1) + ' %') + C.ttRow(null, 'bouts', D.s3.trans_n[r][c])}`,
    });
    // durations
    boxPlot($('du3'), D.s3.top.filter((s) => D.s3.durations[s]).map((s) => {
      const d = D.s3.durations[s], k = D.s3.top.indexOf(s);
      return { label: sName(s), q: d.q, color: P.sylColor(k), tip: `<div class="tt-h">${sName(s)} · ${d.n} bouts</div>${C.ttRow(null, 'median', d.q[2] + ' s')}${C.ttRow(null, 'IQR', `${d.q[1]}–${d.q[3]} s`)}${C.ttRow(null, 'mean', d.mean + ' s')}` };
    }), { xmin: 0.5, xmax: 120, ticks: [0.5, 1, 2, 5, 10, 20, 50, 100] });
    // cohort usage
    const pts = [];
    D.s3.cohort.forEach((c) => c.usage.forEach((u, k) => pts.push({ cat: k, v: u * 100, group: c.group, current: c.current, tip: `<div class="tt-h">${c.session}</div>${C.ttRow(null, c.group, '')}${C.ttRow(null, labs[k], (u * 100).toFixed(1) + ' %')}` })));
    C.dotStrip($('co3'), { cats: labs.map((l) => ({ label: l, strong: l !== 'other' })), points: pts, groups: [{ key: 'Control', color: '--s1' }, { key: 'Social isolation', color: '--s2' }], xfmt: (v) => v + '%', rowH: 30, labelW: 60 });
    // F/S/X strip
    const FSX = ['F', 'S', 'X'], fsxFmt = [(v) => (v * 100).toFixed(1) + ' %', (v) => v.toFixed(3), (v) => v.toFixed(3) + ' bits'];
    let fsxHandle = null;
    const drawFSX = (f) => {
      f = +f;
      const keys = ex.key.map((k) => k.syl), P2 = [];
      ex.cohort_fsx.forEach((c) => keys.forEach((s, k) => {
        const v = c.v[String(s)];
        if (f > 0 && v[0] === 0) return;
        P2.push({ cat: k, v: f === 0 ? v[0] * 100 : v[f], group: c.group, current: c.current, tip: `<div class="tt-h">${c.session}</div>${C.ttRow(null, `${sName(s)} · ${FSX[f]}`, fsxFmt[f](v[f]))}` });
      }));
      const vals = P2.map((p) => p.v), mn = Math.min(...vals), mx = Math.max(...vals);
      const o = { cats: keys.map((s) => ({ label: sName(s), strong: true })), points: P2, groups: [{ key: 'Control', color: '--s1' }, { key: 'Social isolation', color: '--s2' }], rowH: 30, labelW: 50 };
      if (f === 0) { o.xmax = mx * 1.08; o.xfmt = (v) => v + '%'; } else { const pd = (mx - mn) * 0.08 || 0.01; o.xmin = Math.max(0, mn - pd); o.xmax = mx + pd; o.xfmt = (v) => v.toFixed(f === 1 ? 3 : 2); }
      $('fsx3').innerHTML = '';
      fsxHandle = C.dotStrip($('fsx3'), o);
    };
    drawFSX(0); bindSeg('fsx', drawFSX); void fsxHandle;
    // key syllable cards
    const lmax = 0.16;
    const cohortMean = (s, f, grp) => {
      const v = ex.cohort_fsx.filter((c) => c.group === grp).map((c) => c.v[String(s)]).filter((x) => f === 0 || x[0] > 0).map((x) => x[f]);
      return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
    };
    $('ks').innerHTML = ex.key.map((k) => {
      const bars = ['F', 'S', 'X'].map((L, j) => {
        const v = k.loadings[j], w = Math.abs(v) / lmax * 50;
        return `<span class="lbl">${L}</span><span class="axis"><b style="${v >= 0 ? `left:50%;width:${w}%;background:var(--s8)` : `right:50%;width:${w}%;background:var(--s1)`}"></b></span><span class="val">${v >= 0 ? '+' : ''}${v.toFixed(3)}</span>`;
      }).join('');
      const tv = ex.this_fsx[String(k.syl)];
      const row = (j, f) => `<tr><td><b>${'FSX'[j]}</b></td><td class="num">${tv[0] === 0 && j > 0 ? '–' : f(tv[j])}</td><td class="num">${(() => { const m = cohortMean(k.syl, j, 'Control'); return m == null ? '–' : f(m); })()}</td><td class="num">${(() => { const m = cohortMean(k.syl, j, 'Social isolation'); return m == null ? '–' : f(m); })()}</td></tr>`;
      return `<article class="ks-card">
        <div class="ks-plate"><img src="${k.gif}" alt="${sName(k.syl)} pose trajectory animation"></div>
        <div class="ks-body">
          <div class="ks-title"><h3>${sName(k.syl)}</h3><span class="sig">Signature ${k.signature} (${k.signature === 'A' ? 'PC1' : 'PC2'})</span></div>
          <div class="ks-interp">${k.text}</div>
          <div class="ks-src">${esc(k.clip.replace('.mp4', ''))} · 10 stable poses</div>
          <div><span class="enrich"><i style="background:${k.enriched === 'normal' ? 'var(--s1)' : 'var(--s8)'}"></i>Enriched in ${k.enriched}</span></div>
          <div class="fsx">${bars}</div>
          <div class="ks-session"><table><thead><tr><th></th><th class="num">this session</th><th class="num">Control</th><th class="num">SI</th></tr></thead><tbody>
            ${row(0, (v) => (v * 100).toFixed(2) + '%')}${row(1, (v) => v.toFixed(3))}${row(2, (v) => v.toFixed(3))}</tbody></table>
            <div style="margin-top:4px">${k.session_bouts} bouts in this session${tv[0] === 0 ? ' (absent)' : ''}.</div></div>
        </div></article>`;
    }).join('');
  }

  // ═══ S6 ══════════════════════════════════════════════════════════
  const PERI_EVENTS = ['GateOpen', 'ApproachCorrect', 'ApproachIncorrect', 'CorrectDigStart', 'IncorrectDigStart', 'EatingStart'];
  function pageS6() {
    const st = D.s6.dff_steps, m6 = D.s6.meta, fj = D.config.fiber_joint;
    main.innerHTML = `
      ${head({ tag: 'S6', eyebrow: 'Photometry lane · stage 6', title: 'Fiber photometry × kinematics', lede: 'Pairs each TDT photometry block with the custom-DLC pose for the same video, computes a bleaching-robust dF/F, aligns it to video frames with camera epocs, and builds peri-event and per-trial AST measures.',
        chips: [{ v: 'stages/s6_fiber_joint.py', code: true }, { v: 'python stages/s6_fiber_joint.py --steps index traces events peri trials summary', code: true }, { k: 'writes', v: 'outputs/06_fiber_joint/', code: true }] })}
      ${io(['Custom DLC H5 (7 kp, 20 fps): 5 used', `TDT block: 465 / 405 nm, channels A and C, ${M.tdt_fs} Hz, Cam1 epocs`, 'Synapse notes: <code>.tnt</code> names + <code>*SavedNotes.nxt</code> times'],
        ['<b>index</b> pair H5 ↔ TDT block · <b>traces</b> kinematics + dF/F per frame', '<b>events</b> parse notes · <b>peri</b> event-locked averages', '<b>trials</b> one row per GateOpen · <b>figures</b>'],
        ['<code>sessions/*.npz</code> (speed, angvel, bend, dffA/C, isoA/C, cam_t)', '<code>events.csv</code>, <code>event_traces.npz</code>, <code>event_metrics.csv</code>', '<code>trial_table.csv</code>, <code>figures/</code>'])}
      ${tiles([
        { label: 'Motion-fit R², detrended', value: st.r2.toFixed(2), note: `raw-on-raw would give ${st.r2_naive.toFixed(2)}` },
        { label: 'LED settle discarded', value: st.settle_s.toFixed(1), unit: 's', note: 'turn-on transient + 30 s' },
        { label: '5-keypoint tracking yield', value: m6.yield5, unit: '%', note: 'all 5 points ≥ 0.6' },
        { label: 'Body length', value: m6.BL_px.toFixed(1), unit: 'px', note: `neck → tail_base ≙ ${fj.cm_per_body_length} cm` },
      ])}
      ${section('Step A · Bleaching-robust dF/F', `Channel A of this session. Each channel is first detrended against its own rolling ${fj.baseline_pct}th-percentile baseline (${fj.baseline_window_s}-s window); only then is 405 regressed onto 465. Regressing raw traces would let the shared bleaching decay dominate the fit (R² ${st.r2_naive.toFixed(2)}) and subtract real signal with it.`, `
        <div class="grid g2">
          ${card('465 nm (dopamine sensor), low-passed, with F₀', null, '<div id="a465"></div>')}
          ${card('405 nm (isosbestic control), low-passed, with F₀', null, '<div id="a405"></div>')}
        </div>
        <div class="grid g2" style="margin-top:16px">
          ${card('Detrended 465 and fitted motion component (60-s window)', `Fitted motion = ${st.slope.toFixed(2)} × detrended 405 + offset.`, `${legend([[C.col('--s1'), '465 − F₀', 'line'], [C.col('--s2'), 'fitted motion (from 405)', 'line']])}<div id="awin"></div>`)}
          ${card('Resulting dF/F (60-s window)', 'dF/F = (detrended 465 − motion) / F₀(465) × 100. Window includes the clip trial.', '<div id="adff"></div>')}
        </div>`)}
      ${section('Step B · Session timeline: dopamine × movement × syllables × events', 'All panels share the TDT clock. S3 syllables (10 fps) are mapped through the same camera epocs. Shaded spans are the IDS / EDS stages; the outlined box is the video clip. Drag to zoom into any trial.', card(null, null, `
        <div id="tl-dff"></div><div id="tl-spd"></div><div id="tl-syl"></div><div id="tl-ev"></div>
        <p class="chart-hint">Speed is the S6 body_center speed × ${fj.cm_per_body_length} cm/BL. Syllable colours follow the S3 page.</p>`))}
      ${section('Step C · Peri-event dopamine', `Event-locked dF/F (channel A), baselined to [${fj.peri_baseline.join(', ')}] s before each event. Events keep their IDS / EDS stage labels.`, card(null, null, `
        <div class="controls">${seg('pe', PERI_EVENTS.map((e) => [e, P.EVENT_META[e].label]), 'GateOpen')}</div>
        <div class="controls">${seg('pesplit', [['all', 'All trials'], ['stage', 'IDS vs EDS']], 'all')}${seg('pesig', [['dff', 'dF/F'], ['iso', '405 control']], 'dff')}<span class="ctl-label" id="pe-n"></span></div>
        <div class="grid g2"><div><div id="pe-leg"></div><div id="pe-line"></div></div><div><div class="legend"><span>trial dF/F (%)</span>${[-1, -0.5, 0, 0.5, 1].map((v) => `<span><i style="background:${C.div(v)}"></i>${v === -1 ? '−' : v === 1 ? '+' : v === 0 ? '0' : ''}</span>`).join('')}</div><div id="pe-heat"></div></div></div>`))}
      ${section('Step D · Trials', 'One row per GateOpen. Outcome = first dig after the gate opens. The ringed cell is where the six-in-a-row criterion is first met. Click a trial to zoom the timeline above.', card(null, null, `
        <div class="trial-strip" id="tstrip"></div>
        <div class="grid g2" style="margin-top:16px">
          <div><h3 style="font-size:14px">Choice latency (s)</h3><div id="tlat"></div></div>
          <div><h3 style="font-size:14px">First-dig dF/F response (0–1 s minus baseline, %)</h3><div id="tdig"></div></div>
        </div>
        <div class="table-wrap" style="margin-top:14px"><table id="ttable"></table></div>`))}
      ${section('Cohort context · historical Stage 6 outputs', 'Session-mean peri-event dF/F (channel A, all stages) for the AST sessions in <code>outputs/06_fiber_joint</code>, mean ± SEM across sessions.', card(null, null, `
        <div class="controls">${seg('coh', ['GateOpen', 'ApproachCorrect', 'CorrectDigStart', 'EatingStart'].map((e) => [e, P.EVENT_META[e].label]), 'GateOpen')}<span class="ctl-label" id="coh-n"></span></div>
        ${legend([[C.col('--s1'), 'Control', 'line'], [C.col('--s2'), 'Social isolation', 'line']])}<div id="coh"></div>`))}
      ${section('Example output · per-video AST time course', `Generated by <code>${D.s6.example_fig.script}</code> for this session: speed, turning, cumulative distance, distance to the target zone, and event rasters, aligned to the first GateOpen.`, card(null, null, `
        <img class="figure-img" src="${D.s6.example_fig.src}" alt="Per-video AST time course for ${M.mouse}" id="figS6">
        <p class="chart-hint">Source: <code>${D.s6.example_fig.origin}</code> · click to enlarge.</p>`))}
      ${section('Parameters', '<code>config.yaml → fiber_joint</code>', card(null, null, params(fj, {
        keypoints: 'custom-DLC points used for kinematics', conf_threshold: 'mask below', max_interp_gap: 'frames', smooth_window: '≈ 0.45 s at 20 fps', fps: 'assumed video rate',
        cm_per_body_length: 'unit conversion', lowpass_hz: 'Butterworth low-pass', baseline_window_s: 'rolling F₀ window', baseline_pct: 'F₀ percentile', settle_s: 'discarded after LED onset',
        peri_pre_s: 'window before event', peri_post_s: 'window after event', peri_baseline: 'baseline (s)', min_events: 'skip event types below this count', gate_window: 'trial DA window', approach_window: '', eat_window: '',
      }, Object.keys(fj).filter((k) => !['data_dir', 'dlc_project', 'output_dir'].includes(k)))))}`;
    buildS6();
  }

  function buildS6() {
    const st = D.s6.dff_steps;
    // Step A
    const F = st.full, ft = F.t.map((t) => t / 60), gA = new C.LinkGroup();
    const fullDom = [0, ft[ft.length - 1]];
    const mkRaw = (el, y, f0, name, color) => new C.LineChart($(el), {
      series: [{ x: ft, y, name, color, width: 1.2 }, { x: ft, y: f0, name: 'F₀ (rolling 10th pct)', color: '--ink', width: 1.6 }],
      height: 180, xFmt: minFmt, xFmtLong: minFmtLong, group: gA, xDomain: fullDom, vFmt: (v) => v.toFixed(1), yLabel: 'raw (a.u.)', robust: true,
    });
    mkRaw('a465', F.a, F.F0a, '465 nm', '--s1');
    mkRaw('a405', F.r, F.F0r, '405 nm', '--s7');
    const W = st.win;
    new C.LineChart($('awin'), { series: [{ x: W.t, y: W.da, name: '465 − F₀', color: '--s1', width: 1.4 }, { x: W.t, y: W.motion, name: 'fitted motion', color: '--s2', width: 1.6 }], height: 180, xFmt: (v) => v.toFixed(0) + ' s', vFmt: (v) => v.toFixed(2), yLabel: 'a.u.' });
    new C.LineChart($('adff'), { series: [{ x: W.t, y: W.dff, name: 'dF/F', color: '--s1', width: 1.6 }], height: 180, xFmt: (v) => v.toFixed(0) + ' s', vFmt: (v) => v.toFixed(2) + ' %', yLabel: 'dF/F (%)', zeroLine: true,
      vlines: D.s6.events.filter((e) => e.t >= W.t[0] && e.t <= W.t[W.t.length - 1] && ['GateOpen', 'CorrectDigStart', 'EatingStart', 'ApproachCorrect'].includes(e.e)).map((e) => ({ x: e.t, label: P.EVENT_META[e.e].short })) });

    // Step B timeline
    const g = new C.LinkGroup();
    const L = 120;
    const dffC = new C.LineChart($('tl-dff'), { series: [{ x: tMin, y: OV.dff, name: 'dF/F ch A', color: '--s1', width: 1.1 }], xDomain: sessionDomain, group: g, height: 150, yLabel: 'dF/F (%)', bands: stageBands, highlight: clipWin, xFmt: minFmt, xFmtLong: minFmtLong, vFmt: (v) => v.toFixed(2) + ' %', pad: { l: L }, noXAxis: true, robust: true, zeroLine: true });
    new C.LineChart($('tl-spd'), { series: [{ x: tMin, y: OV.speed, name: 'speed', color: '--s3', width: 1.1 }], xDomain: sessionDomain, group: g, height: 110, yLabel: 'cm/s', bands: stageBands, bandLabels: false, highlight: clipWin, xFmt: minFmt, xFmtLong: minFmtLong, vFmt: (v) => v.toFixed(1) + ' cm/s', pad: { l: L }, noXAxis: true, robust: true, yZero: true });
    new C.StripChart($('tl-syl'), { segments: sylSegs, starts: sylStarts, xDomain: sessionDomain, group: g, height: 30, title: 'syllable', color: (c) => P.sylColor(c), label: (c) => P.sylName(c), xFmt: minFmt, noXAxis: true, pad: { l: L, t: 4, b: 4 },
      tip: (s) => `<div class="tt-h">${P.sylName(s[2])}</div>${C.ttRow(null, 'start', minFmtLong(s[0]))}${C.ttRow(null, 'duration', (s[4] / 10).toFixed(1) + ' s')}` });
    new C.RasterChart($('tl-ev'), { rows: rasterRows(), xDomain: sessionDomain, group: g, xFmt: minFmt, xFmtLong: minFmtLong, bands: stageBands, pad: { l: L } });
    void dffC;

    // Step C peri
    const pt = D.s6.peri.t;
    let pe = 'GateOpen', split = 'all', sig = 'dff';
    const peLine = new C.LineChart($('pe-line'), { series: [{ x: pt, y: pt.map(() => 0), name: '', color: '--s1' }], height: 240, xFmt: (v) => v + ' s', vFmt: (v) => v.toFixed(2) + ' %', yLabel: 'Δ dF/F (%)', zeroLine: true, vlines: [{ x: 0, color: '--ink-2' }], brush: false });
    const peHeatEl = $('pe-heat');
    let peHeat = null;
    const drawPeri = () => {
      const ev = D.s6.peri.events[pe], rows = ev[sig];
      const series = [];
      const groups = split === 'all' ? [['all', '--s1', rows.map((_, i) => i)]] : [['IDS', '--s1', ev.stage.map((s, i) => (s === 'IDS' ? i : -1)).filter((i) => i >= 0)], ['EDS', '--s2', ev.stage.map((s, i) => (s === 'EDS' ? i : -1)).filter((i) => i >= 0)]];
      const legItems = [];
      groups.forEach(([name, color, idx]) => {
        if (!idx.length) return;
        const ms = meanSem(idx.map((i) => rows[i]));
        series.push({ x: pt, y: ms.mean, lo: ms.lo, hi: ms.hi, name: `${name === 'all' ? 'mean' : name} (n = ${idx.length})`, color, width: 2 });
        legItems.push([C.col(color), `${name === 'all' ? 'Mean ± SEM' : name + ' mean ± SEM'} · n = ${idx.length}`, 'line']);
      });
      $('pe-leg').innerHTML = legend(legItems);
      $('pe-n').textContent = `${rows.length} events in this session`;
      peLine.setSeries(series.length ? series : [{ x: pt, y: pt.map(() => null), name: 'no events', color: '--s1' }]);
      // heatmap rows sorted by stage then time
      const order = rows.map((_, i) => i).sort((a, b) => (split === 'stage' ? (ev.stage[a] > ev.stage[b] ? -1 : ev.stage[a] < ev.stage[b] ? 1 : 0) : 0) || ev.t[a] - ev.t[b]);
      const data = order.map((i) => rows[i].map((v) => (v == null ? NaN : v)));
      const all = data.flat().filter(isFinite).map(Math.abs).sort((a, b) => a - b);
      const lim = all.length ? all[Math.floor(all.length * 0.97)] || 1 : 1;
      const opts = { rows: order.map((i) => `${ev.stage[i]} ${ev.t[i].toFixed(0)}s`), data: data.length ? data : [[NaN]], color: (v) => C.div(v / lim), cellH: Math.max(6, Math.min(14, 260 / Math.max(1, data.length))), rowLabelW: 72, gap: 0,
        xAxis: { d0: pt[0], d1: pt[pt.length - 1] + 0.05, fmt: (v) => v + ' s' }, vline: 0, hoverRow: true,
        tip: (r, c, v) => `<div class="tt-h">${P.EVENT_META[pe].label} at ${ev.t[order[r]].toFixed(1)} s (${ev.stage[order[r]]})</div>${C.ttRow(null, `t = ${pt[c].toFixed(2)} s`, isFinite(v) ? v.toFixed(2) + ' %' : '–')}` };
      if (!peHeat) peHeat = new C.Heatmap(peHeatEl, opts); else peHeat.update(opts.data, opts);
    };
    bindSeg('pe', (v) => { pe = v; drawPeri(); });
    bindSeg('pesplit', (v) => { split = v; drawPeri(); });
    bindSeg('pesig', (v) => { sig = v; drawPeri(); });
    drawPeri();

    // Step D trials
    const trials = D.s6.trials;
    const streak = {}; const crit = new Set();
    trials.forEach((t) => { streak[t.stage] = t.outcome === 'correct' ? (streak[t.stage] || 0) + 1 : 0; if (streak[t.stage] === 6 && !crit.has(t.stage)) { crit.add(t.stage); t._crit = true; } });
    const stages = [...new Set(trials.map((t) => t.stage))];
    $('tstrip').innerHTML = stages.map((s) => {
      const ts = trials.filter((t) => t.stage === s), nc = ts.filter((t) => t.outcome === 'correct').length;
      return `<div class="trial-stage"><div class="st-h"><b>${s}</b> · ${ts.length} trials · ${nc} correct</div><div class="trial-cells">${ts.map((t) => `<div class="tcell ${t.outcome} ${t._crit ? 'crit' : ''}" data-tr="${t.trial}"><span class="o">${t.outcome === 'correct' ? '✓' : t.outcome === 'incorrect' ? '✗' : '–'}</span>${t.stage_trial}</div>`).join('')}</div></div>`;
    }).join('') + `<div class="legend" style="align-self:end"><span><b style="color:var(--good-ink)">✓</b> correct first dig</span><span><b style="color:var(--crit-ink)">✗</b> incorrect first dig</span><span><i class="ring" style="border-color:var(--accent)"></i>6-correct criterion reached</span></div>`;
    main.querySelectorAll('.tcell').forEach((c) => {
      const t = trials.find((x) => x.trial === +c.dataset.tr);
      tipHover(c, `<div class="tt-h">Trial ${t.trial} · ${t.stage} #${t.stage_trial}</div>${C.ttRow(null, 'gate open', t.gate_t.toFixed(1) + ' s')}${C.ttRow(null, 'outcome', t.outcome)}${C.ttRow(null, 'latency', t.latency == null ? '–' : t.latency.toFixed(1) + ' s')}${C.ttRow(null, 'approaches ✓ / ✗', `${t.app_c} / ${t.app_i}`)}`);
      c.addEventListener('click', () => {
        main.querySelectorAll('.tcell').forEach((x) => x.classList.toggle('sel', x === c));
        const nxt = trials.find((x) => x.trial === t.trial + 1);
        const end = Math.min((t.dig_t || t.gate_t) + 40, nxt ? nxt.gate_t : Infinity);
        g.members[0].setX((t.gate_t - 8) / 60, end / 60);
        document.getElementById('tl-dff').scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });
    const tx = trials.map((t) => t.trial);
    const outColor = (t) => (t.outcome === 'correct' ? '--s3' : '--s8');
    const tBars = (el, key, fmt, unit) => {
      C.barList($(el), trials.map((t) => ({ name: `${t.trial} · ${t.stage} ${t.outcome === 'correct' ? '✓' : '✗'}`, a: Math.max(0, t[key] || 0), colorA: outColor(t), tip: `<div class="tt-h">Trial ${t.trial} (${t.stage}, ${t.outcome})</div>${C.ttRow(null, unit, t[key] == null ? '–' : fmt(t[key]))}` })), { fmt: (v) => fmt(v), nameW: 92 });
    };
    void tx;
    tBars('tlat', 'latency', (v) => v.toFixed(1), 'latency (s)');
    // dig response can be negative → simple diverging bars
    const digEl = $('tdig'); const dmax = Math.max(...trials.map((t) => Math.abs(t.dig_dff || 0))) || 1;
    digEl.innerHTML = `<div class="bars">${trials.map((t) => { const v = t.dig_dff; const w = v == null ? 0 : Math.abs(v) / dmax * 50; return `<div class="bar-row" style="grid-template-columns:92px 1fr 50px" data-tr="${t.trial}"><div class="nm">${t.trial} · ${t.stage} ${t.outcome === 'correct' ? '✓' : '✗'}</div><div class="track" style="background:linear-gradient(var(--axis),var(--axis)) 50% 0/1px 100% no-repeat"><div class="fill single" style="${v >= 0 ? 'left:50%' : `left:${50 - w}%`};width:${w}%;background:var(${outColor(t).slice(0)});border-radius:${v >= 0 ? '0 4px 4px 0' : '4px 0 0 4px'}"></div></div><div class="v">${v == null ? '–' : (v >= 0 ? '+' : '') + v.toFixed(2)}</div></div>`; }).join('')}</div>`;
    digEl.querySelectorAll('.bar-row').forEach((r) => { const t = trials.find((x) => x.trial === +r.dataset.tr); tipHover(r, `<div class="tt-h">Trial ${t.trial} · first dig ${t.first_dig || '–'}</div>${C.ttRow(null, 'Δ dF/F 0–1 s', t.dig_dff == null ? '–' : t.dig_dff.toFixed(2) + ' %')}`); });
    $('ttable').innerHTML = `<thead><tr><th class="num">#</th><th>stage</th><th>outcome</th><th class="num">gate (s)</th><th class="num">latency (s)</th><th class="num">approach ✓/✗</th><th class="num">pre-choice speed (cm/s)</th><th class="num">tracked (%)</th><th class="num">gate dF/F early</th><th class="num">dig dF/F</th></tr></thead><tbody>${trials.map((t) => `<tr><td class="num">${t.trial}</td><td>${t.stage} ${t.stage_trial}</td><td>${t.outcome === 'correct' ? '<b style="color:var(--good-ink)">✓</b>' : '<b style="color:var(--crit-ink)">✗</b>'} ${t.outcome}</td><td class="num">${t.gate_t.toFixed(1)}</td><td class="num">${t.latency == null ? '–' : t.latency.toFixed(1)}</td><td class="num">${t.app_c} / ${t.app_i}</td><td class="num">${t.speed_med == null ? '–' : t.speed_med.toFixed(1)}</td><td class="num">${t.track_pct == null ? '–' : t.track_pct.toFixed(0)}</td><td class="num">${t.gate_dff_early == null ? '–' : t.gate_dff_early.toFixed(2)}</td><td class="num">${t.dig_dff == null ? '–' : t.dig_dff.toFixed(2)}</td></tr>`).join('')}</tbody>`;

    // Step E cohort
    const cp = D.s6.cohort_peri, ct = pt;
    const coh = new C.LineChart($('coh'), { series: [{ x: ct, y: ct.map(() => 0), name: '', color: '--s1' }], height: 240, xFmt: (v) => v + ' s', vFmt: (v) => v.toFixed(2) + ' %', yLabel: 'Δ dF/F (%)', zeroLine: true, vlines: [{ x: 0, color: '--ink-2' }], brush: false });
    const drawCoh = (e) => {
      const d = cp[e], ser = [];
      [['Control', '--s1'], ['Social isolation', '--s2']].forEach(([g2, c]) => {
        if (!d[g2]) return;
        const m = d[g2].mean, s = d[g2].sem;
        ser.push({ x: ct, y: m, lo: m.map((v, i) => (v == null ? null : v - s[i])), hi: m.map((v, i) => (v == null ? null : v + s[i])), name: `${g2} (${d[g2].n} sessions)`, color: c, width: 2 });
      });
      coh.setSeries(ser);
      $('coh-n').textContent = Object.entries(d).map(([g2, v]) => `${g2}: ${v.n} sessions / ${v.n_mice_prefix} ID prefixes`).join(' · ');
    };
    bindSeg('coh', drawCoh); drawCoh('GateOpen');

    $('figS6').addEventListener('click', () => openLightbox(D.s6.example_fig.src));
  }

  // ═══ Provenance ══════════════════════════════════════════════════
  function pageProvenance() {
    const bf = M.built_from;
    const rows = [
      ['S1 SuperAnimal pose', bf.s1_superanimal, 'copied subset (clip) + summaries'],
      ['S1b custom AST pose', bf.s1_custom, 'copied subset (clip) + summaries'],
      ['S2 preprocessed', bf.s2, 'recomputed from S1 with stage functions; matches stored file'],
      ['S3 model_all results', bf.s3, 'syllables, usage, transitions, F/S/X'],
      ['S6 session NPZ', bf.s6, 'dF/F, speed, camera epocs'],
      ['Raw TDT block', bf.tdt, 'dF/F method steps recomputed with stage functions'],
      ['Raw video', bf.video, `${clip.n} frames extracted at 10 fps (${clip.w}×${clip.h} JPEG)`],
      ['AST events', bf.events, 'IDS / EDS stage labels'],
      ['AST trials', bf.trials, 'one row per GateOpen'],
      ['Manual-label QC', 'outputs/methodlab/ast_tracking_qc_20250621/keypoint_error_summary.csv', '50 annotated frames'],
      ['Key syllable GIFs', 'outputs/05_stage5/key_syllable_selectedclip_trajectories/', 'copied'],
      ['F/S/X loadings', 'outputs/05_stage5/pca_analysis_R/signature_{A,B}_*_loadings.csv', 'signed PC loadings'],
      ['Per-video time course', D.s6.example_fig.origin, 'copied figure'],
      ['Cohort peri-event', 'outputs/06_fiber_joint/event_traces.npz + event_metrics.csv', 'AST sessions, channel A, all stages'],
    ];
    main.innerHTML = `
      ${head({ eyebrow: 'Reference', title: 'Provenance', lede: 'Where every number on these pages comes from, what was recomputed versus copied, and how to rebuild the bundle.' })}
      ${section('Sources', null, card(null, null, `<div class="table-wrap"><table><thead><tr><th>content</th><th>source (relative to project root)</th><th>use</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${r[0]}</td><td class="mono" style="font-size:12px;word-break:break-all">${esc(r[1])}</td><td>${r[2]}</td></tr>`).join('')}</tbody></table></div>`))}
      ${section('Checks run while building', null, `<div class="grid">
        ${card('Verified', null, `<ul style="margin:0;padding-left:18px;display:grid;gap:6px;font-size:13.8px;color:var(--ink-2)">${[
          `S2 recomputed from S1 with <code>stages/s2_preprocessing.py</code> functions matches the stored NPZ (max |Δ| ${M.s2_verify_max_diff.toExponential(1)}).`,
          `S1 rows (${nf(M.sa_rows)}) = S2 rows = S3 syllable rows.`,
                    `Clip frames are read at native indices ${clip.native[0]}–${clip.native[clip.n - 1]} (even only) and timed with Stage 6 camera epocs.`,
        ].map((x) => `<li>${x}</li>`).join('')}</ul>`)}

      </div>`)}
      ${section('Rebuild', null, card(null, null, `<p class="desc">From the project root, with the <code>mice</code> environment (needs numpy, pandas, h5py, opencv, tdt, joblib/sklearn):</p>
        <pre style="background:var(--code-bg);padding:12px;border-radius:8px;overflow-x:auto;font-size:12.5px"><code style="background:none">/home/yangliu/miniconda3/envs/mice/bin/python ast_pipeline_demo/build/build_demo_data.py</code></pre>
        <p class="desc">This regenerates <code>data/demo_data.js</code>, <code>frames/</code> and <code>assets/syllables</code> + <code>assets/figures</code>. Change <code>BLOCK</code> / <code>CLIP_T</code> at the top of the script to feature another session or trial. The demo itself is static: open <code>index.html</code> in any modern browser, no server or internet needed.</p>`))}
      ${section('Session', null, card(null, null, params({ block: M.block, mouse: M.mouse, group: M.group, date: M.date, camera: M.cam, context: M.context, video: M.video, tdt_fs: M.tdt_fs, clip_tdt_s: M.clip_t })))}`;
  }

  // ── lightbox ─────────────────────────────────────────────────────
  const lb = document.getElementById('lightbox');
  function openLightbox(src) { lb.querySelector('img').src = src; lb.hidden = false; }
  lb.addEventListener('click', () => { lb.hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lb.hidden = true; });

  // ── routing ──────────────────────────────────────────────────────
  const PAGES = { overview: pageOverview, s1: pageS1, s2: pageS2, s3: pageS3, s6: pageS6, provenance: pageProvenance };
  let cleanups = [];
  function route() {
    const r = (location.hash || '#overview').slice(1);
    const page = PAGES[r] ? r : 'overview';
    P.setPlaying(false);
    players.forEach((p) => p.destroy()); players = [];
    cleanups.forEach((f) => f()); cleanups = [];
    C.hideTip();
    document.querySelectorAll('.sidebar a').forEach((a) => a.classList.toggle('active', a.dataset.route === page));
    document.getElementById('sidebar').classList.remove('open');
    PAGES[page]();
    window.scrollTo(0, 0);
    main.focus({ preventScroll: true });
  }
  window.addEventListener('hashchange', route);

  // ── chrome ───────────────────────────────────────────────────────
  document.getElementById('sessionChip').innerHTML = `<b>${M.mouse}</b> · ${M.group} · ${M.date} · ${M.cam} · AST`;
  document.getElementById('themeBtn').addEventListener('click', () => {
    const next = C.isDark() ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('ast-demo-theme', next); } catch (e) { /* private mode */ }
    C.rerenderAll();
  });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => C.rerenderAll());
  document.getElementById('navToggle').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
  document.addEventListener('keydown', (e) => {
    if (!players.length || /input|select|textarea/i.test(e.target.tagName)) return;
    if (e.key === ' ') { e.preventDefault(); P.setPlaying(!P.state.playing); }
    else if (e.key === 'ArrowRight') { P.setPlaying(false); P.setFrame(P.state.i + 1); }
    else if (e.key === 'ArrowLeft') { P.setPlaying(false); P.setFrame(P.state.i - 1); }
  });
  route();
})();
