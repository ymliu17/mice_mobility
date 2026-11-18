/* Shared clip player: real video frames + per-stage overlays (S1 / S2 / S3). */
(function () {
  'use strict';
  const D = window.DEMO, C = window.Charts, clip = D.clip;

  // ── shared vocab ─────────────────────────────────────────────────
  const SYL_VARS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--other'];
  const sylColor = (code) => C.col(SYL_VARS[code] || '--other');
  const sylName = (code) => (code < D.s3.n_top ? 's' + String(D.s3.top[code]).padStart(2, '0') : 'other');
  const EVENT_META = {
    GateOpen: { label: 'Gate open', short: 'Gate', color: '--ink-2' },
    ApproachCorrect: { label: 'Approach correct', short: 'App ✓', color: '--s3' },
    ApproachIncorrect: { label: 'Approach incorrect', short: 'App ✗', color: '--s4' },
    CorrectDigStart: { label: 'Dig correct', short: 'Dig ✓', color: '--s1' },
    IncorrectDigStart: { label: 'Dig incorrect', short: 'Dig ✗', color: '--s8' },
    EatingStart: { label: 'Eating start', short: 'Eat', color: '--s7' },
    CR: { label: 'CR', short: 'CR', color: '--muted' },
    Miss: { label: 'Miss', short: 'Miss', color: '--muted' },
  };

  const SA = D.s1.sa_bodyparts, CU = D.s1.cu_bodyparts;
  const USE = D.config.keypoints.use_bodyparts;
  const USE_IDX = USE.map((b) => SA.indexOf(b));
  const SA_EDGES = [
    ['nose', 'left_eye'], ['nose', 'right_eye'], ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
    ['left_ear', 'left_ear_tip'], ['right_ear', 'right_ear_tip'], ['nose', 'head_midpoint'], ['head_midpoint', 'neck'],
    ['neck', 'mid_back'], ['mid_back', 'mouse_center'], ['mouse_center', 'mid_backend'], ['mid_backend', 'mid_backend2'],
    ['mid_backend2', 'mid_backend3'], ['mid_backend3', 'tail_base'], ['tail_base', 'tail1'], ['tail1', 'tail2'],
    ['tail2', 'tail3'], ['tail3', 'tail4'], ['tail4', 'tail5'], ['tail5', 'tail_end'],
    ['neck', 'left_shoulder'], ['left_shoulder', 'left_midside'], ['left_midside', 'left_hip'], ['left_hip', 'tail_base'],
    ['neck', 'right_shoulder'], ['right_shoulder', 'right_midside'], ['right_midside', 'right_hip'], ['right_hip', 'tail_base'],
  ].map(([a, b]) => [SA.indexOf(a), SA.indexOf(b)]);
  const USE_EDGES = [['nose', 'left_ear'], ['nose', 'right_ear'], ['nose', 'neck'], ['left_ear', 'neck'], ['right_ear', 'neck'],
    ['neck', 'mid_back'], ['mid_back', 'mouse_center'], ['mouse_center', 'tail_base']].map(([a, b]) => [SA.indexOf(a), SA.indexOf(b)]);
  const CU_EDGES = [['nose', 'left_ear'], ['nose', 'right_ear'], ['left_ear', 'neck'], ['right_ear', 'neck'],
    ['neck', 'body_center'], ['body_center', 'tail_base'], ['tail_base', 'tail_tip']].map(([a, b]) => [CU.indexOf(a), CU.indexOf(b)]);
  // overlay colours sit on a dark video frame, so they are fixed (not themed)
  const REGION = { head: '#ff8a5c', spine: '#5cc8ff', tail: '#ffd54f', flank: '#d7a6ff' };
  const regionOf = (b) => (/nose|ear|eye|head/.test(b) ? 'head' : /tail/.test(b) ? 'tail' : /shoulder|midside|hip/.test(b) ? 'flank' : 'spine');

  // ── global clip state + frame cache ──────────────────────────────
  const state = { i: 0, playing: false, speed: 1, listeners: new Set() };
  const frames = clip.files.map((f) => { const im = new Image(); im.decoding = 'async'; im.src = 'frames/' + f; return im; });
  function setFrame(i) {
    state.i = Math.max(0, Math.min(clip.n - 1, Math.round(i)));
    state.listeners.forEach((fn) => fn(state.i));
  }
  let raf = null, last = 0, acc = 0;
  function tick(ts) {
    if (!state.playing) { raf = null; return; }
    const dt = last ? (ts - last) / 1000 : 0; last = ts; acc += dt * clip.fps * state.speed;
    if (acc >= 1) { const k = Math.floor(acc); acc -= k; setFrame(state.i + k >= clip.n ? 0 : state.i + k); }
    raf = requestAnimationFrame(tick);
  }
  function setPlaying(p) {
    state.playing = p; last = 0; acc = 0;
    if (p && !raf) raf = requestAnimationFrame(tick);
    state.listeners.forEach((fn) => fn(state.i));
  }

  // open on the first well-tracked frame after the gate opens (frames before it show handling)
  (function initialFrame() {
    const gate = D.s6.events.find((e) => e.e === 'GateOpen' && e.t >= clip.t[0]);
    let i0 = gate ? C.bisect(clip.t, gate.t) : 0;
    for (let i = i0; i < clip.n; i++) {
      if (USE_IDX.filter((k) => clip.sa_cf[i][k] >= 0.6).length >= 6) { state.i = i; return; }
    }
  })();
  // follow-cam centre per frame: custom body_center, else SuperAnimal mouse_center; gap-filled, smoothed
  const centers = (function () {
    const bc = CU.indexOf('body_center'), mc = SA.indexOf('mouse_center');
    const raw = clip.cu_xy.map((f, i) => {
      if (clip.cu_cf[i][bc] >= 0.5 && f[bc][0] != null) return f[bc];
      const q = clip.sa_xy[i][mc];
      return clip.sa_cf[i][mc] >= 0.5 && q && q[0] != null ? q : null;
    });
    let last = raw.find((p) => p) || [clip.src_w / 2, clip.src_h / 2];
    const filled = raw.map((p) => (p ? (last = p) : last));
    return filled.map((_, i) => {
      let sx = 0, sy = 0, n = 0;
      for (let j = Math.max(0, i - 6); j <= Math.min(filled.length - 1, i + 6); j++) { sx += filled[j][0]; sy += filled[j][1]; n++; }
      return [sx / n, sy / n];
    });
  })();
  const clipEvents = D.s6.events.filter((e) => e.t >= clip.t[0] && e.t <= clip.t[clip.n - 1] && EVENT_META[e.e]);
  const frameAtT = (t) => C.bisect(clip.t, t);

  // ── S3 bout lookup for the clip frames ───────────────────────────
  const rleStarts = D.s3.rle.map((r) => r[0]);
  function boutOf(row) {
    const k = C.bisect(rleStarts, row + 1) - 1;
    const r = D.s3.rle[k];
    return { start: r[0], len: r[1], code: r[2], syl: D.s3.rle_syl[k] };
  }

  // ── Player ───────────────────────────────────────────────────────
  class ClipPlayer {
    constructor(el, o) {
      this.el = el; this.o = Object.assign({ mode: 's1' }, o);
      this.tracker = 'sa'; this.hideLow = false; this.step = 4; this.trail = true; this.zoom = true;
      el.innerHTML = `
        <div class="player">
          <div>
            <div class="stage-canvas"><canvas></canvas><div class="loading">Loading video frames…</div></div>
            <div class="p-controls">
              <button class="play-btn" aria-label="Play/pause"></button>
              <button class="btn" data-a="prev" title="Previous frame">&#9664;</button>
              <button class="btn" data-a="next" title="Next frame">&#9654;</button>
              <div class="seg" data-g="speed"><button data-v="0.5">0.5×</button><button data-v="1" class="on">1×</button><button data-v="2">2×</button></div>
              <label class="check"><input type="checkbox" data-c="zoom" checked> zoom on mouse</label>
              <span class="mode-ctl"></span>
              <span class="time"></span>
            </div>
            <div class="timeline"><canvas></canvas></div>
          </div>
          <div class="side"></div>
        </div>`;
      this.cv = el.querySelector('.stage-canvas canvas');
      this.tl = el.querySelector('.timeline canvas');
      this.side = el.querySelector('.side');
      this.timeEl = el.querySelector('.time');
      this.playBtn = el.querySelector('.play-btn');
      this.loading = el.querySelector('.loading');
      el.querySelector('[data-a=prev]').onclick = () => { setPlaying(false); setFrame(state.i - 1); };
      el.querySelector('[data-a=next]').onclick = () => { setPlaying(false); setFrame(state.i + 1); };
      this.playBtn.onclick = () => setPlaying(!state.playing);
      el.querySelectorAll('[data-g=speed] button').forEach((b) => b.onclick = () => {
        state.speed = +b.dataset.v;
        el.querySelectorAll('[data-g=speed] button').forEach((x) => x.classList.toggle('on', x === b));
      });
      el.querySelectorAll('[data-g=speed] button').forEach((x) => x.classList.toggle('on', +x.dataset.v === state.speed));
      el.querySelector('[data-c=zoom]').onchange = (e) => { this.zoom = e.target.checked; this.draw(); };
      this._modeControls();
      this._bindTimeline();
      this.listener = () => this.draw();
      state.listeners.add(this.listener);
      frames.forEach((im) => { if (!im.complete) im.addEventListener('load', () => { if (im === frames[state.i]) this.draw(); }, { once: true }); });
      this.ro = new ResizeObserver(() => this.draw()); this.ro.observe(el);
      C.registry.add({ el, render: () => this.draw() });
      this.draw();
    }
    destroy() { state.listeners.delete(this.listener); this.ro.disconnect(); }
    _modeControls() {
      const m = this.el.querySelector('.mode-ctl');
      if (this.o.mode === 's1') {
        m.innerHTML = `<div class="seg" data-g="trk"><button data-v="sa" class="on">SuperAnimal · 27</button><button data-v="cu">Custom AST · 7</button><button data-v="both">Both</button></div>
          <label class="check"><input type="checkbox" data-c="low"> hide conf &lt; 0.6</label>`;
        m.querySelectorAll('[data-g=trk] button').forEach((b) => b.onclick = () => {
          this.tracker = b.dataset.v; m.querySelectorAll('[data-g=trk] button').forEach((x) => x.classList.toggle('on', x === b)); this.draw();
        });
        m.querySelector('[data-c=low]').onchange = (e) => { this.hideLow = e.target.checked; this.draw(); };
      } else if (this.o.mode === 's2') {
        m.innerHTML = `<label class="check"><input type="checkbox" data-c="trail" checked> 2-s trail</label>`;
        m.querySelector('[data-c=trail]').onchange = (e) => { this.trail = e.target.checked; this.draw(); };
      } else {
        m.innerHTML = `<label class="check"><input type="checkbox" data-c="trail" checked> 3-s centroid trail</label>`;
        m.querySelector('[data-c=trail]').onchange = (e) => { this.trail = e.target.checked; this.draw(); };
      }
    }
    setStep(s) { this.step = s; this.draw(); }
    _bindTimeline() {
      const seek = (e) => {
        const r = this.tl.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width;
        setPlaying(false); setFrame(x * (clip.n - 1));
      };
      let down = false;
      this.tl.addEventListener('pointerdown', (e) => { down = true; this.tl.setPointerCapture(e.pointerId); seek(e); });
      this.tl.addEventListener('pointermove', (e) => {
        if (down) { seek(e); return; }
        const r = this.tl.getBoundingClientRect();
        const t = clip.t[0] + (e.clientX - r.left) / r.width * (clip.t[clip.n - 1] - clip.t[0]);
        const near = clipEvents.filter((ev) => Math.abs(ev.t - t) < 0.6);
        if (near.length) C.showTip(near.map((ev) => C.ttRow(C.col(EVENT_META[ev.e].color), EVENT_META[ev.e].label, ev.t.toFixed(2) + ' s')).join(''), e.clientX, e.clientY);
        else C.hideTip();
      });
      this.tl.addEventListener('pointerup', () => { down = false; });
      this.tl.addEventListener('pointerleave', () => C.hideTip());
    }

    // ── drawing ─────────────────────────────────────────────────────
    draw() {
      const i = state.i;
      this.playBtn.innerHTML = state.playing
        ? '<svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="1" width="3.5" height="12" fill="#fff"/><rect x="8.5" y="1" width="3.5" height="12" fill="#fff"/></svg>'
        : '<svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1l10 6-10 6z" fill="#fff"/></svg>';
      const t = clip.t[i];
      this.timeEl.textContent = `TDT ${t.toFixed(2)} s · frame ${i + 1}/${clip.n} · native #${clip.native[i]}`;
      const box = this.cv.parentElement;
      const W = box.clientWidth, H = box.clientHeight;
      if (!W) return;
      const ctx = C.sizeCanvas(this.cv, W, H);
      this.cv.style.height = '100%';
      const im = frames[i];
      const Z = this.zoom ? 2.2 : 1, cw = clip.src_w / Z, ch = clip.src_h / Z;
      let x0 = 0, y0 = 0;
      if (this.zoom) {
        x0 = Math.min(clip.src_w - cw, Math.max(0, centers[i][0] - cw / 2));
        y0 = Math.min(clip.src_h - ch, Math.max(0, centers[i][1] - ch / 2));
      }
      const k = clip.w / clip.src_w;          // JPEG px per source px
      if (im.complete && im.naturalWidth) { ctx.drawImage(im, x0 * k, y0 * k, cw * k, ch * k, 0, 0, W, H); this.loading.style.display = 'none'; }
      else { ctx.fillStyle = '#111'; ctx.fillRect(0, 0, W, H); }
      this.ctx = ctx; this.s = W / cw; this.ox = x0; this.oy = y0;
      if (this.o.mode === 's1') this._drawS1(i);
      else if (this.o.mode === 's2') this._drawS2(i);
      else this._drawS3(i);
      this._drawTimeline(i);
      this._side(i);
    }
    _tx(p) { return [(p[0] - this.ox) * this.s, (p[1] - this.oy) * this.s]; }
    _pt(xy, k) { const p = xy[k]; return p && p[0] != null && p[1] != null ? this._tx(p) : null; }
    _skeleton(xy, edges, color, width, alpha, okFn) {
      const ctx = this.ctx;
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.lineCap = 'round';
      edges.forEach(([a, b]) => {
        if (a < 0 || b < 0) return;
        const A = this._pt(xy, a), B = this._pt(xy, b);
        if (!A || !B || (okFn && (!okFn(a) || !okFn(b)))) return;
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }
    _dot(p, color, r, hollow, shape) {
      const ctx = this.ctx;
      ctx.beginPath();
      if (shape === 'diamond') { ctx.moveTo(p[0], p[1] - r - 1); ctx.lineTo(p[0] + r + 1, p[1]); ctx.lineTo(p[0], p[1] + r + 1); ctx.lineTo(p[0] - r - 1, p[1]); ctx.closePath(); }
      else ctx.arc(p[0], p[1], r, 0, 7);
      if (hollow) { ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke(); }
      else { ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = 'rgba(0,0,0,.65)'; ctx.lineWidth = 1; ctx.stroke(); }
    }
    _label(text, color, x = 10, y = 10) {
      const ctx = this.ctx;
      ctx.font = `600 13px ${C.cssVar('--font')}`;
      const w = ctx.measureText(text).width + (color ? 26 : 14);
      ctx.fillStyle = 'rgba(0,0,0,.62)';
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, 24, 6) : ctx.rect(x, y, w, 24); ctx.fill();
      if (color) { ctx.fillStyle = color; ctx.fillRect(x + 8, y + 7, 10, 10); }
      ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillText(text, x + (color ? 22 : 7), y + 12.5);
    }
    _drawS1(i) {
      const thr = D.config.preprocessing.conf_threshold;
      const showSA = this.tracker !== 'cu', showCU = this.tracker !== 'sa';
      if (showSA) {
        const xy = clip.sa_xy[i], cf = clip.sa_cf[i];
        const ok = (k) => cf[k] >= thr;
        this._skeleton(xy, SA_EDGES, '#ffffff', this.tracker === 'both' ? 1 : 1.4, 0.55, this.hideLow ? ok : null);
        SA.forEach((b, k) => {
          const p = this._pt(xy, k); if (!p) return;
          const low = !ok(k);
          if (low && this.hideLow) return;
          this._dot(p, REGION[regionOf(b)], this.tracker === 'both' ? 3 : 4, low);
        });
      }
      if (showCU) {
        const xy = clip.cu_xy[i], cf = clip.cu_cf[i];
        const ok = (k) => cf[k] >= D.config.fiber_joint.conf_threshold;
        this._skeleton(xy, CU_EDGES, '#7CFFB2', 1.8, 0.8, this.hideLow ? ok : null);
        CU.forEach((b, k) => {
          const p = this._pt(xy, k); if (!p) return;
          const low = !ok(k);
          if (low && this.hideLow) return;
          this._dot(p, '#7CFFB2', 3.6, low, 'diamond');
        });
      }
      this._label(this.tracker === 'sa' ? 'SuperAnimal TopViewMouse · 10 fps' : this.tracker === 'cu' ? 'Custom AST ResNet-50 · 20 fps' : 'SuperAnimal (circles) vs custom (diamonds)');
    }
    _s2Layer(i) {
      // returns per-USE keypoint {p:[x,y]|null, status}
      const thr = D.config.preprocessing.conf_threshold;
      const raw = clip.sa_xy[i], cf = clip.sa_cf[i], it = clip.s2_interp[i], sm = clip.s2_smooth[i];
      return USE_IDX.map((k) => {
        const low = !(cf[k] >= thr);
        const has = (a) => a[k] && a[k][0] != null;
        let p = null, status;
        if (this.step === 0) { p = raw[k]; status = low ? 'low' : 'ok'; }
        else if (this.step === 1) { p = low ? null : raw[k]; status = low ? 'masked' : 'ok'; }
        else if (this.step === 2) { p = has(it) ? it[k] : null; status = !has(it) ? 'nan' : low ? 'interp' : 'ok'; }
        else { p = has(sm) ? sm[k] : null; status = !has(sm) ? 'nan' : low ? 'interp' : 'ok'; }
        return { k, p, status, conf: cf[k] };
      });
    }
    _drawS2(i) {
      const ctx = this.ctx;
      const layer = this._s2Layer(i);
      // trail of nose + tail_base positions under the current step
      if (this.trail) {
        const keep = this.step;
        [USE.indexOf('nose'), USE.indexOf('tail_base')].forEach((u, j) => {
          ctx.strokeStyle = j ? REGION.tail : REGION.head; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85;
          ctx.beginPath(); let pen = false;
          for (let f = Math.max(0, i - 20); f <= i; f++) {
            const L = this._s2LayerAt(f, keep)[u];
            if (!L.p || L.p[0] == null) { pen = false; continue; }
            const [x, y] = this._tx(L.p);
            if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
          }
          ctx.stroke(); ctx.globalAlpha = 1;
        });
      }
      const xyFull = []; layer.forEach((L) => { xyFull[L.k] = L.p; });
      this._skeleton(xyFull, USE_EDGES, '#ffffff', 1.6, 0.75);
      layer.forEach((L) => {
        if (!L.p || L.p[0] == null) return;
        const p = this._tx(L.p);
        const c = REGION[regionOf(USE[USE_IDX.indexOf(L.k)])];
        if (L.status === 'low') this._dot(p, c, 3.6, true);
        else if (L.status === 'interp') { this._dot(p, '#ffffff', 5.5, true); this._dot(p, c, 3, false); }
        else this._dot(p, c, 3.6, false);
      });
      if (this.step === 4) { // body-length scale bar
        const bl = D.meta.s2_body_length_px * this.s;
        const x0 = 14, y0 = this.cv.parentElement.clientHeight - 20;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + bl, y0); ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.font = `12px ${C.cssVar('--font')}`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
        ctx.fillText(`1 body length = ${D.meta.s2_body_length_px.toFixed(1)} px (nose→tail_base median)`, x0, y0 - 5);
      }
      const names = ['Step 0 · raw S1 output', 'Step 1 · confidence mask (< 0.6 removed)', 'Step 2 · linear interpolation (gaps ≤ 10 frames)', 'Step 3 · Savitzky–Golay smoothing', 'Step 4 · body-length normalization'];
      this._label(names[this.step]);
    }
    _s2LayerAt(f, step) { const keep = this.step; this.step = step; const L = this._s2Layer(f); this.step = keep; return L; }
    _drawS3(i) {
      const ctx = this.ctx, code = clip.syl[i], c = sylColor(code);
      if (this.trail) {
        for (let f = Math.max(1, i - 30); f <= i; f++) {
          const a = clip.s2_smooth[f - 1][SA.indexOf('mouse_center')], b = clip.s2_smooth[f][SA.indexOf('mouse_center')];
          if (!a || !b || a[0] == null || b[0] == null) continue;
          ctx.strokeStyle = sylColor(clip.syl[f]); ctx.lineWidth = 3; ctx.lineCap = 'round';
          ctx.globalAlpha = 0.35 + 0.65 * (f - i + 30) / 30;
          const A = this._tx(a), B = this._tx(b);
          ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      const xy = clip.s2_smooth[i];
      this._skeleton(xy, USE_EDGES, c, 3, 1);
      USE_IDX.forEach((k) => { const p = this._pt(xy, k); if (p) this._dot(p, c, 3.6, false); });
      const b = boutOf(clip.rows[i]);
      this._label(`syllable s${String(clip.syl_raw[i]).padStart(2, '0')} · bout ${(b.len / 10).toFixed(1)} s`, c);
    }
    _drawTimeline(i) {
      const W = this.tl.parentElement.clientWidth || 600;
      const showSyl = this.o.mode === 's3';
      const H = showSyl ? 58 : 40;
      const ctx = C.sizeCanvas(this.tl, W, H);
      ctx.clearRect(0, 0, W, H);
      const n = clip.n, sx = (f) => f / (n - 1) * W;
      ctx.fillStyle = C.col('--surface-2'); ctx.fillRect(0, 0, W, H);
      // events
      ctx.font = `10.5px ${C.cssVar('--font')}`; ctx.textBaseline = 'top';
      let lastX = -999;
      clipEvents.forEach((ev) => {
        const x = Math.round(sx(frameAtT(ev.t))) + 0.5, m = EVENT_META[ev.e];
        ctx.strokeStyle = C.col(m.color); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, 22); ctx.stroke();
        if (x - lastX > 44 && x < W - 30) { ctx.fillStyle = C.col('--ink-2'); ctx.textAlign = 'left'; ctx.fillText(m.short, x + 3, 4); lastX = x; }
      });
      if (showSyl) {
        for (let f = 0; f < n; f++) { ctx.fillStyle = sylColor(clip.syl[f]); ctx.fillRect(sx(f), 26, W / n + 0.6, 14); }
      }
      // dF/F mini trace
      const y0 = showSyl ? 42 : 24, yh = H - y0 - 2;
      const v = clip.dff.filter((x) => x != null), lo = Math.min(...v), hi = Math.max(...v);
      ctx.strokeStyle = C.col('--ink-2'); ctx.lineWidth = 1; ctx.globalAlpha = .7; ctx.beginPath();
      clip.dff.forEach((d, f) => { if (d == null) return; const y = y0 + yh - (d - lo) / (hi - lo) * yh; f ? ctx.lineTo(sx(f), y) : ctx.moveTo(sx(f), y); });
      ctx.stroke(); ctx.globalAlpha = 1;
      // playhead
      const x = Math.round(sx(i)) + 0.5;
      ctx.strokeStyle = C.col('--accent'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    _side(i) {
      const t = clip.t[i];
      const base = `<div><h4>Frame</h4><dl class="kv readout">
          <dt>TDT time</dt><dd>${t.toFixed(2)} s</dd>
          <dt>Native frame (20 fps)</dt><dd>${clip.native[i]}</dd>
          <dt>S1–S3 row (10 fps)</dt><dd>${clip.rows[i]}</dd>
          <dt>Speed (S6)</dt><dd>${clip.speed[i] == null ? '–' : clip.speed[i].toFixed(1) + ' cm/s'}</dd>
          <dt>dF/F ch A (S6)</dt><dd>${clip.dff[i] == null ? '–' : clip.dff[i].toFixed(2) + ' %'}</dd>
        </dl></div>`;
      if (this.o.mode === 's1') {
        const useCU = this.tracker === 'cu';
        const names = useCU ? CU : SA, cf = useCU ? clip.cu_cf[i] : clip.sa_cf[i];
        const thr = 0.6;
        const rows = names.map((b, k) => `<div class="conf-row ${cf[k] < thr ? 'low' : ''}"><span class="nm" title="${b}">${b}</span><span class="tr"><b style="width:${(cf[k] || 0) * 100}%"></b><i class="thr"></i></span><span class="v">${cf[k] == null ? '–' : cf[k] < 0 ? '–' : cf[k].toFixed(2)}</span></div>`).join('');
        const nlow = cf.filter((v) => !(v >= thr)).length, nodet = !useCU && cf.every((v) => v < 0);
        this.side.innerHTML = base + `<div><h4>${useCU ? 'Custom model' : 'SuperAnimal'} confidence · ${nodet ? 'no detection' : `${nlow}/${names.length} below 0.6`}</h4><div class="conf-list">${rows}</div></div>`;
      } else if (this.o.mode === 's2') {
        const raw = this._s2LayerAt(i, 0), fin = this._s2LayerAt(i, 3);
        const lab = { ok: 'kept', masked: 'masked', interp: 'interpolated', nan: 'gap (not filled)', low: 'low conf' };
        const rows = USE.map((b, u) => {
          const st = this.step === 0 ? raw[u].status : this._s2LayerAt(i, Math.min(this.step, 3))[u].status;
          const p = fin[u].p;
          return `<tr><td class="mono">${b}</td><td class="num">${raw[u].conf == null ? '–' : raw[u].conf < 0 ? '–' : raw[u].conf.toFixed(2)}</td><td>${lab[st]}</td><td class="num">${p && p[0] != null ? (this.step === 4 ? (p[0] / D.meta.s2_body_length_px).toFixed(2) + ', ' + (p[1] / D.meta.s2_body_length_px).toFixed(2) : p[0].toFixed(0) + ', ' + p[1].toFixed(0)) : '–'}</td></tr>`;
        }).join('');
        this.side.innerHTML = base + `<div><h4>MoSeq keypoints at this frame</h4><div class="table-wrap"><table><thead><tr><th>keypoint</th><th class="num">conf</th><th>status</th><th class="num">${this.step === 4 ? 'x, y (BL)' : 'x, y (px)'}</th></tr></thead><tbody>${rows}</tbody></table></div></div>
          <div class="legend"><span><i style="background:${REGION.head};border-radius:50%"></i>kept</span><span><i class="ring" style="border-color:#888"></i>low confidence (raw)</span><span><i class="ring" style="border-color:var(--ink-2)"></i>ringed = interpolated</span></div>`;
      } else {
        const b = boutOf(clip.rows[i]);
        const code = clip.syl[i];
        const legend = D.s3.top.map((s, k) => `<span><i style="background:${sylColor(k)}"></i>s${String(s).padStart(2, '0')}</span>`).join('') + `<span><i style="background:${sylColor(7)}"></i>other</span>`;
        this.side.innerHTML = base + `<div><h4>Syllable (model_all)</h4>
          <div class="syl-badge"><i style="background:${sylColor(code)}"></i>s${String(clip.syl_raw[i]).padStart(2, '0')}</div>
          <dl class="kv readout" style="margin-top:6px">
            <dt>Bout start (row)</dt><dd>${b.start}</dd>
            <dt>Bout duration</dt><dd>${(b.len / 10).toFixed(1)} s</dd>
            <dt>Session usage</dt><dd>${code < 7 ? (D.s3.usage_session[code] * 100).toFixed(1) + ' %' : '–'}</dd>
          </dl></div><div><h4>Colour key (top 7 by global usage)</h4><div class="legend">${legend}</div></div>`;
      }
    }
  }

  window.Player = { ClipPlayer, state, setFrame, setPlaying, sylColor, sylName, EVENT_META, SYL_VARS, REGION, regionOf, frameAtT, clipEvents };
})();
