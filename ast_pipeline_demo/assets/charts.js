/* Minimal dependency-free chart kit for the AST demo.
 * Canvas time-series with brush zoom, linked cursors, hover tooltips;
 * heatmaps, ethogram strips, event rasters; DOM bar lists and SVG dot strips. */
(function () {
  'use strict';

  const DPR = () => window.devicePixelRatio || 1;
  const rootStyle = () => getComputedStyle(document.documentElement);
  const cssVar = (n) => rootStyle().getPropertyValue(n).trim();
  const col = (c) => (c && c.startsWith('--') ? cssVar(c) : c);
  const isDark = () => {
    const t = document.documentElement.dataset.theme;
    if (t) return t === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  const registry = new Set();
  const rerenderAll = () => registry.forEach((c) => { if (c.el.isConnected) c.render(); else registry.delete(c); });

  // ── tooltip ──────────────────────────────────────────────────────
  const tt = document.getElementById('tooltip');
  function showTip(html, cx, cy) {
    tt.innerHTML = html;
    tt.classList.add('show');
    const r = tt.getBoundingClientRect();
    let x = cx + 14, y = cy + 14;
    if (x + r.width > window.innerWidth - 8) x = cx - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = cy - r.height - 14;
    tt.style.left = Math.max(8, x) + 'px';
    tt.style.top = Math.max(8, y) + 'px';
  }
  const hideTip = () => tt.classList.remove('show');
  const ttRow = (color, name, val) =>
    `<div class="tt-r"><span>${color ? `<i style="background:${color}"></i>` : ''}${name}</span><span>${val}</span></div>`;

  // ── helpers ──────────────────────────────────────────────────────
  function sizeCanvas(c, w, h) {
    const d = DPR();
    c.width = Math.max(1, Math.round(w * d));
    c.height = Math.max(1, Math.round(h * d));
    c.style.height = h + 'px';
    const ctx = c.getContext('2d');
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return ctx;
  }
  function ticks(a, b, n = 5) {
    const span = b - a;
    if (!(span > 0) || !isFinite(span)) return [a];
    const s0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(s0))), e = s0 / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmtNum(v, d) {
    if (v == null || !isFinite(v)) return '–';
    if (d != null) return v.toFixed(d);
    if (Math.abs(v) < 1e-9) return '0';
    const a = Math.abs(v);
    if (a >= 1000) return (v / 1000).toFixed(a >= 10000 ? 0 : 1) + 'K';
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }
  function fmtClock(s) {
    if (!isFinite(s)) return '–';
    const sign = s < 0 ? '-' : '';
    s = Math.abs(s);
    const m = Math.floor(s / 60), r = s - m * 60;
    return `${sign}${m}:${r < 10 ? '0' : ''}${r.toFixed(r < 100 && m < 2 ? 1 : 0)}`;
  }
  function bisect(arr, x) { // first index with arr[i] >= x
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
    return lo;
  }
  function hexToRgb(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mix(a, b, t) {
    const A = hexToRgb(a), B = hexToRgb(b);
    return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
  }
  function rampAt(stops, t) {
    t = Math.min(1, Math.max(0, t));
    const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p));
    return mix(stops[i], stops[i + 1], p - i);
  }
  const SEQ_LIGHT = ['#f3f6fb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  const SEQ_DARK = ['#1d2129', '#163156', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'];
  const seq = (t) => rampAt(isDark() ? SEQ_DARK : SEQ_LIGHT, t);
  function div(t) { // t in [-1, 1]
    const mid = isDark() ? '#383835' : '#f0efec';
    const neg = isDark() ? ['#383835', '#256abf', '#9ec5f4'] : ['#f0efec', '#6da7ec', '#184f95'];
    const pos = isDark() ? ['#383835', '#c94a4a', '#f2a3a3'] : ['#f0efec', '#e98585', '#a32b2a'];
    if (!isFinite(t)) return mid;
    return t < 0 ? rampAt(neg, -t) : rampAt(pos, t);
  }
  function withAlpha(color, a) {
    if (color.startsWith('#')) { const [r, g, b] = hexToRgb(color); return `rgba(${r},${g},${b},${a})`; }
    if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', `,${a})`);
    return color;
  }

  // ── Link group: shared x-domain + cursor ─────────────────────────
  class LinkGroup {
    constructor() { this.members = []; }
    add(c) { this.members.push(c); }
    setX(x0, x1, src) { this.members.forEach((c) => c !== src && c.setX(x0, x1, true)); if (this.onX) this.onX(x0, x1); }
    setCursor(x, src) { this.members.forEach((c) => c !== src && c.setCursor(x, true)); }
    reset(src) { this.members.forEach((c) => c !== src && c.resetX(true)); if (this.onX) this.onX(null, null); }
  }

  // ── TimeBase: shared x-axis behaviour (brush zoom, cursor, click) ─
  class TimeBase {
    constructor(el, o) {
      this.el = el;
      this.o = Object.assign({ height: 160, pad: { l: 54, r: 14, t: 10, b: 24 }, xFmt: (v) => fmtNum(v), brush: true }, o);
      this.o.pad = Object.assign({ l: 54, r: 14, t: 10, b: 24 }, o.pad || {});
      el.classList.add('chart');
      this.canvas = document.createElement('canvas');
      el.appendChild(this.canvas);
      this.full = this.o.xDomain;
      this.x0 = this.full[0]; this.x1 = this.full[1];
      this.cursor = null; this.drag = null;
      if (this.o.group) this.o.group.add(this);
      this._bind();
      registry.add(this);
      this.ro = new ResizeObserver(() => this.render());
      this.ro.observe(el);
    }
    get plot() {
      const p = this.o.pad, w = this.el.clientWidth || 600, h = this.o.height;
      return { l: p.l, r: w - p.r, t: p.t, b: h - p.b, w, h };
    }
    sx(x) { const P = this.plot; return P.l + (x - this.x0) / (this.x1 - this.x0) * (P.r - P.l); }
    ix(px) { const P = this.plot; return this.x0 + (px - P.l) / (P.r - P.l) * (this.x1 - this.x0); }
    setX(x0, x1, fromGroup) {
      this.x0 = x0; this.x1 = x1; this.render();
      if (!fromGroup && this.o.group) this.o.group.setX(x0, x1, this);
    }
    resetX(fromGroup) {
      this.x0 = this.full[0]; this.x1 = this.full[1]; this.render();
      if (!fromGroup && this.o.group) this.o.group.reset(this);
    }
    setCursor(x, fromGroup) {
      this.cursor = x; this.render();
      if (!fromGroup && this.o.group) this.o.group.setCursor(x, this);
    }
    _bind() {
      const c = this.canvas;
      const pos = (e) => { const r = c.getBoundingClientRect(); const pt = e.touches ? e.touches[0] : e; return { x: pt.clientX - r.left, y: pt.clientY - r.top, cx: pt.clientX, cy: pt.clientY }; };
      c.addEventListener('pointerdown', (e) => {
        const p = pos(e), P = this.plot;
        if (p.x < P.l || p.x > P.r) return;
        this.drag = { x: p.x, now: p.x };
        c.setPointerCapture(e.pointerId);
      });
      c.addEventListener('pointermove', (e) => {
        const p = pos(e), P = this.plot;
        if (this.drag) { this.drag.now = Math.min(P.r, Math.max(P.l, p.x)); this.render(); return; }
        if (p.x < P.l || p.x > P.r) { this._leave(); return; }
        const x = this.ix(p.x);
        this.setCursor(x);
        const html = this.tooltip ? this.tooltip(x, p) : null;
        if (html) showTip(html, p.cx, p.cy); else hideTip();
      });
      c.addEventListener('pointerup', (e) => {
        if (!this.drag) return;
        const d = this.drag; this.drag = null;
        if (Math.abs(d.now - d.x) > 6 && this.o.brush) {
          const a = this.ix(Math.min(d.x, d.now)), b = this.ix(Math.max(d.x, d.now));
          this.setX(a, b);
        } else {
          this.render();
          if (this.o.onClick) this.o.onClick(this.ix(d.x));
        }
      });
      c.addEventListener('pointerleave', () => this._leave());
      c.addEventListener('dblclick', () => this.resetX());
    }
    _leave() { hideTip(); if (this.cursor != null) this.setCursor(null); }
    _begin() {
      const P = this.plot;
      const ctx = sizeCanvas(this.canvas, P.w, P.h);
      ctx.clearRect(0, 0, P.w, P.h);
      ctx.font = `11px ${cssVar('--font')}`;
      return { ctx, P };
    }
    _bands(ctx, P) {
      (this.o.bands || []).forEach((b, i) => {
        const a = Math.max(P.l, this.sx(b.x0)), z = Math.min(P.r, this.sx(b.x1));
        if (z <= a) return;
        ctx.fillStyle = col(b.fill || (i % 2 ? '--band-2' : '--band'));
        ctx.fillRect(a, P.t, z - a, P.b - P.t);
        if (b.label && this.o.bandLabels !== false) {
          ctx.fillStyle = col('--ink-2'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
          ctx.font = `600 10.5px ${cssVar('--font')}`;
          if (z - a > 26) ctx.fillText(b.label, a + 4, P.t + 3);
          ctx.font = `11px ${cssVar('--font')}`;
        }
      });
      (this.o.highlight ? [this.o.highlight] : []).forEach((h) => {
        const a = Math.max(P.l, this.sx(h.x0)), z = Math.min(P.r, this.sx(h.x1));
        if (z <= a) return;
        ctx.strokeStyle = col('--accent'); ctx.lineWidth = 1;
        ctx.strokeRect(a + 0.5, P.t + 0.5, z - a - 1, P.b - P.t - 1);
      });
    }
    _xAxis(ctx, P) {
      if (this.o.noXAxis) return;
      ctx.fillStyle = col('--muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const tk = ticks(this.x0, this.x1, Math.max(2, Math.floor((P.r - P.l) / 90)));
      tk.forEach((v) => ctx.fillText(this.o.xFmt(v), this.sx(v), P.b + 6));
      ctx.strokeStyle = col('--axis'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(P.l, P.b + 0.5); ctx.lineTo(P.r, P.b + 0.5); ctx.stroke();
    }
    _cursorAndBrush(ctx, P) {
      if (this.cursor != null && this.cursor >= this.x0 && this.cursor <= this.x1) {
        const x = Math.round(this.sx(this.cursor)) + 0.5;
        ctx.strokeStyle = col('--ink-2'); ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke(); ctx.globalAlpha = 1;
      }
      if (this.o.marker != null && this.o.marker >= this.x0 && this.o.marker <= this.x1) {
        const x = Math.round(this.sx(this.o.marker)) + 0.5;
        ctx.strokeStyle = col('--accent'); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke();
      }
      if (this.drag) {
        const a = Math.min(this.drag.x, this.drag.now), b = Math.max(this.drag.x, this.drag.now);
        ctx.fillStyle = withAlpha(col('--accent'), 0.12);
        ctx.fillRect(a, P.t, b - a, P.b - P.t);
      }
    }
    setMarker(x) { this.o.marker = x; this.render(); }
  }

  // ── LineChart ────────────────────────────────────────────────────
  class LineChart extends TimeBase {
    constructor(el, o) {
      const s0 = o.series[0];
      const xd = o.xDomain || [s0.x[0], s0.x[s0.x.length - 1]];
      super(el, Object.assign({}, o, { xDomain: xd }));
      this.series = o.series;
      this.render();
    }
    setSeries(series, yDomain) { this.series = series; if (yDomain !== undefined) this.o.yDomain = yDomain; this.render(); }
    _range(s) {
      const i0 = Math.max(0, bisect(s.x, this.x0) - 1), i1 = Math.min(s.x.length, bisect(s.x, this.x1) + 1);
      return [i0, i1];
    }
    _yDomain() {
      if (this.o.yDomain) return this.o.yDomain;
      const vals = [];
      this.series.forEach((s) => {
        if (s.noScale) return;
        const [i0, i1] = this._range(s);
        const step = Math.max(1, Math.floor((i1 - i0) / 4000));
        for (let i = i0; i < i1; i += step) {
          const v = s.y[i]; if (v != null && isFinite(v)) vals.push(v);
          if (s.lo) { const a = s.lo[i], b = s.hi[i]; if (a != null && isFinite(a)) vals.push(a); if (b != null && isFinite(b)) vals.push(b); }
        }
      });
      if (!vals.length) return [0, 1];
      vals.sort((a, b) => a - b);
      const q = this.o.robust ? 0.004 : 0;
      let lo = vals[Math.floor(q * (vals.length - 1))], hi = vals[Math.ceil((1 - q) * (vals.length - 1))];
      if (this.o.yZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      if (lo === hi) { lo -= 1; hi += 1; }
      const pad = (hi - lo) * 0.06;
      return [this.o.yZero && lo === 0 ? 0 : lo - pad, hi + pad];
    }
    render() {
      const { ctx, P } = this._begin();
      const [y0, y1] = this._yDomain();
      this.yd = [y0, y1];
      const sy = (v) => P.b - (v - y0) / (y1 - y0) * (P.b - P.t);
      this._bands(ctx, P);
      // grid + y ticks
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ticks(y0, y1, Math.max(2, Math.floor((P.b - P.t) / 36))).forEach((v) => {
        const y = Math.round(sy(v)) + 0.5;
        ctx.strokeStyle = col('--grid'); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(P.r, y); ctx.stroke();
        ctx.fillStyle = col('--muted'); ctx.fillText(this.o.yFmt ? this.o.yFmt(v) : +v.toPrecision(4) + '', P.l - 6, y);
      });
      if (this.o.yLabel) {
        ctx.save(); ctx.translate(11, (P.t + P.b) / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = col('--ink-2');
        ctx.fillText(this.o.yLabel, 0, 0); ctx.restore();
      }
      if (this.o.zeroLine && y0 < 0 && y1 > 0) {
        ctx.strokeStyle = col('--axis'); ctx.lineWidth = 1;
        const y = Math.round(sy(0)) + 0.5; ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(P.r, y); ctx.stroke();
      }
      (this.o.vlines || []).forEach((v) => {
        if (v.x < this.x0 || v.x > this.x1) return;
        const x = Math.round(this.sx(v.x)) + 0.5;
        ctx.strokeStyle = col(v.color || '--axis'); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke();
        if (v.label) { ctx.fillStyle = col('--muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(v.label, x + 3, P.t + 2); }
      });
      ctx.save();
      ctx.beginPath(); ctx.rect(P.l, P.t - 2, P.r - P.l, P.b - P.t + 4); ctx.clip();
      this.series.forEach((s) => this._drawSeries(ctx, P, s, sy));
      ctx.restore();
      this._xAxis(ctx, P);
      // cursor dots
      if (this.cursor != null && this.o.cursorDots !== false) {
        this.series.forEach((s) => {
          if (s.points || s.noCursor) return;
          const i = this._nearest(s, this.cursor);
          if (i < 0) return;
          const v = s.y[i]; if (v == null || !isFinite(v)) return;
          const x = this.sx(s.x[i]), y = sy(v);
          if (y < P.t || y > P.b) return;
          ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fillStyle = col(s.color); ctx.fill();
          ctx.lineWidth = 2; ctx.strokeStyle = col('--surface'); ctx.stroke();
        });
      }
      this._cursorAndBrush(ctx, P);
    }
    _nearest(s, x) {
      const i = bisect(s.x, x);
      if (i <= 0) return 0;
      if (i >= s.x.length) return s.x.length - 1;
      return x - s.x[i - 1] < s.x[i] - x ? i - 1 : i;
    }
    _drawSeries(ctx, P, s, sy) {
      const c = col(s.color);
      const [i0, i1] = this._range(s);
      const n = i1 - i0, pw = P.r - P.l;
      if (s.lo) { // band
        ctx.fillStyle = withAlpha(c, s.bandAlpha || 0.14);
        let open = false;
        const flush = (stack) => {
          if (stack.length < 2) return;
          ctx.beginPath();
          stack.forEach(([x, lo], k) => (k ? ctx.lineTo(x, sy(lo)) : ctx.moveTo(x, sy(lo))));
          for (let k = stack.length - 1; k >= 0; k--) ctx.lineTo(stack[k][0], sy(stack[k][2]));
          ctx.closePath(); ctx.fill();
        };
        let stack = [];
        for (let i = i0; i < i1; i++) {
          const lo = s.lo[i], hi = s.hi[i];
          if (lo == null || hi == null || !isFinite(lo) || !isFinite(hi)) { flush(stack); stack = []; continue; }
          stack.push([this.sx(s.x[i]), lo, hi]);
        }
        flush(stack); void open;
      }
      if (s.points) {
        const r = s.r || 2.5;
        for (let i = i0; i < i1; i++) {
          const v = s.y[i]; if (v == null || !isFinite(v)) continue;
          const x = this.sx(s.x[i]), y = sy(v);
          ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
          if (s.hollow) { ctx.strokeStyle = c; ctx.lineWidth = 1.2; ctx.stroke(); }
          else { ctx.fillStyle = c; ctx.fill(); }
        }
        return;
      }
      if (s.noLine) return;
      ctx.strokeStyle = c; ctx.lineWidth = s.width || 1.6; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (s.dash) ctx.setLineDash(s.dash);
      ctx.globalAlpha = s.alpha || 1;
      if (n > pw * 2.2) { // per-pixel min/max decimation
        let curPx = -1, mn = Infinity, mx = -Infinity, first = null, last = null;
        ctx.beginPath(); let pen = false;
        const emit = () => {
          if (curPx < 0 || mn === Infinity) { pen = false; return; }
          if (!pen) { ctx.moveTo(curPx, sy(first)); pen = true; } else ctx.lineTo(curPx, sy(first));
          ctx.lineTo(curPx, sy(mn)); ctx.lineTo(curPx, sy(mx)); ctx.lineTo(curPx, sy(last));
        };
        for (let i = i0; i < i1; i++) {
          const v = s.y[i];
          const px = Math.round(this.sx(s.x[i]));
          if (px !== curPx) { emit(); curPx = px; mn = Infinity; mx = -Infinity; first = null; }
          if (v == null || !isFinite(v)) { if (mn === Infinity) pen = false; continue; }
          if (first == null) first = v;
          last = v; if (v < mn) mn = v; if (v > mx) mx = v;
        }
        emit(); ctx.stroke();
      } else {
        ctx.beginPath(); let pen = false;
        for (let i = i0; i < i1; i++) {
          const v = s.y[i];
          if (v == null || !isFinite(v)) { pen = false; continue; }
          const x = this.sx(s.x[i]), y = sy(v);
          if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    tooltip(x) {
      if (this.o.tooltip === false) return null;
      const rows = this.series.filter((s) => !s.noTip).map((s) => {
        const i = this._nearest(s, x);
        const v = s.y[i];
        return ttRow(col(s.color), s.name, v == null || !isFinite(v) ? '–' : (s.fmt || this.o.vFmt || fmtNum)(v));
      }).join('');
      return `<div class="tt-h">${this.o.xFmtLong ? this.o.xFmtLong(x) : this.o.xFmt(x)}</div>${rows}`;
    }
  }

  // ── StripChart: categorical segments over time (ethogram) ────────
  class StripChart extends TimeBase {
    // o.segments: [[t0, t1, code, meta]], o.color(code), o.label(code), o.rows? (single row)
    constructor(el, o) {
      super(el, Object.assign({ height: 44, pad: { l: 54, r: 14, t: 4, b: 18 } }, o));
      this.render();
    }
    render() {
      const { ctx, P } = this._begin();
      const segs = this.o.segments;
      const h = P.b - P.t;
      const i0 = Math.max(0, bisect(this.o.starts, this.x0) - 1);
      for (let i = i0; i < segs.length; i++) {
        const [a, b, code] = segs[i];
        if (a > this.x1) break;
        if (b < this.x0) continue;
        const xa = Math.max(P.l, this.sx(a)), xb = Math.min(P.r, this.sx(b));
        ctx.fillStyle = this.o.color(code);
        ctx.fillRect(xa, P.t, Math.max(0.6, xb - xa), h);
      }
      if (this.o.title) {
        ctx.fillStyle = col('--ink-2'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(this.o.title, P.l - 6, P.t + h / 2);
      }
      this._xAxis(ctx, P);
      this._cursorAndBrush(ctx, P);
    }
    segAt(x) {
      const i = bisect(this.o.starts, x) - 1;
      if (i < 0) return null;
      const s = this.o.segments[i];
      return s && x <= s[1] ? s : null;
    }
    tooltip(x) {
      const s = this.segAt(x);
      if (!s) return null;
      return this.o.tip ? this.o.tip(s, x) : `<div class="tt-h">${this.o.label(s[2])}</div>`;
    }
  }

  // ── RasterChart: event ticks in labelled rows ────────────────────
  class RasterChart extends TimeBase {
    // o.rows: [{key, label, color, times:[]}]
    constructor(el, o) {
      const rowH = o.rowH || 16;
      super(el, Object.assign({ height: o.rows.length * rowH + 26, pad: { l: 120, r: 14, t: 4, b: 22 } }, o));
      this.rowH = rowH;
      this.render();
    }
    render() {
      const { ctx, P } = this._begin();
      this._bands(ctx, P);
      this.o.rows.forEach((r, k) => {
        const y = P.t + k * this.rowH;
        ctx.fillStyle = col('--ink-2'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(r.label, P.l - 8, y + this.rowH / 2);
        ctx.strokeStyle = col('--grid'); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(P.l, y + this.rowH - 0.5); ctx.lineTo(P.r, y + this.rowH - 0.5); ctx.stroke();
        ctx.strokeStyle = col(r.color); ctx.lineWidth = 2; ctx.lineCap = 'round';
        r.times.forEach((t) => {
          if (t < this.x0 || t > this.x1) return;
          const x = Math.round(this.sx(t)) + 0.5;
          ctx.beginPath(); ctx.moveTo(x, y + 3); ctx.lineTo(x, y + this.rowH - 3); ctx.stroke();
        });
      });
      this._xAxis(ctx, P);
      this._cursorAndBrush(ctx, P);
    }
    tooltip(x, p) {
      const k = Math.floor((p.y - this.plot.t) / this.rowH);
      const r = this.o.rows[k];
      if (!r) return null;
      const tol = (this.x1 - this.x0) / (this.plot.r - this.plot.l) * 5;
      const near = r.times.filter((t) => Math.abs(t - x) <= tol);
      if (!near.length) return null;
      return `<div class="tt-h">${r.label}</div>${near.slice(0, 4).map((t) => ttRow(null, 'TDT time', this.o.xFmtLong ? this.o.xFmtLong(t) : fmtNum(t))).join('')}`;
    }
  }

  // ── Heatmap (matrix) ─────────────────────────────────────────────
  class Heatmap {
    // o: rows [labels], cols? [labels], data [r][c], color(v, r, c), height, rowLabelW, tip(r,c,v), xAxis {d0,d1,fmt}
    constructor(el, o) {
      this.el = el; this.o = Object.assign({ rowLabelW: 110, height: null, colLabelH: 0 }, o);
      el.classList.add('chart');
      this.canvas = document.createElement('canvas'); el.appendChild(this.canvas);
      registry.add(this);
      this.canvas.addEventListener('pointermove', (e) => this._move(e));
      this.canvas.addEventListener('pointerleave', () => { hideTip(); this.hover = null; this.render(); });
      if (o.onClick) this.canvas.addEventListener('click', () => this.hover && o.onClick(this.hover[0], this.hover[1]));
      this.ro = new ResizeObserver(() => this.render()); this.ro.observe(el);
      this.render();
    }
    get geom() {
      const nr = this.o.data.length, nc = this.o.data[0].length;
      const w = this.el.clientWidth || 500;
      const L = this.o.rowLabelW, T = this.o.colLabelH, B = this.o.xAxis ? 22 : 4;
      const cw = (w - L - 8) / nc;
      const ch = this.o.cellH || (this.o.square ? cw : 12);
      const h = this.o.height || (T + nr * ch + B);
      return { nr, nc, w, h, L, T, cw, ch: this.o.height ? (h - T - B) / nr : ch, B };
    }
    render() {
      const G = this.geom, ctx = sizeCanvas(this.canvas, G.w, G.h);
      ctx.clearRect(0, 0, G.w, G.h);
      ctx.font = `11px ${cssVar('--font')}`;
      const gap = this.o.gap != null ? this.o.gap : (G.cw > 10 ? 1 : 0);
      for (let r = 0; r < G.nr; r++) {
        for (let c = 0; c < G.nc; c++) {
          const v = this.o.data[r][c];
          ctx.fillStyle = this.o.color(v, r, c);
          ctx.fillRect(G.L + c * G.cw, G.T + r * G.ch, G.cw - gap + (gap ? 0 : 0.6), G.ch - (G.ch > 6 ? 1 : 0));
          if (this.o.cellText && G.cw > 30 && G.ch > 16) {
            const txt = this.o.cellText(v, r, c);
            if (txt) {
              const bg = this.o.color(v, r, c).match(/\d+/g) || [255, 255, 255];
              const lum = (0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]) / 255;
              ctx.fillStyle = lum > 0.55 ? '#0b0b0b' : '#ffffff';
              ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(txt, G.L + c * G.cw + G.cw / 2, G.T + r * G.ch + G.ch / 2);
            }
          }
        }
        if (G.ch >= 9) {
          ctx.fillStyle = (this.o.rowStrong && this.o.rowStrong(r)) ? col('--ink') : col('--ink-2');
          ctx.font = `${this.o.rowStrong && this.o.rowStrong(r) ? '600 ' : ''}${G.ch < 12 ? 10 : 11}px ${this.o.mono ? cssVar('--mono') : cssVar('--font')}`;
          ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
          ctx.fillText(this.o.rows[r], G.L - 6, G.T + r * G.ch + G.ch / 2);
        }
      }
      ctx.font = `11px ${cssVar('--font')}`;
      if (this.o.cols && G.T) {
        ctx.fillStyle = col('--ink-2'); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        this.o.cols.forEach((lab, c) => ctx.fillText(lab, G.L + c * G.cw + G.cw / 2, G.T - 4));
      }
      if (this.o.xAxis) {
        const { d0, d1, fmt } = this.o.xAxis;
        ctx.fillStyle = col('--muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const W = G.nc * G.cw;
        ticks(d0, d1, Math.max(2, Math.floor(W / 90))).forEach((v) => {
          const x = G.L + (v - d0) / (d1 - d0) * W;
          ctx.fillText(fmt ? fmt(v) : fmtNum(v), x, G.T + G.nr * G.ch + 6);
        });
        if (this.o.vline != null) {
          const x = G.L + (this.o.vline - d0) / (d1 - d0) * W;
          ctx.strokeStyle = col('--ink'); ctx.lineWidth = 1; ctx.globalAlpha = .6;
          ctx.beginPath(); ctx.moveTo(x + .5, G.T); ctx.lineTo(x + .5, G.T + G.nr * G.ch); ctx.stroke(); ctx.globalAlpha = 1;
        }
      }
      if (this.hover) {
        const [r, c] = this.hover;
        ctx.strokeStyle = col('--ink'); ctx.lineWidth = 1.5;
        if (this.o.hoverRow) ctx.strokeRect(G.L + 0.75, G.T + r * G.ch + 0.75, G.nc * G.cw - 1.5, G.ch - 1.5);
        else ctx.strokeRect(G.L + c * G.cw + 0.75, G.T + r * G.ch + 0.75, G.cw - 1.5, G.ch - 1.5);
      }
    }
    _move(e) {
      const G = this.geom, rc = this.canvas.getBoundingClientRect();
      const x = e.clientX - rc.left, y = e.clientY - rc.top;
      const c = Math.floor((x - G.L) / G.cw), r = Math.floor((y - G.T) / G.ch);
      if (r < 0 || r >= G.nr || c < 0 || c >= G.nc) { hideTip(); this.hover = null; this.render(); return; }
      this.hover = [r, c]; this.render();
      if (this.o.tip) showTip(this.o.tip(r, c, this.o.data[r][c]), e.clientX, e.clientY);
    }
    update(data, extra) { this.o.data = data; Object.assign(this.o, extra || {}); this.render(); }
  }

  // ── DOM bar list ─────────────────────────────────────────────────
  function barList(el, rows, o) {
    // rows: [{name, a, b?, strong, tip}], o: {max, colorA, colorB, fmt, nameW}
    el.innerHTML = '';
    el.classList.add('bars');
    const max = o.max || Math.max(...rows.map((r) => Math.max(r.a || 0, r.b || 0))) || 1;
    rows.forEach((r) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      if (o.nameW || o.valW) row.style.gridTemplateColumns = `${o.nameW || 112}px 1fr ${o.valW || 50}px`;
      const two = r.b != null;
      row.innerHTML = `<div class="nm ${r.strong ? 'use' : ''}" title="${r.name}">${r.name}</div>
        <div class="track">
          <div class="fill ${two ? 'a' : 'single'}" style="width:${Math.max(0, r.a) / max * 100}%;background:${col(r.colorA || o.colorA)}"></div>
          ${two ? `<div class="fill b" style="width:${Math.max(0, r.b) / max * 100}%;background:${col(o.colorB)}"></div>` : ''}
        </div>
        <div class="v">${two && o.both ? `${(o.fmt || fmtNum)(r.a)} / ${(o.fmt || fmtNum)(r.b)}` : (o.fmt || fmtNum)(two && o.showB ? r.b : r.a)}</div>`;
      if (r.tip) {
        row.addEventListener('pointermove', (e) => showTip(r.tip, e.clientX, e.clientY));
        row.addEventListener('pointerleave', hideTip);
      }
      el.appendChild(row);
    });
  }

  // ── SVG dot strip (categories × value, dots by group) ────────────
  function dotStrip(el, o) {
    // o: cats [{label, strong}], points [{cat, v, group, current, tip}], groups [{key, color}], xmax, xfmt, rowH
    const render = () => {
      const w = el.clientWidth || 600, rowH = o.rowH || 30, L = o.labelW || 70, R = 16, T = 6;
      const h = T + o.cats.length * rowH + 22;
      const xmax = o.xmax || Math.max(...o.points.map((p) => p.v)) * 1.08 || 1;
      const xmin = o.xmin || 0;
      const sx = (v) => L + (v - xmin) / (xmax - xmin) * (w - L - R);
      const ns = 'http://www.w3.org/2000/svg';
      let s = `<svg xmlns="${ns}" width="${w}" height="${h}" style="display:block">`;
      ticks(xmin, xmax, Math.max(2, Math.floor((w - L - R) / 80))).forEach((v) => {
        s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${h - 22}" stroke="${col('--grid')}" />`;
        s += `<text x="${sx(v)}" y="${h - 6}" text-anchor="middle" font-size="11" fill="${col('--muted')}">${o.xfmt ? o.xfmt(v) : fmtNum(v)}</text>`;
      });
      o.cats.forEach((c, k) => {
        const y = T + k * rowH + rowH / 2;
        s += `<text x="${L - 8}" y="${y + 4}" text-anchor="end" font-size="12" font-family="${cssVar('--mono')}" fill="${col(c.strong ? '--ink' : '--ink-2')}" ${c.strong ? 'font-weight="600"' : ''}>${c.label}</text>`;
        o.groups.forEach((g, gi) => {
          const pts = o.points.filter((p) => p.cat === k && p.group === g.key);
          if (!pts.length) return;
          const m = pts.reduce((a, p) => a + p.v, 0) / pts.length;
          const yy = y + (gi - (o.groups.length - 1) / 2) * 8;
          s += `<line x1="${sx(m)}" x2="${sx(m)}" y1="${yy - 6}" y2="${yy + 6}" stroke="${col(g.color)}" stroke-width="2.5" stroke-linecap="round" />`;
        });
      });
      o.points.forEach((p, i) => {
        const gi = o.groups.findIndex((g) => g.key === p.group);
        const y = T + p.cat * rowH + rowH / 2 + (gi - (o.groups.length - 1) / 2) * 8;
        const g = o.groups[gi];
        if (p.current) {
          s += `<circle cx="${sx(p.v)}" cy="${y}" r="7" fill="none" stroke="${col('--ink')}" stroke-width="2" />`;
        }
        s += `<circle data-i="${i}" cx="${sx(p.v)}" cy="${y}" r="4" fill="${col(g.color)}" fill-opacity="0.85" stroke="${col('--surface')}" stroke-width="1.5" />`;
      });
      s += '</svg>';
      el.innerHTML = s;
      el.querySelectorAll('circle[data-i]').forEach((c) => {
        const p = o.points[+c.dataset.i];
        c.style.cursor = 'default';
        c.addEventListener('pointermove', (e) => showTip(p.tip, e.clientX, e.clientY));
        c.addEventListener('pointerleave', hideTip);
      });
    };
    const ro = new ResizeObserver(render); ro.observe(el);
    const handle = { el, render };
    registry.add(handle);
    render();
    return handle;
  }

  window.Charts = {
    LinkGroup, LineChart, StripChart, RasterChart, Heatmap, barList, dotStrip,
    showTip, hideTip, ttRow, fmtNum, fmtClock, ticks, col, cssVar, seq, div, withAlpha, mix, isDark,
    rerenderAll, registry, sizeCanvas, bisect,
  };
})();
