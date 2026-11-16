// homelab-mlops dashboard. Renders site/model_history.js (window.MODEL_HISTORY),
// built from REPORTS/model_metrics_*.json by scripts/build_history.py.

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[c]));
const fmtDate = (iso, opts = { day: 'numeric', month: 'short' }) =>
  new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-GB', { ...opts, timeZone: 'UTC' });
const pct = (r, dp = 1) => `${(r * 100).toFixed(dp)}%`;
const MODEL_NAMES = { LocalOutlierFactor: 'Local Outlier Factor', IsolationForest: 'Isolation Forest' };

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  const rows = window.MODEL_HISTORY || [];
  if (rows.length) {
    renderHeadline(rows);
    const step = findStep(rows);
    lineChart('chartSamples', rows, (r) => r.samples, (v) => Math.round(v).toLocaleString('en-GB'), null);
    lineChart('chartRate', rows, (r) => r.anomaly_rate, (v) => pct(v), step);
    renderCaption(rows, step);
  }
  initBand();
  initReveal();
});

function initTheme() {
  const root = document.documentElement;
  const dark = matchMedia('(prefers-color-scheme: dark)');
  document.getElementById('themeToggle').addEventListener('click', () => {
    const next = (root.dataset.theme || (dark.matches ? 'dark' : 'light')) === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
}

/* ---------- Band ---------- */

function renderHeadline(rows) {
  const last = rows[rows.length - 1];
  const wins = rows.filter((r) => r.model === last.model).length;
  document.getElementById('updated').textContent = `Latest report ${fmtDate(last.date, { day: 'numeric', month: 'short', year: 'numeric' })}`;
  const cards = [
    { value: String(rows.length), label: 'daily model reports published since the first run.', context: `since ${fmtDate(rows[0].date, { day: 'numeric', month: 'short', year: 'numeric' })}` },
    { value: last.samples.toLocaleString('en-GB'), label: 'rows of telemetry in the latest training set.', context: `${last.features} features per row` },
    { value: pct(last.anomaly_rate), label: 'of those rows flagged as anomalous.', context: 'contamination set to 1%' },
    { value: `${wins}/${rows.length}`, label: `days won by ${MODEL_NAMES[last.model] || last.model} over the other model.`, context: last.model },
  ];
  document.getElementById('headline').innerHTML = cards.map((c) => `
    <div class="outcome">
      <dt class="outcome-label">${esc(c.label)}</dt>
      <dd class="outcome-value">${esc(c.value)}</dd>
      <dd class="outcome-context">${esc(c.context)}</dd>
    </div>`).join('');
}

// Largest day-on-day change in anomaly rate, if it is a clear step (> 1 point).
function findStep(rows) {
  let best = null;
  for (let i = 1; i < rows.length; i++) {
    const d = Math.abs(rows[i].anomaly_rate - rows[i - 1].anomaly_rate);
    if (d > 0.01 && (!best || d > best.d)) best = { i, d };
  }
  return best;
}

function renderCaption(rows, step) {
  const cap = document.getElementById('historyCap');
  if (!step) { cap.textContent = 'Hover or focus a chart to read each day.'; return; }
  const a = rows[step.i - 1], b = rows[step.i];
  cap.textContent = `The anomaly share dropped in one step on ${fmtDate(b.date)}, from ${pct(a.anomaly_rate)} to ${pct(b.anomaly_rate)}, and has held near ${pct(rows[rows.length - 1].anomaly_rate)} since. A step that sudden most likely reflects a settings change, not the data. Hover or focus a chart to read each day.`;
}

/* ---------- Line chart (one series, own y-scale, crosshair tooltip) ---------- */

function lineChart(id, rows, get, fmt, step) {
  const W = 520, H = 240, m = { t: 16, r: 12, b: 28, l: 56 };
  const vals = rows.map(get);
  const yMax = niceMax(Math.max(...vals));
  const X = (i) => m.l + (i / (rows.length - 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
  const ticks = [0, yMax / 2, yMax];
  const pts = vals.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
  const xTicks = [0, Math.floor((rows.length - 1) / 2), rows.length - 1];
  // Approximate path length for the draw-in animation.
  let len = 0;
  for (let i = 1; i < vals.length; i++) len += Math.hypot(X(i) - X(i - 1), Y(vals[i]) - Y(vals[i - 1]));

  const host = document.getElementById(id);
  host.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(host.previousElementSibling.previousElementSibling.textContent)}: ${fmt(vals[0])} on ${fmtDate(rows[0].date)} to ${fmt(vals[vals.length - 1])} on ${fmtDate(rows[rows.length - 1].date)}">
      <g class="grid">${ticks.map((t) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`).join('')}</g>
      ${ticks.map((t) => `<text class="axis-label" x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join('')}
      ${xTicks.map((i) => `<text class="axis-label" x="${X(i)}" y="${H - 8}" text-anchor="${i === 0 ? 'start' : i === rows.length - 1 ? 'end' : 'middle'}">${fmtDate(rows[i].date)}</text>`).join('')}
      ${step ? `<line class="note-line" x1="${X(step.i)}" x2="${X(step.i)}" y1="${m.t}" y2="${H - m.b}"/><text class="note" x="${X(step.i) + 6}" y="${m.t + 12}">step change</text>` : ''}
      <polygon class="area" points="${m.l},${H - m.b} ${pts.join(' ')} ${X(rows.length - 1)},${H - m.b}"/>
      <polyline class="line" style="--len:${Math.ceil(len)}" points="${pts.join(' ')}"/>
      <line class="cross" y1="${m.t}" y2="${H - m.b}" visibility="hidden"/>
      <circle class="hover-dot" r="5" visibility="hidden"/>
      <rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" tabindex="0"/>
    </svg>`;

  const svg = host.querySelector('svg');
  const cross = svg.querySelector('.cross'), dot = svg.querySelector('.hover-dot'), hit = svg.querySelector('.hit');
  const tip = document.getElementById('tip');
  let cur = rows.length - 1;
  const show = (i) => {
    cur = Math.max(0, Math.min(rows.length - 1, i));
    const x = X(cur), y = Y(vals[cur]);
    cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.setAttribute('visibility', 'visible');
    dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('visibility', 'visible');
    tip.innerHTML = `<strong>${fmt(vals[cur])}</strong><span>${fmtDate(rows[cur].date, { day: 'numeric', month: 'short', year: 'numeric' })} · ${esc(MODEL_NAMES[rows[cur].model] || rows[cur].model)}</span>`;
    tip.hidden = false;
    const r = dot.getBoundingClientRect();
    tip.style.left = `${Math.min(Math.max(8, r.left + r.width / 2 - tip.offsetWidth / 2), innerWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${r.top - tip.offsetHeight - 12}px`;
  };
  const hide = () => {
    tip.hidden = true;
    cross.setAttribute('visibility', 'hidden');
    dot.setAttribute('visibility', 'hidden');
  };
  hit.addEventListener('pointermove', (e) => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const sx = p.matrixTransform(svg.getScreenCTM().inverse()).x;
    show(Math.round(((sx - m.l) / (W - m.l - m.r)) * (rows.length - 1)));
  });
  hit.addEventListener('pointerleave', hide);
  hit.addEventListener('focus', () => show(cur));
  hit.addEventListener('blur', hide);
  hit.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); show(cur - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); show(cur + 1); }
  });
  addEventListener('scroll', hide, { passive: true });
}

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const k of [1, 2, 2.5, 5, 10]) if (v <= k * p) return k * p;
  return 10 * p;
}

/* ---------- Motion ---------- */

function initBand() {
  const band = document.querySelector('.outcomes');
  if (!document.documentElement.classList.contains('motion')) { band.classList.add('in'); return; }
  const open = () => { band.classList.add('in'); removeEventListener('scroll', check); };
  const check = () => { if (scrollY > 80 || band.getBoundingClientRect().top < innerHeight * 0.45) open(); };
  addEventListener('scroll', check, { passive: true });
  check();
}

function initReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!document.documentElement.classList.contains('motion')) { els.forEach((el) => el.classList.add('in')); return; }
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }), { rootMargin: '0px 0px -8% 0px' });
  els.forEach((el) => io.observe(el));
}
