/* Fashion-MNIST embedding explorer (D3 v7)
   data/<model>/metadata.json and data/<model>/<method>_stage<N>.json  ->  [[x, y, label], ...]
   The page draws the coordinates exactly as exported (no re-anchoring here). Axis limits come from
   metadata.axis_limit[method]; if that is missing they are computed from the loaded points.
   Must be served over http (e.g. `python3 -m http.server`), not opened via file://. */

const METHODS = [
  { id: "tsne", name: "t-SNE", lim: null, ticks: [], sel: [] },
  { id: "umap", name: "UMAP",  lim: null, ticks: [], sel: [3] }, // default checked class: Dress
];
const TAB10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const GRAY = "#d3d3d3";

// panel geometry (SVG units); the plot area is a square S x S
const W = 360, H = 386, M = { l: 62, t: 46 }, S = 290;

const app = d3.select("#app");
const status = d3.select("#status");
const cache = new Map();
let classes = [], token = 0, nStages = 0;
let pair = null;          // [actual, predicted] picked in the confusion matrix, or null
let showErrors = false;   // "Show misclassified" button state
let errs = [];            // misclassified test images: [index, true, pred, confidence, base64 28x28 pixels]

/* ---------- hover tooltip for misclassified points ---------- */
const tip = d3.select("body").append("div").attr("class", "tip");
const tipCv = tip.append("canvas").attr("width", 28).attr("height", 28).node();
const tipTxt = tip.append("div");
const showErr = (e, m) => pair ? e[1] === pair[0] && e[2] === pair[1] : (!m.sel.size || m.sel.has(e[1]));   // e[1] actual, e[2] predicted
function showTip(e, ev) {
  const px = Uint8Array.from(atob(e[4]), c => c.charCodeAt(0)), g = tipCv.getContext("2d"), img = g.createImageData(28, 28);
  for (let i = 0; i < 784; i++) { img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = px[i]; img.data[4 * i + 3] = 255; }
  g.putImageData(img, 0, 0);
  tipTxt.html(`<b>Actual:</b> ${classes[e[1]]}<br><b>Predicted:</b> ${classes[e[2]]}<br><b>Confidence:</b> ${(e[3] * 100).toFixed(1)}%<br><span>test image #${e[0]}</span>`);
  tip.style("display", "flex").style("left", `${ev.pageX + 14}px`).style("top", `${ev.pageY + 14}px`);
}   // nStages comes from metadata.stages.length

const fetchJSON = url => {
  if (!cache.has(url)) cache.set(url, d3.json(url));
  return cache.get(url);
};

/* ---------- build static layout once ---------- */
function buildSection(m) {
  m.sel = new Set(m.sel);
  const sec = app.append("div").attr("class", "section");

  const controls = sec.append("div").attr("class", "controls");
  m.cnt = []; m.boxes = [];
  classes.forEach((name, i) => {
    const lab = controls.append("label");
    m.boxes[i] = lab.append("input").attr("type", "checkbox").property("checked", m.sel.has(i))
      .on("change", e => { e.target.checked ? m.sel.add(i) : m.sel.delete(i);
        if (pair) { pair = null; drawMatrix(); METHODS.forEach(x => redraw(x)); } else redraw(m); });
    lab.append("span").text(name);
    m.cnt[i] = lab.append("span").attr("class", "n");
  });

  m.titleEl = sec.append("h3").attr("class", "plot-title");
  const row = sec.append("div").attr("class", "row").style("--n", nStages);
  m.panels = d3.range(nStages).map(i => buildPanel(row, m, i));
  redraw(m);
}

function buildPanel(row, m, i) {
  const div = row.append("div").attr("class", "panel");
  const svg = div.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const axes = svg.append("g");
  // ticks (outward) + labels; y labels only on the first panel (shared y axis). Re-run when the limits change.
  const drawAxes = () => {
    const sx = d3.scaleLinear([-m.lim, m.lim], [M.l, M.l + S]);
    const sy = d3.scaleLinear([-m.lim, m.lim], [M.t + S, M.t]);
    axes.selectAll("*").remove();
    m.ticks.forEach(t => {
      axes.append("line").attr("x1", sx(t)).attr("x2", sx(t)).attr("y1", M.t + S).attr("y2", M.t + S + 4).attr("stroke", "#000");
      axes.append("text").attr("x", sx(t)).attr("y", M.t + S + 17).attr("text-anchor", "middle").attr("font-size", 10.5).text(t);
      axes.append("line").attr("x1", M.l - 4).attr("x2", M.l).attr("y1", sy(t)).attr("y2", sy(t)).attr("stroke", "#000");
      if (i === 0) axes.append("text").attr("x", M.l - 8).attr("y", sy(t)).attr("dy", "0.35em").attr("text-anchor", "end").attr("font-size", 10.5).text(t);
    });
  };
  svg.append("text").attr("x", M.l + S / 2).attr("y", H - 10).attr("text-anchor", "middle").attr("font-size", 11)
    .text(`${m.name} Component 1`);
  svg.append("text").attr("transform", `translate(16,${M.t + S / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-size", 11)
    .text(`${m.name} Component 2`);

  const t1 = svg.append("text").attr("x", M.l + S / 2).attr("y", 20).attr("text-anchor", "middle").attr("font-size", 13);
  const t2 = svg.append("text").attr("x", M.l + S / 2).attr("y", 36).attr("text-anchor", "middle").attr("font-size", 13);

  const cv = div.append("canvas")
    .style("left", `${M.l / W * 100}%`).style("top", `${M.t / H * 100}%`)
    .style("width", `${S / W * 100}%`).style("height", `${S / H * 100}%`);
  cv.on("mousemove", ev => {
    const hit = cv.node()._hit; if (!hit) return;
    const r = cv.node().getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 100;                              // 10 px pick radius (squared)
    for (const h of hit) { const d = (h.x - x) ** 2 + (h.y - y) ** 2; if (d < bd) { bd = d; best = h.e; } }
    best ? showTip(best, ev) : tip.style("display", "none");
  }).on("mouseleave", () => tip.style("display", "none"));
  return { cv: cv.node(), t1, t2, pts: null, drawAxes };
}

/* ---------- drawing ---------- */
/* ---------- confusion matrix (rows = actual, columns = predicted); click an off-diagonal cell to filter ---------- */
function drawMatrix() {
  const n = classes.length, pts = METHODS[0].panels[0].pts; if (!pts) return;
  const tot = Array(n).fill(0), C = Array.from({ length: n }, () => Array(n).fill(0));
  pts.forEach(p => tot[p[2]]++); errs.forEach(e => C[e[1]][e[2]]++);
  for (let t = 0; t < n; t++) C[t][t] = tot[t] - d3.sum(C[t]);               // diagonal = correct predictions
  let maxShare = 0;
  for (let t = 0; t < n; t++) for (let p = 0; p < n; p++) if (t !== p) maxShare = Math.max(maxShare, C[t][p] / tot[t]);

  const cell = 40, ml = 100, mt = 110, Wm = ml + n * cell + 70, Hm = mt + n * cell + 8;
  const svg = d3.select("#matrixSvg").attr("viewBox", `0 0 ${Wm} ${Hm}`);
  svg.selectAll("*").remove();
  svg.append("text").attr("x", ml + n * cell / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 12).text("Predicted class");
  svg.append("text").attr("transform", `translate(12,${mt + n * cell / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-size", 12).text("Actual class");
  classes.forEach((name, i) => {
    svg.append("text").attr("x", ml - 6).attr("y", mt + i * cell + cell / 2).attr("dy", "0.35em").attr("text-anchor", "end").attr("font-size", 11).text(name);
    svg.append("text").attr("transform", `translate(${ml + i * cell + cell / 2 + 4},${mt - 6}) rotate(-45)`).attr("font-size", 11).text(name);
  });
  for (let t = 0; t < n; t++) for (let p = 0; p < n; p++) {
    const v = C[t][p], diag = t === p, share = v / tot[t];
    const k = diag || !v ? 0 : 0.1 + 0.9 * Math.pow(share / maxShare, 0.7);
    const g = svg.append("g").attr("transform", `translate(${ml + p * cell},${mt + t * cell})`);
    const sel = pair && pair[0] === t && pair[1] === p;
    g.append("rect").attr("width", cell - 1).attr("height", cell - 1)
      .attr("fill", diag ? "#eceff3" : d3.interpolateReds(k)).attr("stroke", sel ? "#000" : "none").attr("stroke-width", 2);
    if (v) g.append("text").attr("x", (cell - 1) / 2).attr("y", (cell - 1) / 2).attr("dy", "0.35em").attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", diag ? "#666" : k > 0.55 ? "#fff" : "#000").text(v);
    g.append("title").text(diag ? `${classes[t]}: ${v} of ${tot[t]} correct`
      : `Actual ${classes[t]} → predicted ${classes[p]}: ${v} images (${(share * 100).toFixed(1)}% of ${classes[t]})`);
    if (!diag && v) g.style("cursor", "pointer").on("click", () => pickPair(t, p));
  }
  d3.select("#pairInfo").text(pair
    ? `Showing ${C[pair[0]][pair[1]]} errors: actual ${classes[pair[0]]} → predicted ${classes[pair[1]]}. Click the cell again to clear.`
    : "Click an off-diagonal cell to show only those errors on the plots above.");
}

function pickPair(t, p) {
  pair = pair && pair[0] === t && pair[1] === p ? null : [t, p];
  if (pair) {                                              // turn crosses on and highlight the two classes
    showErrors = true; d3.select("#showErr").classed("on", true).attr("aria-pressed", true);
    METHODS.forEach(m => { m.sel = new Set([t, p]); m.boxes.forEach((b, i) => b.property("checked", m.sel.has(i))); });
  }
  tip.style("display", "none");
  METHODS.forEach(m => redraw(m)); drawMatrix();
}

function updateLabels(m) {                 // class label count: total images, or misclassified ones when the class is selected
  const pts = m.panels[0].pts; if (!pts) return;
  const tot = Array(classes.length).fill(0), bad = Array(classes.length).fill(0);
  pts.forEach(p => tot[p[2]]++); errs.forEach(e => bad[e[1]]++);
  m.cnt.forEach((s, i) => s.text(showErrors && m.sel.has(i) ? `(${bad[i]} misclassified)` : `(${tot[i]})`));
}

function redraw(m) {
  updateLabels(m);
  const names = [...m.sel].sort((a, b) => a - b).map(i => classes[i]);
  m.titleEl.text(`${m.name}: Highlighting ${names.length ? names.join(", ") : "All Classes"}`);
  m.panels.forEach(p => p.pts && drawPoints(p.cv, p.pts, m));
}

function drawPoints(cv, pts, m) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, w = r.width;
  if (!w) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr);
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const k = w / S;
  const sx = d3.scaleLinear([-m.lim, m.lim], [0, w]);
  const sy = d3.scaleLinear([-m.lim, m.lim], [w, 0]);

  g.fillStyle = "#fff"; g.fillRect(0, 0, w, w);
  g.strokeStyle = "#999"; g.lineWidth = 0.8;               // crosshair through the origin
  g.beginPath(); g.moveTo(w / 2, 0); g.lineTo(w / 2, w); g.moveTo(0, w / 2); g.lineTo(w, w / 2); g.stroke();

  const rad = 1.7 * k;
  const dot = (p, c) => { g.fillStyle = c; g.beginPath(); g.arc(sx(p[0]), sy(p[1]), rad, 0, 6.2832); g.fill(); };
  g.globalAlpha = 0.85;
  if (m.sel.size) {
    for (const p of pts) if (!m.sel.has(p[2])) dot(p, GRAY);   // dim others first
    for (const p of pts) if (m.sel.has(p[2])) dot(p, TAB10[p[2]]);
  } else {
    for (const p of pts) dot(p, TAB10[p[2]]);
  }
  g.globalAlpha = 1;

  // black crosses on misclassified images (white halo keeps them visible on dense clusters)
  const hit = [], b = 3 * k;
  for (const e of showErrors ? errs : []) {
    if (!showErr(e, m) || !pts[e[0]]) continue;
    const x = sx(pts[e[0]][0]), y = sy(pts[e[0]][1]);
    hit.push({ x, y, e });
    g.beginPath(); g.moveTo(x - b, y - b); g.lineTo(x + b, y + b); g.moveTo(x - b, y + b); g.lineTo(x + b, y - b);
    g.strokeStyle = "#fff"; g.lineWidth = 3.2; g.stroke();
    g.strokeStyle = "#000"; g.lineWidth = 1.3; g.stroke();
  }
  cv._hit = hit;

  const c = w / 2, ro = 7 * k, ri = 3 * k;                  // black star marker at origin (anchor class centroid)
  g.beginPath();
  for (let i = 0; i < 10; i++) { const r = i % 2 ? ri : ro, t = -Math.PI / 2 + i * Math.PI / 5; g.lineTo(c + r * Math.cos(t), c + r * Math.sin(t)); }
  g.closePath(); g.fillStyle = "#000"; g.fill(); g.strokeStyle = "#fff"; g.lineWidth = 1; g.stroke();

  g.lineWidth = 1.2; g.strokeRect(0.6, 0.6, w - 1.2, w - 1.2); // axes frame
}


/* ---------- model loading ---------- */
async function setModel(model) {
  const my = ++token;
  status.attr("class", "").text("Loading…");
  try {
    pair = null;
    const meta = await fetchJSON(`data/${model}/metadata.json`);
    if (my !== token) return;
    if (nStages !== meta.stages.length) {                       // (re)build the layout for this many stages
      nStages = meta.stages.length; classes = meta.classes; app.selectAll("*").remove();
      METHODS.forEach(buildSection);
    }

    METHODS.forEach(m => m.panels.forEach((p, i) => {          // "Encoder Block 1 — 49 CLS Features" -> two lines
      const [a, b = ""] = meta.stages[i].title.split(" — ");
      p.t1.text(a); p.t2.text(b); p.pts = null;
    }));

    errs = await fetchJSON(`data/${model}/errors.json`).catch(() => []);   // missing file -> no crosses
    if (my !== token) return;
    await Promise.all(METHODS.map(async m => {
      const all = await Promise.all(m.panels.map((p, i) => fetchJSON(`data/${model}/${m.id}_stage${i}.json`)));
      if (my !== token) return;
      m.panels.forEach((p, i) => p.pts = all[i]);

      // axis limit: exported value if present, else same rule as the notebook (99.5th percentile of |coord| x 1.1, shared by all stages)
      const exported = meta.axis_limit && meta.axis_limit[m.id];
      m.lim = exported || 1.1 * d3.max(all, pts => d3.quantile(Float64Array.from(pts.flatMap(p => [Math.abs(p[0]), Math.abs(p[1])])).sort(), 0.995));
      m.ticks = d3.ticks(-m.lim, m.lim, 5).filter(t => Math.abs(t) < m.lim);
      m.panels.forEach(p => { p.drawAxes(); drawPoints(p.cv, p.pts, m); });
    }));
    METHODS.forEach(updateLabels);
    drawMatrix();
    if (my === token) status.text(errs.length ? `${errs.length} of ${METHODS[0].panels[0].pts.length} test images misclassified`+`` : "No errors.json for this model: no misclassification crosses shown.");
  } catch (err) {
    console.error(err);
    status.attr("class", "error").text("Couldn't load data. Serve this folder over http, e.g. `python3 -m http.server`, then open localhost:8000.");
  }
}

d3.select("#showErr").on("click", function () {
  showErrors = !showErrors;
  if (!showErrors) pair = null;
  d3.select(this).classed("on", showErrors).attr("aria-pressed", showErrors);
  tip.style("display", "none");
  METHODS.forEach(m => m.panels && redraw(m));
  drawMatrix();
});
d3.select("#model").on("change", e => setModel(e.target.value));
window.addEventListener("resize", () => METHODS.forEach(m => m.panels && redraw(m)));
setModel(d3.select("#model").property("value"));