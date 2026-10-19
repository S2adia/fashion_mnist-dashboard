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
let classes = [], token = 0, nStages = 0;   // nStages comes from metadata.stages.length

const fetchJSON = url => {
  if (!cache.has(url)) cache.set(url, d3.json(url));
  return cache.get(url);
};

/* ---------- build static layout once ---------- */
function buildSection(m) {
  m.sel = new Set(m.sel);
  const sec = app.append("div").attr("class", "section");

  const controls = sec.append("div").attr("class", "controls");
  classes.forEach((name, i) => {
    const lab = controls.append("label");
    lab.append("input").attr("type", "checkbox").property("checked", m.sel.has(i))
      .on("change", e => { e.target.checked ? m.sel.add(i) : m.sel.delete(i); redraw(m); });
    lab.append("span").text(name);
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
  return { cv: cv.node(), t1, t2, pts: null, drawAxes };
}

/* ---------- drawing ---------- */
function redraw(m) {
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

  const c = w / 2, a = 5 * k;                               // black X marker at origin
  g.strokeStyle = "#000"; g.lineWidth = 1.6;
  g.beginPath(); g.moveTo(c - a, c - a); g.lineTo(c + a, c + a); g.moveTo(c - a, c + a); g.lineTo(c + a, c - a); g.stroke();

  g.lineWidth = 1.2; g.strokeRect(0.6, 0.6, w - 1.2, w - 1.2); // axes frame
}


/* ---------- model loading ---------- */
async function setModel(model) {
  const my = ++token;
  status.attr("class", "").text("Loading…");
  try {
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
    if (my === token) status.text("");
  } catch (err) {
    console.error(err);
    status.attr("class", "error").text("Couldn't load data. Serve this folder over http, e.g. `python3 -m http.server`, then open localhost:8000.");
  }
}

d3.select("#model").on("change", e => setModel(e.target.value));
window.addEventListener("resize", () => METHODS.forEach(m => m.panels && redraw(m)));
setModel(d3.select("#model").property("value"));
