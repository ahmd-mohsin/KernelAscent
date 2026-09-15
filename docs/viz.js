// KernelAscent live mechanistic visuals — rendered as inline SVG from data/mech_analysis.json.
// Three figures: (1) 7-lane internal-failure causality DAG, (2) scale-vs-RSI-gain bubble scatter with
// marginal histograms, (3) a five-task agent-flow diagram. All vanilla JS, no deps, serif to match site.

// ---- shared palette (semantic: outcome class) ----
const OC = {
  rsi:    { fill: "#5b9e6f", line: "#5b9e6f", label: "RSI compounds" },       // green
  flat:   { fill: "#7d9dc9", line: "#7d9dc9", label: "crossed wall, flat" },  // blue
  wall:   { fill: "#d98a3a", line: "#d98a3a", label: "stuck at correctness wall" }, // orange
};
function outClass(m) { return m.rsi ? "rsi" : (m.wall_crossed ? "flat" : "wall"); }
const clamp01 = v => Math.max(0, Math.min(1, v));
const lerp = (a, b, t) => a + (b - a) * t;

// normalize a numeric field across models to [0,1]; guards against constant fields
function normer(models, f) {
  const vs = models.map(m => +m[f] || 0);
  const lo = Math.min(...vs), hi = Math.max(...vs);
  return v => (hi > lo ? ((+v || 0) - lo) / (hi - lo) : 0.5);
}

// ============================================================ 1. CAUSALITY DAG
function renderCausalityDAG(models) {
  const el = document.getElementById("viz-dag"); if (!el) return;
  const LANES = [
    { key: "size_b",          name: "SCALE",     log: true },
    { key: "wall_crossed",    name: "WALL",      bin: true },
    { key: "max_C_train",     name: "GRADIENT" },
    { key: "drift_total",     name: "DRIFT" },
    { key: "mean_retention",  name: "RETENTION" },
    { key: "mean_diversity",  name: "DIVERSITY" },
    { key: "outcome",         name: "OUTCOME",   out: true },
  ];
  const W = 1180, H = 640, padL = 20, padR = 20, top = 78, bot = 92;
  const laneX = i => padL + i / (LANES.length - 1) * (W - padL - padR);
  const bandTop = top, bandBot = H - bot, bandH = bandBot - bandTop;

  // per-lane value → y (0 at bottom band, 1 at top band). Add slight jitter so equal values fan out.
  const normers = {};
  LANES.forEach(L => { if (!L.bin && !L.out) normers[L.key] = normer(models, L.key === "size_b" ? "size_b" : L.key); });
  const logn = (() => { const vs = models.map(m => Math.log10(Math.max(0.3, +m.size_b || 0.5)));
    const lo = Math.min(...vs), hi = Math.max(...vs); return v => (hi > lo ? (Math.log10(Math.max(0.3, v)) - lo) / (hi - lo) : 0.5); })();
  const driftN = normer(models, "drift_total");

  function laneVal(m, L, idx) {
    if (L.log) return logn(+m.size_b || 0.5);
    if (L.bin) return m.wall_crossed ? 0.72 : 0.12;
    if (L.out) return m.rsi ? 0.85 : (m.wall_crossed ? 0.5 : 0.12);
    return clamp01(normers[L.key](m[L.key]));
  }
  const yOf = (v, seed) => {
    const jit = ((Math.sin(seed * 12.9898) * 43758.5453) % 1) * 0.06 - 0.03; // deterministic tiny jitter
    return bandBot - clamp01(v + jit) * bandH;
  };
  const rOf = m => 3 + Math.sqrt(clamp01(driftN(m.drift_total))) * 15; // area ∝ drift

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Internal-failure causality DAG" font-family="Georgia, serif">`;
  // lane guide lines + labels
  LANES.forEach((L, i) => {
    const x = laneX(i);
    svg += `<line x1="${x}" y1="${bandTop - 8}" x2="${x}" y2="${bandBot + 8}" stroke="#ededed" stroke-width="1"/>`;
    svg += `<text x="${x}" y="${bandTop - 24}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#6a6a6a" letter-spacing="1.4">${L.name}</text>`;
  });
  svg += `<text x="${W/2}" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#151515">Internal-failure causality — each model's path SCALE → WALL → GRADIENT → DRIFT → RETENTION → DIVERSITY → OUTCOME</text>`;
  svg += `<text x="${W/2}" y="46" text-anchor="middle" font-size="11" fill="#9a9a9a">one translucent ribbon per model (${models.length} models); edge width &amp; marker area ∝ LoRA drift; color = outcome</text>`;

  // sort so small (wall) drawn first, rsi on top
  const order = [...models].map((m, i) => ({ m, i })).sort((a, b) => ("rsi flat wall".indexOf(outClass(b.m)) - "rsi flat wall".indexOf(outClass(a.m))));
  // edges (bezier ribbons)
  order.forEach(({ m, i }) => {
    const c = OC[outClass(m)];
    const w = 0.7 + Math.sqrt(clamp01(driftN(m.drift_total))) * 5.5;
    const op = m.rsi ? 0.42 : (m.wall_crossed ? 0.22 : 0.5);
    let d = "";
    for (let k = 0; k < LANES.length - 1; k++) {
      const x0 = laneX(k), x1 = laneX(k + 1);
      const y0 = yOf(laneVal(m, LANES[k], i + k), i + k), y1 = yOf(laneVal(m, LANES[k + 1], i + k + 1), i + k + 1);
      const mx = (x0 + x1) / 2;
      d += `M${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1} `;
    }
    svg += `<path d="${d}" fill="none" stroke="${c.line}" stroke-width="${w.toFixed(2)}" stroke-opacity="${op}" stroke-linecap="round"/>`;
  });
  // nodes
  order.forEach(({ m, i }) => {
    const c = OC[outClass(m)];
    LANES.forEach((L, k) => {
      const x = laneX(k), y = yOf(laneVal(m, L, i + k), i + k), r = rOf(m);
      svg += `<circle cx="${x}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${c.fill}" fill-opacity="0.55" stroke="${c.line}" stroke-width="0.8"/>`;
    });
  });
  // legend
  const lx = padL + 6, ly = bandBot + 34;
  let leg = "";
  Object.values(OC).forEach((c, j) => {
    leg += `<circle cx="${lx + j * 200 + 6}" cy="${ly - 4}" r="6" fill="${c.fill}" fill-opacity="0.6" stroke="${c.line}"/><text x="${lx + j * 200 + 18}" y="${ly}" font-size="11.5" fill="#4a4a4a">${c.label}</text>`;
  });
  leg += `<circle cx="${lx + 620}" cy="${ly - 4}" r="3" fill="#9a9a9a"/><text x="${lx + 628}" y="${ly}" font-size="11.5" fill="#9a9a9a">small drift</text>`;
  leg += `<circle cx="${lx + 720}" cy="${ly - 4}" r="10" fill="#9a9a9a" fill-opacity="0.4"/><text x="${lx + 736}" y="${ly}" font-size="11.5" fill="#9a9a9a">large drift</text>`;
  svg += leg + `</svg>`;
  el.innerHTML = svg;
}

// ============================================================ 2. SCALE vs RSI-GAIN scatter + marginals
function renderScaleGain(models) {
  const el = document.getElementById("viz-scatter"); if (!el) return;
  const pts = models.map(m => ({
    x: Math.log10(Math.max(0.3, +m.size_b || 0.5)),
    y: +m.C_held_gain || 0,
    drift: +m.drift_total || 0,
    cls: outClass(m), m,
  }));
  const W = 760, H = 560, mL = 66, mR = 150, mT = 150, mB = 60;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const xLo = -0.55, xHi = 1.3, yLo = Math.min(-0.15, ...pts.map(p => p.y)) - 0.02, yHi = Math.max(0.5, ...pts.map(p => p.y)) + 0.03;
  const X = v => mL + (v - xLo) / (xHi - xLo) * plotW;
  const Y = v => mT + plotH - (v - yLo) / (yHi - yLo) * plotH;
  const dMax = Math.max(...pts.map(p => p.drift), 0.01);
  const R = d => 3 + Math.sqrt(d / dMax) * 22;

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scale vs RSI gain" font-family="Georgia, serif">`;
  svg += `<text x="${mL + plotW/2}" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#151515">Scale vs. RSI gain</text>`;
  svg += `<text x="${mL + plotW/2}" y="44" text-anchor="middle" font-size="11" fill="#9a9a9a">bubble area ∝ LoRA drift · shaded = sub-2B correctness wall · marginals show where models pile up</text>`;
  // correctness-wall shaded region (size < 2B => log10 < 0.301)
  const xw = X(Math.log10(2));
  svg += `<rect x="${mL}" y="${mT}" width="${xw - mL}" height="${plotH}" fill="#f0f0f0"/>`;
  svg += `<text x="${(mL + xw)/2}" y="${mT + 16}" text-anchor="middle" font-size="10.5" fill="#b0b0b0" font-style="italic">correctness wall</text>`;
  // axes + zero line
  svg += `<line x1="${mL}" y1="${mT+plotH}" x2="${mL+plotW}" y2="${mT+plotH}" stroke="#151515" stroke-width="1"/>`;
  svg += `<line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT+plotH}" stroke="#151515" stroke-width="1"/>`;
  svg += `<line x1="${mL}" y1="${Y(0)}" x2="${mL+plotW}" y2="${Y(0)}" stroke="#c9c9c9" stroke-width="1" stroke-dasharray="3 3"/>`;
  // ticks
  [-0.5,0,0.5,1.0].forEach(t => { const x=X(t); svg += `<line x1="${x}" y1="${mT+plotH}" x2="${x}" y2="${mT+plotH+5}" stroke="#151515"/><text x="${x}" y="${mT+plotH+20}" text-anchor="middle" font-size="10" fill="#6a6a6a">${t.toFixed(2)}</text>`; });
  [0,0.2,0.4].forEach(t => { const y=Y(t); svg += `<line x1="${mL-5}" y1="${y}" x2="${mL}" y2="${y}" stroke="#151515"/><text x="${mL-9}" y="${y+3}" text-anchor="end" font-size="10" fill="#6a6a6a">${t.toFixed(1)}</text>`; });
  svg += `<text x="${mL+plotW/2}" y="${mT+plotH+42}" text-anchor="middle" font-size="11.5" fill="#4a4a4a">log10 params (B)</text>`;
  svg += `<text transform="translate(${mL-42},${mT+plotH/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="#4a4a4a">held-out C gain over rounds</text>`;

  // top marginal histogram of x
  const nb = 12;
  const xbins = new Array(nb).fill(0); pts.forEach(p => { const b = Math.min(nb-1, Math.max(0, Math.floor((p.x-xLo)/(xHi-xLo)*nb))); xbins[b]++; });
  const xbMax = Math.max(...xbins, 1);
  xbins.forEach((c, b) => { if (!c) return; const x0 = X(xLo + b/nb*(xHi-xLo)), x1 = X(xLo + (b+1)/nb*(xHi-xLo)); const h = c/xbMax*54;
    svg += `<rect x="${x0+1}" y="${mT-14-h}" width="${x1-x0-2}" height="${h}" fill="#7d9dc9" fill-opacity="0.55"/>`; });
  // right marginal histogram of y
  const ybins = new Array(nb).fill(0); pts.forEach(p => { const b = Math.min(nb-1, Math.max(0, Math.floor((p.y-yLo)/(yHi-yLo)*nb))); ybins[b]++; });
  const ybMax = Math.max(...ybins, 1);
  ybins.forEach((c, b) => { if (!c) return; const y1 = Y(yLo + b/nb*(yHi-yLo)), y0 = Y(yLo + (b+1)/nb*(yHi-yLo)); const w = c/ybMax*70;
    svg += `<rect x="${mL+plotW+14}" y="${y0+1}" width="${w}" height="${y1-y0-2}" fill="#5b9e6f" fill-opacity="0.5"/>`; });

  // bubbles (wall first, rsi on top)
  const ord = [...pts].sort((a,b) => "rsi flat wall".indexOf(b.cls) - "rsi flat wall".indexOf(a.cls));
  ord.forEach(p => { const c = OC[p.cls];
    svg += `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${R(p.drift).toFixed(1)}" fill="${c.fill}" fill-opacity="0.5" stroke="${c.line}" stroke-width="1"><title>${p.m.model} · ${(+p.m.size_b).toFixed(1)}B · gain ${p.y.toFixed(3)} · drift ${p.drift.toFixed(3)}</title></circle>`; });
  // legend
  Object.values(OC).forEach((c, j) => { const yy = mT + 6 + j*20; svg += `<circle cx="${mL+plotW+22}" cy="${yy}" r="6" fill="${c.fill}" fill-opacity="0.55" stroke="${c.line}"/><text x="${mL+plotW+34}" y="${yy+4}" font-size="10.5" fill="#4a4a4a">${c.label}</text>`; });
  svg += `</svg>`;
  el.innerHTML = svg;
}

// ============================================================ 3. FIVE-TASK AGENT FLOW
function renderTaskFlow() {
  const el = document.getElementById("viz-flow"); if (!el) return;
  const W = 1180, H = 300, n = 5, gap = 16, cw = (W - gap * (n - 1)) / n;
  const TASKS = [
    { t: "TASK 1", h: "Capability", sub: "one shot · open + closed", glyph: "one", note: "model → kernel → grade. A snapshot of raw skill; no learning." },
    { t: "TASK 2", h: "Weight RSI", sub: "open weight", glyph: "loop", note: "writes kernels → trains on the correct ones → re-measured. The weights change." },
    { t: "TASK 3", h: "Procedure RSI", sub: "open + closed", glyph: "edit", note: "edits its own executable research procedure; weights frozen. Skill via better method." },
    { t: "TASK 4", h: "Closed → Open", sub: "closed drives open", glyph: "cross", note: "a closed researcher rewrites the training harness that improves an open trainee." },
    { t: "TASK 5", h: "Self-play", sub: "true RSI · open + closed", glyph: "arena", note: "the model authors strictly-harder tasks and improves on them — difficulty co-evolves." },
  ];
  const cx = i => i * (cw + gap);
  const ink = "#151515", mut = "#9a9a9a", accent = "#8C1515";
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Five-task agent flow" font-family="Georgia, serif">`;
  // progression rail beneath
  svg += `<defs><marker id="fa" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="${accent}"/></marker>
    <marker id="fk" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="${ink}"/></marker></defs>`;
  const railY = H - 30;
  svg += `<line x1="${cx(0)+cw/2}" y1="${railY}" x2="${cx(n-1)+cw/2}" y2="${railY}" stroke="#e7d3d3" stroke-width="3"/>`;
  svg += `<text x="${cx(0)+cw/2}" y="${railY+20}" text-anchor="middle" font-size="10.5" fill="${mut}">snapshot</text>`;
  svg += `<text x="${cx(n-1)+cw/2}" y="${railY+20}" text-anchor="middle" font-size="10.5" fill="${mut}">recursive · self-authored</text>`;
  svg += `<text x="${W/2}" y="${railY+20}" text-anchor="middle" font-size="10.5" fill="${accent}" font-style="italic">increasing recursion depth →</text>`;

  TASKS.forEach((T, i) => {
    const x = cx(i), cX = x + cw / 2, boxTop = 44, boxH = 168;
    svg += `<rect x="${x}" y="${boxTop}" width="${cw}" height="${boxH}" rx="12" fill="#fff" stroke="#e6e6e6" stroke-width="1"/>`;
    svg += `<circle cx="${cX}" cy="${railY}" r="5" fill="${accent}"/>`;
    svg += `<line x1="${cX}" y1="${boxTop+boxH}" x2="${cX}" y2="${railY-5}" stroke="#e7d3d3" stroke-width="1.5"/>`;
    svg += `<text x="${cX}" y="${boxTop+20}" text-anchor="middle" font-size="10.5" font-weight="700" fill="${mut}" letter-spacing="1.3">${T.t}</text>`;
    svg += `<text x="${cX}" y="${boxTop+40}" text-anchor="middle" font-size="15" font-weight="700" fill="${ink}">${T.h}</text>`;
    svg += `<text x="${cX}" y="${boxTop+56}" text-anchor="middle" font-size="9.5" fill="${mut}">${T.sub}</text>`;
    // per-task glyph (agent behavior), drawn in a 96px-tall zone starting at gy
    const gy = boxTop + 70, gX = cX;
    const box = (bx, by, bw, bh, lab, f) => `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="6" fill="${f||'#fafafa'}" stroke="${ink}" stroke-width="1"/><text x="${bx+bw/2}" y="${by+bh/2+3}" text-anchor="middle" font-size="9" font-weight="700" fill="${ink}">${lab}</text>`;
    if (T.glyph === "one") {
      svg += box(gX-58, gy+8, 50, 26, "model");
      svg += box(gX+8, gy+8, 50, 26, "kernel");
      svg += `<line x1="${gX-8}" y1="${gy+21}" x2="${gX+8}" y2="${gy+21}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += box(gX-26, gy+50, 52, 26, "score C", "#f3eaea");
      svg += `<line x1="${gX+33}" y1="${gy+34}" x2="${gX}" y2="${gy+50}" stroke="${ink}" marker-end="url(#fk)"/>`;
    } else if (T.glyph === "loop") {
      svg += box(gX-58, gy+6, 52, 24, "model");
      svg += box(gX+10, gy+6, 50, 24, "kernels");
      svg += box(gX+10, gy+50, 50, 24, "LoRA");
      svg += box(gX-58, gy+50, 52, 24, "stronger", "#f3eaea");
      svg += `<line x1="${gX-6}" y1="${gy+18}" x2="${gX+10}" y2="${gy+18}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += `<line x1="${gX+35}" y1="${gy+30}" x2="${gX+35}" y2="${gy+50}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += `<line x1="${gX+10}" y1="${gy+62}" x2="${gX-6}" y2="${gy+62}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += `<path d="M${gX-32} ${gy+50} C ${gX-32} ${gy+40}, ${gX-32} ${gy+34}, ${gX-32} ${gy+30}" stroke="${accent}" fill="none" marker-end="url(#fa)"/>`;
    } else if (T.glyph === "edit") {
      svg += box(gX-30, gy+6, 60, 24, "model");
      svg += `<rect x="${gX-40}" y="${gy+44}" width="80" height="30" rx="6" fill="#fafafa" stroke="${ink}" stroke-dasharray="4 2"/><text x="${gX}" y="${gy+62}" text-anchor="middle" font-size="9" font-weight="700" fill="${ink}">procedure</text>`;
      svg += `<path d="M${gX} ${gy+30} C ${gX} ${gy+37}, ${gX} ${gy+40}, ${gX} ${gy+44}" stroke="${accent}" fill="none" marker-end="url(#fa)"/>`;
      svg += `<text x="${gX+48}" y="${gy+34}" font-size="8.5" fill="${mut}">weights</text><text x="${gX+48}" y="${gy+44}" font-size="8.5" fill="${mut}">locked 🔒</text>`;
    } else if (T.glyph === "cross") {
      svg += box(gX-58, gy+8, 54, 26, "closed", "#f3eaea");
      svg += box(gX+6, gy+8, 54, 26, "harness");
      svg += box(gX-28, gy+52, 56, 26, "open");
      svg += `<line x1="${gX-4}" y1="${gy+21}" x2="${gX+6}" y2="${gy+21}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += `<line x1="${gX+33}" y1="${gy+34}" x2="${gX}" y2="${gy+52}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += `<text x="${gX-31}" y="${gy+46}" text-anchor="middle" font-size="8" fill="${mut}">rewrites</text>`;
    } else if (T.glyph === "arena") {
      svg += box(gX-56, gy+6, 50, 24, "author");
      svg += box(gX+8, gy+6, 50, 24, "solver");
      svg += `<line x1="${gX-6}" y1="${gy+13}" x2="${gX+8}" y2="${gy+13}" stroke="${ink}" marker-end="url(#fk)"/>`;
      svg += `<line x1="${gX+8}" y1="${gy+23}" x2="${gX-6}" y2="${gy+23}" stroke="${accent}" marker-end="url(#fa)"/>`;
      ["S","F","L"].forEach((a,j) => svg += `<circle cx="${gX-30+j*30}" cy="${gy+58}" r="10" fill="${j===2?accent:'#fafafa'}" stroke="${ink}"/><text x="${gX-30+j*30}" y="${gy+61}" text-anchor="middle" font-size="9" font-weight="700" fill="${j===2?'#fff':ink}">${a}</text>`);
      svg += `<text x="${gX}" y="${gy+80}" text-anchor="middle" font-size="8" fill="${mut}">L−F = co-evolution</text>`;
    }
  });
  svg += `</svg>`;
  el.innerHTML = svg;
  const notes = document.getElementById("viz-flow-notes");
  if (notes) notes.innerHTML = TASKS.map(T => `<div class="flownote"><b>${T.h}.</b> ${T.note}</div>`).join("");
}

// ---- load + render ----
fetch("data/mech_analysis.json").then(r => r.json()).then(d => {
  const ms = (d.models || []).filter(m => typeof m.size_b === "number");
  const up = document.getElementById("viz-updated");
  if (up) up.textContent = "updated " + (d.updated || "") + " · " + ms.length + " WHY-RSI probes, 0.5–15B";
  renderCausalityDAG(ms);
  renderScaleGain(ms);
}).catch(e => {});
renderTaskFlow();
