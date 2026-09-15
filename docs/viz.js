// KernelAscent live mechanistic visuals — animated inline SVG from data/mech_analysis.json.
// (1) full-width five-task agent-flow with flowing edges + pulsing nodes, (2) 7-lane causality DAG
// with ribbons that flow left→right, (3) scale-vs-RSI-gain bubble scatter with marginals + grow-in.
// Vanilla JS, SMIL motion, serif to match site.

const OC = {
  rsi:  { fill: "#5b9e6f", line: "#5b9e6f", label: "RSI compounds" },
  flat: { fill: "#7d9dc9", line: "#7d9dc9", label: "crossed wall, flat" },
  wall: { fill: "#d98a3a", line: "#d98a3a", label: "stuck at correctness wall" },
};
function outClass(m) { return m.rsi ? "rsi" : (m.wall_crossed ? "flat" : "wall"); }
const clamp01 = v => Math.max(0, Math.min(1, v));
function normer(models, f) {
  const vs = models.map(m => +m[f] || 0), lo = Math.min(...vs), hi = Math.max(...vs);
  return v => (hi > lo ? ((+v || 0) - lo) / (hi - lo) : 0.5);
}

// ============================================================ 1. FIVE-TASK AGENT FLOW (full width, animated)
function renderTaskFlow() {
  const el = document.getElementById("viz-flow"); if (!el) return;
  const W = 1560, H = 470, n = 5, pad = 26, gap = 20;
  const cw = (W - pad * 2 - gap * (n - 1)) / n;
  // cool → warm ramp encoding escalating recursion depth
  const RAMP = ["#5f83b4", "#4e9e8f", "#c6a13c", "#cf7e34", "#8C1515"];
  const TASKS = [
    { t: "TASK 1", h: "Capability",     sub: "one shot · open + closed", glyph: "one",
      note: "model → kernel → grade. A snapshot of raw skill; no learning." },
    { t: "TASK 2", h: "Weight RSI",     sub: "open weight",              glyph: "loop",
      note: "writes kernels → LoRA-trains on the correct ones → re-measured. The weights change." },
    { t: "TASK 3", h: "Procedure RSI",  sub: "open + closed",            glyph: "edit",
      note: "rewrites its own executable procedure; weights frozen. Skill via a better method." },
    { t: "TASK 4", h: "Closed → Open",  sub: "closed drives open",       glyph: "cross",
      note: "a closed researcher rewrites the harness that trains an open trainee." },
    { t: "TASK 5", h: "Self-play",      sub: "true RSI · open + closed",  glyph: "arena",
      note: "the model authors strictly-harder tasks and improves on them — difficulty co-evolves." },
  ];
  const cx = i => pad + i * (cw + gap);
  const ink = "#1a1a1a", mut = "#9a9a9a";

  // svg defs: soft shadow, per-card gradient, flowing-edge dash. dur staggered per card.
  let defs = `<defs>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="3" stdDeviation="6" flood-color="#000" flood-opacity="0.10"/></filter>`;
  RAMP.forEach((c, i) => { defs += `<linearGradient id="cg${i}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="${c}14"/></linearGradient>`; });
  // rail flowing gradient
  defs += `<linearGradient id="rail" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="${RAMP[0]}"/><stop offset="0.5" stop-color="${RAMP[2]}"/><stop offset="1" stop-color="${RAMP[4]}"/></linearGradient></defs>`;

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Five-task agent flow" font-family="Georgia, serif" class="flowsvg">${defs}`;
  svg += `<text x="${W/2}" y="30" text-anchor="middle" font-size="17" font-weight="700" fill="${ink}">The five tasks, as agent loops</text>`;
  svg += `<text x="${W/2}" y="50" text-anchor="middle" font-size="12" fill="${mut}">what the agent actually does at each rung — left to right is increasing recursion depth</text>`;

  // animated flowing edge: base hairline + moving accent dash
  const flow = (d, c, dur, w) => `<path d="${d}" fill="none" stroke="#dcdcdc" stroke-width="${w||1.4}"/>`
    + `<path d="${d}" fill="none" stroke="${c}" stroke-width="${(w||1.4)+0.6}" stroke-linecap="round" stroke-dasharray="6 12" opacity="0.9">`
    + `<animate attributeName="stroke-dashoffset" from="18" to="0" dur="${dur}s" repeatCount="indefinite"/></path>`;
  const dot = (x, y, r, c, pulse) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${c}">`
    + (pulse ? `<animate attributeName="r" values="${r};${r*1.35};${r}" dur="1.8s" repeatCount="indefinite"/><animate attributeName="fill-opacity" values="1;0.55;1" dur="1.8s" repeatCount="indefinite"/>` : "") + `</circle>`;

  const railY = H - 46, x0 = cx(0) + cw / 2, x1 = cx(n - 1) + cw / 2;
  svg += `<rect x="${x0}" y="${railY-3}" width="${x1-x0}" height="6" rx="3" fill="url(#rail)" opacity="0.85"/>`;
  // moving pulse travelling along the rail (the "recursion depth" carrier)
  svg += `<circle r="5" fill="#fff" stroke="${RAMP[4]}" stroke-width="2"><animate attributeName="cx" from="${x0}" to="${x1}" dur="6s" repeatCount="indefinite"/><animate attributeName="cy" values="${railY};${railY}" dur="6s" repeatCount="indefinite"/></circle>`;
  svg += `<text x="${x0}" y="${railY+22}" text-anchor="middle" font-size="10.5" fill="${mut}">snapshot</text>`;
  svg += `<text x="${x1}" y="${railY+22}" text-anchor="middle" font-size="10.5" fill="${RAMP[4]}">recursive · self-authored</text>`;
  svg += `<text x="${W/2}" y="${railY+22}" text-anchor="middle" font-size="10.5" fill="${RAMP[2]}" font-style="italic">increasing recursion depth →</text>`;

  TASKS.forEach((T, i) => {
    const x = cx(i), cX = x + cw / 2, bT = 66, bH = 300, acc = RAMP[i], dur = 2.2 + i * 0.25;
    svg += `<g class="flowcard" style="animation-delay:${i*90}ms">`;
    svg += `<rect x="${x}" y="${bT}" width="${cw}" height="${bH}" rx="16" fill="url(#cg${i})" stroke="${acc}33" stroke-width="1.2" filter="url(#soft)"/>`;
    svg += `<rect x="${x}" y="${bT}" width="${cw}" height="6" rx="3" fill="${acc}"/>`;
    // connector down to rail
    svg += `<line x1="${cX}" y1="${bT+bH}" x2="${cX}" y2="${railY-5}" stroke="${acc}55" stroke-width="1.5"/>`;
    svg += dot(cX, railY, 5, acc, i === n - 1);
    svg += `<text x="${cX}" y="${bT+24}" text-anchor="middle" font-size="10.5" font-weight="700" fill="${acc}" letter-spacing="1.6">${T.t}</text>`;
    svg += `<text x="${cX}" y="${bT+46}" text-anchor="middle" font-size="17" font-weight="700" fill="${ink}">${T.h}</text>`;
    svg += `<text x="${cX}" y="${bT+63}" text-anchor="middle" font-size="10" fill="${mut}">${T.sub}</text>`;

    const gy = bT + 82;
    const node = (bx, by, bw, bh, lab, hot) => `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="8" fill="${hot?acc:'#fff'}" stroke="${acc}" stroke-width="1.3"/>`
      + `<text x="${bx+bw/2}" y="${by+bh/2+3.5}" text-anchor="middle" font-size="10" font-weight="700" fill="${hot?'#fff':ink}">${lab}</text>`;

    if (T.glyph === "one") {
      svg += node(cX-70, gy, 62, 30, "model");
      svg += node(cX+8, gy, 62, 30, "kernel");
      svg += flow(`M${cX-8} ${gy+15} L ${cX+8} ${gy+15}`, acc, dur);
      svg += node(cX-31, gy+70, 62, 30, "score C", true);
      svg += flow(`M${cX+39} ${gy+30} C ${cX+39} ${gy+55}, ${cX} ${gy+50}, ${cX} ${gy+70}`, acc, dur);
      svg += `<text x="${cX}" y="${gy+128}" text-anchor="middle" font-size="9" fill="${mut}">one shot · no learning</text>`;
    } else if (T.glyph === "loop") {
      svg += node(cX-72, gy, 60, 28, "model");
      svg += node(cX+12, gy, 60, 28, "kernels");
      svg += node(cX+12, gy+70, 60, 28, "LoRA");
      svg += node(cX-72, gy+70, 60, 28, "stronger", true);
      svg += flow(`M${cX-12} ${gy+14} L ${cX+12} ${gy+14}`, acc, dur);
      svg += flow(`M${cX+42} ${gy+28} L ${cX+42} ${gy+70}`, acc, dur);
      svg += flow(`M${cX+12} ${gy+84} L ${cX-12} ${gy+84}`, acc, dur);
      svg += flow(`M${cX-42} ${gy+70} L ${cX-42} ${gy+28}`, acc, dur);
      svg += `<text x="${cX}" y="${gy+128}" text-anchor="middle" font-size="9" fill="${mut}">weights improve each round</text>`;
    } else if (T.glyph === "edit") {
      svg += node(cX-35, gy, 70, 28, "model");
      svg += `<rect x="${cX-46}" y="${gy+62}" width="92" height="34" rx="8" fill="#fff" stroke="${acc}" stroke-width="1.3" stroke-dasharray="5 3"/><text x="${cX}" y="${gy+82}" text-anchor="middle" font-size="10" font-weight="700" fill="${ink}">procedure</text>`;
      svg += flow(`M${cX} ${gy+28} L ${cX} ${gy+62}`, acc, dur, 1.6);
      svg += `<text x="${cX}" y="${gy+124}" text-anchor="middle" font-size="9" fill="${mut}">weights locked · edits its method</text>`;
    } else if (T.glyph === "cross") {
      svg += node(cX-70, gy, 62, 30, "closed", true);
      svg += node(cX+8, gy, 62, 30, "harness");
      svg += node(cX-31, gy+70, 62, 30, "open");
      svg += flow(`M${cX-8} ${gy+15} L ${cX+8} ${gy+15}`, acc, dur);
      svg += flow(`M${cX+39} ${gy+30} C ${cX+39} ${gy+55}, ${cX} ${gy+50}, ${cX} ${gy+70}`, acc, dur);
      svg += `<text x="${cX}" y="${gy+128}" text-anchor="middle" font-size="9" fill="${mut}">closed rewrites the trainer</text>`;
    } else if (T.glyph === "arena") {
      svg += node(cX-70, gy, 62, 30, "author");
      svg += node(cX+8, gy, 62, 30, "solver");
      svg += flow(`M${cX-8} ${gy+11} L ${cX+8} ${gy+11}`, acc, dur);
      svg += flow(`M${cX+8} ${gy+22} L ${cX-8} ${gy+22}`, acc, dur*0.8);
      ["S","F","L"].forEach((a,j) => svg += `<g>${dot(cX-34+j*34, gy+74, 13, j===2?acc:'#fff', j===2)}<text x="${cX-34+j*34}" y="${gy+78}" text-anchor="middle" font-size="10" font-weight="700" fill="${j===2?'#fff':ink}">${a}</text></g>`);
      svg += `<text x="${cX}" y="${gy+108}" text-anchor="middle" font-size="9" fill="${mut}">L − F = author co-evolution</text>`;
    }
    svg += `</g>`;
  });
  svg += `</svg>`;
  el.innerHTML = svg;
  const notes = document.getElementById("viz-flow-notes");
  if (notes) notes.innerHTML = TASKS.map((T, i) => `<div class="flownote" style="border-top:3px solid ${RAMP[i]}"><b>${T.h}.</b> ${T.note}</div>`).join("");
}

// ============================================================ 2. CAUSALITY DAG (ribbons flow left→right)
function renderCausalityDAG(models) {
  const el = document.getElementById("viz-dag"); if (!el) return;
  const LANES = [
    { name: "SCALE", log: true }, { name: "WALL", bin: true }, { key: "max_C_train", name: "GRADIENT" },
    { key: "drift_total", name: "DRIFT" }, { key: "mean_retention", name: "RETENTION" },
    { key: "mean_diversity", name: "DIVERSITY" }, { name: "OUTCOME", out: true },
  ];
  const W = 1400, H = 660, padL = 24, padR = 24, top = 84, bot = 96;
  const laneX = i => padL + i / (LANES.length - 1) * (W - padL - padR);
  const bandTop = top, bandBot = H - bot, bandH = bandBot - bandTop;
  const norms = {}; LANES.forEach(L => { if (L.key) norms[L.key] = normer(models, L.key); });
  const logn = (() => { const vs = models.map(m => Math.log10(Math.max(0.3, +m.size_b || 0.5)));
    const lo = Math.min(...vs), hi = Math.max(...vs); return v => (hi > lo ? (Math.log10(Math.max(0.3, v)) - lo) / (hi - lo) : 0.5); })();
  const driftN = normer(models, "drift_total");
  const laneVal = (m, L) => L.log ? logn(+m.size_b || 0.5) : L.bin ? (m.wall_crossed ? 0.72 : 0.12)
    : L.out ? (m.rsi ? 0.85 : (m.wall_crossed ? 0.5 : 0.12)) : clamp01(norms[L.key](m[L.key]));
  const yOf = (v, s) => { const j = ((Math.sin(s * 12.9898) * 43758.5453) % 1) * 0.06 - 0.03; return bandBot - clamp01(v + j) * bandH; };
  const rOf = m => 3 + Math.sqrt(clamp01(driftN(m.drift_total))) * 15;

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Internal-failure causality DAG" font-family="Georgia, serif">`;
  LANES.forEach((L, i) => { const x = laneX(i);
    svg += `<line x1="${x}" y1="${bandTop-8}" x2="${x}" y2="${bandBot+8}" stroke="#ededed"/>`;
    svg += `<text x="${x}" y="${bandTop-26}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#6a6a6a" letter-spacing="1.4">${L.name}</text>`; });
  svg += `<text x="${W/2}" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#151515">Internal-failure causality — each model's path through the gates it must clear</text>`;
  svg += `<text x="${W/2}" y="48" text-anchor="middle" font-size="11" fill="#9a9a9a">SCALE → WALL → GRADIENT → DRIFT → RETENTION → DIVERSITY → OUTCOME · one flowing ribbon per model (${models.length}) · width &amp; marker ∝ LoRA drift</text>`;

  const order = [...models].map((m, i) => ({ m, i })).sort((a, b) => ("rsi flat wall".indexOf(outClass(b.m)) - "rsi flat wall".indexOf(outClass(a.m))));
  order.forEach(({ m, i }) => {
    const c = OC[outClass(m)];
    const w = 0.7 + Math.sqrt(clamp01(driftN(m.drift_total))) * 5.5;
    const op = m.rsi ? 0.42 : (m.wall_crossed ? 0.2 : 0.5);
    let d = "";
    for (let k = 0; k < LANES.length - 1; k++) {
      const x0 = laneX(k), x1 = laneX(k + 1), y0 = yOf(laneVal(m, LANES[k]), i + k), y1 = yOf(laneVal(m, LANES[k + 1]), i + k + 1), mx = (x0 + x1) / 2;
      d += `M${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1} `;
    }
    // base ribbon + a faint moving dash that flows left→right (green/rsi ribbons flow fastest)
    svg += `<path d="${d}" fill="none" stroke="${c.line}" stroke-width="${w.toFixed(2)}" stroke-opacity="${op}" stroke-linecap="round"/>`;
    if (m.rsi) svg += `<path d="${d}" fill="none" stroke="#fff" stroke-width="${(w*0.5).toFixed(2)}" stroke-opacity="0.5" stroke-dasharray="2 26" stroke-linecap="round"><animate attributeName="stroke-dashoffset" from="28" to="0" dur="${(2.4).toFixed(1)}s" repeatCount="indefinite"/></path>`;
  });
  order.forEach(({ m, i }) => { const c = OC[outClass(m)];
    LANES.forEach((L, k) => { const x = laneX(k), y = yOf(laneVal(m, L), i + k), r = rOf(m);
      svg += `<circle cx="${x}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${c.fill}" fill-opacity="0.55" stroke="${c.line}" stroke-width="0.8"/>`; }); });
  const lx = padL + 6, ly = bandBot + 40; let leg = "";
  Object.values(OC).forEach((c, j) => leg += `<circle cx="${lx+j*210+6}" cy="${ly-4}" r="6" fill="${c.fill}" fill-opacity="0.6" stroke="${c.line}"/><text x="${lx+j*210+18}" y="${ly}" font-size="11.5" fill="#4a4a4a">${c.label}</text>`);
  leg += `<circle cx="${lx+660}" cy="${ly-4}" r="3" fill="#9a9a9a"/><text x="${lx+668}" y="${ly}" font-size="11.5" fill="#9a9a9a">small drift</text><circle cx="${lx+760}" cy="${ly-4}" r="10" fill="#9a9a9a" fill-opacity="0.4"/><text x="${lx+776}" y="${ly}" font-size="11.5" fill="#9a9a9a">large drift</text>`;
  svg += leg + `</svg>`;
  el.innerHTML = svg;
}

// ============================================================ 3. SCALE vs RSI-GAIN scatter + marginals (grow-in)
function renderScaleGain(models) {
  const el = document.getElementById("viz-scatter"); if (!el) return;
  const pts = models.map(m => ({ x: Math.log10(Math.max(0.3, +m.size_b || 0.5)), y: +m.C_held_gain || 0, drift: +m.drift_total || 0, cls: outClass(m), m }));
  const W = 820, H = 600, mL = 66, mR = 158, mT = 152, mB = 62, plotW = W-mL-mR, plotH = H-mT-mB;
  const xLo = -0.55, xHi = 1.3, yLo = Math.min(-0.15, ...pts.map(p=>p.y))-0.02, yHi = Math.max(0.5, ...pts.map(p=>p.y))+0.03;
  const X = v => mL + (v-xLo)/(xHi-xLo)*plotW, Y = v => mT + plotH - (v-yLo)/(yHi-yLo)*plotH;
  const dMax = Math.max(...pts.map(p=>p.drift), 0.01), R = d => 3 + Math.sqrt(d/dMax)*22;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scale vs RSI gain" font-family="Georgia, serif">`;
  svg += `<text x="${mL+plotW/2}" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#151515">Scale vs. RSI gain</text>`;
  svg += `<text x="${mL+plotW/2}" y="46" text-anchor="middle" font-size="11" fill="#9a9a9a">bubble area ∝ LoRA drift · shaded = sub-2B correctness wall · marginals show where models pile up</text>`;
  const xw = X(Math.log10(2));
  svg += `<rect x="${mL}" y="${mT}" width="${xw-mL}" height="${plotH}" fill="#f0f0f0"/><text x="${(mL+xw)/2}" y="${mT+16}" text-anchor="middle" font-size="10.5" fill="#b0b0b0" font-style="italic">correctness wall</text>`;
  svg += `<line x1="${mL}" y1="${mT+plotH}" x2="${mL+plotW}" y2="${mT+plotH}" stroke="#151515"/><line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT+plotH}" stroke="#151515"/>`;
  svg += `<line x1="${mL}" y1="${Y(0)}" x2="${mL+plotW}" y2="${Y(0)}" stroke="#c9c9c9" stroke-dasharray="3 3"/>`;
  [-0.5,0,0.5,1.0].forEach(t=>{const x=X(t); svg+=`<line x1="${x}" y1="${mT+plotH}" x2="${x}" y2="${mT+plotH+5}" stroke="#151515"/><text x="${x}" y="${mT+plotH+20}" text-anchor="middle" font-size="10" fill="#6a6a6a">${t.toFixed(2)}</text>`;});
  [0,0.2,0.4].forEach(t=>{const y=Y(t); svg+=`<line x1="${mL-5}" y1="${y}" x2="${mL}" y2="${y}" stroke="#151515"/><text x="${mL-9}" y="${y+3}" text-anchor="end" font-size="10" fill="#6a6a6a">${t.toFixed(1)}</text>`;});
  svg += `<text x="${mL+plotW/2}" y="${mT+plotH+44}" text-anchor="middle" font-size="11.5" fill="#4a4a4a">log10 params (B)</text>`;
  svg += `<text transform="translate(${mL-44},${mT+plotH/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="#4a4a4a">held-out C gain over rounds</text>`;
  const nb = 12;
  const xb = new Array(nb).fill(0); pts.forEach(p=>xb[Math.min(nb-1,Math.max(0,Math.floor((p.x-xLo)/(xHi-xLo)*nb)))]++);
  const xbM = Math.max(...xb,1);
  xb.forEach((c,b)=>{ if(!c)return; const x0=X(xLo+b/nb*(xHi-xLo)),x1=X(xLo+(b+1)/nb*(xHi-xLo)),h=c/xbM*54;
    svg+=`<rect x="${x0+1}" y="${mT-14-h}" width="${x1-x0-2}" height="${h}" fill="#7d9dc9" fill-opacity="0.55"/>`; });
  const yb = new Array(nb).fill(0); pts.forEach(p=>yb[Math.min(nb-1,Math.max(0,Math.floor((p.y-yLo)/(yHi-yLo)*nb)))]++);
  const ybM = Math.max(...yb,1);
  yb.forEach((c,b)=>{ if(!c)return; const y1=Y(yLo+b/nb*(yHi-yLo)),y0=Y(yLo+(b+1)/nb*(yHi-yLo)),w=c/ybM*72;
    svg+=`<rect x="${mL+plotW+14}" y="${y0+1}" width="${w}" height="${y1-y0-2}" fill="#5b9e6f" fill-opacity="0.5"/>`; });
  const ord = [...pts].sort((a,b)=>"rsi flat wall".indexOf(b.cls)-"rsi flat wall".indexOf(a.cls));
  ord.forEach((p,idx)=>{ const c=OC[p.cls], r=R(p.drift);
    svg+=`<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${r.toFixed(1)}" fill="${c.fill}" fill-opacity="0.5" stroke="${c.line}" stroke-width="1"><title>${p.m.model} · ${(+p.m.size_b).toFixed(1)}B · gain ${p.y.toFixed(3)} · drift ${p.drift.toFixed(3)}</title><animate attributeName="r" from="0" to="${r.toFixed(1)}" begin="${(idx*0.02).toFixed(2)}s" dur="0.5s" fill="freeze"/></circle>`; });
  Object.values(OC).forEach((c,j)=>{const yy=mT+6+j*20; svg+=`<circle cx="${mL+plotW+22}" cy="${yy}" r="6" fill="${c.fill}" fill-opacity="0.55" stroke="${c.line}"/><text x="${mL+plotW+34}" y="${yy+4}" font-size="10.5" fill="#4a4a4a">${c.label}</text>`;});
  svg += `</svg>`;
  el.innerHTML = svg;
}

// ---- load + render ----
fetch("data/mech_analysis.json").then(r => r.json()).then(d => {
  const ms = (d.models || []).filter(m => typeof m.size_b === "number");
  const up = document.getElementById("viz-updated");
  if (up) up.textContent = "updated " + (d.updated || "") + " · " + ms.length + " WHY-RSI probes, 0.5–15B";
  renderCausalityDAG(ms); renderScaleGain(ms);
}).catch(e => {});
renderTaskFlow();
