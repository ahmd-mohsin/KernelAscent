// KernelAscent mechanistic visuals — dark "research instrument" aesthetic (GPT-6-Astra design, 2026-09-18).
// Pure vanilla SVG+CSS, no deps. Driven by data/mech_analysis.json (per-model WHY-RSI series).
// Three large, center-aligned, densely-animated figures:
//   1. renderTaskFlow    — 5-stage RSI pipeline (1440x620), layered flowing edges + SMIL agent tokens
//   2. renderCausalityDAG — 7-gate internal-failure DAG (1440x860), one ribbon/model, failed runs STOP at their gate
//   3. renderScaleGain   — scale-vs-RSI-gain scatter (1440x900) w/ top+right marginals, bubble area = LoRA drift
const C = { cyan:"#54d8ff", violet:"#ae93ff", green:"#65dfb0", amber:"#f4c16b", red:"#f48592",
            ink:"#edf3ff", mut:"#a6b7cf", grid:"#233248", track:"#283b54", panel:"#0d1624", raised:"#121f31" };
const outClass = m => m.rsi ? "rsi" : (m.wall_crossed ? "flat" : "wall");
const outColor = m => m.rsi ? C.green : (m.wall_crossed ? C.amber : C.red);

// ---- shared: layered flowing edge (track + colored route + moving dash highlight) ----
function edge(id, d, col, w) {
  return `<path d="${d}" class="edge-track"/>`
       + `<path d="${d}" class="edge-color" stroke="${col}" ${w?`stroke-width="${w}"`:""} marker-end="url(#${id}-arrow)"/>`
       + `<path d="${d}" class="edge-flow"/>`;
}
// ---- shared: SMIL token particles traveling forward along a route id ----
function tokens(routeId, n, col, dur) {
  let s = `<g aria-hidden="true">`;
  for (let i = 0; i < n; i++) {
    const beg = (-dur * i / n).toFixed(2);
    const op = (0.9 - 0.5 * i / n).toFixed(2);
    s += `<circle r="${i===0?4:3.2}" fill="${i===0?'#e8fbff':col}" opacity="${op}">`
       + `<animateMotion dur="${dur}s" begin="${beg}s" repeatCount="indefinite"><mpath href="#${routeId}"/></animateMotion></circle>`;
  }
  return s + `</g>`;
}
function arrowDef(id, col) {
  return `<marker id="${id}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L9 5 L1 9 Z" fill="${col}"/></marker>`;
}

// ============================================================ 1. FIVE-STAGE PIPELINE
function renderTaskFlow() {
  const el = document.getElementById("viz-flow"); if (!el) return;
  const W = 1440, H = 620, n = 5;
  const stages = [
    { t: "T1 · Capability", s: "one-shot kernel skill", m: "pass@k", acc: C.cyan,
      d: "Write a correct, fast GPU kernel in one shot. Graded vs an fp32 reference + roofline speed." },
    { t: "T2 · Weight-RSI", s: "does self-training compound?", m: "lin−reset", acc: C.violet,
      d: "Open-weight lineage vs matched reset over rounds. The compounding test." },
    { t: "T3 · Procedure-RSI", s: "self-edit the harness", m: "Δ vs frozen", acc: C.green,
      d: "Model rewrites its own training procedure (weights fixed) round over round." },
    { t: "T4 · Closed→Open", s: "frontier improves a trainee", m: "improved−frozen", acc: C.amber,
      d: "A closed researcher edits an open trainee's harness; causal transfer." },
    { t: "T5 · Self-play", s: "author co-evolution", m: "L − F", acc: C.red,
      d: "Author + solver co-train. L−F isolates whether a learning author beats a frozen one." },
  ];
  const pad = 70, cw = 226, gap = (W - 2 * pad - n * cw) / (n - 1);
  const bT = 150, bH = 300, railY = 512;
  let defs = `<defs>${arrowDef("pl", C.violet)}`;
  for (let i = 0; i < n; i++)
    defs += `<linearGradient id="cg${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${stages[i].acc}" stop-opacity=".22"/><stop offset="1" stop-color="${stages[i].acc}" stop-opacity=".04"/></linearGradient>`;
  // routes between cards (for edges + token mpaths)
  const cx = i => pad + i * (cw + gap) + cw / 2;
  for (let i = 0; i < n - 1; i++) {
    const x1 = pad + i * (cw + gap) + cw, x2 = pad + (i + 1) * (cw + gap), y = bT + bH / 2;
    defs += `<path id="pl-route-${i}" d="M${x1} ${y} C${x1 + gap * .5} ${y} ${x2 - gap * .5} ${y} ${x2} ${y}"/>`;
  }
  defs += `<linearGradient id="rail" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C.cyan}"/><stop offset=".5" stop-color="${C.violet}"/><stop offset="1" stop-color="${C.red}"/></linearGradient></defs>`;
  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Five-stage RSI pipeline">${defs}`;
  svg += `<text x="${W/2}" y="54" text-anchor="middle" class="stage-title">The KernelAscent RSI ladder</text>`;
  svg += `<text x="${W/2}" y="86" text-anchor="middle" class="body-label">capability → weight-RSI → procedure-RSI → closed→open → self-play · tokens flow in stage order</text>`;
  // edges + tokens between cards
  for (let i = 0; i < n - 1; i++) {
    const x1 = pad + i * (cw + gap) + cw, x2 = pad + (i + 1) * (cw + gap), y = bT + bH / 2;
    const d = `M${x1} ${y} C${x1 + gap * .5} ${y} ${x2 - gap * .5} ${y} ${x2} ${y}`;
    svg += edge("pl", d, stages[i].acc, 4) + tokens(`pl-route-${i}`, 3, stages[i].acc, 3.2);
  }
  // stage cards
  for (let i = 0; i < n; i++) {
    const x = pad + i * (cw + gap), acc = stages[i].acc, s = stages[i];
    svg += `<rect x="${x}" y="${bT}" width="${cw}" height="${bH}" rx="16" fill="url(#cg${i})" stroke="${acc}" stroke-opacity=".45" stroke-width="1.3"/>`;
    svg += `<rect x="${x}" y="${bT}" width="${cw}" height="7" rx="3.5" fill="${acc}"/>`;
    svg += `<text x="${x+20}" y="${bT+44}" class="stage-title" font-size="20" fill="${acc}">${s.t}</text>`;
    svg += `<text x="${x+20}" y="${bT+70}" class="body-label">${s.s}</text>`;
    svg += `<text x="${x+20}" y="${bT+118}" class="metric" fill="${C.ink}">${s.m}</text>`;
    // wrapped description
    const words = s.d.split(" "); let line = "", ly = bT+156;
    for (const w of words) { if ((line+w).length > 30) { svg += `<text x="${x+20}" y="${ly}" font-size="13" fill="${C.mut}">${line}</text>`; line = w+" "; ly += 19; } else line += w+" "; }
    svg += `<text x="${x+20}" y="${ly}" font-size="13" fill="${C.mut}">${line}</text>`;
    svg += `<line x1="${cx(i)}" y1="${bT+bH}" x2="${cx(i)}" y2="${railY-6}" stroke="${acc}" stroke-opacity=".4" stroke-width="1.4"/>`;
    svg += `<circle cx="${cx(i)}" cy="${railY}" r="6" fill="${acc}"/>`;
  }
  // agent rail with a traveling token
  const x0 = cx(0), x1 = cx(n-1);
  svg += `<rect x="${x0}" y="${railY-3}" width="${x1-x0}" height="6" rx="3" fill="url(#rail)" opacity=".9"/>`;
  svg += `<text x="${W/2}" y="${railY+38}" text-anchor="middle" class="body-label">the same agent is carried across every stage — the axis of the benchmark</text>`;
  svg += `<circle r="7" fill="#fff"><animate attributeName="cx" from="${x0}" to="${x1}" dur="7s" repeatCount="indefinite"/><animate attributeName="cy" values="${railY};${railY}" dur="7s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;.4;1" dur="7s" repeatCount="indefinite"/></circle>`;
  svg += `</svg>`;
  el.innerHTML = svg;
}

// ============================================================ 2. INTERNAL-FAILURE CAUSALITY DAG
function renderCausalityDAG(models) {
  const el = document.getElementById("viz-dag"); if (!el) return;
  const W = 1440, H = 860;
  const gates = ["SCALE","WALL","GRADIENT","DRIFT","RETENTION","DIVERSITY","OUTCOME"];
  const gx = i => 120 + i * (W - 240) / (gates.length - 1);
  const topY = 150, botY = 780;
  // y by size (log), so ribbons fan by scale
  const sizes = models.map(m => m.size_b || 2);
  const lo = Math.log(Math.min(...sizes)), hi = Math.log(Math.max(...sizes));
  const yOf = s => topY + (botY - topY) * (1 - (Math.log(s) - lo) / (hi - lo + 1e-9));
  // how far each model progresses before stopping (failed runs STOP at their gate)
  const stopGate = m => !m.wall_crossed ? 1 : (m.drift_total < 0.05 ? 2 : (m.rsi ? 6 : 5));
  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Internal-failure causality DAG">`;
  svg += `<text x="${W/2}" y="52" text-anchor="middle" class="stage-title">Where recursive self-improvement breaks — internal-failure causality</text>`;
  svg += `<text x="${W/2}" y="84" text-anchor="middle" class="body-label">${models.length} model runs · each ribbon threads the gates it clears and STOPS where it stalls · width ∝ LoRA drift · color = outcome</text>`;
  // gate columns + centered headers
  gates.forEach((g, i) => {
    svg += `<line x1="${gx(i)}" y1="${topY-14}" x2="${gx(i)}" y2="${botY+14}" stroke="${C.grid}" stroke-width="1"/>`;
    svg += `<text x="${gx(i)}" y="${topY-30}" text-anchor="middle" class="gatehdr">${g}</text>`;
  });
  // ribbons (draw failed first / faint, rsi last / bright)
  const ordered = models.slice().sort((a,b) => (a.rsi?2:a.wall_crossed?1:0) - (b.rsi?2:b.wall_crossed?1:0));
  const routeIds = [];
  ordered.forEach((m, idx) => {
    const col = outColor(m), yb = yOf(m.size_b || 2), sg = stopGate(m);
    const w = Math.max(1, Math.min(7, 1 + (m.drift_total || 0) * 8));
    // path from SCALE to its stop gate, easing toward a mild outcome spread
    let d = `M${gx(0)} ${yb}`;
    for (let i = 1; i <= sg; i++) {
      const x = gx(i), prevx = gx(i-1);
      const y = (i === 6) ? (m.rsi ? topY + 60 : yb) : yb + (Math.sin(idx + i) * 6);
      d += ` C${(prevx+x)/2} ${yb} ${(prevx+x)/2} ${y} ${x} ${y}`;
    }
    const op = m.rsi ? 0.95 : (m.wall_crossed ? 0.6 : 0.4);
    svg += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${w.toFixed(2)}" stroke-opacity="${op}" stroke-linecap="round"/>`;
    // failed runs: a small "stop" tick at the gate they died on
    if (!m.rsi) { const ex = gx(sg); svg += `<circle cx="${ex}" cy="${yb}" r="${(w*.7+1).toFixed(1)}" fill="none" stroke="${col}" stroke-width="1.4" stroke-opacity="${op}"/>`; }
    // rsi ribbons get a token route (reuse the same path) — keep the particle budget bounded
    if (m.rsi && routeIds.length < 22) { const rid = `dag-r${idx}`; svg += `<path id="${rid}" d="${d}" fill="none" stroke="none"/>`; routeIds.push(rid); }
  });
  // token particles flow ONLY along rsi ribbons (semantic: successful recursion carries signal through)
  routeIds.forEach((rid, k) => { svg += tokens(rid, 2, "#eafff6", 2.6 + (k % 3) * .4); });
  // gate outcome legend
  const leg = [["RSI compounds", C.green], ["crossed wall, flat", C.amber], ["stuck at wall", C.red]];
  leg.forEach(([t, c], i) => { const lx = 120 + i * 240; svg += `<circle cx="${lx}" cy="${botY+52}" r="6" fill="${c}"/><text x="${lx+14}" y="${botY+57}" class="body-label">${t}</text>`; });
  svg += `</svg>`;
  el.innerHTML = svg;
}

// ============================================================ 3. SCALE vs RSI-GAIN SCATTER (+ marginals)
function renderScaleGain(models) {
  const el = document.getElementById("viz-scatter"); if (!el) return;
  const W = 1440, H = 900, mL = 110, mR = 240, mT = 210, mB = 120;
  const px = W - mR, py = H - mB;
  const xs = models.map(m => Math.log(m.size_b || 2));
  const xlo = Math.min(...xs) - 0.15, xhi = Math.max(...xs) + 0.15;
  const gains = models.map(m => m.C_held_gain || 0);
  const ylo = Math.min(-0.05, ...gains), yhi = Math.max(0.05, ...gains);
  const X = v => mL + (px - mL) * (Math.log(v) - xlo) / (xhi - xlo);
  const Y = v => py - (py - mT) * (v - ylo) / (yhi - ylo);
  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Scale vs RSI gain">`;
  svg += `<text x="${W/2}" y="54" text-anchor="middle" class="stage-title">Scale vs held-out RSI gain — an inverted-U, not a staircase</text>`;
  svg += `<text x="${W/2}" y="86" text-anchor="middle" class="body-label">bubble area ∝ LoRA drift · color = outcome · gain peaks mid-scale; ≥9B drifts most yet gains least (roofline saturation)</text>`;
  // correctness-wall shaded region (<2B)
  const wallX = X(2);
  svg += `<rect x="${mL}" y="${mT}" width="${wallX-mL}" height="${py-mT}" fill="#16233a" opacity=".55"/>`;
  svg += `<text x="${(mL+wallX)/2}" y="${mT+26}" text-anchor="middle" class="body-label" font-style="italic">sub-2B correctness wall</text>`;
  // axes + gridlines
  svg += `<line x1="${mL}" y1="${py}" x2="${px}" y2="${py}" stroke="${C.grid}"/><line x1="${mL}" y1="${mT}" x2="${mL}" y2="${py}" stroke="${C.grid}"/>`;
  svg += `<line x1="${mL}" y1="${Y(0)}" x2="${px}" y2="${Y(0)}" stroke="${C.grid}" stroke-dasharray="4 6"/><text x="${px+6}" y="${Y(0)+4}" class="axis-label">0</text>`;
  [0.5,1,2,3,7,14,32].forEach(s => { if (s>=Math.exp(xlo)&&s<=Math.exp(xhi)) svg += `<text x="${X(s)}" y="${py+30}" text-anchor="middle" class="axis-label mono">${s}B</text>`; });
  svg += `<text x="${(mL+px)/2}" y="${py+62}" text-anchor="middle" class="axis-label">model size (log)</text>`;
  svg += `<text x="26" y="${(mT+py)/2}" text-anchor="middle" class="axis-label" transform="rotate(-90 26 ${(mT+py)/2})">held-out capability gain</text>`;
  // top marginal: mean gain in scale bands
  const bands = [[0,2,"<2B"],[2,8,"2–8B"],[8,99,"≥9B"]];
  bands.forEach(([a,b,lab]) => {
    const inb = models.filter(m => (m.size_b||2)>=a && (m.size_b||2)<b);
    if (!inb.length) return;
    const rsiFrac = inb.filter(m=>m.rsi).length/inb.length;
    const x0 = X(Math.max(a,Math.exp(xlo))), x1 = X(Math.min(b,Math.exp(xhi)));
    const bh = 90 * rsiFrac;
    svg += `<rect x="${x0+4}" y="${mT-20-bh}" width="${x1-x0-8}" height="${bh}" fill="${C.green}" opacity=".5" rx="3"/>`;
    svg += `<text x="${(x0+x1)/2}" y="${mT-26-bh}" text-anchor="middle" class="body-label">${(rsiFrac*100).toFixed(0)}% RSI</text>`;
    svg += `<text x="${(x0+x1)/2}" y="${mT-6}" text-anchor="middle" class="axis-label mono">${lab}</text>`;
  });
  // bubbles
  models.forEach(m => {
    const r = Math.max(3, Math.min(26, 4 + (m.drift_total||0)*26));
    const col = outColor(m);
    svg += `<circle cx="${X(m.size_b||2).toFixed(1)}" cy="${Y(m.C_held_gain||0).toFixed(1)}" r="${r.toFixed(1)}" fill="${col}" fill-opacity=".38" stroke="${col}" stroke-width="1.1"/>`;
  });
  // right marginal: gain distribution
  svg += `<text x="${px+80}" y="${mT-6}" text-anchor="middle" class="axis-label">drift ∝ bubble</text>`;
  const leg = [["RSI", C.green], ["flat", C.amber], ["stuck", C.red]];
  leg.forEach(([t,c],i)=>{ svg += `<circle cx="${px+40}" cy="${mT+40+i*30}" r="7" fill="${c}" fill-opacity=".5" stroke="${c}"/><text x="${px+54}" y="${mT+45+i*30}" class="body-label">${t}</text>`; });
  svg += `</svg>`;
  el.innerHTML = svg;
}

(function init(){
  fetch("data/mech_analysis.json").then(r=>r.json()).then(d=>{
    const models = (d.models||[]).filter(m => typeof m.size_b === "number");
    try { renderTaskFlow(); } catch(e){ console.error("flow",e); }
    try { renderCausalityDAG(models); } catch(e){ console.error("dag",e); }
    try { renderScaleGain(models); } catch(e){ console.error("scatter",e); }
  }).catch(e=>console.error("viz load",e));
})();
