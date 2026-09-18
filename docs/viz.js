// KernelAscent mechanistic visuals — dark "research instrument" (GPT-6-Astra design + review, 2026-09-18).
// Pure vanilla SVG+CSS. Driven by data/mech_analysis.json. Three large, center-aligned figures:
//   1. Pipeline (1440x600)  — ONE packet threads all 5 stages (single clock, dwell+travel), active card emphasized.
//   2. Causality DAG (1440x820) — runs BUNDLED by (terminal-gate, outcome); width = Σ drift, cap labeled n; no spaghetti.
//   3. Scatter (1440x900)  — fixed center dot + area-scaled drift halo; scale vs held-out gain, marginals.
const C = { cyan:"#54d8ff", violet:"#ae93ff", green:"#65dfb0", amber:"#f4c16b", red:"#f48592",
            ink:"#edf3ff", mut:"#a6b7cf", grid:"#233248", track:"#2b3d57" };
const outColor = m => m.rsi ? C.green : (m.wall_crossed ? C.amber : C.red);
const SVGNS = "http://www.w3.org/2000/svg";

// ============================================================ 1. PIPELINE (single-clock packet)
function renderTaskFlow() {
  const el = document.getElementById("viz-flow"); if (!el) return;
  const W = 1440, H = 600, n = 5;
  const stages = [
    { t:"T1 · Capability", s:"one-shot kernel skill", m:"pass@k", acc:C.cyan, d:"Write a correct, fast kernel in one shot — graded vs an fp32 reference and roofline speed." },
    { t:"T2 · Weight-RSI", s:"does self-training compound?", m:"lin − reset", acc:C.violet, d:"Open-weight lineage vs a matched reset over rounds. The core compounding test." },
    { t:"T3 · Procedure-RSI", s:"self-edit the harness", m:"Δ vs frozen", acc:C.green, d:"The model rewrites its own training procedure with weights fixed, round over round." },
    { t:"T4 · Closed→Open", s:"frontier improves a trainee", m:"improved − frozen", acc:C.amber, d:"A closed researcher edits an open trainee's harness — causal transfer through tooling." },
    { t:"T5 · Self-play", s:"author co-evolution", m:"L − F", acc:C.red, d:"Author and solver co-train. L−F isolates whether a learning author beats a frozen one." },
  ];
  const pad = 66, cw = 232, gap = (W - 2*pad - n*cw) / (n-1);
  const bT = 150, bH = 300, midY = bT + bH/2;
  const cardX = i => pad + i*(cw+gap);
  const cx = i => cardX(i) + cw/2;
  let defs = `<defs>`;
  for (let i=0;i<n;i++) defs += `<linearGradient id="cg${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${stages[i].acc}" stop-opacity=".18"/><stop offset="1" stop-color="${stages[i].acc}" stop-opacity=".03"/></linearGradient>`;
  defs += `</defs>`;
  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Five-stage RSI pipeline">${defs}`;
  svg += `<text x="${W/2}" y="52" text-anchor="middle" class="stage-title">The KernelAscent RSI ladder</text>`;
  svg += `<text x="${W/2}" y="84" text-anchor="middle" class="body-label">one agent carried through five stages of increasing recursion depth — a single execution trace</text>`;
  // subdued static connectors (2px neutral)
  for (let i=0;i<n-1;i++) {
    const x1 = cardX(i)+cw, x2 = cardX(i+1);
    svg += `<path id="pl-c${i}" d="M${x1} ${midY} C${x1+gap*.5} ${midY} ${x2-gap*.5} ${midY} ${x2} ${midY}" fill="none" stroke="${C.track}" stroke-width="2"/>`;
  }
  // cards
  for (let i=0;i<n;i++) {
    const x = cardX(i), acc = stages[i].acc, s = stages[i];
    svg += `<g id="pl-card${i}"><rect x="${x}" y="${bT}" width="${cw}" height="${bH}" rx="16" fill="url(#cg${i})" stroke="${acc}" stroke-opacity=".35" stroke-width="1.3" class="pl-cardbox"/>`;
    svg += `<rect x="${x}" y="${bT}" width="${cw}" height="7" rx="3.5" fill="${acc}"/>`;
    svg += `<text x="${x+22}" y="${bT+46}" class="stage-title" font-size="20" fill="${acc}">${s.t}</text>`;
    svg += `<text x="${x+22}" y="${bT+72}" class="body-label">${s.s}</text>`;
    svg += `<text x="${x+22}" y="${bT+124}" class="metric">${s.m}</text>`;
    let line="", ly=bT+162; for (const w of s.d.split(" ")) { if ((line+w).length>32){svg+=`<text x="${x+22}" y="${ly}" font-size="13" fill="${C.mut}">${line}</text>`;line=w+" ";ly+=19;} else line+=w+" "; }
    svg += `<text x="${x+22}" y="${ly}" font-size="13" fill="${C.mut}">${line}</text></g>`;
  }
  // single packet
  svg += `<circle id="pl-packet" r="8" fill="#fff" opacity="0"/><circle id="pl-halo" r="8" fill="none" stroke="#fff" stroke-width="2" opacity="0"/>`;
  svg += `</svg>`;
  el.innerHTML = svg;
  // one-clock animation: dwell at card, travel connector, emphasize active card
  const svgEl = el.querySelector("svg");
  const packet = el.querySelector("#pl-packet"), halo = el.querySelector("#pl-halo");
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    packet.setAttribute("cx", cx(0)); packet.setAttribute("cy", midY); packet.setAttribute("opacity","1"); return;
  }
  const conns = [...Array(n-1)].map((_,i)=>el.querySelector(`#pl-c${i}`));
  const cards = [...Array(n)].map((_,i)=>el.querySelector(`#pl-card${i} .pl-cardbox`));
  const DWELL=650, TRAVEL=750; let stage=0, phase="dwell", t0=performance.now();
  packet.setAttribute("opacity","1");
  function frame(now){
    const el2 = el.querySelector("#pl-packet"); if(!el2) return;         // stop if re-rendered
    const dt = now - t0;
    cards.forEach((c,i)=>{ if(c) c.setAttribute("stroke-opacity", i===stage?"0.95":"0.35"); });
    if (phase==="dwell") {
      const p = cx(stage); packet.setAttribute("cx",p); packet.setAttribute("cy",midY);
      halo.setAttribute("cx",p); halo.setAttribute("cy",midY);
      const k = Math.min(1, dt/DWELL); halo.setAttribute("opacity", (0.5*Math.sin(k*Math.PI)).toFixed(2)); halo.setAttribute("r",(8+10*Math.sin(k*Math.PI)).toFixed(1));
      if (dt>=DWELL){ if(stage>=n-1){stage=0; t0=now; return requestAnimationFrame(frame);} phase="travel"; t0=now; }
    } else {
      const conn = conns[stage]; const L = conn.getTotalLength(); const k=Math.min(1,dt/TRAVEL);
      const pt = conn.getPointAtLength(L*k); packet.setAttribute("cx",pt.x); packet.setAttribute("cy",pt.y); halo.setAttribute("opacity","0");
      if (dt>=TRAVEL){ stage++; phase="dwell"; t0=now; }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ============================================================ 2. CAUSALITY DAG (bundled)
function renderCausalityDAG(models) {
  const el = document.getElementById("viz-dag"); if (!el) return;
  const W = 1440, H = 820;
  const gates = ["SCALE","WALL","GRADIENT","DRIFT","RETENTION","DIVERSITY","OUTCOME"];
  const gx = i => 130 + i*(W-260)/(gates.length-1);
  const topY = 150, botY = 690;
  const stopGate = m => !m.wall_crossed ? 1 : (m.drift_total < 0.05 ? 2 : (m.rsi ? 6 : 5));
  // bundle by (terminalGate, outcome)
  const B = {};
  models.forEach(m => { const sg=stopGate(m), oc=m.rsi?"rsi":(m.wall_crossed?"flat":"wall"); const k=sg+"|"+oc;
    (B[k]=B[k]||{sg,oc,n:0,drift:0,col:outColor(m)}).n++; B[k].drift += (m.drift_total||0); });
  const bundles = Object.values(B).sort((a,b)=> a.sg-b.sg || (a.oc<b.oc?-1:1));
  // stable vertical slots per terminal gate
  const bySg = {}; bundles.forEach(b=> (bySg[b.sg]=bySg[b.sg]||[]).push(b));
  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Internal-failure causality DAG (bundled)">`;
  svg += `<text x="${W/2}" y="52" text-anchor="middle" class="stage-title">Where recursive self-improvement breaks</text>`;
  svg += `<text x="${W/2}" y="84" text-anchor="middle" class="body-label">${models.length} runs bundled by where they stall · ribbon width ∝ total LoRA drift · n = runs · color = outcome (aggregate width ≠ success probability)</text>`;
  gates.forEach((g,i)=>{ svg += `<line x1="${gx(i)}" y1="${topY-14}" x2="${gx(i)}" y2="${botY+14}" stroke="${C.grid}"/><text x="${gx(i)}" y="${topY-28}" text-anchor="middle" class="gatehdr">${g}</text>`; });
  const totalDrift = bundles.reduce((s,b)=>s+b.drift,0) || 1;
  bundles.forEach(b => {
    const slots = bySg[b.sg]; const idx = slots.indexOf(b);
    const yb = topY + (botY-topY)*((idx+1)/(slots.length+1));
    const w = Math.max(3, Math.min(46, 3 + b.drift*10));
    let d = `M${gx(0)} ${yb}`; for (let i=1;i<=b.sg;i++){ const x=gx(i),px=gx(i-1); const y=(i===6&&b.oc==="rsi")?topY+70:yb; d+=` C${(px+x)/2} ${yb} ${(px+x)/2} ${y} ${x} ${y}`; }
    const op = b.oc==="rsi"?0.9:(b.oc==="flat"?0.6:0.42);
    svg += `<path d="${d}" fill="none" stroke="${b.col}" stroke-width="${w.toFixed(1)}" stroke-opacity="${op}" stroke-linecap="round"/>`;
    // termination cap + run count
    const ex = gx(b.sg), ey = (b.sg===6&&b.oc==="rsi")?topY+70:yb;
    svg += `<circle cx="${ex}" cy="${ey}" r="${(w/2+5).toFixed(1)}" fill="${C.panel||'#0d1624'}" stroke="${b.col}" stroke-width="1.6" stroke-opacity="${op}"/>`;
    svg += `<text x="${ex}" y="${ey+4}" text-anchor="middle" class="mono" font-size="12" fill="${b.col}">${b.n}</text>`;
    // subtle single flow only on the main rsi bundle (largest)
  });
  // one animated flow on the biggest rsi bundle for a hint of life
  const rsiB = bundles.filter(b=>b.oc==="rsi").sort((a,b)=>b.drift-a.drift)[0];
  if (rsiB){ const slots=bySg[rsiB.sg],idx=slots.indexOf(rsiB),yb=topY+(botY-topY)*((idx+1)/(slots.length+1));
    let d=`M${gx(0)} ${yb}`; for(let i=1;i<=rsiB.sg;i++){const x=gx(i),px=gx(i-1);const y=(i===6)?topY+70:yb;d+=` C${(px+x)/2} ${yb} ${(px+x)/2} ${y} ${x} ${y}`;}
    svg += `<path d="${d}" fill="none" stroke="#eafff6" stroke-width="2" stroke-dasharray="4 16" opacity=".8" class="edge-flow"/>`; }
  const leg = [["RSI compounds",C.green],["crossed wall, flat",C.amber],["stuck at wall",C.red]];
  leg.forEach(([t,c],i)=>{ const lx=130+i*250; svg += `<circle cx="${lx}" cy="${botY+56}" r="6" fill="${c}"/><text x="${lx+14}" y="${botY+61}" class="body-label">${t}</text>`; });
  svg += `</svg>`; el.innerHTML = svg;
}

// ============================================================ 3. SCATTER (center dot + drift halo)
function renderScaleGain(models) {
  const el = document.getElementById("viz-scatter"); if (!el) return;
  const W=1440,H=900,mL=110,mR=230,mT=210,mB=120, px=W-mR, py=H-mB;
  const xs=models.map(m=>Math.log(m.size_b||2)); const xlo=Math.min(...xs)-0.15,xhi=Math.max(...xs)+0.15;
  const gains=models.map(m=>m.C_held_gain||0); const ylo=Math.min(-0.05,...gains),yhi=Math.max(0.05,...gains);
  const X=v=>mL+(px-mL)*(Math.log(v)-xlo)/(xhi-xlo); const Y=v=>py-(py-mT)*(v-ylo)/(yhi-ylo);
  let svg=`<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Scale vs RSI gain">`;
  svg += `<text x="${W/2}" y="52" text-anchor="middle" class="stage-title">Scale vs held-out RSI gain — an inverted-U, not a staircase</text>`;
  svg += `<text x="${W/2}" y="84" text-anchor="middle" class="body-label">center dot = model · halo area ∝ LoRA drift · color = outcome · gain peaks mid-scale; ≥9B drifts most yet gains least</text>`;
  const wallX=X(2);
  svg += `<rect x="${mL}" y="${mT}" width="${wallX-mL}" height="${py-mT}" fill="#16233a" opacity=".5"/><text x="${(mL+wallX)/2}" y="${mT+26}" text-anchor="middle" class="body-label" font-style="italic">sub-2B correctness wall</text>`;
  svg += `<line x1="${mL}" y1="${py}" x2="${px}" y2="${py}" stroke="${C.grid}"/><line x1="${mL}" y1="${mT}" x2="${mL}" y2="${py}" stroke="${C.grid}"/>`;
  svg += `<line x1="${mL}" y1="${Y(0)}" x2="${px}" y2="${Y(0)}" stroke="${C.grid}" stroke-dasharray="4 6"/><text x="${px+6}" y="${Y(0)+4}" class="axis-label">0</text>`;
  [0.5,1,2,3,7,14,32].forEach(s=>{ if(s>=Math.exp(xlo)&&s<=Math.exp(xhi)) svg+=`<text x="${X(s)}" y="${py+30}" text-anchor="middle" class="axis-label mono">${s}B</text>`; });
  svg += `<text x="${(mL+px)/2}" y="${py+62}" text-anchor="middle" class="axis-label">model size (log)</text>`;
  svg += `<text x="30" y="${(mT+py)/2}" text-anchor="middle" class="axis-label" transform="rotate(-90 30 ${(mT+py)/2})">held-out capability gain</text>`;
  // top marginal: %RSI per band
  [[0,2,"<2B"],[2,8,"2–8B"],[8,99,"≥9B"]].forEach(([a,b,lab])=>{ const inb=models.filter(m=>(m.size_b||2)>=a&&(m.size_b||2)<b); if(!inb.length)return;
    const fr=inb.filter(m=>m.rsi).length/inb.length; const x0=X(Math.max(a,Math.exp(xlo))),x1=X(Math.min(b,Math.exp(xhi))); const bh=90*fr;
    svg+=`<rect x="${x0+4}" y="${mT-24-bh}" width="${x1-x0-8}" height="${bh}" fill="${C.green}" opacity=".45" rx="3"/><text x="${(x0+x1)/2}" y="${mT-30-bh}" text-anchor="middle" class="body-label">${(fr*100).toFixed(0)}% RSI</text><text x="${(x0+x1)/2}" y="${mT-8}" text-anchor="middle" class="axis-label mono">${lab}</text>`; });
  // marks: halo (area ∝ drift) + fixed center dot
  models.forEach(m=>{ const col=outColor(m); const r=Math.max(4,Math.sqrt((m.drift_total||0)*2400/Math.PI)); const x=X(m.size_b||2).toFixed(1),y=Y(m.C_held_gain||0).toFixed(1);
    svg+=`<circle cx="${x}" cy="${y}" r="${r.toFixed(1)}" fill="${col}" fill-opacity=".07" stroke="${col}" stroke-opacity=".28"/><circle cx="${x}" cy="${y}" r="2.6" fill="${col}"/>`; });
  const leg=[["RSI",C.green],["flat",C.amber],["stuck",C.red]]; leg.forEach(([t,c],i)=>{ svg+=`<circle cx="${px+44}" cy="${mT+40+i*30}" r="6" fill="${c}"/><text x="${px+58}" y="${mT+45+i*30}" class="body-label">${t}</text>`; });
  svg += `<text x="${px+44}" y="${mT+40+3*30+8}" class="axis-label" font-size="13">halo ∝ drift</text>`;
  svg += `</svg>`; el.innerHTML = svg;
}

(function init(){
  fetch("data/mech_analysis.json").then(r=>r.json()).then(d=>{
    const models=(d.models||[]).filter(m=>typeof m.size_b==="number");
    try{renderTaskFlow();}catch(e){console.error("flow",e);}
    try{renderCausalityDAG(models);}catch(e){console.error("dag",e);}
    try{renderScaleGain(models);}catch(e){console.error("scatter",e);}
  }).catch(e=>console.error("viz load",e));
})();
