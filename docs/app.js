// KernelAscent leaderboards. Capability loads data/leaderboard.json, RSI loads data/rsi_leaderboard.json.
const NUM = ["correct_rate", "fast_rate", "meanC"];
let DATA = [], sortKey = "meanC", sortDir = -1;

function fmt(v) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") return (Math.round(v * 1000) / 1000).toFixed(3);
  return v;
}
function fmtci(v, ci) {
  if (v === null || v === undefined) return "—";
  const base = fmt(v);
  return (Array.isArray(ci) && ci.length === 2) ? `${base} <span class="mid small">[${fmt(ci[0])}, ${fmt(ci[1])}]</span>` : base;
}
function kindPill(k) {
  const open = (k === "open");
  return `<span class="pill kind">${open ? "open weight" : "closed / API"}</span>`;
}

function render() {
  const tb = document.querySelector("#lb tbody");
  const best = {};
  NUM.forEach(k => { best[k] = Math.max(...DATA.map(m => m[k] ?? -Infinity)); });
  const rows = [...DATA].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (typeof av === "number" || typeof bv === "number") return ((av ?? -Infinity) - (bv ?? -Infinity)) * sortDir;
    return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
  });
  const ciKey = { correct_rate: "correct_ci", fast_rate: "fast_ci" };
  tb.innerHTML = rows.map(m => {
    const cell = (k) => {
      const isBest = m[k] === best[k] && best[k] > -Infinity;
      return `<td class="num ${isBest ? "best" : ""}">${ciKey[k] ? fmtci(m[k], m[ciKey[k]]) : fmt(m[k])}</td>`;
    };
    return `<tr><td><b>${m.model}</b></td><td>${kindPill(m.kind)}</td>${cell("correct_rate")}${cell("fast_rate")}${cell("meanC")}</tr>`;
  }).join("");
}
function wireSort() {
  document.querySelectorAll("#lb th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k; if (!k) return;
    if (k === sortKey) sortDir *= -1; else { sortKey = k; sortDir = NUM.includes(k) ? -1 : 1; }
    render();
  }));
}
fetch("data/leaderboard.json").then(r => r.json()).then(d => {
  DATA = d.models || [];
  const up = document.getElementById("lb-updated"); if (up) up.textContent = "updated " + (d.updated || "");
  wireSort(); render(); renderCapChart(DATA);
}).catch(e => {
  const tb = document.querySelector("#lb tbody"); if (tb) tb.innerHTML = `<tr><td colspan="5" class="mid">Serve over HTTP to load the leaderboard.</td></tr>`;
});

// RSI leaderboard
function verdictPill(v) {
  v = (v || "").toLowerCase();
  if (v.startsWith("compound")) return `<span class="pill good">compounds</span>`;
  if (v.startsWith("gain")) return `<span class="pill gain">gains</span>`;
  if (v.startsWith("degrade")) return `<span class="pill bad">overfits</span>`;
  return `<span class="pill flat">flat</span>`;
}
function num(v){ return (v===null||v===undefined||v==="") ? "—" : (typeof v==="number" ? (Math.round(v*1000)/1000).toFixed(3) : v); }
function deltaBar(v) {
  if (typeof v !== "number") return "—";
  const w = Math.min(100, Math.abs(v) * 100);
  const pos = v >= 0;
  const bar = `<span style="display:inline-block;height:8px;width:${w}%;background:${pos ? "#8C1515" : "#d8bcbc"};border-radius:2px;vertical-align:middle"></span>`;
  return `<span class="mono ${pos ? "pos" : "neg"}">${pos ? "+" : ""}${v.toFixed(3)}</span> ${bar}`;
}
fetch("data/rsi_leaderboard.json").then(r => r.json()).then(d => {
  const up = document.getElementById("rsi-updated"); if (up) up.textContent = (d.updated ? d.updated + " · " : "") + (d.bank || "");
  const tb = document.querySelector("#rsi-lb tbody");
  const ms = d.models || [];
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.model}</b></td><td><span class="pill kind">${m.family}</span></td>
    <td class="num">${num(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.Cfinal)}</td>
    <td class="num">${num(m.correct_rate)}</td><td class="num mono">${num(m.compiled_sp)}</td>
    <td class="num">${deltaBar(m.delta_frozen)}</td><td class="num">${deltaBar(m.self_minus_fresh)}</td>
    <td>${verdictPill(m.verdict)}</td></tr>`).join("");
  renderRsiChart(ms);
}).catch(e => {
  const tb = document.querySelector("#rsi-lb tbody"); if (tb) tb.innerHTML = `<tr><td colspan="8" class="mid">Serve over HTTP to load results.</td></tr>`;
});

// ---- dynamic bar charts ----
function verdictClass(v){ v=(v||"").toLowerCase(); return v.startsWith("compound")?"good":v.startsWith("gain")?"gain":v.startsWith("degrade")?"bad":"flat"; }
function chartRow(label, pct, cls, valText, leftPct){
  const left = leftPct ? `left:${leftPct}%;` : "left:0;";
  return `<div class="row"><div class="lab" title="${label}">${label}</div>
    <div class="track2"><span class="zero" style="left:${leftPct||0}%"></span><span class="fill ${cls}" data-w="${pct}" style="${left}"></span></div>
    <div class="val">${valText}</div></div>`;
}
function animateFills(root){
  requestAnimationFrame(()=>{ root.querySelectorAll(".fill").forEach(f=>{ f.style.width = f.dataset.w + "%"; }); });
}
function renderCapChart(models){
  const el = document.getElementById("cap-chart"); if (!el) return;
  const ms = [...models].filter(m=>typeof m.meanC==="number").sort((a,b)=>b.meanC-a.meanC).slice(0,14);
  el.innerHTML = ms.map(m=>chartRow(m.model, Math.round(m.meanC*100), (m.kind==="open"?"open":"closed"), m.meanC.toFixed(3))).join("")
    + `<div class="chart-legend"><span><i class="fill open" style="background:linear-gradient(90deg,#6f1010,#B1040E)"></i>open weight</span><span><i class="fill closed" style="background:repeating-linear-gradient(45deg,#8C1515,#8C1515 5px,#c0575f 5px,#c0575f 10px)"></i>closed / API</span><span>bar = mean C</span></div>`;
  animateFills(el);
}
function renderRsiChart(models){
  const el = document.getElementById("rsi-chart"); if (!el) return;
  // chart the CAUSAL metric: self vs fresh-frozen producer, centered at 0
  const ms = [...models].filter(m=>typeof m.self_minus_fresh==="number").sort((a,b)=>b.self_minus_fresh-a.self_minus_fresh);
  const rows = ms.map(m=>{
    const d = Math.max(-0.5,Math.min(0.5,m.self_minus_fresh))/0.5;   // scale +-0.5 to full width
    const pct = Math.abs(d)*50; const left = d>=0?50:50-pct;
    return chartRow(m.model, pct, verdictClass(m.verdict), (m.self_minus_fresh>=0?"+":"")+m.self_minus_fresh.toFixed(3), left);
  });
  el.innerHTML = rows.join("")
    + `<div class="chart-legend"><span><i style="background:linear-gradient(90deg,#6f1010,#8C1515)"></i>compounds</span><span><i style="background:linear-gradient(90deg,#b8555b,#d98a90)"></i>gains</span><span><i style="background:#e0cccc"></i>flat</span><span>bar = self minus fresh-frozen producer, center = 0</span></div>`;
  animateFills(el);
}

// Per-scale speed board
fetch("data/tier_speed_rsi.json").then(r => r.json()).then(d => {
  const up = document.getElementById("ts-updated"); if (up) up.textContent = (d.updated ? d.updated + " · " : "") + (d.metric || "");
  const tb = document.querySelector("#tier-lb tbody"); if (!tb) return;
  const ms = [...(d.models||[])].sort((a,b)=>(b.self_minus_fresh??-9)-(a.self_minus_fresh??-9));
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.model}</b></td><td><span class="pill kind">${m.tier}</span></td>
    <td class="num">${num(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.C_self)}</td>
    <td class="num">${deltaBar(m.self_minus_fresh)}</td><td>${verdictPill(m.verdict)}</td></tr>`).join("");
}).catch(e => {});

// Task 3 (closed->open) board
fetch("data/combined_rsi.json").then(r => r.json()).then(d => {
  const up = document.getElementById("c3-updated"); if (up) up.textContent = (d.updated ? d.updated : "");
  const tb = document.querySelector("#rsi3-lb tbody"); if (!tb) return;
  const ms = d.models || [];
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.trainee}</b></td><td><span class="pill kind">${m.researcher}</span></td>
    <td class="num">${num(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.C_improved)}</td>
    <td class="num">${num(m.C_frozen)}</td><td class="num">${deltaBar(m.impr_minus_frozen)}</td>
    <td>${verdictPill(m.verdict==="harness helps"?"compounds":(m.verdict==="hurts"?"overfits":"flat"))}</td></tr>`).join("");
}).catch(e => {});

// Baselines board (recursion vs sampling)
fetch("data/baselines.json").then(r => r.json()).then(d => {
  const up = document.getElementById("bl-updated"); if (up) up.textContent = (d.updated || "");
  const tb = document.querySelector("#bl-lb tbody"); if (!tb) return;
  const ms = [...(d.models || [])].sort((a, b) => (b.recursion_gain ?? -9) - (a.recursion_gain ?? -9));
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.model}</b></td><td class="num">${num(m.rsi_C)}</td><td class="num">${num(m.best_of_k)}</td>
    <td class="num">${num(m.self_refine)}</td><td class="num">${num(m.retrieval)}</td>
    <td class="num">${deltaBar(m.recursion_gain)}</td></tr>`).join("");
}).catch(e => {});

// Procedure RSI (Track C) board
fetch("data/trackc.json").then(r => r.json()).then(d => {
  const up = document.getElementById("tc-updated"); if (up) up.textContent = (d.updated || "");
  const tb = document.querySelector("#tc-lb tbody"); if (!tb) return;
  const ms = [...(d.models || [])].sort((a, b) => (b.delta_vs_base ?? -9) - (a.delta_vs_base ?? -9));
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.model}</b></td><td><span class="pill kind">${m.mode}</span></td>
    <td class="num">${num(m.rounds)}</td><td class="num">${num(m.Q0)}</td><td class="num">${num(m.Qg)}</td>
    <td class="num">${deltaBar(m.delta_vs_base)}</td>
    <td>${verdictPill(m.verdict === "improves" ? "compounds" : (m.verdict === "degrades" ? "overfits" : "flat"))}</td></tr>`).join("");
}).catch(e => {});

// Mechanism dense multi-curve panels
const MECH_COLORS = ["#6f1010", "#B1040E", "#8C1515", "#c0575f", "#d98a90", "#e0b0b0", "#a33", "#7a2222"];
function mechPanel(el, models, key, yminF, ymaxF) {
  if (!el) return;
  const W = 320, H = 180, pad = 28;
  const allv = models.flatMap(m => (m[key] || []).filter(v => typeof v === "number"));
  if (!allv.length) { el.innerHTML = ""; return; }
  let ymin = yminF !== undefined ? yminF : Math.min(0, ...allv);
  let ymax = ymaxF !== undefined ? ymaxF : Math.max(...allv);
  if (ymax === ymin) ymax = ymin + 1;
  const maxR = Math.max(...models.map(m => (m.rounds || []).length)) - 1 || 1;
  const X = r => pad + r / maxR * (W - pad - 8);
  const Y = v => H - pad - (v - ymin) / (ymax - ymin) * (H - pad - 10);
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  svg += `<line class="axis" x1="${pad}" y1="${H-pad}" x2="${W-4}" y2="${H-pad}"/><line class="axis" x1="${pad}" y1="8" x2="${pad}" y2="${H-pad}"/>`;
  if (ymin < 0) svg += `<line class="zero" x1="${pad}" y1="${Y(0)}" x2="${W-4}" y2="${Y(0)}"/>`;
  svg += `<text x="2" y="12" font-size="9" fill="#9a9a9a">${ymax.toFixed(2)}</text><text x="2" y="${H-pad}" font-size="9" fill="#9a9a9a">${ymin.toFixed(2)}</text>`;
  models.forEach((m, i) => {
    const pts = (m[key] || []).map((v, r) => (typeof v === "number") ? `${X(r)},${Y(v)}` : null).filter(Boolean).join(" ");
    if (pts) svg += `<polyline points="${pts}" stroke="${MECH_COLORS[i % MECH_COLORS.length]}"/>`;
  });
  svg += `</svg>`;
  el.innerHTML = svg;
}
fetch("data/mech.json").then(r => r.json()).then(d => {
  const up = document.getElementById("mech-updated"); if (up) up.textContent = (d.updated || "");
  const ms = d.models || [];
  mechPanel(document.getElementById("mech-C_self"), ms, "C_self", 0);
  mechPanel(document.getElementById("mech-diversity"), ms, "diversity", 0, 1);
  mechPanel(document.getElementById("mech-retention"), ms, "retention", 0, 1);
  mechPanel(document.getElementById("mech-self_minus_fresh"), ms, "self_minus_fresh");
  const leg = document.getElementById("mech-legend");
  if (leg) leg.innerHTML = ms.map((m, i) => `<span><i style="background:${MECH_COLORS[i % MECH_COLORS.length]}"></i>${m.model}</span>`).join("");
}).catch(e => {});

// Task 5 open-ended RSI panels
fetch("data/open_rsi.json").then(r => r.json()).then(d => {
  const up = document.getElementById("open-updated"); if (up) up.textContent = (d.updated || "");
  const ms = d.models || [];
  mechPanel(document.getElementById("open-C_frontier"), ms, "C_frontier", 0);
  mechPanel(document.getElementById("open-frontier"), ms, "frontier", 0);
  const leg = document.getElementById("open-legend");
  if (leg) leg.innerHTML = ms.map((m, i) => `<span><i style="background:${MECH_COLORS[i % MECH_COLORS.length]}"></i>${m.model} (${m.sustained||""})</span>`).join("");
}).catch(e => {});

// ---- sticky nav + scroll reveal ----
(function(){
  const nav = document.querySelector(".nav");
  const onScroll = ()=>{ if (nav) nav.classList.toggle("scrolled", window.scrollY > 8); };
  window.addEventListener("scroll", onScroll, {passive:true}); onScroll();
  const io = new IntersectionObserver((es)=>{ es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add("in"); io.unobserve(e.target); } }); }, {threshold:.12});
  document.querySelectorAll("section, figure, .track, .note").forEach((el,i)=>{ el.classList.add("reveal"); el.style.transitionDelay=(Math.min(i,6)*40)+"ms"; io.observe(el); });
})();
