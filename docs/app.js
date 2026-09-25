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
// Counts are integers: rendering a round count as "6.000" or an authored-task count as "1.000"
// reads like a measurement with three significant figures when it is a tally.
function cnt(v){ return (v===null||v===undefined||v==="") ? "—" : (typeof v==="number" ? String(Math.round(v)) : v); }
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
    <td class="num">${cnt(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.Cfinal)}</td>
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
    <td class="num">${cnt(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.C_self)}</td>
    <td class="num">${deltaBar(m.self_minus_fresh)}</td><td>${verdictPill(m.verdict)}</td></tr>`).join("");
}).catch(e => {});

// Task 4 (closed->open harness rewrite) board — sorted by peak desc
fetch("data/combined_rsi.json").then(r => r.json()).then(d => {
  const up = document.getElementById("c3-updated"); if (up) up.textContent = (d.updated ? d.updated : "");
  const tb = document.querySelector("#rsi3-lb tbody"); if (!tb) return;
  const ms = [...(d.models || [])].sort((a, b) => (b.peak ?? -9) - (a.peak ?? -9));
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.researcher}</b></td><td><span class="pill kind">${m.trainee}</span></td>
    <td class="num">${cnt(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.C_improved)}</td>
    <td class="num">${num(m.C_frozen)}</td><td class="num">${deltaBar(m.impr_minus_frozen)}</td><td class="num mono">${num(m.peak)}</td>
    <td>${verdictPill(m.verdict==="harness helps"?"compounds":(m.verdict==="hurts"?"overfits":"flat"))}</td></tr>`).join("");
}).catch(e => {});

// Task 5 self-play dense boards (open weight + closed API), sorted by final L-F desc
// L-F is only defined when the LIVE author actually produced accepted, valid, novel tasks.
// Below this floor a row is an author-yield failure, not a measurement -- so it must not render
// as a green "co-evolves" pill (the largest L-F on the open board rests on ONE accepted task).
const MIN_AUTHORED = 5;
function selfplayVerdict(v, authored){
  if ((authored || 0) < MIN_AUTHORED)
    return `<span class="pill flat" title="Only ${authored||0} accepted model-authored task(s): the live author never produced a frontier, so L−F is undefined rather than measured.">undefined · author yield</span>`;
  v = (v || "").toLowerCase();
  if (v.includes("co-evolution")) return `<span class="pill good">co-evolves</span>`;
  if (v.includes("adaptive")) return `<span class="pill gain">curriculum only</span>`;
  return `<span class="pill flat">no compounding</span>`;
}
function lastNum(a){ if(!Array.isArray(a)) return a; for(let i=a.length-1;i>=0;i--){ if(typeof a[i]==="number") return a[i]; } return null; }
function selfplayBoard(file, tbSel, upId){
  fetch(file).then(r => r.json()).then(d => {
    const up = document.getElementById(upId); if (up) up.textContent = (d.updated || "");
    const tb = document.querySelector(tbSel); if (!tb) return;
    const ms = [...(d.models || [])].sort((a, b) => (b.final_L_minus_F ?? -9) - (a.final_L_minus_F ?? -9));
    if (!ms.length){ tb.innerHTML = `<tr><td colspan="10" class="mid">Runs in progress — rows populate as rounds land.</td></tr>`; return; }
    tb.innerHTML = ms.map(m => `<tr>
      <td><b>${m.model}</b></td><td><span class="pill kind">${m.tier}</span></td>
      <td class="num">${cnt((m.rounds||[]).length)}</td>
      <td class="num">${num(lastNum(m.C_held_live))}</td><td class="num">${num(lastNum(m.C_held_frozen_author))}</td><td class="num">${num(lastNum(m.C_held_static))}</td>
      <td class="num">${(m.total_model_proposed||0) < MIN_AUTHORED
          ? `<span class="undef" title="not interpretable at this author yield">${num(m.final_L_minus_F)}</span>`
          : deltaBar(m.final_L_minus_F)}</td><td class="num mono">${num(lastNum(m.F_minus_S))}</td>
      <td class="num mono">${cnt(m.total_model_proposed)}</td><td>${selfplayVerdict(m.verdict, m.total_model_proposed)}</td></tr>`).join("");
  }).catch(e => {
    const tb = document.querySelector(tbSel); if (tb) tb.innerHTML = `<tr><td colspan="10" class="mid">Serve over HTTP to load results.</td></tr>`;
  });
}
selfplayBoard("data/selfplay.json", "#sp-open-lb tbody", "sp-open-updated");
selfplayBoard("data/selfplay_closed.json", "#sp-closed-lb tbody", "sp-closed-updated");

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
    <td class="num">${cnt(m.rounds)}</td><td class="num">${num(m.Q0)}</td><td class="num">${num(m.Qg)}</td>
    <td class="num">${deltaBar(m.delta_vs_base)}</td>
    <td>${(m.rounds||0) < 3
        ? `<span class="pill flat" title="Only ${cnt(m.rounds)} completed round(s): a genuine ceiling or self-degradation cannot be separated from an interrupted run.">not interpreted · ${cnt(m.rounds)}r</span>`
        : (m.verdict === "improves"
            ? `<span class="pill gain" title="Q-gain vs the frozen procedure. Mostly one-shot: ~76% of the gain arrives at round 0 and the archive saturates by round 1-2, so this is an improvement, not demonstrated compounding.">improves (one-shot)</span>`
            : verdictPill(m.verdict === "degrades" ? "overfits" : "flat"))}</td></tr>`).join("");
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
  mechPanel(document.getElementById("open-C_frontier"), ms, "delta_open_minus_fixed");
  mechPanel(document.getElementById("open-frontier"), ms, "base_correct_on_frontier", 0, 1);
  const leg = document.getElementById("open-legend");
  if (leg) leg.innerHTML = ms.map((m, i) => `<span><i style="background:${MECH_COLORS[i % MECH_COLORS.length]}"></i>${m.model} (${m.verdict||""})</span>`).join("");
}).catch(e => {});

// ---- sticky nav + scroll reveal ----
(function(){
  const nav = document.querySelector(".nav");
  const onScroll = ()=>{ if (nav) nav.classList.toggle("scrolled", window.scrollY > 8); };
  window.addEventListener("scroll", onScroll, {passive:true}); onScroll();
  const io = new IntersectionObserver((es)=>{ es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add("in"); io.unobserve(e.target); } }); }, {threshold:.12});
  document.querySelectorAll("section, figure, .track, .note").forEach((el,i)=>{ el.classList.add("reveal"); el.style.transitionDelay=(Math.min(i,6)*40)+"ms"; io.observe(el); });
})();

// Community board — data/leaderboard_community.json (built by the submission bot)
(function communityBoard(){
  const TRACK_LABEL = { capability:"Capability", weight_rsi:"Weight-RSI", procedure_rsi:"Procedure-RSI",
                        closed_open:"Closed→Open", selfplay:"Self-play", harness:"Harness" };
  const HEADLINE = { capability:["meanC","mean C"], weight_rsi:["lineage_minus_reset","lineage−reset"],
                     procedure_rsi:["Q_gain_vs_frozen","Q gain"], closed_open:["peak","peak"],
                     selfplay:["L_minus_F","L−F"], harness:["improved_minus_frozen","improved−frozen"] };
  fetch("data/leaderboard_community.json").then(r=>r.json()).then(d=>{
    const tb = document.querySelector("#comm-lb tbody"); if(!tb) return;
    const up = document.getElementById("comm-updated"); if(up) up.textContent = "updated "+(d.updated||"");
    const rows = [];
    Object.keys(TRACK_LABEL).forEach(t=>{
      (d.tracks && d.tracks[t] || []).forEach(m=>{
        const [k,lbl] = HEADLINE[t];
        const val = (typeof m[k]==="number") ? (Math.round(m[k]*1000)/1000).toFixed(3) : "—";
        const st = m.state || (m.verified ? "verified" : "self-reported");
        const status = st==="verified" ? `<span class="state verified">✓ held-out verified</span>`
                     : st==="reproduced" ? `<span class="state repro">↻ reproduced</span>`
                     : `<span class="state selfrep">self-reported</span>`;
        const kind = (m.kind==="open_weight") ? "open weight" : (m.kind==="api" ? "closed / API" : m.kind||"");
        rows.push(`<tr><td>${TRACK_LABEL[t]}</td><td><b>${m.model||"?"}</b></td><td><span class="pill kind">${kind}</span></td>`
          + `<td class="num">${val} <span class="mid small">${lbl}</span></td><td>${status}</td><td class="small">${m.submitter||""}</td></tr>`);
      });
    });
    tb.innerHTML = rows.length ? rows.join("")
      : `<tr><td colspan="6" class="mid">No community submissions yet — <a href="https://github.com/ahmd-mohsin/KernelAscent/blob/main/submissions/README.md">be the first</a>.</td></tr>`;
  }).catch(()=>{ const tb=document.querySelector("#comm-lb tbody"); if(tb) tb.innerHTML=`<tr><td colspan="6" class="mid">Serve over HTTP to load the community board.</td></tr>`; });
})();

/* ---------------------------------------------------------------------------
   Headline results, rendered from data/headline.json.

   These four numbers are the page's lead, and every one of them used to be
   typed into index.html by hand. That is the practice this project has already
   been burned by: a figure typed into prose and propagated by hand turned out
   to be unreproducible from any artifact, and the page went on showing it after
   the paper had stopped. scripts/build_site_headline.py re-derives them from
   the stored artifacts through the same module the paper's tables use, so the
   site cannot report a number the report does not.

   Failure is loud on purpose. A headline section that silently renders nothing
   looks like a design choice, and a reader cannot tell a missing result from an
   absent one.
--------------------------------------------------------------------------- */
(function () {
  const FMT = {
    signed3: v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3),
    signed4: v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4),
    f2: v => v.toFixed(2),
    f3: v => v.toFixed(3),
    pct0: v => Math.round(v * 100) + "%",
    int: v => String(Math.round(v)),
  };
  const dig = (o, path) => path.split(".").reduce((a, k) => (a == null ? a : a[k]), o);

  function fill(root, data) {
    root.querySelectorAll("[data-h]").forEach(el => {
      const v = dig(data, el.dataset.h);
      el.textContent = (typeof v === "number") ? (FMT[el.dataset.fmt] || FMT.f3)(v) : "—";
    });
  }

  function cards(d) {
    const c = d.contrast, s = d.starvation, n = d.nonllm;
    const out = [];
    if (c) out.push([FMT.signed4(c.headroom_mean), "headroom contrast",
      `mean over the ${c.n_parity_rounds} rounds where both arms sit at correct-at-parity; never further from zero than ${c.headroom_max_abs.toFixed(3)}`]);
    if (c) out.push([FMT.signed3(c.passrate_mean), "the same rounds, rescored",
      `pass rate on the identical generations, minimum ${FMT.signed3(c.passrate_min)}. At their closest the two ranges are a factor of ${Math.round(c.range_separation)} apart, and they do not approach one another.`]);
    if (s) out.push([FMT.f2(s.mean_per_round), "examples per round",
      `what the registered weight-RSI loop actually received, over ${s.n_rounds} rounds; ${s.n_empty} of them empty. Predicted in advance by n_train × k × yield = ${s.predicted.expected.toFixed(2)}.`]);
    if (n) out.push([FMT.f3(n.median_score), "non-LLM baseline",
      `torch.compile(mode="max-autotune"), scored as a model submission is — below correct-at-parity, because it is slower than default torch.compile on ${n.n_slower_than_compiled} of ${n.n_tasks} tasks.`]);
    return out.map(([big, lab, note]) =>
      `<div class="card"><div class="card-n">${big}</div><div class="card-l">${lab}</div><div class="card-note">${note}</div></div>`).join("");
  }

  function contrastRows(c) {
    return c.rows.map(r => {
      const dag = r.both_at_parity ? '<span class="dag">†</span>' : "";
      const pv = (typeof r.passrate === "number") ? FMT.signed3(r.passrate) : "—";
      const cls = r.both_at_parity ? ' class="parity"' : "";
      return `<tr${cls}><td class="mono">${r.cell.replace("t2kc_", "")}</td><td class="num">${r.round}</td>`
        + `<td class="num">${r.C_lineage.toFixed(2)} / ${r.C_reset.toFixed(2)}</td>`
        + `<td class="num">${FMT.signed3(r.headroom)}${dag}</td>`
        + `<td class="num"><b>${pv}</b></td></tr>`;
    }).join("");
  }

  fetch("data/headline.json").then(r => r.json()).then(d => {
    const sec = document.getElementById("headline");
    if (sec) fill(sec, d);
    fill(document.querySelector(".hero"), d);

    const cw = document.getElementById("hl-cards");
    if (cw) cw.innerHTML = cards(d);

    const ct = document.querySelector("#hl-contrast tbody");
    if (ct && d.contrast) {
      ct.innerHTML = contrastRows(d.contrast);
      const note = document.getElementById("hl-contrast-note");
      const c = d.contrast;
      if (note) note.innerHTML = `Across all ${c.n_parity_rounds} rounds marked † the headroom contrast stays `
        + `within ${c.headroom_max_abs.toFixed(3)} of zero, mean <b>${FMT.signed4(c.headroom_mean)}</b>, while the same rounds `
        + `report a mean of <b>${FMT.signed3(c.passrate_mean)}</b> under pass rate with a minimum of `
        + `${FMT.signed3(c.passrate_min)}. The two boards are never <i>pooled</i> (pre-registration Amendment 1); `
        + `they are placed side by side here to compare the scorers, which is a different act.`;
    }

    const nt = document.querySelector("#hl-nonllm tbody");
    if (nt && d.nonllm) {
      const t = d.nonllm.by_tier || {};
      nt.innerHTML = Object.keys(t).map(k =>
        `<tr><td>${k}</td><td class="num">${t[k].n}</td><td class="num">${t[k].median_score.toFixed(3)}</td></tr>`).join("")
        + `<tr class="total"><td>All</td><td class="num">${d.nonllm.n_tasks}</td><td class="num"><b>${d.nonllm.median_score.toFixed(3)}</b></td></tr>`;
      const nn = document.getElementById("hl-nonllm-note");
      if (nn) nn.innerHTML = `Median speedup against the compiled baseline is `
        + `<b>${d.nonllm.median_speedup_vs_compiled.toFixed(3)}×</b>, and max-autotune is slower than default `
        + `<code>torch.compile</code> on <b>${d.nonllm.n_slower_than_compiled} of ${d.nonllm.n_tasks}</b> tasks. `
        + `A model matching the compiled baseline is therefore outperforming production autotuning on this bank, `
        + `not failing to beat a weak one. Reported per tier because compile gain over eager differs by about 12× between L1 and L2.`;
    }
  }).catch(() => {
    const cw = document.getElementById("hl-cards");
    if (cw) cw.innerHTML = `<div class="card"><div class="card-l">Headline data did not load.</div>`
      + `<div class="card-note">Serve this page over HTTP, or read the numbers in `
      + `<a href="data/headline.json">data/headline.json</a>. They are not typed into this page, so nothing is shown rather than something stale.</div></div>`;
  });
})();
