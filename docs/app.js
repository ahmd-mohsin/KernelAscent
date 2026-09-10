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
  wireSort(); render();
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
  const bar = `<span style="display:inline-block;height:8px;width:${w}%;background:${pos ? "#151515" : "#c9c9c9"};border-radius:2px;vertical-align:middle"></span>`;
  return `<span class="mono ${pos ? "pos" : "neg"}">${pos ? "+" : ""}${v.toFixed(3)}</span> ${bar}`;
}
fetch("data/rsi_leaderboard.json").then(r => r.json()).then(d => {
  const up = document.getElementById("rsi-updated"); if (up) up.textContent = (d.updated ? d.updated + " · " : "") + (d.bank || "");
  const tb = document.querySelector("#rsi-lb tbody");
  const ms = d.models || [];
  tb.innerHTML = ms.map(m => `<tr>
    <td><b>${m.model}</b></td><td><span class="pill kind">${m.family}</span></td>
    <td class="num">${num(m.rounds)}</td><td class="num">${num(m.C0)}</td><td class="num">${num(m.peak)}</td>
    <td class="num">${deltaBar(m.delta_frozen)}</td><td class="num mono">${num(m.self_minus_ctrl)}</td>
    <td>${verdictPill(m.verdict)}</td></tr>`).join("");
}).catch(e => {
  const tb = document.querySelector("#rsi-lb tbody"); if (tb) tb.innerHTML = `<tr><td colspan="8" class="mid">Serve over HTTP to load results.</td></tr>`;
});
