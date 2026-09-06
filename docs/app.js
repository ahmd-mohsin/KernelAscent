// Leaderboard: load JSON, render, sortable. Add models by editing data/leaderboard.json.
const NUM = ["correct_rate", "fast_rate", "meanC"];
let DATA = [], sortKey = "meanC", sortDir = -1;

function fmt(v) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") return (Math.round(v * 1000) / 1000).toFixed(v < 1 ? 3 : 3);
  return v;
}
// "0.360 [0.26, 0.47]" when a CI is present
function fmtci(v, ci) {
  if (v === null || v === undefined) return "—";
  const base = fmt(v);
  return (Array.isArray(ci) && ci.length === 2) ? `${base} <span class="muted small">[${fmt(ci[0])}, ${fmt(ci[1])}]</span>` : base;
}

function render() {
  const tb = document.querySelector("#lb tbody");
  const best = {};
  NUM.forEach(k => { best[k] = Math.max(...DATA.map(m => m[k] ?? -Infinity)); });

  const rows = [...DATA].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (typeof av === "number" || typeof bv === "number")
      return ((av ?? -Infinity) - (bv ?? -Infinity)) * sortDir;
    return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
  });

  const ciKey = { correct_rate: "correct_ci", fast_rate: "fast_ci" };
  tb.innerHTML = rows.map(m => {
    const cell = (k) => {
      const isBest = m[k] === best[k] && best[k] > -Infinity;
      return `<td class="num ${isBest ? "best" : ""}">${ciKey[k] ? fmtci(m[k], m[ciKey[k]]) : fmt(m[k])}</td>`;
    };
    return `<tr>
      <td><b>${m.model}</b></td><td>${m.kind || "—"}</td>
      ${cell("correct_rate")}${cell("fast_rate")}${cell("meanC")}</tr>`;
  }).join("");
}

function wireSort() {
  document.querySelectorAll("#lb th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (k === sortKey) sortDir *= -1; else { sortKey = k; sortDir = NUM.includes(k) ? -1 : 1; }
    render();
  }));
}

fetch("data/leaderboard.json").then(r => r.json()).then(d => {
  DATA = d.models || [];
  document.getElementById("lb-updated").textContent = "updated " + (d.updated || "");
  document.getElementById("lb-note").textContent = d.metric_note || "";
  wireSort(); render();
}).catch(e => {
  document.querySelector("#lb tbody").innerHTML =
    `<tr><td colspan="10" class="muted">Could not load leaderboard.json (${e}). Serve over HTTP.</td></tr>`;
});

// Self-improvement loop leaderboards (RSI + scaffold-RSI) share a schema
function fmtn(v){ return (v===null||v===undefined||v==="") ? "—" : (typeof v==="number" ? (Math.round(v*1000)/1000) : v); }
function renderLoop(file, tableId, noteId){
  fetch(file).then(r => r.json()).then(d => {
    const note = document.getElementById(noteId);
    if (note) note.textContent = (d.status ? d.status + " " : "") + (d.metric_note || "");
    const tb = document.querySelector("#" + tableId + " tbody");
    const ms = d.models || [];
    if (!ms.length) { tb.innerHTML = `<tr><td colspan="8" class="muted">${d.status || "Results pending."}</td></tr>`; return; }
    tb.innerHTML = ms.map(m => `<tr>
      <td><b>${m.model}</b></td><td>${m.params||"—"}</td><td class="num">${fmtn(m.rounds)}</td>
      <td class="num">${fmtn(m.capability_r0)}</td><td class="num">${fmtn(m.capability_rN)}</td>
      <td class="num">${fmtn(m.compounding_b)}</td><td class="num">${fmtn(m.delta_final)}</td>
      <td>${m.verdict||"—"}</td></tr>`).join("");
  }).catch(e => {
    const tb = document.querySelector("#" + tableId + " tbody");
    if (tb) tb.innerHTML = `<tr><td colspan="8" class="muted">Could not load ${file} (${e}).</td></tr>`;
  });
}
renderLoop("data/rsi_leaderboard.json", "rsi-lb", "rsi-note");
renderLoop("data/scaffold_rsi_leaderboard.json", "scaffold-lb", "scaffold-note");
