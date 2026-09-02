// Leaderboard: load JSON, render, sortable. Add models by editing data/leaderboard.json.
const NUM = ["pass_at_k", "fast_1", "fast_1_5", "fast_2", "geomean_pass"];
let DATA = [], sortKey = "pass_at_k", sortDir = -1;

function fmt(v) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") return (Math.round(v * 1000) / 1000).toFixed(v < 1 ? 2 : 3);
  return v;
}

function render() {
  const tb = document.querySelector("#lb tbody");
  const rankable = DATA.filter(m => !/curator/i.test(m.role || ""));
  const best = {};
  NUM.forEach(k => { best[k] = Math.max(...rankable.map(m => m[k] ?? -Infinity)); });

  const rows = [...DATA].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (typeof av === "number" || typeof bv === "number")
      return ((av ?? -Infinity) - (bv ?? -Infinity)) * sortDir;
    return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
  });

  tb.innerHTML = rows.map(m => {
    const cur = /curator/i.test(m.role || "");
    const cell = (k) => {
      const isBest = !cur && NUM.includes(k) && m[k] === best[k] && best[k] > -Infinity;
      const cls = (NUM.includes(k) ? "num " : "") + (isBest ? "best" : "");
      return `<td class="${cls.trim()}">${fmt(m[k])}</td>`;
    };
    return `<tr class="${cur ? "curator" : ""}">
      <td><b>${m.model}</b></td><td>${m.org || "—"}</td><td>${m.params || "—"}</td><td>${m.role || "—"}</td>
      ${cell("pass_at_k")}${cell("fast_1")}${cell("fast_1_5")}${cell("fast_2")}${cell("geomean_pass")}
      <td class="muted small">${m.notes || ""}</td></tr>`;
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
