"""Generate monochrome (black & white) SVG figures for the KernelAscent site from the ACTUAL result
numbers in BENCHMARK_LOG.md. Pure stdlib; emits docs/figures/*.svg. Aesthetic: black ink on white,
hairline grey grid, thin strokes, smooth Bezier curves, CI whiskers -- no color."""
import os, math

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
INK, GRID, MID, PAPER = "#111111", "#e6e6e6", "#9a9a9a", "#ffffff"
FONT = "font-family='Georgia, \"Times New Roman\", serif'"
SANS = "font-family='ui-sans-serif, system-ui, -apple-system, Helvetica, Arial, sans-serif'"


def _hdr(w, h):
    return ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' width='100%%' "
            "role='img' style='max-width:%dpx;display:block;margin:auto'>"
            "<rect width='%d' height='%d' fill='%s'/>" % (w, h, w, w, h, PAPER))


def _smooth(pts):
    """Catmull-Rom -> cubic Bezier path through pts for a beautiful curve."""
    if len(pts) < 2:
        return ""
    d = "M %.1f %.1f" % pts[0]
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[0]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        d += " C %.1f %.1f %.1f %.1f %.1f %.1f" % (c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    return d


# ---------------------------------------------------------------- Fig 1: mechanism loop (develop / revise / fork -> F)
def fig_mechanism():
    w, h = 860, 300
    s = [_hdr(w, h)]
    def box(x, y, bw, bh, title, sub, dash=False):
        st = "stroke='%s' stroke-width='1.4' fill='%s'%s" % (INK, PAPER, " stroke-dasharray='5 4'" if dash else "")
        s.append("<rect x='%d' y='%d' width='%d' height='%d' rx='8' %s/>" % (x, y, bw, bh, st))
        s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='15' font-weight='700' fill='%s'>%s</text>" % (x + bw // 2, y + 24, SANS, INK, title))
        s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='11.5' fill='%s'>%s</text>" % (x + bw // 2, y + 43, SANS, MID, sub))
    def arrow(x1, y1, x2, y2, label=""):
        s.append("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='%s' stroke-width='1.4' marker-end='url(#a)'/>" % (x1, y1, x2, y2, INK))
        if label:
            s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='11' fill='%s'>%s</text>" % ((x1 + x2) // 2, min(y1, y2) - 7, SANS, INK, label))
    s.append("<defs><marker id='a' markerWidth='9' markerHeight='9' refX='7' refY='3' orient='auto'>"
             "<path d='M0,0 L7,3 L0,6 Z' fill='%s'/></marker></defs>" % INK)
    s.append("<text x='%d' y='28' %s font-size='16' font-weight='700' fill='%s'>The improvement loop &amp; the causal contrast</text>" % (24, SANS, INK))
    box(40, 70, 180, 60, "develop(U)", "run procedure U under budget → Q")
    box(300, 70, 200, 60, "revise(actor, target)", "actor edits a COPY of target → child")
    box(600, 70, 220, 60, "common-target fork", "U₁ and U₀ edit the SAME target")
    arrow(220, 100, 300, 100)
    arrow(500, 100, 600, 100)
    # fork detail
    box(560, 190, 130, 54, "U₂ = revise(U₁, T)", "newer producer", dash=True)
    box(710, 190, 130, 54, "V₂ = revise(U₀, T)", "older producer", dash=True)
    arrow(660, 130, 625, 190)
    arrow(760, 130, 775, 190)
    s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='13' font-weight='700' fill='%s'>F = Q(U₂) − Q(V₂)  (does the newer producer build a better child?)</text>" % (700, 280, SANS, INK))
    # left annotation
    s.append("<text x='40' y='170' %s font-size='12.5' fill='%s'>N = Q(child) − Q(target): does the child beat keeping the target?</text>" % (SANS, MID))
    s.append("<text x='40' y='190' %s font-size='12.5' fill='%s'>q₁−q₀: first-order improvement from one self-revision.</text>" % (SANS, MID))
    s.append("</svg>")
    open(os.path.join(OUT, "mechanism.svg"), "w").write("".join(s))


# ---------------------------------------------------------------- Fig 2: causal decomposition forest plot (with CIs)
def fig_forest():
    # (label, mean, lo, hi)  -- opus-5 live-model Gate-4 (n=40), the headline decomposition
    rows = [("q₁−q₀  first-order improvement", 0.298, 0.278, 0.347),
            ("N₁  child beats target", 0.043, 0.007, 0.080),
            ("F₁  causal producer link", 0.019, -0.013, 0.051),
            ("F₂  compounding (repeat)", -0.019, -0.058, 0.020)]
    w, h = 760, 300
    L, R, T = 300, 720, 70
    lo_ax, hi_ax = -0.1, 0.36
    def X(v): return L + (v - lo_ax) / (hi_ax - lo_ax) * (R - L)
    s = [_hdr(w, h)]
    s.append("<text x='24' y='30' %s font-size='16' font-weight='700' fill='%s'>Causal decomposition (opus-5, n=40 lineages, 95%% CI)</text>" % (SANS, INK))
    s.append("<text x='24' y='50' %s font-size='12' fill='%s'>first-order improvement is resolved; the causal producer link and compounding are not</text>" % (SANS, MID))
    # zero + delta lines
    for v, lab, dash in ((0.0, "0", "0"), (0.05, "δ = 0.05 (meaningful effect)", "4 4")):
        s.append("<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='%s' stroke-width='1' stroke-dasharray='%s'/>" % (X(v), T - 10, X(v), T + len(rows) * 46, MID if v else INK, dash))
        s.append("<text x='%.1f' y='%d' text-anchor='middle' %s font-size='10.5' fill='%s'>%s</text>" % (X(v), T + len(rows) * 46 + 16, SANS, MID, lab))
    for i, (lab, m, lo, hi) in enumerate(rows):
        y = T + 18 + i * 46
        resolved = lo > 0 or hi < 0
        s.append("<text x='%d' y='%d' text-anchor='end' %s font-size='12.5' fill='%s'>%s</text>" % (L - 16, y + 4, SANS, INK, lab))
        s.append("<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='%s' stroke-width='1.6'/>" % (X(lo), y, X(hi), y, INK))
        for xx in (X(lo), X(hi)):
            s.append("<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='%s' stroke-width='1.6'/>" % (xx, y - 5, xx, y + 5, INK))
        s.append("<circle cx='%.1f' cy='%d' r='5' fill='%s' stroke='%s' stroke-width='1.5'/>" % (X(m), y, INK if resolved else PAPER, INK))
        s.append("<text x='%.1f' y='%d' %s font-size='11' fill='%s'>%+.3f</text>" % (X(hi) + 8, y + 4, SANS, INK, m))
    s.append("<text x='%d' y='%d' %s font-size='10.5' fill='%s'>● filled = CI excludes 0 (resolved)   ○ open = spans 0</text>" % (L - 16 + 0, T + len(rows) * 46 + 34, SANS, MID))
    s.append("</svg>")
    open(os.path.join(OUT, "forest.svg"), "w").write("".join(s))


# ---------------------------------------------------------------- Fig 3: procedure value vs deployment budget (curves)
def fig_budget():
    # (name, {budget: Q}, style)
    series = [("good procedure (built directly)", {30: 0.779, 60: 0.854, 90: 0.90}, "solid"),
              ("direct brute-force search", {30: 0.598, 60: 0.708, 90: 0.80}, "dash"),
              ("iterative self-improvement", {30: 0.558, 60: 0.671, 90: 0.74}, "dot"),
              ("baseline (no improvement)", {30: 0.508, 60: 0.522, 90: 0.53}, "thin")]
    w, h = 760, 380
    L, R, T, B = 70, 720, 60, 320
    bx = [30, 60, 90]
    ylo, yhi = 0.45, 0.95
    def X(b): return L + (b - 25) / (95 - 25) * (R - L)
    def Y(q): return B - (q - ylo) / (yhi - ylo) * (B - T)
    s = [_hdr(w, h)]
    s.append("<text x='24' y='30' %s font-size='16' font-weight='700' fill='%s'>Procedure value vs deployment budget</text>" % (SANS, INK))
    s.append("<text x='24' y='48' %s font-size='12' fill='%s'>the good procedure's edge over brute-force widens as budget tightens</text>" % (SANS, MID))
    # grid + y labels
    for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
        s.append("<line x1='%d' y1='%.1f' x2='%d' y2='%.1f' stroke='%s' stroke-width='1'/>" % (L, Y(q), R, Y(q), GRID))
        s.append("<text x='%d' y='%.1f' text-anchor='end' %s font-size='10.5' fill='%s'>%.1f</text>" % (L - 8, Y(q) + 3, SANS, MID, q))
    for b in bx:
        s.append("<text x='%.1f' y='%d' text-anchor='middle' %s font-size='10.5' fill='%s'>%d</text>" % (X(b), B + 18, SANS, MID, b))
    s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='11.5' fill='%s'>deployment budget (eval units)</text>" % ((L + R) // 2, B + 40, SANS, INK))
    s.append("<text x='18' y='%d' text-anchor='middle' %s font-size='11.5' fill='%s' transform='rotate(-90 18 %d)'>research productivity Q</text>" % ((T + B) // 2, SANS, INK, (T + B) // 2))
    dash_map = {"solid": "", "dash": " stroke-dasharray='7 5'", "dot": " stroke-dasharray='2 4'", "thin": " stroke-dasharray='1 3'"}
    wid_map = {"solid": 2.4, "dash": 1.8, "dot": 1.8, "thin": 1.2}
    for name, d, st in series:
        pts = [(X(b), Y(d[b])) for b in bx]
        s.append("<path d='%s' fill='none' stroke='%s' stroke-width='%.1f'%s/>" % (_smooth(pts), INK, wid_map[st], dash_map[st]))
        for p in pts:
            s.append("<circle cx='%.1f' cy='%.1f' r='3' fill='%s'/>" % (p[0], p[1], INK))
        lx, ly = pts[-1]
        s.append("<text x='%.1f' y='%.1f' %s font-size='11' fill='%s'>%s</text>" % (lx - 6, ly - 8, SANS, INK, name))
    s.append("</svg>")
    open(os.path.join(OUT, "budget.svg"), "w").write("".join(s))


# ---------------------------------------------------------------- Fig 4: capability gradient + coadaptation (bars)
def fig_bars():
    w, h = 760, 320
    s = [_hdr(w, h)]
    s.append("<text x='24' y='30' %s font-size='16' font-weight='700' fill='%s'>First-order improvement is capability-graded</text>" % (SANS, INK))
    # left: q1-q0 by model
    models = [("opus-5", 0.298), ("gpt-oss-120b", 0.098), ("deepseek", 0.021)]
    L, T, BW, GAP, base = 60, 70, 70, 40, 250
    mx = 0.34
    for i, (m, v) in enumerate(models):
        x = L + i * (BW + GAP); bh = v / mx * 170
        s.append("<rect x='%d' y='%.1f' width='%d' height='%.1f' fill='%s'/>" % (x, base - bh, BW, bh, INK))
        s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='11' fill='%s'>%s</text>" % (x + BW // 2, base + 16, SANS, INK, m))
        s.append("<text x='%d' y='%.1f' text-anchor='middle' %s font-size='11' fill='%s'>+%.2f</text>" % (x + BW // 2, base - bh - 6, SANS, INK, v))
    s.append("<text x='%d' y='%d' %s font-size='11.5' fill='%s'>q₁−q₀ (procedure gain from experience)</text>" % (L, T - 16, SANS, MID))
    # right: coadaptation 2x2
    RL = 440; cells = [("P₀J₀", 0.913), ("P_gJ₀", 0.913), ("P₀J_g", 0.923), ("P_gJ_g", 0.969)]
    cy0, cymin, cymax = 250, 0.90, 0.98
    s.append("<text x='%d' y='%d' %s font-size='11.5' fill='%s'>coadaptation: only proposal × judgment TOGETHER helps</text>" % (RL, T - 16, SANS, MID))
    for i, (lab, v) in enumerate(cells):
        x = RL + i * 60; bh = (v - cymin) / (cymax - cymin) * 170
        fill = INK if lab == "P_gJ_g" else MID
        s.append("<rect x='%d' y='%.1f' width='44' height='%.1f' fill='%s'/>" % (x, cy0 - bh, bh, fill))
        s.append("<text x='%d' y='%d' text-anchor='middle' %s font-size='10.5' fill='%s'>%s</text>" % (x + 22, cy0 + 16, SANS, INK, lab))
        s.append("<text x='%d' y='%.1f' text-anchor='middle' %s font-size='10' fill='%s'>%.2f</text>" % (x + 22, cy0 - bh - 6, SANS, INK, v))
    s.append("<text x='%d' y='%d' %s font-size='10.5' fill='%s'>interaction +0.045 (the compounding channel)</text>" % (RL, cy0 + 40, SANS, INK))
    s.append("</svg>")
    open(os.path.join(OUT, "bars.svg"), "w").write("".join(s))


if __name__ == "__main__":
    fig_mechanism(); fig_forest(); fig_budget(); fig_bars()
    print("wrote:", ", ".join(sorted(os.listdir(OUT))))
