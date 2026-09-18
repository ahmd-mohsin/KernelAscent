## 1. Design direction: an animated research instrument, not a glowing flowchart

The biggest improvement will come from **hierarchy and spatial density**, not adding more glow.

I would build the page around:

1. **A large five-stage pipeline** explaining the benchmark.
2. **A dominant, nearly full-width failure-path diagram** showing where models progress or stop.
3. **A large scatter plot with integrated marginals** connecting scale, drift, and gain.
4. **Three quieter diagnostic panels** for coverage, geometry, and failure taxonomy.

Use motion to explain direction, progression, and correspondence:

- Tokens travel **forward along measured paths**.
- Ribbons reveal **in pipeline order**.
- Hovering a model lights up **the same model across figures**.
- Failed runs **stop at the recorded gate**, rather than merely changing color.
- Selected nodes emit a restrained pulse as tokens arrive.

**Do not animate everything independently.** Dense motion looks good when it has a common rhythm and semantic structure. Random pulsing, perpetual bubble movement, and large blurred glows will make the page harder to read.

---

# 2. Exact layout and visual system

## Figure sizing

Use a substantially wider content region than a typical documentation page:

```css
:root {
  color-scheme: dark;

  --page-bg: #080d16;
  --panel-bg: #0d1624;
  --panel-raised: #121f31;

  --ink: #edf3ff;
  --ink-muted: #a6b7cf;
  --grid: #233248;

  --cyan: #54d8ff;
  --violet: #ae93ff;
  --green: #65dfb0;
  --amber: #f4c16b;
  --red: #f48592;

  --figure-max: 1560px;
}

body {
  margin: 0;
  background: var(--page-bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

.viz-section {
  width: min(var(--figure-max), calc(100% - 48px));
  margin: 64px auto;
}

.viz-heading {
  max-width: 900px;
  margin: 0 auto 24px;
  text-align: center;
}

.viz-heading h2 {
  margin: 0 0 10px;
  font-size: clamp(26px, 3vw, 40px);
  line-height: 1.1;
  letter-spacing: -0.035em;
}

.viz-heading p {
  color: var(--ink-muted);
  line-height: 1.6;
}

.viz-frame {
  position: relative;
  overflow: hidden;
  border: 1px solid #26374d;
  border-radius: 20px;
  background:
    radial-gradient(
      ellipse at 50% 0%,
      rgb(84 216 255 / 6%),
      transparent 65%
    ),
    var(--panel-bg);
}

.viz-svg {
  display: block;
  width: 100%;
  height: auto;
  margin-inline: auto;
}

.viz-caption,
.viz-legend {
  max-width: 1000px;
  margin: 14px auto 0;
  text-align: center;
}

.viz-caption {
  color: var(--ink-muted);
  font-size: 14px;
  line-height: 1.6;
}
```

Recommended desktop coordinate systems:

| Figure | SVG `viewBox` | Approximate rendered height at 1440px width |
|---|---|---:|
| Pipeline | `0 0 1440 620` | 620px |
| Failure-path DAG | `0 0 1440 820` | 820px |
| Scatter + marginals | `0 0 1440 900` | 900px |
| Coverage | `0 0 1000 500` | 500px |
| Geometry | `0 0 1000 520` | 520px |
| Failure taxonomy | `0 0 1000 520` | 520px |

These should be major page sections, **not 300px-high widgets**.

### Centering rule

Center the **figure, heading, legends, and diagram composition**. Do not center every text line inside every card.

- Card titles: left aligned.
- Card descriptions: left aligned.
- Stage number and main metric: optionally centered.
- DAG gate headers: centered over their column.
- Legends and control bars: centered.

This produces a centered composition without sacrificing readability.

## Typography inside SVG

For a desktop `viewBox` width of 1440:

```css
.viz-svg text {
  fill: var(--ink);
  font-family: inherit;
}

.viz-svg .stage-title {
  font-size: 24px;
  font-weight: 650;
  letter-spacing: -0.025em;
}

.viz-svg .body-label {
  font-size: 16px;
  fill: var(--ink-muted);
}

.viz-svg .axis-label {
  font-size: 16px;
  fill: var(--ink-muted);
}

.viz-svg .metric {
  font-size: 34px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.viz-svg .mono {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
```

Do not rely on 10–12px labels to make a diagram “detailed.” Detail should come from **more room and clearer structure**.

## Color semantics

Keep outcome colors consistent across all figures:

- **RSI:** green.
- **Crossed but flat:** amber.
- **Stuck:** rose/red.
- **Selected model/path:** cyan outline or halo.
- **Procedure/task identity:** violet or neutral blue.
- **Unknown/not evaluated:** gray, optionally dashed.

If the taxonomy uses green for `correct`, label it explicitly: **correctness is not the same as RSI success**.

Use text, symbols, or line patterns in addition to color.

---

# 3. Motion architecture: CSS + one shared `requestAnimationFrame` loop

My recommendation:

| Effect | Best implementation |
|---|---|
| Moving dashes along edges | CSS `stroke-dashoffset` |
| One-time edge reveal | CSS + normalized `pathLength` |
| Node arrival pulse | CSS on a separate halo |
| A few fixed-route tokens | SMIL `<animateMotion>` |
| Many data-driven particles | One shared `requestAnimationFrame` loop |
| Selected-path emphasis | CSS classes |
| Model-dependent flow speed/count | JS |
| Genuine ribbon geometry | SVG filled paths, computed in JS |

**Do not use CSS motion paths as the primary mechanism.** SVG path reuse and coordinate handling are more straightforward with `getPointAtLength()` or `<animateMotion>`.

### A useful motion budget

For a typical desktop:

- 8–15 small moving edge patterns.
- 30–70 visible token particles in the DAG.
- 2–4 tokens per prominent pipeline connector.
- At most 1–3 pulsing nodes at once.
- No perpetual movement of scatter-point positions.
- No animated blur filters on large ribbon layers.

Animate only figures currently visible.

---

# 4. SVG pattern: layered, flowing edges

Use three edge layers:

1. A low-opacity structural track.
2. A broad colored route.
3. A thin, moving dashed highlight.

The broad route carries the relationship; the dashes carry the motion.

```html
<svg
  class="viz-svg"
  viewBox="0 0 1440 620"
  role="img"
  aria-labelledby="pipeline-title pipeline-desc"
>
  <title id="pipeline-title">KernelAscent RSI pipeline</title>
  <desc id="pipeline-desc">
    Five benchmark stages from capability through self-play.
    Animated connectors indicate stage order.
  </desc>

  <defs>
    <!-- Prefix IDs per figure: SVG IDs are document-global. -->
    <linearGradient
      id="pipeline-edge-gradient"
      gradientUnits="userSpaceOnUse"
      x1="288" y1="0"
      x2="328" y2="0"
    >
      <stop offset="0%" stop-color="#54d8ff"/>
      <stop offset="100%" stop-color="#ae93ff"/>
    </linearGradient>

    <marker
      id="pipeline-arrow"
      viewBox="0 0 10 10"
      refX="8" refY="5"
      markerWidth="6" markerHeight="6"
      orient="auto"
      markerUnits="userSpaceOnUse"
    >
      <path d="M 1 1 L 9 5 L 1 9 Z" fill="#ae93ff"/>
    </marker>

    <path
      id="pipeline-route-12"
      d="M288 280 C302 280 314 280 328 280"
    />
  </defs>

  <g aria-hidden="true">
    <use
      href="#pipeline-route-12"
      class="edge-track"
    />

    <use
      href="#pipeline-route-12"
      class="edge-color"
      stroke="url(#pipeline-edge-gradient)"
      marker-end="url(#pipeline-arrow)"
    />

    <use
      href="#pipeline-route-12"
      class="edge-flow"
    />
  </g>

  <!-- Cards render above edges. -->
</svg>
```

```css
.edge-track,
.edge-color,
.edge-flow {
  fill: none;
  stroke-linecap: round;
  pointer-events: none;
}

.edge-track {
  stroke: #283b54;
  stroke-width: 12;
  opacity: 0.5;
}

.edge-color {
  stroke-width: 4;
  opacity: 0.85;
}

.edge-flow {
  stroke: #e1faff;
  stroke-width: 2;
  stroke-dasharray: 5 13;
  opacity: 0.8;
}

.viz-frame.is-live .edge-flow {
  animation: edge-flow 900ms linear infinite;
}

@keyframes edge-flow {
  to { stroke-dashoffset: -18; }
}
```

Here, `-18` equals one complete dash pattern: `5 + 13`. That makes the loop seamless.

**Important distinction:** a gradient stroke with moving dashes is not an actually moving gradient. It usually looks better and is simpler. If you genuinely want a traveling gradient, use a repeating gradient and animate its transform—but keep that to a few highlighted edges.

## Staggered reveal

Use a separate solid path, not the same element as the perpetual moving dashes:

```html
<path
  class="edge-reveal"
  pathLength="1"
  d="M300 300 C420 300 420 420 540 420"
  style="--reveal-delay: 160ms"
/>
```

```css
.edge-reveal {
  fill: none;
  stroke: var(--cyan);
  stroke-width: 3;
}

/* Add this class only for the entrance sequence. */
.viz-frame.is-entering .edge-reveal {
  stroke-dasharray: 1 1;
  animation: reveal-edge 650ms ease-out both;
  animation-delay: var(--reveal-delay, 0ms);
}

@keyframes reveal-edge {
  from { stroke-dashoffset: 1; }
  to   { stroke-dashoffset: 0; }
}
```

Suggested entrance sequence:

- Gate headers: 0–250ms.
- Cards: 80ms stagger per stage.
- Edges: 100ms stagger per stage.
- Particles: begin after approximately 700ms.
- Entire reveal: finish within 1.2–1.5s.

Run this once, not whenever the user scrolls back.

---

# 5. SVG pattern: SMIL token particles

For a handful of particles, SMIL is concise and works in current major browsers:

```html
<defs>
  <path
    id="dag-route-demo"
    d="M120 180 C320 180 320 430 540 430"
  />
</defs>

<use
  href="#dag-route-demo"
  fill="none"
  stroke="#54d8ff"
  stroke-opacity=".25"
  stroke-width="8"
/>

<g class="motion-only" aria-hidden="true">
  <circle r="3.5" fill="#e8fbff">
    <animateMotion dur="3.6s" begin="0s" repeatCount="indefinite">
      <mpath href="#dag-route-demo"/>
    </animateMotion>
  </circle>

  <circle r="3.5" fill="#54d8ff" opacity=".8">
    <animateMotion dur="3.6s" begin="-1.2s" repeatCount="indefinite">
      <mpath href="#dag-route-demo"/>
    </animateMotion>
  </circle>

  <circle r="3.5" fill="#54d8ff" opacity=".55">
    <animateMotion dur="3.6s" begin="-2.4s" repeatCount="indefinite">
      <mpath href="#dag-route-demo"/>
    </animateMotion>
  </circle>
</g>
```

Negative begin times distribute particles along the path immediately.

For a directional token rather than a dot:

```html
<g class="motion-only" aria-hidden="true">
  <path d="M-6 -2 L4 -2 L7 0 L4 2 L-6 2 Z" fill="#dcf8ff">
    <animateMotion
      dur="4s"
      repeatCount="indefinite"
      rotate="auto"
    >
      <mpath href="#dag-route-demo"/>
    </animateMotion>
  </path>
</g>
```

### SMIL caveats

- CSS `animation-play-state` **does not pause SMIL**.
- Use `svg.pauseAnimations()` / `svg.unpauseAnimations()`.
- Hide decorative tokens for reduced motion.
- Use SMIL **instead of**, not in addition to, a JS particle system for the same route.

For your failure DAG, I would use JS particles because you will want data-dependent route selection, stopping, and highlighting.

---

# 6. Production pattern: many particles with one shared scheduler

The important optimization is to **sample paths once**, then interpolate cached points each frame.

Avoid calling `getPointAtLength()` hundreds of times on every animation frame.

```js
const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

function samplePath(path, samples = 180) {
  const length = path.getTotalLength();
  const xy = new Float32Array((samples + 1) * 2);

  for (let i = 0; i <= samples; i++) {
    const p = path.getPointAtLength(length * i / samples);
    xy[i * 2] = p.x;
    xy[i * 2 + 1] = p.y;
  }

  return { xy, samples, length };
}

function pointAt(table, t) {
  const f = Math.max(0, Math.min(1, t)) * table.samples;
  const i = Math.min(table.samples - 1, Math.floor(f));
  const a = f - i;
  const j = i * 2;

  return {
    x: table.xy[j] +
       (table.xy[j + 2] - table.xy[j]) * a,
    y: table.xy[j + 1] +
       (table.xy[j + 3] - table.xy[j + 1]) * a
  };
}

function createParticleSystem(
  layer,
  routes,
  speed = 85 // SVG coordinate units per second
) {
  const particles = [];

  for (const route of routes) {
    const table = samplePath(route.path);
    if (table.length <= 0) continue;

    const count = route.count ?? 3;

    for (let i = 0; i < count; i++) {
      const el = svgEl("circle", {
        r: route.radius ?? 2.8,
        fill: route.color ?? "#dffaff",
        "pointer-events": "none"
      });

      layer.append(el);

      particles.push({
        el,
        table,
        phase: i / count,
        speed: route.speed ?? speed
      });
    }
  }

  function update(seconds) {
    for (const p of particles) {
      const t =
        (p.phase + seconds * p.speed / p.table.length) % 1;

      const { x, y } = pointAt(p.table, t);

      // Hide the teleport from route end back to route start.
      const fade = Math.min(1, t / 0.06, (1 - t) / 0.10);

      p.el.setAttribute("transform", `translate(${x} ${y})`);
      p.el.setAttribute("opacity", String(0.85 * fade));
    }
  }

  update(0);
  return { update };
}
```

Keep paths and the particle layer in the same SVG coordinate system. If paths are nested inside transformed groups, either place particles in that group too or explicitly transform the sampled coordinates.

## Visibility, pause, and reduced-motion controller

Every frame should have:

```html
<div class="viz-frame" data-motion="full">
  <button class="motion-toggle" type="button" aria-pressed="false">
    Pause animation
  </button>

  <svg class="viz-svg" viewBox="0 0 1440 820">
    <!-- Static geometry and labels -->
    <g class="motion-only" id="dag-particles" aria-hidden="true"></g>
  </svg>
</div>
```

Shared controller:

```js
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const controllers = new Set();

let raf = 0;
let lastTime = 0;

function ensureLoop() {
  const needed = [...controllers].some(c => c.active);

  if (needed && !raf) {
    lastTime = 0;
    raf = requestAnimationFrame(tick);
  } else if (!needed && raf) {
    cancelAnimationFrame(raf);
    raf = 0;
    lastTime = 0;
  }
}

function tick(now) {
  raf = 0;

  const dt = lastTime ? Math.min((now - lastTime) / 1000, 0.05) : 0;
  lastTime = now;

  for (const c of controllers) {
    if (!c.active) continue;
    c.time += dt;
    c.system.update(c.time);
  }

  if ([...controllers].some(c => c.active)) {
    raf = requestAnimationFrame(tick);
  } else {
    lastTime = 0;
  }
}

function refreshMotion(c) {
  const reduced =
    reduceMotion.matches ||
    c.frame.dataset.motion === "reduced";

  c.active =
    c.visible &&
    !document.hidden &&
    !reduced &&
    !c.userPaused;

  c.frame.classList.toggle("is-live", c.active);
  c.frame.classList.toggle("is-reduced", reduced);

  // Also controls any optional SMIL in this SVG.
  if (c.active) c.svg.unpauseAnimations?.();
  else c.svg.pauseAnimations?.();

  ensureLoop();
}

function registerMotion(frame, system) {
  const c = {
    frame,
    svg: frame.querySelector("svg"),
    system,
    visible: false,
    userPaused: false,
    active: false,
    time: 0
  };

  controllers.add(c);

  const observer = new IntersectionObserver(([entry]) => {
    c.visible = entry.isIntersecting;
    refreshMotion(c);
  }, { threshold: 0 });

  observer.observe(frame);

  const button = frame.querySelector(".motion-toggle");
  button?.addEventListener("click", () => {
    c.userPaused = !c.userPaused;
    button.setAttribute("aria-pressed", String(c.userPaused));
    button.textContent = c.userPaused
      ? "Resume animation"
      : "Pause animation";
    refreshMotion(c);
  });

  refreshMotion(c);
}

document.addEventListener("visibilitychange", () => {
  for (const c of controllers) refreshMotion(c);
});

reduceMotion.addEventListener("change", () => {
  for (const c of controllers) refreshMotion(c);
});
```

```css
.motion-only {
  pointer-events: none;
}

.viz-frame.is-reduced .motion-only {
  display: none;
}

@media (prefers-reduced-motion: reduce) {
  .viz-frame .motion-only {
    display: none;
  }

  .viz-frame .edge-flow,
  .viz-frame .edge-reveal,
  .viz-frame .node-halo,
  .viz-frame .stage-enter {
    animation: none !important;
    transition: none !important;
  }
}
```

All essential information must remain present when animation is paused or removed.

---

# 7. Figure one: make the pipeline a five-stage instrument panel

## Desktop layout

Use `viewBox="0 0 1440 620"`.

Five stage cards:

```js
const pipelineLayout = {
  cardWidth: 224,
  cardHeight: 264,
  cardY: 150,
  stageX: [48, 328, 608, 888, 1168]
};
```

Each card should contain four layers:

1. **Task identifier:** `T1 / CAPABILITY`
2. **Plain-language question**
3. **Protocol or measurement**
4. **Actual summary statistic**

Suggested copy structure:

| Stage | Plain-language question | Detail area |
|---|---|---|
| T1 Capability | Can the model produce a correct kernel? | Extraction, compilation, correctness, performance |
| T2 Weight-RSI | Does a weight update improve performance? | Update protocol, held-out gain, LoRA drift |
| T3 Procedure-RSI | Does changing the procedure improve search? | Procedure changes and evaluation budget |
| T4 Closed → Open | Does improvement transfer to the open setting? | Transfer protocol and retained gain |
| T5 Self-play | Does iterative feedback sustain improvement? | Rounds, diversity, retention, cumulative gain |

Match this wording to the benchmark’s actual definitions. Do not display a measurement that is not present in the data.

### Add a meaningful lower lane

Below the cards, use a band at approximately `y=475–565`:

- Selected model name.
- Its five measured stage results.
- A highlighted progression path.
- A textual outcome summary.

This adds detail without stuffing every card.

Use feedback arrows only if the benchmark truly contains those feedback loops. Otherwise, a visual return arrow can falsely imply that T5 feeds back into T2.

## Animation

- Stage cards reveal left to right.
- Connectors continuously show a small number of forward tokens.
- Selecting a model changes metric values and highlights its stage status.
- The selected model’s token stops at its last evaluated or failed stage.
- Unknown stages use `— / not evaluated`, not red.

A pulse halo should be separate from the card:

```html
<g transform="translate(160 150)">
  <circle class="node-halo" r="22"/>
  <circle r="6" fill="#54d8ff"/>
</g>
```

```css
.node-halo {
  fill: none;
  stroke: var(--cyan);
  stroke-width: 1.5;
  opacity: 0;
  transform-box: fill-box;
  transform-origin: center;
}

.is-live .is-selected .node-halo {
  animation: node-pulse 2.8s ease-out infinite;
}

@keyframes node-pulse {
  0%   { transform: scale(.65); opacity: .65; }
  70%  { transform: scale(1.35); opacity: 0; }
  100% { transform: scale(1.35); opacity: 0; }
}
```

---

# 8. Figure two: failure DAG as a bundled alluvial diagram

This should be the visual centerpiece.

## Structure

Use seven vertical columns:

```js
const gates = [
  "Scale",
  "Correctness wall",
  "Gradient",
  "LoRA drift",
  "Retention",
  "Diversity",
  "Outcome"
];

const gateX = [80, 285, 490, 695, 900, 1105, 1320];
```

Each gate has:

- A header.
- A one-line operational definition.
- One or more state nodes.
- Counts of evaluated / passed / stopped / unavailable.

**Avoid drawing all 133 model labels in the diagram.** Show aggregate structure by default, individual traces on selection.

## Three visual layers

### Layer A: aggregate ribbons

Group models by **observed state transitions** between adjacent columns. Ribbon width represents run count.

Do not group only by outcome and then imply that all intermediate paths were identical.

### Layer B: thin model traces

Render all runs as low-opacity paths:

```css
.model-trace {
  fill: none;
  stroke-width: 1;
  opacity: 0.08;
}
```

These provide fine texture without becoming the primary encoding.

### Layer C: selected path and particles

Selected model:

```css
.model-trace.is-selected {
  stroke: var(--cyan);
  stroke-width: 3;
  opacity: 1;
}
```

Use bright tokens only on:

- Major aggregate routes.
- Selected model paths.
- Optionally one hovered cohort.

## Ribbon path generator

A ribbon should be a filled area, not just a very thick stroke:

```js
function ribbonPath(x0, y0Top, y0Bottom, x1, y1Top, y1Bottom) {
  const dx = (x1 - x0) * 0.48;
  const c0 = x0 + dx;
  const c1 = x1 - dx;

  return [
    `M ${x0} ${y0Top}`,
    `C ${c0} ${y0Top}, ${c1} ${y1Top}, ${x1} ${y1Top}`,
    `L ${x1} ${y1Bottom}`,
    `C ${c1} ${y1Bottom}, ${c0} ${y0Bottom}, ${x0} ${y0Bottom}`,
    "Z"
  ].join(" ");
}

function centerlinePath(x0, y0, x1, y1) {
  const dx = (x1 - x0) * 0.48;
  return `M${x0},${y0} C${x0 + dx},${y0} ${x1 - dx},${y1} ${x1},${y1}`;
}
```

Use a **single global count-to-height factor** so equal counts have equal widths:

```js
const pixelsPerRun = usableRibbonHeight / totalRuns;
const ribbonHeight = transition.count * pixelsPerRun;
```

Reserve room for state-node gaps before calculating that factor. Do not independently rescale each column.

At each node:

1. Sort incoming links by source position.
2. Sort outgoing links by target position.
3. Allocate contiguous vertical slots.
4. Use each slot’s top and bottom as ribbon endpoints.

For a fixed seven-column graph, a few forward/backward barycentric ordering passes can reduce crossings without needing a layout library. Keep semantically important state order fixed.

### Missing and failed data

Make a clear distinction:

- **Failed:** measured failure at this gate.
- **Not evaluated:** no later measurement.
- **Unknown:** missing field.
- **Passed:** satisfied the stated condition.

If a failed run is carried across later columns for count conservation, place it in a visibly muted **“stopped earlier”** lane—not a normal evaluation node.

## Ribbon animation

Keep the ribbon body static:

```css
.ribbon {
  fill-opacity: .16;
  stroke-opacity: .2;
  stroke-width: 1;
}
```

Animate a thin centerline and a few token lanes inside it. For a wide ribbon, sample centerlines at 25%, 50%, and 75% of its thickness.

This produces dense flow without distorting the encoded width.

**Use constant token speed.** If particle frequency encodes run count, say so. Otherwise label motion as illustrative. Speed should not accidentally imply that one model trained or converged faster.

Also: call this an **observed failure-path diagram** unless the data actually establish causal relationships. Gate sequencing alone does not demonstrate causation.

---

# 9. Figure three: bubble scatter with integrated marginals

Use `viewBox="0 0 1440 900"`.

Example regions:

```js
const scatter = {
  left: 120,
  top: 190,
  right: 1190,
  bottom: 760,

  marginalTop: 60,
  marginalHeight: 90,

  marginalRight: 1220,
  marginalWidth: 110
};
```

## Encodings

- **X:** model scale, logarithmic.
- **Y:** RSI gain, linear if the range allows it.
- **Bubble area:** LoRA drift.
- **Fill:** outcome.
- **Outline:** selection.

Explicit scale helper:

```js
function logScale(value, d0, d1, r0, r1) {
  const t = Math.log(value / d0) / Math.log(d1 / d0);
  return r0 + t * (r1 - r0);
}
```

For area proportional to nonnegative drift:

```js
function driftRadius(drift, maxDrift, maxRadius = 22) {
  if (maxDrift <= 0) return 0;
  return maxRadius * Math.sqrt(Math.max(0, drift) / maxDrift);
}
```

Do not use radius directly proportional to drift: that squares the visual difference.

If you apply a minimum visible radius, disclose that the area mapping is floored. Better: keep the visual radius honest and give each point a larger transparent hit target.

## Top marginal

Use a histogram aligned exactly to the X scale:

- If scale values are discrete families, show aligned count bars.
- If treating scale continuously, use bins uniform in log space.
- Label it **run count**, not density, unless you actually normalize for density.

## Right marginal

Show horizontal bars aligned to RSI-gain bins.

A stacked outcome histogram is useful here, but make zero and negative gain unambiguous.

## Overplotting

133 runs do not need a force simulation.

Use:

- Fill opacity around `0.45–0.65`.
- Thin outlines.
- Large bubbles drawn first.
- Selected bubbles redrawn on top.
- A nearby-points list when several bubbles share coordinates.
- Optional cohort filters.

Do not silently jitter scale or gain. That changes the apparent data.

## Useful animation

- Initial bubble opacity/radius reveal, once.
- On hover, a vertical and horizontal guide appears.
- Relevant marginal bin highlights.
- The corresponding DAG path lights up.
- Selected bubble gets a separate expanding ring.

Do **not** make the bubbles drift or orbit. Their positions encode measurements.

A synchronized selection state is more valuable than ten extra animation effects:

```js
const selection = new EventTarget();
let selectedRunId = null;

function selectRun(runId) {
  selectedRunId = runId;

  selection.dispatchEvent(new CustomEvent("change", {
    detail: { runId }
  }));
}

selection.addEventListener("change", event => {
  const { runId } = event.detail;
  updatePipelineSelection(runId);
  updateDagSelection(runId);
  updateScatterSelection(runId);
});
```

Provide a searchable HTML run selector as a keyboard-accessible alternative to targeting SVG bubbles.

---

# 10. The three diagnostic figures

## Coverage versus scale

Use a large line chart with:

- Logarithmic scale axis.
- Actual sampled scales marked.
- Fraction or count clearly labeled.
- Numerator/denominator on hover.
- Uncertainty bands only if you can justify their calculation.

Motion:

- One-time line reveal.
- Highlight an actual point on selection.
- A moving guide may be decorative, but do not interpolate “measurements” between observed scales and present them as real.

Use straight segments unless a fitted curve is part of the analysis.

## Sharpening geometry

Use stacked area bands if the scale axis is genuinely continuous, or stacked bars for a small set of discrete scales:

- Unreachable: muted rose.
- Sharpenable: amber.
- Reliable: green/cyan.

Directly label the bands where there is room.

Require:

```js
unreachable + sharpenable + reliable ≈ 1
```

within a stated numerical tolerance. If they do not form a partition, do not force them into a 100% stack.

Motion:

- Brief reveal of boundaries.
- Selected-scale vertical slice.
- Update a side readout showing the three masses.

Avoid continuously shifting band boundaries; those boundaries are data.

## Zero-score failure taxonomy

Use 100% stacked bars by scale for composition, with sample counts above:

1. `no_extract`
2. `syntax`
3. `wrong`
4. `correct`

Show raw counts in the tooltip or in a toggle.

If `correct` exists among zero-score runs, explain why: correctness may not imply positive benchmark reward or speedup. That is likely an important result, not an edge case to hide.

Motion:

- Reveal scale groups in order.
- Highlight the same category across all scales.
- No perpetual animated stripes through every bar.

---

# 11. Responsive behavior: reflow, do not shrink everything

A 1440-wide SVG squeezed into a 360px viewport makes 16px labels approximately 4px tall.

Use `ResizeObserver` to choose a layout mode:

```js
let previousMode;

const resizeObserver = new ResizeObserver(([entry]) => {
  const mode = entry.contentRect.width < 760
    ? "mobile"
    : "desktop";

  if (mode === previousMode) return;
  previousMode = mode;

  renderPipeline(mode);
});

resizeObserver.observe(document.querySelector("#pipeline-frame"));
```

Recommended mobile adaptations:

- **Pipeline:** vertical five-stage layout.
- **DAG:** cohort summary plus horizontally scrollable detailed graph.
- **Scatter:** main plot full width; move marginals below or hide them behind a toggle.
- **Diagnostics:** one column.

For a scrollable detail graph:

```css
.viz-scroll {
  overflow-x: auto;
  overscroll-behavior-inline: contain;
}

.viz-scroll > svg {
  display: block;
  width: 100%;
  min-width: 1100px;
}
```

Keep scrolling inside the figure, not on the whole page. Add a visible “Scroll to inspect all gates” hint.

If rebuilding an SVG, remove its old particle system and observer registrations, then resample paths. Otherwise detached elements can remain in your animation loop.

---

# 12. Details that will make this look polished

### Use small halos, not giant blur clouds

For selected points, two circles often look better than a filter:

```html
<circle r="13" fill="#54d8ff" opacity=".07"/>
<circle r="7" fill="#54d8ff" opacity=".16"/>
<circle r="3" fill="#e5fbff"/>
```

### Keep decorative layers noninteractive

```css
.grid,
.ribbon-background,
.particle-layer,
.decorative-glow {
  pointer-events: none;
}
```

### Use wider invisible hit targets

```html
<path
  d="..."
  fill="none"
  stroke="transparent"
  stroke-width="18"
  pointer-events="stroke"
  data-run-id="..."
/>
```

Use these for a limited set of important paths, not 133 overlapping hit areas fighting for pointer events.

### Use HTML for tooltips and detailed readouts

SVG is excellent for geometry. HTML is better for wrapped explanatory text, links, controls, and tables.

- Set JSON-derived strings with `textContent`.
- Support focus as well as hover.
- Make click/tap pin a selection.
- Provide a static caption and downloadable or expandable data table.

### Keep animation semantics explicit

A short legend note can prevent misinterpretation:

> Ribbon width shows run count. Particle motion indicates path direction; it does not encode training time.

---

## What I would implement first

1. **Increase figure width and height**, improve typography, and center the composition.
2. Rebuild the pipeline as five substantial, explanatory cards.
3. Build a **static, count-conserving failure DAG** with correct missing-data handling.
4. Add the layered CSS flowing-edge treatment.
5. Add one shared, visibility-aware particle scheduler.
6. Link model selection between pipeline, DAG, and scatter.
7. Add marginals and diagnostic panels.
8. Finish mobile reflow, reduced motion, pause controls, and keyboard access.

The strongest end result is **large static geometry carrying the science, with dense, coordinated motion layered on top**. That gives you the striking “live instrument” aesthetic without making the benchmark feel like an animated screensaver.