## Direction: small research diagrams, not dashboard panels

Use **three centered figures, stacked vertically**: RSI ladder → causality DAG → scatter. No dark surfaces, full-bleed backgrounds, gradients, or decorative glow.

### Palette and sizing

| Role | Hex |
|---|---|
| Page | `#FFFEFB` |
| Figure / node fill | `#FFFFFF` |
| Ink | `#241C1C` |
| Secondary text | `#716361` |
| Rules / inactive edges | `#DCCFCA` |
| Red accent | `#B4232C` |
| Pale red emphasis | `#FCEEEE` |

```css
:root {
  --paper: #fffefb;
  --ink: #241c1c;
  --muted: #716361;
  --rule: #dccfca;
  --red: #b4232c;
  --wash: #fceeee;
}

body { background: var(--paper); color: var(--ink); }

.research-figures {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2rem;
}
.research-figure {
  width: calc(100% - 2rem);
  max-width: 680px;
  margin: 0;
}
.research-figure--dag { max-width: 560px; }
.research-figure--scatter { max-width: 520px; }

.research-figure svg { display: block; width: 100%; height: auto; }
.research-figure figcaption {
  margin-top: .65rem;
  font: .875rem/1.5 Georgia, serif;
  color: var(--muted);
}
.research-figure svg text {
  font-family: Georgia, serif;
  fill: var(--ink);
}
```

**ViewBoxes:** ladder `0 0 640 660`; DAG `0 0 560 300`; scatter `0 0 520 340`. Keep SVG text at **13–15 units**, titles at **17–18**.

## RSI ladder: five compact internal-loop strips

Not a staircase of task names. Make **five vertically stacked, numbered mini-schematics**, roughly 120 units each. Each has a short title, a forward process, and—except T1—a bottom return arrow.

Use square-ish boxes (`rx="3"`), thin gray edges, and **pale-red boxes only for what changes**. Explicitly label frozen components; color alone should not carry meaning.

| Strip | Exact schematic content |
|---|---|
| **T1 · One-shot solve** | `Task → Solver → Kernel → Grade harness`. Small note: **No feedback update**. No return arrow. |
| **T2 · Learn through LoRA** | `Base + adapter → Generate kernels → Grade harness → SFT LoRA adapter → Updated weights`. Return updated weights to the model. Put a small **SFT harness** label around the adapter-training stage. |
| **T3 · Improve external memory** | `Frozen solver → Generate kernels → Grade harness → Edit artifacts`. Inside the artifact box: **strategy library / solver prompt / kernel archive**. Return artifacts to the solver, labeled **next solve context**. |
| **T4 · Rewrite the training process** | `Closed researcher → Edit training harness → Train open trainee → Evaluate`. Harness box lists **data / LoRA hparams / curriculum**. Evaluation returns to the researcher; label trainee update **LoRA weights**. |
| **T5 · Co-evolve** | `Author proposes harder tasks → Solver solves → Grade harness → Update author + solver`. Split the return into **author update** and **solver update**, reconnecting to their respective boxes. |

For T4 and T5, use **two shallow rows inside the strip** rather than shrinking labels to fit. Omit ornamental icons; the harnesses, adapters, and return paths are the explanation.

### Small SVG example: T2 strip

Embed this group within the ladder; its local coordinate area is `640 × 132`.

```svg
<svg viewBox="0 0 640 132" role="img" aria-labelledby="t2-title">
  <title id="t2-title">
    T2: generate and grade kernels, train a LoRA adapter,
    then repeat with updated weights.
  </title>
  <defs>
    <marker id="t2-arrow" viewBox="0 0 8 8"
            refX="7" refY="4" markerWidth="6" markerHeight="6"
            orient="auto-start-reverse">
      <path d="M0 0L8 4L0 8Z" fill="#716361"/>
    </marker>
    <marker id="t2-red-arrow" viewBox="0 0 8 8"
            refX="7" refY="4" markerWidth="6" markerHeight="6"
            orient="auto">
      <path d="M0 0L8 4L0 8Z" fill="#B4232C"/>
    </marker>
  </defs>

  <text x="16" y="20" font-size="17">T2 · Learn through LoRA</text>

  <g fill="white" stroke="#DCCFCA">
    <rect x="16"  y="38" width="100" height="40" rx="3"/>
    <rect x="140" y="38" width="112" height="40" rx="3"/>
    <rect x="276" y="38" width="80"  height="40" rx="3"/>
    <rect x="380" y="38" width="116" height="40" rx="3"
          fill="#FCEEEE" stroke="#B4232C"/>
    <rect x="520" y="38" width="104" height="40" rx="3"/>
  </g>

  <g text-anchor="middle" font-size="13">
    <text x="66" y="55">Base +<tspan x="66" dy="15">adapter</tspan></text>
    <text x="196" y="55">Generate<tspan x="196" dy="15">kernels</tspan></text>
    <text x="316" y="55">Grade<tspan x="316" dy="15">harness</tspan></text>
    <text x="438" y="55">SFT LoRA<tspan x="438" dy="15">adapter</tspan></text>
    <text x="572" y="55">Updated<tspan x="572" dy="15">weights</tspan></text>
  </g>

  <g fill="none" stroke="#716361" marker-end="url(#t2-arrow)">
    <path d="M116 58H138"/>
    <path d="M252 58H274"/>
    <path d="M356 58H378"/>
    <path d="M496 58H518"/>
  </g>

  <path id="t2-return" d="M572 78V108H66V80"
        fill="none" stroke="#B4232C" stroke-opacity=".3"
        marker-end="url(#t2-red-arrow)"/>
  <use href="#t2-return" class="loop-flow"
       marker-end="none" aria-hidden="true"/>

  <text x="438" y="94" text-anchor="middle" font-size="12">SFT harness</text>
  <text x="300" y="126" text-anchor="middle" font-size="12">
    repeat with updated weights
  </text>
</svg>
```

### Motion: only the feedback path

```css
.loop-flow {
  fill: none;
  stroke: var(--red);
  stroke-width: 1.5;
  stroke-opacity: .65;
  stroke-dasharray: 4 20;
  animation: feedback-flow 2.4s linear infinite;
}
@keyframes feedback-flow {
  to { stroke-dashoffset: -24; }
}
@media (prefers-reduced-motion: reduce) {
  .loop-flow { animation: none; display: none; }
}
```

The solid return path remains visible without motion. **No bouncing boxes, pulsing nodes, or animated plot points.** No JavaScript is necessary.

## Keep the DAG and scatter quiet

- **DAG:** preserve existing causal structure and arrow directions. White nodes, gray ordinary edges, red only on the causal path being discussed. Use dashed edges only if they have an explicit semantic meaning—not as decoration.
- **Scatter:** white plotting area, ink tick labels, very light grid (`#EEE5E1`), muted comparison points, red focal points. Use shape differences as well as color for series. Preserve real axes, scales, and data.
- Caption each figure with **one takeaway sentence**, not a UI-style status footer.

For narrow screens, reflow ladder strips into two rows using an alternate SVG layout; **do not simply scale six-box rows until their text becomes unreadable**.