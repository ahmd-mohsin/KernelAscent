Based on the specification—not a screenshot—the biggest risk is **too many marks competing at equal importance**. World-class scientific graphics make the structure obvious first, then reward inspection.

## 1. Pipeline — Replace continuous animation with one coordinated handoff

**Likely weakness:** Five cards, layered connectors, moving dashes, 12 connector particles, and a traveling agent create several competing clocks. That reads as “animated diagram,” not a coherent execution trace.

**Single improvement: one clock, one active transition.**

- Keep connector tracks permanently visible but subdued: roughly `2px`, low-contrast neutral.
- Remove the moving dashes and three-particle trains.
- Run **one packet through the entire pipeline**, activating only its current connector.
- On arrival, briefly emphasize the destination card’s border/header—not the whole card with a glow.
- Make the rail agent follow that same state. It must not wander independently.

Suggested choreography:

```text
T1 dwell 450ms → travel 650ms → T2 dwell 450ms → …
T5 dwell 900ms → reset without a visible backward sweep
```

Use one `requestAnimationFrame` clock and `getPointAtLength()` for packet positioning. For reduced motion, show the static route without packet travel.

**Why this wins:** Animation becomes an explanation of sequence. The inactive diagram gains enough silence to look deliberate and expensive.

## 2. Causality DAG — Make bundles, not individual runs, the primary visual unit

**Likely weakness:** 133 variable-width ribbons across seven columns will become spaghetti, especially where wide paths overlap and failure rings accumulate. Animating only RSI routes also gives that class extra perceptual weight beyond its actual prevalence.

**Single improvement: a bundled overview with individual-run inspection.**

Aggregate runs by:

```js
key = [gatePath, terminalGate, outcome]
```

Each bundle gets:

```js
bundleWidth = k * sum(memberDrifts)
```

- Bundle **only genuinely identical gate paths**; never imply a shared path that the data does not contain.
- Use a stable vertical ordering across columns: terminal gate, then outcome. Do not independently reorder at each gate.
- Replace overlapping failure rings with **one termination cap per bundle**, labeled with run count.
- Hover, keyboard focus, or tap reveals the bundle’s constituent ribbons; selecting a run highlights its exact path.
- Remove overview particles. If retained, animate only the explicitly selected run.

Label the encoding clearly: **width = total drift; n = runs**. Aggregate width is not success probability.

**Why this wins:** Readers can see where flow survives or terminates immediately. You retain all 133 runs without requiring readers to untangle them simultaneously. Lower opacity alone would not solve that structural problem.

## 3. Scatter — Separate precise position from drift magnitude

**Likely weakness:** Large filled bubbles obscure small models and each other. Their visual mass can dominate the gain axis and make the sub-2B boundary look less precise.

**Single improvement: use a fixed center dot plus a quiet, area-scaled drift halo.**

```js
const r = Math.sqrt(k * drift / Math.PI); // halo area ∝ drift
```

```svg
<g class="model" style="color: var(--outcome)">
  <circle r="HALO_RADIUS" fill="currentColor" fill-opacity=".06"
          stroke="currentColor" stroke-opacity=".25"/>
  <circle r="2.5" fill="currentColor"/>
</g>
```

Render **all halos first, largest to smallest, then all center dots above them**. Group-by-group rendering would let later halos cover earlier centers.

On hover/focus, strengthen only the selected halo and show its label. Use an invisible hit target of at least `8px` radius so small halos remain selectable.

**Why this wins:** Every observation has a precise, readable location; drift remains available as a secondary encoding. The plot stops looking like colored bubble decoration and starts looking like a measurement instrument.