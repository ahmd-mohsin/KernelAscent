## 1. Submission architecture — ranked

**P0 — Keep model vs. recipe, but make it a submission-unit distinction—not competing scientific tracks.**  
A model is an artifact; a harness is an intervention. Present two entry points: **“Submit a model/checkpoint”** and **“Submit an improvement recipe.”** Then assign the relevant evaluation track. Keep your existing track IDs internally, but expose plain-language labels. Otherwise, “procedure RSI” versus “harness” will look redundant.

- **Model rows:** capability, plus RSI only when accompanied by a reproducible improvement trajectory.
- **Recipe rows:** gain on the fixed open trainee, under a fixed resource envelope.
- Never combine their scores into one overall ranking. Explicitly disclose external models, tools, and human input.

**P0 — Treat recipe execution as hostile code execution.**  
Maintainer approval is not a sandbox. Run submissions on isolated, ephemeral workers with **no secrets, no unrestricted network, no persistent writable shared cache**, and hard resource limits. The private evaluator should own held-out inputs and scoring—not hand the private split to arbitrary recipe code. Never execute PR-controlled code in a privileged GitHub workflow.

**P0 — Prevent the leaderboard from rewarding search volume disguised as RSI.**  
Require an immutable starting checkpoint, recipe commit, total budget, candidate count, selection rule, and resulting checkpoints. Meter training **and** search, including teacher/API assistance. For recipe verification, run matched **reset and search-only controls**; rank paired gains with uncertainty, not the best submitted endpoint. Add submission quotas and a rotating private confirmation set to limit adaptive holdout fishing.

**P1 — Make submission cheaper than writing a benchmark adapter.**  
Ship one reference recipe and one command:  
`kernelascent submit --track harness --recipe recipe.yaml`  
It should smoke-test locally and generate the PR-ready manifest. Show verification turnaround, available compute sponsorship, and one fully worked submission. Give contributors stable result URLs, attribution, and citable versioned records. Use explicit row states: **schema-valid → reproduced → held-out verified**, with evaluator version and date. “Verified” alone overpromises.

## 2. Website — five high-impact changes, ranked

**1. Turn the hero into a precise editorial statement, not a dashboard.**  
Use **“Verified self-training did not compound.”** Below it: “Under our tested GPU-kernel optimization protocol.” Then one compact evidence strip: `213 evaluations · lineage ≈ reset · BF₀₁ ≈ 14.5`, with equivalence bounds accessible. Two links only: **Read the evidence** / **Submit a recipe**. Put the scale finding below; don’t make three claims compete for the headline.

**2. Establish a disciplined, narrow typographic grid.**  
Use a **760px centered container**, **620px prose**, and 20px mobile gutters. Try *Instrument Sans* for text and *IBM Plex Mono* for metrics. Set body **17/27px**, section heads **28/32px**, hero **46/49px** desktop, **34/38px** mobile. Use tabular numerals throughout boards. Space major sections **80px** apart and heading-to-content **20px**—not repeated card padding.

**3. Give red one semantic job: emphasis—not decoration or trust.**  
Use white, near-black `#171717`, muted gray `#666`, hairlines `#E7E7E7`, and red `#C52228`. Reserve red for the focal result, active navigation, and primary action. Verification should be a **labeled checkmark**, not a red badge. Remove most rounded cards; use thin rules and whitespace for an editorial-paper feel.

**4. Redraw the SVGs as evidence figures with a shared visual grammar.**  
Use live text, consistent strokes, direct labels, and generous annotation space. Make the **five-task ladder vertical on mobile** rather than shrinking it. In the DAG, distinguish **hypothesized and tested relationships** with labeled line styles; arrows alone imply too much. In the scale scatter, use a log parameter axis, visible zero-gain line, uncertainty, and direct annotations; highlight **2–8B** only as strongly as the data justify. Provide captions and downloadable SVGs.

**5. Make interaction quiet and submission-oriented.**  
Add a slim sticky anchor bar: **Findings / Results / Submit**. Default community boards to verified results, but expose pending submissions with counts; expand rows inline for budgets, provenance, and uncertainty. Keep numerical columns aligned and preserve submission actions on mobile. Use only **120–180ms hover/focus transitions** and optional one-time figure reveals—no looping particles, count-up metrics, or scroll-jacking. Respect reduced motion.
