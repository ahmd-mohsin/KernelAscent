# KernelAscent — progress & next steps (2026-09-13)

Snapshot of where the benchmark stands and what to do next. Companion to `ROADMAP.md` (queue), `RESUME.md` (live fleet/ops), `BENCHMARK_LOG.md` (numbers).

## Goal
An award-level benchmark for **compounding** GPU-kernel optimization / recursive self-improvement. The novelty is measuring *recursion* — whether improvement feeds itself — not raw capability. Scores are headroom-normalized against a per-task roofline so the ceiling is the model's skill, never the benchmark's.

## The 5-task ladder (built)
1. **Capability** — one-shot correct+fast kernel. Open + closed.
2. **Weight-RSI** — open model LoRA-trains on its own correct kernels; held-out C over rounds vs frozen-base and round-0-data controls.
3. **Procedure-RSI** — model rewrites its own strategy library + verified archive; weights fixed (closed-capable).
4. **Closed→Open** — closed frontier model rewrites an open trainee's training harness.
5. **Self-play (TRUE RSI)** — model authors its own strictly-harder tasks and improves on them. **This is the headline metric.**

### Task 5 — the 3-arm design (the airtight part)
Three arms from one base, equal budget, one fixed held-out ladder:
- **S STATIC** (fixed frontier), **F FROZEN-AUTHOR** (frontier escalates, author = frozen base), **L LIVE-AUTHOR** (author = current evolving model). Only the author role differs F vs L.
- **Primary metric L−F = author co-evolution** (self-referential signal). L−S = total curriculum benefit; F−S = curriculum without updating the author.
- Runs open (weight channel, `lab_selfplay_rsi.py`) and closed (procedure channel, `lab_selfplay_closed.py`).
- **Anti-reward-hacking gate**: rejects constant-output / identity / no-op / trivial-runtime authored tasks; semantic dedup; provenance logged (model-proposed vs programmatic backstop).

### Mechanism probe (`lab_rsi_mechanism.py`) — *why* RSI fails / *how* it passes
Per round: generation diversity (mode collapse), predictive entropy, **LoRA drift by transformer depth** (→0 = ceiling), retention (forgetting), train−held transfer gap (memorization). Emits a verdict: `diversity_collapse | forgetting | drift_saturation | no_transfer | no_headroom`, or PASS + carrying depth.

## What's running now (72 GPUs, 3 SDB jobs × 9 nodes)
- Open self-play 3-arm: small (DeepSeek-1.3B, Qwen-1.5B) seeds 0+1; mid (Qwen-7B s0+s1, DeepSeek-6.7B) 2-GPU/arm; large (Qwen-14B, StarCoder2-15B) 2-GPU/arm; xlarge (Qwen-32B).
- 8 tiny WHY-RSI probes (0.5–3B, families Qwen/DeepSeek/Yi/OpenCoder/SmolLM/StableCode/StarCoder) + 4 bigger (7–15B, 2-GPU/arm).
- Closed 3-arm self-play: GPT-5.6, Opus-5, Kimi-k2.5, DeepSeek-v3.2, Fable-5.1 (currently Q=0 — **bedrock creds expired**, needs refresh).

## Results so far (EARLY — rounds ~2–7, ~33–75 min/round)
- **Grader fix was decisive**: a path bug made every kernel grade `False` (C=0 everywhere) — fixed to self-locate the repo. All prior C=0 was that bug, not the models.
- **14B mechanism probe compounds**: C_train 0.03→0.16, C_held 0.04→0.12, drift rising, retention recovering — a model that *passes*.
- **Tiny models = C=0 floor**: sub-2B never emit a correct kernel → nothing to learn from (clean negative finding, not a blank).
- **Task 5 L−F ≈ 0 or slightly negative so far** across sizes. The 14B authors tasks but the gate rejects ~19/round as degenerate and the survivors the frozen base already solves (`base=1.0`) → the frontier doesn't actually harden → author co-evolution buys nothing yet. **Interim finding: open models ≤15B can't self-author genuinely-harder valid kernel tasks**, so self-play doesn't compound beyond a frozen-author curriculum. Needs more rounds + bigger/closed authors to confirm.

## Infra / reproducibility (built)
- **S3 progress store + resume**: `scripts/s3ckpt.py` (boto3/IRSA → `greenland-intern-artifacts-…` bucket; CLI fails cross-account). Self-play checkpoints per-arm LoRA adapters + frontier/round each round; `runrole.sh` restores before launch and the runner resumes from the next round. Daemons push every 300s. New nodes resume automatically.
- **Reasoning-chain traces**: per-round `*_trace.jsonl` (raw generation + best kernel + authored task) for post-run qualitative "what changed" analysis (activates on runs launched with current code).
- **Docs live**: README + website (GitHub Pages) + HF dataset card all carry the 5-task taxonomy, 3-arm design, and mechanism analysis.
- Memory-safe launch: 7B+ shards each arm across 2 GPUs (40GB limit); `runrole.sh` direct-nohup (setsid died on ssh-close); grade GPU off the model GPUs.

## What we can do next (menu)
**Finish the current science**
- [ ] Let self-play reach ~8–12 rounds and pull clean L−F / L−S / F−S curves + CIs (seeds 0+1); publish `selfplay.json`.
- [ ] Refresh bedrock creds → get closed-model self-play (GPT-5.6/Opus-5/Kimi) actually scoring; closed authors may clear the "can't author harder tasks" bottleneck.
- [ ] Mechanism aggregation (`rsi_mech.json`) → site panels + per-model verdicts; harvest reasoning-chain traces for qualitative figures.

**Strengthen Task 5 (if L−F stays flat)**
- [ ] Give the author a stronger proposer or a difficulty target (Goldilocks: keep tasks the *current* learner solves 30–70% of the time) so the frontier genuinely hardens.
- [ ] Add the recursion-interruption fork to self-play (clone state, freeze author at checkpoint, continue) — the decisive mechanism control.
- [ ] Future-learning-efficiency meta-metric M (disposable learning assay) per GPT-6 Astra spec.

**Scale / breadth**
- [ ] Bigger open authors (32B+ / 70B) — does author capability cross the threshold where L−F>0?
- [ ] Sealed held-out ladder + fresh-instance final confirmation for the official board.

**Framing**
- [ ] Retitle around "When learning to optimize compounds — and when it collapses"; lead with measurement + the mechanism/why story; Task 5 co-evolution as the centerpiece; tiny-floor and ≤15B-can't-author as honest negative results.
