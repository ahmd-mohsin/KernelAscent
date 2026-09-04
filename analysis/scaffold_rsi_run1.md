# Scaffold-RSI v2 — Run 1 Failure Analysis

**Setup.** 26 Bedrock models, frozen weights. Each round: optimize a held-out **transfer**
set (24 tasks, attention + matmul; MoE sliced out at n=24), graded on GPU against the
`min(eager, torch.compile)` roofline; a frozen-library **control**; the model reflects on a
disjoint **practice** set and grows its own strategy library. 4 rounds. Reasoning enabled
where supported. Metric `C = fast_1` (fraction of tasks beating the roofline).

**Headline.** No model shows recursive self-improvement. 17/26 never open the channel,
7 plateau, 2 degrade. But the *reasons* differ sharply by capability tier, and there are
two separate walls plus real harness bugs. Details below.

---

## 1. Global candidate outcomes (2,482 candidates)

| Outcome | Count | % |
|---|---|---|
| ok (correct) | 1018 | 41.0 |
| other_exception | 328 | 13.2 |
| python_syntax_error | 249 | 10.0 |
| wrong_api (AttributeError) | 171 | 6.9 |
| runtime_error | 161 | 6.5 |
| triton_compile_error | 160 | 6.4 |
| wrong_output | 141 | 5.7 |
| type_error | 107 | 4.3 |
| no_candidate (gen/parse fail) | 64 | 2.6 |
| value_error | 54 | 2.2 |
| bad_shape/NaN | 29 | 1.2 |

**Critical number:** of 1,018 *correct* kernels, only **261 were also faster than the roofline** (~26%). So even when a model writes a working kernel, it beats `torch.compile` only a quarter of the time.

---

## 2. Two distinct walls (by capability tier)

| Tier | n cand | ok(correct) | wrong | compile err | wrong-API | syntax | no-cand |
|---|---|---|---|---|---|---|---|
| **frontier** (opus/sonnet-5/fable) | 384 | **93%** | 0% | 0% | 0% | 0% | 5% |
| **mid/large** (gemma, qwen, deepseek, mistral, gpt-oss, nova, …) | 1522 | 31% | 7% | 7% | 7% | **15%** | 2% |
| **small/other** | 576 | 33% | 6% | 10% | 10% | 3% | 1% |

Two completely different failure regimes:

### Wall A — the *correctness* wall (mid / small models)
Non-frontier models fail to even produce valid, correct Triton ~two-thirds of the time:
- **15% python syntax errors** (mid/large), **10% compile errors**, **7–10% wrong-API `AttributeError`**, plus wrong-output and type errors.
- **Why:** these models have shallow, partly-hallucinated Triton knowledge. They invent APIs that don't exist (`triton.language.matmul`, `tl.tanh`, `tl.mean`, `triton.jit.block_dim`), emit malformed indentation, or call `@triton.jit` incorrectly. They are below the capability floor to reliably emit *valid* Triton, let alone fast Triton. This mirrors the open-weight GRPO observation: "learns valid before fast," and most never reach valid.

### Wall B — the *speed* wall (frontier models)
Frontier models write **correct** kernels 93% of the time (0% syntax/compile/API errors) — their bottleneck is **speed, not correctness**:
- They rarely beat the `min(eager, torch.compile)` roofline. `torch.compile`/cuBLAS are already near-optimal on matmul and fuse elementwise/reduction well, so a hand-written Triton kernel usually ties or loses.
- On attention specifically, only **6% of all attempts** beat the roofline; on matmul, 19%.
- **Why:** matching or beating a mature autotuning compiler requires expert-level tiling / warp-specialization / autotuning that the models produce inconsistently. Their good strategies (call `F.scaled_dot_product_attention`, shape-dispatch, runtime autotune guards) help sometimes but don't reliably clear the bar.

---

## 3. Why self-improvement doesn't compound

### 3a. Frontier: ceiling-limited, library adds noise not lift
opus-5 `[0.50,0.62,0.54,0.58]`, opus-4-8 `[0.33,0.38,0.29,0.33]`, sonnet-5 `[0.38,0.38,0.42,0.33]`.
Round 0 is already near the model's ceiling for this task set; accumulated strategies are
reasonable (see below) but the *execution* — writing a kernel that actually beats compile —
doesn't improve with better advice, because the limiter is generation skill, not knowledge.
Result: flat, noisy `C` (b≈0), `Δ_k` oscillates around 0. Fable rose to 0.67 mid-run then
fell back (b<0) → plateau, not compounding.

### 3b. Non-frontier: the library gets *poisoned*
The self-grown library actively hurts weaker models. Two corruption modes, both measured:

**(i) Reasoning-text leak into the "strategy library"** (extraction bug). For reasoning
models the reflect step captured chain-of-thought instead of strategies. Corruption counts
in the final library (of 12 entries): **qwen3-32b 12/12, qwen3-next 12/12, deepseek-r1 12/12,
kimi 8/12, gpt-oss/nemotron/minimax/deepseek-v3.2 4/12.** Verbatim examples:
```
<reasoning>
Okay, the user wants me to add 1-3 new Triton/kernel strategies to their playbook...
First, the failed cases (pass=0) are attn_s512_d512_600000, t500001_1024x4097, ...
```
These entries fill the prompt with meta-commentary, crowding out real guidance.

**(ii) Hallucinated APIs entering the library** (no grounding check). Weak models "learn"
strategies that reference non-existent APIs, then dutifully try to use them next round:
```
Use `torch.backends.cuda.matmul/matmul_strided` for strided matmuls
Apply `#pragma unroll` to loops...            (CUDA C, not Triton/Python)
convert operations to use cutlass kernels     (no such Triton call)
Use block-level parallelism with triton.jit.block_dim
```
This directly explains the **degrade** cases: llama3-3-70b `[0.25, 0, 0, 0]` and
sonnet-4-5 `[0.21,0.04,0.04,0.08]` — round 0 is fine, then a bad strategy enters the library
and every later round follows the poison. Δ_final = −0.375 (llama3-3-70b) is the worst.

For contrast, frontier libraries are clean and genuinely good:
```
Attention: don't hand-roll FlashAttention on A100 — call F.scaled_dot_product_attention...
Odd trailing dims (4097, 513): flatten to 1D contiguous, flat grid BLOCK=2048–4096...
Ship a runtime guard: microbenchmark Triton vs eager/torch.compile once per (shape,dtype)...
```
Good advice, but (3a) it doesn't move the needle because the limiter is execution.

---

## 4. Harness / measurement issues (not model faults)

1. **`reflect()` has no output sanitation** → reasoning-model CoT leaks into the library
   (Wall 3b-i). Fix: strip `<reasoning>…</reasoning>`, drop lines that aren't imperative
   strategy sentences, dedupe.
2. **No grounding check on learned strategies** → hallucinated APIs poison weak models
   (3b-ii). Fix: reject entries matching known-bogus patterns / non-existent API regexes.
3. **Difficulty floor missing.** The transfer set is attention+matmul only; only 6–19% of
   attempts beat the roofline, and there is nothing an 0.5–7B model can solve. This *guarantees*
   "channel-not-opened" for 17/26 and makes self-improvement unmeasurable below frontier.
4. **MoE was sliced out** (`[:24]` after family filter dropped moe) → the transfer set wasn't
   the intended attention+matmul+moe mix.
5. **Anthropic reasoning is redacted** (encrypted signature, `text=""`), so the reasoning
   trajectory can't be stored for Claude models — only that they reasoned.
6. **Roofline is `torch.compile`, not expert kernels** — appropriate, but it makes Wall B a
   very high bar; ranking on `fast_1.5/2` or vs CUTLASS would change absolute numbers.

---

## 5. Root-cause synthesis

- **Two capability regimes, one metric.** Below frontier the binding constraint is *writing
  valid/correct Triton* (Wall A). At frontier it is *beating a mature compiler* (Wall B). A
  single hard metric collapses both into ~0–0.6 with no headroom to show improvement.
- **Scaffold-RSI's mechanism is knowledge, but the limiter is execution (frontier) or
  validity (non-frontier)** — so better textual strategies don't translate into higher scores,
  and for weak models the self-authored knowledge is wrong and *harms* them.
- **The loop can go negative** — an unfiltered self-editing scaffold is not monotonic; bad
  strategies compound downward (the 2 "degrades").

## 6. Implications (for the next design)

1. **Tiered difficulty (Easy→Ultra).** An Easy tier that 0.5–7B models can pass on validity
   *and* speed, so the channel can open and improvement is measurable; Ultra for frontier
   headroom. Report compounding per tier.
2. **Sanitize + ground the library** (fixes 4.1, 4.2) so self-improvement isn't sabotaged by
   its own extraction/hallucination artifacts.
3. **Report the two walls explicitly** (correctness-rate vs speed-rate) rather than one `fast_p`.
4. **Fix the MoE slice**, and consider expert-kernel rooflines + `fast_1.5/2` for frontier discrimination.

*Data source: `/tmp/instance_storage/ka_data/srsi/*` (26 models × 4 rounds × transfer/control/practice), 2,482 graded candidates.*
