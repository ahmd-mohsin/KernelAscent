# What goes in the main paper, what goes in the appendix, what gets cut

A rubric fixed **before** the final numbers land, so placement is decided by evidential strength
rather than by which result turned out most flattering. Filled in as cells complete.

## The rubric

A result earns **main text** only if it clears all four:

1. **Powered** — n meets the pre-registered target, or the claim is explicitly bounded by its n.
2. **Instrument-validated** — the measurement is shown to have range, and a positive control
   fired on the same hardware and metric.
3. **Policy-robust** — survives the preprocessing choices it depends on (extraction policy,
   scorer, binning), or the dependence is reported.
4. **Load-bearing** — the paper's argument changes if it is removed.

**Appendix** = sound but not load-bearing, or underpowered but worth recording.
**Cut** = cannot be defended under questioning, or is superseded.

## Current assignment

| result | powered | validated | robust | load-bearing | place |
|---|---|---|---|---|---|
| **Instrument validity** (7 defects, gates) | n/a | n/a | n/a | **yes** | **MAIN — lead** |
| **Loss cascade** (348→118→74→4) | yes | yes | policy stated | **yes** | **MAIN** |
| **Reachability** precondition | yes | yes | yes | **yes** | **MAIN** |
| **Compile-gain probe** (1.34× median) | yes | n/a | n/a | yes | **MAIN** (short) |
| **T1-kernel** endpoint contrast | 1 seed, **replicated across routes** | yes | **yes — both policies** | yes | **MAIN** |
| ~~T1-kernel *monotone ordering*~~ | — | — | **fails under lenient** | — | **CUT** (retracted; artifact of strict extraction) |
| **T2-kernel** compounding (headroom) | valid prefixes only | control fires 73% | **metric saturates at 0.50** | yes | **MAIN as an instrument result**, not an RSI one |
| **T2-kernel** teacher injection | matched n=44/arm | — | — | no | **APPENDIX** — not detected; supersedes an earlier positive |
| **T2-kernel** under pass-rate | running (rounds=2) | — | resolves 0.040 vs 0.020 | yes | **MAIN if it reaches depth** — the metric that does not saturate |
| **prereg** `self−fresh_frozen` | **re-running from round 0** (Amendment 5) | — | registered scorer | **yes** | **MAIN if n≥8** — all pre-15:32 data discarded |
| A100 compounding null | yes (57) | **control never fired** | — | yes | MAIN, flagged not admissible under our own gate |
| Search beats training | yes (57) | yes | yes | yes | **MAIN** |
| Coverage-vs-scale, 0-forensics | yes (133) | yes | yes | supporting | MAIN (compressed) |
| WHY-RSI mechanism | yes (133) | associations only | binning fixed | supporting | **APPENDIX** |
| E2 strategy-cap ablation | n=2/cell | yes | yes | supports a retraction | **APPENDIX** |
| Probe-as-intervention | **n=12** | weakest comparator | — | no | **APPENDIX** |
| T3 procedure-RSI gains | n=1/cell | cap-confounded | — | retracted reading | APPENDIX, gains only |
| T4 closed→open | **n=1** | no seeds, no CI | — | no | **CUT** to appendix line |
| T5 self-play `L−F` | 11 qualifying rows | author-yield failure | **1200-token budget vs T1/T2's 2048** | no | **APPENDIX** with interval *and* the budget gap |
| 32B scale probe | retried (queued) | — | — | no | **CUT unless it lands** — group quota is 31x over its block limit and 166x over inodes; not ours to fix |
| **DSL kernel bank** (124 tasks) | 456 screened, 27% kept | held-out-family split | — | supporting | **MAIN (one line) + APPENDIX** — answers "only 29 tasks" |
| **Cross-rung budget asymmetry** | n/a | n/a | n/a | **yes** | **MAIN — stated as a limitation**: T1/T2 at 2048 tokens, T3/T5 at 1200 |

## Framing

Lead with the instrument. The strongest, most novel, most defensible contribution is that a
two-arm self-improvement benchmark manufactures nulls cheaply, with seven worked examples and
the gates that catch each. The RSI ladder becomes the case study that produced them.

This is also the only framing immune to the strongest attack available: that by our own proposed
criteria (positive control required, reachability precondition) our own headline null is
inadmissible. Under an instrument-first framing that inadmissibility is the thesis, not a wound.

## What must be added regardless of results

* a bibliography and a related-work section (currently **zero** `\cite` commands)
* a KernelBench comparison — same niche, visibly shared task format
* a datasheet, licences, and a contamination statement (the RSI bank is model-generated)
* non-LLM baselines: `torch.compile max-autotune`, a Triton autotuner, a human reference kernel
