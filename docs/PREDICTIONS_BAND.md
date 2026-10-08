# Registered predictions: the measurable band, tested prospectively

**Committed before any cell below was submitted (2026-10-07).** Every cell runs `lab_weight_rsi` on
the DSL kernel bank (`KA_PROMPT=kernel`, `KA_SCORE=compiled`, `KA_ROOF_ARCH=h100`), with k=10,
5 rounds, seeds 1–3, and all three arms on one GPU. These are the same settings as the `fed*`
calibration cells, except for the one variable each cell changes. Nothing here may be edited after
the first of these cells reports a round. Corrections go in a dated addendum below the line.

## Outcome definitions (fixed now)

Each configuration is scored at round 5, using the mean over its seeds:

| Outcome | Definition |
|---|---|
| **STARVED** | mean verified examples per round (`n_ex`) < 2, the Amendment 6 floor |
| **SATURATED** | not starved, and mean final `C_ctrl` ≥ 0.45 (90% of the 0.50 correct-at-parity ceiling) |
| **IN-BAND** | neither |

The contrast reported alongside is `delta_self_minus_ctrl`, averaged over the last two rounds and
then over seeds.

## The rules, and how well they fit the calibration cells

**Rule S (starvation, pre-training).** Compute `T0 = n_train × k × y0`, where `y0` is the model's
round-1 yield at n_train=35 from its `fed*` cell. If `T0 < 3`, predict STARVED.

Calibration on the 10 existing configurations: correct on 9 of 10. It predicts starvation for
fed_q14 (T0 0.5), fedyi (0.3), dose3_q15 (0.5) and dose10_q15 (1.5), all four observed starved. It
predicts non-starvation for fed_q05 (8.0), fed_q15 (5.4), fed_q3 (19.7), fedds (4.3) and fedoc
(12.0), all five observed not starved. The dose cells use the fed_q15 yield of 1.54%. **The miss is fed_q7:** T0 = 4.0, yet it starved, because its
example count never grows.

**Saturation has no pre-training rule.** Three calibration cells have nearly identical C0
(0.04–0.08) and y0 (1.1–1.5%), yet end in three different outcomes: fed_q7 starved, fedds saturated
and fed_q15 inside the band. What separates them is how fast the control learns, which can only be
seen after training starts. The abstract's claim that the band is "predictable in advance from
dynamic range and throughput" is supported for starvation and **not yet** for saturation.

The saturation predictions below therefore come from the band hypothesis itself, not from a fitted
rule. **Hypothesis H-vol:** reducing training volume slows the control, so a model that saturates
at n_train=35 lands inside the band at n_train=10, provided Rule S does not predict starvation.

## Predictions

| Cell prefix | Model | n_train | Change vs calibration | y0 used | T0 | **Prediction** | Tests |
|---|---|---|---|---|---|---|---|
| `pa_oc10` | OpenCoder-1.5B-Instruct | 10 | volume 35 → 10 (fedoc SATURATED) | 3.43% | 3.4 | **IN-BAND** | H-vol: pull a saturated model in |
| `pa_q3n10` | Qwen2.5-Coder-3B-Instruct | 10 | volume 35 → 10 (fed_q3 SATURATED) | 5.62% | 5.6 | **IN-BAND** | H-vol, second family member |
| `pa_q05n10` | Qwen2.5-Coder-0.5B-Instruct | 10 | volume 35 → 10 (fed_q05 SATURATED) | 2.29% | 2.3 | **STARVED** | Rule S on the low side |
| `pa_dsn10` | deepseek-coder-1.3b-instruct | 10 | volume 35 → 10 (fedds SATURATED) | 1.24% | 1.2 | **STARVED** | Rule S on the low side |
| `pa_q7n100` | Qwen2.5-Coder-7B-Instruct | 100 | volume 35 → 100 (fed_q7 STARVED) | 1.14% | 11.4 | **IN-BAND** | push a starved model in with volume. Rule S's one miss was this model |
| `pa_yilen` | Yi-Coder-1.5B-Chat | 35 | `KA_EXTRACT=lenient` (fedyi STARVED) | measured at round 1 | conditional | **if mean round-1 `n_ex` ≥ 3: IN-BAND, else STARVED** | push a starved model in by raising yield |

`pa_yilen` is the one conditional prediction. Lenient extraction changes yield by an amount nobody
has measured for Yi. The rule is fixed now, and its input is read after the first generation round,
before any training.

**Known risk, stated in advance:** n_train=10 draws the first 10 tasks of a fixed shuffle. Their
yield may differ from the 35-task average. dose10_q15 had a predicted T0 of 1.5 and observed 0.

### Scoring the test

- A prediction is **correct** if the observed outcome class matches.
- The test **supports** the band hypothesis if at least 5 of 6 predictions are correct, and
  **refutes H-vol** if both `pa_oc10` and `pa_q3n10` come out SATURATED.
- Each class prediction is reported with its seed-level breakdown. A configuration whose seeds
  disagree on class is reported as such, not by majority vote.

## Noise floor (no prediction; a measurement)

- `noise_q15_s1_r1` and `noise_q15_s1_r2` repeat `fed_q15_s1` exactly: same command, same seed.
- With the original they give three same-seed copies. The spread across them is the run-to-run
  noise the contrast must exceed.
- Context: fedds_s1 and famds_s1, an accidental same-seed pair, were +0.188 vs −0.050 at round 3.

---
*Addenda (dated, below this line only):*
