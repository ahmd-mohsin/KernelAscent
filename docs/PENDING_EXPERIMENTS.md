# Pending experiments

All of these were in flight when the Marlowe `-pm06` allocation (Cycle 6) ended on 2026-09-28 at
17:00 PDT. None finished. Every cell stopped between 13:00 and 15:00 that day, and every one
resumes from its last completed round.

- **Partial results** (round JSONs) are in `data/trajectories/<cell>/`.
- **Resume checkpoints** (`adapter_*.pt`, `lineage_adapter.pt`, `resume_aux.json`) exist only on
  Marlowe, in `/users/muahmed/ka_data/<cell>/`. Copy them before running anywhere else.
- **Commands** are recorded in `.jobman.tsv`. They hard-code `/users/muahmed/...` paths, and the
  `fam*` cells put `/scratch/m000215-pm06/...` on `PYTHONPATH`, which is scheduled to be purged.
  Rewrite both on a new machine.

GPU-hours = remaining rounds × measured minutes per round (mean of the last 3 rounds) × GPUs.
They don't include queue time or the round lost each time a chunk hits its walltime.

## Priority 1: band sweep (`lab_compounding`, Qwen2.5-Coder-1.5B, `--held-tail`, n_held=20)

This is the direct test of the measurable-band claim: with the model fixed, the contrast should
peak in the middle as training volume varies. So far it rises and levels off. n_train 5 starves,
and saturation has not appeared at n_train 80 within 4 rounds.

| Cell | n_train | Seed | Rounds | min/round | GPUs | GPU-h left |
|---|---|---|---|---|---|---|
| band5-s1 | 5 | 1 | 6/8 | 27 | 1 | 0.9 |
| band5-s2 | 5 | 2 | 6/8 | 28 | 1 | 0.9 |
| band10-s1 | 10 | 1 | 6/8 | 27 | 1 | 0.9 |
| band10-s2 | 10 | 2 | 6/8 | 27 | 1 | 0.9 |
| band20-s1 | 20 | 1 | 5/8 | 28 | 1 | 1.4 |
| band20-s2 | 20 | 2 | 6/8 | 27 | 1 | 0.9 |
| band40-s1 | 40 | 1 | 6/8 | 35 | 1 | 1.2 |
| band40-s2 | 40 | 2 | 5/8 | 33 | 1 | 1.7 |
| band80-s1 | 80 | 1 | 4/8 | 49 | 1 | 3.3 |
| band80-s2 | 80 | 2 | 4/8 | 44 | 1 | 2.9 |
| **Subtotal** | | | | | | **≈15** |

Seed disagreement is larger than the differences between n_train levels (n_train 10: +0.158 vs
+0.020 over rounds 1–4). A third seed at n_train 10/20/40 is the cheapest way to make the shape
readable. That would add 3 new cells × 8 rounds, about 12 GPU-h.

## Priority 2: family replication (`lab_weight_rsi`, n_train=35, 5 rounds)

This tests whether the band holds outside Qwen. Phi-3.5 starts at C0=0.442 against a maximum of
about 0.5, so it is already saturated at round 0.

| Cell | Model | Seed | Rounds | min/round | GPUs | GPU-h left |
|---|---|---|---|---|---|---|
| famds-s1 | deepseek-coder-1.3b-instruct | 1 | 3/5 | 55 | 1 | 1.8 |
| famds-s2 | deepseek-coder-1.3b-instruct | 2 | 3/5 | 47 | 1 | 1.6 |
| famsm-s1 | SmolLM2-1.7B-Instruct | 1 | 1/5 | ~60 (est.) | 1 | ~4 |
| famsm-s2 | SmolLM2-1.7B-Instruct | 2 | 1/5 | ~60 (est.) | 1 | ~4 |
| famphi-s1 | Phi-3.5-mini-instruct | 1 | 1/5 | ~60 (est.) | 1 | ~4 |
| famphi-s2 | Phi-3.5-mini-instruct | 2 | 0/5 (never started) | ~60 (est.) | 1 | ~5.5 |
| **Subtotal** | | | | | | **≈21** |

## Priority 3: pass-rate scorer at depth (`deepp`, `KA_SCORE=passrate`, 1.5B, 15 rounds)

The depth reversal already reproduces under this scorer in all 3 seeds: a peak of +0.18 to +0.28
around rounds 3–6, then −0.03 to −0.37 at the last completed round. Finishing these only fills in
the tail.

| Cell | Seed | Rounds | min/round | GPUs | GPU-h left |
|---|---|---|---|---|---|
| deepp-q15-s1 | 1 | 13/15 | 39 | 1 | 1.3 |
| deepp-q15-s2 | 2 | 11/15 | 50 | 1 | 3.3 |
| deepp-q15-s3 | 3 | 13/15 | 43 | 1 | 1.4 |
| **Subtotal** | | | | | **≈6** |

## Priority 4: trust-region λ sweep (`deepkl*`, `KA_RSI_KL=λ`, 1.5B, 15 rounds)

The answer is already in: no λ prevents the reversal. λ=2 rises to about +0.1, then reverses to
about −0.1 by rounds 7–9. λ=5 is negative almost throughout. λ=20 starves the loop (5–12 examples
per round). Finishing these mainly gives the third seed for λ=2.

| Cell | λ | Seed | Rounds | min/round | GPUs | GPU-h left |
|---|---|---|---|---|---|---|
| deepkl2-s1 | 2 | 1 | 9/15 | 53 | 1 | 5.3 |
| deepkl2-s2 | 2 | 2 | 7/15 | 58 | 1 | 7.7 |
| deepkl2-s3 | 2 | 3 | 2/15 | 56 | 1 | 12.1 |
| deepkl5-s1 | 5 | 1 | 10/15 | 57 | 1 | 4.8 |
| deepkl5-s2 | 5 | 2 | 10/15 | 58 | 1 | 4.8 |
| deepkl20-s1 | 20 | 1 | 11/15 | 59 | 1 | 3.9 |
| deepkl20-s2 | 20 | 2 | 7/15 | 60 | 1 | 8.0 |
| **Subtotal** | | | | | | **≈47** |

## Priority 5: large-scale starvation (`fed-q14`, `fed-q32`, 5 rounds)

`fed-q14` is already starved, with 0–3 examples per round and a contrast of about 0. The remaining
rounds would only confirm it. `fed-q32` adds one more point at the starvation end, and it is the
most expensive cell on this list.

| Cell | Model | Seed | Rounds | min/round | GPUs | GPU-h left |
|---|---|---|---|---|---|---|
| fed-q14-s1 | Qwen2.5-Coder-14B-Instruct | 1 | 3/5 | ~170 | 3 | ~17 |
| fed-q14-s2 | Qwen2.5-Coder-14B-Instruct | 2 | 3/5 | ~160 | 3 | ~16 |
| fed-q32-s1 | Qwen2.5-Coder-32B-Instruct | 1 | 1/5 | ~275 | 6 | ~110 |
| **Subtotal** | | | | | | **≈143** |

## Totals

| Scope | GPU-h |
|---|---|
| Priority 1 (band sweep) | ≈15 |
| Priorities 1–3 (what the paper needs) | ≈42 |
| Everything except fed-q32 | ≈122 |
| Everything | ≈232 |

Any single GPU with ≥40 GB runs priorities 1–4. The 14B cells need 3 GPUs and 32B needs 6, or a
re-plan onto fewer, larger cards.

## Compute status (2026-10-04)

- **Marlowe:** the login node is reachable, but Slurm is not (`DNS SRV lookup failed`), so nothing
  can be submitted. The user is in groups `marlowe-m000159` and `marlowe-m000215-pm06` (expired),
  and is not in `marlowe-m000215`.
- **`my-ec2`, `vast-h100`, Tailscale `100.100.101.36`:** none reachable.
