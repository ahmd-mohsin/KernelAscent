# E0 — capability-monotonicity validity gate

Assay: base `develop` only (no lineage/revise), FIXED 15 Medium tasks (seed0=0), bounded
attainment C in {0, 0.5, 1.0} vs the torch.compile baseline. Code: `kernelascent/v3/capcheck.py`
(`--k` samples/task, C averaged over samples; Wilson 95% CI on rates). Ladder: Qwen2.5-Coder
0.5/1.5/3/7/14B, Llama-3.1-8B, Fable 5.1 (API).

## k=5 result (75 trials/model)

| model | meanC | correct [95% CI] | fast [95% CI] |
|---|---|---|---|
| Coder-0.5B | 0.007 | 0.013 [0.002, 0.072] | 0 |
| Coder-3B | 0.053 | 0.107 [0.055, 0.197] | 0 |
| Llama-3.1-8B | 0.020 | 0.040 [0.014, 0.111] | 0 |
| Coder-14B | 0.020 | 0.040 [0.014, 0.111] | 0 |
| Coder-1.5B | 0.100 | 0.200 [0.125, 0.304] | 0 |
| Coder-7B | 0.140 | 0.280 [0.191, 0.390] | 0 |
| Fable 5.1 | 0.293 | 0.360 [0.261, 0.473] | 0.227 [0.147, 0.333] |

## Verdict: PASS with one documented exception

1. The metric tracks capability: monotone rise 0.5B < 3B ~ 1.5B < 7B < Fable; frontier (Fable) is
   the ONLY model that beats the compile baseline. 0.5B (near-zero) vs Fable (top) separated with
   non-overlapping CIs. 1.5B vs 3B statistically tied (close in capability).
2. Exception: Coder-14B (0.040) < 7B (0.280), CIs non-overlapping, also low at k=1 (0.133).
   Root-caused by reading candidates (files are complete — not truncation): the 14B-Instruct
   checkpoint hallucinates torch internals, e.g. calls the private JIT pass
   `torch._C._jit_pass_fuse_addmm(x,W,b)` as a matmul and uses `math.sqrt` without importing math.
   Genuine model behavior of that checkpoint, not a measurement artifact.
3. Llama-3.1-8B (0.040) far below same-size code models — expected (not code-specialized);
   confirms the metric reflects task-relevant capability, not size alone.

## Method lesson

k=1 is underpowered for ranking (k=1 3B=0.000 and the 14B<7B gap were partly noise, partly real;
k=5 disentangled them). Capability assays use k>=5 + Wilson CIs; report ties as ties.

## Raw JSON

```
### c0p5b
{
  "who": "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.007,
  "correct_rate": 0.013,
  "correct_ci": [
    0.002,
    0.072
  ],
  "fast_rate": 0.0,
  "fast_ci": [
    0.0,
    0.049
  ]
}
### c1p5b
{
  "who": "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.1,
  "correct_rate": 0.2,
  "correct_ci": [
    0.125,
    0.304
  ],
  "fast_rate": 0.0,
  "fast_ci": [
    0.0,
    0.049
  ]
}
### c3b
{
  "who": "hf:Qwen/Qwen2.5-Coder-3B-Instruct",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.053,
  "correct_rate": 0.107,
  "correct_ci": [
    0.055,
    0.197
  ],
  "fast_rate": 0.0,
  "fast_ci": [
    0.0,
    0.049
  ]
}
### c7b
{
  "who": "hf:Qwen/Qwen2.5-Coder-7B-Instruct",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.14,
  "correct_rate": 0.28,
  "correct_ci": [
    0.191,
    0.39
  ],
  "fast_rate": 0.0,
  "fast_ci": [
    0.0,
    0.049
  ]
}
### c14b
{
  "who": "hf:Qwen/Qwen2.5-Coder-14B-Instruct",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.02,
  "correct_rate": 0.04,
  "correct_ci": [
    0.014,
    0.111
  ],
  "fast_rate": 0.0,
  "fast_ci": [
    0.0,
    0.049
  ]
}
### llama8b
{
  "who": "hf:meta-llama/Llama-3.1-8B-Instruct",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.02,
  "correct_rate": 0.04,
  "correct_ci": [
    0.014,
    0.111
  ],
  "fast_rate": 0.0,
  "fast_ci": [
    0.0,
    0.049
  ]
}
### fable
{
  "who": "api:us.anthropic.claude-fable-5-1",
  "n": 15,
  "k": 5,
  "trials": 75,
  "meanC": 0.293,
  "correct_rate": 0.36,
  "correct_ci": [
    0.261,
    0.473
  ],
  "fast_rate": 0.227,
  "fast_ci": [
    0.147,
    0.333
  ]
}
```
