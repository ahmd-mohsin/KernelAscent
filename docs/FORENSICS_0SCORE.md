# 0-score failure forensics — where every model family gets stuck (mechanistic)

For each model we sample K candidates on every task and classify **why** each scored 0. The 0-score 'correctness wall' below ~3B is really a **kernel-formation wall**: models fail before producing a valid kernel (`no_extract`, `syntax_error`), not by writing runnable-but-wrong kernels. As scale rises the dominant failure moves *downstream*: incoherence → truncation → API-hallucination → wrong-output → correct. Each scale step advances the model one stage down the pipeline.

| model | size | zero% | no_extract | syntax | name/API | wrong_out | correct | dominant stage |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | 99% | 72% | 27% | 0% | 0% | 0% | incoherence/format |
| DeepSeek-Coder-1.3B | 1.3B | 93% | 90% | 8% | 1% | 0% | 0% | incoherence/format |
| Qwen2.5-Coder-1.5B | 1.5B | 90% | 58% | 38% | 1% | 0% | 1% | incoherence/format |
| Yi-Coder-1.5B | 1.5B | 98% | 97% | 2% | 0% | 0% | 0% | incoherence/format |
| Qwen2.5-Coder-3B | 3B | 41% | 52% | 31% | 6% | 2% | 7% | incoherence/format |
| DeepSeek-Coder-6.7B | 6.7B | 53% | 79% | 11% | 3% | 1% | 4% | incoherence/format |
| Qwen2.5-Coder-7B | 7B | 38% | 9% | 75% | 5% | 0% | 8% | truncation/syntax |
| Qwen2.5-Coder-14B | 14B | 14% | 0% | 8% | 8% | 8% | 65% | correct |

## Qwen2.5-Coder-0.5B (0.5B) — example failure chains

## DeepSeek-Coder-1.3B (1.3B) — example failure chains

## Qwen2.5-Coder-1.5B (1.5B) — example failure chains

## Yi-Coder-1.5B (1.5B) — example failure chains

## Qwen2.5-Coder-3B (3B) — example failure chains

## DeepSeek-Coder-6.7B (6.7B) — example failure chains

## Qwen2.5-Coder-7B (7B) — example failure chains

## Qwen2.5-Coder-14B (14B) — example failure chains
