# KernelAscent — open-ended RSI task bank

Hard list-of-int **synthesis** tasks that deliberately **share** number-theory / digit primitives
(primality, prime factorization, digit-sum / digital-root, palindrome, gcd, divisors, perfect squares,
run-length). Shared structure is the point: a reusable, *verified* helper for one task is useful for
others, so a growing **archive** of debugged abstractions can compound. Curated by Fable 5.1 and
executable-validated (reference runs on sampled inputs, outputs vary, references ≥1 shared primitive).

Used by the **open-ended / recursion track** (`kernelascent/v3/lab_open_live.py --bank`): the model writes
`solve(xs)` and may call a growing library of helpers it builds via a build-test-fix loop; graded on
edge-heavy hidden inputs so reusing a debugged helper beats an error-prone inline re-derivation.

## Schema (one JSON object per line)
`name` · `spec` (exact one-sentence return definition, incl. empty-input behavior) · `reference_code`
(`def f(xs)`, correct, stdlib-only, deterministic) · `sampler_code` (`def sample(rng)`) · `components`
(auto-tagged shared primitives) · `tier` · `id`.

## Splits
`public.jsonl` (committed + on HF at `rsi/public.jsonl`) is the development split. A held-out split
(`heldout/`, ~equal size) is **never published** and scores the leaderboard.

## Why these tasks
Strong models re-derive easy helpers inline, so an *optional* library adds nothing (F ≈ 0). These tasks
are edge-case-heavy and compositional so that (a) headroom exists below the frontier ceiling and (b) a
verified archived helper is load-bearing — the setting where recursive compounding can be measured live.

Code + full record: https://github.com/ahmd-mohsin/KernelAscent
