# Batch B: T4 service instrument validated on real vLLM

Isolated-venv vLLM 0.10.2 (torch 2.8+cu128, base grader torch untouched) serving
Qwen2.5-Coder-7B on one A100-40GB. External streaming client, client-observed metrics,
every scheduled request counted, clean teardown.

Baseline (rate 8, unsaturated): offered 120, completed 120, errored 0, SLO-met 120,
goodput 7.65 rps, TTFT p50 27ms / p95 34ms, e2e p50 411ms / p95 902ms, wall 15.7s.

This is the trusted-measurement half of the Batch B exit gate: a real server + client + real
TTFT/TPOT/e2e/goodput. Next: an A/B at saturating load to confirm the instrument detects an
attributable goodput delta, then the agent-artifact -> vLLM patch integration (the remaining
Batch B piece: apply a generated kernel/config to the running server, move goodput, revert).

## A/B sensitivity (validated)
Saturating load (rate 60, 240 reqs, out 96, TTFT SLO 0.5s, e2e SLO 4s):
- baseline (default scheduling): goodput 43.06 rps, 240/240 SLO-met, e2e p95 2.13s
- throttled (--max-num-seqs 1): goodput 0.02 rps, 2/240 SLO-met, e2e p95 86.2s (queue collapse)

A known config change moves client-observed goodput by ~3 orders of magnitude and the
instrument captures it with correct completion/SLO denominators and clean teardown. The
measurement half of the Batch B exit gate is proven. Remaining Batch B piece: the
agent-generated artifact -> vLLM patch integration (apply a produced kernel/config to the
running server, move goodput, revert) -- the harness is ready for it.
