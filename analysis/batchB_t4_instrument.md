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
