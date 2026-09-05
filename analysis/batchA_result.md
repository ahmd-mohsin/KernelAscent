# Batch A result: executed mutable improver U (exit gate MET)

Two validations, both required by the v2 spec.

## 1. Deterministic harness fixture (test_batchA.py) -- ALL PASS
Proves the machinery is correct, independent of any model:
- an accepted U edit is EXECUTED next round and changes behavior
- revert reverses the behavior; rescue restores it (keep/revert/rescue)
- the controller executes the ACTUAL accepted U bytes (provenance gate)
- U transplants onto a fresh solver without carrying S
- the resource ledger charges and cannot be refunded or exceeded
- different bytes with identical behavior are flagged for behavioral (not hash) credit

## 2. Real end-to-end with Fable 5.1 (run_real.py) -- exit gate MET
Real Bedrock model, real Medium kernel tasks, crash-isolated grading, 3 rounds:
- Fable edited its own improver U every round: target -> "wrong" (r0),
  admit_min_score 0.0 -> 0.5 (r1) -> 0.7 (r2)
- each edit was executed the next round; distinct_U_hashes = 4 (U0 + 3 evolved)
- ledger charged all 15 model calls
- VERDICT: "U self-modified and re-executed"

Observation: Fable's self-edit was arguably poor (raising the admission bar while banking 0
skills). That is a real, faithfully captured improver-policy decision -- exactly what the
transplant (L2) and ancestry (L3) experiments will evaluate. It is not a harness fault. The
open-weight (HF) adapter is being confirmed on Qwen2.5-Coder-7B on the same loop.

This is the v2 core the previous rsi_causal lacked: it evolved S (skill memory), never U.
Next: Batch B, one vertical T2/T3/T4 slice wired to a real vLLM service metric.
