# RSI v2 redesign: exact next steps

Grounded in `KernelAscent_RSI_v2_Redesign_Specification.md` (its Batches A-F, section 18, and
the completion checklist). The spec is a multi-week program; do not re-curate tasks or sweep
models first. Follow the exit-gated order. The biggest architectural risk is cheap to resolve
and comes first.

## The core gap to fix first

v2 requires a checkpoint A_k = (theta, S_k, U_k, H_k) where an accepted change to the improver
U is loaded and EXECUTED to generate the next round (spec 4.2). Our current `rsi_causal.py`
does not do this: `improve()` is fixed code that only grows a skill library (that is S, not U).
So nothing in the repo yet measures improver evolution. Everything else in v2 depends on this.

## Step 1 (Batch A, P0): make U a real, executed, mutable artifact

Local code + a deterministic CPU fixture. No API, minimal GPU. Fits the current window.

- Represent U as editable state: the improvement policy as data/code the agent can rewrite
  (which failures to study, candidate/repair strategy, test selection, admission rules, how S
  and U edits are proposed and locally compared). Store it content-addressed with a dependency
  manifest, separate from S.
- Controller loads the actual bytes of the accepted U_k and executes them next round
  (Execute_{theta,U_k}). Log: hash of U that generated each proposal, the S and U diffs
  proposed/accepted, which tests admitted them, and the first subsequent execution of the new U.
- Interfaces per spec 4.3: solve(), improve() (may edit both S and U), grade_submission(),
  snapshot(), fork(). Keep the official evaluator a separate trust boundary.
- Deterministic fixture: a known U edit changes next-round behavior; reverting it reverses the
  behavior; restoring it restores the behavior; the edit is transplantable without S.

Exit gate: an intentional improvement-procedure edit executes in the next generation, survives
checkpointing, and transplants without carrying S. (A fixture validates the harness, not any
model's RSI.)

## Step 2 (Batch B, P0): one complete vertical slice with a real service

- One T2 fused component, one T3 runnable block, one T4 vLLM-served workload.
- Arms R/F/B, one common-anchor probe, one prospective ancestry fork.
- The candidate patch must actually patch the pinned serving engine and be measured by an
  external client (goodput/TTFT/TPOT). Prove a deliberately injected known optimization moves
  the metric and an irrelevant change does not; prove a clean revert.

Exit gate: a generated artifact connects to a measured service effect with a reproducible
revert. The spec says do this BEFORE re-curating 128 projects; it resolves the largest risk
cheaply.

## Step 3 (Batch C, P0): harden verification

Numerical contracts (dtypes, tolerances, layouts, stateful behavior), private conformance
cases, trusted isolated paired timing (never trust a candidate-returned runtime), service
certification, resource-accounting and GPU/host-failure fixtures, official-evaluator isolation.

Exit gate: all release fixtures in spec section 19 behave correctly; no private feedback
reaches the agent.

## Step 4 (Batch D): re-curate the registry with Fable 5.1

This is the task recreation. 128 project roots across four tiers (T1 primitives, T2 fused
components, T3 inference blocks, T4 inference services), split by ROOT lineage (public practice
48, dev validation 16, private final 32, private probe 32). Each project: public contract,
starter implementation + integration point, public dev cases, private conformance + workload
panel, a pinned deployment baseline, declared budget/permissions, apply/exercise/revert path.
Use Fable 5.1 for the strong-optimizer headroom pass (evidence a better implementation exists),
NOT for score normalization. Freeze the registry before confirmatory runs.

Exit gate: every release project fits its deadline, passes a trusted baseline, has resolved
metadata and correct root-level split separation.

## Step 5 (Batch E): compact research pilot

Run one capable API endpoint and one weaker endpoint through 8 matched blocks of the Day24
profile. Inspect completion, unchanged-state variance, U-edit frequency, ancestry eligibility,
serving-packet precision. Exit gate: the report distinguishes no-gain / inconclusive /
invalid-protocol / positive-evidence. Freeze the protocol before confirmatory evaluation.

## Step 6 (Batch F): confirm and release

Preregister the confirmatory contrasts (U4 vs U0 on the balanced probe aggregate; service
goodput separately; checkpoints 0/2/4; materiality thresholds; ancestry rules; Holm
correction). Run untouched allocations. Publish harness, profile, accessible tasks, model
cards, result schema.

## Decisions needed from you

1. Scope confirmation: build in the A -> B -> C -> D order (defer the 128-task Fable
   re-curation until A/B/C pass), rather than recreating tasks first. The spec insists on this.
2. Track B stack: pin vLLM + two A100-40GB-fittable open-weight decode models (~3-4B and
   ~7-8B) per spec 6.4. Confirm or name alternates.
3. Compute reality: v2 is multi-week and needs persistent infra (Docker sandbox, model gateway,
   state store, service integration). The current 24h box + 9h creds fit Batch A (local) and a
   first Batch-B smoke, not the full program. Plan for repeated box/creds windows.
4. Third API reference and the fixed-endpoint model for the standard-harness division.

## What is safe to start immediately

Batch A. It is local code plus a CPU fixture, needs no API and little GPU, directly builds the
v2 core (executed mutable U + transplant-clean snapshots + the harness fixture), and is the
prerequisite for every later claim. Recommended to begin now.
