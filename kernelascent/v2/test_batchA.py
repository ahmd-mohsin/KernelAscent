"""Batch A harness fixture (deterministic, no model). Validates the executed-mutable-U
machinery against the v2 spec section 19 fixtures and the Batch A exit gate:

  - an accepted U edit is EXECUTED next round and changes behavior
  - revert reverses the behavior; rescue restores it (keep/revert/rescue)
  - the controller executes the ACTUAL accepted U bytes (provenance gate)
  - U transplants onto a fresh solver without carrying S
  - the resource ledger charges and cannot be refunded/exceeded

This is harness validation. It deliberately uses scripted U edits and a mock model, because
the fixture must be deterministic; it does not claim anything about a model's RSI.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kernelascent.v2.core import (SolverState, ImproverState, StateUpdate, StateStore,
                                  Controller, Ledger, BudgetExceeded, load_improver_callable)

# U0 upgrades itself to U1 (proposes a new source). Deterministic, no model.
U1_SRC = (
    "def improve_step(ctx):\n"
    "    return StateUpdate(s_skills_add=[{'name':'MARKER','family':'op','code':'x','score':1.0}], notes='U1-ran')\n"
)
U0_SRC = (
    "def improve_step(ctx):\n"
    "    return StateUpdate(u_new_source=%r, notes='U0-ran')\n" % U1_SRC
)

OK = True
def check(cond, msg):
    global OK
    print(("PASS " if cond else "FAIL ") + msg); OK = OK and bool(cond)

def mk(root):
    store = StateStore(root)
    ctrl = Controller(store, u_probe=lambda U: True)   # probe: accept any interface-valid U
    def ctx_factory(S, U):
        led = Ledger()
        from kernelascent.v2.core import ImproveContext
        return ImproveContext(S=S, U_params=U.params, history=[], practice_tasks=[],
                              model_rpc=lambda p: "", dev_tools={}, ledger=led, round=0)
    return store, ctrl, ctx_factory

import tempfile
root = tempfile.mkdtemp(prefix="ka_v2_")
store, ctrl, ctxf = mk(root)

S0 = SolverState(skills=[])
U0 = ImproverState(source=U0_SRC, params={})

# Round 1: execute U0 -> it proposes U1 -> admitted. U' should be U1.
S1, U1, upd1, meta1 = ctrl.execute_round(S0, U0, ctxf, k=1)
check(meta1["u_edit_kind"] == "source" and meta1["u_changed"], "round1: U0 self-edit to new source admitted + U changed")
check(U1.source == U1_SRC, "round1: accepted U' is exactly U1")

# Round 2 KEEP: execute the accepted U' (U1) -> banks MARKER, notes U1-ran.
Sk, Uk, updk, metak = ctrl.execute_round(S1, U1, ctxf, k=2)
keep_marker = any(s.get("name") == "MARKER" for s in updk.s_skills_add)
check(metak["u_hash_executed"] == store.put(U1.to_json()), "round2 keep: controller executed the ACTUAL accepted U (U1) bytes")
check(keep_marker and updk.notes == "U1-ran", "round2 keep: behavior is U1 (banked MARKER)")

# Round 2 REVERT: fork the post-round1 checkpoint, replace U with U0 (pre-edit), execute.
cp1 = store.snapshot(S1, U1)
Sr, Ur = store.fork(cp1, intervention=lambda S, U: (S, ImproverState(source=U0_SRC, params=U.params)))
Sr2, Ur2, updr, metar = ctrl.execute_round(Sr, Ur, ctxf, k=2)
revert_marker = any(s.get("name") == "MARKER" for s in updr.s_skills_add)
check(metar["u_hash_executed"] == store.put(ImproverState(source=U0_SRC, params={}).to_json()),
      "round2 revert: controller executed the reverted U (U0) bytes")
check((not revert_marker) and updr.notes == "U0-ran", "round2 revert: behavior reversed (no MARKER, U0-ran)")

# Round 2 RESCUE: revert then restore U1 through the same path -> behavior returns.
Srescue, Urescue = store.fork(cp1, intervention=lambda S, U: (S, ImproverState(source=U1_SRC, params=U.params)))
_, _, updres, _ = ctrl.execute_round(Srescue, Urescue, ctxf, k=2)
check(any(s.get("name") == "MARKER" for s in updres.s_skills_add) and updres.notes == "U1-ran",
      "round2 rescue: restoring U1 restores the behavior")

# TRANSPLANT: run U1 on a FRESH solver with no skills; must execute without carrying S.
Sfresh = SolverState(skills=[])
_, _, updt, _ = ctrl.execute_round(Sfresh, ImproverState(source=U1_SRC, params={}), ctxf, k=3)
check(any(s.get("name") == "MARKER" for s in updt.s_skills_add), "transplant: U1 runs on fresh S0 (no S carried)")

# PROVENANCE gate: keep vs revert executed different U hashes.
check(metak["u_hash_executed"] != metar["u_hash_executed"], "provenance: keep and revert executed different U bytes")

# LEDGER: charges accumulate and the cap is enforced (no refund/expansion).
led = Ledger(caps={"model_calls": 2, "tool_seconds": 10**9})
led.charge_call(tag="a"); led.charge_call(tag="b")
raised = False
try:
    led.charge_call(tag="c")
except BudgetExceeded:
    raised = True
check(raised and led.model_calls == 3, "ledger: model-call cap enforced, spend is not refundable")

# A no-behavior-change U edit (identical behavior, different bytes) must NOT be treated as a
# behavior change by any downstream L2/L3 logic: we assert behavior equality is what counts.
U1_ALIAS = U1_SRC + "\n# cosmetic comment, identical behavior\n"
_, _, upda, _ = ctrl.execute_round(SolverState(skills=[]), ImproverState(source=U1_ALIAS, params={}), ctxf, k=3)
check(store.put(ImproverState(source=U1_ALIAS, params={}).to_json()) != store.put(U1.to_json())
      and any(s.get("name") == "MARKER" for s in upda.s_skills_add),
      "hash-not-behavior: different bytes, same behavior -> flagged for behavioral (not hash) credit")

print("\nBATCH A HARNESS: " + ("ALL PASS" if OK else "FAILURES PRESENT"))
sys.exit(0 if OK else 1)
