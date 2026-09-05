"""Initial improver U_0 as executable source.

This whole file's text is stored as the initial ImproverState.source and executed each round
by the controller. It defines improve_step(ctx) -> StateUpdate. On each round it:

  1. Solves the practice tasks with the current solver S (via ctx.dev_tools) and banks verified
     skills  -> these are the S edits.
  2. Reflects on the graded practice outcomes and asks the model to improve U ITSELF, returning
     either tuned params or a full new U source -> these are the U edits (the recursion).

`StateUpdate` and `ImproveContext` are injected into this module's namespace by the controller
(kernelascent/v2/core.load_improver_callable), so no import is needed.

Params (ctx.U_params) the model may tune:
  target: which practice failures to prioritize ("compile"|"wrong"|"slow"|"any")
  admit_min_score: minimum log-interp score for a skill to be banked
  self_edit: "params" | "source" | "off"  -- how U proposes to change itself
"""


def improve_step(ctx):
    P = dict(ctx.U_params or {})
    target = P.get("target", "any")
    admit_min = float(P.get("admit_min_score", 0.0))
    self_edit = P.get("self_edit", "params")

    solve = ctx.dev_tools["solve"]              # (task, S) -> {"correct","score","code","reason","family","name"}
    banked = []
    outcomes = []
    for t in ctx.practice_tasks:
        r = solve(t, ctx.S)
        outcomes.append(r)
        if r.get("correct") and r.get("code") and r.get("score", 0.0) >= admit_min:
            banked.append({"name": "%s_r%d" % (r.get("family", "op"), ctx.round),
                           "family": r.get("family", "op"), "code": r["code"],
                           "score": r.get("score", 0.0)})

    # ---- reflect and propose a change to U itself (the self-modification) ----
    n = max(len(outcomes), 1)
    n_correct = sum(1 for r in outcomes if r.get("correct"))
    n_slow = sum(1 for r in outcomes if r.get("correct") and r.get("score", 0.0) <= 0.0)
    n_compile = sum(1 for r in outcomes if "compile" in str(r.get("reason", "")).lower())
    summary = ("practice: %d/%d correct, %d correct-but-not-faster, %d compile-failures; "
               "current U params: %s") % (n_correct, n, n_slow, n_compile, P)

    u_param_edits, u_new_source = {}, None
    if self_edit != "off":
        ask = (
            "You are improving your own optimization-improvement procedure (the policy U that "
            "decides how to turn graded practice outcomes into a better solver). Here is this "
            "round's evidence:\n" + summary + "\n\n"
            "Propose ONE concrete improvement to U as a compact JSON object. Allowed keys:\n"
            '  "target": one of "compile","wrong","slow","any" (which failures to prioritize next)\n'
            '  "admit_min_score": float in [0,1] (bar for banking a skill)\n'
            "Return only the JSON object, no prose.")
        raw = ctx.model_rpc(ask) or ""
        u_param_edits = _parse_param_edits(raw)

    return StateUpdate(
        s_skills_add=banked,
        s_param_edits={},
        u_param_edits=u_param_edits,
        u_new_source=u_new_source,
        notes=summary[:180],
    )


def _parse_param_edits(raw):
    import json, re
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    out = {}
    if d.get("target") in ("compile", "wrong", "slow", "any"):
        out["target"] = d["target"]
    try:
        v = float(d.get("admit_min_score"))
        if 0.0 <= v <= 1.0:
            out["admit_min_score"] = v
    except Exception:
        pass
    return out
