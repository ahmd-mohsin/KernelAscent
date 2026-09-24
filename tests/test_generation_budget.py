"""Every generation path must take its token budget from KA_MAX_NEW, not a literal.

lab_baselines._gen_custom hardcoded max_new=900 while W.generate_batch used KA_MAX_NEW=2048.
best_of_k went through one path and self_refine/retrieval through the other, so
`max(best_of_k, self_refine, retrieval)` -- the quantity weight-RSI is compared against --
was a maximum over arms with different token budgets. Nothing failed; the numbers were just
quietly incomparable. This gate makes that shape of defect fail loudly instead.
"""
import ast
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FROZEN INVENTORY of generation budgets that are still literals.
#
# These are NOT approved -- they are recorded. The ladder is not budget-comparable across
# rungs: T1/T2 generate at KA_MAX_NEW=2048, T3 and T5 at 1200, the probes lower still. Any
# cross-rung claim ("less RSI at T3 than T2") is partly confounded by this and must say so.
#
# They are deliberately NOT changed to read KA_MAX_NEW: KA_MAX_NEW=2048 is exported globally,
# and T3/T5 resume from per-round checkpoints, so flipping them now would put early rounds at
# 1200 and later rounds at 2048 INSIDE one cell -- a within-cell budget change is worse than a
# consistent wrong one. They change only at a clean restart of those rungs.
#
# The gate's job is to freeze this list: a new hardcoded budget, or a change to one of these
# values, fails. lab_baselines is absent on purpose -- it was the one case where two arms of a
# SINGLE comparison ran at different budgets, so it was fixed rather than recorded.
FROZEN = {
    ("kernelascent/v3/lab_combined_rsi.py", 103): 700,
    ("kernelascent/v3/lab_interp_probe.py", 48): 900,
    ("kernelascent/v3/lab_probe_intervene.py", 45): 900,
    ("kernelascent/v3/lab_reasoning_probe.py", 77): 1100,
    ("kernelascent/v3/lab_selfplay_rsi.py", 64): 1200,
    ("kernelascent/v3/lab_track_c.py", 146): 1200,
    ("kernelascent/v3/lab_wall_break.py", 49): 900,
    ("kernelascent/v3/lab_wall_causal.py", 26): 900,
    ("kernelascent/v3/lab_wall_causal.py", 49): 900,
}


def _int_literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


def test_no_hardcoded_generation_budget():
    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, "kernelascent", "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, ROOT)
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=rel)
        for node in ast.walk(tree):
            # a call like mdl.generate(..., max_new_tokens=2048)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in ("max_new_tokens", "max_new") and _int_literal(kw.value):
                        if FROZEN.get((rel, node.lineno)) == kw.value.value:
                            continue                      # recorded in the inventory, unchanged
                        bad.append("%s:%d  %s=%r passed as a literal" % (rel, node.lineno, kw.arg, kw.value.value))
            # a def like def _gen_custom(..., max_new=900)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                defaults = list(zip([p.arg for p in a.args[len(a.args) - len(a.defaults):]], a.defaults))
                defaults += list(zip([p.arg for p in a.kwonlyargs], [d for d in a.kw_defaults if d is not None]))
                for name, d in defaults:
                    if name in ("max_new_tokens", "max_new") and _int_literal(d):
                        bad.append("%s:%d  def %s(%s=%r) -- default must be None, resolved from KA_MAX_NEW"
                                   % (rel, node.lineno, node.name, name, d.value))
    assert not bad, (
        "generation budget hardcoded (or a frozen one moved); arms become incomparable:\n  "
        + "\n  ".join(bad))


def test_frozen_inventory_is_not_stale():
    """Every frozen entry must still exist. A stale entry means a budget moved unnoticed."""
    missing = []
    for (rel, lineno), val in sorted(FROZEN.items()):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append("%s no longer exists" % rel)
            continue
        lines = open(path, encoding="utf-8").read().splitlines()
        if lineno > len(lines) or ("%d" % val) not in lines[lineno - 1]:
            missing.append("%s:%d no longer holds %d -- re-verify and update the inventory"
                           % (rel, lineno, val))
    assert not missing, "frozen budget inventory is stale:\n  " + "\n  ".join(missing)


def test_gate_would_catch_the_real_regression():
    """Negative test: the gate must fire on the exact code that shipped the defect."""
    src = "def _gen_custom(tok, mdl, texts, k, max_new=900, temp=0.8, bs=4):\n    pass\n"
    tree = ast.parse(src)
    fn = tree.body[0]
    a = fn.args
    defaults = list(zip([p.arg for p in a.args[len(a.args) - len(a.defaults):]], a.defaults))
    hit = [n for n, d in defaults if n == "max_new" and _int_literal(d)]
    assert hit == ["max_new"], "gate would not have caught max_new=900"
