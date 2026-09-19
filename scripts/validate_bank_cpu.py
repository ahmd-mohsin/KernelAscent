#!/usr/bin/env python3
"""CPU semantic + degeneracy gate for a generated task bank (P2). No GPU required.

A generated task is only a benchmark item if it actually computes something. Speed grading needs a GPU,
but *semantics do not*: shape and device are parameters of a task, not properties of its computation. So
we re-materialise every task at a small shape on CPU and apply the same anti-reward-hacking gate the
self-play author faces, at bank-build time:

  builds        the source execs, defines DT / class Model / get_inputs(), and Model(...).forward runs
  finite        the output contains no NaN/Inf
  not_constant  the output actually varies (a constant-output task is gradeable but meaningless)
  not_identity  the output is not (numerically) the input -- a no-op task is free to "optimise"
  shape_ok      the output is a tensor with a sensible rank

Catching these on CPU is worth real GPU time: every degenerate task that reaches the cluster costs
~12 s per crash-isolated grade, times K candidates, times rounds.

DTYPE NOTE. Some ops (cumsum, topk, logsumexp) have incomplete half-precision CPU kernels. A task that
fails *only* because CPU lacks an fp16 path is NOT degenerate, so we retry it in float32 and mark it
`cpu_dtype_fallback` rather than rejecting it. Those still need GPU confirmation before use.

  python3 scripts/validate_bank_cpu.py --in dataset/kernel_bank/kernel_tasks_dsl.json \
                                       --out dataset/kernel_bank/kernel_tasks_dsl_validated.json
"""
import json, os, re, argparse, sys, traceback, collections

ROWS_V, HID_V = 64, 128          # validation shape: small, divisible by 8, even (group/rope families)


def rematerialise(src, rows=ROWS_V, hidden=HID_V):
    """Rewrite the bank's shape constants to a CPU-affordable size. The computation is untouched."""
    return re.sub(r"^ROWS,\s*HIDDEN\s*=\s*\d+\s*,\s*\d+\s*$",
                  "ROWS, HIDDEN = %d, %d" % (rows, hidden), src, flags=re.M)


def force_float32(src):
    return re.sub(r"^DT\s*=\s*torch\.\w+\s*$", "DT = torch.float32", src, flags=re.M)


def check_one(src, torch):
    """-> (verdict, detail). verdict in {ok, cpu_dtype_fallback, reject}."""
    def run(s):
        ns = {}
        exec(compile(s, "<task>", "exec"), ns)          # noqa: S102 -- bank sources are ours, generated
        if "Model" not in ns or "get_inputs" not in ns:
            raise RuntimeError("missing Model/get_inputs")
        model = ns["Model"]()
        xs = ns["get_inputs"]()
        if not isinstance(xs, (list, tuple)) or not xs:
            raise RuntimeError("get_inputs returned %r" % type(xs))
        with torch.no_grad():
            y = model(*xs)
        return xs[0], y

    def gate(x, y):
        if not hasattr(y, "shape"):
            return "reject", "output is not a tensor (%s)" % type(y).__name__
        if y.dim() < 1 or y.numel() == 0:
            return "reject", "degenerate output shape %s" % (tuple(y.shape),)
        yf = y.float()
        if not torch.isfinite(yf).all():
            return "reject", "non-finite output"
        if float(yf.std()) <= 1e-8:
            return "reject", "constant output (std=%.2e) -- meaningless task" % float(yf.std())
        if y.shape == x.shape:
            xf = x.float()
            denom = float(xf.abs().max()) + 1e-12
            if float((yf - xf).abs().max()) / denom < 1e-6:
                return "reject", "output == input (identity/no-op task)"
        return "ok", "std=%.3g" % float(yf.std())

    try:
        x, y = run(src)
        return gate(x, y)
    except Exception as e:
        first = "%s: %s" % (type(e).__name__, str(e).strip().splitlines()[0][:150] if str(e).strip() else "")
        # a missing half-precision CPU kernel is an environment limit, not a bad task
        if re.search(r'not implemented for|"?\w+_cpu"? not implemented|Half|BFloat16', first, re.I):
            try:
                x, y = run(force_float32(src))
                v, d = gate(x, y)
                return ("cpu_dtype_fallback" if v == "ok" else "reject"), "fp32 retry: " + d
            except Exception as e2:
                return "reject", "fp32 retry failed: %s: %s" % (type(e2).__name__, str(e2)[:120])
        return "reject", first


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--in", dest="inp", default=os.path.join(root, "dataset", "kernel_bank", "kernel_tasks_dsl.json"))
    ap.add_argument("--out", default=None, help="write the validated bank here (default: no write)")
    ap.add_argument("--report", default=None, help="write a per-task validation report here")
    ap.add_argument("--rows", type=int, default=ROWS_V)
    ap.add_argument("--hidden", type=int, default=HID_V)
    ap.add_argument("--keep-fallback", action="store_true",
                    help="keep cpu_dtype_fallback tasks in the validated bank (they need GPU confirmation)")
    a = ap.parse_args()
    try:
        import torch
    except ImportError:
        print("torch is required (CPU-only is fine): pip install torch")
        return 2
    torch.manual_seed(0)
    if not selftest(torch):
        print("\nABORT: the degeneracy gate failed its own self-test; a bank validated by a broken "
              "gate is worthless.")
        return 3
    print()

    bank = json.load(open(a.inp))
    print("validating %d tasks at %dx%d on CPU (torch %s)\n" % (len(bank), a.rows, a.hidden, torch.__version__))
    kept, report = [], []
    by_fam = collections.defaultdict(lambda: collections.Counter())
    tally = collections.Counter()
    for t in bank:
        src = rematerialise(t["source"], a.rows, a.hidden)
        verdict, detail = check_one(src, torch)
        tally[verdict] += 1
        by_fam[t.get("family", "?")][verdict] += 1
        report.append({"name": t["name"], "family": t.get("family"), "split": t.get("split"),
                       "verdict": verdict, "detail": detail})
        if verdict == "ok" or (verdict == "cpu_dtype_fallback" and a.keep_fallback):
            kept.append({**t, "cpu_validated": verdict})

    print("%-20s %6s %8s %10s %8s" % ("family", "n", "ok", "fallback", "reject"))
    print("-" * 58)
    for fam in sorted(by_fam):
        c = by_fam[fam]
        n = sum(c.values())
        print("%-20s %6d %8d %10d %8d" % (fam, n, c["ok"], c["cpu_dtype_fallback"], c["reject"]))
    print("-" * 58)
    print("%-20s %6d %8d %10d %8d" % ("TOTAL", len(bank), tally["ok"],
                                      tally["cpu_dtype_fallback"], tally["reject"]))

    rej = [r for r in report if r["verdict"] == "reject"]
    if rej:
        print("\nREJECTED (%d) -- first 15:" % len(rej))
        for r in rej[:15]:
            print("   %-52s %s" % (r["name"][:52], r["detail"][:90]))

    if a.out:
        splits = collections.Counter(t.get("split") for t in kept)
        fams = collections.Counter(t.get("family") for t in kept)
        json.dump(kept, open(a.out, "w"), indent=2)
        print("\nwrote %d validated tasks over %d families -> %s" % (len(kept), len(fams), a.out))
        print("  by split:", dict(splits))
    if a.report:
        json.dump({"n_in": len(bank), "n_kept": len(kept), "tally": dict(tally),
                   "by_family": {k: dict(v) for k, v in by_fam.items()}, "tasks": report},
                  open(a.report, "w"), indent=1)
        print("wrote report ->", a.report)
    return 0


# ---------------------------------------------------------------------------------------
# self-test: a gate that never rejects anything is indistinguishable from a broken gate,
# so the gate is checked against known-degenerate tasks before it is trusted.
# ---------------------------------------------------------------------------------------

_STUB = """import torch
import torch.nn as nn
DT = torch.float32
ROWS, HIDDEN = 8, 16

class Model(nn.Module):
    def __init__(self, dt=DT, hidden=HIDDEN):
        super().__init__()
    def forward(self, x):
        %s

def get_inputs():
    g = torch.Generator().manual_seed(1)
    return [torch.randn(ROWS, HIDDEN, generator=g).to(DT)]
"""

SELFTEST = [
    ("healthy",      "return torch.relu(x * 2.0) + 1.0",                     "ok"),
    ("constant",     "return torch.zeros_like(x)",                           "reject"),
    ("constant_ones", "return torch.ones_like(x) * 3.0",                     "reject"),
    ("identity",     "return x",                                             "reject"),
    ("identity_noop", "return x * 1.0",                                      "reject"),
    ("nan",          "return x / torch.zeros_like(x) * 0.0",                 "reject"),
    ("inf",          "return x / torch.zeros_like(x)",                       "reject"),
    ("not_a_tensor", "return float(x.sum())",                                "reject"),
    ("raises",       "raise RuntimeError('boom')",                           "reject"),
]


def selftest(torch):
    bad = 0
    print("gate self-test (known-degenerate tasks must be REJECTED):")
    for name, body, want in SELFTEST:
        got, detail = check_one(_STUB % body, torch)
        got_class = "ok" if got in ("ok", "cpu_dtype_fallback") else "reject"
        mark = "PASS" if got_class == want else "FAIL"
        if mark == "FAIL":
            bad += 1
        print("   [%s] %-14s want=%-7s got=%-7s  %s" % (mark, name, want, got_class, detail[:60]))
    print("   -> %d/%d gate checks passed" % (len(SELFTEST) - bad, len(SELFTEST)))
    return bad == 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import torch as _t
        raise SystemExit(0 if selftest(_t) else 1)
    raise SystemExit(main())
