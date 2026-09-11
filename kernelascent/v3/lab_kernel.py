"""LOAD-BEARING GPU-kernel RSI loop — the substrate where frontier genuinely fails (E0 fast-rate ~0.6),
so a VERIFIED fast kernel is hard-to-re-derive + easy-to-reuse = load-bearing by nature (unlike pure Python
where strong models re-derive one-shot).

State U = an ARCHIVE of VERIFIED fast kernels the agent has built (name -> ModelNew source that graded
correct + faster than the torch baseline). develop(U, task): the model optimizes a task's Model, shown its
archived fast kernels as reusable exemplars; graded on GPU (correct vs fp32-gold + speedup vs eager) ->
speed-resolved Q. revise(actor, target): the actor builds ONE new verified-fast kernel into the archive,
using ITS OWN archive as exemplars -> a richer archive yields faster kernels a poorer one can't reach ->
newer producer builds a better child on a common target (F1>0) and it can repeat (F2>0). Grading reuses the
crash-isolated harness (agent_bench.build_ref/grade); generation uses curate_bedrock.Curator (handles opus
quirks). Estimators = core.run_lineage.
"""
import os, sys, json, argparse, random, statistics, copy
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci
from kernelascent import agent_bench as AB

# reduction/normalization family so optimization PATTERNS transfer across tasks (shared structure)
def _task(fwd, M=4096, Dd=4096):
    return ("import torch, torch.nn as nn\nDT=torch.float16\nclass Model(nn.Module):\n"
            "    def __init__(self, dt=DT):\n        super().__init__(); self.dt=dt\n"
            "    def forward(self, x):\n        %s\n"
            "def get_inputs():\n    return [torch.randn(%d,%d, dtype=DT)]\n" % (fwd, M, Dd))

# ALL task CONTENT is Fable-5.1-curated (curate_kernel_tasks.py); we never hand-write tasks. Load the
# curated bank (KernelBench-style Model sources, tier-graded, GPU-validated). Override path via KA_KERNEL_BANK.
def _load_bank():
    import glob
    cands = [os.environ.get("KA_KERNEL_BANK", ""),
             "/tmp/instance_storage/ka_data/kernel_bank/kernel_tasks.json",
             os.path.join(ROOT, "dataset", "kernel_bank", "kernel_tasks.json")]
    for p in cands:
        if p and os.path.exists(p):
            bank = json.load(open(p))
            return {b["name"]: b["source"] for b in bank}, {b["name"]: b.get("tier", "L1") for b in bank}
    raise FileNotFoundError("Fable-curated kernel bank not found; run curate_kernel_tasks.py (searched %r)" % cands)

TASKS, TIER = _load_bank()
_REF = {}


def _ref(name):
    if name not in _REF:
        src = TASKS[name]
        ref, x, gold, rerr = AB.build_ref(src)
        bound = max(2e-2, 2 * rerr); tbase = AB.time_fn(lambda z: ref(z), (x,))
        _REF[name] = (src, ref, x, gold, bound, tbase)
    return _REF[name]


def _score(ok, sp, ceiling=1.5):
    """Correctness + HEADROOM-NORMALIZED speed. Correct-at-parity = 0.5; correct at the per-task achievable
    ceiling = 1.0, whether that ceiling is 1.1x or 40x. `ceiling` is the roofline-achievable speedup over the
    baseline (from the grader). This makes the score the MODEL's ceiling (did it capture the physically-available
    headroom), not a fixed 1.5x anchor that some tasks can't reach and others trivially saturate. Default 1.5
    recovers the legacy behavior when no ceiling is supplied."""
    if not ok:
        return 0.0
    room = max(ceiling - 1.0, 1e-6)
    return max(0.0, min(1.0, 0.5 + 0.5 * min(1.0, (sp - 1.0) / room)))


def _archive_text(lib):
    if not lib:
        return ""
    return ("You have these VERIFIED fast kernels from prior work — reuse/adapt their techniques:\n" +
            "\n".join("# fast %s:\n%s" % (n, s) for n, s in list(lib.items())[:3]) + "\n")


OPT = ("Optimize this PyTorch module for speed on an A100 GPU, numerically equivalent (same dtype, same "
       "output). The code must be SELF-CONTAINED and RUN AS-IS: import ONLY torch and torch.nn as nn "
       "(optionally `import triton` and `import triton.language as tl` — nothing else, no other triton "
       "submodules, no autocast wrappers, no external packages). Prefer fused torch ops; a plain-torch "
       "kernel that is correct beats a fancy one that errors. Define ONLY `class ModelNew(nn.Module)` with "
       "the same __init__/forward signature, in ONE python code block.\n{arch}\n{src}")


def make_behaviors(gen):
    cache = {}

    def _optimize(task_name, lib, rng):
        key = (task_name, tuple(sorted(lib)))
        if key in cache:
            return cache[key]
        src, ref, x, gold, bound, tbase = _ref(task_name)
        code = AB.extract_modelnew(gen(OPT.format(arch=_archive_text(lib), src=src)) or "")
        if not code:
            cache[key] = (0.0, None); return cache[key]
        try:
            ok, err, sp, msg = AB.grade(src, code, ref, x, gold, bound, tbase)
        except Exception:
            cache[key] = (0.0, None); return cache[key]
        cache[key] = (_score(ok, sp), (code, sp) if ok and sp > 1.02 else None)
        return cache[key]

    def develop(agent, task_name, rng):
        return _optimize(task_name, agent["params"]["lib"], rng)[0]

    def revise(actor, target, rng):
        child = copy.deepcopy(target); lib = child["params"]["lib"]
        # actor builds a verified fast kernel for a task NOT yet archived, using its OWN archive as exemplars
        todo = [t for t in TASKS if t not in lib] or list(TASKS)
        name = todo[rng.randrange(len(todo))]
        _, built = _optimize(name, actor["params"]["lib"], rng)   # actor's archive -> better exemplars
        if built is not None:
            lib[name] = built[0]                                  # add the verified fast kernel
        return child

    return develop, revise


def run(args):
    import curate_bedrock as CB
    cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    cur.resolve(); cur.resolve_reasoning(); who = "api:" + args.api_model
    def gen(p):
        o = ""
        for _ in range(4):
            o = cur.generate(p) or ""
            if o.strip():
                return o
        return o
    for n in TASKS:
        _ref(n)                                                   # warm baselines on GPU
    develop, revise = make_behaviors(gen)
    anchors = list(TASKS)
    U0 = {"params": {"lib": {}}}
    print("KERNEL-RSI %s tasks=%d lineages=%d" % (who, len(anchors), args.lineages), flush=True)
    results = []
    for s in range(args.lineages):
        r = run_lineage(copy.deepcopy(U0), develop, revise, anchors, random.Random(s), reps=1)
        results.append(r); agg = aggregate_lineages(results)
        print("lin%d Q0=%.3f q1-q0=%+.3f N1=%+.3f F1=%+.3f F2=%+.3f (aggF1=%+.3f F2=%+.3f)" %
              (s, r.Q["U0"], r.q1_minus_q0, r.N1, r.F1, r.F2, agg["F1"]["mean"], agg["F2"]["mean"]), flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"who": who, "lineages": s + 1, "Q0": _mean_ci([x.Q["U0"] for x in results]), "agg": agg},
                  open(os.path.join(args.outdir, "kernel_%s.json" % who.replace(":", "_").replace("/", "_").replace(".", "_")), "w"), indent=2)
    agg = aggregate_lineages(results)
    print("\n=== KERNEL-RSI %s ===" % who)
    for k in ("q1_minus_q0", "N1", "F1", "N2", "F2"):
        print("  %-12s %s" % (k, agg[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-model", required=True); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--lineages", type=int, default=8)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/kernel_rsi")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
