"""TRACK C — agent-procedure RSI (open AND closed models).

A closed/API model cannot change its weights, but it can improve its own EXECUTABLE RESEARCH PROCEDURE.
Here the editable state U is the agent's procedure:
  U.strategies   a list of optimization strategy notes the agent maintains
  U.archive      verified fast kernels it has accumulated (name -> ModelNew source)
The FIXED foundation model plays two roles through the same API:
  develop(U, task)      write a kernel for `task`, conditioned on U (strategies + archived exemplars); GPU-graded
  improve(actor, target) the actor REFLECTS on recent evidence and REWRITES the target's strategy library -> child U'
The improved procedure U' governs the NEXT round's solving. This is procedure-level self-improvement with a
frozen model, the STOP / HyperAgents setting.

Causal estimators (actor/target separated), evaluated on a held-out task set:
  Q(U)   = mean best-of-k capability C of develop(U, .) over held tasks     (eager+compiled graded)
  F_g    = Q(U_g) - Q(U_{g-1})   newer procedure vs older procedure (does the improved producer build better?)
  N_g    = Q(U_g) - Q(T)         child procedure vs the unchanged target it was revised from
Controls:
  frozen         U never changes (base procedure), fresh API samples each round  -> value of more search alone
  archive-only   archive grows, but the improve() strategy-rewrite is DISABLED   -> value of memory without reflection
  self-modify    improve() rewrites strategies each round                        -> value of procedure recursion
Grading reuses the crash-isolated GPU grader (grade_batch, eager+compiled). Generation uses curate_bedrock.Curator
(API models) or any callable. Fixed weights throughout; the object of study is the agent SYSTEM's procedure.
"""
import os, sys, json, argparse, random, statistics, time, hashlib, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
from kernelascent import agent_bench as AB

SOLVE_SYS = ("You are an expert GPU performance engineer. You write correct, fast, self-contained PyTorch/Triton "
             "kernels. Import only torch and torch.nn (optionally triton, triton.language). Return ONLY one "
             "class ModelNew(nn.Module) in a python code block.")

def _solve_prompt(U, task_src):
    strat = "\n".join("- " + s for s in U["strategies"][:12]) if U["strategies"] else "(none yet)"
    arch = ""
    if U["archive"]:
        arch = "\nVERIFIED fast kernels you built before (reuse their techniques):\n" + \
               "\n".join("# %s\n%s" % (n, s) for n, s in list(U["archive"].items())[:3])
    return ("Your current optimization strategy notes:\n%s\n%s\n\nOptimize this module for an A100, numerically "
            "equivalent. Return ONLY class ModelNew(nn.Module).\n%s" % (strat, arch, task_src))

IMPROVE_SYS = ("You are improving your OWN kernel-optimization research procedure. You are given your current "
               "strategy notes and evidence from recent attempts (which kernels were correct, their speedups, and "
               "failures). Rewrite the strategy notes into a BETTER, concrete, general list that will make your next "
               "kernels correct AND faster. Return ONLY a JSON list of short strategy strings.")

def _improve_prompt(U, evidence):
    ev = "\n".join(evidence[:24])
    cur = json.dumps(U["strategies"], indent=0)
    return ("Current strategy notes (JSON list):\n%s\n\nEvidence from recent attempts:\n%s\n\nReturn ONLY an "
            "improved JSON list of concise strategy strings (<=12), most useful first." % (cur, ev))


def _grade(src, codes, grade_gpu="0"):
    """crash-isolated GPU grade -> [(ok, sp_eager, sp_compiled), ...]"""
    if not codes:
        return []
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump({"task": src, "codes": codes}, open(path, "w"))
    env = dict(os.environ, PYTHONPATH="/tmp/instance_storage:/tmp/instance_storage/kernelascent",
               HF_HOME="/tmp/instance_storage/ka_data/hf", CUDA_VISIBLE_DEVICES=grade_gpu)
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "grade_batch.py"), path],
                           capture_output=True, text=True, timeout=900, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        res = json.loads(line[-1][len("RESULT"):]) if line else []
    except Exception:
        res = []
    os.remove(path)
    return res + [[False, 0.0, 0.0]] * (len(codes) - len(res))


def _score(ok, se):                                   # capability C, eager-parity=0.5, 1.5x eager=1.0
    return 0.0 if not ok else max(0.0, min(1.0, 0.5 + 0.5 * min(1.0, (se - 1.0) / 0.5)))


def develop(U, names, tasks, gen, k, grade_gpu):
    """Best-of-k capability over `names`; returns mean C, and evidence + newly verified fast kernels."""
    scores = []; evidence = []; verified = {}
    for n in names:
        src = tasks[n]
        codes = [c for c in (AB.extract_modelnew(gen(_solve_prompt(U, src), SOLVE_SYS) or "") for _ in range(k)) if c]
        res = _grade(src, codes, grade_gpu)
        best = 0.0; best_code = None; best_sp = 0.0
        for code, g in zip(codes, res):
            ok, se, sc = (g + [0, 0, 0])[:3]
            s = _score(ok, se)
            if s > best:
                best = s
            if ok and sc > best_sp:
                best_sp = sc; best_code = code
        scores.append(best)
        evidence.append("task %s: best_C=%.2f best_compiled_sp=%.2f n_correct=%d/%d" %
                        (n, best, best_sp, sum(1 for g in res if g and g[0]), len(codes)))
        if best_code is not None and best_sp > 1.0:
            verified["fast_%s" % n] = best_code
    return (statistics.mean(scores) if scores else 0.0), scores, evidence, verified


def improve(U, evidence, gen):
    """Actor rewrites its strategy library from evidence -> child procedure U' (fixed model weights)."""
    out = gen(_improve_prompt(U, evidence), IMPROVE_SYS) or ""
    m = None
    import re
    mm = re.search(r"\[.*\]", out, re.S)
    if mm:
        try:
            m = json.loads(mm.group(0))
        except Exception:
            m = None
    child = {"strategies": (m if isinstance(m, list) and m else U["strategies"]),
             "archive": dict(U["archive"])}
    return child


def run(args):
    random.seed(args.seed)
    import curate_bedrock as CB
    _cache = {}
    def gen(user, system):
        cur = CB.Curator(args.model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))  # fresh Session re-reads creds file -> survives rotation
        if _cache:
            cur.resolved = _cache["r"]; cur.reasoning = _cache["rc"]
        else:
            cur.resolve(); cur.resolve_reasoning(); _cache["r"] = cur.resolved; _cache["rc"] = cur.reasoning
        for _ in range(3):
            o = cur.generate(user) or ""
            if o.strip() and not o.startswith("BEDROCK_ERROR"):
                return o
        return o
    from kernelascent.v3 import lab_kernel as LK       # standardized bank via KA_KERNEL_BANK
    tasks = LK.TASKS
    names = list(tasks); random.Random(1).shuffle(names)
    train, held = names[:args.n_train], names[args.n_train:]
    U0 = {"strategies": [], "archive": {}}
    print("TRACK-C %s mode=%s train=%d held=%d k=%d" % (args.model, args.mode, len(train), len(held), args.k), flush=True)
    Q0, _, _, _ = develop(U0, held, tasks, gen, args.k, args.grade_gpu)
    print("Q0 base-procedure held-out = %.3f" % Q0, flush=True)
    U = {"strategies": [], "archive": {}}; hist = []; prevU = None
    for r in range(args.rounds):
        t0 = time.time()
        _, _, evidence, verified = develop(U, train, tasks, gen, args.k, args.grade_gpu)   # solve train -> evidence
        if args.mode in ("archive-only", "self-modify"):
            U["archive"].update(verified)                                                  # accumulate memory
        prevU = {"strategies": list(U["strategies"]), "archive": dict(U["archive"])}
        if args.mode == "self-modify":
            U = improve(U, evidence, gen)                                                  # rewrite procedure
        Qg, _, _, _ = develop(U, held, tasks, gen, args.k, args.grade_gpu)                  # capability with U_g
        Fg = None
        if prevU is not None and r > 0:
            Qprev, _, _, _ = develop(prevU, held, tasks, gen, args.k, args.grade_gpu)       # producer transplant
            Fg = round(Qg - Qprev, 3)
        row = {"round": r, "Qg": round(Qg, 3), "delta_vs_base": round(Qg - Q0, 3), "F_g": Fg,
               "n_strategies": len(U["strategies"]), "n_archive": len(U["archive"])}
        hist.append(row)
        print("round %d Q=%.3f dBase=%+.3f F=%s strat=%d arch=%d (%.0fs)" %
              (r, Qg, Qg - Q0, ("%+.3f" % Fg if Fg is not None else "-"), len(U["strategies"]), len(U["archive"]), time.time() - t0), flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"model": args.model, "mode": args.mode, "Q0": Q0, "history": hist},
                  open(os.path.join(args.outdir, "track_c.json"), "w"), indent=2)
    fs = [h["F_g"] for h in hist if h["F_g"] is not None]
    print("\n=== TRACK-C SUMMARY (%s, %s) ===" % (args.model, args.mode))
    print("  Q0=%.3f  Q by round:" % Q0, [h["Qg"] for h in hist])
    print("  F (newer procedure beats older):", fs)
    print("  procedure-RSI?", "YES" if len(fs) >= 2 and statistics.mean(fs[-2:]) > 0.05 else "not resolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="API model id (e.g. us.anthropic.claude-fable-5-1)")
    ap.add_argument("--mode", default="self-modify", choices=["frozen", "archive-only", "self-modify"])
    ap.add_argument("--region", default="us-east-1"); ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--grade-gpu", default="0")
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/track_c")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
