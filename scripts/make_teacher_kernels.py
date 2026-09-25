#!/usr/bin/env python3
"""Harvest VERIFIED-CORRECT kernels from a strong model, for the coverage-injection control (E1).

The compounding null says carrying a self-training lineage forward does not beat a matched
reset. That null is only meaningful if the harness COULD have registered compounding -- and
that has never been demonstrated. A benchmark that returns zero for every method is
indistinguishable from a broken one, and this project has already been burned exactly that
way (a wrong KA_ROOT made the grader score every kernel False, which read as a scientific
null for days).

The positive control: inject correct kernels the student CANNOT produce itself, on tasks
where its own success probability is zero, and ask whether lineage-minus-reset then goes
positive. If it does, the instrument has demonstrated dynamic range and the null is a
finding. If it does not, the loop is still broken and the null must not be published.

This script produces the injection material: run a teacher over the bank, grade every
candidate on GPU, and keep only the ones that verify. One pass, saved to JSON, reused by
every downstream arm -- so the expensive part happens once.

  python3 scripts/make_teacher_kernels.py --model Qwen/Qwen2.5-Coder-14B-Instruct \\
      --gpus 0,1 --k 8 --out /scratch/.../teacher_kernels.json
"""
import os, sys, json, argparse, time
from kernelascent import provenance as PROV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="teacher HF id (a model strong enough to solve what the student cannot)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--k", type=int, default=8, help="candidates per task; more = better coverage, linearly more time")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke testing)")
    a = ap.parse_args()

    from kernelascent.v3 import lab_weight_rsi as W
    from kernelascent.v3 import lab_kernel as LK
    from kernelascent import agent_bench as AB

    names = list(LK.TASKS)
    if a.limit:
        names = names[:a.limit]
    print("TEACHER %s over %d tasks, k=%d" % (a.model, len(names), a.k), flush=True)

    attempts = {}          # per-task: candidates, kernel attempts, kernel verifies
    out, t0 = {}, time.time()

    # RESUME. The artifact is rewritten after every task and carries `tasks_attempted`, so a
    # walltime kill leaves a usable prefix -- but without this the next chunk started at task 0
    # and threw it away. That is affordable at 0.5B and not at 32B, where loading the weights
    # alone costs an hour and a 3-hour chunk harvests 29 tasks exactly once.
    _start = 0
    if os.path.exists(a.out):
        try:
            _d = json.load(open(a.out))
            if not _d.get("complete") and _d.get("tasks_attempted"):
                out = _d.get("kernels") or {}
                attempts = _d.get("attempts") or {}
                _start = int(_d["tasks_attempted"])
                print("RESUME  %d of %d tasks already harvested -- continuing at task %d"
                      % (_start, len(names), _start), flush=True)
            elif _d.get("complete"):
                print("already complete (%d tasks); nothing to do" % _d.get("tasks_attempted", 0), flush=True)
                return 0
        except Exception as e:
            print("RESUME failed (%r) -- starting clean" % e, flush=True)
            out, attempts, _start = {}, {}, 0

    tok, mdl = W.build(a.model, tuple(int(g) for g in str(a.gpus).split(",") if g != ""))

    for i, name in enumerate(names):
        if i < _start:
            continue
        src = LK.TASKS[name]
        try:
            # adapter=False: the teacher is frozen, we only want what it can already do
            raw = W.generate_batch(tok, mdl, [src], a.k, adapter=False)[0]
            # A model emits prose + a fenced code block, not a bare class. eval_tasks does this
            # same extraction before grading; skipping it hands the grader unparseable text and
            # EVERY candidate fails to build -- which looks exactly like "the teacher can't solve
            # anything" (0/29 on a model whose published coverage is 86%).
            codes = [c for c in (AB.extract_modelnew(t) for t in raw) if c]
            if not codes:
                print("  [%d/%d] %-34s no parseable ModelNew in %d generations"
                      % (i + 1, len(names), name[:34], len(raw)), flush=True)
                continue
        except Exception as e:
            print("  [%d/%d] %-34s GEN FAIL %s" % (i + 1, len(names), name[:34], e), flush=True)
            continue
        try:
            graded = W._grade_isolated(src, codes)
        except Exception as e:
            print("  [%d/%d] %-34s GRADE FAIL %s" % (i + 1, len(names), name[:34], e), flush=True)
            continue
        # keep every correct candidate, best (fastest) first -- a task may need more than one
        # exemplar, and speed is the tiebreak we care about
        # ATTEMPT vs SUCCESS, per candidate. Only verified kernels are kept below, so a failed
        # attempt leaves no trace -- which made the T1 curve uninterpretable: 0.5B "solved" 8/29
        # and 14B 1/29, but half the small model's solves were plain-torch rewrites (a free pass
        # for declining the task) while 7B/14B attempted a real kernel every time. Counting "any
        # verifying submission" scores WILLINGNESS TO TAKE THE EASY ROUTE, which is
        # anti-correlated with instruction-following and hence with scale.
        def _is_kernel(c):
            return any(t_ in c for t_ in ("triton", "load_inline", "__global__"))
        n_attempt = sum(1 for c in codes if _is_kernel(c))
        n_kernel_ok = 0
        good = []
        for code, g in zip(codes, graded):
            ok, se = (list(g) + [0, 0])[:2]
            if ok:
                good.append({"code": code, "speedup_eager": round(float(se), 3),
                             "is_kernel": _is_kernel(code)})
                if _is_kernel(code):
                    n_kernel_ok += 1
        attempts[name] = {"n_candidates": len(codes), "n_attempted_kernel": n_attempt,
                          "n_kernel_verified": n_kernel_ok,
                          "n_verified": sum(1 for _c, g in zip(codes, graded) if (list(g) + [0])[0])}
        good.sort(key=lambda d: -d["speedup_eager"])
        if good:
            out[name] = good
        print("  [%d/%d] %-34s %d/%d correct%s" % (i + 1, len(names), name[:34], len(good), a.k,
              ("  best %.2fx" % good[0]["speedup_eager"]) if good else ""), flush=True)
        # `kernels` holds only SOLVED tasks, so its length is coverage, not progress -- a
        # half-finished harvest is indistinguishable from a complete one with poor coverage.
        # Record how far we actually got, and whether we reached the end. Reading a partial
        # artifact as a capability number is a live hazard: this file is rewritten every task.
        PROV.dump_atomic({"teacher": a.model, "k": a.k, "n_tasks": len(names), "kernels": out,
                   "attempts": attempts,
                   "tasks_attempted": i + 1, "complete": (i + 1) == len(names)},
                  a.out, indent=1)

    cov = len(out)
    print("\n=== TEACHER COVERAGE ===")
    print("  %d/%d tasks solved at least once (%.0f%%)" % (cov, len(names), 100 * cov / max(len(names), 1)))
    print("  %d verified kernels total" % sum(len(v) for v in out.values()))
    print("  %.1f min" % ((time.time() - t0) / 60))
    print("  wrote", a.out)
    if cov == 0:
        print("\nWARNING: the teacher solved nothing. Injection would be a no-op -- check the")
        print("bank, the grader (a copy-of-reference kernel must grade correct), and KA_ROOT.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
