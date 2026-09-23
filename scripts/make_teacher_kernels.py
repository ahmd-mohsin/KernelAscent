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

    tok, mdl = W.build(a.model, tuple(int(g) for g in str(a.gpus).split(",") if g != ""))

    out, t0 = {}, time.time()
    for i, name in enumerate(names):
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
        good = []
        for code, g in zip(codes, graded):
            ok, se = (list(g) + [0, 0])[:2]
            if ok:
                good.append({"code": code, "speedup_eager": round(float(se), 3)})
        good.sort(key=lambda d: -d["speedup_eager"])
        if good:
            out[name] = good
        print("  [%d/%d] %-34s %d/%d correct%s" % (i + 1, len(names), name[:34], len(good), a.k,
              ("  best %.2fx" % good[0]["speedup_eager"]) if good else ""), flush=True)
        json.dump({"teacher": a.model, "k": a.k, "n_tasks": len(names), "kernels": out},
                  open(a.out, "w"), indent=1)

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
