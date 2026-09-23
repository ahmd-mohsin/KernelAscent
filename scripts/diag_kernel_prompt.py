#!/usr/bin/env python3
"""Why does the KA_PROMPT=kernel arm produce 0/29?

'The model cannot write kernels' and 'the model writes kernels that trip one specific harness
constraint' look identical in a coverage number and are completely different findings. This
samples a few candidates under each prompt and classifies what actually comes back.

    python scripts/diag_kernel_prompt.py --tasks 3 --k 4
"""
import argparse, json, os, re, sys, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def classify(raw, code, graded):
    """Order matters: each class is only reached once the earlier ones are ruled out."""
    if not raw or not raw.strip():
        return "empty-generation"
    if not code:
        return "no-parseable-ModelNew"
    has_tri = "triton" in code or "load_inline" in code or "__global__" in code
    if not has_tri:
        return "plain-torch-rewrite (ignored the instruction)"
    if graded is None:
        return "grader-returned-nothing"
    ok = bool(graded[0])
    if ok:
        return "CORRECT custom kernel"
    return "custom kernel, FAILED verification"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from kernelascent import agent_bench as AB
    from kernelascent.v3 import lab_weight_rsi as W
    from kernelascent.v3 import lab_kernel as LK
    import importlib

    names = list(LK.TASKS)[: a.tasks] if hasattr(LK, "TASKS") else None
    if not names:
        bank = json.load(open(os.path.join(ROOT, "dataset", "kernel_bank", "kernel_tasks.json")))
        names = [t["name"] for t in bank][: a.tasks]
        srcs = {t["name"]: t["source"] for t in bank}
    else:
        srcs = {n: LK.task_src(n) for n in names}

    tok, mdl = W.build(a.model, tuple(int(g) for g in str(a.gpus).split(",") if g != ""))
    report = {}
    for prompt_mode in ("safe", "kernel"):
        os.environ["KA_PROMPT"] = prompt_mode
        importlib.reload(LK)                      # OPT is bound at import, so re-bind it
        importlib.reload(W)
        counts, samples = {}, []
        print("\n" + "=" * 78)
        print("KA_PROMPT=%s" % prompt_mode)
        print("=" * 78)
        for n in names:
            src = srcs[n]
            outs = W.generate(tok, mdl, src, a.k, adapter=False)
            codes = [AB.extract_modelnew(o) for o in outs]
            graded = W._grade_isolated(src, [c for c in codes if c])
            gi = 0
            for raw, code in zip(outs, codes):
                g = None
                if code:
                    g = graded[gi] if gi < len(graded) else None
                    gi += 1
                cls = classify(raw, code, g)
                counts[cls] = counts.get(cls, 0) + 1
                if len(samples) < 3 and code and "triton" in code:
                    samples.append({"task": n, "class": cls, "code": code[:1500]})
            print("  %-34s %s" % (n[:34], counts))
        report[prompt_mode] = {"counts": counts, "samples": samples}
        print("  TOTAL %s" % counts)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for m, d in report.items():
        tot = sum(d["counts"].values()) or 1
        print("  %-8s %s" % (m, "  ".join("%s=%d(%.0f%%)" % (k, v, 100 * v / tot)
                                          for k, v in sorted(d["counts"].items()))))
    print("""
  READ-OUT:
    'plain-torch-rewrite' dominant under kernel prompt -> the model DECLINES to write kernels
    'custom kernel, FAILED verification' dominant      -> it tries and cannot succeed:
        formation-limited, and the paper must report ambition and coverage separately
    'no-parseable-ModelNew' dominant                   -> a FORMAT failure, not a capability
        one -- fix the extractor before claiming anything about the model""")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
        print("\n  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(2)
