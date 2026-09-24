#!/usr/bin/env python3
"""Why do two thirds of generations fail to extract into a submission?

Per-candidate tracking showed 348 generations yielding 118 parseable candidates (34%) under the
kernel prompt. That is the LARGEST single loss in the pipeline -- bigger than the verification
failure it precedes -- and nothing in the harness records its cause, because `extract_modelnew`
returns None and the generation is dropped.

The extractor accepts a fenced ```python block containing `class ModelNew`, or an unfenced
occurrence of `class ModelNew`. So a generation can fail for at least five distinct reasons,
which are different problems with different fixes:

  empty                 the model emitted nothing
  truncated             hit the token limit mid-output (no closing fence, or cut mid-class)
  no-ModelNew           wrote code, but named the class something else / explained instead
  fenced-but-empty      produced a code block with no class in it
  prose-only            answered in prose without code

Distinguishing them matters: "raise max_tokens" and "the model cannot follow the format" are
opposite conclusions, and a 66% loss attributed to the wrong one is a wasted intervention.
"""
import argparse, json, os, re, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def classify(text, max_new):
    t = (text or "").strip()
    if not t:
        return "empty"
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", t, re.DOTALL)
    has_open_fence = t.count("```") % 2 == 1
    if "class ModelNew" in t:
        return "extracted"                         # the extractor's unfenced fallback catches it
    if has_open_fence or (len(t) > 0.9 * max_new * 3):
        # an odd number of fences, or a generation near the character budget, is truncation
        return "truncated"
    if fenced:
        joined = "\n".join(fenced)
        if "class " in joined:
            return "wrong-class-name"
        return "fenced-but-no-class"
    if re.search(r"\bdef |\bimport |\bclass ", t):
        return "code-without-ModelNew"
    return "prose-only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="/users/muahmed/ka_data/extraction_probe.json")
    a = ap.parse_args()

    from kernelascent.v3 import lab_kernel as LK
    from kernelascent.v3 import lab_weight_rsi as W
    from kernelascent import agent_bench as AB
    from kernelascent import provenance as PROV

    names = list(LK.TASKS)[: a.tasks]
    tok, mdl = W.build(a.model, tuple(int(g) for g in str(a.gpus).split(",") if g != ""))
    max_new = int(os.environ.get("KA_MAX_NEW", "1024"))

    counts = collections.Counter()
    classnames = collections.Counter()      # what the model NAMES its class when not ModelNew
    lens = []
    samples = {}
    for i, n in enumerate(names):
        raw = W.generate(tok, mdl, LK.TASKS[n], a.k, adapter=False)
        for t in raw:
            c = classify(t, max_new)
            if c == "wrong-class-name" and _ex(t, "lenient"):
                recovered += 1
            counts[c] += 1
            lens.append(len(t or ""))
            if c != "extracted" and c not in samples:
                samples[c] = (t or "")[:1200]
            if c == "wrong-class-name":
                # the actionable detail: is it a near-miss (ModelNew with different casing or
                # spacing) or a genuinely different name? Those need different fixes -- a looser
                # extractor versus a clearer prompt.
                # only inside fenced CODE -- a bare `class\s+(\w+)` over the whole generation
                # matched English prose ("the class using...", "the class with...") and reported
                # `using`, `with`, `inherits` as class names. A regex over mixed prose and code
                # needs to be told which it is reading.
                for b in re.findall(r"```(?:python)?\s*(.*?)```", t or "", re.DOTALL):
                    for nm in re.findall(r"^\s*class\s+(\w+)", b, re.M):
                        classnames[nm] += 1
        print("  [%d/%d] %-28s %s" % (i + 1, len(names), n[:28], dict(counts)), flush=True)

    # what a lenient extractor would recover, measured rather than assumed
    from kernelascent.agent_bench import extract_modelnew as _ex
    recovered = 0
    tot = sum(counts.values())
    print("\n=== EXTRACTION OUTCOMES (%s, k=%d over %d tasks) ===" % (a.model, a.k, len(names)))
    for c, k in counts.most_common():
        print("  %-22s %4d  (%.0f%%)" % (c, k, 100 * k / tot))
    print("  mean generation length: %d chars" % (sum(lens) / max(len(lens), 1)))
    if recovered:
        print("\n  LENIENT extraction would recover %d of %d wrong-class-name generations"
              % (recovered, counts.get("wrong-class-name", 0)))
        print("  -> extraction %.0f%% strict vs %.0f%% lenient"
              % (100 * counts.get("extracted", 0) / tot,
                 100 * (counts.get("extracted", 0) + recovered) / tot))
    if classnames:
        print("\n  CLASS NAMES emitted when it is not ModelNew:")
        for nm, k in classnames.most_common(8):
            near = "  <- near-miss" if nm.lower().replace("_", "") == "modelnew" else ""
            print("     %-28s %3d%s" % (nm, k, near))
    print("""
  READ-OUT
    truncated dominant        -> raise max_new_tokens. A cheap fix to the pipeline's biggest loss.
    wrong-class-name dominant -> the extractor is too strict, or the prompt's naming is not
                                 being followed; both are harness problems, not capability.
    prose-only dominant       -> the model is declining the format, which IS a capability
                                 statement and belongs in the results.""")
    PROV.dump({"model": a.model, "k": a.k, "n_tasks": len(names),
               "counts": dict(counts), "class_names": dict(classnames),
               "lenient_recovered": recovered,
               "mean_chars": sum(lens) / max(len(lens), 1),
               "samples": samples}, a.out, indent=1)
    print("\n  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
