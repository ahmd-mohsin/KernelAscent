#!/usr/bin/env python3
"""Build the 0-score failure-forensics mechanistic report from forensics_*.json (lab_failure_forensics output).
Emits docs/FORENSICS_0SCORE.md (per-family taxonomy + staged-failure narrative + verbatim example chains) and
docs/data/forensics_summary.json (compact, for the paper table). The headline mechanism: the sub-3B 0-score
'wall' is a kernel-FORMATION failure (no_extract/syntax dominate), not a kernel-CORRECTNESS failure; the failure
mode moves downstream with scale (incoherence -> truncation -> API-hallucination -> wrong-output -> correct)."""
import json, glob, os

NAMES = {"q05": ("Qwen2.5-Coder-0.5B", 0.5), "q15": ("Qwen2.5-Coder-1.5B", 1.5), "q3": ("Qwen2.5-Coder-3B", 3),
         "q7": ("Qwen2.5-Coder-7B", 7), "q14": ("Qwen2.5-Coder-14B", 14), "ds13": ("DeepSeek-Coder-1.3B", 1.3),
         "ds67": ("DeepSeek-Coder-6.7B", 6.7), "yi15": ("Yi-Coder-1.5B", 1.5), "oc15": ("OpenCoder-1.5B", 1.5),
         "mistral7": ("Mistral-7B", 7)}
BUCKETS = ["no_extract", "syntax_error", "name_error", "type_shape", "cuda_error", "oom", "runtime_other", "wrong_output", "correct"]
STAGE = {"no_extract": "incoherence/format", "syntax_error": "truncation/syntax", "name_error": "API-hallucination",
         "type_shape": "wiring", "cuda_error": "bad-kernel", "oom": "resource", "runtime_other": "runtime",
         "wrong_output": "silently-wrong", "correct": "correct"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    fdir = os.environ.get("KA_FOREN_DIR", "/tmp/ka/foren")
    rows = []
    for f in sorted(glob.glob(os.path.join(fdir, "forensics_*.json"))):
        d = json.load(open(f)); t = d["tag"]
        if t.endswith("c2"): continue  # 2nd-cycle stability runs, not in main table
        nm, sz = NAMES.get(t, (t, 99))
        rows.append((sz, nm, t, d))
    rows.sort()
    summ = {"models": []}
    md = ["# 0-score failure forensics — where every model family gets stuck (mechanistic)", "",
          "For each model we sample K candidates on every task and classify **why** each scored 0. The 0-score "
          "'correctness wall' below ~3B is really a **kernel-formation wall**: models fail before producing a "
          "valid kernel (`no_extract`, `syntax_error`), not by writing runnable-but-wrong kernels. As scale "
          "rises the dominant failure moves *downstream*: incoherence → truncation → API-hallucination → "
          "wrong-output → correct. Each scale step advances the model one stage down the pipeline.", "",
          "| model | size | zero% | no_extract | syntax | name/API | wrong_out | correct | dominant stage |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sz, nm, t, d in rows:
        p = d.get("failure_pct", {})
        dom = max(BUCKETS, key=lambda b: p.get(b, 0))
        md.append("| %s | %sB | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %s |" % (
            nm, sz, 100 * d.get("zero_task_frac", 0), p.get("no_extract", 0), p.get("syntax_error", 0),
            p.get("name_error", 0), p.get("wrong_output", 0), p.get("correct", 0), STAGE.get(dom, dom)))
        summ["models"].append({"model": nm, "size_b": sz, "tag": t, "zero_frac": d.get("zero_task_frac", 0),
                               "failure_pct": p, "dominant": dom})
    md.append("")
    # per-family example chains
    for sz, nm, t, d in rows:
        ex = d.get("examples", {})
        md.append("## %s (%sB) — example failure chains" % (nm, sz))
        for b in ["no_extract", "syntax_error", "name_error", "wrong_output"]:
            if ex.get(b):
                e = ex[b][0]; snip = (e.get("gen_tail") or e.get("code_tail", ""))[-300:]
                md.append("**%s** (%s) — task `%s`, error: `%s`" % (b, STAGE.get(b, b), e.get("task", "?"), str(e.get("error", "-"))[:80]))
                md.append("```\n...%s\n```" % snip)
        md.append("")
    open(os.path.join(ROOT, "docs", "FORENSICS_0SCORE.md"), "w").write("\n".join(md))
    json.dump(summ, open(os.path.join(ROOT, "docs", "data", "forensics_summary.json"), "w"), indent=1)
    print("wrote docs/FORENSICS_0SCORE.md (%d families) + docs/data/forensics_summary.json" % len(rows))


if __name__ == "__main__":
    main()
