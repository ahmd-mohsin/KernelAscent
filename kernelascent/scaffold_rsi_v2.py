"""Scaffold-RSI v2 (closed-model-eligible, real-value): a frozen model recursively
improves its own optimization *strategy library* from GRADED feedback, and we measure
whether that produces TRANSFER (generalization to held-out tasks) that COMPOUNDS across
rounds vs a frozen-library control and a one-shot baseline.

Single host: Bedrock generation + GPU grading. Per round:
  - self arm  : optimize the held-out TRANSFER set with the current library -> grade -> C_self[k]
  - control   : optimize TRANSFER with an EMPTY library                     -> grade -> C_control[k]
  - self learns: optimize a disjoint PRACTICE set, grade, reflect on results -> grow library
C = fast_1 (fraction beating the min(eager,torch.compile) roofline). one-shot = C_self[0].
RSI value = C_self compounding (b>0) and Delta_k = C_self - C_control growing across rounds.
"""
import os, re, json, glob, argparse, subprocess, math
from concurrent.futures import ThreadPoolExecutor
import gen_source_tasks as G
import curate_bedrock as C

MAX_LIB = 15                          # cap accumulated library size
_PROSE_FIRST = {"okay", "first", "looking", "let", "let's", "here", "alright",
                "step", "so", "hmm", "well", "now", "given", "next", "based"}
# hallucinated / non-existent APIs weak models "learn" then reuse -> reject
_BOGUS_API = re.compile(
    r"triton\.jit\.\w+|#pragma|matmul_strided|heap_memory|cutlass kernel|"
    r"\btl\.(tanh|mean|matmul|constants|remainder|num_warps|Tensor|inline_assembly)\b|"
    r"torch\.backends\.cuda\.matmul_strided",
    re.I,
)


def sanitize_strategies(text, existing):
    """Extract clean, grounded strategy lines from a reflect() response.
    Drops reasoning-CoT leaks, prose, hallucinated-API lines, dupes; returns list[str]."""
    text = re.sub(r"(?is)<reasoning>.*?</reasoning>", " ", text or "")  # closed reasoning blocks
    text = re.sub(r"(?is)<reasoning>.*$", " ", text)                    # unclosed -> to EOF
    seen = {s.strip().lower() for s in existing}
    out = []
    for raw in text.splitlines():
        s = raw.strip().strip("-*•").strip()
        s = re.sub(r"^\**\s*\d+[\.\):]\s*", "", s).strip().strip("*").strip()  # drop "1." / "2)" / "**"
        low = s.lower()
        if not s or len(s) < 15 or len(s) > 240:
            continue
        if not re.search(r"[a-zA-Z]", s):                     # just numbers/punctuation
            continue
        if "<reasoning>" in low or "</reasoning>" in low:
            continue
        first = low.split()[0].strip(".,:;!?'\"") if low.split() else ""
        if first in _PROSE_FIRST or low.startswith("the user") or s.startswith("BEDROCK_ERROR"):
            continue
        if _BOGUS_API.search(s):                              # hallucinated API -> reject
            continue
        if low in seen:                                       # case-insensitive dedupe
            continue
        seen.add(low)
        out.append(s)
    return out

OPT = """Optimize this PyTorch module for speed on an A100. Keep __init__ identical; only rewrite forward.
Use Triton or fused PyTorch. Produce numerically equivalent output. Output ONE class named ModelNew in a single ```python block. No prose.

Optimization strategies you have learned so far (apply the relevant ones):
{lib}

Reference module:
```python
{src}
```"""


def render(lib):
    return "\n".join("- " + s for s in lib) if lib else "- (none yet)"


def gen_and_grade(cur, tasks, lib, outdir, workers, grader):
    os.makedirs(outdir, exist_ok=True)
    def do(t):
        d = os.path.join(outdir, t["name"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(t["source"])
        json.dump({k: t[k] for k in ("name", "tier", "family", "tags", "meta")}, open(d + "/meta.json", "w"), indent=2)
        raw = cur.generate(OPT.format(lib=render(lib), src=t["source"]))
        open(d + "/raw_0.txt", "w").write(raw or "")
        code = C.extract_modelnew(raw or "")
        if code:
            open(d + "/cand_0.py", "w").write(code)
        open(d + "/DONE", "w").write("1" if code else "0")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, tasks))
    summ = outdir + "/summary.json"
    try:
        subprocess.run(grader.format(candir=outdir, out=summ), shell=True, timeout=1200)
    except Exception:
        pass    # a hung compile shouldn't stall the loop; missing summary -> C treated as 0 this pass
    graded = json.load(open(summ)).get("tasks", []) if os.path.exists(summ) else []
    n = max(len(graded), 1)
    fast1 = sum(1 for t in graded if t.get("best_speedup_roofline", 0) > 1.0) / n
    return fast1, graded


def reflect(cur, graded, lib):
    notes = ["%s: pass=%d best_sp=%.2f (%s)" % (t["name"], t.get("pass_at_k", 0),
             t.get("best_speedup_roofline", 0), (t.get("best") or {}).get("reason", "")) for t in graded[:40]]
    prompt = ("You are refining your GPU-kernel optimization playbook from these graded outcomes:\n"
              + "\n".join(notes) +
              "\n\nAdd 1-3 NEW, concrete, generally-useful Triton/kernel strategies that would raise "
              "speedup or fix failures next time (one per line, terse). No prose.")
    out = cur.generate(prompt)
    add = sanitize_strategies(out or "", lib)[:3]     # up to 3 clean, grounded, novel lines
    lib = lib + add
    return lib[-MAX_LIB:] if len(lib) > MAX_LIB else lib   # keep most-recent MAX_LIB


def pick(seed0, fams, n):
    ts = G.generate_systematic(n_fusion=max(n, 4), seed0=seed0)
    if not fams:
        return ts[:n]
    F = [f.strip() for f in fams.split(",")]
    buckets = {f: [t for t in ts if t["family"] == f] for f in F}
    out, i = [], 0                                     # round-robin so no family is sliced out
    while len(out) < n and any(buckets[f] for f in F):
        f = F[i % len(F)]
        if buckets[f]:
            out.append(buckets[f].pop(0))
        i += 1
    return out[:n]


def fit_b(ys):
    ys = [y for y in ys if y is not None]
    if len(ys) < 3:
        return 0.0
    try:
        import numpy as np; return float(np.polyfit(range(len(ys)), ys, 2)[0])
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=os.environ.get("BEDROCK_PROFILE", "bedrock"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--practice-n", type=int, default=6)
    ap.add_argument("--transfer-n", type=int, default=6)
    ap.add_argument("--families", default="matmul,norm-act,attention")
    ap.add_argument("--practice-seed0", type=int, default=500000)
    ap.add_argument("--transfer-seed0", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grader", default="python grade_candidates.py --candir {candir} --out {out}")
    args = ap.parse_args()

    transfer = pick(args.transfer_seed0, args.families, args.transfer_n)   # held-out, scored
    cur = C.Curator(args.model_id, args.region, args.profile)
    cur.resolve(); cur.resolve_reasoning()
    print("model=%s id=%s max=%d reasoning=%s | practice=%d transfer=%d rounds=%d" %
          (args.model_id, cur.resolved[0], cur.resolved[1],
           bool(cur.reasoning and cur.reasoning != "__unset__"), args.practice_n, args.transfer_n, args.rounds), flush=True)

    lib, C_self, C_ctrl, log = [], [], [], []
    for k in range(args.rounds):
        rd = os.path.join(args.outdir, "round_%d" % k)
        cs, _ = gen_and_grade(cur, transfer, lib, rd + "/self_transfer", args.workers, args.grader)
        cc, _ = gen_and_grade(cur, transfer, [], rd + "/control_transfer", args.workers, args.grader)
        practice = pick(args.practice_seed0 + k * 1000, args.families, args.practice_n)   # fresh practice each round
        _, pg = gen_and_grade(cur, practice, lib, rd + "/self_practice", args.workers, args.grader)
        lib = reflect(cur, pg, lib)
        json.dump({"round": k, "library": lib}, open(rd + "/library.json", "w"), indent=2)
        C_self.append(cs); C_ctrl.append(cc)
        log.append(dict(round=k, C_self=cs, C_control=cc, delta=cs - cc, lib_size=len(lib)))
        print("round %d: C_self=%.3f C_control=%.3f delta=%+.3f lib=%d" % (k, cs, cc, cs - cc, len(lib)), flush=True)

    deltas = [s - c for s, c in zip(C_self, C_ctrl)]
    res = dict(model=args.model_id, rounds=args.rounds, params="n/a", role="scaffold-rsi",
               capability_r0=C_self[0] if C_self else None, capability_rN=C_self[-1] if C_self else None,
               compounding_b=fit_b(C_self), delta_final=(deltas[-1] if deltas else None),
               oneshot=C_self[0] if C_self else None, C_self=C_self, C_control=C_ctrl, deltas=deltas,
               verdict=("compounds" if len(deltas) >= 2 and deltas[-1] > deltas[0] and deltas[-1] > 0 else "plateau/none"),
               log=log, final_library=lib)
    json.dump(res, open(os.path.join(args.outdir, "scaffold_rsi_result.json"), "w"), indent=2)
    print("VERDICT:", res["verdict"], "compounding_b=%.4f delta_final=%s" % (res["compounding_b"], res["delta_final"]))


if __name__ == "__main__":
    main()
