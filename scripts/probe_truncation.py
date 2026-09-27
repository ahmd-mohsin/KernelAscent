#!/usr/bin/env python3
"""Measure how often the generation budget truncates a kernel, as a function of model scale.

Why this exists. The size ladder reports an INVERTED capability curve: the frozen control arm
scores C_fresh 0.468 at 0.5B, 0.240 at 1.5B, 0.457 at 3B, 0.070 at 7B, 0.035 at 14B. Verified
yield collapses with it -- 12% of generations at 0.5B and 3B, 0.1% at 7B, 0.0% at 14B -- while
n_gens is identical (1050/round) at every scale. A 7B Qwen2.5-Coder does not write worse kernels
than a 0.5B, so that curve is an artifact, and lab_weight_rsi's own comment names the candidate:

    "larger models write longer, more elaborate kernels, so a fixed budget truncates them more
     often. An apparent capability curve can invert for no reason but the token limit."

The budget was already raised 900 -> 2048 once for this. This asks whether 2048 is still short.

It is a measurement, not an argument: for each model it generates against the REAL prompt
(LK.OPT via _prompt/_chat) with the REAL logits processor, then reports the completion-length
distribution, the share that run to the cap, and the share from which agent_bench can still
extract a ModelNew. Running the same tasks at a second, larger cap separates the two hypotheses:

  * truncation   -- share-at-cap is high and extraction rises sharply with the larger budget
  * incapability -- extraction stays low at both caps, and lengths sit well under the cap

  python3 scripts/probe_truncation.py --model Qwen/Qwen2.5-Coder-7B-Instruct --caps 2048,6144
"""
import argparse, json, os, sys, time
import torch

from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_kernel as LK
from kernelascent.v3 import lab_weight_rsi as W


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def run_cap(tok, mdl, srcs, k, cap, temp):
    """Generate k per src at this cap. Returns per-completion (n_tokens, hit_cap, extracted)."""
    rows = []
    tok.padding_side = "left"
    for src in srcs:
        text = W._chat(tok, W._prompt(src))
        enc = tok([text], return_tensors="pt").to(W._dev(mdl))
        with torch.no_grad():
            out = mdl.generate(**enc, do_sample=True, temperature=temp, top_p=0.95,
                               num_return_sequences=k, max_new_tokens=cap,
                               pad_token_id=tok.pad_token_id, logits_processor=W._LP)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = row.tolist()
            # strip trailing pad so a short completion is not counted as running to the cap
            while ids and ids[-1] == tok.pad_token_id:
                ids.pop()
            ntok = len(ids)
            txt = tok.decode(row, skip_special_tokens=True)
            rows.append((ntok, ntok >= cap, bool(AB.extract_modelnew(txt))))
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--caps", default="2048,6144", help="comma-separated max_new_tokens to compare")
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gpus = tuple(int(x) for x in a.gpus.split(",") if x != "")
    caps = [int(x) for x in a.caps.split(",") if x]

    # Same accessor the lab uses (lab_weight_rsi line 286), so the probe sees the same bank
    # KA_KERNEL_BANK selects. _load_bank() returns (TASKS, TIER); LK.TASKS is the name->source map.
    names = sorted(LK.TASKS)[: a.tasks]
    srcs = [LK.TASKS[n] for n in names]
    sys.stderr.write("bank=%d tasks, probing %r\n" % (len(LK.TASKS), names))

    t0 = time.time()
    tok, mdl = W.build(a.model, gpus)
    mdl.eval()
    sys.stderr.write("loaded %s in %.0fs\n" % (a.model, time.time() - t0))

    report = {"model": a.model, "tasks": len(srcs), "k": a.k, "caps": {}}
    print("=" * 78)
    print("%s   %d tasks x k=%d" % (a.model, len(srcs), a.k))
    print("=" * 78)
    print("%-8s %6s %8s %8s %8s %9s %10s" % ("cap", "n", "median", "p90", "max", "at-cap", "extracted"))
    for cap in caps:
        rows = run_cap(tok, mdl, srcs, a.k, cap, a.temp)
        lens = sorted(r[0] for r in rows)
        n = len(rows)
        med = lens[n // 2] if n else 0
        p90 = lens[int(n * 0.9)] if n else 0
        atcap = sum(1 for r in rows if r[1])
        ext = sum(1 for r in rows if r[2])
        print("%-8d %6d %8d %8d %8d %8.1f%% %9.1f%%"
              % (cap, n, med, p90, (lens[-1] if lens else 0), pct(atcap, n), pct(ext, n)))
        report["caps"][str(cap)] = {"n": n, "median": med, "p90": p90,
                                    "max": (lens[-1] if lens else 0),
                                    "at_cap_pct": round(pct(atcap, n), 2),
                                    "extracted_pct": round(pct(ext, n), 2)}
    # The discriminating line, stated rather than left to the reader.
    if len(caps) >= 2:
        lo, hi = str(caps[0]), str(caps[-1])
        d = report["caps"][hi]["extracted_pct"] - report["caps"][lo]["extracted_pct"]
        print("\nextraction %+.1f pp going %s -> %s tokens; %.1f%% still run to the larger cap"
              % (d, lo, hi, report["caps"][hi]["at_cap_pct"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=2)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
