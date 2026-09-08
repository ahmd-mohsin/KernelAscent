"""Curate a GRADED CANDIDATE LADDER per curated task, to give the verifier-improvement RSI loop
HEADROOM (so Q doesn't pin at the ceiling). For each task, Fable emits N buggy variants ordered from
most-wrong to least-wrong; we MEASURE each variant's hidden (edge-weighted) pass-rate and keep a
spread of rungs. A weak verifier can't separate the high rungs from the reference; a stronger verifier
(more edge coverage) climbs the ladder -> Q rises gradually -> a better improver can out-compound a
worse one (F1/F2 can be nonzero). No test-taker model at RSI run time: reference + ladder = fixed bank.
"""
import os, sys, json, argparse, random, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3 import curated_loader as CL, rsi_verify as V
import curate_bedrock as CB

LADDER_TMPL = (
    "Here is a CORRECT Python function and its spec.\nSpec: {spec}\n"
    "Correct code:\n```python\n{ref}\n```\nExamples: {ex}\n"
    "Produce {n} DISTINCT buggy variants of it, ordered from MOST-wrong to LEAST-wrong. Every variant MUST: "
    "keep the exact signature named {fn}; return the CORRECT result on typical/random inputs; be WRONG only "
    "on EDGE cases. Variant 1 fails on relatively COMMON edges; each later variant fails on RARER edges "
    "(harder to catch). Pure stdlib, deterministic, no exec/eval/IO. "
    "Return ONLY JSON {{\"variants\": [\"def {fn}(...):...\", ...]}} with exactly {n} entries. No prose."
)


def measure(task, fn, rng, n=240):
    return V.continuous_grade(task, fn, rng, n_hidden=n, edge_frac=0.8)


def build_ladder(task, raw, rng, keep=5):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        variants = json.loads(m.group(0)).get("variants", [])
    except Exception:
        return []
    rungs = []
    for code in variants:
        fn = CL._compile(code, task["fn"])
        if not callable(fn):
            continue
        try:
            pr = measure(task, fn, rng)
        except Exception:
            continue
        # a genuine rung: mostly right (>=0.3) but not perfect (<0.999) -> wrong only on some edges
        if 0.3 <= pr < 0.999:
            rungs.append({"code": code, "pr": round(pr, 3)})
    # dedup by pr bucket + sort ascending; keep a spread
    rungs.sort(key=lambda r: r["pr"])
    seen = set(); uniq = []
    for r in rungs:
        b = round(r["pr"], 2)
        if b in seen:
            continue
        seen.add(b); uniq.append(r)
    return uniq[:keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="/tmp/instance_storage/kernelascent/dataset/tasks/public")
    ap.add_argument("--tiers", default="easy,medium"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--attempts", type=int, default=3); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--effort", default="xhigh"); ap.add_argument("--min-rungs", type=int, default=3)
    ap.add_argument("--min-span", type=float, default=0.2)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/ladders")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cur = CB.Curator("us.anthropic.claude-fable-5-1", args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    rid, mt = cur.resolve(); cur.resolve_reasoning()
    import boto3
    from botocore.config import Config as _Cfg
    cur.rt = boto3.Session(profile_name="bedrock").client("bedrock-runtime", region_name=args.region,
                                                          config=_Cfg(read_timeout=1800, connect_timeout=60, retries={"max_attempts": 1}))
    cur.resolved = (rid, min(mt, 32000))
    cur.reasoning = {"thinking": {"type": "adaptive"}, "output_config": {"effort": args.effort}}
    out = {}
    for tier in args.tiers.split(","):
        projs = CL.load_projects(args.curated, tier)
        tier_out = {}
        for task in projs:
            rng = random.Random(hash(task["name"]) % 10000)
            ex = "; ".join("%r->%r" % (a, o) for a, o in task["examples"][:2])
            best = []
            for at in range(args.attempts):
                raw = cur.generate(LADDER_TMPL.format(spec=task["spec"], ref=task.get("reference_src", ""),
                                   fn=task["fn"], ex=ex, n=args.n))
                rungs = build_ladder(task, raw, rng, keep=args.n)
                if len(rungs) > len(best):
                    best = rungs
                span = (best[-1]["pr"] - best[0]["pr"]) if best else 0
                if len(best) >= args.min_rungs and span >= args.min_span:
                    break
            span = (best[-1]["pr"] - best[0]["pr"]) if best else 0
            ok = len(best) >= args.min_rungs and span >= args.min_span
            print("  %-7s %-32s rungs=%d prs=%s %s" % (tier, task["name"][:32], len(best),
                  [r["pr"] for r in best], "OK" if ok else "THIN"), flush=True)
            if best:
                tier_out[task["name"]] = best
        out[tier] = tier_out
        json.dump(out, open(os.path.join(args.outdir, "ladders.json"), "w"), indent=2)
        print("TIER %s: %d/%d tasks laddered" % (tier, len(tier_out), len(projs)), flush=True)
    print("DONE", {t: len(v) for t, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
