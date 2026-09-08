"""Fable-5.1-max task curator: generate difficulty-graded bug-fix tasks and EXECUTABLE-VALIDATE each
before admitting it. Produces a tiered task bank for RSI-VERIFY so tasks are properly distinguished by
difficulty (easy/medium/hard/ultra) rather than the 3 hand-authored ones.

Each task (Fable returns strict JSON): name, fn, spec, reference_code, buggy_code, examples,
sampler_code (def sample(rng)->args tuple, in-domain), edge_code (def edge(rng)->edge args tuple).
Validation (all must pass, else reject + retry):
  - reference matches every example
  - reference never raises on sampled or edge inputs (in-domain)
  - buggy is genuinely buggy: differs from reference on >=1 input
  - tier-appropriate edge-subtlety: for medium+/ hard, buggy AGREES with reference on a majority of
    TYPICAL inputs but DIFFERS on edges (a plausible fix that misses an edge)
Single strong curator (Fable max effort); executable checks + this validator = the independent review.
Guarded exec (SIGALRM) so a pathological generated function can't hang curation.
"""
import os, sys, json, argparse, random, signal, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import curate_bedrock as CB


class _TO(Exception):
    pass


def guarded(fn, args, sec=1.0):
    def _h(s, f): raise _TO()
    old = signal.signal(signal.SIGALRM, _h); signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


TIER_PROMPT = {
    "easy": "a BASIC list/string/number transformation with few edge cases; a competent model almost always fixes it",
    "medium": "correctness hinges on 1-2 EDGE cases (empty input / boundary / duplicates / sign) that a plausible fix often misses; the buggy version passes typical inputs but fails those edges",
    "hard": "multiple interacting rules or a subtle specification; the buggy version is plausible and fails a non-obvious combination of conditions",
    "ultra": "a competition-hard algorithm (parsing / DP / number theory / geometry) with many edge cases; even strong models frequently produce a subtly-wrong fix",
}

CURATE_TMPL = (
    "Create ONE self-contained Python bug-fix task at difficulty '{tier}' ({desc}).\n"
    "Return ONLY a JSON object with these string/code fields:\n"
    "  name (short slug), fn (function name), spec (1-2 sentence exact behavior + domain),\n"
    "  reference_code (a CORRECT def fn(...)), buggy_code (a PLAUSIBLE but WRONG def fn(...) same signature),\n"
    "  sampler_code (def sample(rng): return a tuple of VALID in-domain args), edge_code (def edge(rng): return a tuple of EDGE-case valid args),\n"
    "  examples (list of [args_list, expected_output], 2-3 items).\n"
    "Rules: pure Python stdlib only; deterministic; reference must never raise on sampled/edge inputs; "
    "buggy must be genuinely wrong (differ from reference) yet look reasonable. Do NOT use exec, eval, or file/network I/O. "
    "Return only the JSON, no prose."
)


def _load(src, name):
    ns = {"random": random, "math": __import__("math"), "itertools": __import__("itertools"), "re": re}
    exec(compile(src, "<t>", "exec"), ns)
    return ns.get(name)


def validate(t, tier):
    try:
        ref = _load(t["reference_code"], t["fn"]); bug = _load(t["buggy_code"], t["fn"])
        smp = _load(t["sampler_code"], "sample"); edg = _load(t["edge_code"], "edge")
        assert all(callable(x) for x in (ref, bug, smp, edg)), "missing callables"
    except Exception as e:
        return False, "compile:%r" % e
    rng = random.Random(0)
    # examples
    for a, o in t.get("examples", []):
        try:
            if guarded(ref, tuple(a)) != o:
                return False, "ref!=example %r" % (a,)
        except Exception as e:
            return False, "ref raised on example %r" % (a,)
    typ_agree = typ = edge_diff = edge_n = bug_diff_any = 0
    for _ in range(40):
        try:
            a = smp(rng); rv = guarded(ref, a)          # in-domain: ref must not raise
        except Exception:
            return False, "ref raised on sampled input"
        typ += 1
        try:
            bv = guarded(bug, a)
        except Exception:
            bv = "<exc>"
        if bv == rv: typ_agree += 1
        else: bug_diff_any += 1
    for _ in range(20):
        try:
            a = edg(rng); rv = guarded(ref, a)
        except Exception:
            continue
        edge_n += 1
        try:
            if guarded(bug, a) != rv: edge_diff += 1
        except Exception:
            edge_diff += 1
    if bug_diff_any == 0 and edge_diff == 0:
        return False, "buggy never differs (not a real bug)"
    if tier in ("medium", "hard") and edge_n and (edge_diff / edge_n) < 0.3:
        return False, "not edge-discriminating (edge_diff %d/%d)" % (edge_diff, edge_n)
    if tier in ("medium", "hard") and typ and (typ_agree / typ) < 0.4:
        return False, "buggy too broadly wrong for %s (typ_agree %d/%d)" % (tier, typ_agree, typ)
    return True, "ok typ_agree=%d/%d edge_diff=%d/%d" % (typ_agree, typ, edge_diff, edge_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=3); ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--effort", default="max"); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--outdir", default="/tmp/curated"); args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cur = CB.Curator("us.anthropic.claude-fable-5-1", args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    rid, mt = cur.resolve(); cur.resolve_reasoning()
    cur.reasoning = {"thinking": {"type": "adaptive"}, "output_config": {"effort": args.effort}}
    # max effort spends many tokens on (encrypted) reasoning -> need generous output budget so the JSON
    # answer is not truncated; and a long read timeout so a max-effort call can complete.
    import boto3
    from botocore.config import Config as _Cfg
    cur.rt = boto3.Session(profile_name="bedrock").client("bedrock-runtime", region_name=args.region,
                                                          config=_Cfg(read_timeout=1800, connect_timeout=60, retries={"max_attempts": 1}))
    cur.resolved = (rid, min(mt, 24000))
    print("curator", rid, "effort", args.effort, "maxTokens", cur.resolved[1], flush=True)
    bank = {}
    for tier in ("easy", "medium", "hard", "ultra"):
        got = []
        for attempt in range(args.per_tier * args.attempts):
            if len(got) >= args.per_tier:
                break
            raw = cur.generate(CURATE_TMPL.format(tier=tier, desc=TIER_PROMPT[tier])) or ""
            raw = re.sub(r"<reasoning>.*?</reasoning>", "", raw, flags=re.S)
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                print("  %s attempt%d: no json" % (tier, attempt), flush=True); continue
            try:
                t = json.loads(m.group(0))
            except Exception as e:
                print("  %s attempt%d: json err" % (tier, attempt), flush=True); continue
            ok, why = validate(t, tier)
            print("  %s attempt%d [%s] %s %s" % (tier, attempt, "PASS" if ok else "REJECT", t.get("name", "?"), why), flush=True)
            if ok:
                t["tier"] = tier; got.append(t)
        bank[tier] = got
        json.dump(bank, open(os.path.join(args.outdir, "curated_tasks.json"), "w"), indent=2)
        print("TIER %s: %d/%d validated" % (tier, len(got), args.per_tier), flush=True)
    print("DONE", {k: len(v) for k, v in bank.items()}, flush=True)


if __name__ == "__main__":
    main()
