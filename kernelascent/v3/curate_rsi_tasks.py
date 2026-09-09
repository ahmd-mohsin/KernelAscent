"""Curate the OPEN-ENDED RSI task bank: hard list-of-int synthesis tasks that COMPOSE a shared set of
number-theory/digit primitives (primality, factorization, digit-sum/digital-root, palindrome, gcd,
divisors, run-length). Shared components are the point: a reusable verified helper for one task helps
others -> a growing archive can compound. Curated by Fable 5.1, executable-validated + deduped.

Task JSON: {name, spec (exact one-liner), reference_code (def f(xs)), sampler_code (def sample(rng)->list),
components (auto-tagged), tier}. Validation: ref+sampler compile; ref runs on 24 sampled inputs w/o raising;
outputs vary (non-trivial); references >=1 shared primitive; buggy-free (deterministic).
"""
import os, sys, json, argparse, random, re, signal, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import curate_bedrock as CB

PRIMS = ["prime", "factor", "digit", "palindrom", "gcd", "divisor", "modulo", "square", "run"]
TMPL = (
    "Create ONE HARD Python task: a function f(xs) where xs is a list of ints (0..300, length 0..10).\n"
    "It must COMPOSE number-theory / digit operations (e.g. primality, prime factorization, digit-sum or "
    "digital-root, base-10 palindrome, gcd, divisors, perfect squares, run-length) so that a reusable helper "
    "for one operation would help solve it. Make it edge-case-heavy (empty list, 0/1/2, ties) so a naive "
    "one-shot attempt often gets edges wrong.\n"
    "Return ONLY JSON: {{\"name\": short-slug, \"spec\": one exact sentence defining the return value incl "
    "empty-input behavior, \"reference_code\": \"def f(xs): ...\" (CORRECT, stdlib only, deterministic, no "
    "I/O), \"sampler_code\": \"def sample(rng): return a list of ints\"}}. No prose.")


class _TO(Exception):
    pass


def _g(fn, a, sec=1.0):
    def h(s, f): raise _TO()
    old = signal.signal(signal.SIGALRM, h); signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn(*a)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


def _load(src, name):
    ns = {"__builtins__": __builtins__, "math": __import__("math")}
    exec(compile(src, "<t>", "exec"), ns); return ns.get(name)


def validate(t):
    try:
        f = _load(t["reference_code"], "f"); smp = _load(t["sampler_code"], "sample")
        assert callable(f) and callable(smp)
    except Exception as e:
        return False, "compile:%r" % e
    rng = random.Random(0); outs = []
    for _ in range(24):
        try:
            a = smp(rng)
        except Exception:
            return False, "sampler raised"
        if not isinstance(a, list):
            return False, "sampler not list"
        try:
            outs.append(repr(_g(f, (a,))))
        except Exception as e:
            return False, "ref raised %r" % e
    if len(set(outs)) < 3:
        return False, "outputs too constant (%d distinct)" % len(set(outs))
    comps = sorted({p for p in PRIMS if p in t["reference_code"].lower()})
    if not comps:
        return False, "no shared primitive referenced"
    t["components"] = comps; t["tier"] = "rsi"
    return True, "ok comps=%s distinct=%d" % (comps, len(set(outs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--effort", default="xhigh"); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/rsi_bank"); args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cur = CB.Curator("us.anthropic.claude-fable-5-1", args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    rid, mt = cur.resolve(); cur.resolve_reasoning()
    import boto3
    from botocore.config import Config as _C
    cur.rt = boto3.Session(profile_name="bedrock").client("bedrock-runtime", region_name=args.region,
                                                          config=_C(read_timeout=1800, connect_timeout=60, retries={"max_attempts": 1}))
    cur.resolved = (rid, min(mt, 32000)); cur.reasoning = {"thinking": {"type": "adaptive"}, "output_config": {"effort": args.effort}}
    got = []; seen = set()
    for attempt in range(args.n * args.attempts):
        if len(got) >= args.n:
            break
        raw = re.sub(r"<reasoning>.*?</reasoning>", "", cur.generate(TMPL) or "", flags=re.S)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            print("  attempt%d: no json" % attempt, flush=True); continue
        try:
            t = json.loads(m.group(0))
        except Exception:
            print("  attempt%d: json err" % attempt, flush=True); continue
        key = (t.get("name", "").lower().strip(), hashlib.sha1(re.sub(r"\s+", "", t.get("reference_code", "")).encode()).hexdigest())
        if key in seen:
            print("  attempt%d: DUP %s" % (attempt, t.get("name")), flush=True); continue
        ok, why = validate(t)
        print("  attempt%d [%s] %s %s" % (attempt, "PASS" if ok else "REJECT", t.get("name"), why), flush=True)
        if ok:
            seen.add(key); got.append(t)
            json.dump(got, open(os.path.join(args.outdir, "rsi_tasks.json"), "w"), indent=2)
    print("DONE %d/%d" % (len(got), args.n), flush=True)


if __name__ == "__main__":
    main()
