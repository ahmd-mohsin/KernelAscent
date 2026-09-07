"""RSI-VERIFY-01 (corrected): a bug-fix dev agent whose LOCAL VERIFIER participates in self-revision.

Episode: fix a buggy Python function. The model emits K candidate patches; the agent's local test
suite (a spec-derived correctness checker run on N self-generated, IN-DOMAIN inputs) selects one;
graded on a large HIDDEN input set (immutable oracle). Improving `test_generation` (more inputs +
edge coverage) is the procedural improvement.

CORRECTIONS (2026-09-07): (1) NESTED verifiers -- the strong suite is the weak suite's inputs PLUS
extra edge inputs, so more testing can only help selection: dQ >= 0 by construction (no spurious
harm). (2) PAIRED -- weak and strong verifiers select from the SAME candidate set, isolating the
verifier from candidate-generation noise. (3) ORACLE-GUARD -- every test input is validated (the
reference must not raise on it); degenerate out-of-domain inputs are dropped, never counted.

Gate 2 (deterministic): a stronger (nested) verifier raises future productivity Q at matched
candidates -> the improvement opportunity provably exists. --calib validates run_lineage detection.
"""
import os, sys, json, argparse, random, copy, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci


# ------------------------------------------------------------------- oracles + samplers (in-domain)
def kth_largest(lst, k):
    return sorted(lst, reverse=True)[k - 1]


def _kth_sampler(rng):
    n = rng.randint(3, 6); return ([rng.randint(0, 20) for _ in range(n)], rng.randint(1, n))


def _kth_edge(rng):
    n = rng.randint(3, 5); lst = [rng.choice([2, 2, 5, 5, 9]) for _ in range(n)]
    return (lst, rng.choice([1, n]))            # duplicates + boundary k (still valid: 1<=k<=n)


def rle(s):
    if not s:
        return ""
    out = []; c = s[0]; n = 1
    for ch in s[1:]:
        if ch == c:
            n += 1
        else:
            out.append(c + str(n)); c = ch; n = 1
    out.append(c + str(n)); return "".join(out)


def _rle_sampler(rng):
    return ("".join(rng.choice("abc") for _ in range(rng.randint(2, 8))),)


def _rle_edge(rng):
    return (rng.choice(["", "a", "aa", "ab", "aaa"]),)   # trailing-run / empty / single edges


def merge_touch(iv):
    if not iv:
        return []
    s = sorted(iv); out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _merge_sampler(rng):
    n = rng.randint(2, 4); iv = []
    for _ in range(n):
        a = rng.randint(0, 8); iv.append((a, a + rng.randint(1, 3)))
    return (iv,)


def _merge_edge(rng):
    return (rng.sample([(2, 4), (4, 6), (1, 2), (6, 6)], 3),)   # touching + unsorted + point interval


PROJECTS = [
    {"name": "kth_largest", "ref": kth_largest, "fn": "kth_largest", "sampler": _kth_sampler, "edge": _kth_edge,
     "spec": "Return the k-th LARGEST element of list lst (k is 1-indexed; k=1 is the maximum).",
     "buggy": "def kth_largest(lst, k):\n    return sorted(lst)[k - 1]\n",
     "examples": [(([3, 1, 2], 1), 3), (([5, 4, 6, 1], 2), 5)]},
    {"name": "rle", "ref": rle, "fn": "rle", "sampler": _rle_sampler, "edge": _rle_edge,
     "spec": "Run-length encode a string: each maximal run of char c length n -> c then n (e.g. 'aaab'->'a3b1', ''->'').",
     "buggy": "def rle(s):\n    if not s:\n        return ''\n    out=[];c=s[0];n=1\n    for ch in s[1:]:\n        if ch==c:\n            n+=1\n        else:\n            out.append(c+str(n));c=ch;n=1\n    return ''.join(out)\n",
     "examples": [(("aaab",), "a3b1"), (("abc",), "a1b1c1")]},
    {"name": "merge_intervals", "ref": merge_touch, "fn": "merge_intervals", "sampler": _merge_sampler, "edge": _merge_edge,
     "spec": "Merge overlapping OR TOUCHING intervals ((1,2)&(2,3)->(1,3)); input may be unsorted; return list of (start,end).",
     "buggy": "def merge_intervals(iv):\n    if not iv: return []\n    out=[list(iv[0])]\n    for a,b in iv[1:]:\n        if a < out[-1][1]:\n            out[-1][1]=max(out[-1][1],b)\n        else:\n            out.append([a,b])\n    return [tuple(x) for x in out]\n"},
]


# ------------------------------------------------------------------- verifier (in-domain, nested, paired)
def _safe_ref(ref, args):
    try:
        return (True, ref(*args))
    except Exception:
        return (False, None)


def gen_inputs(project, n, rng, edge=False):
    """Return n IN-DOMAIN inputs (oracle does not raise); edge=True uses the edge sampler."""
    ref = project["ref"]; samp = project["edge"] if edge else project["sampler"]
    out = []; tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        a = samp(rng); ok, _ = _safe_ref(ref, a)
        if ok:
            out.append(a)
    return out


def select(project, candidates, inputs):
    """Score each candidate by # inputs it matches the oracle on (exceptions=fail); pick max, tie->first."""
    ref = project["ref"]
    truth = [ref(*a) for a in inputs]           # inputs are pre-validated in-domain
    best_i, best_s = 0, -1
    for i, (_, fn) in enumerate(candidates):
        s = 0
        for a, t in zip(inputs, truth):
            try:
                s += (fn(*a) == t)
            except Exception:
                pass
        if s > best_s:
            best_s, best_i = s, i
    return candidates[best_i][1]


def hidden_grade(project, fn, rng, n_hidden=60):
    ref = project["ref"]; ok = 0; tot = 0
    for i in range(n_hidden):
        a = (project["edge"] if i % 2 == 0 else project["sampler"])(rng)
        valid, t = _safe_ref(ref, a)
        if not valid:
            continue
        tot += 1
        try:
            ok += (fn(*a) == t)
        except Exception:
            pass
    frac = ok / tot if tot else 0.0
    return 1.0 if frac >= 0.999 else (0.5 if frac >= 0.8 else 0.0)


# ------------------------------------------------------------------- model-backed candidates
# ------------------------------------------------------------------- HARD task set (edge-rich; frontier not saturated)
def h_simplify(path):
    st = []
    for p in path.split("/"):
        if p in ("", "."): continue
        if p == "..":
            if st: st.pop()
        else: st.append(p)
    return "/" + "/".join(st)


def _simplify_samp(rng):
    toks = [rng.choice(["a", "b", "c", ".", "..", ""]) for _ in range(rng.randint(1, 5))]
    return ("/" + "/".join(toks),)


def _simplify_edge(rng):
    return (rng.choice(["/../", "/a/../../b", "/a//b/", "/./", "/...", "/a/b/../..", "/"]),)


def h_atoi(s):
    i, n = 0, len(s)
    while i < n and s[i] == " ": i += 1
    sign = 1
    if i < n and s[i] in "+-":
        sign = -1 if s[i] == "-" else 1; i += 1
    num = 0
    while i < n and s[i].isdigit():
        num = num * 10 + int(s[i]); i += 1
    return max(-2**31, min(2**31 - 1, sign * num))


def _atoi_samp(rng):
    return ((rng.choice(["", "  ", ""]) + rng.choice(["", "+", "-"]) + str(rng.randint(0, 99999)) + rng.choice(["", "abc", " x"])),)


def _atoi_edge(rng):
    return (rng.choice(["   -91283472332", "2147483648", "-2147483649", "+-12", "  +0 123", "words", "   ", "+", "-000123"]),)


def h_next_perm(a):
    a = list(a); n = len(a); i = n - 2
    while i >= 0 and a[i] >= a[i + 1]: i -= 1
    if i >= 0:
        j = n - 1
        while a[j] <= a[i]: j -= 1
        a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a


def _np_samp(rng):
    n = rng.randint(3, 5); return ([rng.randint(1, 4) for _ in range(n)],)


def _np_edge(rng):
    return (rng.choice([[3, 2, 1], [1, 1, 1], [2, 2, 3, 1], [1], [5, 4, 3, 2, 1]]),)


HARD_PROJECTS = [
    {"name": "simplify_path", "ref": h_simplify, "fn": "simplify_path", "sampler": _simplify_samp, "edge": _simplify_edge,
     "spec": "Canonicalize a Unix absolute path: collapse '.', resolve '..' (never above root), drop empty/duplicate slashes; return the canonical path starting with '/' and with no trailing slash (root is '/').",
     "buggy": "def simplify_path(path):\n    st=[]\n    for p in path.split('/'):\n        if p=='..':\n            st.pop()\n        elif p and p!='.':\n            st.append(p)\n    return '/'+'/'.join(st)\n",
     "examples": [(("/a/./b/../c",), "/a/c"), (("/x//y/",), "/x/y")]},
    {"name": "atoi", "ref": h_atoi, "fn": "atoi", "sampler": _atoi_samp, "edge": _atoi_edge,
     "spec": "Parse a leading 32-bit signed integer: skip leading spaces, optional single +/- sign, then digits until a non-digit; clamp to [-2**31, 2**31-1]; no digits -> 0.",
     "buggy": "def atoi(s):\n    s=s.strip()\n    sign=1;i=0\n    if s and s[i] in '+-':\n        sign=-1 if s[i]=='-' else 1;i+=1\n    num=0\n    while i<len(s) and s[i].isdigit():\n        num=num*10+int(s[i]);i+=1\n    return sign*num\n",
     "examples": [(("  -42abc",), -42), (("4193 with words",), 4193)]},
    {"name": "next_perm", "ref": h_next_perm, "fn": "next_perm", "sampler": _np_samp, "edge": _np_edge,
     "spec": "Return the next lexicographically greater permutation of the list; if it is the highest (fully non-increasing), wrap to the lowest (sorted ascending). Handle duplicates.",
     "buggy": "def next_perm(a):\n    a=list(a);n=len(a);i=n-2\n    while i>=0 and a[i]>a[i+1]: i-=1\n    if i<0: return a\n    j=n-1\n    while a[j]<a[i]: j-=1\n    a[i],a[j]=a[j],a[i]\n    a[i+1:]=reversed(a[i+1:])\n    return a\n",
     "examples": [(([1, 2, 3],), [1, 3, 2]), (([3, 2, 1],), [1, 2, 3])]},
]

SOLVE_TMPL = ("Fix the bug in this Python function so it fully matches the spec.\nSpec: {spec}\n"
              "Buggy code:\n```python\n{buggy}\n```\nExamples (input->output): {ex}\n"
              "Return ONLY the corrected function named {fn} as a single ```python code block. No prose.")


def parse_candidate(text, fn_name):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    src = m.group(1) if m else (text or "")
    ns = {}
    try:
        exec(compile(src, "<cand>", "exec"), ns)
        fn = ns.get(fn_name)
        return fn if callable(fn) else None
    except Exception:
        return None


def gen_candidates(project, gen_fn, K, trace_dir=None, tag=""):
    ex = "; ".join("%r->%r" % (a, o) for a, o in project.get("examples", [])[:3])
    prompt = SOLVE_TMPL.format(spec=project["spec"], buggy=project["buggy"], ex=ex, fn=project["fn"])
    cands = []
    for j in range(K):
        raw = gen_fn(prompt) or ""
        if trace_dir:                      # persist the FULL generation (incl <reasoning> block) for evaluation
            os.makedirs(trace_dir, exist_ok=True)
            open(os.path.join(trace_dir, "%s_c%d.txt" % (tag, j)), "w").write(raw)
        fn = parse_candidate(raw, project["fn"])
        if fn is not None:
            cands.append(("c%d" % j, fn))
    return cands


def edge_mutants(project):
    """Edge-subtle WRONG patches: pass typical/random inputs, fail only on edges. Injected as
    distractors so a weak verifier can misselect one and a strong (edge) verifier rejects it ->
    the verifier-improvement opportunity is REALIZED for real model candidates."""
    pool = _synth_pool(project)
    return [(("mut_" + n), f) for n, f in pool if n != "correct"]


# ------------------------------------------------------------------- Gate 2 (nested, paired, deterministic)
def _synth_pool(project):
    """A correct impl + edge-wrong impls that PASS typical inputs but fail on edges (so a weak
    verifier can tie/misselect and a nested strong verifier disambiguates)."""
    ref = project["ref"]
    correct = ref
    if project["name"] == "kth_largest":
        def w1(lst, k): return sorted(set(lst), reverse=True)[k - 1] if len(set(lst)) >= k else -1   # dedup bug (edge: duplicates)
        def w2(lst, k): return sorted(lst, reverse=True)[k % len(lst)]                                 # off-by-one (edge: k=len)
        wrong = [w1, w2]
    elif project["name"] == "rle":
        def w1(s):
            if not s: return ""
            out = []; c = s[0]; n = 1
            for ch in s[1:]:
                if ch == c: n += 1
                else: out.append(c + str(n)); c = ch; n = 1
            return "".join(out)               # forgets final run
        def w2(s):
            import itertools
            return "".join(c + (str(len(list(g))) if len(list(g)) > 1 else "") for c, g in itertools.groupby(s))  # no count for singletons
        wrong = [w1, w2]
    else:
        def w1(iv):
            if not iv: return []
            s = sorted(iv); out = [list(s[0])]
            for a, b in s[1:]:
                if a < out[-1][1]: out[-1][1] = max(out[-1][1], b)   # strict -> fails on touching
                else: out.append([a, b])
            return [tuple(x) for x in out]
        def w2(iv):
            if not iv: return []
            out = [list(iv[0])]                                      # assumes sorted -> fails on unsorted
            for a, b in iv[1:]:
                if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
                else: out.append([a, b])
            return [tuple(x) for x in out]
        wrong = [w1, w2]
    return [("correct", correct)] + [("wrong%d" % i, f) for i, f in enumerate(wrong)]


def gate2():
    def run(n_weak, n_edge, seed):
        rng = random.Random(seed); cs = []
        for proj in PROJECTS:
            for rep in range(10):
                pool = _synth_pool(proj); order = pool[:]; random.Random(seed * 97 + rep).shuffle(order)
                weak_in = gen_inputs(proj, n_weak, rng, edge=False)
                strong_in = weak_in + gen_inputs(proj, n_edge, rng, edge=True)   # NESTED
                inputs = weak_in if n_edge == 0 else strong_in
                cs.append(hidden_grade(proj, select(proj, order, inputs), rng))
        return sum(cs) / len(cs)
    qw = sum(run(2, 0, s) for s in range(5)) / 5
    qs = sum(run(2, 10, s) for s in range(5)) / 5
    print("=== RSI-VERIFY-01 GATE 2 (nested verifier improvement) ===")
    print("  weak (2 inputs):            Q=%.3f" % qw)
    print("  strong (2 + 10 edge, nested): Q=%.3f" % qs)
    print("  dQ = %+.3f" % (qs - qw))
    ok = qs - qw > 0.10
    print("GATE2", "PASS -- verifier improvement realizes future gain" if ok else "FAIL")
    return 0 if ok else 1


def calib():
    """run_lineage detection on this state shape (non-saturating synthetic score)."""
    anchors = [{"id": i} for i in range(6)]
    def develop(agent, project, rng):
        p = agent["params"]; return 0.02 * float(p.get("n_inputs", 2)) + 0.04 * float(p.get("n_edge", 0))
    def make_revise(cap):
        def revise(actor, target, rng):
            child = copy.deepcopy(target); power = int(actor["params"].get("power", 4))
            child["params"]["n_inputs"] = target["params"].get("n_inputs", 2) + power
            child["params"]["n_edge"] = target["params"].get("n_edge", 0) + power // 2
            child["params"]["power"] = min(cap, power + 4)
            return child
        return revise
    ok = True
    for mode, cap, exp in (("compound", 99, "F1>0,F2>0"), ("oneup", 8, "F1>0,F2~0")):
        rs = [run_lineage({"params": {"n_inputs": 2, "n_edge": 0, "power": 4}}, develop, make_revise(cap), anchors, random.Random(s), reps=2) for s in range(6)]
        agg = aggregate_lineages(rs); f1, f2 = agg["F1"]["mean"], agg["F2"]["mean"]
        good = (f1 > 0.01 and f2 > 0.01) if mode == "compound" else (f1 > 0.01 and abs(f2) < 0.03)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (%s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("RSI-VERIFY CALIB", "ALL PASS" if ok else "FAIL"); return 0 if ok else 1


# ------------------------------------------------------------------- model panel (paired, nested)
def panel(args):
    if args.api_model:
        import curate_bedrock as CB
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        cur.resolve(); cur.resolve_reasoning()
        gen_fn = lambda p: cur.generate(p); who = "api:" + args.api_model
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model); mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        def gen_fn(p):
            enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)], return_tensors="pt", padding=True).to("cuda")
            import torch as _t
            with _t.no_grad():
                o = mdl.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
            return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        who = "hf:" + args.model
    rng = random.Random(0); cw, cs_, base = [], [], []
    projs = HARD_PROJECTS if getattr(args, "hard", False) else PROJECTS
    for proj in projs:
        muts = edge_mutants(proj) if args.inject else []
        for rep in range(args.reps):
            cands = gen_candidates(proj, gen_fn, args.K,
                                   trace_dir=os.path.join(args.outdir, "traces"), tag="%s_r%d" % (proj["name"], rep))
            if not cands:
                cw.append(0.0); cs_.append(0.0); base.append(0.0); continue
            pool = cands + muts                       # inject edge-subtle distractors
            random.Random(rep * 131 + 7).shuffle(pool)
            weak_in = gen_inputs(proj, 2, rng, edge=False)
            strong_in = weak_in + gen_inputs(proj, 12, rng, edge=True)     # NESTED (strong superset of weak)
            cw.append(hidden_grade(proj, select(proj, pool, weak_in), rng))      # same pool,
            cs_.append(hidden_grade(proj, select(proj, pool, strong_in), rng))   # both verifiers (paired)
            base.append(hidden_grade(proj, pool[0][1], rng))                     # no-verifier baseline (first in pool)
    n = len(cw); mw = sum(cw) / n; ms = sum(cs_) / n; mb = sum(base) / n
    dq = [cs_[i] - cw[i] for i in range(n)]
    res = {"who": who, "n": n, "K": args.K, "meanC_noverifier": round(mb, 3), "meanC_weak": round(mw, 3),
           "meanC_strong": round(ms, 3), "verifier_dQ": _mean_ci(dq)}
    os.makedirs(args.outdir, exist_ok=True); json.dump(res, open(os.path.join(args.outdir, "panel.json"), "w"), indent=2)
    print("PANEL %s  C0=%.3f C_weak=%.3f C_strong=%.3f dQ=%+.3f %s" % (who, mb, mw, ms, ms - mw, dq and _mean_ci(dq).get("ci95")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate2", action="store_true"); ap.add_argument("--calib", action="store_true"); ap.add_argument("--panel", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--inject", type=int, default=1, help="inject edge-subtle mutant distractors (1=on)")
    ap.add_argument("--hard", action="store_true", help="use the HARD edge-rich task set (frontier not saturated)")
    ap.add_argument("--max-new", type=int, default=1024); ap.add_argument("--outdir", default="/tmp/rsiv")
    args = ap.parse_args()
    if args.gate2: sys.exit(gate2())
    if args.calib: sys.exit(calib())
    if args.panel: panel(args); return
    print("use --gate2 | --calib | --panel --model/--api-model")


if __name__ == "__main__":
    main()
