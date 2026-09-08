"""Load the Fable-curated code-task bank (dataset/tasks/public/<tier>.jsonl) into the PROJECTS shape
that rsi_verify + rsi_true consume: {name, ref(callable), fn, sampler(callable), edge(callable),
spec, buggy(source), examples[(args_tuple,out)], mutants[(name,fn)]}.

The curated buggy_code IS an edge-subtle wrong patch (curator-validated: agrees with reference on
typical inputs, differs on edges) -> it is injected as the distractor mutant so the verifier-
improvement opportunity is realized for real model candidates without hand-authored synth pools.
"""
import os, json, random, re, math, itertools


def _compile(src, name):
    ns = {"random": random, "math": math, "itertools": itertools, "re": re}
    try:
        exec(compile(src, "<curated>", "exec"), ns)
    except Exception:
        return None
    return ns.get(name)


def load_projects(path, tier, limit=None):
    fp = path if path.endswith(".jsonl") else os.path.join(path, tier + ".jsonl")
    projs = []
    for line in open(fp):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        ref = _compile(t["reference_code"], t["fn"])
        samp = _compile(t["sampler_code"], "sample")
        edge = _compile(t["edge_code"], "edge")
        if not all(callable(x) for x in (ref, samp, edge)):
            continue
        ex = t["examples"]
        if isinstance(ex, str):
            ex = json.loads(ex)
        examples = [(tuple(a), o) for a, o in ex]
        proj = {"name": t["name"], "ref": ref, "fn": t["fn"], "sampler": samp, "edge": edge,
                "spec": t["spec"], "buggy": t["buggy_code"], "examples": examples, "tier": tier}
        buggy_fn = _compile(t["buggy_code"], t["fn"])
        if callable(buggy_fn):
            proj["mutants"] = [("buggy", buggy_fn)]
        projs.append(proj)
        if limit and len(projs) >= limit:
            break
    return projs
