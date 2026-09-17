"""0-SCORE FAILURE FORENSICS + behavioural mechanistic interpretability (panel/user request).

For a model family we ask: on the tasks it scores ZERO on, WHERE does it get stuck? We generate K candidates
per task, and classify every candidate's outcome into a mechanistic failure taxonomy:
  no_extract     : the generation never produced a parseable `class ModelNew` (format/instruction-following fail)
  ref_build_fail : the task reference itself failed to build (harness, excluded from model blame)
  syntax_error   : generated code is syntactically invalid (SyntaxError/IndentationError)
  name_error     : uses undefined names / bad imports (NameError/ImportError/ModuleNotFound/AttributeError)
  type_shape     : TypeError / shape / size mismatch in wiring
  cuda_error     : CUDA illegal-memory / device asserts (bad hand-written kernel)
  oom            : CUDA out-of-memory
  runtime_other  : any other runtime exception
  wrong_output   : runs cleanly but numerically wrong / imprecise (the SUBTLE failure --- looks right, isn't)
  correct        : passed
We record per-task 0-ness, the per-model failure distribution, and up to N verbatim (prompt-tail, code-tail,
error) examples per bucket so the reasoning chains can be read. This localises *why* a family fails: below-wall
models fail overwhelmingly at no_extract/syntax (can't even form a kernel) vs. mid-scale at wrong_output
(forms plausible kernels that are silently incorrect) --- a mechanistic signature of where capability breaks.

CLI: python -m kernelascent.v3.lab_failure_forensics --model <hf> --gpus 0 --K 32 --max-tasks 40 --tag q05 --outdir <d>
"""
import os, sys, json, argparse, random, re, subprocess, signal, tempfile, collections
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _classify(ok, msg):
    if ok:
        return "correct"
    m = (msg or "").lower()
    if msg.startswith("ref_build_fail"):
        return "ref_build_fail"
    if "wrong/imprecise" in m:
        return "wrong_output"
    if "out of memory" in m or "outofmemory" in m:
        return "oom"
    if "illegal memory" in m or "device-side assert" in m or "cuda error" in m:
        return "cuda_error"
    if "syntaxerror" in m or "indentationerror" in m:
        return "syntax_error"
    if "nameerror" in m or "importerror" in m or "modulenotfound" in m or "attributeerror" in m:
        return "name_error"
    if "typeerror" in m or "size mismatch" in m or "shape" in m or "dimension" in m:
        return "type_shape"
    return "runtime_other"


def _grade_forensic(src, codes, timeout=180):
    if not codes:
        return []
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump({"task": src, "codes": codes}, open(path, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("KA_GRADE_GPU", "0"))
    res = []
    try:
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "grade_forensic.py"), path],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, start_new_session=True)
        try:
            so, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
            so = ""
        line = [l for l in so.splitlines() if l.startswith("RESULT")]
        res = json.loads(line[-1][len("RESULT"):]) if line else []
    except Exception:
        res = []
    os.remove(path)
    return res + [[False, 0.0, "grader_timeout"]] * (len(codes) - len(res))


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    gpus = [int(x) for x in str(args.gpus).split(",") if x != ""]
    tok, mdl = W.build(args.model, tuple(gpus))
    names = list(LK.TASKS)
    if args.max_tasks:
        random.Random(1).shuffle(names); names = names[:args.max_tasks]
    srcs = [LK.TASKS[n] for n in names]
    print("FORENSICS %s tasks=%d K=%d" % (args.model, len(names), args.K), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    dist = collections.Counter()
    per_task = {}
    examples = collections.defaultdict(list)   # bucket -> list of {task, code_tail, error}
    zero_tasks = []
    for ni, (name, src) in enumerate(zip(names, srcs)):
        gens = W.generate_batch(tok, mdl, [src], args.K, adapter=False)[0]
        codes, raw = [], []
        for g in gens:
            c = AB.extract_modelnew(g)
            if c:
                codes.append(c); raw.append(g)
            else:
                dist["no_extract"] += 1
                if len(examples["no_extract"]) < args.n_examples:
                    examples["no_extract"].append({"task": name, "gen_tail": g[-400:]})
        grades = _grade_forensic(src, codes) if codes else []
        n_ok = 0; tcount = collections.Counter()
        for code, g, res in zip(codes, raw, grades):
            ok, se, msg = (list(res) + [False, 0.0, "?"])[:3]
            cls = _classify(ok, msg)
            dist[cls] += 1; tcount[cls] += 1
            if ok:
                n_ok += 1
            elif len(examples[cls]) < args.n_examples:
                examples[cls].append({"task": name, "code_tail": code[-500:], "error": msg})
        per_task[name] = {"n_ok": n_ok, "K": args.K, "dominant_fail": (tcount.most_common(1)[0][0] if tcount else "no_extract")}
        if n_ok == 0:
            zero_tasks.append(name)
        if (ni + 1) % 10 == 0:
            print("  %d/%d tasks graded; running dist=%s" % (ni + 1, len(names), dict(dist)), flush=True)

    total = sum(dist.values()) or 1
    summary = {"model": args.model, "tag": args.tag, "K": args.K, "n_tasks": len(names),
               "n_zero_tasks": len(zero_tasks), "zero_task_frac": round(len(zero_tasks) / len(names), 3),
               "failure_dist": dict(dist),
               "failure_pct": {k: round(100 * v / total, 1) for k, v in dist.items()},
               "zero_tasks": zero_tasks, "per_task": per_task, "examples": {k: v for k, v in examples.items()}}
    out = os.path.join(args.outdir, "forensics_%s.json" % args.tag)
    json.dump(summary, open(out, "w"), indent=1)
    top = sorted(dist.items(), key=lambda x: -x[1])
    print("\n=== FORENSICS %s === zero-tasks=%d/%d (%.0f%%)\nfailure mix: %s\nwrote %s" %
          (args.tag, len(zero_tasks), len(names), 100 * len(zero_tasks) / len(names),
           ", ".join("%s %.0f%%" % (k, 100 * v / total) for k, v in top), out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpus", default="0")
    ap.add_argument("--K", type=int, default=32); ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "forensics"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
