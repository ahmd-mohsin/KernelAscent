"""Scaffold-RSI harness (T2, API-eligible): system-level recursive self-improvement.

No weight training. Around a FROZEN model, an agent maintains an optimization scaffold
(base prompt + an evolving strategy memory). Each round it (1) optimizes a fixed set of
kernel tasks with the current scaffold, (2) is graded, (3) reflects on what worked/failed
and edits its own scaffold. A matched CONTROL arm keeps the round-0 scaffold frozen.

We report capability C_k per round, the compounding coefficient (quadratic fit), and
Delta_k = C_self - C_control. Scaffold-RSI is confirmed only if Delta_k grows across
rounds (self-improvement, not just more attempts). Generation is Bedrock converse (API);
grading is delegated to grade_candidates.py (needs a GPU host).

Usage (run where both Bedrock creds and a GPU are available, or wire --grader for remote):
  AWS_SHARED_CREDENTIALS_FILE=/tmp/ka/bedrock_creds AWS_PROFILE=bedrock \
  python scaffold_rsi.py --model-id us.anthropic.claude-fable-5 --rounds 5 \
     --n-fusion 12 --seed0 10000000 --outdir /tmp/ka/scaffold_run
"""
import os, re, json, glob, argparse, subprocess, time
import boto3
import gen_source_tasks as G

SYS = ("You are an elite GPU performance engineer optimizing PyTorch modules to run "
       "faster on an A100 while keeping outputs numerically equivalent.")
BASE = """Optimize the module for speed on an A100. Keep __init__ identical; only rewrite forward.
Use Triton or fused PyTorch. Output ONE class named ModelNew in a single ```python block.

Accumulated optimization strategies (apply the relevant ones):
{memory}

Reference module:
```python
{src}
```"""


def extract(t):
    for b in re.findall(r"```(?:python)?\s*(.*?)```", t or "", re.DOTALL):
        if "class ModelNew" in b:
            return b
    return None


class Bedrock:
    def __init__(self, model_id, region, profile):
        self.rt = (boto3.Session(profile_name=profile) if profile else boto3.Session()).client(
            "bedrock-runtime", region_name=region)
        self.model_id = model_id

    def ask(self, system, user, max_tokens=32000, temp=1.0):
        for a in range(6):
            try:
                r = self.rt.converse(modelId=self.model_id, system=[{"text": system}],
                                     messages=[{"role": "user", "content": [{"text": user}]}],
                                     inferenceConfig={"maxTokens": max_tokens, "temperature": temp})
                return "".join(p.get("text", "") for p in r["output"]["message"]["content"])
            except Exception as e:
                if "Throttl" in repr(e) and a < 5:
                    time.sleep(min(2 ** a, 30)); continue
                return "ERR:" + repr(e)[:80]


def render_memory(mem):
    return "\n".join("- " + m for m in mem) if mem else "- (none yet)"


def optimize_round(bed, tasks, memory, outdir, workers=4):
    os.makedirs(outdir, exist_ok=True)
    mem = render_memory(memory)
    def do(t):
        d = os.path.join(outdir, t["name"]); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "task.py"), "w").write(t["source"])
        json.dump({k: t[k] for k in ("name", "tier", "family", "tags", "meta")},
                  open(os.path.join(d, "meta.json"), "w"), indent=2)
        raw = bed.ask(SYS, BASE.format(memory=mem, src=t["source"]))
        open(os.path.join(d, "raw_0.txt"), "w").write(raw or "")
        code = extract(raw)
        if code:
            open(os.path.join(d, "cand_0.py"), "w").write(code)
        open(os.path.join(d, "DONE"), "w").write("1" if code else "0")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, tasks))


def grade(outdir, grader_tmpl):
    summ = os.path.join(outdir, "summary.json")
    cmd = grader_tmpl.format(candir=outdir, out=summ)
    subprocess.call(cmd, shell=True)
    if not os.path.exists(summ):
        return 0.0
    tasks = json.load(open(summ)).get("tasks", [])
    if not tasks:
        return 0.0
    return sum(1 for t in tasks if t.get("best_speedup_roofline", 0) > 1.0) / len(tasks)  # C = fast_1


def reflect(bed, outdir, memory):
    # summarize this round's outcomes and ask the model to add strategies
    notes = []
    for d in sorted(os.listdir(outdir)):
        rj = os.path.join(outdir, d, "results.json")
        if os.path.exists(rj):
            r = json.load(open(rj))
            notes.append("%s: pass=%d/%d best_sp=%.2f" %
                         (d, r.get("pass_at_k", 0), r.get("n_cand", 0), r.get("best_speedup_roofline", 0)))
    prompt = ("You are improving your own kernel-optimization playbook. Given these outcomes:\n"
              + "\n".join(notes[:40]) +
              "\n\nList 1-3 NEW, concrete, generally-useful Triton/kernel optimization strategies "
              "(one per line, terse) to add to the playbook. No prose.")
    out = bed.ask(SYS, prompt, max_tokens=2000)
    new = [l.strip("-* ").strip() for l in (out or "").splitlines() if l.strip()][:3]
    return memory + [m for m in new if m and not m.startswith("ERR")]


def reflect_selfcritique(bed, outdir, memory, max_show=4):
    """API-only scaffold evolution: the model critiques its OWN kernels this round
    (no ground-truth grades) and proposes new strategies. Used with --defer-grade."""
    snippets = []
    for cf in sorted(glob.glob(outdir + "/*/cand_0.py"))[:max_show]:
        snippets.append(open(cf).read()[:1500])
    shown = "\n\n---\n\n".join(snippets) if snippets else "(no parseable kernels this round)"
    prompt = ("Here are GPU kernels you wrote this round:\n\n" + shown +
              "\n\nReflect on likely correctness/performance issues and list 1-3 NEW, concrete, "
              "generally-useful Triton/kernel optimization strategies to add to your playbook "
              "(one per line, terse). No prose.")
    out = bed.ask(SYS, prompt, max_tokens=2000)
    new = [l.strip("-* ").strip() for l in (out or "").splitlines() if l.strip()][:3]
    return memory + [m for m in new if m and not m.startswith("ERR")]


def fit_b(ys):
    ys = [y for y in ys if y is not None]
    n = len(ys)
    if n < 3:
        return 0.0
    try:
        import numpy as np
        return float(np.polyfit(range(n), ys, 2)[0])
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=os.environ.get("BEDROCK_PROFILE", "bedrock"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n-fusion", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=10_000_000)
    ap.add_argument("--grader", default="python grade_candidates.py --candir {candir} --out {out}",
                    help="shell template to grade a round dir; use {candir} and {out}")
    ap.add_argument("--defer-grade", action="store_true",
                    help="API-only: evolve scaffold via self-critique, skip grading (grade rounds later on GPU)")
    ap.add_argument("--families", default="", help="comma-separated families (default all)")
    ap.add_argument("--limit", type=int, default=0, help="cap tasks per round")
    args = ap.parse_args()

    bed = Bedrock(args.model_id, args.region, args.profile)
    tasks = G.generate_systematic(n_fusion=args.n_fusion, seed0=args.seed0)
    if args.families:
        fams = set(f.strip() for f in args.families.split(","))
        tasks = [t for t in tasks if t["family"] in fams]
    if args.limit:
        tasks = tasks[:args.limit]
    os.makedirs(args.outdir, exist_ok=True)

    self_mem, ctrl_mem = [], []          # control keeps round-0 (empty) scaffold frozen
    C_self, C_ctrl, deltas, log = [], [], [], []
    for k in range(args.rounds):
        sd = os.path.join(args.outdir, "self", "round_%d" % k)
        cd = os.path.join(args.outdir, "control", "round_%d" % k)
        optimize_round(bed, tasks, self_mem, sd)
        optimize_round(bed, tasks, ctrl_mem, cd)
        json.dump({"round": k, "self_scaffold": self_mem, "control_scaffold": ctrl_mem},
                  open(os.path.join(sd, "scaffold.json"), "w"), indent=2)
        if args.defer_grade:
            cs = cc = None                      # grade later on GPU
            self_mem = reflect_selfcritique(bed, sd, self_mem)
            print("round %d: generated (defer-grade); scaffold=%d" % (k, len(self_mem)), flush=True)
        else:
            cs, cc = grade(sd, args.grader), grade(cd, args.grader)
            print("round %d: C_self=%.3f C_control=%.3f delta=%.3f (scaffold=%d)" %
                  (k, cs, cc, cs - cc, len(self_mem)), flush=True)
            self_mem = reflect(bed, sd, self_mem)   # only the self arm evolves
        C_self.append(cs); C_ctrl.append(cc)
        deltas.append((cs - cc) if (cs is not None and cc is not None) else None)
        log.append(dict(round=k, C_self=cs, C_control=cc,
                        delta=(cs - cc) if (cs is not None and cc is not None) else None,
                        scaffold_size=len(self_mem)))

    valid_C = [c for c in C_self if c is not None]
    valid_d = [d for d in deltas if d is not None]
    verdict = ("pending-grade" if not valid_d else
               ("compounds" if len(valid_d) >= 2 and valid_d[-1] > valid_d[0] and valid_d[-1] > 0
                else "plateau/none"))
    result = dict(model=args.model_id, rounds=args.rounds, defer_grade=args.defer_grade,
                  capability_r0=valid_C[0] if valid_C else None,
                  capability_rN=valid_C[-1] if valid_C else None,
                  compounding_b=fit_b(C_self), delta_final=(valid_d[-1] if valid_d else None),
                  delta_series=deltas, C_self=C_self, C_control=C_ctrl,
                  verdict=verdict, log=log)
    json.dump(result, open(os.path.join(args.outdir, "scaffold_rsi_result.json"), "w"), indent=2)
    print("verdict:", verdict, "compounding_b=%.4f" % result["compounding_b"],
          "(grade rounds later on GPU)" if args.defer_grade else "")


if __name__ == "__main__":
    main()
