"""KernelAscent CLI — one-command eval.

  kernelascent gen   --model us.anthropic.claude-opus-4-8 --tiers L1,L2 --k 1 --out runs/opus
  kernelascent grade --candir runs/opus --out runs/opus/summary.json      # needs a GPU
  kernelascent eval  --model us.anthropic.claude-opus-4-8 --tiers L1,L2 --out runs/opus

`gen` calls a Bedrock model via converse (API, no GPU) and stores candidate kernels +
reasoning trajectories. `grade` runs/times them on a GPU against the roofline. `eval`
does both. Auth: AWS profile with Bedrock access (env BEDROCK_PROFILE, default 'bedrock').
"""
import os, sys, argparse, subprocess, glob, json

PKG = os.path.dirname(os.path.abspath(__file__))
TIER_FAMILIES = {"L1": ["norm-act"], "L2": ["matmul", "quant-gemm"],
                 "L3": ["attention", "rope-attention", "moe"]}
SPLIT_SEED = {"public": 0, "heldout": 10_000_000}


def _run(script, args):
    env = os.environ.copy()
    env["PYTHONPATH"] = PKG + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call([sys.executable, os.path.join(PKG, script)] + [str(a) for a in args], env=env)


def _families(tiers):
    fams = []
    for t in tiers.split(","):
        fams += TIER_FAMILIES.get(t.strip().upper(), [])
    return ",".join(dict.fromkeys(fams))


def cmd_gen(a):
    fams = _families(a.tiers)
    seed0 = SPLIT_SEED.get(a.split, 0)
    args = ["--model-id", a.model, "--outdir", a.out, "--k", a.k, "--n-fusion", a.n_fusion,
            "--seed0", seed0, "--workers", a.workers, "--region", a.region]
    if fams:
        args += ["--families", fams]
    return _run("curate_bedrock.py", args)


def cmd_grade(a):
    return _run("grade_candidates.py", ["--candir", a.candir, "--out", a.out])


def cmd_eval(a):
    rc = cmd_gen(a)
    if rc != 0:
        print("generation failed"); return rc
    summ = os.path.join(a.out, "summary.json")
    rc = _run("grade_candidates.py", ["--candir", a.out, "--out", summ])
    if os.path.exists(summ):
        tasks = json.load(open(summ)).get("tasks", [])
        n = max(len(tasks), 1)
        fp = lambda p: sum(1 for t in tasks if t.get("best_speedup_roofline", 0) > p) / n
        print("\n=== %s ===" % a.model)
        print("tasks=%d  pass@1=%.2f  fast_1=%.2f  fast_1.5=%.2f  fast_2=%.2f" %
              (len(tasks), sum(1 for t in tasks if t.get("pass_at_k", 0) > 0) / n, fp(1.0), fp(1.5), fp(2.0)))
    return rc


def main():
    ap = argparse.ArgumentParser(prog="kernelascent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="Bedrock model id (bare or us.* profile)")
        p.add_argument("--tiers", default="L1,L2", help="comma list of L1,L2,L3")
        p.add_argument("--split", default="public", choices=list(SPLIT_SEED))
        p.add_argument("--k", default=1)
        p.add_argument("--n-fusion", default=10)
        p.add_argument("--workers", default=6)
        p.add_argument("--region", default="us-east-1")
        p.add_argument("--out", required=True)
    pg = sub.add_parser("grade")
    pg.add_argument("--candir", required=True)
    pg.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"gen": cmd_gen, "grade": cmd_grade, "eval": cmd_eval}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
