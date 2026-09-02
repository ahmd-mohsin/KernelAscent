"""Evaluate MANY Bedrock models on the KernelAscent benchmark, storing full trajectories.

For each model it runs the curator harness (curate_bedrock) as a test-taker: the model
reads each task.py and writes ModelNew; we save the extracted kernel AND the full raw
reasoning trajectory. Grading (grade_candidates.py, GPU box) then scores each model into
a leaderboard row. Generation is API-only (Bedrock converse); no GPU needed to generate.

Usage:
  AWS_SHARED_CREDENTIALS_FILE=/tmp/ka/bedrock_creds AWS_PROFILE=bedrock \
  python eval_all.py --models eval_models.json --n-fusion 40 --seed0 10000000 \
      --k 3 --outroot /tmp/ka/eval [--limit-models 76] [--workers 4]

seed0=10000000 -> the PRIVATE held-out split (leaderboard). seed0=0 -> public dev split.
Grade afterwards, per model:
  python grade_candidates.py --candir /tmp/.../eval/<slug> --out summary_<slug>.json
"""
import os, json, argparse, subprocess, re

HERE = os.path.dirname(os.path.abspath(__file__))


def slug(model_id):
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="JSON list of Bedrock model ids")
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--n-fusion", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=10_000_000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--families", default="")
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    models = json.load(open(args.models))
    if args.limit_models:
        models = models[:args.limit_models]
    os.makedirs(args.outroot, exist_ok=True)
    print("evaluating %d models -> %s (seed0=%d)" % (len(models), args.outroot, args.seed0))
    for i, mid in enumerate(models):
        outdir = os.path.join(args.outroot, slug(mid))
        cmd = ["python", os.path.join(HERE, "curate_bedrock.py"),
               "--model-id", mid, "--region", args.region, "--outdir", outdir,
               "--k", str(args.k), "--n-fusion", str(args.n_fusion),
               "--seed0", str(args.seed0), "--workers", str(args.workers)]
        if args.families:
            cmd += ["--families", args.families]
        print("[%d/%d] %s" % (i + 1, len(models), mid), flush=True)
        rc = subprocess.call(cmd)
        print("   rc=%d -> %s" % (rc, outdir), flush=True)
    print("generation done. Grade each with grade_candidates.py on the GPU box, then aggregate leaderboard.")


if __name__ == "__main__":
    main()
