"""KernelAscent — single standard evaluation entrypoint.

One command to score a model on the public benchmark and emit a submittable scorecard. Two tracks:

  capability : the curated code-task bank (per tier) -- does the model produce a correct patch, and does
               its local verifier select it (oracle_pool_success / selected_C / verifier_dQ). Reuses the
               validated rsi_verify panel with --curated.
  rsi        : the efficiency-of-experimentation lab -- does the model improve its research PROCEDURE with
               experience and does that compound (q1-q0 / N1 / F1 / F2). Reuses lab_easy Gate-4.

Models: --api-model <bedrock-id> (no GPU) OR --model <hf-id> (GPU). Held-out scoring stays with
maintainers; this evaluates the PUBLIC split and writes /out/scorecard.json.

  python3 -m kernelascent.evaluate --track capability --api-model us.anthropic.claude-opus-5 --tier medium
  python3 -m kernelascent.evaluate --track rsi        --api-model us.anthropic.claude-opus-5 --lineages 20
  python3 -m kernelascent.evaluate --track capability --model Qwen/Qwen2.5-Coder-7B-Instruct --tier hard
"""
import os, sys, json, argparse, types, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

DEFAULT_DATA = os.environ.get("KA_DATA", os.path.join(ROOT, "dataset", "tasks", "public"))


def _resolve_data(path):
    if os.path.isdir(path) or path.endswith(".jsonl"):
        return path
    # fall back: pull the public split from HuggingFace into a local dir
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download("muahmed7338/kernelascent-tasks", repo_type="dataset")
        return os.path.join(d, "data")
    except Exception as e:
        sys.exit("no curated data at %s and HF fetch failed: %r" % (path, e))


def run_capability(args):
    from kernelascent.v3 import rsi_verify as V
    data = _resolve_data(args.curated or DEFAULT_DATA)
    a = types.SimpleNamespace(api_model=args.api_model, model=args.model, region=args.region,
                              K=args.K, reps=args.reps, inject=1, hard=False, veryhard=False,
                              curated=data, tier=args.tier, limit=args.limit, max_new=args.max_new,
                              outdir=args.outdir)
    V.panel(a)
    res = json.load(open(os.path.join(args.outdir, "panel.json")))
    return {"track": "capability", "tier": args.tier,
            "oracle_pool_success": res.get("oracle_pool_success"),
            "selected_C_strong": res.get("selected_C_strong"),
            "verifier_dQ": res.get("verifier_dQ", {}).get("mean"),
            "attainable_C": res.get("attainable_C"), "n_cells": res.get("n_cells"), "K": res.get("K")}


def run_rsi(args):
    from kernelascent.v3 import lab_open_live as LOL
    if not args.api_model:
        sys.exit("rsi track currently runs API models (--api-model); GPU-model rsi is a separate path")
    bank = args.bank or os.path.join(ROOT, "dataset", "rsi_tasks", "public.jsonl")
    a = types.SimpleNamespace(api_model=args.api_model, region=args.region,
                              lineages=args.lineages, bank=(bank if os.path.exists(bank) else ""),
                              outdir=args.outdir)
    LOL.run(a)
    who = "api:" + args.api_model
    f = os.path.join(args.outdir, "lab_open_live_%s.json" % who.replace(":", "_").replace("/", "_").replace(".", "_"))
    d = json.load(open(f)); agg = d["agg"]
    g = lambda k: agg[k]["mean"]
    return {"track": "rsi", "substrate": "open-ended library-learning", "bank": bank, "lineages": d["lineages"],
            "Q0": d["Q0"]["mean"], "q1_minus_q0": g("q1_minus_q0"), "N1": g("N1"), "F1": g("F1"), "F2": g("F2"),
            "F1_ci": agg["F1"].get("ci95"), "F2_ci": agg["F2"].get("ci95")}


def main():
    ap = argparse.ArgumentParser(description="KernelAscent standard evaluation")
    ap.add_argument("--track", choices=["capability", "rsi"], required=True)
    ap.add_argument("--api-model", default=""); ap.add_argument("--model", default="")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tier", default="medium", help="capability tier: easy|medium|hard|ultra")
    ap.add_argument("--curated", default="", help="curated public dir/jsonl (default: bundled / HF)")
    ap.add_argument("--bank", default="", help="rsi task-bank JSONL (default: bundled dataset/rsi_tasks/public.jsonl)")
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--lineages", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--outdir", default="/out")
    args = ap.parse_args()
    if not (args.api_model or args.model):
        sys.exit("give --api-model <bedrock-id> or --model <hf-id>")
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    card = run_capability(args) if args.track == "capability" else run_rsi(args)
    card.update(model=args.api_model or args.model, elapsed_s=round(time.time() - t0, 1))
    json.dump(card, open(os.path.join(args.outdir, "scorecard.json"), "w"), indent=2)
    print("\n=== KERNELASCENT SCORECARD ===")
    for k, v in card.items():
        print("  %-20s %s" % (k, v))
    print("\nwrote", os.path.join(args.outdir, "scorecard.json"),
          "\nSubmit: open a model-submission issue with this scorecard; maintainers verify on the held-out split.")


if __name__ == "__main__":
    main()
