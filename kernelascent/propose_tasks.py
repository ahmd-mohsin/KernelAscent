"""Fable task-PROPOSER (runs on Mac): invents novel benchmark problems.

Complements the procedural generator. Fable is asked to invent fresh, non-trivial
PyTorch `Model` modules across systematic themes and difficulty tiers. Output problems
are written as task.py; they are then validated + deduped + novelty-checked on the box
before joining the pool. The synthetic generator remains the contamination-safe core;
these are an augmentation, so we instruct against copying known public benchmarks and
dedupe structurally.
"""
import os, re, json, argparse, time
from concurrent.futures import ThreadPoolExecutor
import boto3

SYS = ("You are a GPU kernel benchmark designer. You invent novel, self-contained PyTorch "
       "modules that are meaningful GPU performance optimization targets.")
PROMPT = """Invent ONE novel PyTorch module that is a realistic GPU-performance optimization target: a fused sequence of tensor ops a kernel engineer would want to make faster.

Hard requirements:
- Define class Model(nn.Module) with __init__(self, dtype=DT) that does `g = torch.Generator().manual_seed(SEED)` and creates every weight as nn.Parameter(..., generator=g, ... ).to(dtype) with requires_grad=False.
- forward(self, x) is pure functional tensor ops (no data-dependent control flow), numerically stable, returns ONE tensor.
- Define module-level SEED, the shape variables, DT (torch.float16 or torch.bfloat16), and def get_inputs() returning [one tensor] built from torch.Generator().manual_seed(SEED+12345).
- Use a NON-TRIVIAL fused pattern (>=3 ops). Theme: {theme}. Difficulty target: {tier}.
- Keep tensors moderate (< 8 GB). Do NOT copy any known public benchmark (e.g. KernelBench). Invent a fresh combination.
- Output ONLY the module source in a single ```python code block. No prose."""

THEMES = [
    "gated feed-forward SwiGLU-style MLP", "multi-head self-attention", "grouped routed expert matmul (MoE top-1)",
    "rotary positional embedding then attention", "int8 dequantize then matmul epilogue", "residual RMSNorm/LayerNorm stack",
    "softmax attention with additive bias", "cumulative-sum / scan style reduction", "top-k gating with weighted sum",
    "cross-attention between two projections", "depthwise-style channel mixing", "GLU gating with layernorm",
]
TIERS = ["L2", "L3", "L4"]


def extract_model(text):
    for b in re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL):
        if "class Model" in b and "def get_inputs" in b:
            return b
    return None


class Proposer:
    def __init__(self, model_id, region, profile):
        sess = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.rt = sess.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def one(self, theme, tier, max_tokens=32000):
        for attempt in range(6):
            try:
                r = self.rt.converse(
                    modelId=self.model_id, system=[{"text": SYS}],
                    messages=[{"role": "user", "content": [{"text": PROMPT.format(theme=theme, tier=tier)}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 1.0})
                return "".join(p.get("text", "") for p in r["output"]["message"]["content"])
            except Exception as e:
                if "Throttl" in repr(e) and attempt < 5:
                    time.sleep(min(2 ** attempt, 30)); continue
                return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=os.environ.get("BEDROCK_PROFILE", "bedrock"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per-cell", type=int, default=2, help="proposals per (theme,tier)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    prop = Proposer(args.model_id, args.region, args.profile)

    jobs = [(t, tier, i) for tier in TIERS for t in THEMES for i in range(args.per_cell)]
    if args.limit:
        jobs = jobs[:args.limit]

    def run(job):
        theme, tier, i = job
        name = "prop_%s_%s_%d" % (tier, re.sub(r"[^a-z0-9]+", "-", theme.lower())[:20], i)
        d = os.path.join(args.outdir, name)
        if os.path.exists(os.path.join(d, "task.py")):
            return name, True
        text = prop.one(theme, tier)
        src = extract_model(text or "")
        if not src:
            return name, False
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "task.py"), "w").write(src)
        json.dump({"name": name, "tier": tier, "family": "fable-proposed", "theme": theme,
                   "tags": [], "meta": {"theme": theme, "chain": [theme]}},
                  open(os.path.join(d, "meta.json"), "w"), indent=2)
        return name, True

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, good in ex.map(run, jobs):
            ok += int(good)
            print("%-40s %s" % (name, "ok" if good else "NO_SOURCE"), flush=True)
    print("proposed %d/%d -> %s" % (ok, len(jobs), args.outdir))


if __name__ == "__main__":
    main()
