"""Fable-5.1 curator for GPU-kernel optimization tasks (KernelBench-style), tier-graded. ALL task content
comes from Fable (never hand-written); each candidate is executable-VALIDATED on GPU before admission:
the source must define DT + class Model(nn.Module) + get_inputs(); build_ref must run and produce a
finite fp32-gold-consistent output; and the baseline must be non-trivial (measurable time). Deduped.

Tiers (research/optimization difficulty):
  L1 memory/reduction-bound fusions   L2 matmul + fused epilogue   L3 attention / normalization chains
Prompted for the best quality per tier. Output bank JSON: [{name, tier, source}].
"""
import os, sys, json, argparse, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
import curate_bedrock as CB

TIER = {
 "L1": "a MEMORY/REDUCTION-bound fusion (elementwise chains, softmax/normalization pieces, reductions) on a large 2-D tensor; hard to beat torch.compile because it is bandwidth-bound",
 "L2": "a MATMUL with a fused epilogue (bias/activation/scale/residual) on realistic shapes; hard to beat cuBLAS/torch.compile",
 "L3": "an ATTENTION or full NORMALIZATION chain (e.g. scaled-dot-product attention, layernorm+residual+gelu, "
       "a fused SiLU-gate MLP, multi-head attention) on realistic transformer shapes; genuinely hard to fuse "
       "correctly and to beat torch.compile",
}
# how many VALIDATED tasks to admit per tier (weighted hard: most of the bank is L3 to give a strong model headroom)
COUNTS = {"L1": 6, "L2": 24, "L3": 40}   # hard-weighted over-sample; difficulty_filter.py trims to the learnable band

# Concrete, single-op L3 specs. Fable STALLS on the open-ended "deep chain" phrasing but answers a specific
# fusion quickly; we rotate through these so each L3 attempt asks for one well-defined hard kernel.
L3_SPECS = [
 "scaled-dot-product attention (softmax(QK^T/sqrt(d))V) for one multi-head batch, no mask, no dropout",
 "layernorm followed by a linear (matmul+bias) then gelu then a residual add",
 "a fused SiLU-gated MLP: y = down_proj( silu(gate_proj(x)) * up_proj(x) )",
 "RMSNorm followed by a linear projection (matmul+bias)",
 "multi-head attention with a causal mask (softmax over masked QK^T/sqrt(d), then times V)",
 "a bias-add + GELU + dropout-free layernorm chain over the last dim",
 "grouped-query attention: fewer KV heads than Q heads, softmax(QK^T/sqrt(d))V, no mask",
 "softmax over the last dim then a matmul with a weight matrix (attention-style probability-weighted sum)",
]
TMPL = (
 "Create ONE PyTorch module optimization task at tier {t}: {desc}.\n"
 "Return ONLY a python code block defining EXACTLY:\n"
 "  DT = torch.float16\n"
 "  class Model(nn.Module):  with __init__(self, dt=DT) and forward(self, x)  (any needed weights created "
 "in __init__ on the right device/dtype; forward returns one tensor)\n"
 "  def get_inputs():  returns [a single torch tensor of dtype DT] (the x passed to forward)\n"
 "Use realistic large shapes so timing is meaningful. Must import torch, torch.nn as nn. Numerically "
 "well-defined and deterministic. NO prose, ONLY the code block.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=0, help="0 = use per-tier COUNTS map"); ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--effort", default="max"); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--only-tier", default="", help="restrict to one tier (L1/L2/L3) for parallel curation")
    ap.add_argument("--out-name", default="kernel_tasks.json"); ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--read-timeout", type=int, default=120, help="abandon a stalled Fable call after this many seconds")
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/kernel_bank"); args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    from kernelascent import agent_bench as AB
    import torch
    cur = CB.Curator("us.anthropic.claude-fable-5-1", args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    rid, mt = cur.resolve(); cur.resolve_reasoning()
    import boto3
    from botocore.config import Config as _C
    cur.rt = boto3.Session(profile_name="bedrock").client("bedrock-runtime", region_name=args.region,
                                                          config=_C(read_timeout=args.read_timeout, connect_timeout=30, retries={"max_attempts": 1}))
    cur.resolved = (rid, min(mt, args.max_tokens))
    cur.reasoning = {"thinking": {"type": "adaptive"}, "output_config": {"effort": args.effort}}

    def gen(p):
        """Retry on empty OR error/timeout. A stalled Fable call now raises ReadTimeoutError in
        ~read_timeout s (not 30 min), so a hung request is abandoned and retried instead of blocking."""
        for _ in range(4):
            o = cur.generate(p) or ""
            if o.strip() and not o.startswith("BEDROCK_ERROR"):
                return o
        return ""

    def extract(t):
        m = re.search(r"```(?:python)?\s*(.*?)```", t or "", re.S)
        return (m.group(1) if m else (t or "")).strip()

    def validate(src):
        if "class Model" not in src or "def get_inputs" not in src or "DT" not in src:
            return False, "missing DT/Model/get_inputs"
        try:
            ref, x, gold, rerr = AB.build_ref(src)
            tbase = AB.time_fn(lambda z: ref(z), (x,))
            if not (tbase > 0 and torch.isfinite(gold).all().item()):
                return False, "bad ref/time"
        except Exception as e:
            return False, "build_ref: %r" % e
        return True, "ok tbase=%.3fms rerr=%.2g" % (tbase * 1e3, rerr)

    bank = []; seen = set()
    tiers = (args.only_tier,) if args.only_tier else ("L1", "L2", "L3")
    for tier in tiers:
        target = args.per_tier if args.per_tier > 0 else COUNTS[tier]
        got = 0
        for a in range(target * args.attempts):
            if got >= target:
                break
            desc = ("an ATTENTION or NORMALIZATION kernel: " + L3_SPECS[a % len(L3_SPECS)] +
                    " on realistic transformer shapes; genuinely hard to fuse and beat torch.compile") if tier == "L3" else TIER[tier]
            src = extract(gen(TMPL.format(t=tier, desc=desc)))
            key = hashlib.sha1(re.sub(r"\s+", "", src).encode()).hexdigest()
            if key in seen or not src:
                print("  %s a%d: empty/dup" % (tier, a), flush=True); continue
            ok, why = validate(src)
            print("  %s a%d [%s] %s" % (tier, a, "PASS" if ok else "REJECT", why), flush=True)
            if ok:
                seen.add(key); got += 1
                bank.append({"name": "%s_%d" % (tier.lower(), got), "tier": tier, "source": src})
                json.dump(bank, open(os.path.join(args.outdir, args.out_name), "w"), indent=2)
        print("TIER %s: %d/%d" % (tier, got, target), flush=True)
    print("DONE", {t: sum(1 for b in bank if b["tier"] == t) for t in ("L1", "L2", "L3")}, flush=True)


if __name__ == "__main__":
    main()
