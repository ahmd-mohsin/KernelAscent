"""Open-weight test-taker harness for tier calibration.

Runs a single open-weight HF model (e.g. Qwen2.5-Coder-*-Instruct) as a
*test-taker* on the tiered task set and writes candidate ModelNew files, which
are then graded by grade_candidates.py on the GPU. Decoupled generation vs
grading is deliberate (the community-standard KernelBench-style loop): generate
-> extract ```python block -> import as a module -> grade correctness vs an
fp32 gold across N random inputs -> time vs the min(eager, torch.compile) roofline.

Standard practice this follows:
  - HF `AutoModelForCausalLM` + `apply_chat_template` (the model's own chat format),
    bf16 weights, one model pinned to one GPU via CUDA_VISIBLE_DEVICES.
  - Batched sampling, k candidates per task, temperature/top-p sampling.
  - No grading here; grading is a separate GPU pass (grade_candidates.py) so the
    same candidates can be re-graded deterministically.

For faster serving / RL rollouts at scale the recommended stack is vLLM (see
docs/open_weight_track.md); this file uses plain transformers so it runs against
a container's pinned torch without reinstalling anything.
"""
import os, argparse, json
import torch
import gen_source_tasks as G
import curate_bedrock as CB

PROMPT = """Optimize this PyTorch module for speed on an NVIDIA A100 GPU. Keep __init__ identical; only rewrite forward. Use Triton or fused PyTorch ops. The output must be numerically equivalent to the reference.

Output exactly ONE class named ModelNew in a single ```python code block. No prose, no explanation.

Reference module:
```python
{src}
```"""


def build_tasks(tiers, n_per_tier, seed0, nshards=1, shard=0):
    tasks = []
    for tier in tiers:
        tasks += G.generate_tiered(tier, n_per_tier, seed0=seed0)
    if nshards > 1:                              # round-robin keeps every tier represented in each shard
        tasks = [t for i, t in enumerate(tasks) if i % nshards == shard]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tiers", default="Easy,Medium,Hard", help="comma list of Easy,Medium,Hard,Ultra")
    ap.add_argument("--n-per-tier", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="candidates per task")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.outdir, exist_ok=True)
    tasks = build_tasks([t.strip() for t in args.tiers.split(",")], args.n_per_tier, args.seed0,
                        args.nshards, args.shard)
    print("model=%s tasks=%d tiers=%s k=%d gpu=%s" %
          (args.model, len(tasks), args.tiers, args.k, os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="cuda")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    for i, t in enumerate(tasks):
        d = os.path.join(args.outdir, t["name"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(t["source"])
        json.dump({k: t[k] for k in ("name", "tier", "level", "family", "tags", "meta") if k in t},
                  open(d + "/meta.json", "w"), indent=2)
        msgs = [{"role": "user", "content": PROMPT.format(src=t["source"])}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text] * args.k, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=args.temp > 0,
                                 temperature=max(args.temp, 1e-5), top_p=args.top_p,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        n_ok = 0
        for j in range(args.k):
            raw = tok.decode(gen[j], skip_special_tokens=True)
            open(d + "/raw_%d.txt" % j, "w").write(raw)
            code = CB.extract_modelnew(raw)
            if code:
                open(d + "/cand_%d.py" % j, "w").write(code); n_ok += 1
        open(d + "/DONE", "w").write(str(n_ok))
        print("[%d/%d] %-40s tier=%-6s cands=%d/%d" % (i + 1, len(tasks), t["name"], t.get("tier"), n_ok, args.k), flush=True)


if __name__ == "__main__":
    main()
