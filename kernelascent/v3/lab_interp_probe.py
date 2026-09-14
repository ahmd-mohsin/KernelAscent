"""GPU interpretability probe: WHERE (and whether) a model INTERNALLY encodes GPU-kernel correctness.

For a set of tasks the model generates k kernels; each is graded (correct vs not) via the same isolated grader
as weight-RSI. We then run a forward pass over each generated kernel with output_hidden_states, mean-pool tokens
per transformer layer, and — per layer — measure how linearly separable correct-from-incorrect kernels are
(1-D Fisher/LDA projection → rank AUC). Output: per-layer correctness-decodability curve + best layer/AUC.

Reading: a model that will COMPOUND under weight-RSI should already carry a decodable "this kernel is correct"
signal in its mid/late layers (SFT can then amplify it); a model at the correctness wall has no such signal to
amplify. This is the internal, activation-level counterpart to the behavioral WHY-RSI probe.

Runs on 1 GPU (grades on the same GPU via KA_GRADE_GPU). Memory-safe: bf16, no grad, truncated context, small bs.
"""
import argparse, json, os, statistics, torch
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK

SYS = "You are an expert GPU performance engineer. You write correct, fast PyTorch/Triton kernels."

def _auc(scores, labels):
    """Rank AUC of a 1-D score separating label==1 from label==0 (dependency-free)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]; neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg: return None
    ranks = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_of = {};
    for r, i in enumerate(ranks): rank_of[i] = r + 1
    sp = sum(rank_of[i] for i, y in enumerate(labels) if y == 1)
    auc = (sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return round(max(auc, 1 - auc), 3)   # symmetric: separability regardless of sign

@torch.no_grad()
def _layer_feats(tok, mdl, code, dev, max_len=1024):
    ids = tok(code, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    out = mdl(**ids, output_hidden_states=True)
    hs = out.hidden_states                       # tuple (L+1) of [1, seq, H]
    return [h[0].float().mean(0).cpu() for h in hs]   # mean-pool tokens -> [H] per layer

def run(args):
    dev = "cuda:0"
    tok = W.__dict__.get("_tok")  # not used; build fresh base model (no LoRA) with hidden states
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if os.environ.get("KA_DTYPE", "bf16") == "bf16" else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, device_map={"": dev}, trust_remote_code=True).eval()
    names = list(LK.TASKS)[: args.n_tasks]
    print("INTERP-PROBE %s tasks=%d k=%d" % (args.model, len(names), args.k), flush=True)
    # 1) generate + grade -> labeled kernels
    srcs = [LK.TASKS[n] for n in names]
    bs = 1 if any(s in args.model for s in ("7B", "7b", "8B", "8b", "9B", "13b", "14B", "15b", "32B")) else 2
    gl = W.generate_batch(tok, mdl, srcs, args.k, max_new=900, adapter=False, bs=bs)
    feats = []; labels = []
    for n, src, gens in zip(names, srcs, gl):
        codes = [c for c in (AB.extract_modelnew(t) for t in gens) if c]
        if not codes: continue
        grades = W._grade_isolated(src, codes)          # [ok, sp_e, sp_c] per code
        for code, g in zip(codes, grades):
            ok = bool((g + [0])[0]) if isinstance(g, list) else False
            try:
                feats.append(_layer_feats(tok, mdl, code, dev)); labels.append(1 if ok else 0)
            except Exception:
                pass
    nC, nI = sum(labels), len(labels) - sum(labels)
    print("labeled: %d correct / %d incorrect" % (nC, nI), flush=True)
    result = {"model": args.model, "n_correct": nC, "n_incorrect": nI,
              "correctness_rate": round(nC / max(len(labels), 1), 3)}
    if nC >= 3 and nI >= 3:
        L = len(feats[0]); per_layer = []
        for l in range(L):
            X = torch.stack([f[l] for f in feats])                    # [N, H]
            mu1 = X[[i for i,y in enumerate(labels) if y==1]].mean(0)
            mu0 = X[[i for i,y in enumerate(labels) if y==0]].mean(0)
            w = (mu1 - mu0)                                            # Fisher direction (isotropic)
            proj = (X @ w).tolist()
            per_layer.append(_auc(proj, labels))
        best = max(range(L), key=lambda i: (per_layer[i] or 0))
        result.update(n_layers=L, per_layer_auc=per_layer, best_layer=best, best_layer_frac=round(best/(L-1),2),
                      best_auc=per_layer[best], mean_auc=round(statistics.mean([a for a in per_layer if a]),3))
        print("best-layer AUC=%.3f @ layer %d/%d (%.0f%% depth) | mean AUC=%.3f" %
              (per_layer[best], best, L-1, 100*best/(L-1), result["mean_auc"]), flush=True)
    else:
        result.update(note="insufficient correct/incorrect kernels to probe (correctness wall)")
        print("correctness wall: not enough labeled examples to probe internal correctness", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(result, open(os.path.join(args.outdir, "interp_probe.json"), "w"), indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-tasks", type=int, default=12); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "interp_probe"))
    run(ap.parse_args())

if __name__ == "__main__":
    main()
