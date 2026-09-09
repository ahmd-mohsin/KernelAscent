"""WEIGHT-RSI loop (open-weight) — the fix for the frozen-weight ceiling.

An open-weight model writes GPU kernels for a fixed task set; candidates are graded on GPU (correct vs
fp32-gold + speedup vs eager, via agent_bench). Each round we do LoRA REJECTION-SAMPLING SFT on the model's
OWN correct+fast kernels -> the model's WEIGHTS improve -> the improver itself gets better (impossible for
frozen/API models). We measure held-out capability C_r each round and the causal contrast
  Delta_r = C_r(self-trained) - C_0(frozen base)
Compounding / RSI is supported if C_r rises AND Delta_r grows across rounds. Control = the frozen base model
(adapter disabled), evaluated on the same held-out tasks at matched sampling.

Minimal + robust on torch-2.6-nv: transformers==4.47.1 + peft==0.13.2, plain-torch SFT (no trl).
Run: python3 kernelascent/v3/lab_weight_rsi.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --rounds 4
"""
import os, sys, json, argparse, random, statistics, time
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
import torch
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_kernel as LK   # reuse TASKS, _ref, _score, OPT

SYS = "You are an expert GPU performance engineer. You write correct, fast PyTorch/Triton kernels."


def _prompt(task_src):
    return LK.OPT.format(arch="", src=task_src)


def build(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    mdl = get_peft_model(mdl, lcfg)
    return tok, mdl


def _chat(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


from transformers import LogitsProcessor, LogitsProcessorList


class _Sanitize(LogitsProcessor):
    """replace nan/inf logits with finite values so multinomial never sees a bad probability tensor
    (a nan/inf there triggers an uncatchable CUDA device-side assert that poisons the whole process)."""
    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


_LP = LogitsProcessorList([_Sanitize()])


def generate(tok, mdl, task_src, k, max_new=900, temp=0.8, adapter=True):
    ctx = mdl.disable_adapter() if not adapter else _null()
    with ctx:
        text = _chat(tok, _prompt(task_src))
        enc = tok([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = mdl.generate(**enc, do_sample=True, temperature=temp, top_p=0.95, num_return_sequences=k,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id, logits_processor=_LP)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True) for g in gen]


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _grade_isolated(src, codes):
    """Grade candidate kernels in a FRESH subprocess (CUDA crash-isolation). Returns [(ok, sp), ...]."""
    import subprocess, tempfile
    if not codes:
        return []
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump({"task": src, "codes": codes}, open(path, "w"))
    env = dict(os.environ, PYTHONPATH="/tmp/instance_storage:/tmp/instance_storage/kernelascent",
               HF_HOME="/tmp/instance_storage/ka_data/hf")
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "grade_batch.py"), path],
                           capture_output=True, text=True, timeout=240, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        res = json.loads(line[-1][len("RESULT"):]) if line else []
    except Exception:
        res = []
    os.remove(path)
    return res + [[False, 0.0]] * (len(codes) - len(res))


def eval_tasks(tok, mdl, names, k, adapter=True):
    """mean over tasks of best-of-k speed-resolved score (capability C). Training examples = the BEST
    CORRECT candidate per task (rejection-sampling SFT). Generation runs in-process (safe); candidate
    EXECUTION is isolated in a subprocess so a bad kernel can't poison the trainer's CUDA context."""
    scores = []; examples = []
    for n in names:
        src = LK.TASKS[n]
        codes = [c for c in (AB.extract_modelnew(t) for t in generate(tok, mdl, src, k, adapter=adapter)) if c]
        res = _grade_isolated(src, codes)
        best = 0.0
        for code, (ok, sp) in zip(codes, res):
            s = LK._score(ok, sp)
            if s > best:
                best = s
            if ok:                                         # keep ALL correct kernels -> more SFT data
                examples.append((src, code))
        scores.append(best)
    return statistics.mean(scores) if scores else 0.0, examples


def sft(tok, mdl, pairs, steps, lr=2e-5, bs=2):
    """LoRA rejection-sampling SFT on (task_src, ModelNew code) pairs; loss on completion tokens only.
    Stabilized: low lr + grad clipping + steps scaled to data size (avoid the overfit-to-NaN that made
    generation emit inf/nan logits)."""
    if not pairs:
        return 0.0
    mdl.train(); opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)
    data = []
    for src, code in pairs:
        pr = _chat(tok, _prompt(src)); comp = "```python\n" + code.strip() + "\n```" + tok.eos_token
        pids = tok(pr, add_special_tokens=False)["input_ids"]
        cids = tok(comp, add_special_tokens=False)["input_ids"]
        ids = (pids + cids)[:1536]; labels = ([-100] * len(pids) + cids)[:1536]
        data.append((ids, labels))
    steps = min(steps, max(3, 3 * len(data)))              # scale to data -> no catastrophic overfit
    losses = []
    for step in range(steps):
        random.shuffle(data)
        batch = data[:bs]
        m = max(len(i) for i, _ in batch)
        input_ids = torch.tensor([i + [tok.pad_token_id] * (m - len(i)) for i, _ in batch]).cuda()
        lab = torch.tensor([l + [-100] * (m - len(l)) for _, l in batch]).cuda()
        att = (input_ids != tok.pad_token_id).long()
        out = mdl(input_ids=input_ids, attention_mask=att, labels=lab)
        if not torch.isfinite(out.loss):
            opt.zero_grad(); continue                      # skip a non-finite step
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in mdl.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(); losses.append(out.loss.item())
    mdl.eval(); return statistics.mean(losses) if losses else 0.0


def run(args):
    random.seed(0); torch.manual_seed(0)
    for n in LK.TASKS:
        LK._ref(n)                                          # warm GPU baselines
    tok, mdl = build(args.model)
    names = list(LK.TASKS)
    train, held = names[:args.n_train], names[args.n_train:]
    print("WEIGHT-RSI %s train=%s held=%s" % (args.model, train, held), flush=True)
    C0_frozen, _ = eval_tasks(tok, mdl, held, args.k, adapter=False)   # frozen-base control (constant)
    print("C0 frozen-base (held-out) = %.3f" % C0_frozen, flush=True)
    hist = []
    for r in range(args.rounds):
        t0 = time.time()
        # 1. self-generate + grade on TRAIN -> collect correct+fast kernels
        trainC, pairs = eval_tasks(tok, mdl, train, args.k, adapter=True)
        # 2. LoRA rejection-sampling SFT on own kernels -> weights improve
        loss = sft(tok, mdl, pairs, args.sft_steps)
        # 3. measure held-out capability C_r (self, post-update)
        Cr, _ = eval_tasks(tok, mdl, held, args.k, adapter=True)
        delta = Cr - C0_frozen
        hist.append({"round": r, "trainC": round(trainC, 3), "n_train_ex": len(pairs),
                     "sft_loss": round(loss, 3), "C_held": round(Cr, 3), "delta_vs_frozen": round(delta, 3)})
        print("round %d trainC=%.3f ex=%d loss=%.3f  C_held=%.3f  Delta=%+.3f  (%.0fs)" %
              (r, trainC, len(pairs), loss, Cr, delta, time.time() - t0), flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"model": args.model, "C0_frozen": C0_frozen, "history": hist},
                  open(os.path.join(args.outdir, "weight_rsi.json"), "w"), indent=2)
    dl = [h["delta_vs_frozen"] for h in hist]
    print("\n=== WEIGHT-RSI SUMMARY ===")
    print("  C_held by round:", [h["C_held"] for h in hist])
    print("  Delta vs frozen:", dl)
    print("  compounding (Delta rising)?", "YES" if len(dl) >= 2 and dl[-1] > dl[0] + 0.02 else "not resolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=3); ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/weight_rsi")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
