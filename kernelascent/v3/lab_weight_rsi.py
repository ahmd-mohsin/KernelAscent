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


def build(model_id, gpus=(0,)):
    """Load one arm sharded across the given GPU indices (fp32 7B needs >40GB during SFT, so 2 GPUs/arm).
    max_memory forces accelerate to place layers ONLY on `gpus` (others capped at 0), keeping the two arms
    on disjoint devices within one process."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if os.environ.get("KA_DTYPE", "bf16") == "bf16" else torch.float32
    gpus = tuple(gpus)
    if len(gpus) == 1:
        dm = {"": "cuda:%d" % gpus[0]}
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, device_map=dm)
    else:
        n = torch.cuda.device_count()
        mm = {i: ("40GiB" if i in gpus else "0GiB") for i in range(n)}
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, device_map="auto", max_memory=mm)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    mdl = get_peft_model(mdl, lcfg)
    mdl.enable_input_require_grads()                       # needed for grad-checkpointing during SFT
    return tok, mdl


def _dev(mdl):
    return next(mdl.parameters()).device


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
        enc = tok([text], return_tensors="pt").to(_dev(mdl))
        with torch.no_grad():
            out = mdl.generate(**enc, do_sample=True, temperature=temp, top_p=0.95, num_return_sequences=k,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id, logits_processor=_LP)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True) for g in gen]


def generate_batch(tok, mdl, srcs, k, max_new=900, temp=0.8, adapter=True, bs=4):
    """Generate k candidates for EACH src, batching several prompts per forward pass to parallelize on the
    GPU (left-padded). Returns a list (per src) of k decoded strings. This is the big throughput win over
    calling generate() once per task."""
    old_side = tok.padding_side; tok.padding_side = "left"
    results = [[] for _ in srcs]
    ctx = mdl.disable_adapter() if not adapter else _null()
    try:
        with ctx:
            for i in range(0, len(srcs), bs):
                chunk = srcs[i:i + bs]
                texts = [_chat(tok, _prompt(s)) for s in chunk]
                enc = tok(texts, return_tensors="pt", padding=True).to(_dev(mdl))
                with torch.no_grad():
                    out = mdl.generate(**enc, do_sample=True, temperature=temp, top_p=0.95,
                                       num_return_sequences=k, max_new_tokens=max_new,
                                       pad_token_id=tok.pad_token_id, logits_processor=_LP)
                new = out[:, enc["input_ids"].shape[1]:]      # (len(chunk)*k, gen_len), grouped by prompt
                for j in range(len(chunk)):
                    for r in range(k):
                        results[i + j].append(tok.decode(new[j * k + r], skip_special_tokens=True))
                torch.cuda.empty_cache()
    finally:
        tok.padding_side = old_side
    return results


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
               HF_HOME="/tmp/instance_storage/ka_data/hf",
               CUDA_VISIBLE_DEVICES=os.environ.get("KA_GRADE_GPU", "2"))   # grade on a spare GPU (arms hold 0,1)
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "grade_batch.py"), path],
                           capture_output=True, text=True, timeout=240, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        res = json.loads(line[-1][len("RESULT"):]) if line else []
    except Exception:
        res = []
    os.remove(path)
    return res + [[False, 0.0]] * (len(codes) - len(res))


def _grade_isolated_batch(items, chunk=12):
    """Grade MANY tasks' candidates, amortizing torch-init across a whole held-eval: one subprocess per
    `chunk` tasks (not per task). items = [(src, codes), ...]; returns [[(ok,sp),...] per task]. If a chunk's
    subprocess dies (a kernel poisons CUDA), fall back to per-task grading for that chunk so one bad kernel
    only costs its own chunk's speed, never correctness."""
    import subprocess, tempfile
    out = [None] * len(items)
    for i in range(0, len(items), chunk):
        grp = items[i:i + chunk]
        payload = {"batch": [{"task": s, "codes": c} for s, c in grp]}
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        json.dump(payload, open(path, "w"))
        env = dict(os.environ, PYTHONPATH="/tmp/instance_storage:/tmp/instance_storage/kernelascent",
                   HF_HOME="/tmp/instance_storage/ka_data/hf",
                   CUDA_VISIBLE_DEVICES=os.environ.get("KA_GRADE_GPU", "2"))
        res = None
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "grade_batch.py"), path],
                               capture_output=True, text=True, timeout=600, env=env)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            res = json.loads(line[-1][len("RESULT"):]) if line else None
        except Exception:
            res = None
        os.remove(path)
        if not res or len(res) != len(grp):                    # chunk failed -> per-task fallback
            res = [_grade_isolated(s, c) for s, c in grp]
        for j, (s, c) in enumerate(grp):
            rj = res[j] if j < len(res) else []
            out[i + j] = rj + [[False, 0.0]] * (len(c) - len(rj))
    return out


def eval_tasks(tok, mdl, names, k, adapter=True):
    """mean over tasks of best-of-k speed-resolved score (capability C). Training examples = the BEST
    CORRECT candidate per task (rejection-sampling SFT). Generation runs in-process (safe); candidate
    EXECUTION is isolated in a subprocess so a bad kernel can't poison the trainer's CUDA context."""
    srcs = [LK.TASKS[n] for n in names]
    gen_lists = generate_batch(tok, mdl, srcs, k, adapter=adapter)          # batched generation (big speedup)
    per_task_codes = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in gen_lists]
    grades = _grade_isolated_batch(list(zip(srcs, per_task_codes)))         # batched grading (amortized torch-init)
    scores = []; examples = []
    for src, codes, res in zip(srcs, per_task_codes, grades):
        best = 0.0
        for code, (ok, sp) in zip(codes, res):
            s = LK._score(ok, sp)
            if s > best:
                best = s
            if ok:                                         # keep ALL correct kernels -> more SFT data
                examples.append((src, code))
        scores.append(best)
    mean = statistics.mean(scores) if scores else 0.0
    ci = (1.96 * statistics.pstdev(scores) / (len(scores) ** 0.5)) if len(scores) > 1 else 0.0
    return mean, examples, scores, ci


def sft(tok, mdl, pairs, steps, lr=2e-5, bs=2):
    """LoRA rejection-sampling SFT on (task_src, ModelNew code) pairs; loss on completion tokens only.
    Stabilized: low lr + grad clipping + steps scaled to data size (avoid the overfit-to-NaN that made
    generation emit inf/nan logits)."""
    if not pairs:
        return 0.0
    mdl.train(); mdl.gradient_checkpointing_enable(); mdl.config.use_cache = False   # cut 7B activation memory
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)
    dev = _dev(mdl)
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
        input_ids = torch.tensor([i + [tok.pad_token_id] * (m - len(i)) for i, _ in batch]).to(dev)
        lab = torch.tensor([l + [-100] * (m - len(l)) for _, l in batch]).to(dev)
        att = (input_ids != tok.pad_token_id).long()
        out = mdl(input_ids=input_ids, attention_mask=att, labels=lab)
        if not torch.isfinite(out.loss):
            opt.zero_grad(); continue                      # skip a non-finite step
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in mdl.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(); losses.append(out.loss.item())
    mdl.gradient_checkpointing_disable(); mdl.config.use_cache = True; mdl.eval()   # restore fast generation
    del opt; import gc; gc.collect(); torch.cuda.empty_cache()   # reclaim optimizer/activation memory (fp32 7B is tight on 40GB)
    return statistics.mean(losses) if losses else 0.0


def run(args):
    random.seed(0); torch.manual_seed(0)
    for n in LK.TASKS:
        LK._ref(n)                                          # warm GPU baselines
    sg = [int(x) for x in str(args.self_gpu).split(",")]    # SELF arm sharded across these GPUs
    cg = [int(x) for x in str(args.ctrl_gpu).split(",")]    # CONTROL arm sharded across these GPUs
    tok, mdl = build(args.model, sg)                        # SELF arm: trains on its own kernels each round
    tok2, ctrl = build(args.model, cg)                      # CONTROL arm: trains ONLY on round-0 kernels
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    train, held = names[:args.n_train], names[args.n_train:]
    print("WEIGHT-RSI %s train=%d held=%d k=%d" % (args.model, len(train), len(held), args.k), flush=True)
    C0, _, _, c0ci = eval_tasks(tok, mdl, held, args.k, adapter=False)   # frozen-base (no training)
    print("C0 frozen-base held-out = %.3f +-%.3f" % (C0, c0ci), flush=True)
    hist = []; ex0 = None
    for r in range(args.rounds):
        t0 = time.time()
        trainC, pairs, _, _ = eval_tasks(tok, mdl, train, args.k, adapter=True)   # self: generate+grade train
        if ex0 is None:
            ex0 = pairs                                     # freeze round-0 self-generated data for the control
        def _try_sft(t, m, p):                              # resilient: a rare OOM round skips training, run continues
            try:
                return sft(t, m, p, args.sft_steps)
            except torch.cuda.OutOfMemoryError:
                import gc; gc.collect(); torch.cuda.empty_cache()
                print("  [oom] skipped sft this round", flush=True); return float("nan")
        loss = _try_sft(tok, mdl, pairs)                    # SELF: SFT on THIS round's own kernels
        _try_sft(tok2, ctrl, ex0)                           # CONTROL: SFT again on round-0 kernels only
        Cs, _, _, sci = eval_tasks(tok, mdl, held, args.k, adapter=True)
        Cc, _, _, cci = eval_tasks(tok2, ctrl, held, args.k, adapter=True)
        row = {"round": r, "trainC": round(trainC, 3), "n_ex": len(pairs), "loss": round(loss, 3),
               "C_self": round(Cs, 3), "C_self_ci": round(sci, 3), "C_ctrl": round(Cc, 3), "C_ctrl_ci": round(cci, 3),
               "delta_self": round(Cs - C0, 3), "delta_self_minus_ctrl": round(Cs - Cc, 3)}
        hist.append(row)
        print("round %d trainC=%.3f ex=%d | C_self=%.3f+-%.3f C_ctrl=%.3f+-%.3f | dSelf=%+.3f self-ctrl=%+.3f (%.0fs)" %
              (r, trainC, len(pairs), Cs, sci, Cc, cci, Cs - C0, Cs - Cc, time.time() - t0), flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"model": args.model, "C0_frozen": C0, "C0_ci": c0ci, "history": hist},
                  open(os.path.join(args.outdir, "weight_rsi.json"), "w"), indent=2)
    print("\n=== WEIGHT-RSI SUMMARY (%s) ===" % args.model)
    print("  frozen-base C0 = %.3f +-%.3f" % (C0, c0ci))
    print("  C_self by round :", [h["C_self"] for h in hist])
    print("  C_ctrl by round :", [h["C_ctrl"] for h in hist])
    print("  self-minus-ctrl :", [h["delta_self_minus_ctrl"] for h in hist])
    scm = [h["delta_self_minus_ctrl"] for h in hist]
    print("  RSI (self-training on NEW kernels beats retraining on round-0)?",
          "YES" if len(scm) >= 3 and statistics.mean(scm[-2:]) > 0.05 else "not resolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=3); ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--self-gpu", default="0,1"); ap.add_argument("--ctrl-gpu", default="2,3")
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/weight_rsi")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
