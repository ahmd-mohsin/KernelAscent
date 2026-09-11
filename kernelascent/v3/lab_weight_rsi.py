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


def _diversity(codes):
    """Self-generated-data diversity (roadmap #2). Returns (n_unique, mean_pairwise_distinctness) where
    distinctness = 1 - difflib similarity ratio, averaged over pairs (0=identical clones, 1=all different).
    Diversity collapse of a model's own correct kernels is the proposed mechanism for large-model overfit."""
    import difflib
    uniq = list({c.strip() for c in codes if c and c.strip()})
    if len(uniq) < 2:
        return len(uniq), 0.0
    sample = uniq[:12]                                          # cap pairs for speed
    ds = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            ds.append(1.0 - difflib.SequenceMatcher(None, sample[i], sample[j]).ratio())
    return len(uniq), round(sum(ds) / len(ds), 3) if ds else 0.0


def build(model_id, gpus=(0,)):
    """Load one arm sharded across the given GPU indices (fp32 7B needs >40GB during SFT, so 2 GPUs/arm).
    max_memory forces accelerate to place layers ONLY on `gpus` (others capped at 0), keeping the two arms
    on disjoint devices within one process."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if os.environ.get("KA_DTYPE", "bf16") == "bf16" else torch.float32
    gpus = tuple(gpus)
    if len(gpus) == 1:
        dm = {"": "cuda:%d" % gpus[0]}
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, device_map=dm, trust_remote_code=True)
    else:
        n = torch.cuda.device_count()
        cap = os.environ.get("KA_MAXMEM_GIB", "20")            # per-GPU cap FORCES even sharding across the arm's GPUs
        mm = {i: ("%sGiB" % cap if i in gpus else "0GiB") for i in range(n)}
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, device_map="auto", max_memory=mm, trust_remote_code=True)
    try:                                                   # standard attention-proj names (Qwen/Llama/Mistral/Gemma/DeepSeek/StarCoder2)
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        mdl = get_peft_model(mdl, lcfg)
    except (ValueError, KeyError):                         # fused/odd names (e.g. Phi-3 qkv_proj) -> target all linears
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
        mdl = get_peft_model(mdl, lcfg)
    mdl.enable_input_require_grads()                       # needed for grad-checkpointing during SFT
    return tok, mdl


def _dev(mdl):
    return next(mdl.parameters()).device


def _chat(tok, user):
    """Robust across families: try system+user; if the template rejects a system role (Gemma etc.), merge
    system into the user turn; if there's no chat template at all (base models), fall back to plain text."""
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            try:
                return tok.apply_chat_template([{"role": "user", "content": SYS + "\n\n" + user}],
                                               tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
    return SYS + "\n\n" + user + "\n\n"


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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("KA_GRADE_GPU", "2"))   # inherit PYTHONPATH/KA_*/HF_HOME; grade on a spare GPU
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "grade_batch.py"), path],
                           capture_output=True, text=True, timeout=900, env=env)   # compile baseline is slow
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        res = json.loads(line[-1][len("RESULT"):]) if line else []
    except Exception:
        res = []
    os.remove(path)
    return res + [[False, 0.0, 0.0]] * (len(codes) - len(res))


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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("KA_GRADE_GPU", "2"))   # inherit PYTHONPATH/KA_*/HF_HOME
        res = None
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "grade_batch.py"), path],
                               capture_output=True, text=True, timeout=1200, env=env)   # compile baseline is slow
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            res = json.loads(line[-1][len("RESULT"):]) if line else None
        except Exception:
            res = None
        os.remove(path)
        if not res or len(res) != len(grp):                    # chunk failed -> per-task fallback
            res = [_grade_isolated(s, c) for s, c in grp]
        for j, (s, c) in enumerate(grp):
            rj = res[j] if j < len(res) else []
            out[i + j] = rj + [[False, 0.0, 0.0]] * (len(c) - len(rj))
    return out


def eval_tasks(tok, mdl, names, k, adapter=True):
    """mean over tasks of best-of-k speed-resolved score (capability C). Training examples = the BEST
    CORRECT candidate per task (rejection-sampling SFT). Generation runs in-process (safe); candidate
    EXECUTION is isolated in a subprocess so a bad kernel can't poison the trainer's CUDA context."""
    srcs = [LK.TASKS[n] for n in names]
    gen_lists = generate_batch(tok, mdl, srcs, k, adapter=adapter)          # batched generation (big speedup)
    per_task_codes = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in gen_lists]
    grades = _grade_isolated_batch(list(zip(srcs, per_task_codes)))         # [ok, sp_eager, sp_compiled] per candidate
    # KA_SCORE=compiled scores capability on the torch.compile speedup (headroom for correct-but-slow models
    # of ANY size), instead of the correctness-heavy eager score. Isolates SPEED optimization from correctness.
    use_compiled = os.environ.get("KA_SCORE", "eager") == "compiled"
    scores = []; examples = []; corr = []; comp = []
    for src, codes, res in zip(srcs, per_task_codes, grades):
        best = 0.0; n_ok = 0; best_c = 0.0
        for code, g in zip(codes, res):
            ok, se, sc = (g + [0.0, 0.0, 0.0])[:3]
            s = LK._score(ok, sc if use_compiled else se)  # compiled ratio => parity-with-compile=0.5, 1.5x=1.0
            if s > best:
                best = s
            if ok:
                n_ok += 1
                if sc > best_c:
                    best_c = sc
                examples.append((src, code))               # keep ALL correct kernels -> more SFT data
        scores.append(best)
        corr.append(1.0 if n_ok > 0 else 0.0)              # per-task solved-at-all (correctness, not speed)
        comp.append(best_c)                                # best compiled speedup among correct candidates
    mean = statistics.mean(scores) if scores else 0.0
    ci = (1.96 * statistics.pstdev(scores) / (len(scores) ** 0.5)) if len(scores) > 1 else 0.0
    stats = {"correct_rate": round(statistics.mean(corr), 3) if corr else 0.0,
             "compiled_sp": round(statistics.mean(comp), 3) if comp else 0.0}
    return mean, examples, scores, ci, stats


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


def _manifest(args, train, held):
    import hashlib, transformers, peft
    bank = json.dumps({n: LK.TASKS[n] for n in LK.TASKS}, sort_keys=True)
    return {"model": args.model, "seed": args.seed, "rounds": args.rounds, "k": args.k,
            "n_train": args.n_train, "sft_steps": args.sft_steps, "dtype": os.environ.get("KA_DTYPE", "bf16"),
            "bank_path": os.environ.get("KA_KERNEL_BANK", "?"),
            "bank_sha1": hashlib.sha1(bank.encode()).hexdigest()[:12], "n_tasks": len(LK.TASKS),
            "train_tasks": train, "held_tasks": held, "lora_targets": "q/k/v/o or all-linear",
            "transformers": transformers.__version__, "peft": peft.__version__,
            "grader": "eager+compiled, 3-input", "arms": ["self", "fresh_frozen", "round0_replay"]}


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    sg = [int(x) for x in str(args.self_gpu).split(",")]
    cg = [int(x) for x in str(args.ctrl_gpu).split(",")] if args.ctrl_gpu else None
    fg = [int(x) for x in str(args.fresh_gpu).split(",")] if args.fresh_gpu else None
    tok, mdl = build(args.model, sg)                        # SELF: producer = improving model, learner = itself
    tok2, ctrl = (build(args.model, cg) if cg else (None, None))  # ROUND0-REPLAY control (skipped if --ctrl-gpu empty)
    tok3, fr = (build(args.model, fg) if fg else (None, None))  # FRESH-FROZEN: producer = frozen base, fresh each round
    names = list(LK.TASKS); random.Random(1).shuffle(names)  # split seed fixed so train/held is stable across arms
    train, held = names[:args.n_train], names[args.n_train:]
    print("WEIGHT-RSI %s seed=%d train=%d held=%d k=%d arms=%s" %
          (args.model, args.seed, len(train), len(held), args.k,
           ",".join(["self"] + (["fresh"] if fg else []) + (["round0"] if cg else []))), flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(_manifest(args, train, held), open(os.path.join(args.outdir, "manifest.json"), "w"), indent=2)
    C0, _, _, c0ci, st0 = eval_tasks(tok, mdl, held, args.k, adapter=False)   # frozen-base held-out
    print("C0 frozen-base held-out = %.3f +-%.3f (correct=%.2f compiled_sp=%.2f)" %
          (C0, c0ci, st0["correct_rate"], st0["compiled_sp"]), flush=True)

    def _try_sft(t, m, p):
        try:
            return sft(t, m, p, args.sft_steps)
        except torch.cuda.OutOfMemoryError:
            import gc; gc.collect(); torch.cuda.empty_cache()
            print("  [oom] skipped sft this round", flush=True); return float("nan")

    n_gpus = len(sg) + (len(cg) if cg else 0) + (len(fg) if fg else 0)   # GPUs this run holds (arms)
    cum_gpu_s = 0.0; round0_solved = None
    hist = []; ex0 = None
    for r in range(args.rounds):
        t0 = time.time()
        trainC, pairs, _, _, _ = eval_tasks(tok, mdl, train, args.k, adapter=True)     # self producer, fresh
        if ex0 is None:
            ex0 = pairs                                                                # freeze round-0 self data
        loss = _try_sft(tok, mdl, pairs)                                               # SELF learns on its own fresh data
        if ctrl is not None:
            _try_sft(tok2, ctrl, ex0)                                                  # ROUND0 control re-trains on ex0
        Cf = fci = None; stf = {"correct_rate": None, "compiled_sp": None}
        if fr is not None:
            _, frpairs, _, _, _ = eval_tasks(tok, mdl, train, args.k, adapter=False)   # FROZEN base producer, fresh each round
            _try_sft(tok3, fr, frpairs)                                                # FRESH learner trains on frozen-base data
        Cs, _, held_scores, sci, sts = eval_tasks(tok, mdl, held, args.k, adapter=True)
        # forgetting/retention (roadmap #2): fraction of round-0-solved held tasks still solved this round
        solved = [1 if s > 0 else 0 for s in held_scores]
        if round0_solved is None:
            round0_solved = solved[:]
        r0 = [i for i, v in enumerate(round0_solved) if v]
        retention = round(sum(solved[i] for i in r0) / len(r0), 3) if r0 else None
        n_uniq, diversity = _diversity([c for (_, c) in pairs])       # diversity of THIS round's self data
        Cc = cci = None
        if ctrl is not None:
            Cc, _, _, cci, _ = eval_tasks(tok2, ctrl, held, args.k, adapter=True)
        if fr is not None:
            Cf, _, _, fci, stf = eval_tasks(tok3, fr, held, args.k, adapter=True)
        dt = time.time() - t0
        cum_gpu_s += dt * max(n_gpus, 1)                              # cost (roadmap #6): GPU-seconds this run
        n_gens = args.k * len(train) * (1 + (1 if fr is not None else 0) + (1 if ctrl is not None else 0))
        row = {"round": r, "trainC": round(trainC, 3), "n_ex": len(pairs), "loss": round(loss, 3),
               "C_self": round(Cs, 3), "C_self_ci": round(sci, 3),
               "C_ctrl": (round(Cc, 3) if Cc is not None else None), "C_ctrl_ci": (round(cci, 3) if cci is not None else None),
               "C_fresh": (round(Cf, 3) if Cf is not None else None), "C_fresh_ci": (round(fci, 3) if Cf is not None else None),
               "correct_rate_self": sts["correct_rate"], "compiled_sp_self": sts["compiled_sp"],
               "delta_self": round(Cs - C0, 3), "delta_self_minus_ctrl": (round(Cs - Cc, 3) if Cc is not None else None),
               "delta_self_minus_fresh": (round(Cs - Cf, 3) if Cf is not None else None),
               "n_uniq_self": n_uniq, "diversity_self": diversity, "retention": retention,
               "round_sec": round(dt, 1), "cum_gpu_hours": round(cum_gpu_s / 3600, 3), "n_gens": n_gens}
        hist.append(row)
        print("round %d trainC=%.3f ex=%d | C_self=%.3f C_fresh=%s C_ctrl=%s | dSelf=%+.3f self-fresh=%s self-ctrl=%s corr=%.2f csp=%.2f (%.0fs)" %
              (r, trainC, len(pairs), Cs, ("%.3f" % Cf if Cf is not None else "-"), ("%.3f" % Cc if Cc is not None else "-"), Cs - C0,
               ("%+.3f" % (Cs - Cf) if Cf is not None else "-"), ("%+.3f" % (Cs - Cc) if Cc is not None else "-"), sts["correct_rate"], sts["compiled_sp"], time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "C0_frozen": C0, "C0_ci": c0ci,
                   "C0_correct_rate": st0["correct_rate"], "C0_compiled_sp": st0["compiled_sp"], "history": hist},
                  open(os.path.join(args.outdir, "weight_rsi.json"), "w"), indent=2)
    print("\n=== WEIGHT-RSI SUMMARY (%s seed %d) ===" % (args.model, args.seed))
    print("  C0=%.3f  C_self:" % C0, [h["C_self"] for h in hist])
    print("  self-minus-fresh (producer-quality):", [h["delta_self_minus_fresh"] for h in hist])
    print("  self-minus-round0 :", [h["delta_self_minus_ctrl"] for h in hist])
    smf = [h["delta_self_minus_fresh"] for h in hist if h["delta_self_minus_fresh"] is not None]
    print("  producer-causal RSI (self beats fresh-frozen)?",
          "YES" if len(smf) >= 3 and statistics.mean(smf[-2:]) > 0.05 else "not resolved")
    print("  diversity_self (1=all distinct):", [h["diversity_self"] for h in hist])
    print("  retention (round-0 solved still solved):", [h["retention"] for h in hist])
    gpu_h = hist[-1]["cum_gpu_hours"] if hist else 0.0
    gain = (hist[-1]["C_self"] - C0) if hist else 0.0
    print("  cost: %.3f GPU-hours total | final gain vs C0 = %+.3f | GPU-hours per +0.01 C = %s"
          % (gpu_h, gain, ("%.3f" % (gpu_h / (gain * 100)) if gain > 0 else "n/a")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=3); ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--self-gpu", default="0,1"); ap.add_argument("--ctrl-gpu", default="2,3")
    ap.add_argument("--fresh-gpu", default="", help="GPUs for the fresh-frozen-producer arm (empty = disable)")
    ap.add_argument("--seed", type=int, default=0, help="lineage seed for cross-run CIs")
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/weight_rsi")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
