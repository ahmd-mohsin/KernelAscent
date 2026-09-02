"""KernelAscent agent benchmark.

Shows each generated Model source to an LLM agent (Qwen2.5-Coder-7B), which returns an
optimized ModelNew. We sample k candidates per task (multiple agents in parallel via
num_return_sequences), grade each against an fp32 gold, and keep the fastest correct one.
Reports pass@k (any correct) and fast_p over best-of-k speedups vs the eager baseline.
"""
import argparse, re, time, statistics, math, json, os, sys, uuid, importlib.util, torch
import gen_source_tasks

CAND_DIR = "/tmp/instance_storage/ka_data/cand_modules"
os.makedirs(CAND_DIR, exist_ok=True)


def load_module(source):
    """Write candidate to a real file and import it, so triton.jit can inspect source."""
    name = "cand_%s" % uuid.uuid4().hex
    path = os.path.join(CAND_DIR, name + ".py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYS = ("You are an elite GPU performance engineer. You optimize PyTorch modules to run "
       "faster on an NVIDIA A100, keeping outputs numerically equivalent.")

PROMPT = """Optimize the following PyTorch module for speed on an A100 GPU.

Rules:
- Keep __init__ EXACTLY the same (same parameters, shapes, and initialization order). Only rewrite forward.
- You may use Triton (import triton, triton.language as tl) or fused PyTorch ops. Prefer custom Triton kernels for reductions and elementwise fusion.
- Produce numerically equivalent outputs.
- Output ONE complete class named ModelNew with the same interface, inside a single ```python code block. No prose.

Reference module:
```python
{src}
```"""


def rel_l2(a, b):
    return (torch.linalg.vector_norm((a - b).float()) /
            (torch.linalg.vector_norm(b.float()) + 1e-12)).item()


def flush_l2():
    x = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda"); x.zero_(); del x


def time_fn(fn, inputs, iters=20, warmup=8):
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush_l2(); torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(*inputs); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def build_ref(task_source):
    ns = {}
    exec(task_source, ns)
    DT = ns["DT"]
    ref = ns["Model"](DT).cuda().eval()
    gold_m = ns["Model"](torch.float32).cuda().eval()
    x = ns["get_inputs"]()[0].cuda()
    with torch.no_grad():
        ref_out = ref(x)
        gold = gold_m(x.float())
    ref_err = rel_l2(ref_out, gold)
    return ref, x, gold, ref_err


def extract_modelnew(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    for b in blocks:
        if "class ModelNew" in b:
            return b
    if "class ModelNew" in text:                      # unfenced fallback
        return text[text.index("class ModelNew"):]
    return None


def grade(task_source, llm_code, ref, x, gold, bound, tbase):
    try:
        mod = load_module(task_source + "\n" + llm_code)
        MN = mod.ModelNew; DT = mod.DT
        try:
            cand = MN(DT).cuda().eval()
        except TypeError:
            cand = MN().cuda().eval()
        with torch.no_grad():
            out = cand(x)
        err = rel_l2(out, gold)
        ok = (out.shape == gold.shape) and (err <= bound) and torch.isfinite(out).all().item()
    except Exception as e:
        return False, float("inf"), 0.0, repr(e)[:80]
    if not ok:
        return False, err, 0.0, "wrong/imprecise"
    t_cand = time_fn(lambda z: cand(z), (x,))
    return True, err, tbase / t_cand, "ok"


class Agent:
    def __init__(self, gpu=0):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16).cuda().eval()

    def optimize(self, src, k, temp, max_new_tokens=1200):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": PROMPT.format(src=src)}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inp, do_sample=True, temperature=temp, top_p=0.95,
                                      num_return_sequences=k, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[:, inp["input_ids"].shape[1]:]
        return [extract_modelnew(t) for t in self.tok.batch_decode(gen, skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="samples (agents) per task")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = gen_source_tasks.generate(args.n, args.seed0)
    agent = Agent()
    print("device=%s  model=%s  n=%d  k=%d" %
          (torch.cuda.get_device_name(0), MODEL_ID, args.n, args.k))
    print("%-4s %-9s %10s %6s %8s %9s  %s" %
          ("tier", "family", "MxD", "pass@k", "best_sp", "best_err", "chain"))
    records, best_sps, any_correct = [], [], 0
    for task in tasks:
        ref, x, gold, ref_err = build_ref(task["source"])
        bound = max(args.tol, args.margin * ref_err)
        tbase = time_fn(lambda z: ref(z), (x,))
        cands = agent.optimize(task["source"], args.k, args.temp)
        best_sp, best_err, n_ok = 0.0, float("inf"), 0
        for code in cands:
            if not code:
                continue
            ok, err, sp, _ = grade(task["source"], code, ref, x, gold, bound, tbase)
            if ok:
                n_ok += 1
                if sp > best_sp:
                    best_sp, best_err = sp, err
        any_correct += int(n_ok > 0)
        best_sps.append(best_sp)
        records.append(dict(name=task["name"], tier=task["tier"], family=task["family"],
                            meta=task["meta"], pass_at_k=n_ok, best_speedup=best_sp,
                            best_err=(best_err if best_err != float("inf") else None)))
        print("%-4s %-9s %10s %6s %8.3f %9.2e  %s" %
              (task["tier"], task["family"], "%dx%d" % (task["meta"]["M"], task["meta"]["D"]),
               "%d/%d" % (n_ok, args.k), best_sp,
               best_err if best_err != float("inf") else float("nan"),
               "-".join(task["meta"]["chain"])[:40]))

    n = max(len(best_sps), 1)
    fp = lambda p: sum(1 for s in best_sps if s > p) / n
    summ = dict(n=len(best_sps), pass_at_k=any_correct / n, k=args.k,
                fast_1=fp(1.0), fast_1_5=fp(1.5), fast_2=fp(2.0),
                geomean_pass=math.exp(sum(math.log(s) for s in best_sps if s > 0) /
                                      max(1, sum(1 for s in best_sps if s > 0))) if any(s > 0 for s in best_sps) else 0.0)
    print("\nTOTAL n=%d pass@%d=%.2f  fast_1=%.2f fast_1.5=%.2f fast_2=%.2f  geomean(pass)=%.3f" %
          (summ["n"], args.k, summ["pass_at_k"], summ["fast_1"], summ["fast_1_5"],
           summ["fast_2"], summ["geomean_pass"]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summ, "tasks": records}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
