"""COMBINED (TRUE) RSI — a researcher improves the training HARNESS, the harness trains an open model,
and the loop closes.

This is the flagship. It ties procedure improvement to a downstream capability payoff, with a clean causal
control. A RESEARCHER model (open OR closed, weights fixed) edits an executable HARNESS. The harness then
TRAINS a separate open-weight TRAINEE (LoRA). We ask whether a researcher-improved harness produces a better
trainee than a frozen harness, and whether iterating compounds.

Editable HARNESS H (what the researcher rewrites each round):
  H.strategy   guidance text prepended to the trainee's kernel-generation prompt
  H.min_sp     data-admission policy: keep a correct kernel for SFT only if compiled-speedup >= min_sp
Round r:
  1. trainee M writes kernels for train tasks, guided by H.strategy      (GPU graded: eager + compiled)
  2. admit SFT data per H.min_sp                                         (data-selection policy)
  3. M <- LoRA SFT on admitted data                                     (trainee WEIGHTS change)
  4. researcher R rewrites H from the round's evidence -> H'            (procedure improves; R weights fixed)
  5. measure held-out capability of M
Arms (paired, same trainee init, same tasks, same budget):
  frozen-harness    H never changes (base strategy, base admission)     -> baseline training
  improved-harness  R rewrites H each round                             -> value of an IMPROVING procedure
Primary causal contrast: C(improved) - C(frozen). Positive => the researcher's harness improvements CAUSED a
better trained open model. Sustained/rising across rounds => compounding. This is the weight x harness
interaction, isolated by a matched control.

Reuses lab_weight_rsi (build/generate_batch/sft/_grade_isolated_batch, eager+compiled grader) and the
standardized bank. Researcher generation: open model via lab_weight_rsi.generate, or an API callable.
"""
import os, sys, json, argparse, random, statistics, time, re
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
import torch
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_kernel as LK
from kernelascent.v3 import lab_weight_rsi as W

BASE_STRATEGY = "Write a correct, self-contained kernel. Prefer fused torch ops; use Triton only if it clearly helps."

RESEARCH_SYS = ("You improve the TRAINING HARNESS for a smaller model that writes GPU kernels. You are given the "
                "current harness (a strategy note and a data-admission speedup threshold) and evidence from the last "
                "round (per-task correctness and compiled speedups). Rewrite the harness to make the NEXT round of "
                "training produce a faster, still-correct kernel writer. Return ONLY JSON: "
                '{"strategy": "<one concise paragraph>", "min_sp": <float 0.0-1.3>}.')


def _prompt_with_strategy(strategy, task_src):
    return ("Optimize this PyTorch module for speed on an A100, numerically equivalent. %s Return ONLY a class "
            "ModelNew(nn.Module) in a python code block.\n%s" % (strategy, task_src))


def eval_harness(tok, mdl, names, k, strategy, grade, adapter=True):
    """Generate (guided by `strategy`) + grade. Returns mean best-C, and admitted-example candidates with sp."""
    srcs = [LK.TASKS[n] for n in names]
    old = W._prompt
    W._prompt = lambda s: _prompt_with_strategy(strategy, s)      # inject harness strategy into the trainee prompt
    try:
        gen_lists = W.generate_batch(tok, mdl, srcs, k, adapter=adapter)
    finally:
        W._prompt = old
    per = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in gen_lists]
    grades = W._grade_isolated_batch(list(zip(srcs, per)))
    scores = []; cand = []
    for src, codes, res in zip(srcs, per, grades):
        best = 0.0
        for code, g in zip(codes, res):
            ok, se, sc = (g + [0, 0, 0])[:3]
            best = max(best, W.LK._score(ok, se))
            if ok:
                cand.append((src, code, sc))
        scores.append(best)
    return (statistics.mean(scores) if scores else 0.0), cand


def researcher_revise(H, evidence, researcher):
    """Researcher rewrites the harness from evidence. `researcher` is a callable(user, system)->text (open or API)."""
    prompt = ("Current harness: %s\nEvidence:\n%s\n\nReturn improved JSON." %
              (json.dumps(H), "\n".join(evidence[:24])))
    out = researcher(prompt, RESEARCH_SYS) or ""
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"strategy": str(d.get("strategy", H["strategy"]))[:600],
                    "min_sp": float(max(0.0, min(1.3, d.get("min_sp", H["min_sp"]))))}
        except Exception:
            pass
    return dict(H)


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    tg = [int(x) for x in str(args.trainee_gpu).split(",")]
    cg = [int(x) for x in str(args.ctrl_gpu).split(",")]
    tok, mdl = W.build(args.trainee, tg)          # improved-harness trainee
    tok2, ctrl = W.build(args.trainee, cg)        # frozen-harness control trainee
    # researcher backend: open model on its own GPUs, or (future) an API callable
    if args.researcher_gpu:
        rg = [int(x) for x in str(args.researcher_gpu).split(",")]
        rtok, rmdl = W.build(args.researcher, rg)
        def researcher(user, system):
            old = W.SYS; W.SYS = system
            try:
                outs = W.generate(rtok, rmdl, user, 1, max_new=700, adapter=False)
            finally:
                W.SYS = old
            return outs[0] if outs else ""
    else:                                          # API researcher (closed model) via curate_bedrock
        import curate_bedrock as CB
        cur = CB.Curator(args.researcher, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        cur.resolve(); cur.resolve_reasoning()
        def researcher(user, system):
            return cur.generate(user) or ""
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    train, held = names[:args.n_train], names[args.n_train:]
    H = {"strategy": BASE_STRATEGY, "min_sp": 0.0}          # improved arm harness (evolves)
    H0 = {"strategy": BASE_STRATEGY, "min_sp": 0.0}         # frozen arm harness (constant)
    print("COMBINED-RSI trainee=%s researcher=%s train=%d held=%d k=%d" %
          (args.trainee, args.researcher, len(train), len(held), args.k), flush=True)
    grade = None
    C0, _ = eval_harness(tok, mdl, held, args.k, H0["strategy"], grade, adapter=False)
    print("C0 trainee (no training) = %.3f" % C0, flush=True)
    hist = []
    for r in range(args.rounds):
        t0 = time.time()
        # improved arm: solve with current H, admit per H.min_sp, train
        _, cand = eval_harness(tok, mdl, train, args.k, H["strategy"], grade, adapter=True)
        ev = ["task_correct=%d avg_sp=%.2f min_sp_policy=%.2f" % (len(cand), (statistics.mean([c[2] for c in cand]) if cand else 0), H["min_sp"])]
        pairs = [(s, c) for (s, c, sp) in cand if sp >= H["min_sp"]]
        try: W.sft(tok, mdl, pairs, args.sft_steps)
        except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache()
        # frozen arm: solve with base H0, admit-all, train
        _, cand0 = eval_harness(tok2, ctrl, train, args.k, H0["strategy"], grade, adapter=True)
        pairs0 = [(s, c) for (s, c, sp) in cand0]
        try: W.sft(tok2, ctrl, pairs0, args.sft_steps)
        except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache()
        # researcher improves the harness for next round
        Hn = researcher_revise(H, ev, researcher)
        # measure held-out for both arms
        Ci, _ = eval_harness(tok, mdl, held, args.k, H["strategy"], grade, adapter=True)
        Cf, _ = eval_harness(tok2, ctrl, held, args.k, H0["strategy"], grade, adapter=True)
        row = {"round": r, "C_improved": round(Ci, 3), "C_frozen": round(Cf, 3),
               "delta_improved_minus_frozen": round(Ci - Cf, 3), "n_admitted": len(pairs),
               "min_sp": round(H["min_sp"], 2), "strategy": H["strategy"][:120]}
        hist.append(row)
        print("round %d C_improved=%.3f C_frozen=%.3f improved-frozen=%+.3f admit=%d min_sp=%.2f (%.0fs)" %
              (r, Ci, Cf, Ci - Cf, len(pairs), H["min_sp"], time.time() - t0), flush=True)
        H = Hn                                                 # adopt improved harness for next round
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"trainee": args.trainee, "researcher": args.researcher, "C0": C0, "history": hist},
                  open(os.path.join(args.outdir, "combined_rsi.json"), "w"), indent=2)
    d = [h["delta_improved_minus_frozen"] for h in hist]
    print("\n=== COMBINED-RSI SUMMARY ===")
    print("  improved-minus-frozen by round:", d)
    print("  harness-causal RSI (improved harness trains a better model)?",
          "YES" if len(d) >= 3 and statistics.mean(d[-2:]) > 0.05 else "not resolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainee", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--researcher", default="Qwen/Qwen2.5-Coder-14B-Instruct", help="open hf-id, or API model id if --researcher-gpu empty")
    ap.add_argument("--trainee-gpu", default="0"); ap.add_argument("--ctrl-gpu", default="1")
    ap.add_argument("--researcher-gpu", default="2,3", help="GPUs for an OPEN researcher; empty => API (Bedrock)")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=20); ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/combined_rsi")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
