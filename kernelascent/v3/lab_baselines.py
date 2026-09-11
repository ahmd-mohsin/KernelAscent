"""Compute-matched NON-RECURSIVE baselines for KernelAscent (roadmap #1).

Purpose: show that weight-RSI / procedure-RSI improvement comes FROM recursion, not merely from spending
more generation budget. Every baseline is scored with the SAME grader and the SAME best-of-k _score as
lab_weight_rsi.eval_tasks, on the SAME held split, so its C is directly comparable to the RSI C.

Baselines (all frozen weights):
  best-of-k   : draw B = rounds*k samples per held task from the frozen model, keep the best. Pure sampling.
  self-refine : in-context iterative repair. Each iteration re-shows the model its own best kernel so far and
                the grader verdict (correct? compiled speedup?) and asks for a better one. No weight change.
                Budget-matched: iters*k total draws per task = rounds*k.
  retrieval   : build an archive of the frozen model's best correct kernels from the TRAIN split, then solve
                each held task with the top-n archive kernels injected as few-shot context (LK.OPT `arch`
                slot). No weight change. Budget-matched on the held draws.

Primary comparison: weight-RSI final C_self  vs  max(best-of-k, self-refine, retrieval) at equal samples.
If RSI wins, recursion adds value beyond sampling. Output: baselines.json {method: {C, ci, correct_rate, compiled_sp}}.
"""
import os, sys, json, argparse, statistics, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch


def _score_lists(srcs, code_lists):
    """Grade per-task candidate lists and return (mean_C, scores, correct_rate, compiled_sp) exactly as
    eval_tasks does (best-of-k speed-resolved score, KA_SCORE=compiled aware). Also returns correct (src,code)
    pairs for archive building."""
    grades = W._grade_isolated_batch(list(zip(srcs, code_lists)))
    use_compiled = os.environ.get("KA_SCORE", "eager") == "compiled"
    scores, corr, comp, correct_pairs = [], [], [], []
    for src, codes, res in zip(srcs, code_lists, grades):
        best = 0.0; n_ok = 0; best_c = 0.0
        for code, g in zip(codes, res):
            ok, se, sc, ceil = (list(g) + [0.0, 0.0, 0.0, 1.5])[:4]
            s = LK._score(ok, sc if use_compiled else se, ceiling=(ceil if use_compiled else 1.5))
            if s > best:
                best = s
            if ok:
                n_ok += 1; best_c = max(best_c, sc); correct_pairs.append((src, code))
        scores.append(best); corr.append(1.0 if n_ok > 0 else 0.0); comp.append(best_c)
    mean = statistics.mean(scores) if scores else 0.0
    ci = (1.96 * statistics.pstdev(scores) / (len(scores) ** 0.5)) if len(scores) > 1 else 0.0
    return mean, ci, round(statistics.mean(corr), 3) if corr else 0.0, \
        round(statistics.mean(comp), 3) if comp else 0.0, scores, correct_pairs


def _gen_custom(tok, mdl, user_texts, k, max_new=900, temp=0.8, bs=4):
    """Generate k candidates for each pre-built USER text (already the full instruction, not a raw src).
    Mirrors W.generate_batch but lets us inject refine/retrieval prompts. Frozen model (adapter disabled)."""
    old = tok.padding_side; tok.padding_side = "left"
    out_lists = [[] for _ in user_texts]
    try:
        with mdl.disable_adapter() if hasattr(mdl, "disable_adapter") else W._null():
            for i in range(0, len(user_texts), bs):
                chunk = user_texts[i:i + bs]
                texts = [W._chat(tok, u) for u in chunk]
                enc = tok(texts, return_tensors="pt", padding=True).to(W._dev(mdl))
                with torch.no_grad():
                    o = mdl.generate(**enc, do_sample=True, temperature=temp, top_p=0.95,
                                     num_return_sequences=k, max_new_tokens=max_new,
                                     pad_token_id=tok.pad_token_id, logits_processor=W._LP)
                new = o[:, enc["input_ids"].shape[1]:]
                for j in range(len(chunk)):
                    for r in range(k):
                        out_lists[i + j].append(tok.decode(new[j * k + r], skip_special_tokens=True))
                torch.cuda.empty_cache()
    finally:
        tok.padding_side = old
    return out_lists


def best_of_k(tok, mdl, held, budget):
    """Frozen best-of-B sampling (B = rounds*k). This is exactly eval_tasks on the frozen base at high k."""
    C, ex, scores, ci, st = W.eval_tasks(tok, mdl, held, budget, adapter=False)
    return {"method": "best_of_k", "budget_per_task": budget, "C": round(C, 4), "ci": round(ci, 4),
            "correct_rate": st["correct_rate"], "compiled_sp": st["compiled_sp"]}


def retrieval(tok, mdl, train, held, k, n_shot=3):
    """Build an archive from the frozen model's correct TRAIN kernels, then solve held with top-n_shot
    archive kernels as few-shot context (LK.OPT arch slot). No weight change."""
    tsrcs = [LK.TASKS[n] for n in train]
    tgen = _gen_custom(tok, mdl, [LK.OPT.format(arch="", src=s) for s in tsrcs], k)
    tcodes = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in tgen]
    _, _, _, _, _, pairs = _score_lists(tsrcs, tcodes)
    archive = [code for (_, code) in pairs][:24]
    def arch_text(pool):
        pick = random.sample(pool, min(n_shot, len(pool))) if pool else []
        return "".join("\n# Example optimized kernel:\n```python\n%s\n```\n" % c for c in pick)
    hsrcs = [LK.TASKS[n] for n in held]
    hgen = _gen_custom(tok, mdl, [LK.OPT.format(arch=arch_text(archive), src=s) for s in hsrcs], k)
    hcodes = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in hgen]
    C, ci, cr, csp, _, _ = _score_lists(hsrcs, hcodes)
    return {"method": "retrieval", "n_shot": n_shot, "archive": len(archive), "C": round(C, 4),
            "ci": round(ci, 4), "correct_rate": cr, "compiled_sp": csp}


def self_refine(tok, mdl, held, iters, k):
    """In-context iterative repair. Each round re-shows the model its best kernel so far + grader verdict and
    asks for a better one. Budget = iters*k draws per task (matched to rounds*k). No weight change."""
    hsrcs = [LK.TASKS[n] for n in held]
    best_score = [0.0] * len(hsrcs); best_code = [None] * len(hsrcs)
    for it in range(iters):
        prompts = []
        for i, s in enumerate(hsrcs):
            if best_code[i] is None:
                prompts.append(LK.OPT.format(arch="", src=s))
            else:
                verdict = ("Your previous attempt was correct but only %.2fx vs torch.compile; make it faster."
                           % best_score[i] if best_score[i] > 0 else
                           "Your previous attempt was INCORRECT or failed; fix correctness then optimize.")
                prompts.append(LK.OPT.format(arch="", src=s) +
                               "\n\n# Previous attempt:\n```python\n%s\n```\n# %s\n" % (best_code[i], verdict))
        gen = _gen_custom(tok, mdl, prompts, k)
        codes = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in gen]
        grades = W._grade_isolated_batch(list(zip(hsrcs, codes)))
        use_compiled = os.environ.get("KA_SCORE", "eager") == "compiled"
        for i, (codes_i, res_i) in enumerate(zip(codes, grades)):
            for code, g in zip(codes_i, res_i):
                ok, se, sc, ceil = (list(g) + [0.0, 0.0, 0.0, 1.5])[:4]
                sval = LK._score(ok, sc if use_compiled else se, ceiling=(ceil if use_compiled else 1.5))
                if sval > best_score[i]:
                    best_score[i] = sval
                    if ok:
                        best_code[i] = code
    mean = statistics.mean(best_score) if best_score else 0.0
    ci = (1.96 * statistics.pstdev(best_score) / (len(best_score) ** 0.5)) if len(best_score) > 1 else 0.0
    cr = round(statistics.mean([1.0 if c else 0.0 for c in best_code]), 3)
    return {"method": "self_refine", "iters": iters, "k": k, "C": round(mean, 4), "ci": round(ci, 4),
            "correct_rate": cr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--methods", default="best_of_k,self_refine,retrieval")
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "baselines"))
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    gpus = [int(x) for x in str(args.gpus).split(",")]
    tok, mdl = W.build(args.model, gpus)
    names = list(LK.TASKS); random.Random(1).shuffle(names)           # SAME split seed as weight-RSI.run
    train, held = names[:args.n_train], names[args.n_train:]
    budget = args.rounds * args.k                                     # compute-matched to cumulative RSI sampling
    print("BASELINES %s seed=%d held=%d budget=%d (rounds*k)" % (args.model, args.seed, len(held), budget), flush=True)
    out = {"model": args.model, "seed": args.seed, "held": len(held), "budget_per_task": budget, "results": {}}
    want = args.methods.split(",")
    if "best_of_k" in want:
        out["results"]["best_of_k"] = best_of_k(tok, mdl, held, budget); print(out["results"]["best_of_k"], flush=True)
        json.dump(out, open(os.path.join(args.outdir, "baselines.json"), "w"), indent=2)
    if "self_refine" in want:
        out["results"]["self_refine"] = self_refine(tok, mdl, held, args.rounds, args.k); print(out["results"]["self_refine"], flush=True)
        json.dump(out, open(os.path.join(args.outdir, "baselines.json"), "w"), indent=2)
    if "retrieval" in want:
        out["results"]["retrieval"] = retrieval(tok, mdl, train, held, budget); print(out["results"]["retrieval"], flush=True)
        json.dump(out, open(os.path.join(args.outdir, "baselines.json"), "w"), indent=2)
    print("BASELINES DONE", json.dumps(out["results"]), flush=True)


if __name__ == "__main__":
    main()
