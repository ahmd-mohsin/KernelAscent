"""P2 (panel: highest impact/GPU-day) — IS THE CORRECTNESS WALL A COVERAGE GAP OR A SEARCH GAP?

For a wall-bound sub-2B student, matched training tokens, compare one SFT round on:
  SELF  : the student's OWN self-generated verified-correct kernels (few, because it is at the wall).
  EXT   : EXTERNAL verified-correct kernels from a strong teacher (Qwen-14B) on the SAME wall tasks.
If EXT lifts held-out C and SELF does not, the wall is a COVERAGE/KNOWLEDGE gap (the student cannot produce the
right kernels to learn from), not a search/decoding gap — which also explains why a correctness probe (reranking
what is already sampled) cannot help. Complement: a verification-budget sweep E in {4,16,64} — if best-of-E correct
rate does NOT rise with search, the wall is not a search gap either.

Reuses lab_weight_rsi (build/generate_batch/_grade_isolated/sft) + lab_probe_intervene (_gen_grade). Matched SFT
token budget across arms. bf16; grades isolated.
CLI: python -m kernelascent.v3.lab_wall_causal --student Qwen/Qwen2.5-Coder-1.5B-Instruct --teacher Qwen/Qwen2.5-Coder-14B-Instruct \
     --s-gpu 0 --t-gpu 1,2 --wall 16 --held 12 --K 16 --outdir <d>
"""
import os, sys, json, argparse, random, statistics
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def _harvest_correct(tok, mdl, srcs, K, model):
    """generate K per src, grade, return list of (src, code) for verified-correct kernels."""
    gl = W.generate_batch(tok, mdl, srcs, K, max_new=900, adapter=False, bs=4)
    pairs = []
    for src, gens in zip(srcs, gl):
        codes = [c for c in (AB.extract_modelnew(t) for t in gens) if c]
        if not codes: continue
        for code, g in zip(codes, W._grade_isolated(src, codes)):
            if bool((list(g) + [0, 0, 0])[0]): pairs.append((src, code))
    return pairs


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    g = lambda s: tuple(int(x) for x in str(s).split(",") if x != "")
    tok, student = W.build(args.student, g(args.s_gpu)); student.eval()
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    wall = names[:args.wall]; held = names[args.wall:args.wall + args.held]
    wsrc = [LK.TASKS[n] for n in wall]
    print("WALL-CAUSAL student=%s teacher=%s wall=%d held=%d K=%d" % (args.student, args.teacher, len(wall), len(held), args.K), flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    C0, _, _, _, _ = W.eval_tasks(tok, student, held, args.K, adapter=False)
    # verify-budget sweep: does best-of-E correct rate on wall tasks rise with search?
    sweep = {}
    for E in [e for e in (4, 16, 64) if e <= args.Emax]:
        gl = W.generate_batch(tok, student, wsrc, E, max_new=900, adapter=False, bs=4)
        ok = 0
        for src, gens in zip(wsrc, gl):
            codes = [c for c in (AB.extract_modelnew(t) for t in gens) if c]
            if codes and any(bool((list(x) + [0,0,0])[0]) for x in W._grade_isolated(src, codes)): ok += 1
        sweep[E] = round(ok / max(len(wsrc), 1), 3)
        print("verify-budget E=%d wall-solve=%.3f" % (E, sweep[E]), flush=True)
    # SELF harvest (student's own correct kernels on wall tasks)
    self_pairs = _harvest_correct(tok, student, wsrc, args.K, args.student)
    print("SELF harvested %d correct pairs" % len(self_pairs), flush=True)
    # EXTERNAL harvest (teacher's correct kernels on the SAME wall tasks)
    tok_t, teacher = W.build(args.teacher, g(args.t_gpu)); teacher.eval()
    ext_pairs = _harvest_correct(tok_t, teacher, wsrc, args.K, args.teacher)
    print("EXT (teacher) harvested %d correct pairs" % len(ext_pairs), flush=True)
    del teacher; torch.cuda.empty_cache()
    # match SFT token budget: cap both arms to the same #pairs
    n = min(len(self_pairs), len(ext_pairs)) if (self_pairs and ext_pairs) else max(len(self_pairs), len(ext_pairs))
    init = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(student).items()}
    def sft_eval(pairs):
        set_peft_model_state_dict(student, {k: v.to(W._dev(student)) for k, v in init.items()})
        if pairs:
            try: W.sft(tok, student, pairs, args.sft_steps)
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache()
        C, _, _, _, _ = W.eval_tasks(tok, student, held, args.K, adapter=True)
        return round(C, 3)
    C_self = sft_eval(self_pairs[:n] if n else self_pairs)
    C_ext = sft_eval(ext_pairs[:n] if n else ext_pairs)
    res = {"student": args.student, "teacher": args.teacher, "C0_held": round(C0, 3),
           "n_self": len(self_pairs), "n_ext": len(ext_pairs), "n_matched": n,
           "C_held_after_self_sft": C_self, "C_held_after_ext_sft": C_ext,
           "ext_minus_self": round(C_ext - C_self, 3), "verify_budget_sweep": sweep,
           "wall_is_coverage_gap": bool(C_ext > C0 + 0.05 and C_self <= C0 + 0.02),
           "wall_moves_with_search": bool(len(sweep) >= 2 and max(sweep.values()) - min(sweep.values()) > 0.1)}
    json.dump(res, open(os.path.join(args.outdir, "wall_causal.json"), "w"), indent=2)
    print("RESULT C0=%.3f C_self=%.3f C_ext=%.3f (ext-self=%+.3f) | coverage_gap=%s search_moves_wall=%s" %
          (C0, C_self, C_ext, C_ext - C_self, res["wall_is_coverage_gap"], res["wall_moves_with_search"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--s-gpu", default="0"); ap.add_argument("--t-gpu", default="1,2")
    ap.add_argument("--wall", type=int, default=16); ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--Emax", type=int, default=64)
    ap.add_argument("--sft-steps", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "wall_causal"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
