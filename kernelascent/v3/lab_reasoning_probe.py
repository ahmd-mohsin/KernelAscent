"""Reasoning / failure-mechanism probe for Tasks 1-3 — WHERE and WHY models fail, with full chains saved.

For a model and a task bank it generates k full completions per task (the model's reasoning + kernel), grades each
in isolation, and assigns every attempt to an exact FAILURE STAGE:

  no_modelnew   the model never emitted a parseable `class ModelNew`               (can't even format)
  compile_fail  emitted code but the reference/candidate build or run threw         (writes non-runnable code)
  incorrect     compiled+ran but output != fp32 gold                               (wrong algorithm/numerics)
  correct_slow  correct but speedup <= 1.0 over eager                              (no real optimization)
  correct_fast  correct AND faster than eager                                     (genuine win)

It then mines each raw chain for optimization-strategy mentions (shared memory, tiling, fusion, vectorization,
coalescing, warp/register, unroll, recompute, ...) and correlates MENTION vs SUCCESS — exposing the
"strategy-application gap": models that name an optimization but cannot implement it. All raw chains are saved to
reasoning_chains.jsonl for deep/qualitative analysis; aggregate stats to reasoning_probe.json.

These are the measurements behind the theorems in docs/THEOREMS.md (correctness wall, strategy-application gap,
diversity floor). Task 1 = base model (adapter off). Task 2/3 = pass a trained adapter / procedure to see how the
failure histogram SHIFTS with self-improvement (which stage the model climbs out of).
"""
import os, sys, json, argparse, statistics, time, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

STRATS = {  # optimization concept -> regex over the raw chain (lowercased)
    "shared_memory": r"shared[ _]?mem|__shared__|smem",
    "tiling": r"\btil(e|ing)\b|block[ _]?tile",
    "fusion": r"\bfus(e|ed|ion)\b|epilogue|fused",
    "vectorization": r"vectori|float4|float2|\bsimd\b|packed",
    "coalescing": r"coalesc|memory[ _]?access[ _]?pattern|stride",
    "warp": r"\bwarp\b|shuffle|__shfl|warp[ _]?level",
    "register": r"\bregister\b|register[ _]?tiling|per[- ]?thread",
    "unroll": r"unroll|#pragma unroll",
    "recompute": r"recompute|online softmax|running (max|sum)|numerically stable",
    "custom_kernel": r"triton|@triton|cuda kernel|extern \"C\"|load_inline|torch.compile",
}


def _stage(code, grade):
    if not code:
        return "no_modelnew"
    ok, se, sc, ceil = (list(grade) + [0, 0, 0, 1.5])[:4]
    if not ok:
        # distinguish compile/run failure from numeric-incorrect: build the candidate alone
        try:
            AB.build_ref(code); return "incorrect"          # built but graded wrong
        except Exception:
            return "compile_fail"
    return "correct_fast" if (se or 0) > 1.0 else "correct_slow"


def _strats(text):
    t = (text or "").lower()
    return {k: bool(re.search(rx, t)) for k, rx in STRATS.items()}


def run(args):
    gpus = [int(x) for x in str(args.gpu).split(",")]
    tok, mdl = W.build(args.model, gpus)
    adapter = (args.adapter == "on")
    if args.adapter_path and adapter:
        try:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(mdl, torch.load(args.adapter_path, map_location=next(mdl.parameters()).device))
            print("loaded adapter", args.adapter_path, flush=True)
        except Exception as e:
            print("adapter load failed:", e, flush=True)
    names = list(LK.TASKS); random_names = names[:args.n]
    srcs = [LK.TASKS[n] for n in random_names]
    os.makedirs(args.outdir, exist_ok=True)
    chainf = os.path.join(args.outdir, "reasoning_chains.jsonl")
    print("REASONING-PROBE %s adapter=%s n=%d k=%d" % (args.model, args.adapter, len(srcs), args.k), flush=True)
    gl = W.generate_batch(tok, mdl, srcs, args.k, max_new=1100, adapter=adapter)
    codes = [[c for c in (AB.extract_modelnew(t) for t in g) if c] for g in gl]
    # grade one code per generation slot (align raw text <-> extracted code <-> grade)
    stage_hist = {}; strat_total = {k: [0, 0] for k in STRATS}   # [mentions, mention&correct]
    per_task = []
    for name, src, gens in zip(random_names, srcs, gl):
        # extract per-generation (keep None where unparseable) so stage aligns with the raw chain
        gcodes = [AB.extract_modelnew(t) for t in gens]
        parseable = [(i, c) for i, c in enumerate(gcodes) if c]
        grades = W._grade_isolated_batch([(src, [c for _, c in parseable])]) if parseable else [[]]
        gmap = {}
        if parseable:
            for (i, _), gr in zip(parseable, grades[0]):
                gmap[i] = gr
        best = "no_modelnew"; order = ["no_modelnew", "compile_fail", "incorrect", "correct_slow", "correct_fast"]
        for i, (raw, code) in enumerate(zip(gens, gcodes)):
            st = _stage(code, gmap.get(i, []))
            stage_hist[st] = stage_hist.get(st, 0) + 1
            ok = st.startswith("correct")
            sm = _strats(raw)
            for k, present in sm.items():
                if present:
                    strat_total[k][0] += 1
                    if ok: strat_total[k][1] += 1
            if order.index(st) > order.index(best): best = st
            with open(chainf, "a") as fh:                       # persist the full chain for deep analysis
                fh.write(json.dumps({"task": name, "attempt": i, "stage": st, "adapter": args.adapter,
                                     "strategies": [k for k, v in sm.items() if v],
                                     "chain": raw[:6000]}) + "\n")
        per_task.append({"task": name, "best_stage": best})
    # strategy-application gap: P(correct | mentioned) per strategy
    strat_gap = {k: {"mentions": m, "success_rate_when_mentioned": round(c / m, 3) if m else None}
                 for k, (m, c) in strat_total.items()}
    summary = {"model": args.model, "adapter": args.adapter, "n_tasks": len(srcs), "k": args.k,
               "stage_histogram": stage_hist, "strategy_application_gap": strat_gap,
               "correct_rate": round(sum(1 for p in per_task if p["best_stage"].startswith("correct")) / max(1, len(per_task)), 3),
               "per_task_best_stage": per_task}
    json.dump(summary, open(os.path.join(args.outdir, "reasoning_probe.json"), "w"), indent=2)
    print("STAGE HISTOGRAM:", stage_hist, flush=True)
    print("STRATEGY GAP:", {k: v["success_rate_when_mentioned"] for k, v in strat_gap.items() if v["mentions"]}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--adapter", choices=["on", "off"], default="off")   # off = Task-1 base capability
    ap.add_argument("--adapter-path", default="")                        # a trained LoRA (Task 2) to see the failure-stage shift
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "reasoning"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
