"""Experiment #1 — COVERAGE TRANSPLANT 2x2 (panel top pick; Astra 1c, Fable 1c).

Tests whether restricted COVERAGE is the CAUSAL mediator of the compounding null (vs. just associational).
The mechanism claim: self-training SHARPENS tasks the model already covers but cannot EXPAND coverage;
so injecting verified teacher kernels on UNCOVERED tasks should restore compounding, while injecting the
same token budget on already-COVERED tasks (pure sharpening) should not.

Protocol (single scale, e.g. 1.5B student; teacher e.g. 7B/14B):
  1. Estimate student p0(t) over K0 samples -> covered set Cov={t:p0>0}, uncovered U={t:p0=0} (train pool only).
  2. Teacher harvests verified kernels per task -> pool of (task,kernel) pairs, split by student coverage.
  3. Four arms, each starts from the SAME fresh LoRA, matched INJECTED-PAIR COUNT (approx token-match):
       L        : lineage self-harvest only
       L+T_in   : + teacher kernels on COVERED tasks   (sharpening only)
       L+T_out  : + teacher kernels on UNCOVERED tasks  (coverage expansion)
       L+T_rand : + random teacher kernels
     Round 0 = SFT on (self-harvest [+ injected]); rounds 1..R-1 = SELF-ONLY harvest+SFT (does injected coverage
     persist and get sharpened?). Each round record held-out C and coverage-set size on held tasks.
  Prediction: T_out > T_rand > T_in ~ L, and injected coverage persists across self-rounds.
  Optional dose-response: --dose 0.1,0.25,0.5,1.0 scales the injected fraction for the T_out arm.

Reuses lab_weight_rsi build/eval_tasks/sft. Sequential arms on the same GPUs (LoRA reset between arms).
CLI: python -m kernelascent.v3.lab_transplant --student <hf> --teacher <hf> --gpus 0,1 --rounds 4 --outdir <d>
"""
import os, json, argparse, random, time
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def _coverage(tok, mdl, names, k, adapter=False):
    """per-task correct (p>0) via a SINGLE batched eval_tasks call (per_task_correct aligns with names order).
    Batching is critical: per-task calls spawned hundreds of isolated graders and were impractically slow."""
    if not names:
        return [], [], []
    _, allpairs, _, _, st = W.eval_tasks(tok, mdl, names, k, adapter=adapter)
    corr = st.get("per_task_correct", [0] * len(names))
    covered = [n for n, c in zip(names, corr) if c > 0]
    uncovered = [n for n, c in zip(names, corr) if c <= 0]
    return covered, uncovered, allpairs


def _harvest_teacher(tok_t, tea, names, k):
    """teacher generates on tasks; return dict task-> list of verified (src,code) pairs."""
    bytask = {}
    for n in names:
        _, ex, _, _, _ = W.eval_tasks(tok_t, tea, [n], k, adapter=False)
        if ex:
            bytask[n] = ex
    return bytask


def _sample_pairs(bytask, task_subset, n):
    """flatten teacher pairs restricted to task_subset, shuffle, take n."""
    pool = [p for t in task_subset for p in bytask.get(t, [])]
    random.shuffle(pool)
    return pool[:n]


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    gpus = [int(x) for x in str(args.gpus).split(",") if x != ""]
    tok, mdl = W.build(args.student, tuple(gpus))
    lora0 = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(mdl).items()}

    names = list(LK.TASKS); random.Random(1).shuffle(names)
    train = names[:args.n_train]; held = names[args.n_train:args.n_train + args.n_held]
    print("TRANSPLANT student=%s teacher=%s train=%d held=%d K0=%d rounds=%d" %
          (args.student, args.teacher, len(train), len(held), args.k0, args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    # 1. student coverage on TRAIN pool (frozen base)
    cov, unc, _ = _coverage(tok, mdl, train, args.k0, adapter=False)
    print("student coverage(train): covered=%d uncovered=%d" % (len(cov), len(unc)), flush=True)
    C0, _, _, _, cst0 = W.eval_tasks(tok, mdl, held, args.k, adapter=False)
    cov_h0, _, _ = _coverage(tok, mdl, held, args.k0, adapter=False)
    print("held C0=%.3f coverage_held=%d/%d" % (C0, len(cov_h0), len(held)), flush=True)

    # 2. teacher harvest on TRAIN tasks (loaded on same gpus after student coverage done to save memory)
    tok_t, tea = W.build(args.teacher, tuple(gpus))
    tea_by = _harvest_teacher(tok_t, tea, train, args.k_teacher)
    n_tin = sum(len(tea_by.get(t, [])) for t in cov); n_tout = sum(len(tea_by.get(t, [])) for t in unc)
    inj = args.inject if args.inject > 0 else min(n_tin, n_tout) or 8
    print("teacher harvested: covered-pairs=%d uncovered-pairs=%d inject_n=%d" % (n_tin, n_tout, inj), flush=True)
    del tea, tok_t; torch.cuda.empty_cache()

    arms = {"L": None, "L+T_in": ("in", 1.0), "L+T_out": ("out", 1.0), "L+T_rand": ("rand", 1.0)}
    if args.dose:
        for d in [float(x) for x in args.dose.split(",")]:
            arms["L+T_out@%g" % d] = ("out", d)

    results = {}
    for arm, spec in arms.items():
        set_peft_model_state_dict(mdl, {k: v.to(W._dev(mdl)) for k, v in lora0.items()})  # fresh LoRA per arm
        traj = []; prevC = C0
        for r in range(args.rounds):
            t0 = time.time()
            _, self_pairs, _, _, _ = W.eval_tasks(tok, mdl, train, args.k, adapter=(r > 0))
            pairs = list(self_pairs)
            if r == 0 and spec is not None:               # inject teacher pairs ONLY at round 0
                kind, dose = spec; nn = int(inj * dose)
                if kind == "in":   pairs += _sample_pairs(tea_by, cov, nn)
                elif kind == "out": pairs += _sample_pairs(tea_by, unc, nn)
                else:               pairs += _sample_pairs(tea_by, train, nn)
            try: loss = W.sft(tok, mdl, pairs, args.sft_steps)
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); loss = float("nan")
            C, _, _, _, _ = W.eval_tasks(tok, mdl, held, args.k, adapter=True)
            cov_h, _, _ = _coverage(tok, mdl, held, args.k0, adapter=True)
            # DIRECT MEDIATOR: did this arm expand coverage into previously-UNCOVERED train tasks?
            # (held-C is floored when the small bank's held set is all-hard; coverage expansion on the
            #  injection-target domain is the quantity the mechanism actually predicts.)
            new_cov, _, _ = _coverage(tok, mdl, unc, args.k0, adapter=True)
            row = {"round": r, "n_ex": len(pairs), "loss": round(loss, 3), "C": round(C, 3),
                   "marginal_gain": round(C - prevC, 3), "gain_vs_C0": round(C - C0, 3),
                   "coverage_held": len(cov_h), "coverage_uncovered_train": len(new_cov),
                   "uncovered_train_total": len(unc), "sec": round(time.time() - t0, 1)}
            traj.append(row); prevC = C
            print("  [%s] r%d C=%.3f cov_held=%d/%d cov_UNC=%d/%d (%.0fs)" %
                  (arm, r, C, len(cov_h), len(held), len(new_cov), len(unc), time.time() - t0), flush=True)
            results[arm] = traj   # CHECKPOINT AFTER EVERY ROUND: partial arms visible + survive node recycle
            json.dump({"student": args.student, "teacher": args.teacher, "C0": C0, "inject_n": inj,
                       "covered_train": len(cov), "uncovered_train": len(unc), "arms": results},
                      open(os.path.join(args.outdir, "transplant.json"), "w"), indent=2)

    # headline contrast: does coverage injection EXPAND coverage more than sharpening injection?
    def fin(a, key): return results[a][-1][key] if a in results and results[a] else None
    dC = (fin("L+T_out", "gain_vs_C0") - fin("L+T_in", "gain_vs_C0")) \
        if fin("L+T_out", "gain_vs_C0") is not None and fin("L+T_in", "gain_vs_C0") is not None else None
    # coverage-expansion mediator (final-round covered-uncovered-train count) per arm
    cov = {a: fin(a, "coverage_uncovered_train") for a in results}
    dCov = (cov.get("L+T_out", 0) - cov.get("L+T_in", 0)) if cov.get("L+T_out") is not None and cov.get("L+T_in") is not None else None
    print("\n=== TRANSPLANT SUMMARY ===", flush=True)
    print("  final gain_vs_C0(held): L=%s Tin=%s Tout=%s Trand=%s | Tout-Tin=%s" %
          (fin("L", "gain_vs_C0"), fin("L+T_in", "gain_vs_C0"), fin("L+T_out", "gain_vs_C0"),
           fin("L+T_rand", "gain_vs_C0"), (("%+.3f" % dC) if dC is not None else "n/a")), flush=True)
    print("  final coverage_uncovered_train: %s | Tout-Tin=%s" % (cov, dCov), flush=True)
    print("COVERAGE IS CAUSAL MEDIATOR?",
          "YES (Tout expands coverage >> Tin)" if (dCov or 0) >= 2 else "NO/weak", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--k0", type=int, default=24)
    ap.add_argument("--k-teacher", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=24); ap.add_argument("--n-held", type=int, default=20)
    ap.add_argument("--sft-steps", type=int, default=14)
    ap.add_argument("--inject", type=int, default=0)     # 0 = auto = min(covered,uncovered) pair count
    ap.add_argument("--dose", default="")                # e.g. "0.1,0.25,0.5,1.0" -> extra T_out dose arms
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "transplant"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
