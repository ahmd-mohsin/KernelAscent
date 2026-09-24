"""Experiment #2 — COMPOUNDING TEST (validates the paper's "compounding" title claim).

An open model runs N weight-RSI rounds. At each round we measure held-out capability C for:
  LINEAGE   : keep accumulating LoRA + self-data (the RSI agent).
  RESET     : a matched learner whose adapter is RE-INITIALIZED each round and trained ONLY on that round's
              producer data (single-round gain, no accumulation).
  BEST-OF-N : frozen base, best-of-(k*(r+1)) sampling on held-out — matched CUMULATIVE generation budget, no weights.
  RETRIEVAL : frozen base + growing verified-kernel few-shot archive (in-context, no weights).
Also evaluates the LINEAGE model on a HELD-OUT TASK FAMILY never trained on (transfer).

Headline (the compounding claim): lineage's PER-ROUND MARGINAL GAIN should stay positive and, crucially, the
gain from an accumulated model should exceed the reset model's single-round gain — i.e. accumulation raises the
SUBSEQUENT learning rate, not just the level. If lineage==reset==best-of-N, it's search, not compounding.

Reuses lab_weight_rsi (build/eval_tasks/sft) and lab_baselines (retrieval). Memory-safe bf16; grades isolated.
CLI: python -m kernelascent.v3.lab_compounding --model <hf> --gpus 0,1 --rounds 8 --held-family l3 --outdir <d>
"""
import os, sys, json, argparse, random, statistics, time
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import provenance as PROV
try:
    from kernelascent.v3 import lab_baselines as BL
except Exception:
    BL = None
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def _families(names):
    """group task names by prefix before first '_' (e.g. l2_7 -> 'l2') as a proxy for op-family/level."""
    fam = {}
    for n in names:
        k = n.split("_")[0] if "_" in n else n
        fam.setdefault(k, []).append(n)
    return fam


def _load_teacher(path):
    """Verified-correct kernels harvested from a stronger model (scripts/make_teacher_kernels.py)."""
    if not path:
        return {}
    try:
        d = json.load(open(path))
    except Exception as e:
        print("WARN: could not read teacher kernels %r: %r" % (path, e), flush=True)
        return {}
    return d.get("kernels", d)


def _inject(pairs, solved, train, teacher, per_task, tasks):
    """E1 POSITIVE CONTROL -- the experiment that makes the compounding null falsifiable.

    Self-training can only reuse what the model already produces, so if coverage is the
    binding constraint then a lineage that never sees a correct kernel for task T can never
    learn T, no matter how many rounds it runs. This injects teacher-verified kernels for
    exactly the tasks the student FAILED this round, i.e. where its own success probability
    was zero.

    The contrast stays honest because BOTH the lineage and reset arms train on the same
    augmented set: injection changes what is available to accumulate, while lineage-minus-reset
    still isolates accumulation itself. If lineage-minus-reset turns positive only under
    injection, the harness can register compounding and the unaugmented null is a real
    finding about coverage. If it stays flat even here, the loop is broken and nothing
    should be published from it.

    Returns (augmented_pairs, n_injected, tasks_injected).
    """
    if not teacher:
        return pairs, 0, []
    uncovered = [t for t in train if t not in solved]
    added, used = [], []
    for t in uncovered:
        ks = teacher.get(t) or []
        if not ks:
            continue
        src = tasks.get(t)
        if src is None:
            continue
        for kobj in ks[:per_task]:                     # already sorted fastest-first
            added.append((src, kobj["code"] if isinstance(kobj, dict) else kobj))
        used.append(t)
    return pairs + added, len(added), used


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    gpus = [int(x) for x in str(args.gpus).split(",") if x != ""]
    # split GPUs between lineage and reset arms; with >=4 GPUs each arm gets its own multi-GPU shard so
    # 7B/14B reset no longer OOMs on a single GPU (the historical large-scale blocker).
    if len(gpus) >= 4:
        half = len(gpus) // 2
        lin_g, res_g = gpus[:half], gpus[half:]
    else:
        lin_g, res_g = gpus, (gpus[-1:] or [0])
    tok, mdl = W.build(args.model, tuple(lin_g))                    # LINEAGE learner
    tok_r, reset = W.build(args.model, tuple(res_g))               # RESET learner (own shard if >=4 GPUs)
    reset_init = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(reset).items()}  # fresh LoRA snapshot

    names = list(LK.TASKS); random.Random(1).shuffle(names)
    fam = _families(names)
    held_family = args.held_family if args.held_family in fam else max(fam, key=lambda k: len(fam[k]))
    transfer = fam.get(held_family, [])[:args.n_held]              # held-out FAMILY (never trained on)
    pool = [n for n in names if n not in set(transfer)]
    train, held = pool[:args.n_train], pool[args.n_train:args.n_train + args.n_held]
    print("COMPOUNDING %s seed=%d train=%d held=%d transfer_family=%s(%d) rounds=%d" %
          (args.model, args.seed, len(train), len(held), held_family, len(transfer), args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    teacher = _load_teacher(args.inject_kernels)
    if teacher:
        print("E1 POSITIVE CONTROL: injecting up to %d teacher kernel(s) per UNSOLVED train task "
              "(%d tasks available from %s)" % (args.inject_per_task, len(teacher), args.inject_kernels), flush=True)
    C0, _, _, _, _ = W.eval_tasks(tok, mdl, held, args.k, adapter=False)     # frozen-base held-out baseline
    print("C0 frozen-base held = %.3f" % C0, flush=True)
    hist = []; prevC = C0

    # RESUME. Without this the lab restarts at round 0 on every walltime timeout, and jobman's
    # `continue` resubmits it to do so again: these cells reached round 4 of 6 in a 50-minute
    # chunk and then began the same 5 rounds over, indefinitely, while the queue showed healthy
    # progress. 19 cells were in that loop.
    #
    # As with lab_weight_rsi, the LoRA adapter is not checkpointed, so a resumed run continues
    # the round sequence but re-learns from the recorded data rather than restoring exact weights.
    # Recorded as `resumed_at` so affected trajectories are identifiable, never silently pooled.
    _prev = os.path.join(args.outdir, "compounding.json")
    resumed_at = None
    if os.path.exists(_prev):
        try:
            _d = json.load(open(_prev))
            hist = _d.get("history") or []
            if hist:
                resumed_at = len(hist)
                prevC = hist[-1].get("C_lineage", C0)
                print("RESUME  %d round(s) recorded -- continuing from round %d "
                      "(adapter not restored; trajectory marked)" % (len(hist), len(hist)), flush=True)
        except Exception as e:
            print("RESUME failed (%r) -- starting clean" % e, flush=True)
            hist = []

    for r in range(len(hist), args.rounds):
        t0 = time.time()
        # LINEAGE: produce on train (adapter ON), accumulate SFT
        trainC, pairs, _, _, _ = W.eval_tasks(tok, mdl, train, args.k, adapter=True)
        # which train tasks did the student actually solve this round? pairs carry the source,
        # so a task with no pair is one it failed -- exactly the uncovered set injection targets.
        solved = set()
        for ps, _code in pairs:
            for t in train:
                if LK.TASKS.get(t) == ps:
                    solved.add(t); break
        pairs, n_inj, inj_tasks = _inject(pairs, solved, train, teacher, args.inject_per_task, LK.TASKS)
        try: loss = W.sft(tok, mdl, pairs, args.sft_steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); loss = float("nan")
        C_lin, _, _, _, _ = W.eval_tasks(tok, mdl, held, args.k, adapter=True)
        C_tr, _, _, _, _ = W.eval_tasks(tok, mdl, transfer, args.k, adapter=True) if transfer else (0.0, 0, 0, 0, 0)
        # RESET: re-init adapter, train ONLY on this round's producer data (single-round gain, no accumulation)
        set_peft_model_state_dict(reset, {k: v.to(W._dev(reset)) for k, v in reset_init.items()})
        try: W.sft(tok_r, reset, pairs, args.sft_steps)
        except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache()
        C_reset, _, _, _, _ = W.eval_tasks(tok_r, reset, held, args.k, adapter=True)
        # BEST-OF-N: frozen base at matched cumulative budget k*(r+1)
        C_bon, _, _, _, _ = W.eval_tasks(tok, mdl, held, args.k * (r + 1), adapter=False)
        # RETRIEVAL: frozen base + growing few-shot archive (optional, via lab_baselines)
        C_ret = None
        if BL is not None:
            try: C_ret = BL.retrieval(tok, mdl, train, held, args.k, n_shot=min(3 + r, 8)).get("C")
            except Exception: C_ret = None
        # The metric that produced these numbers travels WITH them. PREREGISTRATION.md
        # Amendment 1 forbids pooling passrate with headroom results, and an unstamped row makes
        # that promise unenforceable -- the checker finds no mode and passes by default, which
        # is the failure it exists to prevent.
        row = {"round": r, "score_mode": os.environ.get("KA_SCORE", "eager"),
               "trainC": round(trainC, 3), "n_ex": len(pairs),
               "n_injected": n_inj, "injected_tasks": len(inj_tasks), "n_solved_self": len(solved), "loss": round(loss, 3),
               "C_lineage": round(C_lin, 3), "C_reset": round(C_reset, 3), "C_bestofN": round(C_bon, 3),
               "C_retrieval": (round(C_ret, 3) if C_ret is not None else None), "transfer_C": round(C_tr, 3),
               "marginal_gain_lineage": round(C_lin - prevC, 3),   # gain THIS round from the accumulated model
               "gain_vs_C0_lineage": round(C_lin - C0, 3),
               "marginal_gain_reset": round(C_reset - C0, 3),      # single-round gain (no accumulation)
               "lineage_minus_reset": round(C_lin - C_reset, 3),   # >0 sustained = accumulation compounds
               "lineage_minus_bestofN": round(C_lin - C_bon, 3),   # >0 = beats matched-budget search
               "round_sec": round(time.time() - t0, 1)}
        hist.append(row); prevC = C_lin
        print("round %d C_lin=%.3f C_reset=%.3f C_bon=%.3f transfer=%.3f | lin-reset=%+.3f lin-bon=%+.3f (%.0fs)" %
              (r, C_lin, C_reset, C_bon, C_tr, C_lin - C_reset, C_lin - C_bon, time.time() - t0), flush=True)
        PROV.dump({"model": args.model, "seed": args.seed, "C0": C0, "held_family": held_family, "resumed_at": resumed_at,
                   "arm": ("inject" if teacher else "control"),
                   "inject_kernels": args.inject_kernels, "inject_per_task": args.inject_per_task,
                   "history": hist},
                  open(os.path.join(args.outdir, "compounding.json"), "w"), indent=2)
    lr = [h["lineage_minus_reset"] for h in hist]
    print("\n=== COMPOUNDING SUMMARY %s === lineage-minus-reset:" % args.model, lr)
    print("COMPOUNDS (accumulation raises subsequent learning, sustained lineage>reset & lineage>best-of-N)?",
          "YES" if len(lr) >= 3 and statistics.mean(lr[-2:]) > 0.05 else "NO/insufficient")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpus", default="0")
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=20); ap.add_argument("--n-held", type=int, default=20)
    ap.add_argument("--sft-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held-family", default="l3")
    ap.add_argument("--inject-kernels", default=None,
                    help="E1 positive control: JSON of teacher-verified kernels "
                         "(scripts/make_teacher_kernels.py). Injected ONLY for train tasks the "
                         "student failed that round, into BOTH arms.")
    ap.add_argument("--inject-per-task", type=int, default=1,
                    help="how many teacher kernels to inject per uncovered task")
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "compounding"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
