"""A1 (GPT-6 Astra's decisive control): recursion-interruption fork.

Trains a SELF producer for `fork_round` rounds of rejection-sampling LoRA-SFT, snapshots its adapter, then forks
three producers and continues ALL at equal budget for the remaining rounds:
  continued_self  : the producer keeps evolving (ongoing recursion)
  ckpt_frozen     : producer FROZEN at the fork checkpoint; a fresh learner trains on its fresh samples
  base_frozen     : producer = untrained base; a fresh learner trains on its fresh samples
Held-out learning curves after the fork answer the reviewer's question: does CONTINUING to evolve the producer
beat freezing an already-improved one? continued_self > ckpt_frozen => genuine compounding, not a one-time upgrade.
continued_self == ckpt_frozen => iterative adaptation only. Output: recursion_fork.json.

Reuses lab_weight_rsi (build/eval_tasks/sft) + lab_kernel TASKS. Small/mid models (bf16) fit 3-4 instances.
"""
import os, sys, json, argparse, random, statistics, time, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
import torch


def _save_adapter(mdl, path):
    from peft import get_peft_model_state_dict
    os.makedirs(path, exist_ok=True)
    torch.save(get_peft_model_state_dict(mdl), os.path.join(path, "adapter.pt"))


def _load_adapter(mdl, path):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(mdl, torch.load(os.path.join(path, "adapter.pt"), map_location=W._dev(mdl)))


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    sg = [int(x) for x in str(args.self_gpu).split(",")]
    pg = [int(x) for x in str(args.prod_gpu).split(",")]      # ckpt-frozen producer
    lg = [int(x) for x in str(args.learn_gpu).split(",")]     # ckpt-frozen learner
    bg = [int(x) for x in str(args.base_gpu).split(",")]      # base-frozen learner
    tok, self_m = W.build(args.model, sg)
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    train, held = names[:args.n_train], names[args.n_train:]
    print("RECURSION-FORK %s seed=%d fork_round=%d rounds=%d train=%d held=%d" %
          (args.model, args.seed, args.fork_round, args.rounds, len(train), len(held)), flush=True)
    C0, _, _, _, _ = W.eval_tasks(tok, self_m, held, args.k, adapter=False)
    hist = []
    ckpt_dir = os.path.join(args.outdir, "ckpt"); os.makedirs(args.outdir, exist_ok=True)
    prod_m = learn_m = base_m = None

    for r in range(args.rounds):
        t0 = time.time()
        row = {"round": r}
        # SELF (continued): produce with its evolving adapter, train on own correct kernels
        _, pairs, _, _, _ = W.eval_tasks(tok, self_m, train, args.k, adapter=True)
        W.sft(tok, self_m, pairs, args.sft_steps)
        Cs, _, _, _, _ = W.eval_tasks(tok, self_m, held, args.k, adapter=True)
        row["C_self"] = round(Cs, 3)

        if r == args.fork_round:                              # snapshot + spin up the two frozen-producer arms
            _save_adapter(self_m, ckpt_dir)
            _, prod_m = W.build(args.model, pg); _load_adapter(prod_m, ckpt_dir)   # FROZEN producer @ checkpoint
            for p in prod_m.parameters(): p.requires_grad_(False)
            _, learn_m = W.build(args.model, lg)              # fresh learner for ckpt-frozen data
            _, base_m = W.build(args.model, bg)               # fresh learner for base-frozen data
            print("  [fork @ round %d] snapshot saved; ckpt-frozen + base-frozen arms started" % r, flush=True)

        if r >= args.fork_round and prod_m is not None:
            # ckpt-frozen: fresh data from the FROZEN checkpoint producer -> train ckpt learner
            _, cpairs, _, _, _ = W.eval_tasks(tok, prod_m, train, args.k, adapter=True)   # producer frozen (adapter fixed)
            W.sft(tok, learn_m, cpairs, args.sft_steps)
            Cc, _, _, _, _ = W.eval_tasks(tok, learn_m, held, args.k, adapter=True)
            # base-frozen: fresh data from base -> train base learner
            _, bpairs, _, _, _ = W.eval_tasks(tok, self_m, train, args.k, adapter=False)  # base producer (adapter off)
            W.sft(tok, base_m, bpairs, args.sft_steps)
            Cb, _, _, _, _ = W.eval_tasks(tok, base_m, held, args.k, adapter=True)
            row["C_ckpt_frozen"] = round(Cc, 3); row["C_base_frozen"] = round(Cb, 3)
            row["self_minus_ckpt"] = round(Cs - Cc, 3); row["self_minus_base"] = round(Cs - Cb, 3)
        hist.append(row)
        print("round %d C_self=%.3f ckpt=%s base=%s | self-ckpt=%s self-base=%s (%.0fs)" %
              (r, Cs, row.get("C_ckpt_frozen", "-"), row.get("C_base_frozen", "-"),
               row.get("self_minus_ckpt", "-"), row.get("self_minus_base", "-"), time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "fork_round": args.fork_round, "C0": C0, "history": hist},
                  open(os.path.join(args.outdir, "recursion_fork.json"), "w"), indent=2)
    smc = [h.get("self_minus_ckpt") for h in hist if h.get("self_minus_ckpt") is not None]
    print("\n=== FORK SUMMARY %s === self-minus-ckpt(post-fork): %s" % (args.model, smc))
    print("COMPOUNDS (continuing beats frozen-checkpoint)?",
          "YES" if len(smc) >= 2 and statistics.mean(smc[-2:]) > 0.03 else "NO (one-time upgrade / iterative only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--self-gpu", default="0"); ap.add_argument("--prod-gpu", default="1")
    ap.add_argument("--learn-gpu", default="2"); ap.add_argument("--base-gpu", default="3")
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--fork-round", type=int, default=4)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--sft-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "recursion_fork"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
