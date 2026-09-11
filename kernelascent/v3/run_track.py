"""One-command track dispatcher for KernelAscent (roadmap #3, reproducibility).

  python -m kernelascent.v3.run_track --track capability --model <hf-or-api> --bank bank_mid.json --gpus 0
  python -m kernelascent.v3.run_track --track rsi        --model <hf-id>     --bank bank_small.json --gpus 0,1 --rounds 5
  python -m kernelascent.v3.run_track --track baselines  --model <hf-id>     --bank bank_small.json --gpus 0
  python -m kernelascent.v3.run_track --track procedure  --model <api-id>    --bank bank_mid.json --grade-gpu 0
  python -m kernelascent.v3.run_track --track combined   --trainee <hf-id> --researcher <api-id> --trainee-gpu 0 --ctrl-gpu 1

Sets the standard env (headroom score, size-matched bank) and forwards to the appropriate lab's main(), so a
submission reproduces a scorecard with a single command. GPU tracks need CUDA; procedure/combined need Bedrock creds.
"""
import os, sys, argparse, runpy


def _run(module, argv):
    os.environ.setdefault("KA_SCORE", "compiled")          # headroom-normalized speed score
    sys.argv = [module] + argv
    runpy.run_module(module, run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["capability", "rsi", "procedure", "combined", "baselines"])
    ap.add_argument("--model", default=""); ap.add_argument("--trainee", default=""); ap.add_argument("--researcher", default="")
    ap.add_argument("--bank", default="bank_mid.json")
    ap.add_argument("--gpus", default="0"); ap.add_argument("--grade-gpu", default="0")
    ap.add_argument("--trainee-gpu", default="0"); ap.add_argument("--ctrl-gpu", default="1")
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a, extra = ap.parse_known_args()
    dd = os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data")
    os.environ["KA_KERNEL_BANK"] = a.bank if os.path.isabs(a.bank) else os.path.join(dd, "kernel_bank", a.bank)
    os.environ["KA_GRADE_GPU"] = a.grade_gpu

    if a.track == "rsi":
        _run("kernelascent.v3.lab_weight_rsi",
             ["--model", a.model, "--gpus" if False else "--self-gpu", a.gpus, "--fresh-gpu", a.gpus,
              "--ctrl-gpu", "", "--rounds", str(a.rounds), "--k", str(a.k), "--seed", str(a.seed)] + extra)
    elif a.track == "baselines":
        _run("kernelascent.v3.lab_baselines",
             ["--model", a.model, "--gpus", a.gpus, "--rounds", str(a.rounds), "--k", str(a.k), "--seed", str(a.seed)] + extra)
    elif a.track == "procedure":
        _run("kernelascent.v3.lab_track_c",
             ["--model", a.model, "--rounds", str(a.rounds), "--k", str(a.k), "--grade-gpu", a.grade_gpu, "--seed", str(a.seed)] + extra)
    elif a.track == "combined":
        _run("kernelascent.v3.lab_combined_rsi",
             ["--trainee", a.trainee, "--researcher", a.researcher, "--researcher-gpu", "",
              "--trainee-gpu", a.trainee_gpu, "--ctrl-gpu", a.ctrl_gpu, "--rounds", str(a.rounds), "--k", str(a.k), "--seed", str(a.seed)] + extra)
    elif a.track == "capability":
        # capability = frozen best-of-k on the bank (weight-RSI eval with 0 training rounds via baselines best_of_k)
        _run("kernelascent.v3.lab_baselines",
             ["--model", a.model, "--gpus", a.gpus, "--methods", "best_of_k", "--rounds", str(a.rounds), "--k", str(a.k), "--seed", str(a.seed)] + extra)


if __name__ == "__main__":
    main()
