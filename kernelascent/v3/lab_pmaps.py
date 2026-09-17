"""Experiment #2 — PER-TASK p-MAPS + SHARPENING LAW (panel-requested; Astra 1a/1b, Fable 1a/1b).

For a model checkpoint (frozen base, or a lineage/reset LoRA adapter), estimate for EVERY task t:
  p_K(t)      = fraction of K sampled kernels that are verified-correct  (per-sample success prob)
  passk(t,k)  = unbiased pass@k for k=1..K (Chen et al. estimator) from the same K samples
  best_roof(t)= best roofline fraction (compiled speedup / ceiling) among correct candidates
  n_ok(t), K
Then the harness-level artifacts the panels want, all from grading (no training):
  * coverage set  C = {t : p_K(t) > 0}     -> coverage-set growth across checkpoints
  * mass-shift    histogram of p_K(t)       -> where does self-training move probability mass
  * pass@1 vs pass@K trajectory              -> sharpening (pass@1 up) vs frontier (pass@K flat/down)
  * sharpening-law inputs  (p0, K, dp)       -> fit dp(p0,K) across scales/optimizers

Run one checkpoint at a time; point --adapter-dir at a saved LoRA to map a lineage round (else frozen base).
Compute-cheap: only generation+grading. Use large K (e.g. 128) and it dominates; grading is isolated.
CLI: python -m kernelascent.v3.lab_pmaps --model <hf> --gpus 0 --K 128 --tag base_q15 --outdir <d>
"""
import os, json, argparse, random, math, time
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch


def _passk(n, c, k):
    """Unbiased pass@k = 1 - C(n-c,k)/C(n,k) for c correct out of n samples (Chen et al. 2021)."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _grade_all(srcs, per_task_codes):
    """Grade every candidate; return per-task (n_ok, best_compiled_speedup, best_ceiling)."""
    grades = W._grade_isolated_batch(list(zip(srcs, per_task_codes)))
    out = []
    for codes, res in zip(per_task_codes, grades):
        n_ok = 0; best_c = 0.0; ceil = 1.5
        for g in res:
            ok, se, sc, cl = (list(g) + [0.0, 0.0, 0.0, 1.5])[:4]
            if ok:
                n_ok += 1
                if sc > best_c:
                    best_c = sc
                ceil = cl
        out.append((n_ok, best_c, ceil))
    return out


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    gpus = [int(x) for x in str(args.gpus).split(",") if x != ""]
    tok, mdl = W.build(args.model, tuple(gpus))
    use_adapter = False
    if args.adapter_dir and os.path.isdir(args.adapter_dir):
        try:
            from peft import PeftModel  # noqa
            mdl.load_adapter(args.adapter_dir, adapter_name="lineage"); mdl.set_adapter("lineage")
            use_adapter = True
            print("loaded adapter", args.adapter_dir, flush=True)
        except Exception as e:
            print("adapter load failed (%s); using frozen base" % e, flush=True)

    names = list(LK.TASKS)
    if args.max_tasks:
        random.Random(1).shuffle(names); names = names[:args.max_tasks]
    srcs = [LK.TASKS[n] for n in names]
    print("PMAPS %s adapter=%s tasks=%d K=%d" % (args.model, use_adapter, len(names), args.K), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    # generate K candidates/task in bs chunks (memory-safe), accumulate valid extracted kernels
    per_task_codes = [[] for _ in names]
    done = 0
    while done < args.K:
        step = min(args.gen_chunk, args.K - done)
        gl = W.generate_batch(tok, mdl, srcs, step, adapter=use_adapter)
        for i, cand in enumerate(gl):
            for t in cand:
                c = AB.extract_modelnew(t)
                if c:
                    per_task_codes[i].append(c)
        done += step
        print("  generated %d/%d" % (done, args.K), flush=True)

    graded = _grade_all(srcs, per_task_codes)
    rows = []
    for i, (n, (n_ok, best_c, ceil)) in enumerate(zip(names, graded)):
        n_samp = len(per_task_codes[i])
        p = (n_ok / n_samp) if n_samp else 0.0
        roof = (best_c / ceil) if ceil else 0.0
        passk = {str(k): round(_passk(n_samp, n_ok, k), 4) for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= n_samp}
        rows.append({"task": n, "tier": LK.TIER.get(n, "?"), "n_samples": n_samp, "n_ok": n_ok,
                     "p": round(p, 4), "best_roofline_frac": round(roof, 4), "passk": passk})

    cov = [r["task"] for r in rows if r["p"] > 0]
    # p-mass histogram over fixed bins
    bins = [0.0, 1e-9, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
    hist = [0] * (len(bins) - 1)
    for r in rows:
        for b in range(len(bins) - 1):
            if bins[b] <= r["p"] < bins[b + 1]:
                hist[b] += 1; break
    summary = {"model": args.model, "adapter_dir": (args.adapter_dir if use_adapter else None),
               "tag": args.tag, "K": args.K, "n_tasks": len(rows),
               "coverage_size": len(cov), "coverage_frac": round(len(cov) / len(rows), 3) if rows else 0.0,
               "mean_p": round(sum(r["p"] for r in rows) / len(rows), 4) if rows else 0.0,
               "mean_pass1": round(sum(r["passk"].get("1", 0.0) for r in rows) / len(rows), 4) if rows else 0.0,
               "mean_passK": round(sum(max(r["passk"].values()) for r in rows) / len(rows), 4) if rows else 0.0,
               "p_hist_bins": bins, "p_hist_counts": hist, "coverage_tasks": cov, "rows": rows}
    out = os.path.join(args.outdir, "pmap_%s.json" % args.tag)
    json.dump(summary, open(out, "w"), indent=2)
    print("\n=== PMAP %s === coverage=%d/%d (%.1f%%) mean_p=%.3f mean_pass1=%.3f mean_pass@K=%.3f\nwrote %s" %
          (args.tag, summary["coverage_size"], summary["n_tasks"], 100 * summary["coverage_frac"],
           summary["mean_p"], summary["mean_pass1"], summary["mean_passK"], out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpus", default="0")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--gen-chunk", type=int, default=16)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--adapter-dir", default="")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "pmaps"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
