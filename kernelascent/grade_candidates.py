"""Box-side grader: turn each curated task into a COMPLETE benchmark bundle.

Per task it writes:
  results.json          per-candidate correctness/time/speedup, best, roofline
  reference_solution.py  the best correct+fastest kernel (the achievable target)
  meta.json (updated)    + achievable_speedup, pass_rate, difficulty label

Speedup is roofline-relative: t_base / t_cand where t_base = min(eager, torch.compile).
Difficulty is calibrated from the strong curator: if even it cannot beat the roofline
the task is 'hard'/'frontier'; large speedups mean 'accessible'. Supports --nshards.
"""
import os, glob, json, argparse, shutil, torch
import agent_bench as A


def grade_cand(task_src, code, x, gold, bound):
    try:
        mod = A.load_module(task_src + "\n" + code)
        MN = mod.ModelNew; DT = mod.DT
        try:
            cand = MN(DT).cuda().eval()
        except TypeError:
            cand = MN().cuda().eval()
        with torch.no_grad():
            out = cand(x)
        err = A.rel_l2(out, gold)
        ok = (out.shape == gold.shape) and (err <= bound) and torch.isfinite(out).all().item()
    except Exception as e:
        return dict(ok=False, err=None, t_cand=None, reason=repr(e)[:100])
    if not ok:
        return dict(ok=False, err=err, t_cand=None, reason="wrong_or_imprecise")
    t_cand = A.time_fn(lambda z: cand(z), (x,))
    return dict(ok=True, err=err, t_cand=t_cand, reason="ok")


def difficulty(best_sp_roofline, any_ok, n_cand):
    if n_cand == 0:
        return "no_candidates"     # curator produced no parseable solution (generation gap)
    if not any_ok:
        return "frontier"          # candidates generated, but none correct: genuinely hard
    if best_sp_roofline <= 1.0:
        return "hard"              # correct but cannot beat the strong automated baseline
    if best_sp_roofline <= 1.5:
        return "medium"
    return "accessible"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    tdirs = sorted(d for d in glob.glob(args.candir + "/*") if os.path.exists(os.path.join(d, "task.py")))
    tdirs = [d for i, d in enumerate(tdirs) if i % args.nshards == args.shard]
    summary = []
    for d in tdirs:
        name = os.path.basename(d)
        meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {"name": name}
        task_src = open(os.path.join(d, "task.py")).read()
        try:
            ref, x, gold, ref_err = A.build_ref(task_src)
        except Exception as e:
            print("SKIP %s ref-build %s" % (name, repr(e)[:50])); continue
        bound = max(args.tol, args.margin * ref_err)
        t_eager = A.time_fn(lambda z: ref(z), (x,))
        try:
            cf = torch.compile(ref)
            with torch.no_grad():
                _ = cf(x)
            t_compile = A.time_fn(lambda z: cf(z), (x,))
        except Exception:
            t_compile = None
        t_base = min([t for t in [t_eager, t_compile] if t is not None])

        cands = []
        cand_files = sorted(glob.glob(d + "/cand_*.py"))
        for cf_path in cand_files:
            r = grade_cand(task_src, open(cf_path).read(), x, gold, bound)
            rec = dict(file=os.path.basename(cf_path), ok=r["ok"], err=r["err"], reason=r["reason"],
                       t_cand=r["t_cand"],
                       speedup_vs_eager=(t_eager / r["t_cand"]) if r["t_cand"] else 0.0,
                       speedup_vs_roofline=(t_base / r["t_cand"]) if r["t_cand"] else 0.0)
            cands.append(rec)
        ok_cands = [c for c in cands if c["ok"]]
        best = max(ok_cands, key=lambda c: c["speedup_vs_roofline"], default=None)
        best_sp = best["speedup_vs_roofline"] if best else 0.0
        diff = difficulty(best_sp, bool(ok_cands), len(cands))

        results = dict(name=name, tier=meta.get("tier"), family=meta.get("family"),
                       tags=meta.get("tags", []), meta=meta.get("meta", {}),
                       ref_err=ref_err, bound=bound, t_eager=t_eager, t_compile=t_compile,
                       t_roofline=t_base, n_cand=len(cands), pass_at_k=len(ok_cands),
                       best=best, best_speedup_roofline=best_sp, difficulty=diff, candidates=cands)
        json.dump(results, open(os.path.join(d, "results.json"), "w"), indent=2)
        if best:
            shutil.copyfile(os.path.join(d, best["file"]),
                            os.path.join(d, "reference_solution.py"))
        meta.update(achievable_speedup=best_sp, pass_rate=len(ok_cands) / max(len(cands), 1),
                    difficulty=diff)
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
        summary.append(dict(name=name, family=meta.get("family"), tier=meta.get("tier"),
                            difficulty=diff, pass_at_k=len(ok_cands), n_cand=len(cands),
                            best_speedup_roofline=best_sp))
        print("%-28s %-14s %-3s %-10s pass=%d/%d best_sp=%.3f" %
              (name, meta.get("family"), meta.get("tier"), diff, len(ok_cands), len(cands), best_sp), flush=True)

    json.dump({"tasks": summary}, open(args.out, "w"), indent=2)
    # rollups
    LABELS = ("no_candidates", "frontier", "hard", "medium", "accessible")
    def roll(key):
        b = {}
        for s in summary:
            b.setdefault(s[key], []).append(s)
        for k in sorted(b, key=str):
            ss = b[k]
            print("  %-12s n=%3d  " % (k, len(ss)) +
                  " ".join("%s=%d" % (x, sum(1 for s in ss if s["difficulty"] == x)) for x in LABELS))
    print("== by family =="); roll("family")
    print("== by difficulty ==")
    for dlab in LABELS:
        print("  %-13s %d" % (dlab, sum(1 for s in summary if s["difficulty"] == dlab)))
    print("wrote %s (%d tasks)" % (args.out, len(summary)))


if __name__ == "__main__":
    main()
