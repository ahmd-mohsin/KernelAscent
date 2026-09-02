"""Box-side grader -> complete, reward-hack-resistant benchmark bundles.

Robustness: correctness is checked on MULTIPLE fresh random inputs (not just the one
seeded input), against an fp32 gold, with tolerance no tighter than the working dtype's
own rounding error. This defeats the main hacks — hardcoding the output for the fixed
input, constant/no-op outputs, or seed-specific tricks — because a real kernel must be
correct on every input while a hack fails the extra draws.

Per task writes results.json + reference_solution.py + difficulty in meta.json. Speedup
is roofline-relative (min of eager, torch.compile). Supports --nshards for multi-GPU.
"""
import os, glob, json, argparse, shutil, torch
import agent_bench as A

N_INPUTS = 4  # correctness must hold on all of these


def build_models(task_src):
    ns = {}
    exec(task_src, ns)
    DT = ns["DT"]
    ref = ns["Model"](DT).cuda().eval()
    gold = ns["Model"](torch.float32).cuda().eval()
    x0 = ns["get_inputs"]()[0].cuda()
    return ref, gold, DT, x0


def draw_inputs(x0, n):
    xs = [x0]
    for s in range(1, n):
        g = torch.Generator(device=x0.device).manual_seed(9973 * s + 17)
        xs.append(torch.randn(*x0.shape, generator=g, device=x0.device, dtype=x0.dtype))
    return xs


def golds_and_bounds(ref, gold, xs, tol, margin):
    gs, bs = [], []
    with torch.no_grad():
        for x in xs:
            g = gold(x.float())
            r = ref(x)
            gs.append(g); bs.append(max(tol, margin * A.rel_l2(r, g)))
    return gs, bs


def grade_cand(task_src, code, xs, golds, bounds):
    """Correct only if within bound on ALL inputs and output actually depends on input."""
    try:
        mod = A.load_module(task_src + "\n" + code)
        MN = mod.ModelNew; DT = mod.DT
        try:
            cand = MN(DT).cuda().eval()
        except TypeError:
            cand = MN().cuda().eval()
        errs, outs = [], []
        with torch.no_grad():
            for x, g in zip(xs, golds):
                o = cand(x)
                if o.shape != g.shape or not torch.isfinite(o).all().item():
                    return dict(ok=False, err=None, t_cand=None, reason="bad_shape_or_nonfinite", n_pass=0)
                errs.append(A.rel_l2(o, g)); outs.append(o)
    except Exception as e:
        return dict(ok=False, err=None, t_cand=None, reason=repr(e)[:100], n_pass=0)
    n_pass = sum(1 for e, b in zip(errs, bounds) if e <= b)
    # input-sensitivity: outputs for two different inputs must differ (catches constant/no-op)
    insensitive = len(outs) >= 2 and A.rel_l2(outs[0], outs[1]) < 1e-6
    ok = (n_pass == len(xs)) and not insensitive
    if not ok:
        reason = "input_insensitive" if insensitive else "wrong_on_%d/%d" % (len(xs) - n_pass, len(xs))
        return dict(ok=False, err=max(errs), t_cand=None, reason=reason, n_pass=n_pass)
    t_cand = A.time_fn(lambda z: cand(z), (xs[0],))
    return dict(ok=True, err=max(errs), t_cand=t_cand, reason="ok", n_pass=n_pass)


def difficulty(best_sp, any_ok, n_cand):
    if n_cand == 0:
        return "no_candidates"
    if not any_ok:
        return "frontier"
    if best_sp <= 1.0:
        return "hard"
    if best_sp <= 1.5:
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
            ref, gold, DT, x0 = build_models(task_src)
            xs = draw_inputs(x0, N_INPUTS)
            golds, bounds = golds_and_bounds(ref, gold, xs, args.tol, args.margin)
        except Exception as e:
            print("SKIP %s build: %s" % (name, repr(e)[:50])); continue
        t_eager = A.time_fn(lambda z: ref(z), (x0,))
        try:
            cf = torch.compile(ref)
            with torch.no_grad():
                _ = cf(x0)
            t_compile = A.time_fn(lambda z: cf(z), (x0,))
        except Exception:
            t_compile = None
        t_base = min([t for t in [t_eager, t_compile] if t is not None])

        cands = []
        for cf_path in sorted(glob.glob(d + "/cand_*.py")):
            r = grade_cand(task_src, open(cf_path).read(), xs, golds, bounds)
            cands.append(dict(file=os.path.basename(cf_path), ok=r["ok"], err=r["err"],
                              reason=r["reason"], n_pass=r["n_pass"], n_inputs=N_INPUTS,
                              t_cand=r["t_cand"],
                              speedup_vs_eager=(t_eager / r["t_cand"]) if r["t_cand"] else 0.0,
                              speedup_vs_roofline=(t_base / r["t_cand"]) if r["t_cand"] else 0.0))
        ok_cands = [c for c in cands if c["ok"]]
        best = max(ok_cands, key=lambda c: c["speedup_vs_roofline"], default=None)
        best_sp = best["speedup_vs_roofline"] if best else 0.0
        # anomaly flag: passed but implausibly fast (candidate for manual hack review)
        suspicious = [c["file"] for c in ok_cands if c["speedup_vs_roofline"] > 5.0]
        diff = difficulty(best_sp, bool(ok_cands), len(cands))

        results = dict(name=name, tier=meta.get("tier"), family=meta.get("family"),
                       tags=meta.get("tags", []), meta=meta.get("meta", {}),
                       n_inputs=N_INPUTS, t_eager=t_eager, t_compile=t_compile, t_roofline=t_base,
                       n_cand=len(cands), pass_at_k=len(ok_cands), best=best,
                       best_speedup_roofline=best_sp, difficulty=diff,
                       suspicious=suspicious, candidates=cands)
        json.dump(results, open(os.path.join(d, "results.json"), "w"), indent=2)
        if best:
            shutil.copyfile(os.path.join(d, best["file"]), os.path.join(d, "reference_solution.py"))
        meta.update(achievable_speedup=best_sp, pass_rate=len(ok_cands) / max(len(cands), 1),
                    difficulty=diff, suspicious=bool(suspicious))
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
        summary.append(dict(name=name, family=meta.get("family"), tier=meta.get("tier"),
                            difficulty=diff, pass_at_k=len(ok_cands), n_cand=len(cands),
                            best_speedup_roofline=best_sp, suspicious=bool(suspicious)))
        print("%-26s %-14s %-3s %-12s pass=%d/%d sp=%.3f%s" %
              (name, meta.get("family"), meta.get("tier"), diff, len(ok_cands), len(cands), best_sp,
               "  SUSPICIOUS" if suspicious else ""), flush=True)

    json.dump({"tasks": summary}, open(args.out, "w"), indent=2)
    LABELS = ("no_candidates", "frontier", "hard", "medium", "accessible")
    print("== by difficulty ==")
    for L in LABELS:
        print("  %-13s %d" % (L, sum(1 for s in summary if s["difficulty"] == L)))
    print("  suspicious   %d" % sum(1 for s in summary if s["suspicious"]))
    print("wrote %s (%d tasks)" % (args.out, len(summary)))


if __name__ == "__main__":
    main()
