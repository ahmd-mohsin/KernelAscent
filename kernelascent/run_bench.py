"""KernelAscent harness: pull generated tasks, grade a candidate against a roofline baseline.

- Correctness: candidate passes if its relative L2 error to an fp32 GOLD is no worse than
  the reference's own fp16/bf16 rounding error (times a margin). No over-demanding precision.
- Speedup is roofline-relative: t_baseline / t_candidate, where the baseline is a strong
  AUTOMATED reference (torch.compile max-autotune). Procedural tasks cannot have hand-tuned
  experts, so the strong baseline must also be automated. Beating it is the difficulty.
- Primary metric: fast_p. Geomean over passing tasks only. Results persisted to JSON.
"""
import argparse, time, statistics, math, json, torch
import gen_tasks


def flush_l2():
    x = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
    x.zero_(); del x


def time_fn(fn, inputs, iters=30, warmup=10):
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush_l2(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*inputs)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def rel_l2(a, b):
    return (torch.linalg.vector_norm((a - b).float()) /
            (torch.linalg.vector_norm(b.float()) + 1e-12)).item()


def build_eager(task):
    return task.ref

def build_compile(task):
    return torch.compile(task.ref)

def build_compile_max(task):
    return torch.compile(task.ref, mode="max-autotune")

BUILDERS = {"eager": build_eager, "compile": build_compile, "compile_max": build_compile_max}


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def graded(fn, inp, gold, bound):
    """Return (ok, err). ok requires finite output within bound rel-L2 of gold."""
    try:
        out = fn(*inp)
        err = rel_l2(out, gold)
        return (err <= bound and torch.isfinite(out).all().item()), err
    except Exception:
        return False, float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--candidate", choices=list(BUILDERS), default="compile")
    ap.add_argument("--baseline", choices=list(BUILDERS), default="compile_max",
                    help="roofline denominator: speedup = t_baseline / t_candidate")
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="write per-task results JSON here")
    args = ap.parse_args()
    cbuild, bbuild = BUILDERS[args.candidate], BUILDERS[args.baseline]
    tasks = gen_tasks.generate(args.n, args.seed0)

    print("device=%s  candidate=%s  baseline=%s  tasks=%d" %
          (torch.cuda.get_device_name(0), args.candidate, args.baseline, len(tasks)))
    print("%-4s %-9s %10s %8s %8s %8s %8s %9s %5s  %s" %
          ("tier", "family", "MxD", "eager_ms", "base_ms", "cand_ms", "speedup", "cand_err", "ok", "chain"))
    records, rows_sp, passed = [], [], 0
    per_tier = {}
    for task in tasks:
        inp = task.make_inputs()
        try:
            ref_out = task.ref(*inp)
            gold = task.ref(inp[0].float())
        except Exception as e:
            print("%-4s %-9s  ref raised: %s" % (task.tier, task.family, repr(e)[:60]))
            continue
        bound = max(args.tol, args.margin * rel_l2(ref_out, gold))
        cfn, bfn = cbuild(task), bbuild(task)
        c_ok, c_err = graded(cfn, inp, gold, bound)
        b_ok, _ = graded(bfn, inp, gold, bound)
        t_eager = time_fn(task.ref, inp)
        t_base = time_fn(bfn, inp) if b_ok else float("nan")
        t_cand = time_fn(cfn, inp) if c_ok else float("nan")
        sp = (t_base / t_cand) if (c_ok and b_ok) else 0.0
        rows_sp.append(sp); passed += int(c_ok)
        per_tier.setdefault(task.tier, []).append((sp, c_ok))
        records.append(dict(name=task.name, tier=task.tier, family=task.family, tags=task.tags,
                            meta=task.meta, ok=c_ok, cand_err=c_err,
                            t_eager=t_eager, t_base=t_base, t_cand=t_cand, speedup=sp))
        print("%-4s %-9s %10s %8.3f %8.3f %8.3f %8.3f %9.2e %5s  %s" %
              (task.tier, task.family, "%dx%d" % (task.meta["M"], task.meta["D"]),
               t_eager * 1e3, t_base * 1e3, t_cand * 1e3, sp, c_err, str(c_ok),
               "-".join(task.meta["chain"])[:42]))

    n = max(len(rows_sp), 1)
    fp = lambda p: sum(1 for s in rows_sp if s > p) / n
    summary = dict(candidate=args.candidate, baseline=args.baseline, n=len(rows_sp),
                   correct=passed, fast_1=fp(1.0), fast_1_5=fp(1.5), fast_2=fp(2.0),
                   geomean_pass=geomean([s for s in rows_sp if s > 0]))
    print("\nTOTAL tasks=%d correct=%d (%.0f%%)  fast_1=%.2f fast_1.5=%.2f fast_2=%.2f  geomean(pass)=%.3f" %
          (summary["n"], passed, 100.0 * passed / n, summary["fast_1"], summary["fast_1_5"],
           summary["fast_2"], summary["geomean_pass"]))
    for tier in sorted(per_tier):
        ss = [s for s, _ in per_tier[tier]]
        ok_n = sum(1 for _, o in per_tier[tier] if o)
        print("  %s: n=%d correct=%d fast_1=%.2f geomean(pass)=%.3f" %
              (tier, len(ss), ok_n, sum(1 for s in ss if s > 1.0) / len(ss),
               geomean([s for s in ss if s > 0])))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "tasks": records}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
