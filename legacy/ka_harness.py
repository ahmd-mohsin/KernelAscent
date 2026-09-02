"""KernelAscent locked eval harness (pilot).

Grades a candidate GPU kernel against a PyTorch reference: correctness at a fixed
numerical tolerance, then measured speedup on the target GPU. Timing uses warmup,
median-of-N, L2 cache flush and fresh syncs. Any correctness failure scores 0.

Candidate file must expose: run(*inputs) -> tensor.
"""
import argparse, importlib.util, time, statistics, math, torch

def load_candidate(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def flush_l2():
    x = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
    x.zero_()
    del x

def time_fn(fn, inputs, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush_l2()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

def reference_softmax(x):
    return torch.softmax(x, dim=-1)

# includes non-power-of-2 columns and a huge-row adversarial case a naive kernel fails
SHAPES = [(4096, 4096), (8192, 2048), (2048, 8193), (16384, 1024), (1, 131072)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args()
    cand = load_candidate(args.candidate)
    dev = torch.cuda.get_device_name(0)
    clk = torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else "n/a"
    print("device=%s" % dev)
    print("%16s %9s %9s %8s %10s %5s" % ("shape", "ref_ms", "cand_ms", "speedup", "maxerr", "ok"))
    speedups = []
    passed = 0
    for shape in SHAPES:
        inp = (torch.randn(*shape, device="cuda", dtype=torch.float16),)
        ref = reference_softmax(*inp)
        try:
            out = cand.run(*inp)
            err = (out.float() - ref.float()).abs().max().item()
            ok = torch.allclose(out.float(), ref.float(), atol=args.atol, rtol=args.rtol)
        except Exception as e:
            err = float("inf"); ok = False; out = None
            print("%16s  candidate raised: %s" % (str(shape), repr(e)[:80]))
        if ok:
            t_ref = time_fn(reference_softmax, inp)
            t_cand = time_fn(cand.run, inp)
            sp = t_ref / t_cand
        else:
            t_ref = time_fn(reference_softmax, inp); t_cand = float("nan"); sp = 0.0
        speedups.append(sp); passed += int(ok)
        print("%16s %9.3f %9.3f %8.3f %10.2e %5s" %
              (str(shape), t_ref * 1e3, t_cand * 1e3, sp, err, str(ok)))
    gm = math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))
    fast1 = sum(1 for s in speedups if s > 1.0) / len(speedups)
    print("tasks=%d passed=%d fast_1=%.2f geomean_speedup=%.3f" %
          (len(SHAPES), passed, fast1, gm))

if __name__ == "__main__":
    main()
