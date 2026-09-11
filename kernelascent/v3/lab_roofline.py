"""Absolute grounding for KernelAscent speedups (roadmap #5).

"1.5x vs torch.compile" is relative. This module grounds every task against the HARDWARE roofline: given the
task's FLOPs (counted) and bytes moved (from tensor sizes), the achievable time is max(FLOPs/peak_FLOPS,
bytes/bandwidth). We then report what fraction of that ceiling the fp32 reference already reaches, so a kernel's
score can be expressed as % of theoretical peak, not just relative to a baseline. Optionally compares against an
expert reference kernel per task if one is registered.

Reports per task: flops, bytes, arithmetic_intensity, bound (compute|memory), roofline_ms, ref_ms,
ref_pct_of_peak. A candidate with measured speedup S over the reference reaches ref_pct_of_peak * S of peak.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_kernel as LK
import torch

# A100 peak (SXM/40GB). Override via env for H100 etc.
PEAK = {
    "fp16": float(os.environ.get("KA_PEAK_FP16_TFLOPS", "312")) * 1e12,   # tensor-core
    "bf16": float(os.environ.get("KA_PEAK_BF16_TFLOPS", "312")) * 1e12,
    "fp32": float(os.environ.get("KA_PEAK_FP32_TFLOPS", "19.5")) * 1e12,
    "tf32": float(os.environ.get("KA_PEAK_TF32_TFLOPS", "156")) * 1e12,
}
BW = float(os.environ.get("KA_PEAK_HBM_GBPS", "1555")) * 1e9              # bytes/s (A100-40GB ~1.55 TB/s)


def _bytes_of(tensors):
    seen = set(); tot = 0
    for t in tensors:
        if torch.is_tensor(t) and id(t) not in seen:
            seen.add(id(t)); tot += t.numel() * t.element_size()
    return tot


def roofline_task(name, src):
    """Return roofline analysis for one task, or {'error':...}. Uses the fp32-consistent reference module."""
    try:
        ref, x, gold, rerr = AB.build_ref(src)
    except Exception as e:
        return {"name": name, "error": "build_ref: %r" % repr(e)[:80]}
    dt = x.dtype if torch.is_tensor(x) else torch.float16
    peak_key = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(dt, "fp16")
    # count FLOPs of one forward
    flops = 0
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            _ = ref(x)
        flops = fc.get_total_flops()
    except Exception:
        flops = 0
    # bytes moved: inputs read + output written + params read (lower bound on traffic)
    params = [p for p in getattr(ref, "parameters", lambda: [])()] if hasattr(ref, "parameters") else []
    out = None
    try:
        out = ref(x)
    except Exception:
        pass
    bts = _bytes_of([x] + list(params) + ([out] if out is not None else []))
    ref_ms = AB.time_fn(lambda z: ref(z), (x,)) * 1e3
    t_compute = flops / PEAK[peak_key] if flops else 0.0
    t_mem = bts / BW if bts else 0.0
    roof_s = max(t_compute, t_mem)
    roof_ms = roof_s * 1e3
    bound = "compute" if t_compute >= t_mem else "memory"
    ai = (flops / bts) if bts else 0.0
    ref_pct = round(100.0 * roof_ms / ref_ms, 2) if ref_ms > 0 and roof_ms > 0 else None
    return {"name": name, "dtype": str(dt).replace("torch.", ""), "peak_used": peak_key,
            "flops": int(flops), "bytes": int(bts), "arithmetic_intensity": round(ai, 2),
            "bound": bound, "roofline_ms": round(roof_ms, 4), "ref_ms": round(ref_ms, 4),
            "ref_pct_of_peak": ref_pct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.environ.get("KA_KERNEL_BANK", ""))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "roofline"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.outdir, exist_ok=True)
    tasks = LK.TASKS
    rows = []
    for name, src in tasks.items():
        r = roofline_task(name, src)
        rows.append(r)
        print(r.get("name"), r.get("bound"), "ref_pct_of_peak=", r.get("ref_pct_of_peak"), r.get("error", ""), flush=True)
        json.dump(rows, open(os.path.join(args.outdir, "roofline.json"), "w"), indent=2)
    ok = [r for r in rows if r.get("ref_pct_of_peak") is not None]
    if ok:
        import statistics
        print("DONE %d tasks | median ref%%-of-peak=%.1f | compute-bound=%d memory-bound=%d" %
              (len(ok), statistics.median([r["ref_pct_of_peak"] for r in ok]),
               sum(r["bound"] == "compute" for r in ok), sum(r["bound"] == "memory" for r in ok)), flush=True)


if __name__ == "__main__":
    main()
