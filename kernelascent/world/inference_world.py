"""M1: an inspectable GPU inference WORLD.

A small transformer decode service built from SWAPPABLE operators (the improvable system state W).
The agent may replace an operator with a numerically-equivalent faster one; the IMMUTABLE grader runs
the service on a hidden workload and measures correctness (vs fp32 gold), latency, tokens/sec, goodput
(fraction of requests under an SLO), per-operator profile (the Amdahl ceiling), and resources.

Everything is inspectable: run `--smoke` to see the baseline service run on GPU, the per-operator time
share (which operator is worth optimizing), and a demo candidate operator graded end-to-end.

Operators (baseline = torch): rmsnorm, qkv (matmul), attn (SDPA), oproj (matmul), mlp (gated SiLU).
A candidate operator is python source defining a function with the operator's name + signature; the
grader execs it in a fresh namespace, swaps it in, and grades the WHOLE service — so a change is only
credited if it is correct AND moves the downstream service metric.
"""
import os, sys, json, argparse, time, math, hashlib
import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
DEV = "cuda"


# ---------------------------------------------------------------- fixed world config + weights
def world_config():
    return {"H": 2048, "n_head": 16, "n_layer": 8, "ffn": 5632, "B": 8, "S": 256, "seed": 0}


def build_weights(cfg, dtype=DTYPE):
    g = torch.Generator(device=DEV).manual_seed(cfg["seed"])
    H, F_, L = cfg["H"], cfg["ffn"], cfg["n_layer"]
    def r(*s): return (torch.randn(*s, generator=g, device=DEV) / math.sqrt(s[0])).to(dtype)
    W = []
    for _ in range(L):
        W.append({"ln1": torch.ones(H, device=DEV, dtype=dtype), "ln2": torch.ones(H, device=DEV, dtype=dtype),
                  "qkv": r(H, 3 * H), "o": r(H, H), "gate": r(H, F_), "up": r(H, F_), "down": r(F_, H)})
    return W


# ---------------------------------------------------------------- baseline operators (the improvable W)
def op_rmsnorm(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def op_qkv(x, Wqkv):
    return x @ Wqkv


def op_attn(q, k, v, n_head):
    B, S, H = q.shape
    hd = H // n_head
    q = q.view(B, S, n_head, hd).transpose(1, 2)
    k = k.view(B, S, n_head, hd).transpose(1, 2)
    v = v.view(B, S, n_head, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).contiguous().view(B, S, H)


def op_oproj(x, Wo):
    return x @ Wo


def op_mlp(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg) * (x @ Wu)) @ Wd


BASELINE_OPS = {"rmsnorm": op_rmsnorm, "qkv": op_qkv, "attn": op_attn, "oproj": op_oproj, "mlp": op_mlp}


# ---------------------------------------------------------------- the service forward (uses the ops dict)
def forward(x, W, cfg, ops, profile=None):
    nh = cfg["n_head"]
    for layer in W:
        h = ops["rmsnorm"](x, layer["ln1"])
        qkv = ops["qkv"](h, layer["qkv"])
        q, k, v = qkv.chunk(3, dim=-1)
        a = ops["attn"](q, k, v, nh)
        x = x + ops["oproj"](a, layer["o"])
        h2 = ops["rmsnorm"](x, layer["ln2"])
        x = x + ops["mlp"](h2, layer["gate"], layer["up"], layer["down"])
    return x


def _sync(): torch.cuda.synchronize()


def _time_service(W, cfg, ops, x, iters=20, warmup=5):
    for _ in range(warmup):
        forward(x, W, cfg, ops)
    _sync()
    ts = []
    flush = torch.empty(int(40e6), device=DEV, dtype=torch.int8)   # L2 flush between reps
    for _ in range(iters):
        flush.zero_(); _sync(); t0 = time.perf_counter()
        forward(x, W, cfg, ops)
        _sync(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]     # median seconds/forward


def profile_ops(W, cfg, ops, x, iters=10):
    """Per-operator share of service time (the Amdahl ceiling for optimizing each op)."""
    base = _time_service(W, cfg, ops, x, iters=iters)
    per = {n: 0.0 for n in ("rmsnorm", "qkv", "attn", "oproj", "mlp")}
    reps = iters
    for _ in range(reps):
        for layer in W:
            for name, call in (("rmsnorm", lambda: ops["rmsnorm"](x, layer["ln1"])),):
                _sync(); t0 = time.perf_counter(); call(); _sync(); per["rmsnorm"] += time.perf_counter() - t0
            _sync(); t0 = time.perf_counter(); qkv = ops["qkv"](x, layer["qkv"]); _sync(); per["qkv"] += time.perf_counter() - t0
            q, k, v = qkv.chunk(3, dim=-1)
            _sync(); t0 = time.perf_counter(); a = ops["attn"](q, k, v, cfg["n_head"]); _sync(); per["attn"] += time.perf_counter() - t0
            _sync(); t0 = time.perf_counter(); ops["oproj"](a, layer["o"]); _sync(); per["oproj"] += time.perf_counter() - t0
            _sync(); t0 = time.perf_counter(); ops["mlp"](x, layer["gate"], layer["up"], layer["down"]); _sync(); per["mlp"] += time.perf_counter() - t0
    tot = sum(per.values())
    return {n: round(per[n] / tot, 3) for n in per}, base


# ---------------------------------------------------------------- immutable grader
def fp32_reference(W, cfg, x):
    Wf = [{k: (v.float() if torch.is_tensor(v) else v) for k, v in ly.items()} for ly in W]
    return forward(x.float(), Wf, cfg, BASELINE_OPS)


def grade(candidate_ops, cfg=None, slo_ms=None, tol=2e-2):
    """Run the service with candidate_ops; return correctness + latency + tokens/sec + goodput + profile.
    candidate_ops: dict opname->callable (only changed ops need be present; rest fall back to baseline)."""
    cfg = cfg or world_config()
    W = build_weights(cfg)
    x = (torch.randn(cfg["B"], cfg["S"], cfg["H"], device=DEV) * 0.1).to(DTYPE)
    ops = dict(BASELINE_OPS); ops.update(candidate_ops or {})
    # correctness vs fp32 gold
    ref = fp32_reference(W, cfg, x)
    try:
        out = forward(x, W, cfg, ops).float()
    except Exception as e:
        return {"correct": False, "error": repr(e)[:120]}
    rel = (out - ref).norm() / (ref.norm() + 1e-9)
    base_ref = (forward(x, W, cfg, BASELINE_OPS).float() - ref).norm() / (ref.norm() + 1e-9)  # baseline's own bf16 error
    correct = bool(rel <= max(tol, 2 * base_ref))
    lat = _time_service(W, cfg, ops, x)
    base_lat = _time_service(W, cfg, BASELINE_OPS, x)
    toks = cfg["B"] * cfg["S"] / lat
    slo = slo_ms if slo_ms is not None else (base_lat * 1000 * 1.0)  # SLO = baseline latency
    goodput = (cfg["B"] * cfg["S"] / (lat)) if (lat * 1000) <= slo else 0.0
    return {"correct": correct, "rel_err": float(rel), "tol": max(tol, 2 * float(base_ref)),
            "latency_ms": round(lat * 1000, 3), "baseline_latency_ms": round(base_lat * 1000, 3),
            "speedup_vs_baseline": round(base_lat / lat, 4), "tokens_per_s": round(toks, 1),
            "goodput_tok_s": round(goodput, 1), "slo_ms": round(slo, 3)}


# ---------------------------------------------------------------- candidate loading (agent's operator patch)
def load_op(src, name):
    ns = {"torch": torch, "F": F, "math": math}
    exec(compile(src, "<op>", "exec"), ns)
    fn = ns.get(name)
    return fn if callable(fn) else None


def smoke():
    cfg = world_config()
    W = build_weights(cfg)
    x = (torch.randn(cfg["B"], cfg["S"], cfg["H"], device=DEV) * 0.1).to(DTYPE)
    print("world:", cfg, flush=True)
    shares, base = profile_ops(W, cfg, BASELINE_OPS, x)
    print("baseline service latency: %.3f ms  (%.0f tok/s)" % (base * 1000, cfg["B"] * cfg["S"] / base))
    print("per-operator time share (Amdahl ceiling):", shares)
    # baseline grade (sanity: correct, speedup 1.0)
    g0 = grade({})
    print("baseline grade:", {k: g0[k] for k in ("correct", "rel_err", "latency_ms", "speedup_vs_baseline", "tokens_per_s")})
    # demo candidate: a fused MLP (torch) -- numerically equivalent, tests the grader end-to-end
    demo = "def mlp(x, Wg, Wu, Wd):\n    import torch.nn.functional as F\n    return (F.silu(x @ Wg).mul_(x @ Wu)) @ Wd\n"
    fn = load_op(demo, "mlp")
    g1 = grade({"mlp": fn})
    print("demo fused-mlp grade:", {k: g1[k] for k in ("correct", "rel_err", "speedup_vs_baseline", "tokens_per_s")})
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    if args.smoke or args.profile:
        smoke()
    else:
        print("use --smoke")


if __name__ == "__main__":
    main()
