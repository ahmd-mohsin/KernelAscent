"""Procedural task generator for KernelAscent.

Samples a fused op-graph (matmul + pointwise + reduction chain), randomizes shapes
(including non-power-of-2), dtype, and length, then composes the PyTorch reference
automatically. Weights are stored in fp32 and cast to the working dtype at apply time,
so the identical graph can be run in fp16/bf16 (the graded reference) or fp32 (the
gold used to define achievable precision). A seed range gives a private held-out split.
"""
import math, random
import torch
import torch.nn.functional as F
from task_schema import Task

SHAPES_M = [512, 1024, 2048, 4096, 8192]
SHAPES_D = [512, 1024, 2048, 4096]
DTYPES = [torch.float16, torch.bfloat16]


def op_matmul(D, g, rng):
    N = rng.choice([512, 1024, 2048, 4096])
    W = torch.randn(D, N, generator=g, device="cuda", dtype=torch.float32) / math.sqrt(D)
    return (lambda x: x @ W.to(x.dtype)), N, ["tiling", "tensor-core"], "matmul->%d" % N

def op_bias(D, g, rng):
    b = torch.randn(D, generator=g, device="cuda", dtype=torch.float32)
    return (lambda x: x + b.to(x.dtype)), D, ["elementwise-fusion"], "bias"

def op_scale(D, g, rng):
    s = 1.0 + 0.5 * rng.random()
    return (lambda x: x * s), D, ["elementwise-fusion"], "scale"

def op_relu(D, g, rng):
    return (lambda x: torch.relu(x)), D, ["elementwise-fusion"], "relu"

def op_gelu(D, g, rng):
    return (lambda x: F.gelu(x)), D, ["elementwise-fusion"], "gelu"

def op_softmax(D, g, rng):
    return (lambda x: torch.softmax(x, dim=-1)), D, ["reduction", "online-softmax"], "softmax"

def op_layernorm(D, g, rng):
    gamma = torch.randn(D, generator=g, device="cuda", dtype=torch.float32)
    beta = torch.randn(D, generator=g, device="cuda", dtype=torch.float32)
    return (lambda x: F.layer_norm(x, (x.shape[-1],), gamma.to(x.dtype), beta.to(x.dtype))), \
        D, ["reduction", "fusion"], "layernorm"

def op_rmsnorm(D, g, rng):
    w = torch.randn(D, generator=g, device="cuda", dtype=torch.float32)
    def fn(x):
        xf = x.float()
        r = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return r.to(x.dtype) * w.to(x.dtype)
    return fn, D, ["reduction"], "rmsnorm"

POINTWISE = [op_bias, op_scale, op_relu, op_gelu]
REDUCTIONS = [op_softmax, op_layernorm, op_rmsnorm]


def _tier(tags):
    if "tiling" in tags:
        return "L2"
    if "reduction" in tags:
        return "L1"
    return "L0"


def gen_task(seed):
    rng = random.Random(seed)
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    M = rng.choice(SHAPES_M)
    D = rng.choice(SHAPES_D)
    if rng.random() < 0.3:
        D += 1                      # non-power-of-2 width
    dtype = rng.choice(DTYPES)
    L = rng.randint(2, 5)

    fns, labels, tags = [], [], set()
    curD, n_mm = D, 0
    if rng.random() < 0.5:
        fn, curD, t, lab = op_matmul(curD, g, rng); n_mm += 1
        fns.append(fn); labels.append(lab); tags.update(t)
    for _ in range(L):
        pool = list(POINTWISE + REDUCTIONS)
        if n_mm < 2 and rng.random() < 0.25:
            pool.append(op_matmul)
        op = rng.choice(pool)
        fn, curD, t, lab = op(curD, g, rng)
        if lab.startswith("matmul"):
            n_mm += 1
        fns.append(fn); labels.append(lab); tags.update(t)
    if "reduction" not in tags:
        fn, curD, t, lab = op_softmax(curD, g, rng)
        fns.append(fn); labels.append(lab); tags.update(t)

    def ref(x):
        for fn in fns:
            x = fn(x)
        return x

    def make_inputs():
        gi = torch.Generator(device="cuda"); gi.manual_seed(seed + 12345)
        return (torch.randn(M, D, generator=gi, device="cuda", dtype=torch.float32).to(dtype),)

    dtname = "fp16" if dtype == torch.float16 else "bf16"
    return Task(
        name="t%d_%dx%d_%s" % (seed, M, D, dtname),
        tier=_tier(tags),
        family=("matmul" if "tiling" in tags else "norm-act"),
        tags=sorted(tags),
        ref=ref,
        make_inputs=make_inputs,
        meta={"M": M, "D": D, "dtype": dtname, "chain": labels},
    )


def generate(n, seed0=0):
    return [gen_task(seed0 + i) for i in range(n)]
