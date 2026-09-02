"""Procedural KernelBench-style source-task generator for KernelAscent.

Each task is emitted as runnable Python source defining a `Model(nn.Module)` (with a
seeded, dtype-parametric __init__ and a forward that is the fused op-graph), plus
module-level M, D, DT and get_inputs(). An agent is shown this source and must return
an optimized `ModelNew` with the SAME __init__ and equivalent forward. Because tasks
are seed-generated, a held-out seed range gives a private, un-memorizable split.
"""
import math, random

HEADER = """import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""

def _mm(i, curD, rng):
    N = rng.choice([512, 1024, 2048, 4096])
    init = ("        self.W%d = nn.Parameter((torch.randn(%d, %d, generator=g) / math.sqrt(%d)).to(dtype), requires_grad=False)"
            % (i, curD, N, curD))
    fwd = "        x = x @ self.W%d" % i
    return init, fwd, N, ["tiling", "tensor-core"], "matmul->%d" % N

def _bias(i, curD, rng):
    init = "        self.b%d = nn.Parameter(torch.randn(%d, generator=g).to(dtype), requires_grad=False)" % (i, curD)
    return init, "        x = x + self.b%d" % i, curD, ["elementwise-fusion"], "bias"

def _scale(i, curD, rng):
    s = round(1.0 + 0.5 * rng.random(), 4)
    return None, "        x = x * %s" % s, curD, ["elementwise-fusion"], "scale"

def _relu(i, curD, rng):
    return None, "        x = torch.relu(x)", curD, ["elementwise-fusion"], "relu"

def _gelu(i, curD, rng):
    return None, "        x = F.gelu(x)", curD, ["elementwise-fusion"], "gelu"

def _softmax(i, curD, rng):
    return None, "        x = torch.softmax(x, dim=-1)", curD, ["reduction", "online-softmax"], "softmax"

def _layernorm(i, curD, rng):
    init = ("        self.ln%d_g = nn.Parameter(torch.randn(%d, generator=g).to(dtype), requires_grad=False)\n"
            "        self.ln%d_b = nn.Parameter(torch.randn(%d, generator=g).to(dtype), requires_grad=False)"
            % (i, curD, i, curD))
    fwd = "        x = F.layer_norm(x, (x.shape[-1],), self.ln%d_g, self.ln%d_b)" % (i, i)
    return init, fwd, curD, ["reduction", "fusion"], "layernorm"

def _rmsnorm(i, curD, rng):
    init = "        self.rms%d_w = nn.Parameter(torch.randn(%d, generator=g).to(dtype), requires_grad=False)" % (i, curD)
    fwd = ("        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms%d_w" % i)
    return init, fwd, curD, ["reduction"], "rmsnorm"

POINTWISE = [_bias, _scale, _relu, _gelu]
REDUCTIONS = [_softmax, _layernorm, _rmsnorm]


def _tier(tags):
    if "tiling" in tags:
        return "L2"
    if "reduction" in tags:
        return "L1"
    return "L0"


def gen_task(seed):
    rng = random.Random(seed)
    M = rng.choice([512, 1024, 2048, 4096, 8192])
    D = rng.choice([512, 1024, 2048, 4096])
    if rng.random() < 0.3:
        D += 1
    dtype = rng.choice(["torch.float16", "torch.bfloat16"])
    L = rng.randint(2, 5)

    inits, fwds, labels, tags = [], [], [], set()
    curD, i, n_mm = D, 0, 0
    def add(fn):
        nonlocal curD, i, n_mm
        init, fwd, curD, t, lab = fn(i, curD, rng)
        if init:
            inits.append(init)
        fwds.append(fwd); labels.append(lab); tags.update(t)
        if lab.startswith("matmul"):
            n_mm += 1
        i += 1
    if rng.random() < 0.5:
        add(_mm)
    for _ in range(L):
        pool = list(POINTWISE + REDUCTIONS)
        if n_mm < 2 and rng.random() < 0.25:
            pool.append(_mm)
        add(rng.choice(pool))
    if "reduction" not in tags:
        add(_softmax)

    init_body = "\n".join(inits) if inits else "        pass"
    fwd_body = "\n".join(fwds)
    src = HEADER + (
        "\nSEED = %d\nM, D, DT = %d, %d, %s\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, dtype=DT):\n"
        "        super().__init__()\n"
        "        g = torch.Generator().manual_seed(SEED)\n"
        "%s\n\n"
        "    def forward(self, x):\n"
        "%s\n"
        "        return x\n\n"
        "def get_inputs():\n"
        "    g = torch.Generator().manual_seed(SEED + 12345)\n"
        "    return [torch.randn(M, D, generator=g).to(DT)]\n"
        % (seed, M, D, dtype, init_body, fwd_body)
    )
    return dict(name="t%d_%dx%d" % (seed, M, D), tier=_tier(tags),
                family=("matmul" if "tiling" in tags else "norm-act"),
                tags=sorted(tags), meta={"M": M, "D": D, "dtype": dtype, "chain": labels},
                source=src)


def generate(n, seed0=0):
    return [gen_task(seed0 + i) for i in range(n)]


if __name__ == "__main__":
    print(gen_task(0)["source"])
