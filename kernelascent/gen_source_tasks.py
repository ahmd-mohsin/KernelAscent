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


# ---- Additional real-bottleneck families for systematic curation ----

ATTN_HEADER = """import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""

def attn_task(seed, S, D, dtype, causal):
    mask = ("        scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)"
            if causal else "        pass")
    src = ATTN_HEADER + (
        "\nSEED = %d\nS, D, DT = %d, %d, %s\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, dtype=DT):\n"
        "        super().__init__()\n"
        "        g = torch.Generator().manual_seed(SEED)\n"
        "        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)\n"
        "        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)\n"
        "        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)\n\n"
        "    def forward(self, x):\n"
        "        q = x @ self.Wq\n        k = x @ self.Wk\n        v = x @ self.Wv\n"
        "        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])\n"
        "%s\n"
        "        a = torch.softmax(scores, dim=-1)\n"
        "        return a @ v\n\n"
        "def get_inputs():\n"
        "    g = torch.Generator().manual_seed(SEED + 12345)\n"
        "    return [torch.randn(S, D, generator=g).to(DT)]\n"
        % (seed, S, D, dtype, mask)
    )
    tags = ["tiling", "tensor-core", "online-softmax", "reduction", "fusion"]
    if causal:
        tags.append("masking")
    return dict(name="attn%s_s%d_d%d_%d" % ("c" if causal else "", S, D, seed),
                tier="L3", family="attention", tags=sorted(tags),
                meta={"S": S, "D": D, "dtype": ("fp16" if "float16" in dtype else "bf16"),
                      "causal": causal, "chain": ["qkv", "scores", ("causal" if causal else "full"), "softmax", "av"]},
                source=src)


def rope_attn_task(seed, S, D, dtype, causal):
    mask = ("        scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)"
            if causal else "        pass")
    src = ATTN_HEADER + f"""
SEED = {seed}
S, D, DT = {S}, {D}, {dtype}

def _rope(t):
    S, E = t.shape
    half = E // 2
    pos = torch.arange(S, device=t.device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    t1 = t[..., :half].float(); t2 = t[..., half:half * 2].float()
    out = t.float().clone()
    out[..., :half] = t1 * cos - t2 * sin
    out[..., half:half * 2] = t1 * sin + t2 * cos
    return out.to(t.dtype)

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        q = _rope(x @ self.Wq); k = _rope(x @ self.Wk); v = x @ self.Wv
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
{mask}
        a = torch.softmax(scores, dim=-1)
        return a @ v

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
"""
    return dict(name="rope%s_s%d_d%d_%d" % ("c" if causal else "", S, D, seed),
                tier="L3", family="rope-attention",
                tags=sorted(["tiling", "tensor-core", "online-softmax", "reduction", "fusion", "rope"] + (["masking"] if causal else [])),
                meta={"S": S, "D": D, "dtype": ("fp16" if "float16" in dtype else "bf16"),
                      "causal": causal, "chain": ["qkv", "rope", "scores", "softmax", "av"]},
                source=src)


def quant_gemm_task(seed, M, D, N, dtype):
    src = ATTN_HEADER + f"""
SEED = {seed}
M, D, N, DT = {M}, {D}, {N}, {dtype}

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
        x = x @ w + self.bias
        return F.gelu(x)

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
"""
    return dict(name="quant_%dx%dx%d_%d" % (M, D, N, seed),
                tier="L2", family="quant-gemm",
                tags=sorted(["tiling", "tensor-core", "quantization", "dequant-fusion", "elementwise-fusion"]),
                meta={"M": M, "D": D, "N": N, "dtype": ("fp16" if "float16" in dtype else "bf16"),
                      "chain": ["int8-dequant", "matmul->%d" % N, "bias", "gelu"]},
                source=src)


def moe_task(seed, S, D, N, E, dtype):
    src = ATTN_HEADER + f"""
SEED = {seed}
S, D, N, E, DT = {S}, {D}, {N}, {E}, {dtype}

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        gate = torch.softmax(x @ self.Wr, dim=-1)
        outs = torch.stack([x @ self.We[e] for e in range(E)], dim=0)
        y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
        return F.gelu(y)

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
"""
    return dict(name="moe_s%d_d%d_n%d_e%d_%d" % (S, D, N, E, seed),
                tier="L3", family="moe",
                tags=sorted(["tiling", "tensor-core", "reduction", "online-softmax", "grouped-gemm", "fusion"]),
                meta={"S": S, "D": D, "N": N, "E": E, "dtype": ("fp16" if "float16" in dtype else "bf16"),
                      "chain": ["router-softmax", "%d-expert-gemm" % E, "weighted-sum", "gelu"]},
                source=src)


def generate_systematic(n_fusion=160, seed0=0):
    """Dense, structured coverage across families: fusion op-graphs, attention, RoPE
    attention, quantized dequant-GEMM, and soft-MoE grids."""
    tasks = [gen_task(seed0 + i) for i in range(n_fusion)]
    s = seed0 + 100000
    for S in [512, 1024, 2048]:
        for D in [512, 1024, 2048]:
            for dtype in ["torch.float16", "torch.bfloat16"]:
                for causal in [False, True]:
                    tasks.append(attn_task(s, S, D, dtype, causal)); s += 1
    s = seed0 + 200000
    for S in [512, 1024]:
        for D in [512, 1024, 2048]:
            for dtype in ["torch.float16", "torch.bfloat16"]:
                for causal in [False, True]:
                    tasks.append(rope_attn_task(s, S, D, dtype, causal)); s += 1
    s = seed0 + 300000
    for M in [1024, 4096]:
        for D in [1024, 2048]:
            for N in [1024, 4096]:
                for dtype in ["torch.float16", "torch.bfloat16"]:
                    tasks.append(quant_gemm_task(s, M, D, N, dtype)); s += 1
    s = seed0 + 400000
    for S in [512, 1024]:
        for D in [1024, 2048]:
            for N in [1024, 2048]:
                for E in [4, 8]:
                    tasks.append(moe_task(s, S, D, N, E, "torch.float16")); s += 1
    return tasks


if __name__ == "__main__":
    ts = generate_systematic()
    fams = {}
    for t in ts:
        fams[t["family"]] = fams.get(t["family"], 0) + 1
    print("total", len(ts), "families", fams)
