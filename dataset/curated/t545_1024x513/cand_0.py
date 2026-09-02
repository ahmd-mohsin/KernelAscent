import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 545
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_ln_softmax_rms(
    X, G, B, B3, RW, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # ---- load row in fp32 ----
    x = tl.load(X + base + offs).to(tl.float32)

    # ---- LayerNorm (fp32 math, fp16 output like F.layer_norm on half) ----
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    x = d * inv * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 math, fp16 output) ----
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # ---- + b3 (fp16 tensors, fp32 compute -> fp16) ----
    b3 = tl.load(B3 + offs).to(tl.float32)
    x = (x.to(tl.float32) + b3).to(tl.float16)

    # ---- * 1.4014 (fp32 compute -> fp16) ----
    x = (x.to(tl.float32) * 1.4014).to(tl.float16)

    # ---- RMSNorm in fp32, cast to fp16, then * rms5_w in fp16 ----
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * r).to(tl.float16)
    rw = tl.load(RW + offs)
    y = xh * rw

    tl.store(Y + base + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_softmax_rms[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b3, self.rms5_w, out,
            N=N, BLOCK=N,
            num_warps=4,
        )
        return out
