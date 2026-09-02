import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 150
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_rms3_kernel(
    X, G, B, W3, W4, W5, Y,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    base = X + row * N
    x = tl.load(base + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to bf16 like torch.softmax) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- layer norm (fp32 stats, bf16 output) ----
    x = tl.where(mask, x, 0.0)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- RMS norm #1 ----
    x = tl.where(mask, x, 0.0)
    ms = tl.sum(x * x, 0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    n = (x * r).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (n * w3).to(tl.bfloat16).to(tl.float32)

    # ---- RMS norm #2 ----
    x = tl.where(mask, x, 0.0)
    ms = tl.sum(x * x, 0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    n = (x * r).to(tl.bfloat16).to(tl.float32)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (n * w4).to(tl.bfloat16).to(tl.float32)

    # ---- RMS norm #3 ----
    x = tl.where(mask, x, 0.0)
    ms = tl.sum(x * x, 0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    n = (x * r).to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (n * w5).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS bf16 tensor-core GEMM
        if not h.is_contiguous():
            h = h.contiguous()
        rows, N = h.shape[0], h.shape[1]
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_ln_rms3_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b,
            self.rms3_w, self.rms4_w, self.rms5_w,
            y, N,
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
