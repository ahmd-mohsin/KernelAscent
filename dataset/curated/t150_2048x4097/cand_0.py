import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 150
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    LN_G, LN_B, W3, W4, W5,
    N, stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to bf16) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- layer norm (eps=1e-5) ----
    n_f = N.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / n_f
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * inv * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- rms norm 3 (eps=1e-6) ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), 0) / n_f
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w3)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- rms norm 4 ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), 0) / n_f
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w4)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- rms norm 5 ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), 0) / n_f
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w5)

    tl.store(Y + row * stride_y + offs, x.to(tl.bfloat16), mask=mask)


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
        # GEMM on tensor cores
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_norm_kernel[(m,)](
            h, y,
            self.ln2_g, self.ln2_b, self.rms3_w, self.rms4_w, self.rms5_w,
            n, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return y
