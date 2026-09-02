import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 414
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, OUT,
    W1, G2, B2, G3, B3,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 math, cast to fp16, then fp16-weight mul in fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    a = (x * r).to(tl.float16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    h = (a.to(tl.float32) * w1).to(tl.float16)

    # ---- LayerNorm 2 ----
    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    h = ((hf - mean) * rstd * g2 + b2).to(tl.float16)

    # ---- LayerNorm 3 ----
    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    h = ((hf - mean) * rstd * g3 + b3).to(tl.float16)

    # ---- scalar scale (fp32 opmath, round to fp16) ----
    out = (h.to(tl.float32) * 1.1262).to(tl.float16)
    tl.store(OUT + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(Mrows,)](
            y, out,
            self.rms1_w, self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
