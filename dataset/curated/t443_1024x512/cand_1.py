import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 443
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_ln_softmax2_rms_kernel(
    X, G, B, W, Out,
    N, stride,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, round to bf16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #1 (fp32 math, round to bf16) ----
    m1 = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #2 (fp32 math, round to bf16) ----
    m2 = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y = e2 / s2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm-style scale (fp32, round to bf16, then * weight) ----
    ms = tl.sum(y * y, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y * r).to(tl.bfloat16).to(tl.float32) * w

    tl.store(Out + row * stride + cols, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax2_rms_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.rms4_w, out,
            N, h.stride(0),
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
