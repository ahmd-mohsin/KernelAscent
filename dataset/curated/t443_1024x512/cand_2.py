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
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, bf16 rounding to match PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #1 (fp32 math on bf16-rounded input) ----
    y_inf = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y_inf, axis=0)
    e1 = tl.exp(y_inf - m1)
    e1 = tl.where(mask, e1, 0.0)
    p = e1 / tl.sum(e1, axis=0)
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #2 ----
    p_inf = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p_inf, axis=0)
    e2 = tl.exp(p_inf - m2)
    e2 = tl.where(mask, e2, 0.0)
    q = e2 / tl.sum(e2, axis=0)
    q = q.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (explicit fp32, matching reference exactly) ----
    ms = tl.sum(tl.where(mask, q * q, 0.0), axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    t = (q * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (t * w).to(tl.bfloat16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax2_rms_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
