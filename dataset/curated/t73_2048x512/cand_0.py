import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 73
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_post_kernel(
    X, RMS_W, LN_G, LN_B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accumulate, fp16 output like PyTorch) ----
    mx = tl.max(x, 0)
    e = tl.math.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- GELU (exact, fp32 opmath like PyTorch, fp16 output) ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # ---- softmax 2 ----
    x = tl.where(mask, x, float('-inf'))
    mx = tl.max(x, 0)
    e = tl.math.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32 math, cast to fp16, mul by fp16 weight) ----
    ms = tl.sum(x * x, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(RMS_W + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.float16).to(tl.float32)
    x = (x * w).to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 stats like PyTorch, fp16 output) ----
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + offs, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b

    tl.store(OUT + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            x, self.rms4_w, self.ln5_g, self.ln5_b, out,
            x.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
