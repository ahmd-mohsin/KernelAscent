import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 281
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W0, W1, B3, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms0 = tl.sum(x * x, axis=0) / N
    r0 = 1.0 / tl.sqrt(ms0 + 1e-6)
    t = (x * r0).to(tl.bfloat16).to(tl.float32)          # cast to bf16 as reference
    x1 = (t * w0).to(tl.bfloat16).to(tl.float32)         # bf16 multiply semantics

    # ---- RMSNorm 1 ----
    x1m = tl.where(mask, x1, 0.0)
    ms1 = tl.sum(x1m * x1m, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    t2 = (x1 * r1).to(tl.bfloat16).to(tl.float32)
    x2 = (t2 * w1).to(tl.bfloat16).to(tl.float32)

    # ---- ReLU + bias (bf16 add semantics) ----
    x3 = tl.maximum(x2, 0.0)
    x4 = (x3 + b3).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, bf16 output) ----
    x4m = tl.where(mask, x4, float('-inf'))
    mx = tl.max(x4m, axis=0)
    e = tl.exp(x4m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(OUT + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.rms1_w, self.b3, out,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
