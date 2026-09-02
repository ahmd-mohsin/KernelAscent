import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 649
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_rms_softmax_kernel(
    X, W1, W3, Y,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (fp32 compute, fp16 rounding to match reference) ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    x = (x * r).to(tl.float16).to(tl.float32)          # cast to fp16 as in reference
    x = (x * w1).to(tl.float16).to(tl.float32)         # fp16 multiply semantics

    # ---- Softmax 1 (fp32 accumulate, fp16 output as in reference) ----
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xm)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    x = (x * r).to(tl.float16).to(tl.float32)
    x = (x * w3).to(tl.float16).to(tl.float32)

    # ---- Softmax 2 ----
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xm)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, self.rms3_w, y,
            x.stride(0), y.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
