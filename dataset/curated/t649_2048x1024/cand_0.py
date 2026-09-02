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
    X_ptr, W1_ptr, W3_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    y = (xf * inv).to(tl.float16) * w1  # fp16 multiply, matching reference

    # ---- Softmax 1 (float32 accumulation, like PyTorch half softmax) ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m1 = tl.max(yf, axis=0)
    e1 = tl.exp(yf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = (e1 / s1).to(tl.float16)

    # ---- RMSNorm 2 ----
    pf = p.to(tl.float32)
    ms2 = tl.sum(pf * pf, axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0)
    z = (pf * inv2).to(tl.float16) * w3

    # ---- Softmax 2 ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    m2 = tl.max(zf, axis=0)
    e2 = tl.exp(zf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_kernel[(rows,)](
            x, self.rms1_w, self.rms3_w, out,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
