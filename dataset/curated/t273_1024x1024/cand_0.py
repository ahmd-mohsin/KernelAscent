import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 273
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_rms2_kernel(
    X_ptr, W2_ptr, W3_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (bf16 -> fp32) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, matches torch bf16 softmax) ----
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, axis=0)

    # cast to bf16 (softmax output dtype), then re-upcast as torch does with .float()
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms1 = tl.sum(tl.where(mask, s * s, 0.0), axis=0) / N
    r1 = tl.rsqrt(ms1 + 1e-6)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = (s * r1).to(tl.bfloat16).to(tl.float32) * w2
    # result of bf16*bf16 elementwise mul is rounded to bf16
    x1 = x1.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(tl.where(mask, x1 * x1, 0.0), axis=0) / N
    r2 = tl.rsqrt(ms2 + 1e-6)
    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x1 * r2).to(tl.bfloat16).to(tl.float32) * w3

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms2_kernel[(Mrows,)](
            h, self.rms2_w, self.rms3_w, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
