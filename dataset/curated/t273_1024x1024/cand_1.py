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
    X_ptr, W2_ptr, W3_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    # cast to bf16 (softmax output dtype), back to fp32 for RMS
    sm_bf = sm.to(tl.bfloat16)
    xf = sm_bf.to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * r).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w2).to(tl.bfloat16)

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (yf * r2).to(tl.bfloat16)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w3).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms2_kernel[(Mrows,)](
            h, self.rms2_w, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
