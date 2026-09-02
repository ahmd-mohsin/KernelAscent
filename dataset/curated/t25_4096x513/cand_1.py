import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 25
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_xm, stride_om,
    S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, then rounded to bf16 as in reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 op with fp32 opmath, rounded to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xn = (xn * w).to(tl.bfloat16).to(tl.float32)

    # scalar multiplies (each rounds to bf16, matching PyTorch)
    xn = (xn * S1).to(tl.bfloat16).to(tl.float32)
    xn = (xn * S2).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (PyTorch upcasts bf16 softmax internally)
    xn = tl.where(mask, xn, float('-inf'))
    m = tl.max(xn, axis=0)
    e = tl.exp(xn - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    out = (sm * S3).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, out,
            N, x.stride(0), out.stride(0),
            1.2879, 1.2562, 1.2744,
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
