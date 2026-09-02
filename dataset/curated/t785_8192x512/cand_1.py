import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

SEED = 785
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 internal, matching PyTorch's bf16 softmax path)
    row_max = tl.max(x, axis=0)
    e = tld.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round to bf16 (softmax output dtype), then back to fp32 for RMS norm
    sm_bf16 = sm.to(tl.bfloat16)
    xf = sm_bf16.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    y = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = y * w  # bf16 * bf16 -> fp32 mul, rounded back like torch

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM, tensor cores
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
