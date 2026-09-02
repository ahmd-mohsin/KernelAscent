import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 981
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_softmax_bias_rms_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_xm, stride_om,
    SCALE1: tl.constexpr, SCALE2: tl.constexpr, EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's float accumulation for bf16 softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    # round to bf16 as torch.softmax returns bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # x * 1.4911 (float compute, bf16 round)
    y = (y * SCALE1).to(tl.bfloat16).to(tl.float32)

    # x + b3 (float compute, bf16 round)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    y = (y * inv).to(tl.bfloat16).to(tl.float32)

    # * rms4_w (float compute, bf16 round)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # * 1.1793 (float compute, bf16 round)
    y = (y * SCALE2).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_softmax_bias_rms_kernel[(Mrows,)](
            h, self.b3, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            SCALE1=1.4911, SCALE2=1.1793, EPS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
