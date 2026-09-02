import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 739
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps, scale,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    # x = x * 1.4683 in fp16 (match reference half-precision multiply)
    x = x * scale.to(tl.float16)

    xf = x.to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    y = xn * w  # fp16 multiply, as in reference

    # softmax in fp32 (matches PyTorch half softmax internal fp32 accumulation)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(N)
        _fused_rms_softmax_kernel[(Mrows,)](
            x, self.rms2_w, out,
            N, x.stride(0), out.stride(0),
            1e-6, 1.4683,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,
        )
        return out
