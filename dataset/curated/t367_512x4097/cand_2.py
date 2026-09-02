import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 367
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _gelu_rms_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    n = (g * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    v = (n * w).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.math.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        rows, cols = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(cols)
        _gelu_rms_softmax_kernel[(rows,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=cols,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
