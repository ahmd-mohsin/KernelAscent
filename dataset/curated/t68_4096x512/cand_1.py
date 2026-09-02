import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 68
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 compute, round to fp16 like PyTorch output)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # softmax 2
    m2 = tl.max(tl.where(mask, p1, float('-inf')), axis=0)
    e2 = tl.exp(p1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16).to(tl.float32)

    # RMSNorm (fp32), cast to fp16 like reference
    ms = tl.sum(p2 * p2, axis=0) / N
    r = (p2 * tl.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x * w  (half op, opmath fp32, round to half)
    y = (r * w).to(tl.float16).to(tl.float32)
    # x + b
    y = (y + b).to(tl.float16).to(tl.float32)
    # exact gelu: 0.5*x*(1+erf(x/sqrt(2)))
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(Out_ptr + row * stride_o + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            y, self.rms3_w, self.b4, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
