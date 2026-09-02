import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 729
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _gelu_scale_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    stride_xm, stride_om,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, matching F.gelu default
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # replicate fp16 intermediate rounding of the reference
    g = g.to(tl.float16).to(tl.float32)
    g = (g * 1.3162).to(tl.float16).to(tl.float32)

    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    v = (g + b).to(tl.float16).to(tl.float32)

    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = e / s

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _gelu_scale_bias_softmax_kernel[(Mrows,)](
            y, self.b3, out,
            y.stride(0), out.stride(0),
            N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
