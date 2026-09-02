import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 316
M, D, DT = 512, 512, torch.float16


@triton.jit
def _bias_softmax_scale_kernel(
    X, B, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 to match reference (x + b1 in fp16), then softmax in fp32
    xb = (x + b).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    row_max = tl.max(xb, axis=0)
    e = tl.exp(xb - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = (e / denom) * SCALE

    tl.store(Out + row * stride_om + cols, y.to(Out.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS TF32/FP16 tensor-core matmul
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _bias_softmax_scale_kernel[(m,)](
            h, self.b1, out,
            h.stride(0), out.stride(0),
            n, BLOCK=BLOCK, SCALE=1.018,
            num_warps=4,
        )
        return out
