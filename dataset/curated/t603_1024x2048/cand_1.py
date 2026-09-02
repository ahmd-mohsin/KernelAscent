import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 603
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_scale_bias_softmax(
    X, B, OUT,
    N, stride_xm, stride_om,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * scale + b
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(OUT + row * stride_om + cols, y.to(OUT.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_bias_softmax[(m,)](
            h, self.b2, out,
            n, h.stride(0), out.stride(0),
            1.2045,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
