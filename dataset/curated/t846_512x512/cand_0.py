import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 846
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_softmax_kernel(
    X, B1, B2, OUT,
    stride_xm, stride_om,
    N, SCALE,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # match fp16 intermediate rounding of reference: (x*s), +b1, +b2 each in fp16
    v = (x * SCALE).to(tl.float16).to(tl.float32)
    v = (v + b1).to(tl.float16).to(tl.float32)
    v = (v + b2).to(tl.float16).to(tl.float32)

    v = tl.where(mask, v, float('-inf'))
    vmax = tl.max(v, axis=0)
    e = tl.exp(v - vmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            v = x * 1.1929
            v = v + self.b1
            v = v + self.b2
            return torch.softmax(v, dim=-1)

        x = x.contiguous()
        shape = x.shape
        N = shape[-1]
        x2 = x.view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_softmax_kernel[(Mrows,)](
            x2, self.b1, self.b2, out,
            x2.stride(0), out.stride(0),
            N, 1.1929,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out.view(shape)
