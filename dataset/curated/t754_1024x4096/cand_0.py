import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 754
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_relu_bias_softmax(X, B, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.maximum(x, 0.0) + b
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * D + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            return torch.softmax(x, dim=-1)
        x = x.contiguous()
        Mrows, Dcols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, Dcols)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 4096 else 4
        _fused_relu_bias_softmax[(x2.shape[0],)](
            x2, self.b1, y, Dcols, BLOCK=BLOCK, num_warps=num_warps
        )
        return y.view_as(x)
