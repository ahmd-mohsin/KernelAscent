import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 539
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # x = x + b  (fp16 rounding to match reference)
    x = (x + b).to(tl.float16)
    # x = x * 1.1541  (fp16 rounding)
    x = (x * 1.1541).to(tl.float16)

    # exact GELU computed in fp32 (as PyTorch does internally), cast back to fp16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32 (PyTorch upcasts internally for half)
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = y * 1.1541
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        rows, d = x.shape[0], x.shape[-1]
        x2 = x.view(-1, d)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(x2.shape[0],)](
            x2, self.b0, out, d, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view_as(x)
