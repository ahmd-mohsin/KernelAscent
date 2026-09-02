import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 60
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b1_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0)

    # relu + bias (in bf16 to match reference)
    x = tl.maximum(x, 0.0)
    x = (x + b1).to(x_ptr.dtype.element_ty)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    y = (xf * inv).to(x_ptr.dtype.element_ty)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = y * w

    tl.store(out_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.b1, self.rms2_w, out, N, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
