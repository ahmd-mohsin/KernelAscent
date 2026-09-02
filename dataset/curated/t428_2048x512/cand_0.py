import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 428
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_relu_bias_gelu_bias_softmax(
    x_ptr, b1_ptr, b3_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # relu + bias
    x = tl.maximum(x, 0.0) + b1
    # exact (erf-based) GELU, matching F.gelu default
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # bias
    x = x + b3

    # softmax along the row
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * D + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            x = F.gelu(x)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_relu_bias_gelu_bias_softmax[(rows,)](
            x2, self.b1, self.b3, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
