import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 560
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_gelu_softmax_kernel(
    x_ptr, b_ptr, out_ptr,
    D,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf) computed in fp32, rounded back to fp16 like PyTorch half ops
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # scale by 1.2123
    g = g * 1.2123
    g = g.to(tl.float16).to(tl.float32)

    # relu (exact in fp16, no rounding needed)
    g = tl.maximum(g, 0.0)

    # gelu again
    g = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # add bias
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.float16).to(tl.float32)

    # softmax over the row with fp32 accumulation
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(out_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = F.gelu(x)
            x = x * 1.2123
            x = torch.relu(x)
            x = F.gelu(x)
            x = x + self.b4
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, d = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_softmax_kernel[(rows,)](
            x2, self.b4, out, d,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
