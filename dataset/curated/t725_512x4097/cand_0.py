import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 725
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_gelu2_bias_softmax(
    X, B, Out,
    n_cols,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf, fp32 math, round to bf16 like eager)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g2 = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # bias add (fp32 accumulate, round to bf16 like eager)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    z = (g2 + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * stride_o + offs, out.to(Out.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.gelu(y)
            y = y + self.b2
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        b = self.b2
        if b.device != x.device:
            b = b.to(x.device)

        n_rows, n_cols = x.shape[-2], x.shape[-1]
        orig_shape = x.shape
        x2d = x.view(-1, n_cols)
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)

        _fused_gelu2_bias_softmax[(n_rows,)](
            x2d, b, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=16 if BLOCK >= 4096 else 8,
        )
        return out.view(orig_shape)
