import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 576
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask,
                other=-float('inf')).to(tl.float32)

    # ---- softmax #1 (fp32 math, round to bf16 like PyTorch output) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)          # exp(-inf) = 0 for masked lanes
    s1 = tl.sum(e1, 0)
    y = e1 / s1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf), fp32 math, round to bf16 ----
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # ---- LayerNorm (fp32 accumulation, biased var, eps=1e-5) ----
    mean = tl.sum(g, 0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, 0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    z = diff * rstd * w + b
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- scale (fp32 opmath, round to bf16) ----
    z = z * SCALE
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- softmax #2 ----
    z = tl.where(mask, z, -float('inf'))
    m2 = tl.max(z, 0)
    e2 = tl.exp(z - m2)
    s2 = tl.sum(e2, 0)
    out = e2 / s2

    tl.store(out_ptr + row * stride_row + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = y * 1.3939
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(rows,)](
            x2d, self.ln2_g, self.ln2_b, out,
            N, x2d.stride(0),
            EPS=1e-5,
            SCALE=1.3939,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)
