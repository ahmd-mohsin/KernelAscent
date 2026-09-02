import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 299
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, g1_ptr, b1_ptr, g2_ptr, b2_ptr, out_ptr,
    N, stride_row, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)

    # x = x + b0 (bf16 add semantics: fp32 add then round to bf16)
    x = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm 1 (accumulate in fp32, output rounded to bf16)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean1 = tl.sum(xf, axis=0) / N
    d1 = tl.where(mask, xf - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(g1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = (d1 * rstd1 * g1 + b1).to(tl.bfloat16)

    # LayerNorm 2
    yf = y1.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean2 = tl.sum(yf, axis=0) / N
    d2 = tl.where(mask, yf - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(g2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * rstd2 * g2 + b2).to(tl.bfloat16)

    # GELU (exact, erf-based), computed in fp32 on bf16 input
    z = y2.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(out_ptr + row * stride_row + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return F.gelu(x)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, out,
            n, x2.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
