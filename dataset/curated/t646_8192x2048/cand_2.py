import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 646
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, out_ptr,
    g1_ptr, b1_ptr, g4_ptr, b4_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
    EPS: tl.constexpr, SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32, rounded to fp16 like PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm 1 (fp32 math, fp16 output)
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(g1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = d1 * rstd1 * g1 + b1
    x = x.to(tl.float16).to(tl.float32)

    # Softmax (fp32 math, fp16 output)
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.where(mask, tl.exp(x - xm), 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.float16).to(tl.float32)

    # scale
    x = x * SCALE
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm 4
    mean4 = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    d4 = tl.where(mask, x - mean4, 0.0)
    var4 = tl.sum(d4 * d4, axis=0) / D
    rstd4 = 1.0 / tl.sqrt(var4 + EPS)
    g4 = tl.load(g4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d4 * rstd4 * g4 + b4

    tl.store(out_ptr + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            y = y * 1.1269
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.ln4_g, self.ln4_b,
            D=d, BLOCK=BLOCK, EPS=1e-5, SCALE=1.1269,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
