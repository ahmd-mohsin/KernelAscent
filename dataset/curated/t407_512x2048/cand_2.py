import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 407
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, g_ptr, b_ptr, out_ptr,
                  D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (rounded to bf16 like reference)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm (fp32 stats, bf16 output like PyTorch)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + bb
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation)
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xm)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu again
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    tl.store(out_ptr + row * D + offs, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = torch.softmax(y, dim=-1)
            return F.gelu(y)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, d = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.b0, self.ln2_g, self.ln2_b, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
