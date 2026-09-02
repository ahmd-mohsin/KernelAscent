import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 853
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, G, B, Y, N, stride_x, stride_y, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # scale, round to bf16 (match reference bf16 intermediate)
    x = (x * 1.3715).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, stored back as bf16
    h = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    h = h.to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z = (h - mean) * rstd * g + b
    z = z.to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, y,
            N, x.stride(0), y.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
