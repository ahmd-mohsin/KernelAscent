import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 853
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, g_ptr, b_ptr, out_ptr,
                  N, stride_row,
                  EPS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # scale, round to bf16 (matches eager op-by-op rounding)
    x = (x * 1.3715).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), round to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm in fp32
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    ym = tl.where(mask, y, float('-inf'))
    ymax = tl.max(ym, axis=0)
    e = tl.exp(ym - ymax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(out_ptr + row * stride_row + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, out,
            N, x.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
