import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 336
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(
    X, B1, B2, OUT,
    N,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to bf16 (matches PyTorch)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16)

    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)

    # bf16 adds with rounding after each op (matches PyTorch elementwise adds)
    y = (g + b1).to(tl.bfloat16)
    y = (y + b2).to(tl.bfloat16)

    # softmax in fp32 accumulation (matches PyTorch softmax acc type)
    f = y.to(tl.float32)
    f = tl.where(mask, f, float("-inf"))
    row_max = tl.max(f, axis=0)
    e = tl.exp(f - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(OUT + row * stride_om + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = y + self.b1
            y = y + self.b2
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_bias_softmax[(m,)](
            x, self.b1, self.b2, out,
            n,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
