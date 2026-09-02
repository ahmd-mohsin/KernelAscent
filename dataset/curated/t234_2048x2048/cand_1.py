import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 234
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, B2, B4, OUT, N, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # x = x * 1.4841 (round to bf16 to match reference precision)
    x = (x * 1.4841).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf-based) computed in fp32, rounded to bf16
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # add b2
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # gelu again
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # add b4
    x = (x + b4).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(OUT + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.4841
            x = F.gelu(x)
            x = x + self.b2
            x = F.gelu(x)
            x = x + self.b4
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        x2d = x.view(-1, N)
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(x2d.shape[0],)](
            x2d, self.b2, self.b4, out,
            N, x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view_as(x)
