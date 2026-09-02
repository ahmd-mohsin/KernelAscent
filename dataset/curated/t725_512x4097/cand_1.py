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
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf), round to bf16 to match reference intermediate dtype
    y = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)

    # bias add (bf16 result, matching reference)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y, axis=0)
    e = tl.exp(y - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


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
        rows, n = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2d = x.view(rows, n)
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu2_bias_softmax[(rows,)](
            x2d, self.b2, out,
            n, x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(x.shape)
