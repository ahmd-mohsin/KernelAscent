import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 795
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf), round to bf16 to match reference elementwise op
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # + bias (bf16 arithmetic)
    x = (x.to(tl.bfloat16) + b.to(tl.bfloat16)).to(tl.float32)

    # * scale (bf16 arithmetic)
    x = (x.to(tl.bfloat16) * tl.full((), 1.4866, tl.bfloat16)).to(tl.float32)

    # gelu again
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's internal upcast)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, cols = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(rows, cols)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(rows,)](
            x2, self.b1, y, cols, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y.view(x.shape)
