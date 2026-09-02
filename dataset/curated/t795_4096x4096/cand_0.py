import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 795
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf), round to bf16 like the reference elementwise op
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # + bias
    x = x + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # * scale
    x = x * 1.4866
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu again
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal accumulation)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * D_ + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x + self.b1
            x = x * 1.4866
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, d = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        orig_shape = x.shape
        x2 = x.view(-1, d)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(x2.shape[0],)](
            x2, self.b1, y, d, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 2048 else 4,
        )
        return y.view(orig_shape)
